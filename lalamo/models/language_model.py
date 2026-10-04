from collections.abc import Iterable
from dataclasses import dataclass, fields, replace
from typing import NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import Array
from jaxtyping import Bool, Float, Int, Key
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import AssistantMessage, ChatCodec, ChatCodecConfig, Message, ReasoningEffort
from lalamo.models.thinking_budget import (
    MISSING_TOKEN_ID,
    ThinkingProgress,
    advance_thinking_state,
    apply_thinking_budget,
    prompt_closes_thinking,
    resolve_end_of_thinking_token_ids,
)
from lalamo.module import LogicalAxis
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    ForwardPassMode,
    Keychain,
    KeychainBroadcastMode,
)
from lalamo.modules.token_mixer import State
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy
from lalamo.utils.sharding import reshard_as

__all__ = [
    "GenerationConfig",
    "GenerationResults",
    "LanguageModel",
    "LanguageModelConfig",
]


_COMPILED_PROMPT_LENGTHS = tuple(256 * 2**i for i in range(12))


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State


class Chunk(eqx.Module):
    tokens: Int[Array, "num_chunks batch chunk_size"]
    indices: Int[Array, "num_chunks batch chunk_size"]
    sequence_ends: Int[Array, "num_chunks batch"]
    is_last_token_inside: Bool[Array, "num_chunks batch"]


class DecodingState(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State
    stop_flags: Bool[Array, " batch"]
    sampling_policy: SamplingPolicy


class GenerationStepResults(NamedTuple):
    token_ids: Int[Array, " batch"]
    top_k_token_ids: Int[Array, " batch k"] | None
    top_k_token_logits: Float[Array, " batch k"] | None


class GenerationResults(NamedTuple):
    token_ids: Int[Array, "batch response_tokens"]
    top_k_token_ids: Int[Array, "batch response_tokens k"] | None
    top_k_token_logits: Float[Array, "batch response_tokens k"] | None


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling controls for one generation call.

    `thinking_budget` caps tokens before the model's end-of-thinking tag.
    `None` leaves generation unchanged. See `lalamo.models.thinking_budget`.
    """

    stop_token_ids: tuple[int, ...] = ()
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    banned_tokens: tuple[int, ...] | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    suffix_repetition_length: int | None = None
    thinking_budget: int | None = None

    def override_with(self, other: Self) -> Self:
        default_config = GenerationConfig()
        merged_fields = {"stop_token_ids", "banned_tokens"}
        overrides = {}
        for config_field in fields(GenerationConfig):
            field_name = config_field.name
            ours = getattr(self, field_name)
            theirs = getattr(other, field_name)
            default = getattr(default_config, field_name)

            if theirs == default:
                continue
            if field_name in merged_fields and ours != default:
                overrides[field_name] = tuple(sorted({*ours, *theirs}))
            else:
                overrides[field_name] = theirs

        return replace(self, **overrides)

    def default_policy(self) -> SamplingPolicy:
        return SamplingPolicy.init(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            banned_tokens=self.banned_tokens,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            suffix_repetition_length=self.suffix_repetition_length,
        )


@dataclass(frozen=True)
class LanguageModelConfig(ModelConfig[ChatCodecConfig]):
    decoder_config: DecoderConfig
    generation_config: GenerationConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "LanguageModel":
        decoder = self.decoder_config.init(initializer)
        token_codec = self.token_codec_config.init(tokenizer)
        return LanguageModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=token_codec,
            decoder=decoder,
        )


class LanguageModel(Model[ChatCodecConfig, LanguageModelConfig, ChatCodec]):
    token_codec: ChatCodec
    decoder: Decoder

    def default_sampling_policy(self) -> SamplingPolicy:
        return self.config.generation_config.default_policy()

    def trim_at_eos(self, token_ids: list[int]) -> list[int]:
        if not self.config.generation_config.stop_token_ids:
            return token_ids
        stop_token_ids = set(self.config.generation_config.stop_token_ids)
        response_length = next(
            (idx + 1 for idx, token_id in enumerate(token_ids) if token_id in stop_token_ids),
            len(token_ids),
        )
        return token_ids[:response_length]

    @eqx.filter_jit
    def _make_attention_chunks(
        self,
        token_ids: Int[Array, "batch tokens"],
        lengths_without_padding: Int[Array, " batch"] | None,
        chunk_size: int,
    ) -> Chunk:
        batch_size, sequence_length = token_ids.shape
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        if lengths_without_padding is None:
            lengths_without_padding = jax.device_put(
                jnp.full((batch_size,), sequence_length, dtype=jnp.int32),
                self.sharding_config.make_sharding((batch_axis,)),
            )

        chunk_size = min(chunk_size, sequence_length)
        num_chunks = (sequence_length + chunk_size - 1) // chunk_size
        padded_length = num_chunks * chunk_size
        chunk_token_sharding = self.sharding_config.make_sharding((None, batch_axis, None))
        chunk_vector_sharding = self.sharding_config.make_sharding((None, batch_axis))

        tokens = rearrange(
            jnp.pad(token_ids, ((0, 0), (0, padded_length - sequence_length))),
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )
        tokens = jax.device_put(tokens, chunk_token_sharding)
        indices = rearrange(
            jnp.broadcast_to(jnp.arange(padded_length, dtype=jnp.int32), (batch_size, padded_length)),
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )
        indices = jax.device_put(indices, chunk_token_sharding)
        chunk_starts = jnp.arange(num_chunks, dtype=jnp.int32) * chunk_size
        sequence_ends = jnp.clip(lengths_without_padding[None, :] - chunk_starts[:, None], 0, chunk_size)
        sequence_ends = jax.device_put(sequence_ends, chunk_vector_sharding)
        last_token_idx = lengths_without_padding - 1
        is_last_token_inside = (last_token_idx[None, :] >= chunk_starts[:, None]) & (
            last_token_idx[None, :] < chunk_starts[:, None] + chunk_size
        )
        is_last_token_inside = jax.device_put(is_last_token_inside, chunk_vector_sharding)

        return Chunk(
            tokens=tokens,
            indices=indices,
            sequence_ends=sequence_ends,
            is_last_token_inside=is_last_token_inside,
        )

    @eqx.filter_jit
    def prefill_tokens(
        self,
        token_ids: Int[Array, "batch tokens"],
        state_capacity: int,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: DecoderForwardPassConfig | None = None,
        chunk_size: int = 512,
        *,
        keychain: Keychain,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        batch_vector_sharding = self.sharding_config.make_sharding((batch_axis,))
        if lengths_without_padding is None:
            lengths_without_padding = jax.device_put(
                jnp.full((batch_size,), sequence_length, dtype=jnp.int32),
                batch_vector_sharding,
            )
        if forward_pass_config is None:
            forward_pass_config = DecoderForwardPassConfig.for_inference()

        chunks = self._make_attention_chunks(token_ids, lengths_without_padding, chunk_size)
        num_chunks, _, chunk_size = chunks.tokens.shape
        state_dtype = forward_pass_config.embedding_forward_pass_config.activation_dtype
        state = self.decoder.init_static_state(
            batch_size,
            max(state_capacity, num_chunks * chunk_size),
            state_dtype,
        )
        logits_like = jax.device_put(
            jnp.zeros((batch_size, self.decoder.vocab_size), dtype=jnp.float32),
            self.sharding_config.make_sharding((batch_axis, None)),
        )
        logits_sharding = self.sharding_config.make_sharding((batch_axis, None))
        chunk_keychains = jax.tree.map(lambda *nodes: jnp.stack(nodes), *keychain.split(num_chunks))

        def apply_chunk(
            state_and_logits: tuple[State, Float[Array, "batch vocabulary"]],
            chunk_inputs: tuple[Chunk, Keychain],
        ) -> tuple[tuple[State, Float[Array, "batch vocabulary"]], None]:
            current_state, previous_logits = state_and_logits
            chunk, current_chunk_keychain = chunk_inputs
            decoder_result = self.decoder(
                token_ids=chunk.tokens,
                token_positions=chunk.indices,
                state=current_state,
                return_updated_state=True,
                lengths_without_padding=chunk.sequence_ends,
                forward_pass_config=forward_pass_config,
                return_suffix_tokens=1,
                keychain=current_chunk_keychain,
            )
            assert decoder_result.updated_state is not None

            chunk_logits = jax.device_put(decoder_result.logits[:, 0, :], logits_sharding)
            return (
                decoder_result.updated_state,
                jnp.where(chunk.is_last_token_inside[:, None], chunk_logits, previous_logits),
            ), None

        (state, last_token_logits), _ = jax.lax.scan(
            apply_chunk,
            (state, logits_like),
            (chunks, chunk_keychains),
        )

        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_indices=jnp.maximum(lengths_without_padding - 1, 0),
            state=state,
        )

    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        num_top_logits_to_return: int | None = None,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        keychain: Keychain,
    ) -> GenerationResults:
        thinking_budget = None if generation_config is None else generation_config.thinking_budget
        thinking_token_ids = resolve_end_of_thinking_token_ids(
            thinking_budget=thinking_budget,
            end_of_thinking_tag=self.token_codec.config.end_of_thinking_tag,
            encode_tag=self.token_codec.encode_text,
        )
        batch_size = prompt_token_ids.shape[0]
        if thinking_token_ids is None:
            initial_thinking_closed = jnp.zeros((batch_size,), dtype=jnp.bool_)
        else:
            initial_thinking_closed = self._prompts_already_closed_thinking(
                prompt_token_ids,
                prompt_lengths_without_padding,
            )
        return self._generate_tokens(
            prompt_token_ids,
            generation_config,
            prompt_lengths_without_padding,
            max_output_length,
            eos_token_ids,
            num_top_logits_to_return,
            prefill_forward_pass_config,
            decode_forward_pass_config,
            thinking_token_ids,
            initial_thinking_closed,
            keychain=keychain,
        )

    def _prompts_already_closed_thinking(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        prompt_lengths_without_padding: Int[Array, " batch"] | None,
    ) -> Bool[Array, " batch"]:
        rows = prompt_token_ids.tolist()
        if prompt_lengths_without_padding is None:
            lengths = [len(row) for row in rows]
        else:
            lengths = [int(length) for length in prompt_lengths_without_padding.tolist()]
        tag = self.token_codec.config.end_of_thinking_tag
        closed = [
            prompt_closes_thinking(
                self.token_codec.decode_tokens([int(token_id) for token_id in row[:length]]),
                tag,
            )
            for row, length in zip(rows, lengths, strict=True)
        ]
        return jnp.asarray(closed, dtype=jnp.bool_)

    @eqx.filter_jit
    def _generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        generation_config: GenerationConfig | None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None,
        max_output_length: int,
        eos_token_ids: Int[Array, " eos_tokens"] | None,
        num_top_logits_to_return: int | None,
        prefill_forward_pass_config: DecoderForwardPassConfig | None,
        decode_forward_pass_config: DecoderForwardPassConfig | None,
        thinking_token_ids: tuple[int, ...] | None,
        initial_thinking_closed: Bool[Array, " batch"],
        *,
        keychain: Keychain,
    ) -> GenerationResults:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")
        if prefill_forward_pass_config is None:
            prefill_forward_pass_config = DecoderForwardPassConfig.for_inference()
        if decode_forward_pass_config is None:
            decode_forward_pass_config = DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN)

        batch_size, prompt_length = prompt_token_ids.shape
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        batch_vector_sharding = self.sharding_config.make_sharding((batch_axis,))
        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy()
        use_count_penalties = sampling_policy.has_count_penalties
        sampling_policy = sampling_policy.broadcast(batch_size)

        def shard_sampling_leaf(leaf: object) -> object:
            if not isinstance(leaf, jax.Array):
                return leaf
            if leaf.ndim > 0 and leaf.shape[0] == batch_size:
                sharding_axes = (batch_axis, *((None,) * (leaf.ndim - 1)))
            else:
                sharding_axes = (None,) * leaf.ndim
            leaf_sharding = self.sharding_config.make_sharding(sharding_axes)
            return jax.device_put(leaf, leaf_sharding)

        sampling_policy = jax.tree.map(shard_sampling_leaf, sampling_policy)
        if prompt_lengths_without_padding is None:
            prompt_lengths_without_padding = jax.device_put(
                jnp.full((batch_size,), prompt_length, dtype=jnp.int32),
                batch_vector_sharding,
            )
        if use_count_penalties:
            sampling_policy = call_vmapped(
                lambda policy, prompt_token_ids, prompt_length: policy.with_prompt_token_counts(
                    prompt_token_ids,
                    prompt_length,
                    self.decoder.vocab_size,
                ),
                sampling_policy,
                prompt_token_ids,
                prompt_lengths_without_padding,
            )
        if eos_token_ids is None:
            eos_token_ids = jax.device_put(
                jnp.asarray(self.config.generation_config.stop_token_ids, dtype=jnp.int32),
                self.sharding_config.make_sharding((None,)),
            )

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self.prefill_tokens(
            prompt_token_ids,
            prompt_length + max_output_length + 1,
            prompt_lengths_without_padding,
            prefill_forward_pass_config,
            keychain=prefill_keychain,
        )
        initial_state = DecodingState(
            last_token_logits=prefill_results.last_token_logits,
            last_token_indices=prefill_results.last_token_indices,
            state=prefill_results.state,
            stop_flags=jax.device_put(jnp.zeros(batch_size, dtype=jnp.bool_), batch_vector_sharding),
            sampling_policy=sampling_policy,
        )
        stopped_token_ids = jax.device_put(jnp.zeros(batch_size, dtype=jnp.int32), batch_vector_sharding)

        def sample_token(
            logits: Float[Array, " vocabulary"],
            sample_key: Key[Array, ""],
        ) -> Int[Array, ""]:
            return jax.random.categorical(sample_key, logits)

        def step_after_token(
            state: DecodingState,
            chosen_token_ids: Int[Array, " batch"],
            processed_logits: Float[Array, "batch vocabulary"],
            decoding_key: Key[Array, "..."],
        ) -> tuple[DecodingState, GenerationStepResults]:
            next_token_ids = jnp.where(state.stop_flags, stopped_token_ids, chosen_token_ids)
            next_sampling_policy = call_vmapped(
                lambda policy, token_id, should_count: policy.with_next_token_count(token_id, should_count),
                state.sampling_policy,
                next_token_ids,
                jnp.logical_not(state.stop_flags),
            )

            if num_top_logits_to_return is None:
                top_k_token_ids = None
                top_k_token_logits = None
            else:
                top_k_token_logits, top_k_token_ids = jax.lax.top_k(processed_logits, num_top_logits_to_return)

            next_token_indices = state.last_token_indices + 1
            next_stop_flags = state.stop_flags | jnp.any(next_token_ids[:, None] == eos_token_ids[None, :], axis=-1)
            decoder_result = self.decoder(
                token_ids=next_token_ids[:, None],
                token_positions=next_token_indices[:, None],
                state=state.state,
                return_updated_state=True,
                forward_pass_config=decode_forward_pass_config,
                keychain=Keychain(
                    vmapped_keys=decoding_key,
                    batch_key=decoding_keychain.batch_key,
                    sharding_config=decoding_keychain.sharding_config,
                ),
            )
            assert decoder_result.updated_state is not None

            return (
                DecodingState(
                    last_token_logits=decoder_result.logits[:, 0, :].astype(jnp.float32),
                    last_token_indices=next_token_indices,
                    state=decoder_result.updated_state,
                    stop_flags=next_stop_flags,
                    sampling_policy=next_sampling_policy,
                ),
                GenerationStepResults(
                    token_ids=next_token_ids,
                    top_k_token_ids=top_k_token_ids,
                    top_k_token_logits=top_k_token_logits,
                ),
            )

        def sample_step(
            state: DecodingState,
            step_keys: tuple[Key[Array, " batch"], Key[Array, "..."]],
        ) -> tuple[Int[Array, " batch"], Float[Array, "batch vocabulary"], Key[Array, "..."]]:
            sampling_keys, decoding_key = step_keys
            processed_logits = call_vmapped(
                lambda policy, logits: policy.process_logits(logits),
                state.sampling_policy,
                state.last_token_logits.astype(jnp.float32),
            )
            sampled_token_ids = call_vmapped(sample_token, processed_logits, sampling_keys)
            return sampled_token_ids, processed_logits, decoding_key

        sampling_keys = sampling_keychain.rolling_broadcast(
            (max_output_length, batch_size),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys
        sampling_keys = jax.device_put(
            sampling_keys,
            self.sharding_config.make_sharding((None, batch_axis)),
        )
        decoding_keys = decoding_keychain.rolling_broadcast(
            (max_output_length, *decoding_keychain.vmapped_keys.shape),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys
        step_inputs = (sampling_keys, decoding_keys)

        if thinking_token_ids is None:

            def loop_iteration(
                state: DecodingState,
                step_keys: tuple[Key[Array, " batch"], Key[Array, "..."]],
            ) -> tuple[DecodingState, GenerationStepResults]:
                sampled_token_ids, processed_logits, decoding_key = sample_step(state, step_keys)
                return step_after_token(state, sampled_token_ids, processed_logits, decoding_key)

            _, generated = jax.lax.scan(loop_iteration, initial_state, step_inputs)
        else:
            assert generation_config is not None
            assert generation_config.thinking_budget is not None
            budget = generation_config.thinking_budget
            tag_ids = jax.device_put(
                jnp.asarray(thinking_token_ids, dtype=jnp.int32),
                self.sharding_config.make_sharding((None,)),
            )
            recent_token_ids = jax.device_put(
                jnp.full((batch_size, len(thinking_token_ids)), MISSING_TOKEN_ID, dtype=jnp.int32),
                self.sharding_config.make_sharding((batch_axis, None)),
            )
            unclosed_token_count = jax.device_put(
                jnp.zeros((batch_size,), dtype=jnp.int32),
                batch_vector_sharding,
            )
            thinking_closed = jax.device_put(
                initial_thinking_closed,
                batch_vector_sharding,
            )

            def loop_iteration_with_budget(
                carry: tuple[
                    DecodingState,
                    Int[Array, "batch tag"],
                    Int[Array, " batch"],
                    Bool[Array, " batch"],
                ],
                step_keys: tuple[Key[Array, " batch"], Key[Array, "..."]],
            ) -> tuple[
                tuple[DecodingState, Int[Array, "batch tag"], Int[Array, " batch"], Bool[Array, " batch"]],
                GenerationStepResults,
            ]:
                state, recent, unclosed, closed = carry
                sampled_token_ids, processed_logits, decoding_key = sample_step(state, step_keys)
                chosen_token_ids = apply_thinking_budget(
                    sampled_token_ids,
                    state.stop_flags,
                    recent,
                    unclosed,
                    closed,
                    tag_ids,
                    budget,
                )
                new_state, results = step_after_token(state, chosen_token_ids, processed_logits, decoding_key)
                new_recent, new_unclosed, new_closed = advance_thinking_state(
                    recent,
                    unclosed,
                    closed,
                    results.token_ids,
                    state.stop_flags,
                    tag_ids,
                )
                return (new_state, new_recent, new_unclosed, new_closed), results

            _, generated = jax.lax.scan(
                loop_iteration_with_budget,
                (initial_state, recent_token_ids, unclosed_token_count, thinking_closed),
                step_inputs,
            )

        token_ids = rearrange(generated.token_ids, "step batch -> batch step")
        if num_top_logits_to_return is None:
            top_k_token_ids = None
            top_k_token_logits = None
        else:
            top_k_token_ids = rearrange(generated.top_k_token_ids, "step batch k -> batch step k")
            top_k_token_logits = rearrange(generated.top_k_token_logits, "step batch k -> batch step k")

        return GenerationResults(
            token_ids=token_ids,
            top_k_token_ids=top_k_token_ids,
            top_k_token_logits=top_k_token_logits,
        )

    def reply(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        keychain: Keychain,
    ) -> AssistantMessage:
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        token_ids = jax.device_put(
            jnp.asarray(
                self.token_codec.encode_request(messages, reasoning_effort=reasoning_effort),
                dtype=jnp.int32,
            )[None, :],
            self.sharding_config.make_sharding((batch_axis, None)),
        )
        response_ids = self.generate_tokens(
            token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            keychain=keychain,
        ).token_ids[0]
        return self.token_codec.decode_response(self.trim_at_eos(response_ids.tolist()))

    def stream_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        keychain: Keychain,
    ) -> Iterable[Int[Array, ""]]:
        if max_output_length < 1:
            raise ValueError("max_output_length must be at least 1.")
        thinking_budget = None if generation_config is None else generation_config.thinking_budget
        thinking_token_ids = resolve_end_of_thinking_token_ids(
            thinking_budget=thinking_budget,
            end_of_thinking_tag=self.token_codec.config.end_of_thinking_tag,
            encode_tag=self.token_codec.encode_text,
        )
        already_closed = False
        if thinking_token_ids is not None:
            prompt_text = self.token_codec.decode_tokens([int(token_id) for token_id in prompt_token_ids.tolist()])
            already_closed = prompt_closes_thinking(prompt_text, self.token_codec.config.end_of_thinking_tag)
        thinking_progress = (
            None if thinking_token_ids is None else ThinkingProgress.start(thinking_token_ids, closed=already_closed)
        )
        if prefill_forward_pass_config is None:
            prefill_forward_pass_config = DecoderForwardPassConfig.for_inference()
        if decode_forward_pass_config is None:
            decode_forward_pass_config = DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN)

        if eos_token_ids is not None:
            stop_token_ids = eos_token_ids
        else:
            stop_token_ids = jnp.asarray(self.config.generation_config.stop_token_ids, dtype=jnp.int32)

        (input_length,) = prompt_token_ids.shape
        padded_input_length = next(
            (length for length in _COMPILED_PROMPT_LENGTHS if length >= input_length),
            None,
        )
        if padded_input_length is None:
            raise ValueError(f"Input sequence length {input_length} exceeds largest compiled prompt bucket.")

        padded_token_ids = jnp.zeros((padded_input_length,), dtype=prompt_token_ids.dtype)
        padded_token_ids = padded_token_ids.at[:input_length].set(prompt_token_ids)
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        batched_token_ids = jax.device_put(
            padded_token_ids[None, :],
            self.sharding_config.make_sharding((batch_axis, None)),
        )
        length_without_padding = jax.device_put(
            jnp.asarray([input_length], dtype=jnp.int32),
            self.sharding_config.make_sharding((batch_axis,)),
        )

        sampling_policy = self.default_sampling_policy()
        if generation_config is not None:
            sampling_policy = generation_config.default_policy()
        if sampling_policy.has_count_penalties:
            sampling_policy = sampling_policy.with_prompt_token_counts(
                padded_token_ids,
                jnp.asarray(input_length, dtype=jnp.int32),
                self.decoder.vocab_size,
            )

        prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
        prefill_results = self.prefill_tokens(
            batched_token_ids,
            padded_input_length + max_output_length + 1,
            length_without_padding,
            prefill_forward_pass_config,
            keychain=prefill_keychain,
        )

        last_token_logits = prefill_results.last_token_logits[0]
        last_token_index = prefill_results.last_token_indices[0]
        state = prefill_results.state
        sampling_keys = sampling_keychain.rolling_broadcast(
            (max_output_length, 1),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys[:, 0]
        decoding_keys = decoding_keychain.rolling_broadcast(
            (max_output_length, 1),
            mode=KeychainBroadcastMode.PREFIX,
        ).vmapped_keys
        for sampling_key, decoding_key in zip(sampling_keys, decoding_keys, strict=True):
            processed_logits = sampling_policy.process_logits(last_token_logits.astype(jnp.float32))
            next_token_id = jax.random.categorical(sampling_key, processed_logits)
            if thinking_progress is not None and thinking_token_ids is not None and thinking_budget is not None:
                forced_token_id = thinking_progress.forced_token_id(thinking_token_ids, thinking_budget)
                if forced_token_id is not None:
                    # A host scalar does not carry the sampled id's NamedSharding, and the
                    # embedding lookup rejects it. Place the forced id on that same sharding.
                    next_token_id = reshard_as(
                        jnp.asarray(forced_token_id, dtype=next_token_id.dtype),
                        next_token_id,
                    )
            yield next_token_id

            if bool(jnp.any(next_token_id == stop_token_ids).item()):
                return

            if thinking_progress is not None and thinking_token_ids is not None:
                thinking_progress = thinking_progress.after_token(int(next_token_id), thinking_token_ids)

            sampling_policy = sampling_policy.with_next_token_count(next_token_id)
            next_token_index = last_token_index + 1
            decoder_result = self.decoder(
                token_ids=next_token_id.reshape(1, 1),
                token_positions=next_token_index.reshape(1, 1),
                state=state,
                return_updated_state=True,
                forward_pass_config=decode_forward_pass_config,
                keychain=Keychain(
                    vmapped_keys=decoding_key,
                    batch_key=decoding_keychain.batch_key,
                    sharding_config=decoding_keychain.sharding_config,
                ),
            )
            assert decoder_result.updated_state is not None

            last_token_logits = decoder_result.logits[0, 0, :].astype(jnp.float32)
            last_token_index = next_token_index
            state = decoder_result.updated_state

    def stream_reply_text(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        keychain: Keychain,
    ) -> Iterable[str]:
        token_ids = jnp.asarray(
            self.token_codec.encode_request(messages, reasoning_effort=reasoning_effort),
            dtype=jnp.int32,
        )
        response_token_ids: list[int] = []
        previous_text = ""
        for token_id in self.stream_tokens(
            token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            keychain=keychain,
        ):
            response_token_ids.append(int(token_id.item()))
            current_text = self.token_codec.decode_tokens(response_token_ids, hide_invalid_utf_chars=True)
            yield current_text[len(previous_text) :]
            previous_text = current_text
