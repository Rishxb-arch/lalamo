import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from typer.testing import CliRunner

from lalamo.inference.batch_scheduler import BatchSchedulerConfig, ContinuousBatchScheduler
from lalamo.initializer import RandomInitializer
from lalamo.main import _chat_generation_config, app
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig, UserMessage
from lalamo.models.language_model import _COMPILED_PROMPT_LENGTHS, GenerationConfig, LanguageModelConfig
from lalamo.models.thinking_budget import (
    ThinkingProgress,
    advance_thinking_state,
    apply_thinking_budget,
    matched_tag_prefix_length,
    matched_tag_prefix_lengths,
    prompt_closes_thinking,
    resolve_end_of_thinking_token_ids,
)
from lalamo.module import Keychain
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import build_tiny_attention_decoder_config

TAG = (7, 8, 9)
TAG_TEXT = "t7 t8 t9"
PROMPT = (1, 2)


def _replay(samples: list[int], tag: tuple[int, ...], budget: int) -> list[int]:
    progress = ThinkingProgress.start(tag)
    remaining = list(samples)
    emitted: list[int] = []
    for _ in range(len(samples) + len(tag) + 1):
        forced_token_id = progress.forced_token_id(tag, budget)
        if forced_token_id is None:
            if not remaining:
                break
            token_id = remaining.pop(0)
        else:
            token_id = forced_token_id
        emitted.append(token_id)
        progress = progress.after_token(token_id, tag)
    return emitted


def _tag_index(token_ids: list[int], tag: tuple[int, ...]) -> int:
    tag_length = len(tag)
    for index in range(len(token_ids) - tag_length + 1):
        if tuple(token_ids[index : index + tag_length]) == tag:
            return index
    raise AssertionError(f"end-of-thinking tag {tag} not found in {token_ids}")


def test_prompt_closes_thinking_only_when_the_tag_ends_the_prompt() -> None:
    assert prompt_closes_thinking("assistant\n<think>\n\n</think>\n\n", "\n</think>")
    assert not prompt_closes_thinking("assistant\n", "\n</think>")
    assert not prompt_closes_thinking("the user wrote </think> earlier\nassistant\n", "\n</think>")
    assert not prompt_closes_thinking("assistant\n</think> and then more", "\n</think>")
    assert not prompt_closes_thinking("assistant\n</think>\n\n", None)
    progress = ThinkingProgress.start(TAG, closed=True)
    assert progress.forced_token_id(TAG, thinking_budget=0) is None


def test_budget_zero_appends_the_tag_before_sampling() -> None:
    assert _replay([4, 5, 6], TAG, budget=0) == [7, 8, 9, 4, 5, 6]
    assert _replay([4, 5], (7,), budget=0) == [7, 4, 5]


def test_natural_close_is_left_unchanged() -> None:
    samples = [1, 7, 8, 9, 4]
    assert _replay(samples, TAG, budget=10) == samples


def test_completing_token_at_the_budget_is_not_duplicated() -> None:
    assert _replay([7, 8, 9, 4], TAG, budget=3) == [7, 8, 9, 4]


def test_partial_tag_is_completed_instead_of_repeated() -> None:
    assert _replay([1, 7, 8, 4, 5], TAG, budget=3) == [1, 7, 8, 9, 4, 5]


def test_repeated_tag_tokens_keep_the_longest_prefix() -> None:
    tag = (7, 8, 7)
    assert _replay([1, 7, 8, 3], tag, budget=3) == [1, 7, 8, 7, 3]
    assert matched_tag_prefix_length((1, 7, 8), tag) == 2
    assert matched_tag_prefix_length((8, 7), tag) == 1


def test_tokens_before_the_tag_stay_within_the_budget() -> None:
    for budget in (0, 1, 2, 4):
        emitted = _replay([1, 2, 3, 4, 5, 6], TAG, budget=budget)
        assert _tag_index(emitted, TAG) <= budget


def _encode_one(text: str) -> Sequence[int]:
    return [len(text)]


def _encode_nothing(text: str) -> Sequence[int]:
    token_ids: list[int] = []
    if text:
        return token_ids
    return token_ids


def test_resolve_rejects_a_missing_or_empty_tag_and_a_negative_budget() -> None:
    assert (
        resolve_end_of_thinking_token_ids(
            thinking_budget=None,
            end_of_thinking_tag=None,
            encode_tag=_encode_one,
        )
        is None
    )
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        resolve_end_of_thinking_token_ids(
            thinking_budget=-1,
            end_of_thinking_tag="</think>",
            encode_tag=_encode_one,
        )
    with pytest.raises(ValueError, match="end_of_thinking_tag"):
        resolve_end_of_thinking_token_ids(
            thinking_budget=4,
            end_of_thinking_tag=None,
            encode_tag=_encode_one,
        )
    with pytest.raises(ValueError, match="empty token sequence"):
        resolve_end_of_thinking_token_ids(
            thinking_budget=4,
            end_of_thinking_tag="</think>",
            encode_tag=_encode_nothing,
        )


def test_array_step_matches_python_progress_for_each_row() -> None:
    tag = TAG
    rows = ([1, 4, 5, 9, 2], [7, 8, 3, 1, 6])
    budget = 3
    progresses = [ThinkingProgress.start(tag) for _ in rows]
    recent = jnp.full((len(rows), len(tag)), -1, dtype=jnp.int32)
    unclosed = jnp.zeros((len(rows),), dtype=jnp.int32)
    closed = jnp.zeros((len(rows),), dtype=jnp.bool_)
    stop_flags = jnp.zeros((len(rows),), dtype=jnp.bool_)
    tag_ids = jnp.asarray(tag, dtype=jnp.int32)

    for step in range(len(rows[0])):
        samples = jnp.asarray([row[step] for row in rows], dtype=jnp.int32)
        chosen = apply_thinking_budget(samples, stop_flags, recent, unclosed, closed, tag_ids, budget)
        for row_index, progress in enumerate(progresses):
            forced_token_id = progress.forced_token_id(tag, budget)
            expected = samples[row_index] if forced_token_id is None else forced_token_id
            assert int(chosen[row_index]) == expected
            progresses[row_index] = progress.after_token(int(chosen[row_index]), tag)
        recent, unclosed, closed = advance_thinking_state(recent, unclosed, closed, chosen, stop_flags, tag_ids)
        for row_index, progress in enumerate(progresses):
            assert progress.unclosed_token_count == int(unclosed[row_index])
            assert progress.closed == bool(closed[row_index])
            assert progress.recent_token_ids == tuple(int(token_id) for token_id in recent[row_index].tolist())

    prefix_lengths = matched_tag_prefix_lengths(recent, tag_ids)
    for row_index, progress in enumerate(progresses):
        assert int(prefix_lengths[row_index]) == matched_tag_prefix_length(progress.recent_token_ids, tag)


def test_single_token_tag_is_forced_once_then_sampling_resumes() -> None:
    tag_ids = jnp.asarray([7], dtype=jnp.int32)
    recent = jnp.full((1, 1), -1, dtype=jnp.int32)
    unclosed = jnp.zeros((1,), dtype=jnp.int32)
    closed = jnp.zeros((1,), dtype=jnp.bool_)
    stop_flags = jnp.zeros((1,), dtype=jnp.bool_)
    forced = apply_thinking_budget(
        jnp.asarray([4], dtype=jnp.int32),
        stop_flags,
        recent,
        unclosed,
        closed,
        tag_ids,
        thinking_budget=0,
    )
    assert int(forced[0]) == 7
    recent, unclosed, closed = advance_thinking_state(recent, unclosed, closed, forced, stop_flags, tag_ids)
    assert bool(closed[0])
    sampled = apply_thinking_budget(
        jnp.asarray([4], dtype=jnp.int32),
        stop_flags,
        recent,
        unclosed,
        closed,
        tag_ids,
        thinking_budget=0,
    )
    assert int(sampled[0]) == 4


def test_stopped_row_is_not_forced_closed() -> None:
    tag_ids = jnp.asarray(TAG, dtype=jnp.int32)
    recent = jnp.full((1, len(TAG)), -1, dtype=jnp.int32)
    unclosed = jnp.asarray([len(TAG)], dtype=jnp.int32)
    closed = jnp.asarray([False])
    stop_flags = jnp.asarray([True])
    chosen = apply_thinking_budget(
        jnp.asarray([4], dtype=jnp.int32),
        stop_flags,
        recent,
        unclosed,
        closed,
        tag_ids,
        thinking_budget=1,
    )
    assert int(chosen[0]) == 4
    new_recent, new_unclosed, new_closed = advance_thinking_state(
        recent,
        unclosed,
        closed,
        jnp.asarray([0], dtype=jnp.int32),
        stop_flags,
        tag_ids,
    )
    assert int(new_unclosed[0]) == len(TAG)
    assert not bool(new_closed[0])
    assert jnp.array_equal(new_recent, recent)


def test_generation_config_override_treats_zero_as_a_real_budget() -> None:
    base = GenerationConfig(temperature=0.3, thinking_budget=4)
    assert base.override_with(GenerationConfig()).thinking_budget == 4
    assert base.override_with(GenerationConfig(thinking_budget=0)).thinking_budget == 0
    assert base.override_with(GenerationConfig(temperature=0.1)).temperature == 0.1
    assert base.override_with(GenerationConfig(temperature=0.1)).thinking_budget == 4


def test_chat_generation_config_preserves_model_sampling_when_only_the_budget_is_set() -> None:
    model_config = GenerationConfig(temperature=0.7, top_k=5)
    assert _chat_generation_config(model_config, temperature=None, thinking_budget=None) is None
    assert _chat_generation_config(model_config, temperature=None, thinking_budget=0) == replace(
        model_config,
        thinking_budget=0,
    )


def test_chat_help_documents_thinking_budget_and_rejects_a_negative_value() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["chat", "--help"])
    assert help_result.exit_code == 0
    assert "--thinking-budget" in help_result.output

    negative_result = runner.invoke(app, ["chat", "missing-model", "--thinking-budget", "-1", "--message", "hi"])
    assert negative_result.exit_code != 0


def _tokenizer() -> Tokenizer:
    vocab = {"[UNK]": 0}
    vocab.update({f"t{token_id}": token_id for token_id in range(1, 32)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _tiny_model(end_of_thinking_tag: str | None) -> LanguageModel:
    sharding_config = ShardingConfig.replicated()
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template="{{ messages[-1].content }}",
            output_parser_regex=None,
            system_role_name="system",
            user_role_name="user",
            assistant_role_name="assistant",
            eos_token=None,
            bos_token=None,
            end_of_thinking_tag=end_of_thinking_tag,
        ),
        decoder_config=build_tiny_attention_decoder_config((None,)),
        generation_config=GenerationConfig(temperature=0.0, stop_token_ids=()),
    )
    return config.init(
        _tokenizer(),
        RandomInitializer(jnp.float32, sharding_config, key=jax.random.key(0)),
    )


def _generate(
    model: LanguageModel,
    prompt: tuple[int, ...],
    generation_config: GenerationConfig | None,
    max_output_length: int,
    seed: int,
) -> list[int]:
    token_ids = jnp.asarray(prompt, dtype=jnp.int32)[None, :]
    with jax.set_mesh(model.sharding_config.mesh):
        generated = model.generate_tokens(
            token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            keychain=Keychain.init(seed, sharding_config=model.sharding_config),
        )
    return [int(token_id) for token_id in generated.token_ids[0].tolist()]


def test_tiny_model_without_a_tag_rejects_a_budget_and_still_generates_without_one() -> None:
    model = _tiny_model(None)
    with pytest.raises(ValueError, match="end_of_thinking_tag"):
        _generate(model, PROMPT, GenerationConfig(temperature=0.0, thinking_budget=2), 4, seed=1)
    assert len(_generate(model, PROMPT, None, 4, seed=1)) == 4


def test_unset_budget_matches_a_budget_the_model_cannot_reach(tmp_path: Path) -> None:
    model = _tiny_model(TAG_TEXT)
    assert tuple(model.token_codec.encode_text(TAG_TEXT)) == TAG
    baseline = _generate(model, PROMPT, None, 6, seed=2)
    explicit = _generate(model, PROMPT, GenerationConfig(temperature=0.0), 6, seed=2)
    unreachable = _generate(model, PROMPT, GenerationConfig(temperature=0.0, thinking_budget=10_000), 6, seed=2)
    assert baseline == explicit == unreachable

    model.save(tmp_path)
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    assert isinstance(payload, dict)
    generation_config = payload["generation_config"]
    assert isinstance(generation_config, dict)
    generation_config.pop("thinking_budget")
    config_path.write_text(json.dumps(payload))
    loaded = LanguageModel.load(tmp_path, ShardingConfig.replicated())
    assert loaded.config.generation_config.thinking_budget is None


def test_tiny_model_forces_the_tag_and_keeps_generating_the_answer() -> None:
    model = _tiny_model(TAG_TEXT)
    unlimited = _generate(model, PROMPT, GenerationConfig(temperature=0.0), 8, seed=4)
    limited = _generate(model, PROMPT, GenerationConfig(temperature=0.0, thinking_budget=2), 8, seed=4)
    tag_index = _tag_index(limited, TAG)
    assert tag_index <= 2
    assert limited[:tag_index] == unlimited[:tag_index]
    assert tuple(limited[tag_index : tag_index + len(TAG)]) == TAG
    assert len(limited) == 8
    assert tag_index + len(TAG) < len(limited)

    skipped = _generate(model, PROMPT, GenerationConfig(temperature=0.0, thinking_budget=0), 6, seed=4)
    assert tuple(skipped[: len(TAG)]) == TAG
    truncated = _generate(model, PROMPT, GenerationConfig(temperature=0.0, thinking_budget=0), 1, seed=4)
    assert truncated == [TAG[0]]


def test_batched_generate_forces_the_tag_on_every_row() -> None:
    model = _tiny_model(TAG_TEXT)
    prompts = jnp.asarray([PROMPT, (3, 4)], dtype=jnp.int32)
    with jax.set_mesh(model.sharding_config.mesh):
        generated = model.generate_tokens(
            prompts,
            generation_config=GenerationConfig(temperature=0.0, thinking_budget=0),
            max_output_length=5,
            keychain=Keychain.init(7, sharding_config=model.sharding_config),
        )
    assert generated.token_ids.shape == (2, 5)
    for row in generated.token_ids.tolist():
        assert tuple(int(token_id) for token_id in row[: len(TAG)]) == TAG


def _stream(
    model: LanguageModel,
    prompt: tuple[int, ...],
    generation_config: GenerationConfig,
    limit: int,
    seed: int,
) -> list[int]:
    token_ids = []
    for token_id in model.stream_tokens(
        jnp.asarray(prompt, dtype=jnp.int32),
        generation_config=generation_config,
        max_output_length=limit,
        keychain=Keychain.init(seed, sharding_config=model.sharding_config),
    ):
        token_ids.append(int(token_id))
        if len(token_ids) >= limit:
            break
    return token_ids


def test_already_closed_prompt_is_not_forced_again() -> None:
    model = _tiny_model(TAG_TEXT)
    closed_prompt = (*PROMPT, *TAG)
    assert prompt_closes_thinking(model.token_codec.decode_tokens(list(closed_prompt)), TAG_TEXT)
    budget = GenerationConfig(temperature=0.0, thinking_budget=0)
    plain = GenerationConfig(temperature=0.0)
    assert _generate(model, closed_prompt, budget, 5, seed=5) == _generate(model, closed_prompt, plain, 5, seed=5)
    assert _stream(model, closed_prompt, budget, 4, seed=5) == _stream(model, closed_prompt, plain, 4, seed=5)

    open_row = (*PROMPT, 0, 0, 0)
    prompts = jnp.asarray([open_row, closed_prompt], dtype=jnp.int32)
    lengths = jnp.asarray([len(PROMPT), len(closed_prompt)], dtype=jnp.int32)
    with jax.set_mesh(model.sharding_config.mesh):
        forced = model.generate_tokens(
            prompts,
            generation_config=budget,
            prompt_lengths_without_padding=lengths,
            max_output_length=5,
            keychain=Keychain.init(6, sharding_config=model.sharding_config),
        ).token_ids
        baseline = model.generate_tokens(
            prompts,
            generation_config=plain,
            prompt_lengths_without_padding=lengths,
            max_output_length=5,
            keychain=Keychain.init(6, sharding_config=model.sharding_config),
        ).token_ids
    assert tuple(int(token_id) for token_id in forced[0, : len(TAG)].tolist()) == TAG
    assert forced[1].tolist() == baseline[1].tolist()


def test_stream_forces_the_tag_without_an_explicit_mesh() -> None:
    model = _tiny_model(TAG_TEXT)
    streamed = [
        int(token_id)
        for token_id in model.stream_tokens(
            jnp.asarray(PROMPT, dtype=jnp.int32),
            generation_config=GenerationConfig(temperature=0.0, thinking_budget=0),
            max_output_length=4,
            keychain=Keychain.init(11, sharding_config=model.sharding_config),
        )
    ]
    assert tuple(streamed[: len(TAG)]) == TAG


def test_stream_matches_generate_when_a_budget_is_set() -> None:
    model = _tiny_model(TAG_TEXT)
    prompt = jnp.asarray(PROMPT, dtype=jnp.int32)
    padded_length = next(length for length in _COMPILED_PROMPT_LENGTHS if length >= prompt.size)
    padded_prompt = jnp.pad(prompt, (0, padded_length - prompt.size))
    generation_config = GenerationConfig(temperature=0.0, thinking_budget=0)
    keychain = Keychain.init(8, sharding_config=model.sharding_config)
    with jax.set_mesh(model.sharding_config.mesh):
        eager = model.generate_tokens(
            padded_prompt[None, :],
            generation_config=generation_config,
            prompt_lengths_without_padding=jnp.asarray([prompt.size], dtype=jnp.int32),
            max_output_length=5,
            keychain=keychain,
        ).token_ids[0]
        streamed = [
            int(token_id)
            for token_id in model.stream_tokens(
                prompt,
                generation_config=generation_config,
                max_output_length=5,
                keychain=keychain,
            )
        ]
        reply = "".join(
            model.stream_reply_text(
                [UserMessage("t1 t2")],
                generation_config=generation_config,
                max_output_length=4,
                keychain=Keychain.init(9, sharding_config=model.sharding_config),
            ),
        )
    assert [int(token_id) for token_id in eager.tolist()] == streamed
    assert tuple(streamed[: len(TAG)]) == TAG
    assert reply.startswith(model.token_codec.decode_tokens(list(TAG)))


def test_continuous_batching_rejects_a_thinking_budget() -> None:
    model = _tiny_model(TAG_TEXT)
    with pytest.raises(ValueError, match="continuous batching"):
        next(
            ContinuousBatchScheduler(model=model).generate_tokens_many(
                [list(PROMPT)],
                generation_config=GenerationConfig(thinking_budget=4),
                batch_scheduler_config=BatchSchedulerConfig(batch_size=1, padded_length=8, max_output_length=4),
            ),
        )
