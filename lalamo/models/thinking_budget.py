"""Cap a reasoning section, then continue with the answer.

`thinking_budget` counts generated tokens before the model's end-of-thinking
tag. The tag itself is not counted, whether the model emits it or generation
appends it. When the count reaches the budget, the remaining tag tokens are
appended and sampling continues, so the answer is still produced. A budget of
zero appends the tag before any reasoning token. `None` leaves generation
unchanged.

The forced tokens are real sequence positions: they are fed through the
decoder, they are not counted again if the model had already started the tag,
and they do count toward `max_output_length`. Generation still stops at EOS.
If EOS or the length limit arrives before the tag fits, the tag can be cut
short.

This assumes the prompt left the reasoning section open. A template that
already closed thinking (reasoning disabled) will receive another copy of the
tag if a budget is also set. Models with no `end_of_thinking_tag` raise
`ValueError` when a budget is set, because there is no boundary to force.
Continuous batching does not implement the cap and rejects it instead of
ignoring it.

Top-k ids and logits returned by `generate_tokens` stay the distribution
sampled before a forced transition.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Int

__all__ = [
    "MISSING_TOKEN_ID",
    "ThinkingProgress",
    "advance_thinking_state",
    "apply_thinking_budget",
    "matched_tag_prefix_length",
    "matched_tag_prefix_lengths",
    "resolve_end_of_thinking_token_ids",
]

MISSING_TOKEN_ID = -1


def resolve_end_of_thinking_token_ids(
    *,
    thinking_budget: int | None,
    end_of_thinking_tag: str | None,
    encode_tag: Callable[[str], Sequence[int]],
) -> tuple[int, ...] | None:
    if thinking_budget is None:
        return None
    if thinking_budget < 0:
        raise ValueError("thinking_budget must be greater than or equal to zero.")
    if end_of_thinking_tag is None:
        raise ValueError(
            "thinking_budget requires a model with an end_of_thinking_tag. "
            "This model does not define one, so generation cannot close a reasoning section.",
        )
    token_ids = tuple(encode_tag(end_of_thinking_tag))
    if not token_ids:
        raise ValueError("end_of_thinking_tag encodes to an empty token sequence.")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("end_of_thinking_tag encoded to a negative token id.")
    return token_ids


def matched_tag_prefix_length(recent_token_ids: tuple[int, ...], tag_token_ids: tuple[int, ...]) -> int:
    """Longest suffix of `recent_token_ids` that is a proper prefix of the tag."""
    best = 0
    for length in range(1, len(tag_token_ids)):
        if recent_token_ids[-length:] == tag_token_ids[:length]:
            best = length
    return best


def matched_tag_prefix_lengths(
    recent_token_ids: Int[Array, "batch tag"],
    tag_token_ids: Int[Array, " tag"],
) -> Int[Array, " batch"]:
    batch_size, tag_length = recent_token_ids.shape
    best = jnp.zeros((batch_size,), dtype=jnp.int32)
    for length in range(1, tag_length):
        suffix = recent_token_ids[:, tag_length - length :]
        matches = jnp.all(suffix == tag_token_ids[:length][None, :], axis=-1)
        best = jnp.where(matches, jnp.asarray(length, dtype=jnp.int32), best)
    return best


@dataclass(frozen=True)
class ThinkingProgress:
    unclosed_token_count: int
    recent_token_ids: tuple[int, ...]
    closed: bool

    @staticmethod
    def start(tag_token_ids: tuple[int, ...]) -> "ThinkingProgress":
        return ThinkingProgress(
            unclosed_token_count=0,
            recent_token_ids=(MISSING_TOKEN_ID,) * len(tag_token_ids),
            closed=False,
        )

    def forced_token_id(self, tag_token_ids: tuple[int, ...], thinking_budget: int) -> int | None:
        if self.closed or self.unclosed_token_count < thinking_budget:
            return None
        return tag_token_ids[matched_tag_prefix_length(self.recent_token_ids, tag_token_ids)]

    def after_token(self, token_id: int, tag_token_ids: tuple[int, ...]) -> "ThinkingProgress":
        if self.closed:
            return self
        recent_token_ids = (*self.recent_token_ids[1:], token_id)
        return ThinkingProgress(
            unclosed_token_count=self.unclosed_token_count + 1,
            recent_token_ids=recent_token_ids,
            closed=recent_token_ids == tag_token_ids,
        )


def apply_thinking_budget(
    sampled_token_ids: Int[Array, " batch"],
    stop_flags: Bool[Array, " batch"],
    recent_token_ids: Int[Array, "batch tag"],
    unclosed_token_count: Int[Array, " batch"],
    thinking_closed: Bool[Array, " batch"],
    tag_token_ids: Int[Array, " tag"],
    thinking_budget: int,
) -> Int[Array, " batch"]:
    prefix_lengths = matched_tag_prefix_lengths(recent_token_ids, tag_token_ids)
    forced_token_ids = tag_token_ids[prefix_lengths]
    should_force = (~thinking_closed) & (~stop_flags) & (unclosed_token_count >= thinking_budget)
    return jnp.where(should_force, forced_token_ids, sampled_token_ids)


def advance_thinking_state(
    recent_token_ids: Int[Array, "batch tag"],
    unclosed_token_count: Int[Array, " batch"],
    thinking_closed: Bool[Array, " batch"],
    emitted_token_ids: Int[Array, " batch"],
    stop_flags: Bool[Array, " batch"],
    tag_token_ids: Int[Array, " tag"],
) -> tuple[Int[Array, "batch tag"], Int[Array, " batch"], Bool[Array, " batch"]]:
    live = (~stop_flags) & (~thinking_closed)
    shifted = jnp.concatenate([recent_token_ids[:, 1:], emitted_token_ids[:, None]], axis=1)
    new_recent_token_ids = jnp.where(live[:, None], shifted, recent_token_ids)
    new_unclosed_token_count = jnp.where(live, unclosed_token_count + 1, unclosed_token_count)
    completed = live & jnp.all(new_recent_token_ids == tag_token_ids[None, :], axis=-1)
    return new_recent_token_ids, new_unclosed_token_count, thinking_closed | completed
