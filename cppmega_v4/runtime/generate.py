"""V7-F04: minimal generation loop with EOS detection.

Pure-Python decode loop that wraps a `step_fn(token) -> next_token`
callable and stops when EOS is emitted OR max_new_tokens is reached.
Returns (generated_tokens, finish_reason in {"eos", "length"}).

This is the smallest building block needed to land EOS detection;
the surrounding gen.run RPC + KV-cache + sampler live under V7-F01,
V7-F02, V7-F03.
"""

from __future__ import annotations

from typing import Callable, Literal


FinishReason = Literal["eos", "length"]


def generate_until_eos(
    *,
    initial_tokens: list[int],
    step_fn: Callable[[int], int],
    eos_token_id: int,
    max_new_tokens: int,
) -> tuple[list[int], FinishReason]:
    """Run an autoregressive decode loop with EOS early-exit.

    Args:
        initial_tokens: prompt tokens (returned unchanged at the head
            of the output).
        step_fn: takes the last token id, returns next predicted id.
        eos_token_id: stop when step_fn returns this value.
        max_new_tokens: cap on generated tokens (excluding prompt).

    Returns:
        (full_tokens, finish_reason). finish_reason is "eos" when the
        loop halted on EOS, "length" when it hit max_new_tokens.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    out = list(initial_tokens)
    last = out[-1] if out else 0
    for _ in range(max_new_tokens):
        nxt = step_fn(last)
        out.append(int(nxt))
        if int(nxt) == int(eos_token_id):
            return out, "eos"
        last = int(nxt)
    return out, "length"


__all__ = ["generate_until_eos", "FinishReason"]
