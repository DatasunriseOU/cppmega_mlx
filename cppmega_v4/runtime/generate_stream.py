"""V7-F06: per-token streaming generation event emitter.

Wraps generate_until_eos (V7-F04) with a callback fired per token,
so the gen.run RPC can push WS frames as decoding progresses. The
collector form `collect_stream(...)` returns the same finish_reason
+ full tokens plus a list of every per-token event for tests.
"""

from __future__ import annotations

from typing import Callable, Iterator, Literal

from cppmega_v4.runtime.generate import generate_until_eos

FinishReason = Literal["eos", "length"]


def stream_generate(
    *, initial_tokens: list[int],
    step_fn: Callable[[int], int],
    eos_token_id: int,
    max_new_tokens: int,
    on_token: Callable[[dict], None] | None = None,
) -> tuple[list[int], FinishReason]:
    """Generate tokens; fire on_token({step, token_id, finish_reason})
    for every newly emitted token (including the EOS one when fired)."""
    out: list[int] = list(initial_tokens)
    last = out[-1] if out else 0
    step = 0
    for _ in range(max_new_tokens):
        nxt = int(step_fn(last))
        out.append(nxt)
        ev = {"step": step, "token_id": nxt,
              "finish_reason": "eos" if nxt == int(eos_token_id)
                                else None}
        if on_token:
            on_token(ev)
        step += 1
        if nxt == int(eos_token_id):
            return out, "eos"
        last = nxt
    return out, "length"


def collect_stream(*, initial_tokens, step_fn, eos_token_id,
                    max_new_tokens) -> tuple[list[int], FinishReason,
                                              list[dict]]:
    events: list[dict] = []
    tokens, reason = stream_generate(
        initial_tokens=initial_tokens, step_fn=step_fn,
        eos_token_id=eos_token_id, max_new_tokens=max_new_tokens,
        on_token=events.append,
    )
    return tokens, reason, events


__all__ = ["stream_generate", "collect_stream"]
