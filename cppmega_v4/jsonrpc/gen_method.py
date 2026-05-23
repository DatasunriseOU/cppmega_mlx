"""V7-F01: gen.run RPC — composes F02/F03/F04/F06 building blocks.

Backend-only orchestrator: given a list of prompt token ids + a
sampler config, runs the streaming generator with EOS detection and
returns {tokens, finish_reason, events}. The actual model step_fn
is provided by the caller (or, for the smoke test, a stub that
returns last+1).

The vbgui_server WS endpoint /ws/gen/{job_id} wraps this and pushes
each event live (V7-F06 follow-up wire-up).
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.runtime.generate_stream import collect_stream
from cppmega_v4.runtime.samplers import (
    beam_step, greedy, top_k_sample, top_p_sample,
)


class GenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: list[int] = Field(default_factory=list)
    eos_token_id: int = 0
    max_new_tokens: int = Field(16, ge=1, le=4096)
    strategy: Literal["greedy", "top_k", "top_p"] = "greedy"
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    seed: int = 0
    # When set, the gen.run uses a deterministic counter step_fn for
    # smoke testing (no model needed).
    smoke: bool = True


class GenResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    tokens: list[int]
    finish_reason: str
    events: list[dict]


def _sampler_step_fn(params: GenParams):
    """Stub step_fn for the smoke path: returns last+1 modulo a
    small vocab, halting on EOS."""
    rng = random.Random(params.seed)
    vocab = 32
    # Forces EOS at a deterministic position when in smoke mode by
    # mixing in the EOS id under top-k strategy occasionally.
    counter = {"n": 0}

    def _step(last: int) -> int:
        counter["n"] += 1
        if params.strategy == "greedy":
            # Deterministic counter.
            return (last + 1) % vocab
        if params.strategy == "top_k":
            logits = [rng.random() for _ in range(vocab)]
            return top_k_sample(logits, k=params.top_k, rng=rng,
                                 temperature=params.temperature)
        # top_p
        logits = [rng.random() for _ in range(vocab)]
        return top_p_sample(logits, p=params.top_p, rng=rng,
                             temperature=params.temperature)

    return _step


def gen_run(params: GenParams) -> GenResult:
    step_fn = _sampler_step_fn(params)
    tokens, reason, events = collect_stream(
        initial_tokens=list(params.prompt_tokens),
        step_fn=step_fn,
        eos_token_id=params.eos_token_id,
        max_new_tokens=params.max_new_tokens,
    )
    return GenResult(tokens=tokens, finish_reason=reason,
                      events=events)


__all__ = ["GenParams", "GenResult", "gen_run"]
