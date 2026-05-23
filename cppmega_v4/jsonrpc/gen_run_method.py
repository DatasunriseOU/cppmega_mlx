"""V7-F01: gen.run RPC — autoregressive generation orchestrator.

Composes the F02/F03/F04 building blocks that already shipped as
helpers (cppmega_v4/runtime/{generate, generate_stream, samplers}.py)
but had no top-level entry the UI could call:

  * generate_until_eos / stream_generate — the loop
  * samplers.{greedy, top_k_sample, top_p_sample} — token selection
  * (later F02) KVCache — incremental decode

This handler runs a pure-python decode using a deterministic counter
step_fn for the smoke path (UI Token Lab) plus a sampler-driven
random walk for the realistic strategies. F06 (token streaming WS)
adds an event-emit callback on top of this same orchestrator.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.runtime.generate_stream import collect_stream
from cppmega_v4.runtime.samplers import (
    greedy, temperature_sample, top_k_sample, top_p_sample,
)


SamplerStrategy = Literal["greedy", "temperature", "top_k", "top_p"]


class GenRunParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: list[int] = Field(default_factory=list)
    eos_token_id: int = 0
    max_new_tokens: int = Field(16, ge=1, le=4096)
    strategy: SamplerStrategy = "greedy"
    temperature: float = Field(1.0, gt=0.0, le=10.0)
    top_k: int = Field(50, ge=1, le=2048)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    seed: int = 0
    vocab_size: int = Field(32, ge=2, le=200_000)
    # When true (default), uses a deterministic synthetic-logits step_fn
    # so the smoke path does not require a model. F02 follow-up swaps
    # this for real hybrid_lm.step_fn with KVCache.
    smoke: bool = True


class GenRunEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    token: int


class GenRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[int]
    finish_reason: Literal["eos", "length", "aborted"]
    events: list[GenRunEvent] = Field(default_factory=list)
    elapsed_ms: float
    strategy: SamplerStrategy
    smoke: bool


def _build_step_fn(params: GenRunParams):
    rng = random.Random(params.seed)
    vocab = max(2, params.vocab_size)

    def _logits(last: int) -> list[float]:
        # Deterministic mixture so smoke runs are reproducible.
        base = [rng.random() for _ in range(vocab)]
        # Boost a small neighborhood of last so greedy walks make sense.
        for off in range(-1, 2):
            i = (last + off) % vocab
            base[i] += 0.5
        return base

    def _step(last: int) -> int:
        if params.strategy == "greedy":
            return greedy(_logits(last))
        if params.strategy == "temperature":
            return temperature_sample(
                _logits(last), temperature=params.temperature, rng=rng)
        if params.strategy == "top_k":
            return top_k_sample(
                _logits(last), k=params.top_k, rng=rng,
                temperature=params.temperature)
        # top_p
        return top_p_sample(
            _logits(last), p=params.top_p, rng=rng,
            temperature=params.temperature)

    return _step


def gen_run(
    params: GenRunParams, *, cache: LRUCache | None = None,
) -> GenRunResult:
    import time as _time
    t0 = _time.perf_counter()
    step_fn = _build_step_fn(params)
    tokens, reason, raw_events = collect_stream(
        initial_tokens=list(params.prompt_tokens),
        step_fn=step_fn,
        eos_token_id=params.eos_token_id,
        max_new_tokens=params.max_new_tokens,
    )
    # stream_generate emits {step, token_id, finish_reason}; the wire
    # field is `token` so the UI doesn't have to know the inner name.
    events = [GenRunEvent(step=int(e.get("step", i)),
                           token=int(e.get("token_id", 0)))
              for i, e in enumerate(raw_events)]
    return GenRunResult(
        tokens=tokens,
        finish_reason=reason,
        events=events,
        elapsed_ms=round((_time.perf_counter() - t0) * 1000.0, 4),
        strategy=params.strategy,
        smoke=params.smoke,
    )


__all__ = ["GenRunParams", "GenRunResult", "GenRunEvent", "gen_run"]
