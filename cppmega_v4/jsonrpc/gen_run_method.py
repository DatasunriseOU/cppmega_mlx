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
from cppmega_v4.runtime import gen_event_bus
from cppmega_v4.runtime.generate_stream import stream_generate
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
    # V7-F02: when > 0, gen.run instantiates a KVCache with this many
    # layers, appends one synthetic (B=1, S=1, H=head_dim) row per
    # decode step, and reports total_bytes + per-layer length in
    # GenRunResult.kv_cache. 0 disables KV-cache reporting.
    kv_cache_layers: int = Field(0, ge=0, le=128)
    kv_cache_head_dim: int = Field(16, ge=1, le=4096)
    # V7-F06: optional job_id; when set, gen_run publishes each token
    # onto gen_event_bus so /ws/gen/{job_id} subscribers see streaming.
    job_id: str | None = None


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
    # V7-F02: KVCache observability. None when kv_cache_layers=0.
    kv_cache: dict | None = None


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
    # V7-F02: optional KVCache wiring. When kv_cache_layers > 0 we
    # instantiate a KVCache and append one synthetic (B=1, S=1, H_kv)
    # row per token per layer so the user can see the cache grow.
    kv_cache_state: dict | None = None
    kv_obj = None
    if params.kv_cache_layers > 0:
        import mlx.core as mx
        from cppmega_v4.runtime.kv_cache import KVCache
        kv_obj = KVCache(num_layers=params.kv_cache_layers)
        kv_cache_state = {"num_layers": params.kv_cache_layers,
                          "head_dim": params.kv_cache_head_dim,
                          "growth_events": 0,
                          "total_bytes": 0,
                          "lengths_per_layer": []}

    inner_step = _build_step_fn(params)

    def _step(last: int) -> int:
        tok = inner_step(last)
        if kv_obj is not None:
            import mlx.core as mx
            new_k = mx.zeros(
                (1, 1, params.kv_cache_head_dim), dtype=mx.float32)
            new_v = mx.zeros(
                (1, 1, params.kv_cache_head_dim), dtype=mx.float32)
            for layer in range(params.kv_cache_layers):
                kv_obj.append(layer, new_k, new_v)
            kv_cache_state["growth_events"] += 1
        return tok

    # V7-F06: when a job_id was supplied, route through stream_generate
    # with an on_token callback that publishes onto the bus, then mirror
    # the same collected events for the synchronous return path.
    raw_events: list[dict] = []

    def _on_token(ev: dict) -> None:
        raw_events.append(ev)
        if params.job_id:
            gen_event_bus.publish(params.job_id, ev)

    tokens, reason = stream_generate(
        initial_tokens=list(params.prompt_tokens),
        step_fn=_step,
        eos_token_id=params.eos_token_id,
        max_new_tokens=params.max_new_tokens,
        on_token=_on_token,
    )
    if params.job_id:
        gen_event_bus.publish(params.job_id, None)
    if kv_obj is not None:
        kv_cache_state["total_bytes"] = int(kv_obj.total_bytes())
        kv_cache_state["lengths_per_layer"] = [
            kv_obj.length(i) for i in range(params.kv_cache_layers)
        ]
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
        kv_cache=kv_cache_state,
    )


__all__ = ["GenRunParams", "GenRunResult", "GenRunEvent", "gen_run"]
