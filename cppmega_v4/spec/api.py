"""GUI-facing public API: verify + estimate in one call, adapter
suggestion lookup, ergonomic dim_env builders.

This module is what the visual-builder front-end actually imports. It
wires together Stages A-D into the three operations the GUI cares about:

  - :func:`verify_and_estimate` — on every graph mutation, return both
    the diagnostics list (red/yellow/green per edge) and the memory
    report (bar at the top: ``18.2 / 80 GB``).
  - :func:`suggest_adapters` — when the user hovers a red edge, propose
    the adapter chain that would close it.
  - :func:`suggest_dim_env` — sensible default named-dim env for a given
    preset name, so the GUI doesn't have to ship a full editor before
    the first verify lands.

Designed to run in <50 ms per call for any of the 12 architecture
presets at production-scale dim envs (B=1, S=4096, H=4096) — the
GUI workflow test in tests/v4/test_vbspec_stage_e.py enforces this.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cppmega_v4.fusion import FusionRegionPlan, plan_fusion_regions
from cppmega_v4.fusion.brick_graph import BrickGraph
from cppmega_v4.spec.adapters import (
    AdapterSuggestion,
    suggest_adapter_chain,
)
from cppmega_v4.spec.memory_report import (
    MemoryReport,
    estimate_memory,
)
from cppmega_v4.spec.resolver import (
    ResolvedBrickGraph,
    resolve_shapes,
)


# ---------------------------------------------------------------------------
# Default named-dim envs per preset
# ---------------------------------------------------------------------------


# Production-scale defaults. The GUI ships these so a fresh-open model
# already has a sensible env without the user touching dim sliders.
# Override by passing your own ``dim_env`` to verify_and_estimate.
_PROD_BASE: Mapping[str, int] = {
    "B": 1, "S": 4096, "H": 4096,
    "nh": 32, "nkv": 4, "head_dim": 128,
    "num_experts": 8, "top_k": 2,
}


_PROD_MLA_EXTRAS: Mapping[str, int] = {
    "q_lora_rank": 1536, "kv_lora_rank": 512,
    "qk_rope_head_dim": 64, "qk_nope_head_dim": 128, "v_head_dim": 128,
}


_PROD_GEMMA_EXTRAS = {"sliding_window_size": 1024}
_PROD_NEMOTRON_EXTRAS = {"d_state": 64}
_PROD_ZAYA_EXTRAS = {"fine_window": 256, "coarse_block_size": 16}


_PRESET_DIM_ENVS: Mapping[str, Mapping[str, int]] = {
    "qwen3_next":        dict(_PROD_BASE),
    "kimi_linear":       {**_PROD_BASE, **_PROD_MLA_EXTRAS},
    "kimi_k2":           {**_PROD_BASE, **_PROD_MLA_EXTRAS},
    "deepseek_v3":       {**_PROD_BASE, **_PROD_MLA_EXTRAS},
    "deepseek_v4_flash": dict(_PROD_BASE),
    "gemma4":            {**_PROD_BASE, **_PROD_GEMMA_EXTRAS},
    "mistral4":          {**_PROD_BASE, **_PROD_MLA_EXTRAS},
    "ling26":            {**_PROD_BASE, **_PROD_MLA_EXTRAS},
    "longcat":           {**_PROD_BASE, **_PROD_MLA_EXTRAS},
    "nemotron3":         {**_PROD_BASE, **_PROD_NEMOTRON_EXTRAS},
    "zaya1":             {**_PROD_BASE, **_PROD_ZAYA_EXTRAS},
    "arcee_trinity":     {**_PROD_BASE, **_PROD_GEMMA_EXTRAS},
}


def suggest_dim_env(preset_name: str | None = None) -> dict[str, int]:
    """Return a sensible default named-dim env.

    With ``preset_name=None``, returns the generic production-scale
    base env (B=1, S=4096, H=4096, nh=32, nkv=4, head_dim=128,
    num_experts=8, top_k=2). With a preset name, returns the env that
    includes any preset-specific extra dims (MLA LoRA ranks, sliding
    window size, etc.).
    """
    if preset_name is None:
        return dict(_PROD_BASE)
    if preset_name in _PRESET_DIM_ENVS:
        return dict(_PRESET_DIM_ENVS[preset_name])
    # GalCov-A: dynamically derive env from the preset's spec kinds so
    # that newly-added preset factories don't need a manual table entry.
    try:
        from cppmega_v4.architectures import PRESETS, build_preset_specs
    except ImportError:
        raise KeyError(
            f"unknown preset {preset_name!r}; "
            f"available: {sorted(_PRESET_DIM_ENVS)}"
        ) from None
    if preset_name not in PRESETS:
        raise KeyError(
            f"unknown preset {preset_name!r}; "
            f"available: {sorted(set(_PRESET_DIM_ENVS) | set(PRESETS))}"
        )
    specs = build_preset_specs(preset_name, hidden_size=_PROD_BASE["H"])
    # V7-Q04: tolerate parallel-block dicts ({"parallel": [...]} without
    # a top-level "kind"). Walk into the branch and collect kinds from
    # the children. Preserves the existing flat-spec contract for
    # linear presets.
    def _collect_kinds(items: list) -> set[str]:
        out: set[str] = set()
        for it in items:
            if "kind" in it:
                out.add(it["kind"])
            elif "parallel" in it and isinstance(it["parallel"], list):
                out |= _collect_kinds(it["parallel"])
        return out
    kinds = _collect_kinds(specs)
    env = dict(_PROD_BASE)
    # Pull in extras based on which brick categories appear.
    if kinds & {"mla", "mla_absorb", "mistral4_mla", "bailing_mla"}:
        env.update(_PROD_MLA_EXTRAS)
    if "gqa_sliding" in kinds:
        env.update(_PROD_GEMMA_EXTRAS)
    if "mamba3" in kinds:
        env.update(_PROD_NEMOTRON_EXTRAS)
    if "cca_attention" in kinds:
        env.update(_PROD_ZAYA_EXTRAS)
    return env


# ---------------------------------------------------------------------------
# verify_and_estimate
# ---------------------------------------------------------------------------


_DEFAULT_SIDE_CHANNELS = frozenset({"doc_ids", "token_ids"})


@dataclass(frozen=True)
class VerificationResult:
    """One-shot result of :func:`verify_and_estimate`."""

    resolved: ResolvedBrickGraph
    fusion_plan: tuple[FusionRegionPlan, ...]
    memory: MemoryReport
    elapsed_ms: float
    fits_on_device: bool | None = None

    @property
    def has_errors(self) -> bool:
        return self.resolved.has_errors

    def summary(self) -> dict[str, Any]:
        """Compact dict the GUI can render directly."""
        return {
            "errors":    len(self.resolved.errors),
            "warnings":  len(self.resolved.warnings),
            "regions":   len(self.fusion_plan),
            "memory":    self.memory.summary(),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "fits_on_device": self.fits_on_device,
        }


def verify_and_estimate(
    graph: BrickGraph,
    dim_env: Mapping[str, int] | None = None,
    *,
    preset_name: str | None = None,
    device_hbm_bytes: int | None = None,
    headroom: float = 0.9,
    training: bool = True,
    optimizer: str = "adamw",
    dtype_bytes: int = 2,
    kv_cache_dtype_bytes: int = 1,
    available_side_channels: frozenset[str] = _DEFAULT_SIDE_CHANNELS,
    strict: bool = False,
) -> VerificationResult:
    """Resolve shapes, plan fusion regions, estimate memory — one shot.

    Args:
      graph: the BrickGraph to verify.
      dim_env: concrete int values per named dim. If None, falls back
        to :func:`suggest_dim_env(preset_name)`.
      preset_name: lookup key for the default dim_env if dim_env=None.
      device_hbm_bytes: when supplied, the result includes
        ``fits_on_device`` (Memory.fits_on(device, headroom)).
      headroom: HBM headroom fraction for fits_on (default 0.9).
      training: forwarded to estimate_memory.
      optimizer, dtype_bytes, kv_cache_dtype_bytes: forwarded.
      available_side_channels: forwarded to resolve_shapes; defaults to
        ``{"doc_ids", "token_ids"}`` (the most common case for our bricks).
      strict: when True, resolver raises on first ERROR instead of
        collecting diagnostics.
    """
    if dim_env is None:
        dim_env = suggest_dim_env(preset_name)

    t0 = time.perf_counter()
    resolved = resolve_shapes(
        graph, dim_env,
        strict=strict,
        available_side_channels=available_side_channels,
    )
    plans = tuple(plan_fusion_regions(graph))
    memory = estimate_memory(
        resolved,
        fusion_plan=plans,
        training=training,
        optimizer=optimizer,
        dtype_bytes=dtype_bytes,
        kv_cache_dtype_bytes=kv_cache_dtype_bytes,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    fits = (
        memory.fits_on(device_hbm_bytes, headroom=headroom)
        if device_hbm_bytes is not None else None
    )
    return VerificationResult(
        resolved=resolved,
        fusion_plan=plans,
        memory=memory,
        elapsed_ms=elapsed_ms,
        fits_on_device=fits,
    )


# ---------------------------------------------------------------------------
# suggest_adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterProposal:
    """Bundle returned by :func:`suggest_adapters`.

    Fields:
      producer / consumer: the names of the edge endpoints.
      producer_shape / consumer_shape: resolved shapes on the edge.
      chain: ordered list of AdapterSuggestion steps; empty when shapes
        already match; None when no chain ≤ max_steps exists.
      reason: human-readable explanation for the GUI tooltip.
    """

    producer: str
    consumer: str
    producer_shape: tuple[int, ...]
    consumer_shape: tuple[int, ...]
    chain: list[AdapterSuggestion] | None
    reason: str


def suggest_adapters(
    resolved: ResolvedBrickGraph,
    producer: str,
    consumer: str,
    *,
    max_steps: int = 4,
) -> AdapterProposal:
    """Look up the resolved edge ``producer -> consumer`` and produce
    an :class:`AdapterProposal`. Raises KeyError if no such edge."""
    edge = resolved.edge(producer, consumer)
    chain = suggest_adapter_chain(
        edge.producer_shape, edge.consumer_shape, max_steps=max_steps,
    )
    if edge.matched:
        reason = "shapes already match — no adapter needed"
    elif chain is None:
        reason = (
            f"no adapter chain (≤ {max_steps} hops) bridges "
            f"{edge.producer_shape} -> {edge.consumer_shape}"
        )
    elif not chain:
        reason = "shapes match by value but resolver disagreed — no adapter needed"
    else:
        reason = f"chain of {len(chain)} adapter step(s): " + " -> ".join(
            s.kind for s in chain
        )
    return AdapterProposal(
        producer=producer,
        consumer=consumer,
        producer_shape=edge.producer_shape,
        consumer_shape=edge.consumer_shape,
        chain=chain,
        reason=reason,
    )


__all__ = [
    "AdapterProposal",
    "VerificationResult",
    "suggest_adapters",
    "suggest_dim_env",
    "verify_and_estimate",
]
