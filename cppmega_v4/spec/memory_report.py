"""Memory roll-up over a ResolvedBrickGraph.

Given a graph that resolved cleanly under some ``dim_env``, this module
produces a :class:`MemoryReport` with byte-level estimates for:

  - weights (Σ params per brick × dtype_bytes)
  - gradients (= weights bytes; only if training=True)
  - optimizer state (AdamW: m + v at fp32 = 8 bytes/elem; Muon: 4)
  - activations (peak forward — fusion-aware: bricks inside one
    :class:`FusionRegionPlan` share registers, so we take the MAX of
    their per-brick activations, not the sum)
  - KV-cache (decode-mode only; can be quantised via
    ``kv_cache_dtype_bytes``)
  - edge handoff (per-edge tensor materialisation between non-fused
    bricks — almost free under DLPack zero-copy, charged conservatively
    here as one full tensor allocation)

The report carries per-brick and per-region rows so the GUI can render
both a sidebar total ("18.2 / 80 GB") and a detail view ("brick X is
4 GB peak; region 0 saves 1.2 GB by fusing").

Public surface:
  - :class:`BrickMemoryRow` / :class:`RegionMemoryRow` / :class:`MemoryReport`
  - :func:`estimate_memory`
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from cppmega_v4.fusion.auto_planner import FusionRegionPlan
from cppmega_v4.spec.resolver import ResolvedBrickGraph
from cppmega_v4.spec.shape_contract import (
    BrickShapeContract,
    contract_for,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_OPTIMIZER_BYTES_PER_PARAM: Final[dict[str, int]] = {
    # AdamW keeps two fp32 momenta (m, v) = 2 * 4 = 8 bytes/param
    "adamw": 8,
    # Muon stores Newton-Schulz state (fp32) ≈ 4 bytes/param
    "muon":  4,
    # No optimizer state (eval mode / SGD)
    "none":  0,
    "sgd":   0,
}


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrickMemoryRow:
    name: str
    kind: str
    params_bytes: int
    activations_bytes: int
    kv_cache_bytes: int


@dataclass(frozen=True)
class RegionMemoryRow:
    region_idx: int
    brick_names: tuple[str, ...]
    params_bytes: int
    activations_bytes: int       # max() across bricks for fused regions
    is_fused: bool


@dataclass(frozen=True)
class MemoryReport:
    """Byte-level memory estimate for a Resolved BrickGraph."""

    dim_env: Mapping[str, int]
    dtype_bytes: int
    optimizer: str
    training: bool
    kv_cache_dtype_bytes: int
    weights_bytes: int
    grads_bytes: int
    optimizer_bytes: int
    activations_bytes: int
    kv_cache_bytes: int
    edge_handoff_bytes: int
    total_bytes: int
    per_brick: dict[str, BrickMemoryRow]
    per_region: dict[int, RegionMemoryRow] = field(default_factory=dict)

    def fits_on(
        self, device_hbm_bytes: int, *, headroom: float = 0.9,
    ) -> bool:
        """True if the estimate lands below ``device_hbm_bytes * headroom``."""
        if not 0.0 < headroom <= 1.0:
            raise ValueError(
                f"headroom must be in (0, 1], got {headroom!r}"
            )
        if device_hbm_bytes < 1:
            raise ValueError("device_hbm_bytes must be ≥ 1")
        return self.total_bytes <= device_hbm_bytes * headroom

    def summary(self) -> dict[str, int]:
        """Compact dict for logging / GUI bars."""
        return {
            "weights":     self.weights_bytes,
            "grads":       self.grads_bytes,
            "optimizer":   self.optimizer_bytes,
            "activations": self.activations_bytes,
            "kv_cache":    self.kv_cache_bytes,
            "edge":        self.edge_handoff_bytes,
            "total":       self.total_bytes,
        }


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _resolve_elem_count(expr_contract: BrickShapeContract, env, field_name):
    """Resolve a 1-d ShapeExpr to a single non-negative int, or 0 on
    resolve failure (best-effort: opaque/missing-env bricks contribute
    0 to the corresponding bucket)."""
    expr = getattr(expr_contract, field_name)
    try:
        result = expr.resolve(env)
    except Exception:
        return 0
    return max(0, int(result[0]))


def estimate_memory(
    resolved: ResolvedBrickGraph,
    *,
    fusion_plan: Sequence[FusionRegionPlan] | None = None,
    dtype_bytes: int = 2,                 # bf16
    optimizer: str = "adamw",
    training: bool = True,
    kv_cache_dtype_bytes: int = 1,        # int8 quantised cache
    include_edge_handoff: bool = True,
) -> MemoryReport:
    """Roll up ``resolved`` into a :class:`MemoryReport`.

    Args:
      resolved: the output of :func:`cppmega_v4.spec.resolve_shapes`.
      fusion_plan: when supplied, activations within one FusionRegionPlan
        are accounted with ``max`` (shared registers) instead of ``sum``.
        When None, all bricks are charged independently (worst case).
      dtype_bytes: bytes per weight / activation element (bf16=2).
      optimizer: "adamw" | "muon" | "none" / "sgd".
      training: when False, gradients and optimizer state are 0.
      kv_cache_dtype_bytes: 0 to disable KV cache accounting (training mode).
      include_edge_handoff: when False, edge tensors aren't charged
        (assumes perfect DLPack zero-copy across every boundary).
    """
    if optimizer not in _OPTIMIZER_BYTES_PER_PARAM:
        raise ValueError(
            f"unknown optimizer {optimizer!r}; "
            f"choose from {sorted(_OPTIMIZER_BYTES_PER_PARAM)}"
        )
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes must be ≥ 1")
    if kv_cache_dtype_bytes < 0:
        raise ValueError("kv_cache_dtype_bytes must be ≥ 0")

    env = dict(resolved.dim_env)

    per_brick: dict[str, BrickMemoryRow] = {}
    for node in resolved.original.nodes:
        try:
            c = contract_for(node.kind)
        except KeyError:
            # No contract → can't account; report a zero row.
            per_brick[node.name] = BrickMemoryRow(
                name=node.name, kind=node.kind,
                params_bytes=0, activations_bytes=0, kv_cache_bytes=0,
            )
            continue
        params_elems = _resolve_elem_count(c, env, "params_elems")
        act_elems    = _resolve_elem_count(c, env, "activations_elems")
        kv_elems     = _resolve_elem_count(c, env, "kv_cache_elems")
        per_brick[node.name] = BrickMemoryRow(
            name=node.name,
            kind=node.kind,
            params_bytes=params_elems * dtype_bytes,
            activations_bytes=act_elems * dtype_bytes,
            kv_cache_bytes=(
                kv_elems * kv_cache_dtype_bytes if training is False else 0
            ),
        )

    weights_bytes = sum(r.params_bytes for r in per_brick.values())
    grads_bytes = weights_bytes if training else 0
    optimizer_bytes = (
        sum(_OPTIMIZER_BYTES_PER_PARAM[optimizer]
            * (r.params_bytes // dtype_bytes)
            for r in per_brick.values())
        if training else 0
    )

    # Activations — fusion-aware when a plan is supplied.
    per_region: dict[int, RegionMemoryRow] = {}
    if fusion_plan:
        accounted: set[str] = set()
        for idx, plan in enumerate(fusion_plan):
            rows = [per_brick[n] for n in plan.brick_names if n in per_brick]
            if not rows:
                continue
            if plan.is_fused:
                region_act = max(r.activations_bytes for r in rows)
            else:
                region_act = sum(r.activations_bytes for r in rows)
            per_region[idx] = RegionMemoryRow(
                region_idx=idx,
                brick_names=plan.brick_names,
                params_bytes=sum(r.params_bytes for r in rows),
                activations_bytes=region_act,
                is_fused=plan.is_fused,
            )
            accounted.update(r.name for r in rows)
        # Unaccounted (e.g. adapter nodes inserted after plan was built)
        # contribute their own activations as-is.
        leftover = sum(
            r.activations_bytes for n, r in per_brick.items() if n not in accounted
        )
        activations_bytes = (
            sum(rr.activations_bytes for rr in per_region.values()) + leftover
        )
    else:
        activations_bytes = sum(r.activations_bytes for r in per_brick.values())

    kv_cache_bytes = sum(r.kv_cache_bytes for r in per_brick.values())

    # Edge handoff: each non-fused edge moves one tensor; estimate by the
    # bigger of producer/consumer activation rows on the edge.
    edge_handoff_bytes = 0
    if include_edge_handoff:
        fused_pairs: set[tuple[str, str]] = set()
        if fusion_plan:
            for plan in fusion_plan:
                if not plan.is_fused:
                    continue
                names = plan.brick_names
                for i in range(len(names) - 1):
                    fused_pairs.add((names[i], names[i + 1]))
        for edge in resolved.edges:
            if (edge.producer, edge.consumer) in fused_pairs:
                continue
            p = per_brick.get(edge.producer)
            c = per_brick.get(edge.consumer)
            if p is None or c is None:
                continue
            edge_handoff_bytes += max(p.activations_bytes, c.activations_bytes)

    total_bytes = (
        weights_bytes + grads_bytes + optimizer_bytes
        + activations_bytes + kv_cache_bytes + edge_handoff_bytes
    )

    return MemoryReport(
        dim_env=env,
        dtype_bytes=dtype_bytes,
        optimizer=optimizer,
        training=training,
        kv_cache_dtype_bytes=kv_cache_dtype_bytes,
        weights_bytes=weights_bytes,
        grads_bytes=grads_bytes,
        optimizer_bytes=optimizer_bytes,
        activations_bytes=activations_bytes,
        kv_cache_bytes=kv_cache_bytes,
        edge_handoff_bytes=edge_handoff_bytes,
        total_bytes=total_bytes,
        per_brick=per_brick,
        per_region=per_region,
    )


__all__ = [
    "BrickMemoryRow",
    "MemoryReport",
    "RegionMemoryRow",
    "estimate_memory",
]
