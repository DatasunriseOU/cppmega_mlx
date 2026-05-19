"""Stage C — greedy fusion-region planner over a BrickGraph.

Walks a ``BrickGraph`` left-to-right and groups adjacent fusion-eligible
bricks into ``FusionRegionPlan`` records. The planner is intentionally
conservative:

  - Eligibility check delegates entirely to
    :func:`cppmega_v4.fusion.compatibility.can_fuse_pair` (table-driven).
  - A region is **extended** when the trailing brick can fuse with the
    next candidate AND no hard limit is breached.
  - A region is **closed** (and a new one started at the next brick)
    when eligibility says ``can_fuse=False``, when adding the next brick
    would exceed ``max_region_size`` bricks, or when the estimated
    shared-memory footprint would exceed ``max_shared_mem_bytes``.

A simple cost model estimates microsecond savings of grouping (saved
kernel launches + saved DLPack handoffs) minus a register-pressure
penalty for category-mixed regions. The cost model never **forces** a
split — it only reports an ``estimated_savings_us`` field for telemetry
and downstream cost-aware tuning.

Public surface:
  - :class:`FusionRegionPlan`
  - :func:`plan_fusion_regions`
  - :func:`auto_fuse_model` — annotates an :class:`mlx.nn.Module` with
    its planned regions (does not yet replace child modules; the actual
    Path C compilation lands in Stage E).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.nn as nn

from cppmega_v4.fusion.brick_graph import BrickGraph, BrickNode, from_mlx_model
from cppmega_v4.fusion.compatibility import (
    FusionEligibility,
    _CATEGORY_BY_KIND,
    can_fuse_pair,
)


# ---------------------------------------------------------------------------
# Hard limits and cost-model constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_REGION_SIZE: int = 8
"""Maximum bricks per region. Apple Metal register pressure starts hurting
past ~8 fused passes; beyond that, codegen often spills to threadgroup
memory or fragments live ranges across kernels."""

DEFAULT_MAX_SHARED_MEM_BYTES: int = 32 * 1024
"""Per-region Apple Metal threadgroup memory budget (32 KiB)."""

# Per-brick estimated threadgroup-memory footprint, in bytes. These are
# rough rule-of-thumb numbers used by the planner's hard-limit check; the
# real Path C lowering computes exact figures during codegen.
_SHARED_MEM_ESTIMATE_BY_CATEGORY: dict[str, int] = {
    "linear_attn": 4096,       # recurrent state per head
    "ssm": 8192,               # chunkwise scan buffer
    "sdpa_attention": 2048,    # output gate fragment
    "cross_attn": 2048,
    "nonlinear_rnn": 4096,
    "moe": 4096,               # route + combine buffers
    "sparse_attn": 4096,
    "norm_or_proj": 1024,
    "unknown": 2048,
}

# Cost-model constants (microseconds, derived from rough Apple M-series
# launch-latency measurements). Used only for telemetry; the planner's
# decision is driven by eligibility + hard limits.
_KERNEL_LAUNCH_OVERHEAD_US: float = 5.0
_DLPACK_HANDOFF_OVERHEAD_US: float = 1.0
_REGISTER_PRESSURE_PENALTY_PER_MIX_US: float = 1.5


# ---------------------------------------------------------------------------
# FusionRegionPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionRegionPlan:
    """A planned fusion region — a contiguous slice of brick names that
    will be compiled together (or, for size-1 regions, kept as a single
    standalone brick).

    Fields:
      brick_names: ordered tuple of node names belonging to this region.
      categories: per-brick category (linear_attn / sdpa_attention / ...).
      backend: ``"path_c"`` | ``"metal_inline"`` | ``"dlpack_handoff"``
        — the codegen backend the planner selected. Size-1 regions
        always carry the brick's own native backend (single-brick passes
        through whatever it already implements).
      estimated_savings_us: positive when the planner expects time saved
        by fusing this region vs running its bricks separately. Always 0
        for size-1 regions.
      reason: short human-readable explanation (used by tests/telemetry
        to surface *why* a region was closed at this boundary).
    """

    brick_names: tuple[str, ...]
    categories: tuple[str, ...]
    backend: str
    estimated_savings_us: float
    reason: str

    def __post_init__(self) -> None:
        if not self.brick_names:
            raise ValueError("FusionRegionPlan must contain ≥1 brick")
        if len(self.brick_names) != len(self.categories):
            raise ValueError(
                "FusionRegionPlan: brick_names and categories must align"
            )

    @property
    def size(self) -> int:
        return len(self.brick_names)

    @property
    def is_fused(self) -> bool:
        """True when this region groups ≥2 bricks under a fusing backend."""
        return self.size > 1 and self.backend in {"path_c", "metal_inline"}


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def _shared_mem_estimate(category: str) -> int:
    return _SHARED_MEM_ESTIMATE_BY_CATEGORY.get(category, 2048)


def _estimate_savings_us(categories: Sequence[str], backend: str) -> float:
    """Estimated microseconds saved by fusing this region vs running each
    brick as a standalone kernel. Returns 0.0 for size-1 regions and for
    dlpack-only regions (no real fusion happens there)."""
    n = len(categories)
    if n <= 1:
        return 0.0
    if backend not in {"path_c", "metal_inline"}:
        return 0.0
    saved_launches = (n - 1) * _KERNEL_LAUNCH_OVERHEAD_US
    saved_handoffs = (n - 1) * _DLPACK_HANDOFF_OVERHEAD_US
    distinct_categories = len(set(categories))
    mix_penalty = (
        (distinct_categories - 1) * _REGISTER_PRESSURE_PENALTY_PER_MIX_US
    )
    return max(0.0, saved_launches + saved_handoffs - mix_penalty)


# ---------------------------------------------------------------------------
# Planner core
# ---------------------------------------------------------------------------


@dataclass
class _OpenRegion:
    """Mutable scratch state for the greedy walker."""

    nodes: list[BrickNode] = field(default_factory=list)
    backend: str = "dlpack_handoff"
    shared_mem_bytes: int = 0
    close_reason: str = "end of graph"


def _category_of(node: BrickNode) -> str:
    return _CATEGORY_BY_KIND.get(node.kind, "unknown")


def _finalise(region: _OpenRegion) -> FusionRegionPlan:
    cats = tuple(_category_of(n) for n in region.nodes)
    return FusionRegionPlan(
        brick_names=tuple(n.name for n in region.nodes),
        categories=cats,
        backend=region.backend,
        estimated_savings_us=_estimate_savings_us(cats, region.backend),
        reason=region.close_reason,
    )


def plan_fusion_regions(
    graph: BrickGraph,
    *,
    max_region_size: int = DEFAULT_MAX_REGION_SIZE,
    max_shared_mem_bytes: int = DEFAULT_MAX_SHARED_MEM_BYTES,
) -> list[FusionRegionPlan]:
    """Greedy bottom-up region planner.

    Walks ``graph.nodes`` in declaration order. Starts a new region at
    the first node, then for each subsequent node decides whether to
    *extend* the open region or *close* it and open a new one.

    The decision is:

      1. If ``can_fuse_pair(tail, next).can_fuse`` is False → close.
      2. If extending would put the region above ``max_region_size`` → close.
      3. If extending would push estimated threadgroup memory above
         ``max_shared_mem_bytes`` → close.
      4. Otherwise extend.

    Returns a list of :class:`FusionRegionPlan` covering every node in
    ``graph`` exactly once, in original order.
    """
    if max_region_size < 1:
        raise ValueError("max_region_size must be ≥ 1")
    if max_shared_mem_bytes < 1:
        raise ValueError("max_shared_mem_bytes must be ≥ 1")

    plans: list[FusionRegionPlan] = []
    if not graph.nodes:
        return plans

    open_region = _OpenRegion()
    first = graph.nodes[0]
    open_region.nodes.append(first)
    open_region.shared_mem_bytes = _shared_mem_estimate(_category_of(first))
    # A size-1 region carries the "passthrough" backend; it gets upgraded
    # to path_c / metal_inline if (and only if) a fusable neighbour
    # extends it below.
    open_region.backend = "dlpack_handoff"

    for prev, current in zip(graph.nodes[:-1], graph.nodes[1:]):
        tail = open_region.nodes[-1]
        elig: FusionEligibility = can_fuse_pair(tail, current)
        next_shared = _shared_mem_estimate(_category_of(current))

        close = False
        reason = ""
        if not elig.can_fuse:
            close = True
            reason = elig.reason
        elif len(open_region.nodes) + 1 > max_region_size:
            close = True
            reason = (
                f"region would exceed max_region_size={max_region_size}"
            )
        elif open_region.shared_mem_bytes + next_shared > max_shared_mem_bytes:
            close = True
            reason = (
                "region would exceed shared-mem budget "
                f"({open_region.shared_mem_bytes + next_shared} > "
                f"{max_shared_mem_bytes} bytes)"
            )

        if close:
            open_region.close_reason = reason
            plans.append(_finalise(open_region))
            open_region = _OpenRegion()
            open_region.nodes.append(current)
            open_region.shared_mem_bytes = next_shared
            open_region.backend = "dlpack_handoff"
        else:
            open_region.nodes.append(current)
            open_region.shared_mem_bytes += next_shared
            # Promote backend to the chosen fusing backend (path_c or
            # metal_inline). The first promotion wins; subsequent fuses
            # within the same region must agree on the same backend.
            if open_region.backend == "dlpack_handoff":
                open_region.backend = elig.backend
            elif open_region.backend != elig.backend:
                # Mixed-backend region — must close, start fresh.
                open_region.nodes.pop()
                open_region.shared_mem_bytes -= next_shared
                open_region.close_reason = (
                    f"backend mismatch: region={open_region.backend!r} "
                    f"vs pair={elig.backend!r}"
                )
                plans.append(_finalise(open_region))
                open_region = _OpenRegion()
                open_region.nodes.append(current)
                open_region.shared_mem_bytes = next_shared
                open_region.backend = "dlpack_handoff"

    plans.append(_finalise(open_region))
    return plans


# ---------------------------------------------------------------------------
# Public model-level entry point
# ---------------------------------------------------------------------------


def auto_fuse_model(
    model: nn.Module,
    *,
    max_region_size: int = DEFAULT_MAX_REGION_SIZE,
    max_shared_mem_bytes: int = DEFAULT_MAX_SHARED_MEM_BYTES,
) -> nn.Module:
    """Annotate ``model`` with its planned fusion regions.

    Walks the model's direct children, builds a :class:`BrickGraph` from
    them (see :func:`cppmega_v4.fusion.brick_graph.from_mlx_model`),
    runs :func:`plan_fusion_regions`, and attaches the resulting list
    to ``model._v4_fusion_plan``.

    The model object is returned unchanged structurally — Stage E will
    add the actual region compilation + module replacement step. Until
    then, downstream consumers can read ``model._v4_fusion_plan`` to
    drive their own lowering.
    """
    graph = from_mlx_model(model)
    plan = plan_fusion_regions(
        graph,
        max_region_size=max_region_size,
        max_shared_mem_bytes=max_shared_mem_bytes,
    )
    setattr(model, "_v4_fusion_plan", tuple(plan))
    setattr(model, "_v4_fusion_brick_graph", graph)
    return model


__all__ = [
    "DEFAULT_MAX_REGION_SIZE",
    "DEFAULT_MAX_SHARED_MEM_BYTES",
    "FusionRegionPlan",
    "auto_fuse_model",
    "plan_fusion_regions",
]
