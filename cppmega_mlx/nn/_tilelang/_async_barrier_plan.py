"""Barrier/sync planning helpers for Metal Path C reductions.

The planner is intentionally small and conservative. It does not rewrite MSL;
it classifies a reduction schedule so Path C gates can prefer simdgroup-local
async reductions and identify schedules that still need threadgroup barriers.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


_Z3_DISABLE_ENV = (
    "TILELANG_DISABLE_Z3",
    "TILELANG_DISABLE_Z3_BARRIER_ELISION",
    "CPPMEGA_DISABLE_Z3",
)


@dataclass(frozen=True)
class MetalReductionSyncPlan:
    """Sync plan for one Metal Path C reduction schedule."""

    strategy: str
    barrier_count: int
    wait_count: int
    async_stages: int
    z3_used: bool
    z3_proved: bool
    reduction_isolated: bool
    reason: str
    outputs_per_block: int
    reduce_threads: int
    vec: int
    k_extent: int
    simdgroup_width: int = 32

    @property
    def is_async(self) -> bool:
        return self.strategy == "simdgroup_async"

    def as_feature_dict(self) -> dict[str, int | bool | str]:
        return {
            "sync_plan_strategy": self.strategy,
            "sync_plan_barrier_count": self.barrier_count,
            "sync_plan_wait_count": self.wait_count,
            "sync_plan_async_stages": self.async_stages,
            "sync_plan_z3_used": self.z3_used,
            "sync_plan_z3_proved": self.z3_proved,
            "sync_plan_reduction_isolated": self.reduction_isolated,
            "sync_plan_reason": self.reason,
            "sync_plan_outputs_per_block": self.outputs_per_block,
            "sync_plan_reduce_threads": self.reduce_threads,
            "sync_plan_vec": self.vec,
            "sync_plan_k_extent": self.k_extent,
            "sync_plan_simdgroup_width": self.simdgroup_width,
        }


def _z3_disabled() -> bool:
    return any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}
        for name in _Z3_DISABLE_ENV
    )


def _ceil_log2(value: int) -> int:
    return 0 if value <= 1 else math.ceil(math.log2(value))


def _z3_proves_simdgroup_output_isolated(
    *, outputs_per_block: int, reduce_threads: int, simdgroup_width: int
) -> tuple[bool, bool, str]:
    """Prove that one simdgroup never spans two output columns.

    For the Path C QK reducer TileLang's thread axes are ``(kr, ni)``. A pure
    simdgroup reduction is safe only when all lanes in a hardware simdgroup
    share the same ``ni`` and differ only in ``kr``.
    """

    if _z3_disabled():
        return False, False, "z3 disabled by environment"
    try:
        import z3  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional dep
        return False, False, f"z3 unavailable: {type(exc).__name__}: {exc}"

    x0 = z3.Int("x0")
    y0 = z3.Int("y0")
    x1 = z3.Int("x1")
    y1 = z3.Int("y1")
    linear0 = y0 * reduce_threads + x0
    linear1 = y1 * reduce_threads + x1
    solver = z3.Solver()
    solver.set("timeout", 50)
    solver.add(0 <= x0, x0 < reduce_threads)
    solver.add(0 <= x1, x1 < reduce_threads)
    solver.add(0 <= y0, y0 < outputs_per_block)
    solver.add(0 <= y1, y1 < outputs_per_block)
    solver.add(linear0 / simdgroup_width == linear1 / simdgroup_width)
    solver.add(y0 != y1)
    try:
        result = solver.check()
    except Exception as exc:  # pragma: no cover - defensive z3 boundary
        return True, False, f"z3 raised {type(exc).__name__}: {exc}"
    if result == z3.unsat:
        return True, True, "z3 proved each simdgroup is output-isolated"
    if result == z3.unknown:
        return True, False, "z3 returned unknown for simdgroup isolation"
    return True, False, "z3 found a cross-output simdgroup witness"


def plan_metal_path_c_reduction_sync(
    *,
    outputs_per_block: int,
    reduce_threads: int,
    vec: int,
    k_extent: int,
    simdgroup_width: int = 32,
) -> MetalReductionSyncPlan:
    """Classify the cheapest safe sync strategy for a Path C dot reduction."""

    values = {
        "outputs_per_block": outputs_per_block,
        "reduce_threads": reduce_threads,
        "vec": vec,
        "k_extent": k_extent,
        "simdgroup_width": simdgroup_width,
    }
    bad = {name: value for name, value in values.items() if value <= 0}
    if bad:
        return MetalReductionSyncPlan(
            strategy="invalid",
            barrier_count=0,
            wait_count=0,
            async_stages=0,
            z3_used=False,
            z3_proved=False,
            reduction_isolated=False,
            reason=f"non-positive schedule values: {bad}",
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
            k_extent=k_extent,
            simdgroup_width=simdgroup_width,
        )

    if reduce_threads == simdgroup_width:
        z3_used, z3_proved, z3_reason = _z3_proves_simdgroup_output_isolated(
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            simdgroup_width=simdgroup_width,
        )
        return MetalReductionSyncPlan(
            strategy="simdgroup_async",
            barrier_count=0,
            wait_count=0,
            async_stages=max(1, math.ceil(k_extent / (reduce_threads * vec))),
            z3_used=z3_used,
            z3_proved=z3_proved,
            reduction_isolated=True,
            reason=(
                "full simdgroup reduction: no threadgroup memory or barrier is "
                f"needed; {z3_reason}"
            ),
            outputs_per_block=outputs_per_block,
            reduce_threads=reduce_threads,
            vec=vec,
            k_extent=k_extent,
            simdgroup_width=simdgroup_width,
        )

    if reduce_threads < simdgroup_width:
        reason = (
            "sub-simdgroup reduction may share one hardware simdgroup across "
            "multiple output columns; keep threadgroup sync until masked "
            "simdgroup collectives are emitted"
        )
    else:
        reason = (
            "reduction spans multiple hardware simdgroups; needs shared "
            "threadgroup rendezvous"
        )

    barriers = max(1, _ceil_log2(reduce_threads))
    return MetalReductionSyncPlan(
        strategy="threadgroup_sync",
        barrier_count=barriers,
        wait_count=barriers,
        async_stages=0,
        z3_used=False,
        z3_proved=False,
        reduction_isolated=False,
        reason=reason,
        outputs_per_block=outputs_per_block,
        reduce_threads=reduce_threads,
        vec=vec,
        k_extent=k_extent,
        simdgroup_width=simdgroup_width,
    )
