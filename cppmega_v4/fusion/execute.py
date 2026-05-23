"""Stage F — executable fusion graph over auto-compiled brick regions.

This module provides the runtime executor that downstream consumers were
missing in Stages A-E:

  * Stages A-E plan and *describe* fusion regions, register V4 brick
    descriptors with the cppmega_mlx schedule machinery, and emit
    AutoCompiledRegion records that pair a planner region with the
    TileLang schedule template the codegen consumes.
  * Stage F walks those regions in their planner order and executes the
    contained bricks against MLX inputs, in the right order, with the
    backend the planner chose:
      - ``backend == "path_c"`` and the region has a compiled artifact
        attached: drive the artifact's forward (artifact must accept the
        region's leading hidden state via DLPack and write its tail to
        the same buffer convention).
      - otherwise (single-brick passthrough, dlpack-handoff chain,
        metal_inline, or any region whose Path C lowering has not yet
        landed): execute every brick eagerly via its existing
        ``nn.Module`` instance, threading the running hidden state from
        one brick to the next.

The executor is intentionally honest about which regions are running
fused vs eager — see :class:`RegionExecution.backend` and the
:attr:`ExecutableGraph.execution_log` field — so callers (1B path_c
matrix, GUI, benchmarks) can decide whether the planner's promises are
being honoured for a given input. No tensor data is silently copied or
re-typed beyond the explicit DLPack handoff helpers in
:mod:`cppmega_v4.fusion.dlpack_bridge`.

Public surface:
  - :class:`ExecutableGraph` — bound to one mlx model + one plan.
  - :class:`RegionExecution` — per-region outcome record.
  - :func:`build_executable_graph(model, plan_regions=None)` —
    convenience entry point that auto-plans the regions from
    ``auto_fuse_model`` when ``plan_regions`` is omitted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from cppmega_v4.fusion.auto_compile import (
    AutoCompiledRegion,
    RegionPattern,
    auto_compile_plan,
)
from cppmega_v4.fusion.auto_planner import (
    DEFAULT_MAX_REGION_SIZE,
    DEFAULT_MAX_SHARED_MEM_BYTES,
    FusionRegionPlan,
    plan_fusion_regions,
)
from cppmega_v4.fusion.brick_graph import (
    BrickGraph,
    BrickNode,
    from_mlx_model,
)


# Reasons a region must run via the eager-brick path instead of a fused
# artifact. Keep these stable — the matrix/GUI surface them verbatim.
_REGION_RUN_EAGER_REASONS: dict[str, str] = {
    "single_brick": "single-brick region; no fused PrimFunc emitted",
    "dlpack_handoff": "planner chose dlpack_handoff; each brick keeps its native kernel",
    "no_template": "AutoCompiledRegion has no schedule template",
    "no_artifact": "no compiled TileLang artifact attached to this region",
    "no_module": "brick instance missing module reference (instantiate=True required)",
}


@dataclass(frozen=True)
class RegionExecution:
    """One region's outcome from a single ExecutableGraph.forward call.

    Fields:
      region_index: position in the planner output, 0-based.
      pattern: planner pattern label (single-brick, fused, etc.).
      backend: "path_c_artifact" | "eager_bricks" — what actually ran.
      brick_names: ordered tuple of brick names inside this region.
      eager_reason: when ``backend == "eager_bricks"``, the canonical
        reason string from :data:`_REGION_RUN_EAGER_REASONS` (or
        free-form fallback). ``""`` when the artifact ran.
      duration_ns: wall-clock spent in this region's execution call.
    """

    region_index: int
    pattern: RegionPattern
    backend: str
    brick_names: tuple[str, ...]
    eager_reason: str
    duration_ns: int

    @property
    def ran_fused(self) -> bool:
        return self.backend == "path_c_artifact"


@dataclass
class ExecutableGraph:
    """Forward-only executor for a BrickGraph + planner regions + bricks.

    The executor never owns the bricks — it dispatches to their existing
    ``nn.Module`` instances. Per-region compiled artifacts can be
    attached after construction via :meth:`attach_artifact` and will be
    used for that region's forward pass when present. Otherwise the
    region falls back to walking the bricks eagerly with a sequential
    hidden-state thread.

    The executor is intentionally forward-only. Training (value_and_grad,
    backward) goes through the m04 fused train-block runtime, which is a
    separate seam — see ``scripts/m04_train_step.py``.
    """

    graph: BrickGraph
    plans: tuple[FusionRegionPlan, ...]
    regions: tuple[AutoCompiledRegion, ...]
    # Mutable: per-region attached artifact (callable; takes leading
    # hidden mx.array, returns trailing hidden mx.array). Populated by
    # callers via attach_artifact after compile.
    artifacts: dict[int, Callable[[mx.array], mx.array]] = field(
        default_factory=dict
    )
    # Mutable: rolling per-region execution log (last forward call).
    execution_log: list[RegionExecution] = field(default_factory=list)

    # --- introspection -------------------------------------------------

    @property
    def fused_region_count(self) -> int:
        return sum(1 for region in self.regions if region.has_compiled_template)

    @property
    def eager_region_count(self) -> int:
        return len(self.regions) - self.fused_region_count

    def region_summary(self) -> tuple[dict[str, Any], ...]:
        """Per-region summary suitable for receipts / GUI tables."""

        return tuple(
            {
                "index": index,
                "pattern": region.pattern.value,
                "backend_planned": region.plan.backend,
                "brick_names": list(region.plan.brick_names),
                "has_compiled_template": region.has_compiled_template,
                "has_artifact": index in self.artifacts,
                "estimated_savings_us": float(region.plan.estimated_savings_us),
            }
            for index, region in enumerate(self.regions)
        )

    # --- artifact wiring -----------------------------------------------

    def attach_artifact(
        self,
        region_index: int,
        artifact: Callable[[mx.array], mx.array],
    ) -> None:
        """Bind a compiled TileLang artifact to a specific region index.

        The artifact must accept the leading hidden state as its single
        positional ``mx.array`` argument and return the trailing hidden
        state as an ``mx.array``. Any device or layout requirements are
        the artifact's responsibility — callers are expected to wire
        DLPack-friendly artifacts produced by Path C codegen.
        """
        if not callable(artifact):
            raise TypeError(
                f"artifact for region {region_index} must be callable"
            )
        if region_index < 0 or region_index >= len(self.regions):
            raise IndexError(
                f"region_index {region_index} out of range "
                f"[0, {len(self.regions)})"
            )
        region = self.regions[region_index]
        if not region.has_compiled_template:
            raise ValueError(
                f"region {region_index} ({region.pattern.value}) has no "
                "compiled schedule template; cannot attach a fused artifact"
            )
        self.artifacts[region_index] = artifact

    # --- forward -------------------------------------------------------

    def forward(self, hidden: mx.array) -> mx.array:
        """Run every region in planner order against the input ``hidden``.

        Each region's leading brick consumes the current hidden state;
        the trailing brick's output becomes the next region's input.
        Per-region timing and backend choice are recorded in
        :attr:`execution_log` (replaced on every call).
        """
        self.execution_log.clear()
        running = hidden
        for index, region in enumerate(self.regions):
            t0 = time.perf_counter_ns()
            artifact = self.artifacts.get(index)
            if artifact is not None:
                running = artifact(running)
                backend = "path_c_artifact"
                eager_reason = ""
            else:
                running = self._run_region_eager(region, running)
                backend = "eager_bricks"
                eager_reason = self._eager_reason_for(region)
            duration = time.perf_counter_ns() - t0
            self.execution_log.append(
                RegionExecution(
                    region_index=index,
                    pattern=region.pattern,
                    backend=backend,
                    brick_names=region.plan.brick_names,
                    eager_reason=eager_reason,
                    duration_ns=int(duration),
                )
            )
        return running

    __call__ = forward

    # --- helpers -------------------------------------------------------

    def _run_region_eager(
        self,
        region: AutoCompiledRegion,
        hidden: mx.array,
    ) -> mx.array:
        """Walk a region's bricks eagerly, threading the hidden state."""
        running = hidden
        for brick_name in region.plan.brick_names:
            node = self._node_by_name(brick_name)
            module = node.module
            if module is None:
                raise ValueError(
                    f"brick {brick_name!r} has no module attached; the "
                    "executor needs instantiated modules to run eagerly. "
                    "Re-build the graph with instantiate=True or attach "
                    "the module via BrickNode.module."
                )
            out = module(running)
            running = self._sanitise_brick_output(out, running, brick_name)
        return running

    def _node_by_name(self, name: str) -> BrickNode:
        for node in self.graph.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    @staticmethod
    def _sanitise_brick_output(
        out: Any, prev_hidden: mx.array, brick_name: str
    ) -> mx.array:
        """Brick modules sometimes return tuples ``(hidden, state)``.

        Reduce them to the leading hidden ``mx.array`` so the next brick
        sees a clean state. Returns ``prev_hidden`` if the brick output
        is not a usable ``mx.array``.
        """
        if isinstance(out, mx.array):
            return out
        if isinstance(out, tuple) and out and isinstance(out[0], mx.array):
            return out[0]
        if isinstance(out, Mapping):
            for value in out.values():
                if isinstance(value, mx.array):
                    return value
        return prev_hidden

    def _eager_reason_for(self, region: AutoCompiledRegion) -> str:
        if region.pattern is RegionPattern.SINGLE_BRICK_PASSTHROUGH:
            return _REGION_RUN_EAGER_REASONS["single_brick"]
        if region.pattern is RegionPattern.DLPACK_HANDOFF_CHAIN:
            return _REGION_RUN_EAGER_REASONS["dlpack_handoff"]
        if not region.has_compiled_template:
            return _REGION_RUN_EAGER_REASONS["no_template"]
        return _REGION_RUN_EAGER_REASONS["no_artifact"]


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def build_executable_graph(
    model: nn.Module,
    *,
    attr_order: Sequence[str] | None = None,
    max_region_size: int = DEFAULT_MAX_REGION_SIZE,
    max_shared_mem_bytes: int = DEFAULT_MAX_SHARED_MEM_BYTES,
) -> ExecutableGraph:
    """Walk ``model``, plan regions, auto-compile descriptors, and bind.

    Returns an :class:`ExecutableGraph` with no artifacts attached yet —
    every region defaults to eager execution. Callers wire artifacts
    explicitly via :meth:`ExecutableGraph.attach_artifact` after their
    own compile step (so we never mix the planner with TileLang codegen
    here).
    """
    graph = from_mlx_model(model, attr_order=attr_order)
    plans = tuple(
        plan_fusion_regions(
            graph,
            max_region_size=max_region_size,
            max_shared_mem_bytes=max_shared_mem_bytes,
        )
    )
    kinds_by_name = {n.name: n.kind for n in graph.nodes}
    regions = tuple(auto_compile_plan(list(plans), kinds_by_name))
    return ExecutableGraph(graph=graph, plans=plans, regions=regions)


__all__ = [
    "ExecutableGraph",
    "RegionExecution",
    "build_executable_graph",
]
