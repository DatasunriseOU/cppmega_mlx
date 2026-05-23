"""V8-R06: ``compile.trace`` RPC — surface the fusion plan as a wire-
form compile trace that the UI can render with per-op chips.

Each op in the trace carries:
  - name: the brick name
  - fused: True iff its region has 2+ bricks
  - group: the region label (e.g. ``"gemm_softmax_0"``)
  - materialised: True iff the region is the ``dlpack_handoff`` backend
    (a materialised boundary tensor is unavoidable across DLPack)
  - dlpack_boundary: True iff this op crosses a backend boundary
  - backend: tilelang | mlx | torch_inductor | metal_inline | dlpack_handoff

For now there is one shared planner — ``plan_fusion_regions`` — and
the same trace is returned for all three ``backend`` selectors. The
backend label in each op is taken from the FusionRegionPlan rather
than the request, so the UI can render the actual backend chosen by
the planner. The ``backend`` request parameter is reserved for the
upcoming inductor-specific path.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cppmega_v4.fusion.auto_planner import plan_fusion_regions
from cppmega_v4.fusion.brick_graph import from_block_specs
from cppmega_v4.jsonrpc.cache import LRUCache
from cppmega_v4.jsonrpc.methods import (
    _cache_lookup, _cache_store, _graph_to_specs,
)
from cppmega_v4.jsonrpc.schema import VerifyParams


__all__ = [
    "CompileTraceParams",
    "CompileTraceOp",
    "CompileTraceResult",
    "compile_trace",
]


CompileBackend = Literal["tilelang", "mlx", "torch_inductor"]


class CompileTraceParams(BaseModel):
    """Input — VerifyParams payload + the backend perspective."""

    model_config = ConfigDict(extra="forbid")

    spec: VerifyParams
    backend: CompileBackend = "mlx"


class CompileTraceOp(BaseModel):
    """One row in the compile trace — chip-renderable in the UI."""

    model_config = ConfigDict(extra="forbid")

    name: str
    fused: bool
    group: str
    materialised: bool
    dlpack_boundary: bool
    backend: str


class CompileTraceResult(BaseModel):
    """Full trace + aggregate counters that surface in extras.train."""

    model_config = ConfigDict(extra="forbid")

    ops: list[CompileTraceOp] = Field(default_factory=list)
    fused_groups: list[str] = Field(default_factory=list)
    dlpack_crossings: int = 0
    materialised_ops: list[str] = Field(default_factory=list)
    compile_artifact_path: str | None = None
    backend: str


def compile_trace(
    params: CompileTraceParams, *, cache: LRUCache | None = None,
) -> CompileTraceResult:
    """Plan fusion regions for ``params.spec`` and render the trace."""
    key, hit = _cache_lookup(cache, "compile.trace", params)
    if hit is not None:
        return hit

    specs = _graph_to_specs(params.spec.graph)
    hidden = params.spec.dim_env.get("H", 64)
    graph = from_block_specs(specs, hidden_size=hidden, instantiate=False)
    plans = list(plan_fusion_regions(graph))

    ops: list[CompileTraceOp] = []
    fused_groups: list[str] = []
    materialised: list[str] = []
    dlpack_crossings = 0
    last_backend: str | None = None

    for idx, plan in enumerate(plans):
        group_label = f"region_{idx:02d}_{plan.backend}"
        is_materialised = plan.backend == "dlpack_handoff"
        if is_materialised and last_backend is not None and \
                last_backend != plan.backend:
            dlpack_crossings += 1
        if plan.is_fused:
            fused_groups.append(group_label)
        for i, name in enumerate(plan.brick_names):
            ops.append(CompileTraceOp(
                name=name,
                fused=plan.is_fused,
                group=group_label,
                materialised=is_materialised,
                dlpack_boundary=(is_materialised and i == 0
                                  and idx > 0),
                backend=plan.backend,
            ))
            if is_materialised and name not in materialised:
                materialised.append(name)
        last_backend = plan.backend

    out = CompileTraceResult(
        ops=ops,
        fused_groups=fused_groups,
        dlpack_crossings=dlpack_crossings,
        materialised_ops=materialised,
        compile_artifact_path=None,   # set by tilelang codegen path
        backend=params.backend,
    )
    _cache_store(cache, key, out)
    return out
