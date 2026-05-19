"""Path C region-level fusion planning.

This module is the high-level boundary for TileLang/TVM fusion work.  It
records producer/consumer surfaces before MSL lowering, so future codegen can
compile a whole Path C train region instead of trying to concatenate already
lowered Metal kernels.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence


__all__ = [
    "BenchmarkAcceptanceRow",
    "CompiledPathCRegion",
    "FusionCompilePlan",
    "FusionEdge",
    "FusionGroup",
    "FusionKernelSurface",
    "FusionNode",
    "PathCFusionMode",
    "PathCFusionRegion",
    "PathCFusionRegionBuilder",
    "WarmCacheAudit",
    "Z3SyncSpec",
    "audit_warm_cache_reuse",
    "build_mamba3_fp8_train_region",
    "compile_path_c_region",
    "fused_path_c_default_eligible",
    "path_b_baseline_is_clean",
    "selected_path_c_fusion_mode",
]


FUSION_MODE_ENV = "CPPMEGA_PATH_C_FUSION"
DEFAULT_MAX_PATH_B_CACHE_GIB = 55.0
DEFAULT_MAX_PATH_B_MEDIAN_STEP_S = 10.0
DEFAULT_MIN_C_OVER_B = 1.0
DEFAULT_MAX_PEAK_DELTA_GIB = 0.0
DEFAULT_MAX_WARM_FIRST_STEP_S = 12.0


class PathCFusionMode(Enum):
    """Top-level Path C fusion optimization mode."""

    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


_FUSION_MODE_ALIASES = {
    "": PathCFusionMode.OFF,
    "0": PathCFusionMode.OFF,
    "false": PathCFusionMode.OFF,
    "no": PathCFusionMode.OFF,
    "off": PathCFusionMode.OFF,
    "auto": PathCFusionMode.AUTO,
    "1": PathCFusionMode.AUTO,
    "true": PathCFusionMode.AUTO,
    "yes": PathCFusionMode.AUTO,
    "on": PathCFusionMode.AUTO,
    "force": PathCFusionMode.FORCE,
}


def selected_path_c_fusion_mode(
    env: Mapping[str, str] | None = None,
) -> PathCFusionMode:
    """Return the fail-closed fusion mode from ``CPPMEGA_PATH_C_FUSION``."""

    raw = (env or os.environ).get(FUSION_MODE_ENV, "")
    return _FUSION_MODE_ALIASES.get(raw.strip().lower(), PathCFusionMode.OFF)


@dataclass(frozen=True)
class Z3SyncSpec:
    """Z3-backed sync/async scheduling attachment for a fusion plan."""

    enabled: bool
    objective: str
    candidates: tuple[str, ...]
    proof_required: bool

    @classmethod
    def disabled(cls) -> "Z3SyncSpec":
        return cls(
            enabled=False,
            objective="none",
            candidates=(),
            proof_required=False,
        )

    @classmethod
    def minimize_sync_async(cls) -> "Z3SyncSpec":
        return cls(
            enabled=True,
            objective="minimize_sync_async",
            candidates=("sync", "async"),
            proof_required=True,
        )


@dataclass(frozen=True)
class FusionKernelSurface:
    """A pre-MSL Path C surface that can participate in region fusion."""

    name: str
    op_name: str
    path: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    backend: str
    backward: str

    @classmethod
    def path_c(
        cls,
        *,
        name: str,
        op_name: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        backward: str,
        backend: str = "tilelang_tvm_ffi",
    ) -> "FusionKernelSurface":
        return cls(
            name=_require_identifier(name, label="name"),
            op_name=_require_identifier(op_name, label="op_name"),
            path="path_c",
            inputs=_require_names(inputs, label="inputs"),
            outputs=_require_names(outputs, label="outputs"),
            backend=backend,
            backward=_require_identifier(backward, label="backward"),
        )


@dataclass(frozen=True)
class FusionNode:
    """A node in the Path C fusion IR."""

    name: str
    op_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    backend: str
    backward: str


@dataclass(frozen=True)
class FusionEdge:
    """Producer/consumer edge inferred from named buffers."""

    producer: str
    output: str
    consumer: str
    input: str


@dataclass(frozen=True)
class PathCFusionRegion:
    """FX-like region graph consumed by Path C lowering."""

    name: str
    nodes: tuple[FusionNode, ...]
    edges: tuple[FusionEdge, ...]
    z3_sync: Z3SyncSpec

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)


@dataclass(frozen=True)
class FusionGroup:
    """A group intended to lower through one TileLang/TVM region."""

    node_names: tuple[str, ...]
    schedule_template: str


@dataclass(frozen=True)
class FusionCompilePlan:
    """Compile plan produced from a Path C fusion region."""

    region_name: str
    lowering_boundary: str
    backend: str
    compiler: str
    fusion_groups: tuple[FusionGroup, ...]
    backward_graph: str
    z3_sync: Z3SyncSpec
    cache_key_parts: tuple[str, ...]
    requires_msl_post_fusion: bool = False
    large_tensor_staging_allowed: bool = False


@dataclass(frozen=True)
class CompiledPathCRegion:
    """Result wrapper for a future TileLang/TVM region compiler."""

    plan: FusionCompilePlan
    artifact: object | None = None


class PathCFusionRegionBuilder:
    """Build pre-lowering Path C fusion regions."""

    def __init__(
        self,
        name: str,
        *,
        z3_sync: Z3SyncSpec | None = None,
    ) -> None:
        self._name = _require_identifier(name, label="name")
        self._z3_sync = z3_sync or Z3SyncSpec.disabled()
        self._nodes: list[FusionNode] = []

    def add_kernel(self, surface: FusionKernelSurface) -> FusionNode:
        if not isinstance(surface, FusionKernelSurface):
            raise TypeError("surface must be FusionKernelSurface")
        if surface.path != "path_c":
            raise ValueError("Path C fusion only accepts path_c surfaces")
        if any(node.name == surface.name for node in self._nodes):
            raise ValueError(f"duplicate fusion node: {surface.name!r}")
        node = FusionNode(
            name=surface.name,
            op_name=surface.op_name,
            inputs=surface.inputs,
            outputs=surface.outputs,
            backend=surface.backend,
            backward=surface.backward,
        )
        self._nodes.append(node)
        return node

    def add_msl_kernel(self, name: str, source: str) -> None:
        """Reject post-lowering MSL surfaces explicitly."""

        _ = (name, source)
        raise ValueError(
            "Path C fusion must operate before MSL lowering; MSL kernels are "
            "opaque and cannot be safely fused here"
        )

    def build(self) -> PathCFusionRegion:
        if not self._nodes:
            raise ValueError("fusion region must contain at least one node")
        return PathCFusionRegion(
            name=self._name,
            nodes=tuple(self._nodes),
            edges=_infer_edges(self._nodes),
            z3_sync=self._z3_sync,
        )


def build_mamba3_fp8_train_region() -> PathCFusionRegion:
    """Return the high-level Path C train-block template requested for 1B."""

    builder = PathCFusionRegionBuilder(
        "mamba3_fp8_train_block",
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    builder.add_kernel(
        FusionKernelSurface.path_c(
            name="mamba3_scan",
            op_name="mamba3_mimo",
            inputs=("hidden", "mamba_state"),
            outputs=("scan_y", "scan_state"),
            backward="aot_autograd",
        )
    )
    builder.add_kernel(
        FusionKernelSurface.path_c(
            name="m2rnn_packed_post",
            op_name="m2rnn",
            inputs=("scan_y",),
            outputs=("post_y",),
            backward="aot_autograd",
        )
    )
    builder.add_kernel(
        FusionKernelSurface.path_c(
            name="sparse_mla_fp8_prepared",
            op_name="sparse_mla",
            inputs=("post_y",),
            outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
            backward="owner_output",
        )
    )
    return builder.build()


def compile_path_c_region(
    region: PathCFusionRegion,
    *,
    compiler: Callable[[FusionCompilePlan], object] | None = None,
) -> FusionCompilePlan | CompiledPathCRegion:
    """Create a TileLang/TVM region compile plan, optionally invoking compiler."""

    if not isinstance(region, PathCFusionRegion):
        raise TypeError("region must be PathCFusionRegion")
    backends = {node.backend for node in region.nodes}
    if backends != {"tilelang_tvm_ffi"}:
        raise ValueError(f"unsupported fusion backends: {sorted(backends)!r}")
    backward_graph = (
        "aot_autograd"
        if any(node.backward == "aot_autograd" for node in region.nodes)
        else "owner_output"
    )
    tilelang_plan = _tilelang_compile_plan_for(region)
    plan = FusionCompilePlan(
        region_name=region.name,
        lowering_boundary=tilelang_plan.lowering_boundary,
        backend=tilelang_plan.backend,
        compiler="tilelang.engine.fusion",
        fusion_groups=(
            FusionGroup(
                node_names=region.node_names,
                schedule_template=f"{region.name}:producer_consumer_region",
            ),
        ),
        backward_graph=backward_graph,
        z3_sync=region.z3_sync,
        cache_key_parts=tuple(tilelang_plan.cache_key_material),
    )
    if compiler is None:
        return plan
    return CompiledPathCRegion(plan=plan, artifact=compiler(plan))


def _tilelang_compile_plan_for(region: PathCFusionRegion):
    capture = io.StringIO()
    with redirect_stdout(capture), redirect_stderr(capture):
        from tilelang.engine.fusion import (
            FusionRegionBuilder as TileLangFusionRegionBuilder,
            plan_fusion_region,
        )

    builder = TileLangFusionRegionBuilder(region.name)
    for node in region.nodes:
        builder.add_node(
            node.name,
            op=node.op_name,
            inputs=node.inputs,
            outputs=node.outputs,
            attrs={"backward": node.backward},
        )
    for edge in region.edges:
        builder.connect(edge.producer, edge.consumer, buffer=edge.output)
    builder.set_schedule_template(_missing_tilelang_fused_schedule_template)
    if region.z3_sync.enabled:
        builder.enable_z3_sync_async_optimization()
    return plan_fusion_region(builder.build())


def _missing_tilelang_fused_schedule_template(_region):
    raise RuntimeError(
        "Path C fusion planning reached TileLang IR boundary, but the real "
        "fused train-block schedule template has not been supplied yet"
    )


@dataclass(frozen=True)
class BenchmarkAcceptanceRow:
    """Path B/C row used by the fused Path C default-eligibility gate."""

    dtype: str
    optimizer: str
    path_b_tok_sec: float
    path_c_warm_tok_sec: float
    path_b_median_step_s: float
    path_b_cache_gib: float
    path_c_peak_delta_gib: float

    @property
    def c_over_b(self) -> float:
        if self.path_b_tok_sec <= 0:
            return 0.0
        return self.path_c_warm_tok_sec / self.path_b_tok_sec


def path_b_baseline_is_clean(
    row: BenchmarkAcceptanceRow,
    *,
    max_cache_gib: float = DEFAULT_MAX_PATH_B_CACHE_GIB,
    max_median_step_s: float = DEFAULT_MAX_PATH_B_MEDIAN_STEP_S,
) -> bool:
    """Return whether a Path B row is trustworthy enough for Path C gating."""

    return (
        row.path_b_cache_gib <= max_cache_gib
        and row.path_b_median_step_s <= max_median_step_s
    )


def fused_path_c_default_eligible(
    row: BenchmarkAcceptanceRow,
    *,
    min_c_over_b: float = DEFAULT_MIN_C_OVER_B,
    max_peak_delta_gib: float = DEFAULT_MAX_PEAK_DELTA_GIB,
) -> bool:
    """Gate fused Path C defaulting on clean baselines and real wins."""

    if not path_b_baseline_is_clean(row):
        return False
    if row.c_over_b < min_c_over_b:
        return False
    return row.path_c_peak_delta_gib <= max_peak_delta_gib


@dataclass(frozen=True)
class WarmCacheAudit:
    case_id: str
    status: str
    reason: str
    cache_hit: bool
    compile_cache_key: str


def audit_warm_cache_reuse(
    *,
    case_id: str,
    cold_first_step_s: float,
    warm_first_step_s: float,
    cache_hit: bool,
    compile_cache_key: str,
    max_warm_first_step_s: float = DEFAULT_MAX_WARM_FIRST_STEP_S,
) -> WarmCacheAudit:
    """Classify whether a warm Path C cell really avoided cold startup cost."""

    if not cache_hit:
        return WarmCacheAudit(
            case_id=case_id,
            status="miss",
            reason="cache metadata reports a miss",
            cache_hit=cache_hit,
            compile_cache_key=compile_cache_key,
        )
    if warm_first_step_s > max_warm_first_step_s:
        return WarmCacheAudit(
            case_id=case_id,
            status="incomplete",
            reason=(
                "warm first step still exceeds threshold; graph specialization "
                "or TileLang/TVM cache reuse is incomplete"
            ),
            cache_hit=cache_hit,
            compile_cache_key=compile_cache_key,
        )
    if warm_first_step_s > cold_first_step_s:
        return WarmCacheAudit(
            case_id=case_id,
            status="incomplete",
            reason="warm first step is slower than cold first step",
            cache_hit=cache_hit,
            compile_cache_key=compile_cache_key,
        )
    return WarmCacheAudit(
        case_id=case_id,
        status="ok",
        reason="warm first step is within threshold",
        cache_hit=cache_hit,
        compile_cache_key=compile_cache_key,
    )


def _infer_edges(nodes: Sequence[FusionNode]) -> tuple[FusionEdge, ...]:
    producer_by_output: dict[str, str] = {}
    edges: list[FusionEdge] = []
    for node in nodes:
        for input_name in node.inputs:
            producer = producer_by_output.get(input_name)
            if producer is not None:
                edges.append(
                    FusionEdge(
                        producer=producer,
                        output=input_name,
                        consumer=node.name,
                        input=input_name,
                    )
                )
        for output_name in node.outputs:
            producer_by_output[output_name] = node.name
    return tuple(edges)


def _require_names(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    return tuple(_require_identifier(value, label=label) for value in values)


def _require_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
