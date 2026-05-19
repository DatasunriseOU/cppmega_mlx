"""Path C region-level fusion planning.

This module is the high-level boundary for TileLang/TVM fusion work.  It
records producer/consumer surfaces before MSL lowering, so future codegen can
compile a whole Path C train region instead of trying to concatenate already
lowered Metal kernels.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import io
import json
import os
from typing import Any, Callable, Mapping, Sequence


__all__ = [
    "BenchmarkAcceptanceRow",
    "CompiledPathCRegion",
    "FusionCompilePlan",
    "FusionEdge",
    "FusionCacheKeyAudit",
    "FusionGroup",
    "FusionKernelSurface",
    "FusionScheduleAbiBlocker",
    "FusionBufferSignature",
    "FusionSemanticBlocker",
    "FusionScheduleContract",
    "FusionScheduleContractStatus",
    "FusionScheduleStatus",
    "MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS",
    "FusionNode",
    "PathCFusionMode",
    "PathCFusionRegion",
    "PathCFusionRegionBuilder",
    "PathCModelBrick",
    "PathCModelShapeEnv",
    "SparseMLAFp8PrepareSpec",
    "WarmCacheAudit",
    "Z3SyncSpec",
    "audit_fusion_cache_key",
    "audit_warm_cache_reuse",
    "build_path_c_aot_autograd_region",
    "build_path_c_fusion_region",
    "build_path_c_model_region_from_bricks",
    "build_path_c_model_regions_from_bricks",
    "build_path_c_model_region_from_route_symbols",
    "build_path_c_model_regions_from_model",
    "build_path_c_model_regions_from_route_symbols",
    "build_mamba3_fp8_train_acceptance_fixture_region",
    "build_mamba3_fp8_train_region",
    "build_mamba3_fp8_train_tilelang_region_from_prim_funcs",
    "compile_mamba3_fp8_train_tilelang_region_from_prim_funcs",
    "compile_path_c_region",
    "fused_path_c_default_eligible",
    "fused_path_c_plan_default_eligible",
    "mamba3_fp8_train_schedule_status_from_prim_funcs",
    "mark_path_c_schedule_template_for_region",
    "path_b_baseline_is_clean",
    "selected_path_c_fusion_mode",
    "tilelang_single_entry_lowerer",
    "trusted_path_c_production_schedule_ids",
]


FUSION_MODE_ENV = "CPPMEGA_PATH_C_FUSION"
DEFAULT_MAX_PATH_B_CACHE_GIB = 55.0
DEFAULT_MAX_PATH_B_MEDIAN_STEP_S = 10.0
DEFAULT_MIN_C_OVER_B = 1.0
DEFAULT_MAX_PEAK_DELTA_GIB = 0.0
DEFAULT_MAX_WARM_FIRST_STEP_S = 12.0
_TRUSTED_PRODUCTION_SCHEDULE_IDS: frozenset[str] = frozenset()
_MAMBA3_REAL_ABI_INPUTS = (
    "mamba3_in_proj_weight",
    "mamba3_out_proj_weight",
    "mamba3_conv_weight",
    "mamba3_conv_bias",
    "mamba3_dt_bias",
    "mamba3_B_norm_weight",
    "mamba3_B_bias",
    "mamba3_C_norm_weight",
    "mamba3_C_bias",
    "mamba3_D",
    "mamba3_h0",
)
_M2RNN_REAL_ABI_INPUTS = (
    "m2rnn_in_proj_weight",
    "m2rnn_conv_weight",
    "m2rnn_conv_bias",
    "m2rnn_state_weight",
    "m2rnn_A_log",
    "m2rnn_dt_bias",
    "m2rnn_D",
    "m2rnn_g_norm_weight",
    "m2rnn_out_proj_weight",
    "m2rnn_h0",
    "m2rnn_conv_state",
)
_ATTENTION_QKV_REAL_ABI_INPUTS = (
    "attention_q_proj_weight",
    "attention_q_proj_bias",
    "attention_sparse_kv_proj_weight",
    "attention_sparse_kv_proj_bias",
    "attention_rope_inv_freq",
)
_ATTENTION_OUT_PROJ_REAL_ABI_INPUTS = (
    "attention_out_proj_weight",
    "attention_out_proj_bias",
)
_ATTENTION_REAL_ABI_INPUTS = (
    *_ATTENTION_QKV_REAL_ABI_INPUTS,
    *_ATTENTION_OUT_PROJ_REAL_ABI_INPUTS,
)
_SPARSE_MLA_REAL_ABI_INPUTS = (
    "sparse_mla_sm_scale",
    "sparse_mla_sinks",
    "sparse_mla_has_sinks",
)
_MAMBA3_FP8_TRAIN_RESIDUAL_RMSNORM_INPUTS = (
    "mamba3_residual_to_m2rnn_norm_weight",
    "m2rnn_residual_to_attention_norm_weight",
)
MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS = (
    *_MAMBA3_REAL_ABI_INPUTS,
    *_MAMBA3_FP8_TRAIN_RESIDUAL_RMSNORM_INPUTS,
    *_M2RNN_REAL_ABI_INPUTS,
    *_ATTENTION_REAL_ABI_INPUTS,
    *_SPARSE_MLA_REAL_ABI_INPUTS,
)


def trusted_path_c_production_schedule_ids() -> frozenset[str]:
    """Return production schedule IDs accepted by the current build."""

    return _TRUSTED_PRODUCTION_SCHEDULE_IDS


class PathCFusionMode(Enum):
    """Top-level Path C fusion optimization mode."""

    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


@dataclass(frozen=True)
class PathCModelBrick:
    """Allocation-free model-brick descriptor used by automatic Path C discovery."""

    name: str
    kind: str
    route_symbol: str | None = None


@dataclass(frozen=True)
class _ResolvedPathCModelBrick:
    name: str
    kind: str
    route_symbol: str


@dataclass(frozen=True)
class _PathCBrickSurfaceLoweringResult:
    delta_output: str


@dataclass
class _PathCBrickSurfaceLoweringContext:
    surfaces: list[FusionKernelSurface] = field(default_factory=list)
    residual_hidden: str = "hidden"
    route_hidden: str = "hidden"


@dataclass(frozen=True)
class _PathCModelBrickSurfaceLowerer:
    route_symbol: str
    emit: Callable[
        [_ResolvedPathCModelBrick, _PathCBrickSurfaceLoweringContext],
        _PathCBrickSurfaceLoweringResult,
    ]


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)


@dataclass(frozen=True)
class PathCModelShapeEnv:
    """Shape data carried from a concrete model into generic Path C schedules."""

    sequence_length: int
    hidden_size: int
    attention_num_q_heads: int
    attention_num_kv_heads: int
    attention_head_dim: int
    attention_sparse_topk: int
    mamba_expand: int
    mamba_head_dim: int
    mamba_state_dim: int
    mamba_groups: int
    mamba_mimo_rank: int
    mamba_is_mimo: bool
    mamba_conv_kernel: int
    mamba_rope_fraction: float
    m2rnn_k_head_dim: int
    m2rnn_v_head_dim: int
    m2rnn_num_q_heads: int
    m2rnn_num_k_heads: int
    m2rnn_num_v_heads: int
    m2rnn_num_f_heads: int
    m2rnn_num_g_heads: int
    m2rnn_num_weight_heads: int
    m2rnn_conv_kernel: int

    @property
    def mamba_inner_dim(self) -> int:
        return self.hidden_size * self.mamba_expand

    @property
    def mamba_num_heads(self) -> int:
        return self.mamba_inner_dim // self.mamba_head_dim

    @property
    def mamba_effective_mimo_rank(self) -> int:
        return self.mamba_mimo_rank if self.mamba_is_mimo else 1

    @property
    def mamba_bc_dim(self) -> int:
        return (
            self.mamba_state_dim
            * self.mamba_groups
            * self.mamba_effective_mimo_rank
        )

    @property
    def mamba_num_rope_angles(self) -> int:
        split = int(self.mamba_state_dim * self.mamba_rope_fraction)
        if split % 2 != 0:
            split -= 1
        return max(1, split // 2)

    @property
    def mamba_in_proj_dim(self) -> int:
        return (
            2 * self.mamba_inner_dim
            + 2 * self.mamba_bc_dim
            + 3 * self.mamba_num_heads
            + self.mamba_num_rope_angles
        )

    @property
    def mamba_conv_channels(self) -> int:
        return self.mamba_inner_dim + 2 * self.mamba_bc_dim

    @property
    def m2rnn_num_heads(self) -> int:
        return max(
            self.m2rnn_num_q_heads,
            self.m2rnn_num_k_heads,
            self.m2rnn_num_v_heads,
            self.m2rnn_num_f_heads,
            self.m2rnn_num_g_heads,
            self.m2rnn_num_weight_heads,
        )

    @property
    def m2rnn_q_dim(self) -> int:
        return self.m2rnn_num_q_heads * self.m2rnn_k_head_dim

    @property
    def m2rnn_k_dim(self) -> int:
        return self.m2rnn_num_k_heads * self.m2rnn_k_head_dim

    @property
    def m2rnn_v_dim(self) -> int:
        return self.m2rnn_num_v_heads * self.m2rnn_v_head_dim

    @property
    def m2rnn_conv_dim(self) -> int:
        return self.m2rnn_q_dim + self.m2rnn_k_dim + self.m2rnn_v_dim

    @property
    def m2rnn_g_dim(self) -> int:
        return self.m2rnn_num_g_heads * self.m2rnn_v_head_dim

    @property
    def m2rnn_in_proj_dim(self) -> int:
        return self.m2rnn_conv_dim + self.m2rnn_num_f_heads + self.m2rnn_g_dim


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
    fusion_kind: str = "tilelang_ir_graph_region"
    schedule_name: str = "manual"
    schedule_status: str = "missing_real_fused_schedule_template"
    single_kernel_fused: bool = False
    autograd_mode: str = "none"
    autograd_status: str = "none"
    autograd_backward_nodes: tuple[str, ...] = ()
    autograd_backward_edges: tuple[tuple[str, str, str], ...] = ()
    autograd_missing_backward_nodes: tuple[str, ...] = ()
    semantic_blockers: tuple["FusionSemanticBlocker", ...] = ()
    schedule_contract: "FusionScheduleContractStatus | None" = None
    requires_msl_post_fusion: bool = False
    large_tensor_staging_allowed: bool = False


@dataclass(frozen=True)
class CompiledPathCRegion:
    """Result wrapper for a future TileLang/TVM region compiler."""

    plan: FusionCompilePlan
    artifact: object | None = None


@dataclass(frozen=True)
class FusionBufferSignature:
    """A raw PrimFunc buffer ABI signature used for fusion diagnostics."""

    name: str
    shape: tuple[str, ...]
    dtype: str

    @property
    def summary(self) -> str:
        return f"{self.name}:{'x'.join(self.shape)}:{self.dtype}"


@dataclass(frozen=True)
class FusionSemanticBlocker:
    """A graph-level blocker that cannot be solved by raw ABI renaming."""

    kind: str
    producer: str
    consumer: str
    required_node: str
    reason: str


@dataclass(frozen=True)
class FusionScheduleAbiBlocker:
    producer: str
    consumer: str
    buffer: str
    producer_buffers: tuple[str, ...]
    consumer_buffers: tuple[str, ...]
    producer_signature: str = ""
    consumer_signature: str = ""
    kind: str = "raw_abi_mismatch"
    reason: str = (
        "raw PrimFunc ABI does not expose the logical producer/consumer buffer "
        "on both endpoints"
    )


@dataclass(frozen=True)
class FusionScheduleStatus:
    status: str
    reason: str
    single_kernel_fused: bool
    blocked_edges: tuple[FusionScheduleAbiBlocker, ...] = ()


@dataclass(frozen=True)
class FusionScheduleContract:
    """Single-entry schedule contract derived from a Path C fusion region."""

    name: str
    key: str
    op_signature: tuple[str, ...]
    required_internal_buffers: tuple[str, ...]
    required_external_buffers: tuple[str, ...]
    shape_env_key: str = ""


@dataclass(frozen=True)
class FusionScheduleContractStatus:
    """Verification status for a region-level fused schedule contract."""

    name: str
    key: str
    status: str
    reason: str
    op_signature: tuple[str, ...]
    required_internal_buffers: tuple[str, ...]
    required_external_buffers: tuple[str, ...]
    shape_env_key: str = ""
    declared_key: str = ""
    declared_implementation_kind: str = ""
    declared_schedule_id: str = ""
    declared_required_real_abi_inputs: tuple[str, ...] = ()
    missing_real_abi_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SparseMLAFp8PrepareSpec:
    """Shape-specialized FP8 prepare producer for mamba3 train-block fusion."""

    q_rows: int
    kv_rows: int
    K: int
    in_dtype: str
    storage_dtype: str = "uint8"

    def build_prim_func(self) -> Any:
        from cppmega_mlx.nn._tilelang import sparse_mla_fp8_path_c

        return sparse_mla_fp8_path_c.make_fp8_sparse_mla_prepare_kernel(
            q_rows=self.q_rows,
            kv_rows=self.kv_rows,
            K=self.K,
            in_dtype=self.in_dtype,
            storage_dtype=self.storage_dtype,
        )


class PathCFusionRegionBuilder:
    """Build pre-lowering Path C fusion regions."""

    def __init__(
        self,
        name: str,
        *,
        z3_sync: Z3SyncSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._name = _require_identifier(name, label="name")
        self._z3_sync = z3_sync or Z3SyncSpec.disabled()
        self._metadata = dict(metadata or {})
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

    def add_kernels(
        self,
        surfaces: Sequence[FusionKernelSurface],
    ) -> "PathCFusionRegionBuilder":
        for surface in surfaces:
            self.add_kernel(surface)
        return self

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
        edges = _infer_edges(self._nodes)
        nodes = _nodes_in_dependency_order(self._nodes, edges)
        return PathCFusionRegion(
            name=self._name,
            nodes=nodes,
            edges=_edges_in_dependency_order(edges, nodes),
            z3_sync=self._z3_sync,
            metadata=dict(self._metadata),
        )


def _mamba3_fp8_train_surfaces() -> tuple[FusionKernelSurface, ...]:
    return (
        FusionKernelSurface.path_c(
            name="mamba3_scan",
            op_name="mamba3_mimo",
            inputs=("hidden", "mamba_state"),
            outputs=("scan_y", "scan_state"),
            backward="aot_autograd",
        ),
        FusionKernelSurface.path_c(
            name="m2rnn_packed_post",
            op_name="m2rnn",
            inputs=("scan_y",),
            outputs=("post_y",),
            backward="aot_autograd",
        ),
        FusionKernelSurface.path_c(
            name="fp8_prepare",
            op_name="sparse_mla_fp8_prepare",
            inputs=("post_y",),
            outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
            backward="owner_output",
        ),
        FusionKernelSurface.path_c(
            name="sparse_mla_fp8_apply",
            op_name="sparse_mla_fp8_apply",
            inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"),
            outputs=("attention_out", "lse"),
            backward="owner_output",
        ),
    )


_PATH_C_BRICK_KIND_ROUTE_SYMBOLS: Mapping[str, str] = {
    "M": "M",
    "mamba3": "M",
    "mamba3_mimo": "M",
    "mamba3_scan": "M",
    "R": "R",
    "m2rnn": "R",
    "m2rnn_packed_post": "R",
    "A": "A",
    "sparse_mla_fp8": "A",
    "sparse_mla_fp8_apply": "A",
    "sparse_mla_fp8_attention": "A",
}


def build_path_c_model_region_from_route_symbols(
    *,
    region_name: str,
    route_symbols: Sequence[str],
    z3_sync: Z3SyncSpec | None = None,
    include_backward: bool = False,
    shape_env: PathCModelShapeEnv | None = None,
    model_config: Any | None = None,
    acceptance_tags: Sequence[str] = (),
) -> PathCFusionRegion:
    """Build a Path C fusion region from a model route-symbol chain.

    This is the generic model-brick entrypoint.  Named schedules such as the
    Mamba3 FP8 train block may still apply acceptance metadata later, but the
    region itself is derived from the caller's route-symbol chain.
    """

    return _build_path_c_model_region_from_route_symbols(
        region_name=region_name,
        route_symbols=route_symbols,
        z3_sync=z3_sync,
        include_backward=include_backward,
        shape_env=shape_env,
        model_config=model_config,
        acceptance_tags=acceptance_tags,
        acceptance_fixture_abi=False,
    )


def _build_path_c_acceptance_fixture_region_from_route_symbols(
    *,
    region_name: str,
    route_symbols: Sequence[str],
    z3_sync: Z3SyncSpec | None = None,
    include_backward: bool = False,
    shape_env: PathCModelShapeEnv | None = None,
    model_config: Any | None = None,
    acceptance_tags: Sequence[str] = (),
) -> PathCFusionRegion:
    """Build an explicit acceptance fixture with legacy shared ABI names."""

    return _build_path_c_model_region_from_route_symbols(
        region_name=region_name,
        route_symbols=route_symbols,
        z3_sync=z3_sync,
        include_backward=include_backward,
        shape_env=shape_env,
        model_config=model_config,
        acceptance_tags=acceptance_tags,
        acceptance_fixture_abi=True,
    )


def _build_path_c_model_region_from_route_symbols(
    *,
    region_name: str,
    route_symbols: Sequence[str],
    z3_sync: Z3SyncSpec | None,
    include_backward: bool,
    shape_env: PathCModelShapeEnv | None,
    model_config: Any | None,
    acceptance_tags: Sequence[str],
    acceptance_fixture_abi: bool,
) -> PathCFusionRegion:
    resolved_shape_env = shape_env or _path_c_model_shape_env_from_config(model_config)
    metadata: dict[str, Any] = {
        "path_c_route_symbols": tuple(route_symbols),
        "path_c_acceptance_tags": tuple(str(tag) for tag in acceptance_tags),
        "path_c_acceptance_fixture_abi": acceptance_fixture_abi,
    }
    if resolved_shape_env is not None:
        metadata["path_c_model_shape_env"] = resolved_shape_env
    region = build_path_c_fusion_region(
        region_name=region_name,
        surfaces=_path_c_model_surfaces_from_route_symbols(
            route_symbols,
            shared_acceptance_abi=acceptance_fixture_abi,
        ),
        z3_sync=z3_sync or Z3SyncSpec.minimize_sync_async(),
        metadata=metadata,
    )
    if include_backward:
        return build_path_c_aot_autograd_region(region)
    return region


def build_path_c_model_regions_from_route_symbols(
    route_symbols: Sequence[str],
    *,
    region_prefix: str = "path_c_model_segment",
    z3_sync: Z3SyncSpec | None = None,
    include_backward: bool = False,
    min_route_bricks: int = 2,
    shape_env: PathCModelShapeEnv | None = None,
    model_config: Any | None = None,
    acceptance_tags: Sequence[str] = (),
) -> tuple[PathCFusionRegion, ...]:
    """Split a model route pattern into supported Path C fusion regions."""

    if min_route_bricks <= 0:
        raise ValueError("min_route_bricks must be positive")
    resolved_shape_env = shape_env or _path_c_model_shape_env_from_config(model_config)
    regions: list[PathCFusionRegion] = []
    segment: list[str] = []
    segment_start = 0
    for index, raw_symbol in enumerate(route_symbols):
        symbol = _normalized_route_symbol(raw_symbol)
        if _path_c_model_brick_surface_lowerer_for(symbol) is not None:
            if not segment:
                segment_start = index
            segment.append(symbol)
            continue
        _append_path_c_model_segment(
            regions,
            segment,
            segment_start=segment_start,
            region_prefix=region_prefix,
            z3_sync=z3_sync,
            include_backward=include_backward,
            min_route_bricks=min_route_bricks,
            shape_env=resolved_shape_env,
            acceptance_tags=acceptance_tags,
        )
        segment = []
    _append_path_c_model_segment(
        regions,
        segment,
        segment_start=segment_start,
        region_prefix=region_prefix,
        z3_sync=z3_sync,
        include_backward=include_backward,
        min_route_bricks=min_route_bricks,
        shape_env=resolved_shape_env,
        acceptance_tags=acceptance_tags,
    )
    return tuple(regions)


def build_path_c_model_regions_from_bricks(
    bricks: Sequence[Any],
    *,
    region_prefix: str = "path_c_model_segment",
    z3_sync: Z3SyncSpec | None = None,
    include_backward: bool = False,
    min_route_bricks: int = 2,
    shape_env: PathCModelShapeEnv | None = None,
    model_config: Any | None = None,
    acceptance_tags: Sequence[str] = (),
) -> tuple[PathCFusionRegion, ...]:
    """Split an allocation-free model brick graph into Path C fusion regions."""

    if min_route_bricks <= 0:
        raise ValueError("min_route_bricks must be positive")
    resolved_shape_env = shape_env or _path_c_model_shape_env_from_config(model_config)
    resolved_bricks = tuple(
        _resolved_path_c_model_brick(brick, index=index)
        for index, brick in enumerate(bricks)
    )
    regions: list[PathCFusionRegion] = []
    segment: list[_ResolvedPathCModelBrick] = []
    segment_start = 0
    for index, brick in enumerate(resolved_bricks):
        if _path_c_model_brick_surface_lowerer_for(brick.route_symbol) is not None:
            if not segment:
                segment_start = index
            segment.append(brick)
            continue
        _append_path_c_model_brick_segment(
            regions,
            segment,
            segment_start=segment_start,
            region_prefix=region_prefix,
            z3_sync=z3_sync,
            include_backward=include_backward,
            min_route_bricks=min_route_bricks,
            shape_env=resolved_shape_env,
            acceptance_tags=acceptance_tags,
        )
        segment = []
    _append_path_c_model_brick_segment(
        regions,
        segment,
        segment_start=segment_start,
        region_prefix=region_prefix,
        z3_sync=z3_sync,
        include_backward=include_backward,
        min_route_bricks=min_route_bricks,
        shape_env=resolved_shape_env,
        acceptance_tags=acceptance_tags,
    )
    return tuple(regions)


def build_path_c_model_region_from_bricks(
    *,
    region_name: str,
    bricks: Sequence[Any],
    z3_sync: Z3SyncSpec | None = None,
    include_backward: bool = False,
    shape_env: PathCModelShapeEnv | None = None,
    model_config: Any | None = None,
    acceptance_tags: Sequence[str] = (),
) -> PathCFusionRegion:
    """Build a Path C fusion region directly from allocation-free model bricks."""

    resolved_bricks = tuple(
        _resolved_path_c_model_brick(brick, index=index)
        for index, brick in enumerate(bricks)
    )
    if not resolved_bricks:
        raise ValueError("bricks must contain at least one model brick")
    supported_symbols = _path_c_supported_route_symbols()
    unsupported = sorted(
        {
            brick.route_symbol
            for brick in resolved_bricks
            if brick.route_symbol not in supported_symbols
        }
    )
    if unsupported:
        raise ValueError(
            "Path C model region only supports route symbols "
            f"{sorted(supported_symbols)!r}, got unsupported "
            f"{unsupported!r}"
        )
    resolved_shape_env = shape_env or _path_c_model_shape_env_from_config(model_config)
    metadata: dict[str, Any] = {
        "path_c_route_symbols": tuple(brick.route_symbol for brick in resolved_bricks),
        "path_c_bricks": tuple(_path_c_brick_metadata(brick) for brick in resolved_bricks),
        "path_c_acceptance_tags": tuple(str(tag) for tag in acceptance_tags),
    }
    if resolved_shape_env is not None:
        metadata["path_c_model_shape_env"] = resolved_shape_env
    region = build_path_c_fusion_region(
        region_name=region_name,
        surfaces=_path_c_model_surfaces_from_bricks(resolved_bricks),
        z3_sync=z3_sync or Z3SyncSpec.minimize_sync_async(),
        metadata=metadata,
    )
    if include_backward:
        return build_path_c_aot_autograd_region(region)
    return region


def build_path_c_model_regions_from_model(
    model: Any,
    *,
    region_prefix: str | None = None,
    z3_sync: Z3SyncSpec | None = None,
    include_backward: bool = False,
    min_route_bricks: int = 2,
) -> tuple[PathCFusionRegion, ...]:
    """Return Path C fusion regions discovered from a model's brick graph."""

    bricks = _path_c_bricks_from_model(model)
    shape_env = _path_c_model_shape_env_from_model(model)
    return build_path_c_model_regions_from_bricks(
        bricks,
        region_prefix=region_prefix or _path_c_region_prefix_for_model(model),
        z3_sync=z3_sync,
        include_backward=include_backward,
        min_route_bricks=min_route_bricks,
        shape_env=shape_env,
    )


def _path_c_bricks_from_model(model: Any) -> tuple[Any, ...]:
    bricks = getattr(model, "path_c_bricks", None)
    if callable(bricks):
        bricks = bricks()
    if bricks is not None:
        return tuple(bricks)
    kinds = getattr(model, "kinds", None)
    if callable(kinds):
        return tuple(
            PathCModelBrick(name=f"block_{index}_{kind}", kind=str(kind))
            for index, kind in enumerate(kinds())
        )
    return tuple(
        PathCModelBrick(
            name=f"route_{index}_{symbol}",
            kind=str(symbol),
            route_symbol=str(symbol),
        )
        for index, symbol in enumerate(_route_symbols_from_model(model))
    )


def _path_c_route_symbol_from_brick(brick: Any) -> str:
    route_symbol = _brick_attr(brick, "route_symbol")
    if route_symbol is not None:
        return _normalized_route_symbol(str(route_symbol))
    kind = _brick_attr(brick, "kind")
    if kind is None:
        raise ValueError("Path C model brick must expose kind or route_symbol")
    kind_text = str(kind)
    return _PATH_C_BRICK_KIND_ROUTE_SYMBOLS.get(
        kind_text,
        _PATH_C_BRICK_KIND_ROUTE_SYMBOLS.get(kind_text.lower(), kind_text),
    )


def _resolved_path_c_model_brick(
    brick: Any,
    *,
    index: int,
) -> _ResolvedPathCModelBrick:
    kind = _brick_attr(brick, "kind")
    route_symbol = _path_c_route_symbol_from_brick(brick)
    raw_name = _brick_attr(brick, "name")
    name = str(raw_name).strip() if raw_name is not None else ""
    if not name:
        name = f"brick_{index}_{route_symbol.lower()}"
    if kind is None:
        kind = route_symbol
    return _ResolvedPathCModelBrick(
        name=_require_identifier(name, label="brick.name"),
        kind=str(kind),
        route_symbol=route_symbol,
    )


def _path_c_brick_metadata(
    brick: _ResolvedPathCModelBrick,
) -> dict[str, str]:
    return {
        "name": brick.name,
        "kind": brick.kind,
        "route_symbol": brick.route_symbol,
    }


def _brick_attr(brick: Any, name: str) -> Any | None:
    if isinstance(brick, Mapping):
        return brick.get(name)
    return getattr(brick, name, None)


def _route_symbols_from_model(model: Any) -> tuple[str, ...]:
    route_symbols = getattr(model, "route_symbols", None)
    if route_symbols is None:
        raise ValueError(
            "model must expose path_c_bricks, kinds(), or route_symbols for "
            "Path C fusion discovery"
        )
    if isinstance(route_symbols, str):
        return tuple(route_symbols)
    return tuple(str(symbol) for symbol in route_symbols)


def _path_c_region_prefix_for_model(model: Any) -> str:
    raw_name = (
        getattr(model, "name", None)
        or getattr(getattr(model, "cfg", None), "name", None)
        or getattr(getattr(model, "config", None), "name", None)
        or type(model).__name__
    )
    safe = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(raw_name)
    ).strip("_")
    return f"{safe or 'model'}_path_c"


def _path_c_model_shape_env_from_model(model: Any) -> PathCModelShapeEnv | None:
    config = (
        getattr(model, "config", None)
        or getattr(model, "cfg", None)
        or getattr(model, "model_config", None)
    )
    return _path_c_model_shape_env_from_config(config)


def _path_c_model_shape_env_from_config(config: Any | None) -> PathCModelShapeEnv | None:
    if config is None:
        return None
    try:
        mamba_config = config.mamba3_config()
        m2rnn_config = config.m2rnn_config()
        attention_config = config.attention_config("dsa")
        sequence_length = int(getattr(config, "max_seq_length"))
        hidden_size = int(getattr(config, "hidden_size"))
    except AttributeError:
        return None
    return PathCModelShapeEnv(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        attention_num_q_heads=int(attention_config.num_q_heads),
        attention_num_kv_heads=int(attention_config.kv_heads),
        attention_head_dim=int(attention_config.q_head_dim),
        attention_sparse_topk=int(attention_config.sparse_topk),
        mamba_expand=int(mamba_config.expand),
        mamba_head_dim=int(mamba_config.headdim),
        mamba_state_dim=int(mamba_config.d_state),
        mamba_groups=int(mamba_config.ngroups),
        mamba_mimo_rank=int(mamba_config.mimo_rank),
        mamba_is_mimo=bool(mamba_config.is_mimo),
        mamba_conv_kernel=int(mamba_config.d_conv),
        mamba_rope_fraction=float(mamba_config.rope_fraction),
        m2rnn_k_head_dim=int(m2rnn_config.k_head_dim),
        m2rnn_v_head_dim=int(m2rnn_config.v_head_dim),
        m2rnn_num_q_heads=int(m2rnn_config.num_q_heads),
        m2rnn_num_k_heads=int(m2rnn_config.num_k_heads),
        m2rnn_num_v_heads=int(m2rnn_config.num_v_heads),
        m2rnn_num_f_heads=int(m2rnn_config.num_f_heads),
        m2rnn_num_g_heads=int(m2rnn_config.num_g_heads),
        m2rnn_num_weight_heads=int(m2rnn_config.num_weight_heads),
        m2rnn_conv_kernel=int(m2rnn_config.conv_kernel),
    )


def _append_path_c_model_segment(
    regions: list[PathCFusionRegion],
    segment: Sequence[str],
    *,
    segment_start: int,
    region_prefix: str,
    z3_sync: Z3SyncSpec | None,
    include_backward: bool,
    min_route_bricks: int,
    shape_env: PathCModelShapeEnv | None,
    acceptance_tags: Sequence[str],
) -> None:
    if len(segment) < min_route_bricks:
        return
    end = segment_start + len(segment) - 1
    regions.append(
        build_path_c_model_region_from_route_symbols(
            region_name=f"{region_prefix}_{segment_start}_{end}",
            route_symbols=segment,
            z3_sync=z3_sync,
            include_backward=include_backward,
            shape_env=shape_env,
            acceptance_tags=acceptance_tags,
        )
    )


def _append_path_c_model_brick_segment(
    regions: list[PathCFusionRegion],
    segment: Sequence[_ResolvedPathCModelBrick],
    *,
    segment_start: int,
    region_prefix: str,
    z3_sync: Z3SyncSpec | None,
    include_backward: bool,
    min_route_bricks: int,
    shape_env: PathCModelShapeEnv | None,
    acceptance_tags: Sequence[str],
) -> None:
    if len(segment) < min_route_bricks:
        return
    end = segment_start + len(segment) - 1
    regions.append(
        build_path_c_model_region_from_bricks(
            region_name=f"{region_prefix}_{segment_start}_{end}",
            bricks=segment,
            z3_sync=z3_sync,
            include_backward=include_backward,
            shape_env=shape_env,
            acceptance_tags=acceptance_tags,
        )
    )


def _path_c_model_surfaces_from_route_symbols(
    route_symbols: Sequence[str],
    *,
    shared_acceptance_abi: bool,
) -> tuple[FusionKernelSurface, ...]:
    normalized = tuple(_normalized_route_symbol(symbol) for symbol in route_symbols)
    if not normalized:
        raise ValueError("route_symbols must contain at least one route")
    supported_symbols = _path_c_supported_route_symbols()
    unsupported = sorted(set(normalized) - supported_symbols)
    if unsupported:
        raise ValueError(
            "Path C model region only supports route symbols "
            f"{sorted(supported_symbols)!r}, got unsupported "
            f"{unsupported!r}"
        )

    surfaces: list[FusionKernelSurface] = []
    residual_hidden = "hidden"
    route_hidden = "hidden"
    mamba_count = 0
    m2rnn_count = 0
    attention_count = 0
    for index, symbol in enumerate(normalized):
        next_symbol = normalized[index + 1] if index + 1 < len(normalized) else ""
        if symbol == "M":
            mamba_count += 1
            name = "mamba3_scan" if shared_acceptance_abi and mamba_count == 1 else f"mamba3_scan_{mamba_count}"
            delta = "mamba3_delta" if shared_acceptance_abi and mamba_count == 1 else f"{name}_delta"
            state = "scan_state" if shared_acceptance_abi and mamba_count == 1 else f"{name}_state"
            surfaces.append(
                FusionKernelSurface.path_c(
                    name=name,
                    op_name="mamba3_mimo",
                    inputs=(route_hidden, "mamba_state", *_MAMBA3_REAL_ABI_INPUTS)
                    if shared_acceptance_abi and mamba_count == 1
                    else (
                        route_hidden,
                        f"{name}_state_in",
                        *_prefixed_abi_inputs(name, _MAMBA3_REAL_ABI_INPUTS),
                    ),
                    outputs=(delta, state),
                    backward="aot_autograd",
                )
            )
            if next_symbol:
                norm_name = (
                    "mamba3_residual_to_m2rnn_norm"
                    if shared_acceptance_abi and mamba_count == 1 and next_symbol == "R"
                    else f"{name}_residual_norm"
                )
                next_hidden = (
                    "m2rnn_hidden"
                    if shared_acceptance_abi and next_symbol == "R"
                    else f"{norm_name}_hidden"
                )
                hidden_after = (
                    "hidden_after_mamba3"
                    if shared_acceptance_abi and mamba_count == 1
                    else f"{name}_hidden_after"
                )
                surfaces.append(
                    _residual_norm_surface(
                        name=norm_name,
                        hidden=residual_hidden,
                        delta=delta,
                        norm_weight=(
                            "mamba3_residual_to_m2rnn_norm_weight"
                            if shared_acceptance_abi
                            and mamba_count == 1
                            and next_symbol == "R"
                            else f"{norm_name}_weight"
                        ),
                        hidden_after=hidden_after,
                        next_hidden=next_hidden,
                    )
                )
                residual_hidden = hidden_after
                route_hidden = next_hidden
            continue
        if symbol == "R":
            m2rnn_count += 1
            name = "m2rnn_packed_post" if shared_acceptance_abi and m2rnn_count == 1 else f"m2rnn_packed_post_{m2rnn_count}"
            delta = "m2rnn_delta" if shared_acceptance_abi and m2rnn_count == 1 else f"{name}_delta"
            surfaces.append(
                FusionKernelSurface.path_c(
                    name=name,
                    op_name="m2rnn",
                    inputs=(route_hidden, *_M2RNN_REAL_ABI_INPUTS)
                    if shared_acceptance_abi and m2rnn_count == 1
                    else (
                        route_hidden,
                        *_prefixed_abi_inputs(name, _M2RNN_REAL_ABI_INPUTS),
                    ),
                    outputs=(delta,),
                    backward="aot_autograd",
                )
            )
            if next_symbol:
                norm_name = (
                    "m2rnn_residual_to_attention_norm"
                    if shared_acceptance_abi and m2rnn_count == 1 and next_symbol == "A"
                    else f"{name}_residual_norm"
                )
                next_hidden = (
                    "attention_hidden"
                    if shared_acceptance_abi and next_symbol == "A"
                    else f"{norm_name}_hidden"
                )
                hidden_after = (
                    "hidden_after_m2rnn"
                    if shared_acceptance_abi and m2rnn_count == 1
                    else f"{name}_hidden_after"
                )
                surfaces.append(
                    _residual_norm_surface(
                        name=norm_name,
                        hidden=residual_hidden,
                        delta=delta,
                        norm_weight=(
                            "m2rnn_residual_to_attention_norm_weight"
                            if shared_acceptance_abi
                            and m2rnn_count == 1
                            and next_symbol == "A"
                            else f"{norm_name}_weight"
                        ),
                        hidden_after=hidden_after,
                        next_hidden=next_hidden,
                    )
                )
                residual_hidden = hidden_after
                route_hidden = next_hidden
            continue
        attention_count += 1
        projection_name = (
            "attention_qkv_projection"
            if shared_acceptance_abi and attention_count == 1
            else f"attention_qkv_projection_{attention_count}"
        )
        apply_name = (
            "sparse_mla_fp8_apply"
            if shared_acceptance_abi and attention_count == 1
            else f"sparse_mla_fp8_apply_{attention_count}"
        )
        q_fp8 = "q_fp8" if shared_acceptance_abi and attention_count == 1 else f"{projection_name}_q_fp8"
        q_scale = "q_scale" if shared_acceptance_abi and attention_count == 1 else f"{projection_name}_q_scale"
        kv_fp8 = "kv_fp8" if shared_acceptance_abi and attention_count == 1 else f"{projection_name}_kv_fp8"
        kv_scale = "kv_scale" if shared_acceptance_abi and attention_count == 1 else f"{projection_name}_kv_scale"
        indices = "indices" if shared_acceptance_abi and attention_count == 1 else f"{projection_name}_indices"
        surfaces.append(
            FusionKernelSurface.path_c(
                name=projection_name,
                op_name="attention_qkv_projection",
                inputs=(route_hidden, *_ATTENTION_QKV_REAL_ABI_INPUTS)
                if shared_acceptance_abi and attention_count == 1
                else (
                    route_hidden,
                    *_prefixed_abi_inputs(
                        projection_name,
                        _ATTENTION_QKV_REAL_ABI_INPUTS,
                    ),
                ),
                outputs=(q_fp8, q_scale, kv_fp8, kv_scale, indices),
                backward="aot_autograd",
            )
        )
        surfaces.append(
            FusionKernelSurface.path_c(
                name=apply_name,
                op_name="sparse_mla_fp8_apply",
                inputs=(
                    q_fp8,
                    q_scale,
                    kv_fp8,
                    kv_scale,
                    indices,
                    *_SPARSE_MLA_REAL_ABI_INPUTS,
                    *_ATTENTION_OUT_PROJ_REAL_ABI_INPUTS,
                )
                if shared_acceptance_abi and attention_count == 1
                else (
                    q_fp8,
                    q_scale,
                    kv_fp8,
                    kv_scale,
                    indices,
                    *_prefixed_abi_inputs(
                        apply_name,
                        (
                            *_SPARSE_MLA_REAL_ABI_INPUTS,
                            *_ATTENTION_OUT_PROJ_REAL_ABI_INPUTS,
                        ),
                    ),
                ),
                outputs=("attention_out", "lse")
                if shared_acceptance_abi and attention_count == 1
                else (f"{apply_name}_out", f"{apply_name}_lse"),
                backward="owner_output",
            )
        )
    return tuple(surfaces)


def _path_c_model_surfaces_from_bricks(
    bricks: Sequence[_ResolvedPathCModelBrick],
) -> tuple[FusionKernelSurface, ...]:
    if not bricks:
        raise ValueError("bricks must contain at least one route")

    context = _PathCBrickSurfaceLoweringContext()
    for index, brick in enumerate(bricks):
        lowerer = _path_c_model_brick_surface_lowerer_for(brick.route_symbol)
        if lowerer is None:
            supported_symbols = _path_c_supported_route_symbols()
            raise ValueError(
                "Path C model region only supports route symbols "
                f"{sorted(supported_symbols)!r}, got unsupported "
                f"{brick.route_symbol!r}"
            )
        result = lowerer.emit(brick, context)
        if index + 1 < len(bricks):
            _append_path_c_inter_brick_residual_norm(
                context,
                brick=brick,
                delta=result.delta_output,
            )
    return tuple(context.surfaces)


def _path_c_supported_route_symbols() -> frozenset[str]:
    return frozenset(default_path_c_model_brick_surface_lowerers())


def default_path_c_model_brick_surface_lowerers() -> (
    Mapping[str, _PathCModelBrickSurfaceLowerer]
):
    """Return route-brick lowerers used by automatic Path C graph discovery."""

    return {
        "M": _PathCModelBrickSurfaceLowerer(
            route_symbol="M",
            emit=_emit_mamba3_model_brick_surfaces,
        ),
        "R": _PathCModelBrickSurfaceLowerer(
            route_symbol="R",
            emit=_emit_m2rnn_model_brick_surfaces,
        ),
        "A": _PathCModelBrickSurfaceLowerer(
            route_symbol="A",
            emit=_emit_attention_model_brick_surfaces,
        ),
    }


def _path_c_model_brick_surface_lowerer_for(
    route_symbol: str,
) -> _PathCModelBrickSurfaceLowerer | None:
    return default_path_c_model_brick_surface_lowerers().get(
        _normalized_route_symbol(route_symbol)
    )


def _emit_mamba3_model_brick_surfaces(
    brick: _ResolvedPathCModelBrick,
    context: _PathCBrickSurfaceLoweringContext,
) -> _PathCBrickSurfaceLoweringResult:
    name = brick.name
    delta = f"{name}_delta"
    state = f"{name}_state"
    context.surfaces.append(
        FusionKernelSurface.path_c(
            name=name,
            op_name="mamba3_mimo",
            inputs=(
                context.route_hidden,
                f"{name}_state_in",
                *_prefixed_abi_inputs(name, _MAMBA3_REAL_ABI_INPUTS),
            ),
            outputs=(delta, state),
            backward="aot_autograd",
        )
    )
    return _PathCBrickSurfaceLoweringResult(delta_output=delta)


def _emit_m2rnn_model_brick_surfaces(
    brick: _ResolvedPathCModelBrick,
    context: _PathCBrickSurfaceLoweringContext,
) -> _PathCBrickSurfaceLoweringResult:
    name = brick.name
    delta = f"{name}_delta"
    context.surfaces.append(
        FusionKernelSurface.path_c(
            name=name,
            op_name="m2rnn",
            inputs=(
                context.route_hidden,
                *_prefixed_abi_inputs(name, _M2RNN_REAL_ABI_INPUTS),
            ),
            outputs=(delta,),
            backward="aot_autograd",
        )
    )
    return _PathCBrickSurfaceLoweringResult(delta_output=delta)


def _emit_attention_model_brick_surfaces(
    brick: _ResolvedPathCModelBrick,
    context: _PathCBrickSurfaceLoweringContext,
) -> _PathCBrickSurfaceLoweringResult:
    projection_name = f"{brick.name}_qkv_projection"
    apply_name = f"{brick.name}_sparse_mla_fp8_apply"
    q_fp8 = f"{projection_name}_q_fp8"
    q_scale = f"{projection_name}_q_scale"
    kv_fp8 = f"{projection_name}_kv_fp8"
    kv_scale = f"{projection_name}_kv_scale"
    indices = f"{projection_name}_indices"
    delta = f"{apply_name}_out"
    context.surfaces.append(
        FusionKernelSurface.path_c(
            name=projection_name,
            op_name="attention_qkv_projection",
            inputs=(
                context.route_hidden,
                *_prefixed_abi_inputs(
                    projection_name,
                    _ATTENTION_QKV_REAL_ABI_INPUTS,
                ),
            ),
            outputs=(q_fp8, q_scale, kv_fp8, kv_scale, indices),
            backward="aot_autograd",
        )
    )
    context.surfaces.append(
        FusionKernelSurface.path_c(
            name=apply_name,
            op_name="sparse_mla_fp8_apply",
            inputs=(
                q_fp8,
                q_scale,
                kv_fp8,
                kv_scale,
                indices,
                *_prefixed_abi_inputs(
                    apply_name,
                    (
                        *_SPARSE_MLA_REAL_ABI_INPUTS,
                        *_ATTENTION_OUT_PROJ_REAL_ABI_INPUTS,
                    ),
                ),
            ),
            outputs=(delta, f"{apply_name}_lse"),
            backward="owner_output",
        )
    )
    return _PathCBrickSurfaceLoweringResult(delta_output=delta)


def _append_path_c_inter_brick_residual_norm(
    context: _PathCBrickSurfaceLoweringContext,
    *,
    brick: _ResolvedPathCModelBrick,
    delta: str,
) -> None:
    norm_name = f"{brick.name}_residual_norm"
    hidden_after = f"{brick.name}_hidden_after"
    next_hidden = f"{norm_name}_hidden"
    context.surfaces.append(
        _residual_norm_surface(
            name=norm_name,
            hidden=context.residual_hidden,
            delta=delta,
            norm_weight=f"{norm_name}_weight",
            hidden_after=hidden_after,
            next_hidden=next_hidden,
        )
    )
    context.residual_hidden = hidden_after
    context.route_hidden = next_hidden


def _residual_norm_surface(
    *,
    name: str,
    hidden: str,
    delta: str,
    norm_weight: str,
    hidden_after: str,
    next_hidden: str,
) -> FusionKernelSurface:
    return FusionKernelSurface.path_c(
        name=name,
        op_name="residual_rmsnorm",
        inputs=(hidden, delta, norm_weight),
        outputs=(hidden_after, next_hidden),
        backward="aot_autograd",
    )


def _prefixed_abi_inputs(
    prefix: str,
    names: Sequence[str],
) -> tuple[str, ...]:
    return tuple(f"{prefix}_{name}" for name in names)


def _normalized_route_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if not normalized:
        raise ValueError("route symbol must not be empty")
    return normalized


def build_path_c_fusion_region(
    *,
    region_name: str,
    surfaces: Sequence[FusionKernelSurface],
    z3_sync: Z3SyncSpec | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PathCFusionRegion:
    """Build a Path C region from caller-supplied pre-lowering surfaces."""

    return (
        PathCFusionRegionBuilder(region_name, z3_sync=z3_sync, metadata=metadata)
        .add_kernels(surfaces)
        .build()
    )


_NON_DIFFERENTIABLE_FUSION_BUFFERS = frozenset({"indices", "lse", "scan_state"})


def _differentiable_buffer_names(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        name
        for name in names
        if name not in _NON_DIFFERENTIABLE_FUSION_BUFFERS and not name.endswith("_state")
    )


def _grad_buffer_names(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{name}_grad" for name in _differentiable_buffer_names(names))


def _surface_from_node(node: FusionNode) -> FusionKernelSurface:
    return FusionKernelSurface.path_c(
        name=node.name,
        op_name=node.op_name,
        inputs=node.inputs,
        outputs=node.outputs,
        backward=node.backward,
        backend=node.backend,
    )


def _aot_backward_surface_for_node(node: FusionNode) -> FusionKernelSurface:
    inputs = _grad_buffer_names(node.outputs)
    if node.op_name == "residual_rmsnorm":
        inputs = (*inputs, *node.inputs)
    return FusionKernelSurface.path_c(
        name=f"{node.name}_bwd",
        op_name=f"{node.op_name}_bwd",
        inputs=inputs,
        outputs=_grad_buffer_names(node.inputs),
        backward="owner_output",
    )


def _aot_backward_surfaces_for(region: PathCFusionRegion) -> tuple[FusionKernelSurface, ...]:
    return tuple(
        _aot_backward_surface_for_node(node)
        for node in reversed(region.nodes)
        if node.backward == "aot_autograd"
    )


def build_path_c_aot_autograd_region(
    region: PathCFusionRegion,
    *,
    region_name: str | None = None,
) -> PathCFusionRegion:
    """Return ``region`` plus symbolic AOT backward graph surfaces."""

    if not isinstance(region, PathCFusionRegion):
        raise TypeError("region must be PathCFusionRegion")
    return build_path_c_fusion_region(
        region_name=region_name or region.name,
        surfaces=(
            *(_surface_from_node(node) for node in region.nodes),
            *_aot_backward_surfaces_for(region),
        ),
        z3_sync=region.z3_sync,
        metadata=region.metadata,
    )


def build_mamba3_fp8_train_region() -> PathCFusionRegion:
    """Return the high-level Path C train-block template requested for 1B."""

    return build_path_c_fusion_region(
        region_name="mamba3_fp8_train_block",
        surfaces=_mamba3_fp8_train_surfaces(),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )


def build_mamba3_fp8_train_acceptance_fixture_region(
    *,
    include_backward: bool = False,
) -> PathCFusionRegion:
    """Return the explicit named acceptance fixture for MRA train-block tests.

    This helper is deliberately not the production model-discovery path.  Real
    model routes must enter through ``build_path_c_model_regions_from_model`` or
    ``build_path_c_model_regions_from_bricks`` so the fused chain follows the
    model's actual brick graph.
    """

    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile

    acceptance_profile = local_gb10_quarter_profile()
    return _build_path_c_acceptance_fixture_region_from_route_symbols(
        region_name="mamba3_m2rnn_attention_fp8_train_block",
        route_symbols=("M", "R", "A"),
        include_backward=include_backward,
        model_config=acceptance_profile.hybrid_config(),
        acceptance_tags=("mamba3_fp8_train_acceptance",),
    )


def build_mamba3_fp8_train_tilelang_region_from_prim_funcs(
    *,
    mamba3_scan: Any,
    m2rnn_packed_post: Any,
    fp8_prepare: Any | None = None,
    fp8_prepare_spec: SparseMLAFp8PrepareSpec | None = None,
    sparse_mla_fp8_prepared: Any,
    schedule_template: Callable[[Any], Any] | None = None,
) -> Any:
    """Build the TileLang pre-source train-block region from raw PrimFuncs."""

    fp8_prepare = _resolve_fp8_prepare_prim_func(fp8_prepare, fp8_prepare_spec)
    capture = io.StringIO()
    with redirect_stdout(capture), redirect_stderr(capture):
        from tilelang.engine.fusion import FusionRegionBuilder as TileLangFusionRegionBuilder

    builder = TileLangFusionRegionBuilder("mamba3_fp8_train_block")
    use_logical_fused_edges = schedule_template is not None
    builder.add_prim_func_node(
        "mamba3_scan",
        mamba3_scan,
        op="mamba3_mimo",
        inputs=("hidden", "mamba_state")
        if use_logical_fused_edges
        else ("x", "B", "C", "z", "A", "dt", "D", "h0"),
        outputs=("scan_y", "scan_state")
        if use_logical_fused_edges
        else ("y", "h_last", "h_snap"),
        attrs={"backward": "aot_autograd"},
    )
    builder.add_prim_func_node(
        "m2rnn_packed_post",
        m2rnn_packed_post,
        op="m2rnn",
        inputs=("scan_y",)
        if use_logical_fused_edges
        else ("conv_input", "W", "xf", "h0", "D", "projected"),
        outputs=("post_y",)
        if use_logical_fused_edges
        else ("h_last", "tanh_cache", "post"),
        attrs={"backward": "aot_autograd"},
    )
    if schedule_template is not None:
        if fp8_prepare is None:
            builder.add_node(
                "fp8_prepare",
                op="sparse_mla_fp8_prepare",
                inputs=("post_y",),
                outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
                attrs={"backward": "owner_output"},
            )
        else:
            builder.add_prim_func_node(
                "fp8_prepare",
                fp8_prepare,
                op="sparse_mla_fp8_prepare",
                inputs=("post_y",),
                outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
                attrs={"backward": "owner_output"},
            )
        builder.add_prim_func_node(
            "sparse_mla_fp8_apply",
            sparse_mla_fp8_prepared,
            op="sparse_mla_fp8_apply",
            inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale"),
            outputs=("attention_out", "lse"),
            attrs={"backward": "owner_output"},
        )
        builder.set_schedule_template(schedule_template)
    else:
        builder.add_prim_func_node(
            "sparse_mla_fp8_apply",
            sparse_mla_fp8_prepared,
            op="sparse_mla_fp8_apply",
            inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices", "sm_scale_buf", "sinks", "has_sinks"),
            outputs=("out", "lse"),
            attrs={"backward": "owner_output"},
        )
    return builder.enable_z3_sync_async_optimization().build()


def compile_mamba3_fp8_train_tilelang_region_from_prim_funcs(
    *,
    mamba3_scan: Any,
    m2rnn_packed_post: Any,
    fp8_prepare: Any | None = None,
    fp8_prepare_spec: SparseMLAFp8PrepareSpec | None = None,
    sparse_mla_fp8_prepared: Any,
    schedule_template: Callable[[Any], Any] | None = None,
    target: str = "metal",
    lowerer: Callable[..., Any] | None = None,
    require_single_kernel: bool = True,
) -> Any:
    """Compile the mamba3+packed_post+fp8 train region through TileLang fusion.

    The automatic raw-PrimFunc path is deliberately fail-closed until the real
    fused schedule exists.  Current raw surfaces do not expose the logical
    ``scan_y`` and ``post_y`` producer/consumer buffers needed for a single
    train-block kernel.
    """

    if schedule_template is None:
        status = mamba3_fp8_train_schedule_status_from_prim_funcs(
            mamba3_scan=mamba3_scan,
            m2rnn_packed_post=m2rnn_packed_post,
            fp8_prepare=fp8_prepare,
            fp8_prepare_spec=fp8_prepare_spec,
            sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
        )
        blocked_edges = "; ".join(
            (
                f"{edge.kind}:{edge.producer}->{edge.consumer}:{edge.buffer} "
                f"reason={edge.reason!r} "
                f"producer_buffers={list(edge.producer_buffers)!r} "
                f"consumer_buffers={list(edge.consumer_buffers)!r} "
                f"producer_signature={edge.producer_signature!r} "
                f"consumer_signature={edge.consumer_signature!r}"
            )
            for edge in status.blocked_edges
        )
        raise RuntimeError(
            f"{status.status}: {status.reason}; blocked_edges={blocked_edges}"
        )

    capture = io.StringIO()
    with redirect_stdout(capture), redirect_stderr(capture):
        from tilelang.engine.fusion import compile_fusion_region

    region = build_mamba3_fp8_train_tilelang_region_from_prim_funcs(
        mamba3_scan=mamba3_scan,
        m2rnn_packed_post=m2rnn_packed_post,
        fp8_prepare=fp8_prepare,
        fp8_prepare_spec=fp8_prepare_spec,
        sparse_mla_fp8_prepared=sparse_mla_fp8_prepared,
        schedule_template=schedule_template,
    )
    return compile_fusion_region(
        region,
        target=target,
        lowerer=lowerer,
        require_single_kernel=require_single_kernel,
    )


def mamba3_fp8_train_schedule_status_from_prim_funcs(
    *,
    mamba3_scan: Any,
    m2rnn_packed_post: Any,
    fp8_prepare: Any | None = None,
    fp8_prepare_spec: SparseMLAFp8PrepareSpec | None = None,
    sparse_mla_fp8_prepared: Any,
) -> FusionScheduleStatus:
    """Explain whether the current raw PrimFunc ABI can form the logical train-block edges."""

    fp8_prepare = _resolve_fp8_prepare_prim_func(fp8_prepare, fp8_prepare_spec)
    node_buffers = {
        "mamba3_scan": _prim_func_buffer_names(mamba3_scan),
        "m2rnn_packed_post": _prim_func_buffer_names(m2rnn_packed_post),
        "sparse_mla_fp8_apply": _prim_func_buffer_names(sparse_mla_fp8_prepared),
    }
    node_signatures = {
        "mamba3_scan": _prim_func_buffer_signatures(mamba3_scan),
        "m2rnn_packed_post": _prim_func_buffer_signatures(m2rnn_packed_post),
        "sparse_mla_fp8_apply": _prim_func_buffer_signatures(sparse_mla_fp8_prepared),
    }
    if fp8_prepare is not None:
        node_buffers["fp8_prepare"] = _prim_func_buffer_names(fp8_prepare)
        node_signatures["fp8_prepare"] = _prim_func_buffer_signatures(fp8_prepare)
    blocked_edges = []
    for producer, consumer, buffer, producer_aliases, consumer_aliases in (
        ("mamba3_scan", "m2rnn_packed_post", "scan_y", ("scan_y", "y"), ("scan_y", "conv_input")),
    ):
        producer_buffers = node_buffers[producer]
        consumer_buffers = node_buffers[consumer]
        if buffer not in producer_buffers or buffer not in consumer_buffers:
            blocked_edges.append(
                _abi_blocker_from_candidate_aliases(
                    producer=producer,
                    consumer=consumer,
                    buffer=buffer,
                    producer_buffers=producer_buffers,
                    consumer_buffers=consumer_buffers,
                    producer_signatures=node_signatures[producer],
                    consumer_signatures=node_signatures[consumer],
                    producer_aliases=producer_aliases,
                    consumer_aliases=consumer_aliases,
                )
            )
    fp8_prepare_ready = False
    if fp8_prepare is None:
        blocked_edges.append(
            FusionScheduleAbiBlocker(
                producer="m2rnn_packed_post",
                consumer="fp8_prepare",
                buffer="post_y",
                producer_buffers=tuple(sorted(node_buffers["m2rnn_packed_post"])),
                consumer_buffers=(),
                kind="fp8_prepare_tilelang_prim_func_missing",
                reason=(
                    "no raw TileLang PrimFunc currently projects post_y and emits "
                    "q_fp8/q_scale/kv_fp8/kv_scale inside the fused train block"
                ),
            )
        )
    else:
        required_prepare_buffers = {"post_y", "q_fp8", "q_scale", "kv_fp8", "kv_scale"}
        prepare_buffers = node_buffers["fp8_prepare"]
        fp8_prepare_ready = required_prepare_buffers.issubset(prepare_buffers)
        producer_buffers = node_buffers["m2rnn_packed_post"]
        if "post_y" not in producer_buffers or "post_y" not in prepare_buffers:
            blocked_edges.append(
                _abi_blocker_from_candidate_aliases(
                    producer="m2rnn_packed_post",
                    consumer="fp8_prepare",
                    buffer="post_y",
                    producer_buffers=producer_buffers,
                    consumer_buffers=prepare_buffers,
                    producer_signatures=node_signatures["m2rnn_packed_post"],
                    consumer_signatures=node_signatures["fp8_prepare"],
                    producer_aliases=("post_y", "post"),
                    consumer_aliases=("post_y",),
                )
            )
        if not fp8_prepare_ready:
            blocked_edges.append(
                FusionScheduleAbiBlocker(
                    producer="m2rnn_packed_post",
                    consumer="fp8_prepare",
                    buffer="post_y",
                    producer_buffers=tuple(sorted(node_buffers["m2rnn_packed_post"])),
                    consumer_buffers=tuple(sorted(prepare_buffers)),
                    kind="fp8_prepare_raw_abi_mismatch",
                    reason=(
                        "fp8_prepare PrimFunc must consume post_y and emit "
                        "q_fp8/q_scale/kv_fp8/kv_scale for the fused train block"
                    ),
                )
            )
    sparse_mla_buffers = node_buffers["sparse_mla_fp8_apply"]
    prepared_buffers = {"q_fp8", "q_scale", "kv_fp8", "kv_scale"}
    if fp8_prepare_ready:
        prepare_buffers = node_buffers["fp8_prepare"]
        for buffer in sorted(prepared_buffers):
            prepare_signature = node_signatures["fp8_prepare"].get(buffer)
            apply_signature = node_signatures["sparse_mla_fp8_apply"].get(buffer)
            if buffer not in prepare_buffers or buffer not in sparse_mla_buffers:
                blocked_edges.append(
                    FusionScheduleAbiBlocker(
                        producer="fp8_prepare",
                        consumer="sparse_mla_fp8_apply",
                        buffer=buffer,
                        producer_buffers=tuple(sorted(prepare_buffers)),
                        consumer_buffers=tuple(sorted(sparse_mla_buffers)),
                        producer_signature=_candidate_signature_summary(
                            node_signatures["fp8_prepare"],
                            (buffer,),
                        ),
                        consumer_signature=_candidate_signature_summary(
                            node_signatures["sparse_mla_fp8_apply"],
                            (buffer,),
                        ),
                    )
                )
            elif not _same_buffer_storage_signature(prepare_signature, apply_signature):
                blocked_edges.append(
                    FusionScheduleAbiBlocker(
                        producer="fp8_prepare",
                        consumer="sparse_mla_fp8_apply",
                        buffer=buffer,
                        producer_buffers=tuple(sorted(prepare_buffers)),
                        consumer_buffers=tuple(sorted(sparse_mla_buffers)),
                        producer_signature=prepare_signature.summary
                        if prepare_signature is not None
                        else "",
                        consumer_signature=apply_signature.summary
                        if apply_signature is not None
                        else "",
                        kind="raw_abi_signature_mismatch",
                        reason=(
                            "raw PrimFunc ABI exposes the logical buffer name on both "
                            "endpoints, but shape or dtype differs"
                        ),
                    )
                )
    apply_outputs = {"out", "lse"}
    if (
        not fp8_prepare_ready
        and prepared_buffers.issubset(sparse_mla_buffers)
        and apply_outputs.issubset(sparse_mla_buffers)
    ):
        blocked_edges.append(
            FusionScheduleAbiBlocker(
                producer="m2rnn_packed_post",
                consumer="sparse_mla_fp8_apply",
                buffer="fp8_prepare",
                producer_buffers=tuple(sorted(node_buffers["m2rnn_packed_post"])),
                consumer_buffers=tuple(sorted(sparse_mla_buffers)),
                kind="prepared_apply_consumer_not_prepare_producer",
                reason=(
                    "sparse_mla_fp8_prepared consumes already prepared "
                    "q_fp8/q_scale/kv_fp8/kv_scale buffers and produces out/lse; "
                    "it is not the FP8 prepare producer from post_y"
                ),
            )
        )

    if blocked_edges:
        has_prepare_blocker = any(
            edge.kind == "prepared_apply_consumer_not_prepare_producer"
            for edge in blocked_edges
        )
        return FusionScheduleStatus(
            status=(
                "blocked_fp8_prepare_producer_missing"
                if has_prepare_blocker
                else "blocked_raw_abi_mismatch"
            ),
            reason=(
                "the current raw Sparse-MLA FP8 surface is an apply consumer, not "
                "the producer that projects post_y and emits prepared FP8 buffers"
                if has_prepare_blocker
                else (
                    "raw PrimFunc surfaces do not expose the logical producer/consumer "
                    "buffers required for the mamba3+packed_post+fp8_prepared fused schedule"
                )
            ),
            single_kernel_fused=False,
            blocked_edges=tuple(blocked_edges),
        )
    return FusionScheduleStatus(
        status="ready_for_explicit_schedule",
        reason="raw PrimFunc ABI exposes all logical fusion edges",
        single_kernel_fused=False,
    )


def _resolve_fp8_prepare_prim_func(
    fp8_prepare: Any | None,
    fp8_prepare_spec: SparseMLAFp8PrepareSpec | None,
) -> Any | None:
    if fp8_prepare is not None and fp8_prepare_spec is not None:
        raise ValueError("pass either fp8_prepare or fp8_prepare_spec, not both")
    if fp8_prepare is not None:
        return fp8_prepare
    if fp8_prepare_spec is None:
        return None
    return fp8_prepare_spec.build_prim_func()


def compile_path_c_region(
    region: PathCFusionRegion,
    *,
    schedule_template: Callable[[Any], Any] | None = None,
    schedule_name: str | None = None,
    schedule_status: str = "ready",
    tilelang_lowerer: Callable[..., Any] | None = None,
    target: str = "metal",
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
    if tilelang_lowerer is not None and compiler is not None:
        raise ValueError("compiler and tilelang_lowerer are mutually exclusive")
    if tilelang_lowerer is None:
        tilelang_plan = _tilelang_compile_plan_for(
            region,
            schedule_template=schedule_template,
            schedule_name=schedule_name,
            schedule_status=schedule_status,
        )
        artifact = None
    else:
        tilelang_result = _tilelang_compile_result_for(
            region,
            schedule_template=schedule_template,
            schedule_name=schedule_name,
            schedule_status=schedule_status,
            target=target,
            lowerer=tilelang_lowerer,
        )
        tilelang_plan = tilelang_result.plan
        artifact = tilelang_result.artifact
    semantic_blockers = _semantic_blockers_for(region)
    schedule_contract = _schedule_contract_status_for(
        region,
        schedule_template=schedule_template,
        tilelang_lowerer_provided=tilelang_lowerer is not None,
        tilelang_schedule_status=tilelang_plan.schedule_status,
        semantic_blockers=semantic_blockers,
    )
    plan = FusionCompilePlan(
        region_name=region.name,
        lowering_boundary=tilelang_plan.lowering_boundary,
        backend=tilelang_plan.backend,
        compiler="tilelang.engine.fusion",
        fusion_groups=(
            FusionGroup(
                node_names=region.node_names,
                schedule_template=tilelang_plan.schedule_name,
            ),
        ),
        backward_graph=backward_graph,
        z3_sync=region.z3_sync,
        cache_key_parts=(
            *tuple(tilelang_plan.cache_key_material),
            *_region_shape_cache_key_parts(region),
        ),
        schedule_name=tilelang_plan.schedule_name,
        schedule_status=tilelang_plan.schedule_status,
        single_kernel_fused=(
            tilelang_lowerer is not None
            and bool(tilelang_plan.require_single_kernel)
            and tilelang_plan.schedule_status == "ready"
            and not semantic_blockers
            and schedule_contract.status == "verified"
            and bool(schedule_contract.declared_required_real_abi_inputs)
            and not schedule_contract.missing_real_abi_inputs
        ),
        autograd_mode=tilelang_plan.autograd_plan.mode,
        autograd_status=tilelang_plan.autograd_plan.status,
        autograd_backward_nodes=tuple(tilelang_plan.autograd_plan.backward_node_names),
        autograd_backward_edges=tuple(tilelang_plan.autograd_plan.backward_edges),
        autograd_missing_backward_nodes=tuple(
            tilelang_plan.autograd_plan.missing_backward_node_names
        ),
        semantic_blockers=semantic_blockers,
        schedule_contract=schedule_contract,
    )
    if tilelang_lowerer is not None:
        return CompiledPathCRegion(plan=plan, artifact=artifact)
    if compiler is None:
        return plan
    return CompiledPathCRegion(plan=plan, artifact=compiler(plan))


def tilelang_single_entry_lowerer(
    func_or_mod: Any,
    *,
    target: str = "metal",
    execution_backend: str = "tvm_ffi",
    **_kwargs: Any,
) -> Any:
    """Lower a single-entry fusion IRModule through ``tilelang.compile``."""

    prim_func = _single_entry_prim_func(func_or_mod)
    return _compile_tilelang_prim_func(
        prim_func,
        target=target,
        execution_backend=execution_backend,
    )


def _single_entry_prim_func(func_or_mod: Any) -> Any:
    capture = io.StringIO()
    with redirect_stdout(capture), redirect_stderr(capture):
        from tvm import tir

    if isinstance(func_or_mod, tir.PrimFunc):
        return func_or_mod

    functions = getattr(func_or_mod, "functions", None)
    if functions is None:
        raise TypeError(
            "tilelang_single_entry_lowerer expected a PrimFunc or single-entry IRModule"
        )
    funcs = list(functions.values())
    if len(funcs) != 1:
        raise ValueError(
            f"tilelang_single_entry_lowerer requires exactly one entry, got {len(funcs)}"
        )
    prim_func = funcs[0]
    if not isinstance(prim_func, tir.PrimFunc):
        raise TypeError(
            "tilelang_single_entry_lowerer expected the IRModule entry to be a PrimFunc"
        )
    return prim_func


def _compile_tilelang_prim_func(
    prim_func: Any,
    *,
    target: str,
    execution_backend: str,
) -> Any:
    import tilelang

    return tilelang.compile(
        prim_func,
        target=target,
        execution_backend=execution_backend,
    )


def _tilelang_compile_plan_for(
    region: PathCFusionRegion,
    *,
    schedule_template: Callable[[Any], Any] | None = None,
    schedule_name: str | None = None,
    schedule_status: str = "ready",
):
    optimizer = _tilelang_optimizer_for(
        region,
        schedule_template=schedule_template,
        schedule_name=schedule_name,
        schedule_status=schedule_status,
    )
    return optimizer.plan()


def _tilelang_compile_result_for(
    region: PathCFusionRegion,
    *,
    schedule_template: Callable[[Any], Any] | None,
    schedule_name: str | None,
    schedule_status: str,
    target: str,
    lowerer: Callable[..., Any],
):
    if schedule_template is None:
        raise ValueError("tilelang_lowerer requires an explicit fused schedule_template")
    optimizer = _tilelang_optimizer_for(
        region,
        schedule_template=schedule_template,
        schedule_name=schedule_name,
        schedule_status=schedule_status,
    )
    return optimizer.compile(target=target, lowerer=lowerer, require_single_kernel=True)


def _tilelang_optimizer_for(
    region: PathCFusionRegion,
    *,
    schedule_template: Callable[[Any], Any] | None,
    schedule_name: str | None,
    schedule_status: str,
):
    capture = io.StringIO()
    with redirect_stdout(capture), redirect_stderr(capture):
        from tilelang.engine.fusion import (
            FusionOptimizer,
            FusionScheduleRegistry,
        )

    op_signature = tuple(node.op_name for node in region.nodes)
    entry_name = schedule_name or f"{region.name}:producer_consumer_region"
    resolved_schedule = (
        _missing_tilelang_fused_schedule_template
        if schedule_template is None
        else schedule_template
    )
    resolved_status = (
        "missing_real_fused_schedule_template"
        if schedule_template is None
        else schedule_status
    )
    schedule_registry = FusionScheduleRegistry().register(
        op_signature,
        resolved_schedule,
        name=entry_name,
        status=resolved_status,
    )
    optimizer = FusionOptimizer(
        region.name,
        schedule_registry=schedule_registry,
        require_single_kernel=True,
        enable_z3_sync_async=region.z3_sync.enabled,
    )
    for node in region.nodes:
        optimizer.add_node(
            node.name,
            op=node.op_name,
            inputs=node.inputs,
            outputs=node.outputs,
            attrs=_tilelang_node_attrs(node),
        )
    return optimizer


def _tilelang_node_attrs(node: FusionNode) -> dict[str, str]:
    attrs = {"backward": node.backward}
    if node.name.endswith("_bwd"):
        attrs["autograd"] = "aot_backward"
        attrs["role"] = "backward"
    return attrs


def _region_shape_env_payload(region: PathCFusionRegion) -> dict[str, Any]:
    metadata = region.metadata if isinstance(region.metadata, Mapping) else {}
    shape_env = metadata.get("path_c_model_shape_env")
    if not isinstance(shape_env, PathCModelShapeEnv):
        return {}
    return asdict(shape_env)


def _region_shape_env_key(region: PathCFusionRegion) -> str:
    payload = _region_shape_env_payload(region)
    if not payload:
        return ""
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _region_shape_cache_key_parts(region: PathCFusionRegion) -> tuple[str, ...]:
    shape_key = _region_shape_env_key(region)
    if not shape_key:
        return ()
    return (f"shape_env:{shape_key}",)


def _schedule_contract_for(region: PathCFusionRegion) -> FusionScheduleContract:
    internal_buffers = tuple(
        dict.fromkeys(edge.input for edge in region.edges)
    )
    internal_buffer_set = set(internal_buffers)
    external_buffers: list[str] = []
    seen_external: set[str] = set()
    for node in region.nodes:
        for buffer_name in (*node.inputs, *node.outputs):
            if buffer_name in internal_buffer_set or buffer_name in seen_external:
                continue
            seen_external.add(buffer_name)
            external_buffers.append(buffer_name)
    op_signature = tuple(node.op_name for node in region.nodes)
    required_external_buffers = tuple(external_buffers)
    shape_env_payload = _region_shape_env_payload(region)
    shape_env_key = _region_shape_env_key(region)
    key_payload = {
        "region_name": region.name,
        "op_signature": op_signature,
        "required_internal_buffers": internal_buffers,
        "required_external_buffers": required_external_buffers,
        "shape_env": shape_env_payload,
    }
    return FusionScheduleContract(
        name=f"{region.name}:single_entry_contract",
        key=sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest(),
        op_signature=op_signature,
        required_internal_buffers=internal_buffers,
        required_external_buffers=required_external_buffers,
        shape_env_key=shape_env_key,
    )


def mark_path_c_schedule_template_for_region(
    schedule_template: Callable[[Any], Any],
    region: PathCFusionRegion,
    *,
    implementation_kind: str = "diagnostic",
    production_schedule_id: str = "",
    required_real_abi_inputs: Sequence[str] = (),
) -> Callable[[Any], Any]:
    """Return a schedule template wrapper attested for ``region``'s contract."""

    if not callable(schedule_template):
        raise TypeError("schedule_template must be callable")
    if not isinstance(region, PathCFusionRegion):
        raise TypeError("region must be PathCFusionRegion")
    if implementation_kind not in {"diagnostic", "prototype", "scaffold", "production"}:
        raise ValueError(
            "implementation_kind must be one of 'diagnostic', 'prototype', 'scaffold', or 'production'"
        )
    if implementation_kind == "production" and not production_schedule_id:
        raise ValueError("production schedule templates must declare a production_schedule_id")
    contract = _schedule_contract_for(region)

    def attested_schedule_template(template_region: Any) -> Any:
        if getattr(template_region, "metadata", None) is None and region.metadata:
            return schedule_template(region)
        return schedule_template(template_region)

    attested_schedule_template._cppmega_path_c_schedule_contract_key = contract.key
    attested_schedule_template._cppmega_path_c_schedule_contract_name = contract.name
    attested_schedule_template._cppmega_path_c_schedule_implementation_kind = implementation_kind
    attested_schedule_template._cppmega_path_c_production_schedule_id = production_schedule_id
    attested_schedule_template._cppmega_path_c_required_real_abi_inputs = tuple(
        required_real_abi_inputs
    )
    return attested_schedule_template


def _declared_schedule_contract_key(
    schedule_template: Callable[[Any], Any] | None,
) -> str:
    if schedule_template is None:
        return ""
    declared = getattr(
        schedule_template,
        "_cppmega_path_c_schedule_contract_key",
        "",
    )
    return str(declared) if declared is not None else ""


def _declared_schedule_implementation_kind(
    schedule_template: Callable[[Any], Any] | None,
) -> str:
    if schedule_template is None:
        return ""
    declared = getattr(
        schedule_template,
        "_cppmega_path_c_schedule_implementation_kind",
        "",
    )
    return str(declared) if declared is not None else ""


def _declared_production_schedule_id(
    schedule_template: Callable[[Any], Any] | None,
) -> str:
    if schedule_template is None:
        return ""
    declared = getattr(
        schedule_template,
        "_cppmega_path_c_production_schedule_id",
        "",
    )
    return str(declared) if declared is not None else ""


def _declared_required_real_abi_inputs(
    schedule_template: Callable[[Any], Any] | None,
) -> tuple[str, ...]:
    if schedule_template is None:
        return ()
    declared = getattr(
        schedule_template,
        "_cppmega_path_c_required_real_abi_inputs",
        (),
    )
    return tuple(str(name) for name in declared)


def _schedule_contract_status_for(
    region: PathCFusionRegion,
    *,
    schedule_template: Callable[[Any], Any] | None,
    tilelang_lowerer_provided: bool,
    tilelang_schedule_status: str,
    semantic_blockers: Sequence[FusionSemanticBlocker],
) -> FusionScheduleContractStatus:
    contract = _schedule_contract_for(region)
    declared_key = _declared_schedule_contract_key(schedule_template)
    declared_kind = _declared_schedule_implementation_kind(schedule_template)
    declared_schedule_id = _declared_production_schedule_id(schedule_template)
    declared_required_real_abi_inputs = _declared_required_real_abi_inputs(
        schedule_template
    )
    missing_real_abi_inputs = tuple(
        name
        for name in declared_required_real_abi_inputs
        if name not in contract.required_external_buffers
    )
    if semantic_blockers:
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="blocked_semantic_contract",
            reason="semantic blockers must be resolved before a schedule can satisfy the production fusion contract",
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    if schedule_template is None:
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="missing_schedule_template",
            reason="no fused TileLang/TIR schedule template was supplied for this region contract",
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    if declared_key != contract.key:
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="unattested_schedule_template",
            reason=(
                "schedule template was supplied but was not explicitly attested "
                "for this region contract"
            ),
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    if declared_kind != "production":
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="attested_non_production_schedule",
            reason=(
                "schedule template is attested for this contract but is not "
                "marked as a production implementation"
            ),
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    if declared_schedule_id not in _TRUSTED_PRODUCTION_SCHEDULE_IDS:
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="untrusted_production_schedule",
            reason=(
                "schedule template declares production implementation, but its "
                "production_schedule_id is not trusted by this build"
            ),
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    if missing_real_abi_inputs:
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="incomplete_real_abi_contract",
            reason=(
                "production schedule declares real model ABI inputs that are "
                "missing from this region contract"
            ),
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    if not tilelang_lowerer_provided:
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="registered_not_lowered",
            reason="schedule template is registered but not lowered, so internal buffer materialization is not verified",
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    if tilelang_schedule_status != "ready":
        return FusionScheduleContractStatus(
            name=contract.name,
            key=contract.key,
            status="schedule_not_ready",
            reason=f"TileLang schedule status is {tilelang_schedule_status!r}, not 'ready'",
            op_signature=contract.op_signature,
            required_internal_buffers=contract.required_internal_buffers,
            required_external_buffers=contract.required_external_buffers,
            shape_env_key=contract.shape_env_key,
            declared_key=declared_key,
            declared_implementation_kind=declared_kind,
            declared_schedule_id=declared_schedule_id,
            declared_required_real_abi_inputs=declared_required_real_abi_inputs,
            missing_real_abi_inputs=missing_real_abi_inputs,
        )
    return FusionScheduleContractStatus(
        name=contract.name,
        key=contract.key,
        status="verified",
        reason="TileLang lowered a single-entry region and verified internal edge materialization",
        op_signature=contract.op_signature,
        required_internal_buffers=contract.required_internal_buffers,
        required_external_buffers=contract.required_external_buffers,
        shape_env_key=contract.shape_env_key,
        declared_key=declared_key,
        declared_implementation_kind=declared_kind,
        declared_schedule_id=declared_schedule_id,
        declared_required_real_abi_inputs=declared_required_real_abi_inputs,
        missing_real_abi_inputs=missing_real_abi_inputs,
    )


def _semantic_blockers_for(region: PathCFusionRegion) -> tuple[FusionSemanticBlocker, ...]:
    nodes_by_name = {node.name: node for node in region.nodes}
    op_names = tuple(node.op_name for node in region.nodes)
    blockers: list[FusionSemanticBlocker] = []

    if (
        "mamba3_mimo" in op_names
        and "m2rnn" in op_names
        and not any(node.op_name == "residual_rmsnorm" for node in region.nodes)
    ):
        blockers.append(
            FusionSemanticBlocker(
                kind="residual_norm_bridge_missing",
                producer="mamba3_scan",
                consumer="m2rnn_packed_post",
                required_node="mamba3_residual_to_m2rnn_norm",
                reason=(
                    "HybridTinyBlock routes are pre-norm residual blocks: m2rnn "
                    "does not consume raw mamba3 scan output. A fused schedule "
                    "must materialize the residual add and next-layer RMSNorm "
                    "inside the generated region before the m2rnn producer."
                ),
            )
        )

    fp8_prepare = nodes_by_name.get("fp8_prepare")
    if fp8_prepare is not None and fp8_prepare.inputs == ("post_y",):
        blockers.append(
            FusionSemanticBlocker(
                kind="attention_qkv_projection_missing",
                producer="m2rnn_packed_post",
                consumer="fp8_prepare",
                required_node="attention_qkv_projection",
                reason=(
                    "Sparse-MLA FP8 prepared buffers are produced by "
                    "CausalSelfAttention.prepare_sparse_mla_fp8: attention "
                    "Q/KV projection, RoPE, and FP8 quantization from the "
                    "attention-normalized hidden state. They are not a direct "
                    "consumer of m2rnn post_y."
                ),
            )
        )

    return tuple(blockers)


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


def fused_path_c_plan_default_eligible(
    plan: FusionCompilePlan,
    row: BenchmarkAcceptanceRow,
    *,
    min_c_over_b: float = DEFAULT_MIN_C_OVER_B,
    max_peak_delta_gib: float = DEFAULT_MAX_PEAK_DELTA_GIB,
) -> bool:
    """Gate default Path C fusion on both the plan and clean benchmark row."""

    if not plan.single_kernel_fused:
        return False
    if plan.schedule_status != "ready":
        return False
    if plan.autograd_status != "ready" or plan.autograd_missing_backward_nodes:
        return False
    if plan.semantic_blockers:
        return False
    if plan.schedule_contract is None or plan.schedule_contract.status != "verified":
        return False
    if not plan.schedule_contract.declared_required_real_abi_inputs:
        return False
    if plan.schedule_contract.missing_real_abi_inputs:
        return False
    return fused_path_c_default_eligible(
        row,
        min_c_over_b=min_c_over_b,
        max_peak_delta_gib=max_peak_delta_gib,
    )


@dataclass(frozen=True)
class WarmCacheAudit:
    case_id: str
    status: str
    reason: str
    cache_hit: bool
    compile_cache_key: str


@dataclass(frozen=True)
class FusionCacheKeyAudit:
    case_id: str
    status: str
    reason: str
    expected_cache_key: str
    observed_cache_key: str
    cache_hit: bool


def audit_fusion_cache_key(
    *,
    case_id: str,
    expected_cache_key: str,
    observed_cache_key: str,
    cache_hit: bool,
) -> FusionCacheKeyAudit:
    """Classify cache-key stability for a Path C fusion compile attempt."""

    if expected_cache_key != observed_cache_key:
        return FusionCacheKeyAudit(
            case_id=case_id,
            status="key_changed",
            reason="fusion cache key changed between equivalent compile attempts",
            expected_cache_key=expected_cache_key,
            observed_cache_key=observed_cache_key,
            cache_hit=cache_hit,
        )
    if not cache_hit:
        return FusionCacheKeyAudit(
            case_id=case_id,
            status="recompiled_same_key",
            reason="fusion cache key was stable but the compile cache did not hit",
            expected_cache_key=expected_cache_key,
            observed_cache_key=observed_cache_key,
            cache_hit=cache_hit,
        )
    return FusionCacheKeyAudit(
        case_id=case_id,
        status="ok",
        reason="fusion cache key was stable and the compile cache hit",
        expected_cache_key=expected_cache_key,
        observed_cache_key=observed_cache_key,
        cache_hit=cache_hit,
    )


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
    ambiguous_outputs: dict[str, tuple[str, str]] = {}
    edges: list[FusionEdge] = []
    for node in nodes:
        for output_name in node.outputs:
            existing = producer_by_output.get(output_name)
            if existing is not None and existing != node.name:
                ambiguous_outputs[output_name] = (existing, node.name)
                continue
            producer_by_output[output_name] = node.name

    for node in nodes:
        for input_name in node.inputs:
            ambiguous = ambiguous_outputs.get(input_name)
            if ambiguous is not None:
                raise ValueError(
                    f"ambiguous fusion producer for buffer {input_name!r}: "
                    f"{ambiguous[0]!r} and {ambiguous[1]!r}"
                )
            producer = producer_by_output.get(input_name)
            if producer is not None and producer != node.name:
                edges.append(
                    FusionEdge(
                        producer=producer,
                        output=input_name,
                        consumer=node.name,
                        input=input_name,
                    )
                )
    return tuple(edges)


def _nodes_in_dependency_order(
    nodes: Sequence[FusionNode],
    edges: Sequence[FusionEdge],
) -> tuple[FusionNode, ...]:
    if len(nodes) < 2:
        return tuple(nodes)

    nodes_by_name = {node.name: node for node in nodes}
    original_index = {node.name: index for index, node in enumerate(nodes)}
    indegree = {node.name: 0 for node in nodes}
    successors: dict[str, set[str]] = {node.name: set() for node in nodes}
    for edge in edges:
        if edge.producer not in nodes_by_name or edge.consumer not in nodes_by_name:
            raise ValueError(
                f"fusion edge {edge.producer!r}->{edge.consumer!r}:{edge.input!r} "
                "references a missing node"
            )
        if edge.producer == edge.consumer:
            continue
        if edge.consumer not in successors[edge.producer]:
            successors[edge.producer].add(edge.consumer)
            indegree[edge.consumer] += 1

    ready = sorted(
        (name for name, count in indegree.items() if count == 0),
        key=original_index.__getitem__,
    )
    ordered_names: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered_names.append(name)
        for successor in sorted(successors[name], key=original_index.__getitem__):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort(key=original_index.__getitem__)

    if len(ordered_names) != len(nodes):
        raise ValueError("fusion region contains a cycle in producer/consumer dependencies")
    return tuple(nodes_by_name[name] for name in ordered_names)


def _edges_in_dependency_order(
    edges: Sequence[FusionEdge],
    nodes: Sequence[FusionNode],
) -> tuple[FusionEdge, ...]:
    node_order = {node.name: index for index, node in enumerate(nodes)}
    input_order = {
        (node.name, input_name): index
        for node in nodes
        for index, input_name in enumerate(node.inputs)
    }
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                node_order.get(edge.consumer, len(nodes)),
                input_order.get((edge.consumer, edge.input), len(nodes)),
                node_order.get(edge.producer, len(nodes)),
                edge.input,
            ),
        )
    )


def _prim_func_buffer_names(prim_func: Any) -> set[str]:
    buffer_map = getattr(prim_func, "buffer_map", None)
    if buffer_map is None:
        raise TypeError("expected a tvm.tir.PrimFunc-like object with buffer_map")
    return {str(buffer.name) for buffer in buffer_map.values()}


def _prim_func_buffer_signatures(
    prim_func: Any,
) -> dict[str, FusionBufferSignature]:
    buffer_map = getattr(prim_func, "buffer_map", None)
    if buffer_map is None:
        raise TypeError("expected a tvm.tir.PrimFunc-like object with buffer_map")
    return {
        str(buffer.name): FusionBufferSignature(
            name=str(buffer.name),
            shape=tuple(_tir_dim_to_str(dim) for dim in buffer.shape),
            dtype=str(buffer.dtype),
        )
        for buffer in buffer_map.values()
    }


def _candidate_signature_summary(
    signatures: Mapping[str, FusionBufferSignature],
    names: Sequence[str],
) -> str:
    return ";".join(
        signature.summary for name in names if (signature := signatures.get(name)) is not None
    )


def _first_candidate_signature(
    signatures: Mapping[str, FusionBufferSignature],
    names: Sequence[str],
) -> FusionBufferSignature | None:
    for name in names:
        signature = signatures.get(name)
        if signature is not None:
            return signature
    return None


def _abi_blocker_from_candidate_aliases(
    *,
    producer: str,
    consumer: str,
    buffer: str,
    producer_buffers: set[str],
    consumer_buffers: set[str],
    producer_signatures: Mapping[str, FusionBufferSignature],
    consumer_signatures: Mapping[str, FusionBufferSignature],
    producer_aliases: Sequence[str],
    consumer_aliases: Sequence[str],
) -> FusionScheduleAbiBlocker:
    producer_signature = _first_candidate_signature(producer_signatures, producer_aliases)
    consumer_signature = _first_candidate_signature(consumer_signatures, consumer_aliases)
    producer_summary = _candidate_signature_summary(producer_signatures, producer_aliases)
    consumer_summary = _candidate_signature_summary(consumer_signatures, consumer_aliases)
    if producer_signature is not None and consumer_signature is not None:
        if _same_buffer_storage_signature(producer_signature, consumer_signature):
            return FusionScheduleAbiBlocker(
                producer=producer,
                consumer=consumer,
                buffer=buffer,
                producer_buffers=tuple(sorted(producer_buffers)),
                consumer_buffers=tuple(sorted(consumer_buffers)),
                producer_signature=producer_summary,
                consumer_signature=consumer_summary,
                kind="raw_abi_name_mismatch",
                reason=(
                    "raw PrimFunc ABI exposes compatible candidate buffers but "
                    "not the logical fusion buffer name"
                ),
            )
        return FusionScheduleAbiBlocker(
            producer=producer,
            consumer=consumer,
            buffer=buffer,
            producer_buffers=tuple(sorted(producer_buffers)),
            consumer_buffers=tuple(sorted(consumer_buffers)),
            producer_signature=producer_summary,
            consumer_signature=consumer_summary,
            kind="raw_abi_signature_mismatch",
            reason=(
                "raw PrimFunc ABI exposes candidate producer/consumer buffers, "
                "but shape or dtype differs; this requires a real schedule-level "
                "layout transform, not a buffer rename"
            ),
        )
    return FusionScheduleAbiBlocker(
        producer=producer,
        consumer=consumer,
        buffer=buffer,
        producer_buffers=tuple(sorted(producer_buffers)),
        consumer_buffers=tuple(sorted(consumer_buffers)),
        producer_signature=producer_summary,
        consumer_signature=consumer_summary,
    )


def _same_buffer_storage_signature(
    lhs: FusionBufferSignature | None,
    rhs: FusionBufferSignature | None,
) -> bool:
    if lhs is None or rhs is None:
        return False
    return lhs.shape == rhs.shape and lhs.dtype == rhs.dtype


def _tir_dim_to_str(dim: Any) -> str:
    try:
        return str(int(dim))
    except (TypeError, ValueError):
        return str(dim)


def _require_names(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{label} must not be empty")
    return tuple(_require_identifier(value, label=label) for value in values)


def _require_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
