"""Descriptor-driven schedule planning for Path C fusion regions.

The named Mamba3 FP8 train-block target is an acceptance preset, not the source
of truth.  Schedule construction starts from a region graph, resolves each
known op through a brick descriptor, and then emits a single-entry TileLang
template for that chain.  The production schedule ID remains untrusted by
default until compile, profile, memory, and 1B matrix receipts prove it.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import keyword
import linecache
import re
from typing import Any, cast

from cppmega_mlx.runtime.path_c_fusion import (
    CompiledPathCRegion,
    FusionCompilePlan,
    FusionKernelSurface,
    FusionScheduleContractStatus,
    MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
    PathCFusionRegion,
    PathCModelShapeEnv,
    PathCSemanticGraphSideChannelBatch,
    Z3SyncSpec,
    build_path_c_aot_autograd_region,
    build_path_c_fusion_region,
    build_path_c_model_regions_from_model,
    build_mamba3_fp8_train_acceptance_fixture_region,
    compile_path_c_region,
    mark_path_c_schedule_template_for_region,
    tilelang_single_entry_lowerer,
    trusted_path_c_production_schedule_ids,
)


__all__ = [
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID",
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME",
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS",
    "MAMBA3_FP8_TRAIN_BUFFER_EXTENT",
    "MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME",
    "MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS",
    "CompiledMamba3Fp8TrainFusionSchedule",
    "Mamba3Fp8TrainFusionSchedulePlan",
    "Mamba3Fp8TrainFusionScheduleSpec",
    "DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN",
    "DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL",
    "DESCRIPTOR_LOOP_POLICY_FLAT",
    "DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN",
    "PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR",
    "PathCBrickScheduleDescriptor",
    "PathCBrickScheduleFragment",
    "PathCBrickScheduleDescriptorRegistry",
    "PathCFusionScheduleAcceptanceProfile",
    "PathCFusionScheduleChainPlan",
    "PathCFusionScheduleChainSegment",
    "PathCFusionScheduleOptimizer",
    "PathCFusionScheduleOptimizerPlan",
    "PathCFusionScheduleRegistry",
    "PathCFusionScheduleSpec",
    "PathCFusionScheduleTarget",
    "build_path_c_descriptor_prim_func",
    "default_path_c_brick_schedule_descriptor_registry",
    "default_path_c_fusion_schedule_registry",
    "mamba3_fp8_train_prototype_schedule_target",
    "mamba3_fp8_train_fusion_schedule_spec",
    "mamba3_fp8_train_fusion_schedule_template",
    "mamba3_fp8_train_fusion_schedule_target",
    "mamba3_fp8_train_prototype_schedule_template",
    "make_path_c_descriptor_schedule_template",
    "path_c_fusion_schedule_spec",
    "path_c_fusion_schedule_template",
    "compile_mamba3_fp8_train_fusion_schedule",
    "path_c_semantic_graph_schedule_inputs",
    "plan_path_c_direct_fusion_chain_for_region",
    "plan_path_c_direct_fusion_chains_for_model",
    "plan_path_c_fusion_schedule_for_region",
    "plan_path_c_fusion_schedules_for_model",
    "plan_mamba3_fp8_train_fusion_schedule",
    "prototype_path_c_fusion_schedule_registry",
    "select_path_c_fusion_schedule_target",
]


MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID = (
    "mamba3_m2rnn_attention_fp8_train_block_fwd_bwd_v1"
)
MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME = (
    "mamba3_m2rnn_attention_fp8_train_block:production_fwd_bwd_v1"
)
MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME = (
    "mamba3_m2rnn_attention_fp8_train_block:prototype_fwd_bwd"
)
MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS = "prototype"
MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS = "ready"
PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR = "dynamic_brick_descriptor_generator"
DESCRIPTOR_DEFAULT_BUFFER_EXTENT = 4
DESCRIPTOR_DEFAULT_THREADS = 256
DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL = "scalar_local"
DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN = "row_local_hidden"
DESCRIPTOR_LOOP_POLICY_FLAT = "flat"
DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN = "row_phased_hidden"
DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT = "direct_buffers"
DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_DTYPE = "banked_by_dtype"
DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE = "banked_by_role"
DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT = 31
DESCRIPTOR_SHARED_SCRATCH_SPILL_THRESHOLD_BYTES = 4 * 1024
DESCRIPTOR_SHARED_SCRATCH_BUDGET_BYTES = 24 * 1024
MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL = 64
_DESCRIPTOR_ROOT_READS_MARKER = "cppmega_path_c_root_reads"
_DESCRIPTOR_ROOT_WRITES_MARKER = "cppmega_path_c_root_writes"
_EXACT_ROW_PHASED_BACKWARD_OPS = frozenset(
    {
        "attention_qkv_projection_bwd",
        "sparse_mla_fp8_apply_bwd",
        "m2rnn_bwd",
        "mamba3_mimo_bwd",
    }
)
DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_BUFFERS = frozenset(
    {
        "attention_hidden",
        "attention_hidden_grad",
        "hidden_after_mamba3",
        "hidden_after_mamba3_grad",
        "m2rnn_hidden",
        "m2rnn_hidden_grad",
        "m2rnn_delta",
        "m2rnn_delta_grad",
        "mamba3_delta",
        "mamba3_delta_grad",
    }
)
DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH = 64
DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS = "grid_chunks"
DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS = "launcher_chunks"
DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE = DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS
DESCRIPTOR_ROW_CHUNK_INDEX_BUFFER = "path_c_row_chunk_index"
DESCRIPTOR_ROW_CHUNK_INDEX_PARAM = "path_c_row_chunk_index"
DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM = "path_c_row_subchunk_index"
DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH = 8
DESCRIPTOR_BACKWARD_GATE_PARAM = "path_c_run_backward"
_GENERIC_MODEL_REAL_ABI_INPUT_SUFFIXES = (
    "residual_norm_weight",
    # Block A: the per-brick entry RMSNorm weight emitted by the model-
    # level surface lowering for the first in-region brick.
    "entry_rmsnorm_weight",
)
_TRAIN_STEP_SCALAR_OUTPUT_ABI_NAMES = ("loss", "ntokens")
_TRAIN_STEP_SCALAR_OUTPUT_ABI_REASON = (
    "train-step scalar ABI slots are declared, but suffix loss codegen is not "
    "fused into the descriptor body yet"
)
_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_NAMES = (
    "target_ids",
    "target_mask",
    "final_norm_weight",
    "lm_head_weight",
)
_TRAIN_STEP_SUFFIX_LOSS_PARAMETER_GRAD_ABI_NAMES = (
    "final_norm_weight_grad",
    "lm_head_weight_grad",
)
_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_REASON = (
    "train-step suffix loss inputs are declared for fused loss codegen; "
    "target_mask is consumed for ntokens, while full loss codegen is pending"
)
_DTYPE_NBYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "uint16": 2,
    "int16": 2,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "float64": 8,
    "uint64": 8,
    "int64": 8,
}


def path_c_semantic_graph_schedule_inputs(
    graph: PathCSemanticGraphSideChannelBatch,
) -> dict[str, object]:
    """Return caller-owned semantic graph buffers in Path C ABI names."""

    inputs: dict[str, object] = {}
    if graph.call_edges is not None:
        inputs["path_c_semantic_call_edges"] = graph.call_edges
    if graph.call_edge_mask is not None:
        inputs["path_c_semantic_call_edge_mask"] = graph.call_edge_mask
    if graph.type_edges is not None:
        inputs["path_c_semantic_type_edges"] = graph.type_edges
    if graph.type_edge_mask is not None:
        inputs["path_c_semantic_type_edge_mask"] = graph.type_edge_mask
    return inputs


def _mamba3_fp8_train_buffer_extent() -> int:
    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile

    return int(local_gb10_quarter_profile().max_seq_length)


MAMBA3_FP8_TRAIN_BUFFER_EXTENT = _mamba3_fp8_train_buffer_extent()

_PRODUCTION_SCHEDULE_REASON = (
    "dynamic brick descriptors can construct a single-entry TileLang/TIR "
    "schedule for the model-semantic mamba3 + residual/RMSNorm + m2rnn + "
    "attention_qkv_projection + sparse_mla_fp8_apply fwd/bwd region, but the "
    "schedule remains untrusted by default until compile, 1B matrix, profiling, "
    "memory, and cache receipts pass"
)
_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE = (
    # Block A: the fused region applies an eager pre-block RMSNorm to
    # ``hidden_entry`` before the first brick consumes it. The bwd is
    # auto-fissioned in reverse order and runs last.
    "entry_rmsnorm",
    "mamba3_mimo",
    "residual_rmsnorm",
    "m2rnn",
    "residual_rmsnorm",
    "attention_qkv_projection",
    "sparse_mla_fp8_apply",
    "sparse_mla_fp8_apply_bwd",
    "attention_qkv_projection_bwd",
    "residual_rmsnorm_bwd",
    "m2rnn_bwd",
    "residual_rmsnorm_bwd",
    "mamba3_mimo_bwd",
    "entry_rmsnorm_bwd",
)


@dataclass(frozen=True)
class PathCBrickScheduleDescriptor:
    """Schedule metadata for one reusable Path C model brick."""

    op_name: str
    implementation_status: str
    required_codegen_steps: tuple[str, ...]
    schedule_family: str = "loop_descriptor_dataflow"
    supports_backward: bool = True
    description: str = ""
    production_source: str = ""
    production_fragment_status: str = "not_inlined"
    production_fragment_reason: str = ""
    preferred_internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL
    preferred_loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT
    production_fragment_policy: str = ""
    production_fragment_codegen_step: str = ""
    production_fragment_inlined_reason: str = ""
    max_production_hidden_size: int | None = None
    max_production_op_occurrences: int | None = None
    max_production_op_occurrences_min_hidden_size: int | None = None
    fragment_emitter: Callable[..., Any] | None = None


@dataclass(frozen=True)
class PathCFusionScheduleSpec:
    """Schedule contract selected from a concrete Path C model region."""

    schedule_id: str
    schedule_name: str
    region_name: str
    implementation_kind: str
    implementation_status: str
    missing_reason: str
    trusted_by_default: bool
    contract_name: str
    contract_key: str
    shape_env_key: str
    op_signature: tuple[str, ...]
    required_internal_buffers: tuple[str, ...]
    required_external_buffers: tuple[str, ...]
    required_real_abi_inputs: tuple[str, ...]
    required_real_abi_input_shapes: tuple[str, ...]
    missing_real_abi_inputs: tuple[str, ...]
    real_abi_contract_complete: bool
    required_codegen_steps: tuple[str, ...]
    schedule_generator: str
    schedule_generator_status: str
    internal_buffer_policy: str
    loop_policy: str
    buffer_extent: int
    loop_extent: int
    brick_ops: tuple[str, ...]
    brick_schedule_families: tuple[str, ...]
    brick_descriptor_statuses: tuple[str, ...]
    brick_production_fragment_statuses: tuple[str, ...]
    brick_production_fragment_reasons: tuple[str, ...]
    brick_production_fragment_blockers: tuple[str, ...]
    production_fragments_complete: bool


@dataclass(frozen=True)
class Mamba3Fp8TrainFusionScheduleSpec(PathCFusionScheduleSpec):
    """Named acceptance schedule target for the 1B Path C train block."""


@dataclass(frozen=True)
class PathCFusionScheduleTarget:
    """Registry entry for a high-level Path C fused schedule pattern."""

    schedule_id: str
    schedule_name: str
    op_signature: tuple[str, ...]
    schedule_status: str
    implementation_kind: str
    missing_reason: str
    required_codegen_steps: tuple[str, ...]
    schedule_template: Callable[[Any], Any]
    required_real_abi_inputs: tuple[str, ...] = ()
    brick_descriptors: tuple[PathCBrickScheduleDescriptor, ...] = ()
    schedule_generator: str = PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT
    physical_abi_policy: str = DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT
    max_rows_per_launch: int | None = None
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE


@dataclass(frozen=True)
class PathCFusionScheduleAcceptanceProfile:
    """Metadata applied to a descriptor-built target selected from a live region."""

    op_signature: tuple[str, ...]
    schedule_id: str
    schedule_name: str
    schedule_status: str
    implementation_kind: str
    missing_reason: str
    required_codegen_steps: tuple[str, ...]
    entry_symbol: str | None = None
    required_real_abi_inputs: tuple[str, ...] = ()
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT
    required_region_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PathCFusionScheduleOptimizerPlan:
    """Generic high-level Path C optimization plan."""

    region: PathCFusionRegion
    plan: FusionCompilePlan
    schedule_target: PathCFusionScheduleTarget | None


@dataclass(frozen=True)
class PathCFusionScheduleChainSegment:
    """One contiguous fused segment in a generic Path C schedule chain."""

    index: int
    node_start: int
    node_end: int
    region: PathCFusionRegion
    plan: FusionCompilePlan | None
    schedule_target: PathCFusionScheduleTarget | None
    kernel_parameter_count: int | None
    physical_abi_policy: str
    status: str
    reason: str
    execution_phase: str


@dataclass(frozen=True)
class PathCFusionScheduleChainPlan:
    """Generic direct-buffer chain plan for a Path C region."""

    source_region: PathCFusionRegion
    max_kernel_buffers: int
    segments: tuple[PathCFusionScheduleChainSegment, ...]
    status: str
    reason: str


@dataclass(frozen=True)
class Mamba3Fp8TrainFusionSchedulePlan:
    """High-level planner output for the Mamba3 FP8 train-block target."""

    region: PathCFusionRegion
    plan: FusionCompilePlan
    schedule_spec: Mamba3Fp8TrainFusionScheduleSpec


@dataclass(frozen=True)
class CompiledMamba3Fp8TrainFusionSchedule:
    """Lowered named Mamba3 FP8 train-block schedule with its contract."""

    region: PathCFusionRegion
    compiled: CompiledPathCRegion
    schedule_spec: Mamba3Fp8TrainFusionScheduleSpec


@dataclass(frozen=True)
class _ScheduleNodeView:
    name: str
    op_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    backward: str = ""


def _path_c_schedule_node_execution_phase(node: Any) -> str:
    backward = str(getattr(node, "backward", ""))
    op_name = str(getattr(node, "op_name", ""))
    if backward == "owner_output" or op_name.endswith("_bwd"):
        return "backward"
    return "forward"


def _path_c_schedule_segment_execution_phase(nodes: Iterable[Any]) -> str:
    phases = {_path_c_schedule_node_execution_phase(node) for node in nodes}
    if not phases:
        return "empty"
    if len(phases) == 1:
        return next(iter(phases))
    return "mixed"


@dataclass(frozen=True)
class PathCBrickScheduleFragment:
    allocations: tuple[str, ...]
    statements: tuple[str, ...]


_ScheduleNodeFragment = PathCBrickScheduleFragment


@dataclass(frozen=True)
class _PhysicalAbiPlan:
    param_lines: tuple[str, ...]
    external_access_by_buffer: Mapping[str, str]
    physical_buffer_shapes: Mapping[str, tuple[int, ...]]
    logical_to_physical: Mapping[str, Mapping[str, Any]]


class PathCBrickScheduleDescriptorRegistry:
    """Registry mapping reusable brick op names to schedule descriptors."""

    def __init__(
        self,
        descriptors: Sequence[PathCBrickScheduleDescriptor] = (),
    ) -> None:
        self._descriptors: dict[str, PathCBrickScheduleDescriptor] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(
        self,
        descriptor: PathCBrickScheduleDescriptor,
    ) -> "PathCBrickScheduleDescriptorRegistry":
        if not isinstance(descriptor, PathCBrickScheduleDescriptor):
            raise TypeError("descriptor must be PathCBrickScheduleDescriptor")
        if not descriptor.op_name:
            raise ValueError("Path C brick descriptor op_name must not be empty")
        self._descriptors[descriptor.op_name] = descriptor
        return self

    def descriptor_for(self, op_name: str) -> PathCBrickScheduleDescriptor | None:
        descriptor = self._descriptors.get(op_name)
        if descriptor is not None:
            return descriptor
        if not op_name.endswith("_bwd"):
            return None
        base_op_name = op_name[: -len("_bwd")]
        base = self._descriptors.get(base_op_name)
        if base is None or not base.supports_backward:
            return None
        return PathCBrickScheduleDescriptor(
            op_name=op_name,
            implementation_status=f"{base.implementation_status}:aot_backward",
            required_codegen_steps=(f"{base_op_name}_aot_backward_descriptor",),
            schedule_family=base.schedule_family,
            supports_backward=False,
            description=f"AOT backward descriptor for {base_op_name}",
            production_source=base.production_source,
            production_fragment_status="region_fragment_inlined_unoptimized",
            production_fragment_reason=(
                "synthesized AOT backward descriptors use the generic owner-output "
                "gradient fragment; register an explicit backward descriptor before "
                "this can be treated as production-inlined"
            ),
            fragment_emitter=None,
        )

    def descriptors_for_signature(
        self,
        op_signature: Sequence[str],
    ) -> tuple[PathCBrickScheduleDescriptor, ...] | None:
        descriptors: list[PathCBrickScheduleDescriptor] = []
        for op_name in op_signature:
            descriptor = self.descriptor_for(str(op_name))
            if descriptor is None:
                return None
            descriptors.append(descriptor)
        return tuple(descriptors)


def default_path_c_brick_schedule_descriptor_registry() -> (
    PathCBrickScheduleDescriptorRegistry
):
    """Return descriptors for model bricks that can participate in Path C chains."""

    return PathCBrickScheduleDescriptorRegistry(
        (
            PathCBrickScheduleDescriptor(
                op_name="entry_rmsnorm",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "entry_rmsnorm_descriptor",
                    "entry_rmsnorm_pre_block_internal_buffers",
                ),
                description=(
                    "Per-brick entry RMSNorm descriptor applied to the "
                    "first in-region brick's hidden input"
                ),
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_emit_entry_rmsnorm_source"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "entry RMSNorm is emitted into the fused TileLang region "
                    "with an explicit norm-weight ABI input, but it is still "
                    "scalar descriptor code rather than the production vector "
                    "RMSNorm schedule"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "entry_rmsnorm_row_phased_production_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits the per-brick entry "
                    "RMSNorm full-row sum-of-squares reduction, inverse RMS, "
                    "and weighted normalized output without full activation "
                    "staging"
                ),
                fragment_emitter=_emit_entry_rmsnorm_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="entry_rmsnorm_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "entry_rmsnorm_bwd_recompute_descriptor",
                ),
                supports_backward=False,
                description=(
                    "Per-brick entry RMSNorm recompute backward descriptor"
                ),
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_append_row_phased_entry_rmsnorm_bwd_body"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "entry RMSNorm backward has an explicit descriptor; "
                    "row-phased descriptor codegen now recomputes full-hidden "
                    "forward state and accumulates norm-weight gradients, but "
                    "it is still tracked as a policy-gated production fragment "
                    "until row-local hidden scheduling is selected"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "entry_rmsnorm_bwd_row_phased_recompute_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes entry RMSNorm "
                    "state from forward inputs and accumulates norm-weight "
                    "grads without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_mimo",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_scan_descriptor",
                    "mamba3_scan_fwd_internal_buffers",
                    "mamba3_row_phased_dense_in_projection",
                    "mamba3_row_phased_causal_conv_ring_history",
                    "mamba3_row_phased_bc_norm_rope",
                    "mamba3_row_phased_state_recurrence",
                    "mamba3_row_phased_gate_out_projection",
                ),
                description="Mamba3 scan brick descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_path_c:"
                    "mamba3_mimo_fwd_path_c/mamba3_mimo_bwd_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits a Mamba3 descriptor "
                    "fragment; it only becomes production-inlined when the "
                    "row-local hidden policy is selected so the schedule can "
                    "carry scan state without staging full projected x/B/C/z/A/dt "
                    "activation tensors"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "mamba3_row_phased_fused_project_conv_scan_out"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen fuses Mamba3 dense input "
                    "projection, causal depthwise convolution, B/C norm+RoPE, "
                    "scan-state recurrence, gate, and output projection from "
                    "the block-level ABI without full activation staging"
                ),
                fragment_emitter=_emit_mamba3_mimo_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="mamba3_mimo_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_mimo_bwd_descriptor",
                    "mamba3_mimo_bwd_final_gradient_owner_outputs",
                ),
                supports_backward=False,
                description="Mamba3 MIMO backward descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_path_c:"
                    "mamba3_mimo_bwd_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "Mamba3 backward now has an explicit descriptor tied to the "
                    "Path C backward source and emits stage-specific "
                    "project/conv/dt/state/out gradient owner outputs, but it "
                    "is still scalar descriptor code; the existing Path C "
                    "backward kernel consumes scan-level dy/x/B/C/z/A/dt/D/h0 "
                    "tensors rather than block-level model weights"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "mamba3_mimo_bwd_row_phased_weight_state_recompute"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes Mamba3 backward "
                    "owner outputs from block-level weights, state, h0, and "
                    "row-local hidden gradients without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="residual_rmsnorm",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "residual_rmsnorm_descriptor",
                    "residual_rmsnorm_bridge_internal_buffers",
                ),
                description="Residual bridge plus RMSNorm descriptor",
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_emit_residual_rmsnorm_source"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "residual/RMSNorm bridge is now emitted into the fused "
                    "TileLang region with an explicit norm-weight ABI input, "
                    "but it is still scalar descriptor code rather than the "
                    "production vector RMSNorm schedule"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "residual_rmsnorm_row_phased_production_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits the residual bridge, "
                    "full-row sum-of-squares reduction, inverse RMS, and "
                    "weighted normalized output without full activation staging"
                ),
                fragment_emitter=_emit_residual_rmsnorm_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="residual_rmsnorm_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "residual_rmsnorm_bwd_recompute_descriptor",
                ),
                supports_backward=False,
                description="Residual bridge plus RMSNorm recompute backward descriptor",
                production_source=(
                    "cppmega_mlx.runtime.path_c_fusion_schedules:"
                    "_append_row_phased_residual_rmsnorm_bwd_body"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "residual/RMSNorm backward has an explicit descriptor; "
                    "row-phased descriptor codegen now recomputes full-hidden "
                    "forward state and accumulates norm-weight gradients, but "
                    "it is still tracked as a policy-gated production fragment "
                    "until row-local hidden scheduling is selected"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "residual_rmsnorm_bwd_row_phased_recompute_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes residual/RMSNorm "
                    "state from forward inputs and accumulates norm-weight grads "
                    "without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="m2rnn",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "m2rnn_descriptor",
                    "m2rnn_packed_post_internal_buffers",
                    "m2rnn_row_phased_dense_in_projection",
                    "m2rnn_row_phased_causal_conv_ring_history",
                    "m2rnn_row_phased_state_recurrence",
                    "m2rnn_row_phased_gate_norm_out_projection",
                ),
                description="M2RNN packed post descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.m2rnn_path_c:"
                    "m2rnn_apply_mapped_packed_post_with_state_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits an M2RNN descriptor "
                    "fragment; it only becomes production-inlined when the "
                    "row-local hidden policy is selected so the schedule can "
                    "carry recurrent state without staging full projected, "
                    "conv_input, or post tensors"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "m2rnn_row_phased_fused_project_conv_recurrence_post"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen fuses M2RNN dense input "
                    "projection, causal depthwise convolution, mapped state "
                    "recurrence, gate/RMSNorm, and output projection from the "
                    "block-level ABI without full activation staging"
                ),
                fragment_emitter=_emit_m2rnn_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="m2rnn_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "m2rnn_bwd_descriptor",
                    "m2rnn_bwd_final_gradient_owner_outputs",
                ),
                supports_backward=False,
                description="M2RNN packed backward descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.m2rnn_path_c:"
                    "m2rnn_mapped_packed_bwd_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "M2RNN backward now has an explicit descriptor tied to the "
                    "Path C packed backward source and emits stage-specific "
                    "project/conv/recurrent/post gradient owner outputs, but "
                    "it is still scalar descriptor code; the existing Path C "
                    "backward kernels consume mapped-packed recurrent/post "
                    "intermediates rather than block-level model weights"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "m2rnn_bwd_row_phased_weight_state_recompute"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen recomputes M2RNN backward "
                    "owner outputs from block-level projection, convolution, "
                    "state, gate, post, h0, and row-local hidden gradients "
                    "without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="attention_qkv_projection",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "attention_qkv_projection_descriptor",
                    "attention_qkv_projection_fp8_prepare_uint8",
                ),
                description="Attention Q/KV projection and FP8 prepare descriptor",
                production_source=(
                    "cppmega_mlx.nn.attention:"
                    "CausalSelfAttention.prepare_sparse_mla_fp8"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits an attention projection "
                    "descriptor fragment with the real ABI, but it remains "
                    "policy-gated until row-phased hidden scheduling is selected"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "attention_qkv_projection_row_phased_rope_fp8_fragment"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits real q/sparse-kv "
                    "dot-products, split-half RoPE, per-head FP8 scaling, "
                    "uint8 FP8 storage, and full-window causal sparse indices "
                    "without full activation staging"
                ),
                fragment_emitter=_emit_attention_qkv_projection_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="attention_qkv_projection_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "attention_qkv_projection_bwd_descriptor",
                    "attention_qkv_projection_bwd_weight_bias_gradients",
                ),
                supports_backward=False,
                description="Attention Q/KV projection backward descriptor",
                production_source=(
                    "cppmega_mlx.nn.attention:"
                    "CausalSelfAttention.prepare_sparse_mla_fp8"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "attention Q/KV projection backward now has an explicit "
                    "descriptor and emits stage-specific q/kv/RoPE gradient "
                    "owner outputs, but it is still scalar descriptor code "
                    "rather than the production q/kv weight, bias, RoPE, and "
                    "FP8-prepare backward schedule"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "attention_qkv_projection_bwd_row_phased_weight_bias_rope"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits attention Q/KV "
                    "projection backward weight, bias, hidden, and RoPE owner "
                    "gradients from block-level ABI and row-local prepared-FP8 "
                    "gradients without full activation staging"
                ),
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="sparse_mla_fp8_apply",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "sparse_mla_fp8_apply_descriptor",
                    "sparse_mla_fp8_apply_owner_output",
                    "sparse_mla_fp8_apply_softmax_lse_out_proj",
                    "sparse_mla_fp8_apply_lse_reuses_softmax_stats",
                    "sparse_mla_fp8_apply_row_topk_indices_cache",
                    "sparse_mla_fp8_apply_invalid_index_sentinel",
                ),
                supports_backward=False,
                description="Sparse MLA FP8 apply descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c:"
                    "make_fp8_sparse_mla_prepare_kernel/"
                    "sparse_mla_fp8_path_c_apply"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits a Sparse-MLA FP8 apply "
                    "descriptor fragment; it becomes production-inlined when "
                    "row-local hidden scheduling is selected so q/kv FP8 "
                    "prepare, sparse top-k scores, softmax stats, LSE, and "
                    "attention out-projection stay in the train-block region"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "sparse_mla_fp8_apply_row_phased_prepared_apply"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits prepared-FP8 sparse "
                    "attention apply with score max/sumexp, row-local cached "
                    "top-k indices, weighted KV values, invalid-index score "
                    "sentinels, attention out-projection, and LSE from the same "
                    "softmax stats without full activation staging"
                ),
                fragment_emitter=_emit_sparse_mla_fp8_apply_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="sparse_mla_fp8_apply_bwd",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "sparse_mla_fp8_apply_bwd_descriptor",
                    "sparse_mla_fp8_apply_bwd_prepared_grad_owner_outputs",
                    "sparse_mla_fp8_apply_bwd_out_proj_grad_owner_outputs",
                ),
                supports_backward=False,
                description="Sparse MLA FP8 apply backward descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c:"
                    "sparse_mla_fp8_path_c_apply VJP"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "Sparse-MLA FP8 apply backward now has an explicit "
                    "descriptor and emits prepared q/kv FP8, scale, and "
                    "attention out-projection owner gradients inside the "
                    "train-block graph, but it is still scalar descriptor code "
                    "rather than the production softmax/out-projection VJP"
                ),
                preferred_internal_buffer_policy=(
                    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
                ),
                preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
                production_fragment_codegen_step=(
                    "sparse_mla_fp8_apply_bwd_row_phased_prepared_grad"
                ),
                production_fragment_inlined_reason=(
                    "row-phased descriptor codegen emits Sparse-MLA apply "
                    "backward owner outputs for prepared q/kv FP8 values, "
                    "prepared scales, and attention out-projection gradients "
                    "without exposing q/kv prepared gradients as external ABI"
                ),
                fragment_emitter=None,
            ),
        )
    )


def make_path_c_descriptor_schedule_template(
    brick_descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    entry_symbol: str | None = None,
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT,
    shape_env: PathCModelShapeEnv | None = None,
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT,
    physical_abi_policy: str = DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
    train_step_output_abi: bool = False,
    max_rows_per_launch: int | None = None,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
) -> Callable[[Any], Any]:
    """Return a schedule template generated from brick descriptors."""

    descriptors = tuple(brick_descriptors)
    if not descriptors:
        raise ValueError("descriptor schedule template requires at least one brick")
    extent = _validated_buffer_extent(buffer_extent)
    validated_internal_buffer_policy = _validated_internal_buffer_policy(
        internal_buffer_policy
    )
    validated_loop_policy = _validated_loop_policy(loop_policy)
    validated_physical_abi_policy = _validated_physical_abi_policy(
        physical_abi_policy
    )
    validated_max_rows_per_launch = _validated_max_rows_per_launch(
        max_rows_per_launch
    )
    validated_row_dispatch_mode = _validated_row_dispatch_mode(row_dispatch_mode)

    def descriptor_schedule_template(template_region: Any) -> Any:
        return build_path_c_descriptor_prim_func(
            template_region,
            descriptors,
            entry_symbol=entry_symbol,
            buffer_extent=extent,
            shape_env=shape_env,
            internal_buffer_policy=validated_internal_buffer_policy,
            loop_policy=validated_loop_policy,
            physical_abi_policy=validated_physical_abi_policy,
            train_step_output_abi=bool(train_step_output_abi),
            max_rows_per_launch=validated_max_rows_per_launch,
            row_dispatch_mode=validated_row_dispatch_mode,
        )

    descriptor_schedule_metadata = cast(Any, descriptor_schedule_template)
    descriptor_schedule_metadata._cppmega_path_c_schedule_generator = (
        PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    )
    descriptor_schedule_metadata._cppmega_path_c_brick_ops = tuple(
        descriptor.op_name for descriptor in descriptors
    )
    descriptor_schedule_metadata._cppmega_path_c_buffer_extent = extent
    descriptor_schedule_metadata._cppmega_path_c_shape_env = shape_env
    descriptor_schedule_metadata._cppmega_path_c_internal_buffer_policy = (
        validated_internal_buffer_policy
    )
    descriptor_schedule_metadata._cppmega_path_c_loop_policy = validated_loop_policy
    descriptor_schedule_metadata._cppmega_path_c_physical_abi_policy = (
        validated_physical_abi_policy
    )
    descriptor_schedule_metadata._cppmega_path_c_max_rows_per_launch = (
        validated_max_rows_per_launch
    )
    descriptor_schedule_metadata._cppmega_path_c_row_dispatch_mode = (
        validated_row_dispatch_mode
    )
    descriptor_schedule_metadata._cppmega_path_c_workspace_edge_buffers = (
        ("q_fp8", "q_scale", "kv_fp8", "kv_scale")
        if _descriptor_chain_uses_kv_history_workspace(descriptors)
        else ()
    )
    descriptor_schedule_metadata._cppmega_path_c_train_step_output_abi_enabled = (
        bool(train_step_output_abi)
    )
    return descriptor_schedule_template


def mamba3_fp8_train_fusion_schedule_template(region: Any) -> Any:
    """Generate the explicit Mamba3 acceptance schedule for ``region``."""

    resolved_region = _require_path_c_region_graph(
        region,
        function_name="mamba3_fp8_train_fusion_schedule_template",
    )
    target = _profiled_descriptor_target_for_region(
        resolved_region,
        _mamba3_fp8_train_acceptance_profile(),
    )
    return target.schedule_template(resolved_region)


def mamba3_fp8_train_prototype_schedule_template(region: Any) -> Any:
    """Generate the explicit prototype Mamba3 schedule for ``region``."""

    resolved_region = _require_path_c_region_graph(
        region,
        function_name="mamba3_fp8_train_prototype_schedule_template",
    )
    target = _profiled_descriptor_target_for_region(
        resolved_region,
        _mamba3_fp8_train_prototype_profile(),
    )
    return target.schedule_template(resolved_region)


cast(
    Any,
    mamba3_fp8_train_fusion_schedule_template,
)._cppmega_path_c_workspace_edge_buffers = (
    "q_fp8",
    "q_scale",
    "kv_fp8",
    "kv_scale",
)
cast(
    Any,
    mamba3_fp8_train_prototype_schedule_template,
)._cppmega_path_c_workspace_edge_buffers = (
    "q_fp8",
    "q_scale",
    "kv_fp8",
    "kv_scale",
)


def build_path_c_descriptor_prim_func(
    region: Any,
    brick_descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    entry_symbol: str | None = None,
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT,
    shape_env: PathCModelShapeEnv | None = None,
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT,
    physical_abi_policy: str = DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
    train_step_output_abi: bool = False,
    max_rows_per_launch: int | None = None,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
) -> Any:
    """Generate a single-entry TileLang PrimFunc from a descriptor chain."""

    nodes = _node_views_for_region(region)
    descriptors = tuple(brick_descriptors)
    if len(descriptors) != len(nodes):
        raise ValueError(
            "descriptor count must match region node count: "
            f"{len(descriptors)} descriptors for {len(nodes)} nodes"
        )
    _validate_descriptors_match_nodes(nodes, descriptors)
    resolved_shape_env = shape_env or _shape_env_for_region(region)
    validated_internal_buffer_policy = _validated_internal_buffer_policy(
        internal_buffer_policy
    )
    validated_loop_policy = _validated_loop_policy(loop_policy)
    validated_physical_abi_policy = _validated_physical_abi_policy(
        physical_abi_policy
    )
    validated_max_rows_per_launch = _validated_max_rows_per_launch(
        max_rows_per_launch
    )
    validated_row_dispatch_mode = _validated_row_dispatch_mode(row_dispatch_mode)
    descriptors = _descriptors_with_policy_fragment_statuses(
        descriptors,
        internal_buffer_policy=validated_internal_buffer_policy,
        loop_policy=validated_loop_policy,
        shape_env=resolved_shape_env,
    )
    row_chunk_count = _row_phased_chunk_count(
        loop_policy=validated_loop_policy,
        shape_env=resolved_shape_env,
        max_rows_per_launch=validated_max_rows_per_launch,
    )
    row_phased_launcher_carry_buffers = _row_phased_launcher_carry_buffers_for_nodes(
        nodes,
        shape_env=resolved_shape_env,
        loop_policy=validated_loop_policy,
        max_rows_per_launch=validated_max_rows_per_launch,
        row_dispatch_mode=validated_row_dispatch_mode,
    )
    internal_buffers = _internal_buffers_for_nodes(
        nodes,
        shape_env=resolved_shape_env,
        internal_buffer_policy=validated_internal_buffer_policy,
        loop_policy=validated_loop_policy,
    )
    dtype_by_buffer = {
        name: _buffer_dtype(name, shape_env=resolved_shape_env)
        for node in nodes
        for name in (*node.inputs, *node.outputs)
    }
    external_buffers = _external_buffers_for_nodes(nodes, internal_buffers)
    train_step_loss_source_buffers = _train_step_suffix_loss_source_buffers(
        nodes,
    )
    train_step_computed_output_buffers = _train_step_computed_output_buffers(
        declared=bool(train_step_output_abi),
        shape_env=resolved_shape_env,
        loss_source_buffers=train_step_loss_source_buffers,
    )
    train_step_output_abi_payload = _train_step_output_abi_payload(
        declared=bool(train_step_output_abi),
        computed_logical_outputs=train_step_computed_output_buffers,
    )
    train_step_suffix_loss_input_abi_payload = (
        _train_step_suffix_loss_input_abi_payload(
            declared=bool(train_step_output_abi),
        )
    )
    train_step_suffix_loss_input_buffers = tuple(
        train_step_suffix_loss_input_abi_payload["logical_inputs"]
        if train_step_suffix_loss_input_abi_payload["declared"]
        else ()
    )
    train_step_output_buffers = tuple(
        train_step_output_abi_payload["logical_outputs"]
        if train_step_output_abi_payload["declared"]
        else ()
    )
    train_step_suffix_loss_parameter_grad_buffers = (
        _train_step_suffix_loss_parameter_grad_buffers(
            declared=bool(train_step_output_abi),
        )
    )
    external_buffers_for_abi = _append_unique_names(
        external_buffers,
        (
            *row_phased_launcher_carry_buffers,
            *train_step_suffix_loss_input_buffers,
            *train_step_output_buffers,
            *train_step_suffix_loss_parameter_grad_buffers,
        ),
    )
    extent = _validated_buffer_extent(buffer_extent)
    entry_name = _safe_identifier(
        entry_symbol or getattr(region, "entry_symbol", None) or getattr(region, "name", None)
        or "path_c_descriptor_region"
    )
    internal_buffer_shapes = _internal_buffer_shapes(
        internal_buffers,
        validated_internal_buffer_policy,
        resolved_shape_env,
    )
    shape_by_buffer = {
        name: _buffer_shape(name, extent, resolved_shape_env)
        for name in external_buffers
    }
    dtype_by_buffer = dict(dtype_by_buffer)
    for name in row_phased_launcher_carry_buffers:
        dtype_by_buffer[name] = _buffer_dtype(name, shape_env=resolved_shape_env)
        shape_by_buffer[name] = _buffer_shape(name, extent, resolved_shape_env)
    for name in train_step_suffix_loss_input_buffers:
        dtype_by_buffer[name] = _buffer_dtype(name, shape_env=resolved_shape_env)
        shape_by_buffer[name] = _buffer_shape(name, extent, resolved_shape_env)
    for name in train_step_output_buffers:
        dtype_by_buffer[name] = "float32"
        shape_by_buffer[name] = (1,)
    for name in train_step_suffix_loss_parameter_grad_buffers:
        dtype_by_buffer[name] = _buffer_dtype(name, shape_env=resolved_shape_env)
        shape_by_buffer[name] = _buffer_shape(name, extent, resolved_shape_env)
    loop_extent = _descriptor_loop_extent(
        external_buffers,
        extent,
        resolved_shape_env,
    )
    physical_abi_plan = _physical_abi_plan(
        external_buffers=external_buffers_for_abi,
        shape_by_buffer=shape_by_buffer,
        dtype_by_buffer=dtype_by_buffer,
        buffer_extent=extent,
        loop_extent=loop_extent,
        shape_env=resolved_shape_env,
        physical_abi_policy=validated_physical_abi_policy,
    )
    train_step_loss_cotangent_buffers = _train_step_suffix_loss_cotangent_buffers(
        train_step_loss_source_buffers,
        physical_abi_plan,
    )
    train_step_loss_cotangent_abi_payload = (
        _train_step_loss_cotangent_abi_payload(
            source_logical_buffers=train_step_loss_source_buffers,
            logical_cotangent_buffers=train_step_loss_cotangent_buffers,
            cotangents_computed=(
                "loss" in train_step_computed_output_buffers
                and len(train_step_loss_cotangent_buffers)
                == len(train_step_loss_source_buffers)
            ),
        )
    )
    train_step_suffix_loss_parameter_grad_abi_payload = (
        _train_step_suffix_loss_parameter_grad_abi_payload(
            declared=bool(train_step_output_abi),
            logical_gradient_buffers=train_step_suffix_loss_parameter_grad_buffers,
            physical_abi_plan=physical_abi_plan,
        )
    )
    source, spilled_shared_scratch = _descriptor_prim_func_source(
        entry_name=entry_name,
        nodes=nodes,
        descriptors=descriptors,
        internal_buffers=internal_buffers,
        internal_buffer_shapes=internal_buffer_shapes,
        physical_abi_plan=physical_abi_plan,
        internal_buffer_policy=validated_internal_buffer_policy,
        loop_policy=validated_loop_policy,
        max_rows_per_launch=validated_max_rows_per_launch,
        row_dispatch_mode=validated_row_dispatch_mode,
        external_buffers=external_buffers_for_abi,
        shape_by_buffer=shape_by_buffer,
        dtype_by_buffer=dtype_by_buffer,
        buffer_extent=extent,
        loop_extent=loop_extent,
        shape_env=resolved_shape_env,
        train_step_computed_output_buffers=train_step_computed_output_buffers,
        train_step_loss_source_buffers=train_step_loss_source_buffers,
        train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
        train_step_loss_parameter_grad_buffers=(
            train_step_suffix_loss_parameter_grad_buffers
            if train_step_suffix_loss_parameter_grad_abi_payload["gradients_computed"]
            else ()
        ),
    )

    import tilelang.language as T
    from tilelang.tileop.metal_quant import (
        float_to_fp8_e4m3fn_bits,
        fp8_e4m3fn_to_float,
    )

    filename = f"<path_c_descriptor_schedule:{entry_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, Any] = {
        "T": T,
        "float_to_fp8_e4m3fn_bits": float_to_fp8_e4m3fn_bits,
        "fp8_e4m3fn_to_float": fp8_e4m3fn_to_float,
    }
    exec(compile(source, filename, "exec"), namespace)
    prim_func = namespace[entry_name]
    internal_scratch_abi_buffers = tuple(
        name
        for name, info in spilled_shared_scratch.items()
        if bool(info.get("internal_scratch_abi"))
    )
    internal_scratch_abi_aliases = {
        name: {
            "bank": str(info["bank"]),
            "offset": int(info["offset"]),
            "shape": tuple(info["shape"]),
            "dtype": str(info["dtype"]),
        }
        for name, info in spilled_shared_scratch.items()
        if bool(info.get("internal_scratch_abi"))
        and bool(info.get("coalesced_scratch_bank"))
    }
    if internal_scratch_abi_buffers:
        prim_func = prim_func.with_attr(
            "tl.fusion.internal_scratch_abi_buffers",
            json.dumps(internal_scratch_abi_buffers),
        )
    if internal_scratch_abi_aliases:
        prim_func = prim_func.with_attr(
            "tl.fusion.internal_scratch_abi_aliases",
            json.dumps(internal_scratch_abi_aliases, sort_keys=True),
        )
    prim_func = prim_func.with_attr(
        "tl.fusion.physical_abi.policy",
        validated_physical_abi_policy,
    ).with_attr(
        "tl.fusion.physical_abi.logical_to_physical",
        json.dumps(physical_abi_plan.logical_to_physical, sort_keys=True),
    ).with_attr(
        "tl.fusion.physical_abi.physical_buffer_shapes",
        json.dumps(physical_abi_plan.physical_buffer_shapes, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_output_abi",
        json.dumps(train_step_output_abi_payload, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_suffix_loss_input_abi",
        json.dumps(train_step_suffix_loss_input_abi_payload, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_loss_cotangent_abi",
        json.dumps(train_step_loss_cotangent_abi_payload, sort_keys=True),
    ).with_attr(
        "tl.fusion.train_step_suffix_loss_parameter_grad_abi",
        json.dumps(
            train_step_suffix_loss_parameter_grad_abi_payload,
            sort_keys=True,
        ),
    )
    if validated_max_rows_per_launch is not None:
        prim_func = prim_func.with_attr(
            "tl.fusion.max_rows_per_launch",
            validated_max_rows_per_launch,
        ).with_attr(
            "tl.fusion.row_dispatch_mode",
            validated_row_dispatch_mode,
        )
    if row_chunk_count is not None:
        prim_func = prim_func.with_attr(
            "tl.fusion.row_chunk_count",
            row_chunk_count,
        )
        if (
            validated_row_dispatch_mode
            == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        ):
            prim_func = prim_func.with_attr(
                "tl.fusion.row_chunk_index_param",
                DESCRIPTOR_ROW_CHUNK_INDEX_PARAM,
            ).with_attr(
                "tl.fusion.row_subchunk_index_param",
                DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM,
            ).with_attr(
                "tl.fusion.rows_per_kernel_launch",
                DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
            ).with_attr(
                "tl.fusion.row_subchunk_count",
                max(
                    1,
                    (
                        int(validated_max_rows_per_launch)
                        + DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH
                        - 1
                    )
                    // DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
                ),
            )
    if any(node.op_name.endswith("_bwd") for node in nodes):
        prim_func = prim_func.with_attr(
            "tl.fusion.backward_gate_param",
            DESCRIPTOR_BACKWARD_GATE_PARAM,
        )
    compile_pass_configs = _descriptor_tilelang_compile_pass_configs(
        descriptors,
        loop_policy=validated_loop_policy,
    )
    if compile_pass_configs:
        prim_func = prim_func.with_attr(
            "tilelang_pass_configs",
            compile_pass_configs,
        )
    if validated_loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        prim_func = prim_func.with_attr("tl.fusion.disable_tir_simplify", True)
    owner_output_param_indices = tuple(
        idx
        for idx, param in enumerate(prim_func.params)
        if param in prim_func.buffer_map
    )
    if owner_output_param_indices:
        prim_func = prim_func.with_attr(
            "tilelang_out_idx",
            list(owner_output_param_indices),
        ).with_attr(
            "tilelang_metal_zero_init_output_positions",
            [],
        )
    prim_func._cppmega_path_c_schedule_generator = PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    prim_func._cppmega_path_c_brick_ops = tuple(
        descriptor.op_name for descriptor in descriptors
    )
    prim_func._cppmega_path_c_buffer_extent = extent
    prim_func._cppmega_path_c_shape_env = resolved_shape_env
    prim_func._cppmega_path_c_internal_buffer_policy = (
        validated_internal_buffer_policy
    )
    prim_func._cppmega_path_c_loop_policy = validated_loop_policy
    prim_func._cppmega_path_c_physical_abi_policy = validated_physical_abi_policy
    prim_func._cppmega_path_c_max_rows_per_launch = validated_max_rows_per_launch
    prim_func._cppmega_path_c_row_dispatch_mode = validated_row_dispatch_mode
    prim_func._cppmega_path_c_row_chunk_count = row_chunk_count
    prim_func._cppmega_path_c_internal_buffer_shapes = internal_buffer_shapes
    prim_func._cppmega_path_c_buffer_abi_shapes = {
        name: shape_by_buffer[name]
        for name in external_buffers_for_abi
    }
    prim_func._cppmega_path_c_physical_buffer_abi_shapes = dict(
        physical_abi_plan.physical_buffer_shapes
    )
    prim_func._cppmega_path_c_physical_buffer_abi_map = dict(
        physical_abi_plan.logical_to_physical
    )
    prim_func._cppmega_path_c_spilled_shared_scratch_shapes = dict(
        spilled_shared_scratch
    )
    prim_func._cppmega_path_c_internal_scratch_abi_buffers = (
        internal_scratch_abi_buffers
    )
    prim_func._cppmega_path_c_internal_scratch_abi_aliases = dict(
        internal_scratch_abi_aliases
    )
    prim_func._cppmega_path_c_loop_extent = loop_extent
    prim_func._cppmega_path_c_generated_source = source
    prim_func._cppmega_path_c_compile_pass_configs = compile_pass_configs
    prim_func._cppmega_path_c_train_step_output_abi = dict(
        train_step_output_abi_payload
    )
    prim_func._cppmega_path_c_train_step_suffix_loss_input_abi = dict(
        train_step_suffix_loss_input_abi_payload
    )
    prim_func._cppmega_path_c_train_step_suffix_loss_source_buffers = (
        train_step_loss_source_buffers
    )
    prim_func._cppmega_path_c_train_step_loss_cotangent_abi = dict(
        train_step_loss_cotangent_abi_payload
    )
    prim_func._cppmega_path_c_train_step_suffix_loss_parameter_grad_abi = dict(
        train_step_suffix_loss_parameter_grad_abi_payload
    )
    return prim_func


def _descriptor_tilelang_compile_pass_configs(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    loop_policy: str,
) -> dict[str, bool]:
    configs: dict[str, bool] = {}
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        configs["tl.disable_thread_storage_sync"] = True
        configs["tirx.merge_static_smem"] = False
        configs["tirx.disable_cse_tir"] = True
    if (
        loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        and any(descriptor.op_name.endswith("_bwd") for descriptor in descriptors)
    ):
        configs["tirx.disable_storage_rewrite"] = True
    return configs


def _append_unique_names(
    names: Sequence[str],
    extra_names: Sequence[str],
) -> tuple[str, ...]:
    values = [str(name) for name in names]
    seen = set(values)
    for raw_name in extra_names:
        name = str(raw_name)
        if name in seen:
            continue
        values.append(name)
        seen.add(name)
    return tuple(values)


def _row_phased_launcher_carry_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
    *,
    shape_env: PathCModelShapeEnv | None,
    loop_policy: str,
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
) -> tuple[str, ...]:
    if (
        shape_env is None
        or loop_policy != DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        or max_rows_per_launch is None
        or row_dispatch_mode != DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    ):
        return ()
    carries: list[str] = []
    if any(node.op_name == "mamba3_mimo" for node in nodes):
        if int(shape_env.mamba_num_heads) * int(shape_env.mamba_num_rope_angles) > 0:
            carries.append("mamba3_angle_state")
        if max(0, int(shape_env.mamba_conv_kernel) - 1) > 0:
            carries.append("mamba3_conv_state")
    if any(node.op_name == "m2rnn" for node in nodes):
        carries.append("m2rnn_h_state")
    return tuple(carries)


def _train_step_output_abi_payload(
    *,
    declared: bool,
    computed_logical_outputs: Sequence[str] = (),
) -> dict[str, Any]:
    logical_outputs = _TRAIN_STEP_SCALAR_OUTPUT_ABI_NAMES if declared else ()
    computed_outputs = tuple(
        name
        for name in logical_outputs
        if name in {str(output) for output in computed_logical_outputs}
    )
    pending_outputs = tuple(
        name for name in logical_outputs if name not in set(computed_outputs)
    )
    outputs_computed = bool(declared and logical_outputs and not pending_outputs)
    return {
        "declared": bool(declared),
        "outputs_computed": bool(outputs_computed),
        "computed_logical_outputs": computed_outputs,
        "pending_logical_outputs": pending_outputs,
        "logical_outputs": logical_outputs,
        "reason": _TRAIN_STEP_SCALAR_OUTPUT_ABI_REASON
        if declared and not computed_outputs
        else (
            "train-step scalar ABI computes ntokens in the descriptor body, "
            "but loss remains pending fused suffix codegen"
        )
        if declared and not outputs_computed
        else (
            "train-step scalar ABI slots are generated and populated by fused "
            "suffix loss codegen"
            if declared
            else "train-step scalar ABI slots are not required for this descriptor"
        ),
    }


def _train_step_computed_output_buffers(
    *,
    declared: bool,
    shape_env: PathCModelShapeEnv | None,
    loss_source_buffers: Sequence[str] = (),
) -> tuple[str, ...]:
    if not declared or shape_env is None:
        return ()
    outputs: list[str] = []
    if loss_source_buffers and int(getattr(shape_env, "vocab_size", 0) or 0) > 0:
        outputs.append("loss")
    outputs.append("ntokens")
    return tuple(outputs)


def _train_step_suffix_loss_source_buffers(
    nodes: Sequence[_ScheduleNodeView],
) -> tuple[str, ...]:
    produced = {output for node in nodes for output in node.outputs}
    seed_gradients = sorted(
        input_name
        for node in nodes
        if node.backward == "owner_output"
        for input_name in node.inputs
        if input_name.endswith("_grad") and input_name not in produced
    )
    return tuple(
        name[: -len("_grad")]
        for name in seed_gradients
        if name.endswith("_grad")
    )


def _train_step_suffix_loss_cotangent_buffers(
    source_logical_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
) -> tuple[str, ...]:
    return tuple(
        cotangent_name
        for source_name in source_logical_buffers
        for cotangent_name in (f"{source_name}_grad",)
        if cotangent_name in physical_abi_plan.logical_to_physical
    )


def _train_step_loss_cotangent_abi_payload(
    *,
    source_logical_buffers: Sequence[str],
    logical_cotangent_buffers: Sequence[str],
    cotangents_computed: bool,
) -> dict[str, Any]:
    source_buffers = tuple(str(name) for name in source_logical_buffers)
    cotangent_buffers = tuple(str(name) for name in logical_cotangent_buffers)
    missing_cotangents = tuple(
        f"{name}_grad"
        for name in source_buffers
        if f"{name}_grad" not in set(cotangent_buffers)
    )
    return {
        "declared": bool(source_buffers),
        "cotangents_computed": bool(cotangents_computed and not missing_cotangents),
        "source_logical_buffers": source_buffers,
        "logical_cotangent_buffers": cotangent_buffers,
        "missing_logical_cotangent_buffers": missing_cotangents,
        "reason": (
            "train-step suffix loss cotangents are generated into backward seed buffers"
            if cotangents_computed and not missing_cotangents
            else "train-step suffix loss cotangents are not required for this descriptor"
            if not source_buffers
            else "train-step suffix loss cotangent seed buffers are missing from the physical ABI"
        ),
    }


def _train_step_suffix_loss_parameter_grad_buffers(
    *,
    declared: bool,
) -> tuple[str, ...]:
    return _TRAIN_STEP_SUFFIX_LOSS_PARAMETER_GRAD_ABI_NAMES if declared else ()


def _train_step_suffix_loss_parameter_grad_abi_payload(
    *,
    declared: bool,
    logical_gradient_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
) -> dict[str, Any]:
    parameter_buffers = ("final_norm_weight", "lm_head_weight") if declared else ()
    gradient_buffers = tuple(str(name) for name in logical_gradient_buffers)
    missing_gradients = tuple(
        name
        for name in _TRAIN_STEP_SUFFIX_LOSS_PARAMETER_GRAD_ABI_NAMES
        if name not in physical_abi_plan.logical_to_physical
    )
    gradients_computed = bool(
        declared
        and parameter_buffers
        and not missing_gradients
        and all(name in physical_abi_plan.logical_to_physical for name in parameter_buffers)
    )
    return {
        "declared": bool(declared),
        "parameter_logical_buffers": parameter_buffers,
        "logical_gradient_buffers": gradient_buffers,
        "gradients_computed": gradients_computed,
        "missing_logical_gradient_buffers": missing_gradients,
        "reason": (
            "train-step suffix loss parameter gradients are generated for "
            "final_norm_weight and lm_head_weight"
            if gradients_computed
            else "train-step suffix loss parameter gradients are not required for this descriptor"
            if not declared
            else "train-step suffix loss parameter gradient buffers are missing from the physical ABI"
        ),
    }


def _train_step_suffix_loss_input_abi_payload(
    *,
    declared: bool,
) -> dict[str, Any]:
    return {
        "declared": bool(declared),
        "logical_inputs": _TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_NAMES
        if declared
        else (),
        "reason": _TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_REASON
        if declared
        else "train-step suffix loss inputs are not required for this descriptor",
    }


def _descriptor_prim_func_source(
    *,
    entry_name: str,
    nodes: Sequence[_ScheduleNodeView],
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    internal_buffers: Sequence[str],
    internal_buffer_shapes: Mapping[str, tuple[int, ...]],
    physical_abi_plan: _PhysicalAbiPlan,
    internal_buffer_policy: str,
    loop_policy: str,
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
    external_buffers: Sequence[str],
    shape_by_buffer: Mapping[str, tuple[int, ...]],
    dtype_by_buffer: dict[str, str],
    buffer_extent: int,
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
    train_step_computed_output_buffers: Sequence[str] = (),
    train_step_loss_source_buffers: Sequence[str] = (),
    train_step_loss_cotangent_buffers: Sequence[str] = (),
    train_step_loss_parameter_grad_buffers: Sequence[str] = (),
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    indent = " " * 4
    param_lines = list(physical_abi_plan.param_lines)
    if not param_lines:
        param_lines = [
            f"{indent}_dummy: T.Tensor(({buffer_extent},), \"float32\"),"
        ]
    chunked_row_count = _row_phased_chunk_count(
        loop_policy=loop_policy,
        shape_env=shape_env,
        max_rows_per_launch=max_rows_per_launch,
    )
    chunked_by_launcher = (
        chunked_row_count is not None
        and row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    if chunked_by_launcher:
        param_lines.append(
            f"{indent}{DESCRIPTOR_ROW_CHUNK_INDEX_PARAM}: T.int32,"
        )
        param_lines.append(
            f"{indent}{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM}: T.int32,"
        )
    has_backward = any(node.op_name.endswith("_bwd") for node in nodes)
    if has_backward:
        param_lines.append(
            f"{indent}{DESCRIPTOR_BACKWARD_GATE_PARAM}: T.int32,"
        )
    access_by_buffer = {
        buffer_name: _internal_buffer_ref(
            buffer_name,
            internal_buffer_shapes[buffer_name],
            shape_env,
        )
        for buffer_name in internal_buffers
    }
    for buffer_name, shape in shape_by_buffer.items():
        access_by_buffer[buffer_name] = physical_abi_plan.external_access_by_buffer[
            buffer_name
        ]
    fragments = tuple(
        _descriptor_node_source(
            node=node,
            node_index=node_index,
            descriptor=descriptors[node_index],
            dtype_by_buffer=dtype_by_buffer,
            access_by_buffer=access_by_buffer,
        )
        for node_index, node in enumerate(nodes)
    )
    thread_count = min(DESCRIPTOR_DEFAULT_THREADS, max(1, loop_extent))
    block_count = (loop_extent + thread_count - 1) // thread_count
    if loop_policy == DESCRIPTOR_LOOP_POLICY_FLAT:
        kernel_line = f"{indent}with T.Kernel({block_count}, threads={thread_count}) as bx:"
    elif chunked_by_launcher:
        kernel_line = f"{indent}with T.Kernel(1, threads={thread_count}) as chunk:"
    elif chunked_row_count is not None:
        kernel_line = (
            f"{indent}with T.Kernel({chunked_row_count}, threads={thread_count}) as chunk:"
        )
    else:
        kernel_line = f"{indent}with T.Kernel(1, threads={thread_count}):"

    body: list[str] = [
        "@T.prim_func",
        f"def {entry_name}(",
        *param_lines,
        "):",
        kernel_line,
        f"{indent * 2}# internal_buffer_policy: {internal_buffer_policy}",
        f"{indent * 2}# loop_policy: {loop_policy}",
    ]
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        root_regions = _root_kernel_buffer_regions(
            physical_abi_plan.physical_buffer_shapes
        )
        body.append(
            f"{indent * 2}T.reads({', '.join(root_regions)})  "
            f"# {_DESCRIPTOR_ROOT_READS_MARKER}"
        )
        body.append(
            f"{indent * 2}T.writes({', '.join(root_regions)})  "
            f"# {_DESCRIPTOR_ROOT_WRITES_MARKER}"
        )
    if max_rows_per_launch is not None:
        body.append(f"{indent * 2}# max_rows_per_launch: {max_rows_per_launch}")
        body.append(f"{indent * 2}# row_dispatch_mode: {row_dispatch_mode}")
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        body.append(f"{indent * 2}lane = T.get_thread_binding(0)")
    internal_allocator = (
        "T.alloc_shared"
        if internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        else "T.alloc_local"
    )
    for buffer_name in internal_buffers:
        body.append(
            f"{indent * 2}{_safe_identifier(buffer_name)} = "
            f"{internal_allocator}({_shape_literal(internal_buffer_shapes[buffer_name])}, "
            f"\"{dtype_by_buffer[buffer_name]}\")"
        )
    for node, descriptor, fragment in zip(nodes, descriptors, fragments, strict=True):
        if (
            loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
            and (
                _is_row_phased_mamba3(node, descriptor, shape_env)
                or node.op_name == "residual_rmsnorm"
                or _is_row_phased_m2rnn(node, descriptor, shape_env)
                or _is_row_phased_residual_rmsnorm_bwd(node, descriptor)
                or _is_row_phased_bwd_descriptor(node, descriptor, shape_env)
            )
        ):
            continue
        for allocation in fragment.allocations:
            body.append(f"{indent * 2}{allocation}")
    if not internal_buffers and not any(
        fragment.allocations for fragment in fragments
    ):
        body.append(f"{indent * 2}_scratch = T.alloc_local((1,), \"float32\")")
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        _append_row_phased_hidden_body(
            body,
            nodes=nodes,
            descriptors=descriptors,
            fragments=fragments,
            dtype_by_buffer=dtype_by_buffer,
            access_by_buffer=access_by_buffer,
            shape_env=shape_env,
            train_step_computed_output_buffers=train_step_computed_output_buffers,
            train_step_loss_source_buffers=train_step_loss_source_buffers,
            physical_abi_plan=physical_abi_plan,
            train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
            train_step_loss_parameter_grad_buffers=(
                train_step_loss_parameter_grad_buffers
            ),
            max_rows_per_launch=max_rows_per_launch,
            row_dispatch_mode=row_dispatch_mode,
            indent=indent,
        )
    else:
        _reject_proxy_backward_fragments(
            tuple(zip(nodes, descriptors, fragments, strict=True))
        )
        body.append(f"{indent * 2}tid = T.get_thread_binding(0)")
        body.append(f"{indent * 2}i = bx * {thread_count} + tid")
        body.append(f"{indent * 2}if i < {loop_extent}:")
        for node, descriptor, fragment in zip(
            nodes, descriptors, fragments, strict=True
        ):
            _append_descriptor_node_comments(
                body,
                node=node,
                descriptor=descriptor,
                indent=indent * 3,
            )
            for statement in fragment.statements:
                body.append(f"{indent * 3}{statement}")
        _append_train_step_suffix_scalar_outputs(
            body,
            computed_outputs=train_step_computed_output_buffers,
            loss_source_buffers=train_step_loss_source_buffers,
            physical_abi_plan=physical_abi_plan,
            train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
            train_step_loss_parameter_grad_buffers=(
                train_step_loss_parameter_grad_buffers
            ),
            loop_policy=loop_policy,
            shape_env=shape_env,
            indent=indent,
        )
    return _spill_large_shared_scratch_to_abi(
        body,
        existing_parameter_count=len(param_lines),
        internal_buffer_names=frozenset(internal_buffers),
    )


def _append_train_step_suffix_scalar_outputs(
    body: list[str],
    *,
    computed_outputs: Sequence[str],
    loss_source_buffers: Sequence[str],
    train_step_loss_cotangent_buffers: Sequence[str],
    train_step_loss_parameter_grad_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    loop_policy: str,
    shape_env: PathCModelShapeEnv | None,
    indent: str,
) -> None:
    computed_output_set = {str(output) for output in computed_outputs}
    compute_ntokens = "ntokens" in computed_output_set
    compute_loss = "loss" in computed_output_set
    compute_cotangents = bool(train_step_loss_cotangent_buffers) and compute_loss
    compute_parameter_grads = (
        bool(train_step_loss_parameter_grad_buffers) and compute_loss
    )
    if not compute_ntokens and not compute_loss:
        return
    if shape_env is None:
        return
    if "target_mask" not in physical_abi_plan.logical_to_physical:
        return
    if compute_ntokens and "ntokens" not in physical_abi_plan.logical_to_physical:
        return
    if compute_loss and not _train_step_suffix_loss_can_emit(
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
    ):
        compute_loss = False
        compute_cotangents = False
        compute_parameter_grads = False
    if compute_cotangents and not _train_step_suffix_loss_cotangents_can_emit(
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        cotangent_buffers=train_step_loss_cotangent_buffers,
    ):
        compute_cotangents = False
    if (
        compute_parameter_grads
        and not _train_step_suffix_loss_parameter_grads_can_emit(
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            parameter_grad_buffers=train_step_loss_parameter_grad_buffers,
        )
    ):
        compute_parameter_grads = False

    sequence_length = int(shape_env.sequence_length)
    hidden_size = int(shape_env.hidden_size)
    vocab_size = int(getattr(shape_env, "vocab_size", 0) or 0)
    target_mask_ref = _physical_logical_buffer_ref(
        physical_abi_plan,
        "target_mask",
        "token_row",
    )
    target_mask_value = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "target_mask",
        "token_row",
    )
    target_id_ref = _physical_logical_buffer_ref(
        physical_abi_plan,
        "target_ids",
        "token_row",
    )
    loss_ref = _physical_logical_buffer_ref(physical_abi_plan, "loss", "0")
    ntokens_ref = _physical_logical_buffer_ref(physical_abi_plan, "ntokens", "0")
    guard = (
        "lane == 0"
        if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        else "i == 0"
    )
    body.append(f"{indent * 2}# train_step_suffix_loss_scalar")
    if compute_loss:
        for scratch_name in (
            "train_step_suffix_loss_accum",
            "train_step_suffix_row_sum_sq",
            "train_step_suffix_inv_rms",
            "train_step_suffix_max_logit",
            "train_step_suffix_target_logit",
            "train_step_suffix_logit",
            "train_step_suffix_sum_exp",
            "train_step_suffix_hidden_value",
        ):
            body.append(
                f"{indent * 2}{scratch_name} = T.alloc_local((1,), \"float32\")"
            )
    if compute_cotangents:
        for scratch_name in (
            "train_step_suffix_seed_dot",
            "train_step_suffix_seed_grad_norm",
            "train_step_suffix_seed_softmax",
            "train_step_suffix_seed_class_grad",
            "train_step_suffix_seed_hidden_grad",
        ):
            body.append(
                f"{indent * 2}{scratch_name} = T.alloc_local((1,), \"float32\")"
            )
    if compute_parameter_grads:
        for scratch_name in (
            "train_step_suffix_param_grad_norm",
            "train_step_suffix_param_class_grad",
        ):
            body.append(
                f"{indent * 2}{scratch_name} = T.alloc_local((1,), \"float32\")"
            )
    body.append(f"{indent * 2}if {guard}:")
    if compute_ntokens:
        body.append(f"{indent * 3}# train_step_suffix_loss_ntokens")
        body.append(f"{indent * 3}{ntokens_ref} = T.cast(0.0, \"float32\")")
        body.append(
            f"{indent * 3}for token_row in T.serial(0, {sequence_length}):"
        )
        body.append(
            f"{indent * 4}{ntokens_ref} = {ntokens_ref} + "
            f"T.cast({target_mask_ref}, \"float32\")"
        )
    if compute_loss:
        body.append(f"{indent * 3}{loss_ref} = T.cast(0.0, \"float32\")")
        body.append(
            f"{indent * 3}train_step_suffix_loss_accum[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(
            f"{indent * 3}for token_row in T.serial(0, {sequence_length}):"
        )
        body.append(f"{indent * 4}if T.cast({target_mask_ref}, \"float32\") != 0.0:")
        body.append(
            f"{indent * 5}train_step_suffix_row_sum_sq[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(
            f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):"
        )
        _append_suffix_hidden_value(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target="train_step_suffix_hidden_value[0]",
            indent=indent * 6,
        )
        body.append(
            f"{indent * 6}train_step_suffix_row_sum_sq[0] = "
            "train_step_suffix_row_sum_sq[0] + "
            "(train_step_suffix_hidden_value[0] * "
            "train_step_suffix_hidden_value[0])"
        )
        body.append(
            f"{indent * 5}train_step_suffix_inv_rms[0] = "
            f"T.rsqrt((train_step_suffix_row_sum_sq[0] / {float(hidden_size)}) "
            "+ 0.00001)"
        )
        body.append(
            f"{indent * 5}train_step_suffix_max_logit[0] = "
            "T.float32(-3.4028234663852886e38)"
        )
        body.append(
            f"{indent * 5}train_step_suffix_target_logit[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
        _append_suffix_logit(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            row_expr="token_row",
            vocab_expr="vocab_col",
            indent=indent * 6,
        )
        body.append(
            f"{indent * 6}if train_step_suffix_logit[0] > "
            "train_step_suffix_max_logit[0]:"
        )
        body.append(
            f"{indent * 7}train_step_suffix_max_logit[0] = "
            "train_step_suffix_logit[0]"
        )
        body.append(f"{indent * 6}if vocab_col == {target_id_ref}:")
        body.append(
            f"{indent * 7}train_step_suffix_target_logit[0] = "
            "train_step_suffix_logit[0]"
        )
        body.append(
            f"{indent * 5}train_step_suffix_sum_exp[0] = "
            'T.cast(0.0, "float32")'
        )
        body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
        _append_suffix_logit(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            row_expr="token_row",
            vocab_expr="vocab_col",
            indent=indent * 6,
        )
        body.append(
            f"{indent * 6}train_step_suffix_sum_exp[0] = "
            "train_step_suffix_sum_exp[0] + "
            "T.exp(train_step_suffix_logit[0] - "
            "train_step_suffix_max_logit[0])"
        )
        body.append(
            f"{indent * 5}train_step_suffix_loss_accum[0] = "
            "train_step_suffix_loss_accum[0] + "
            f"(T.cast({target_mask_value}, \"float32\") * "
            "((T.log(train_step_suffix_sum_exp[0]) + "
            "train_step_suffix_max_logit[0]) - "
            "train_step_suffix_target_logit[0]))"
        )
        body.append(
            f"{indent * 3}{loss_ref} = train_step_suffix_loss_accum[0] / "
            f'T.max(T.cast({ntokens_ref}, "float32"), T.cast(1.0, "float32"))'
        )
    if compute_cotangents:
        _append_train_step_suffix_loss_cotangent_seeds(
            body,
            loss_source_buffers=loss_source_buffers,
            cotangent_buffers=train_step_loss_cotangent_buffers,
            physical_abi_plan=physical_abi_plan,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            target_mask_ref=target_mask_ref,
            target_mask_value=target_mask_value,
            target_id_ref=target_id_ref,
            ntokens_ref=ntokens_ref,
            indent=indent,
        )
    if compute_parameter_grads:
        _append_train_step_suffix_loss_parameter_grads(
            body,
            loss_source_buffers=loss_source_buffers,
            parameter_grad_buffers=train_step_loss_parameter_grad_buffers,
            physical_abi_plan=physical_abi_plan,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            target_mask_ref=target_mask_ref,
            target_mask_value=target_mask_value,
            target_id_ref=target_id_ref,
            ntokens_ref=ntokens_ref,
            indent=indent,
        )
    if loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        body.append(f"{indent * 2}T.sync_threads()")


def _train_step_suffix_loss_can_emit(
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
) -> bool:
    required = {
        "loss",
        "ntokens",
        "target_ids",
        "target_mask",
        "final_norm_weight",
        "lm_head_weight",
    }
    required.update(str(name) for name in loss_source_buffers)
    return all(name in physical_abi_plan.logical_to_physical for name in required)


def _train_step_suffix_loss_cotangents_can_emit(
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    cotangent_buffers: Sequence[str],
) -> bool:
    return (
        bool(loss_source_buffers)
        and len(loss_source_buffers) == len(cotangent_buffers)
        and all(
            str(name) in physical_abi_plan.logical_to_physical
            for name in (*loss_source_buffers, *cotangent_buffers)
        )
    )


def _train_step_suffix_loss_parameter_grads_can_emit(
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    parameter_grad_buffers: Sequence[str],
) -> bool:
    required = {
        "final_norm_weight",
        "lm_head_weight",
        "ntokens",
        "target_ids",
        "target_mask",
        *tuple(str(name) for name in loss_source_buffers),
        *tuple(str(name) for name in parameter_grad_buffers),
    }
    return bool(loss_source_buffers) and all(
        name in physical_abi_plan.logical_to_physical for name in required
    )


def _append_train_step_suffix_loss_cotangent_seeds(
    body: list[str],
    *,
    loss_source_buffers: Sequence[str],
    cotangent_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    target_mask_ref: str,
    target_mask_value: str,
    target_id_ref: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    body.append(f"{indent * 3}# train_step_suffix_loss_cotangent_seeds")
    body.append(f"{indent * 3}for token_row in T.serial(0, {sequence_length}):")
    body.append(f"{indent * 4}for suffix_hidden_col in T.serial(0, {hidden_size}):")
    for cotangent_name in cotangent_buffers:
        cotangent_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            str(cotangent_name),
            "token_row",
            "suffix_hidden_col",
        )
        body.append(f"{indent * 5}{cotangent_ref} = T.cast(0.0, \"float32\")")
    body.append(f"{indent * 4}if T.cast({target_mask_ref}, \"float32\") != 0.0:")
    body.append(
        f"{indent * 5}train_step_suffix_row_sum_sq[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(
        f"{indent * 5}for seed_hidden_dot_col in T.serial(0, {hidden_size}):"
    )
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="seed_hidden_dot_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    body.append(
        f"{indent * 6}train_step_suffix_row_sum_sq[0] = "
        "train_step_suffix_row_sum_sq[0] + "
        "(train_step_suffix_hidden_value[0] * "
        "train_step_suffix_hidden_value[0])"
    )
    body.append(
        f"{indent * 5}train_step_suffix_inv_rms[0] = "
        f"T.rsqrt((train_step_suffix_row_sum_sq[0] / {float(hidden_size)}) "
        "+ 0.00001)"
    )
    body.append(
        f"{indent * 5}train_step_suffix_max_logit[0] = "
        "T.float32(-3.4028234663852886e38)"
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_seed_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}if train_step_suffix_logit[0] > "
        "train_step_suffix_max_logit[0]:"
    )
    body.append(
        f"{indent * 7}train_step_suffix_max_logit[0] = "
        "train_step_suffix_logit[0]"
    )
    body.append(
        f"{indent * 5}train_step_suffix_sum_exp[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_seed_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_sum_exp[0] = "
        "train_step_suffix_sum_exp[0] + "
        "T.exp(train_step_suffix_logit[0] - "
        "train_step_suffix_max_logit[0])"
    )
    body.append(
        f"{indent * 5}train_step_suffix_seed_dot[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(
        f"{indent * 5}for seed_hidden_dot_col in T.serial(0, {hidden_size}):"
    )
    _append_train_step_suffix_seed_grad_norm(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        row_expr="token_row",
        hidden_expr="seed_hidden_dot_col",
        target_id_ref=target_id_ref,
        target_mask_value=target_mask_value,
        ntokens_ref=ntokens_ref,
        indent=indent * 6,
    )
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="seed_hidden_dot_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    seed_final_norm = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "final_norm_weight",
        "seed_hidden_dot_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_seed_dot[0] = "
        "train_step_suffix_seed_dot[0] + "
        "(train_step_suffix_seed_grad_norm[0] * "
        f"{seed_final_norm} * train_step_suffix_hidden_value[0])"
    )
    body.append(f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):")
    _append_train_step_suffix_seed_grad_norm(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        row_expr="token_row",
        hidden_expr="suffix_hidden_col",
        target_id_ref=target_id_ref,
        target_mask_value=target_mask_value,
        ntokens_ref=ntokens_ref,
        indent=indent * 6,
    )
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="suffix_hidden_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    final_norm = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "final_norm_weight",
        "suffix_hidden_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_seed_hidden_grad[0] = "
        f"(train_step_suffix_inv_rms[0] * {final_norm} * "
        "train_step_suffix_seed_grad_norm[0]) - "
        "(train_step_suffix_hidden_value[0] * train_step_suffix_inv_rms[0] * "
        "train_step_suffix_inv_rms[0] * train_step_suffix_inv_rms[0] * "
        f"train_step_suffix_seed_dot[0] / {float(hidden_size)})"
    )
    for cotangent_name in cotangent_buffers:
        cotangent_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            str(cotangent_name),
            "token_row",
            "suffix_hidden_col",
        )
        body.append(
            f"{indent * 6}{cotangent_ref} = "
            "train_step_suffix_seed_hidden_grad[0]"
        )


def _append_train_step_suffix_loss_parameter_grads(
    body: list[str],
    *,
    loss_source_buffers: Sequence[str],
    parameter_grad_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    target_mask_ref: str,
    target_mask_value: str,
    target_id_ref: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    buffer_set = {str(name) for name in parameter_grad_buffers}
    body.append(f"{indent * 3}# train_step_suffix_loss_parameter_grads")
    if "final_norm_weight_grad" in buffer_set:
        body.append(f"{indent * 3}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        final_norm_grad_ref = _physical_logical_buffer_ref(
            physical_abi_plan,
            "final_norm_weight_grad",
            "suffix_hidden_col",
        )
        body.append(f"{indent * 4}{final_norm_grad_ref} = T.cast(0.0, \"float32\")")
    if "lm_head_weight_grad" in buffer_set:
        body.append(f"{indent * 3}for vocab_col in T.serial(0, {vocab_size}):")
        body.append(f"{indent * 4}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        lm_head_grad_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            "lm_head_weight_grad",
            "vocab_col",
            "suffix_hidden_col",
        )
        body.append(f"{indent * 5}{lm_head_grad_ref} = T.cast(0.0, \"float32\")")

    body.append(f"{indent * 3}for token_row in T.serial(0, {sequence_length}):")
    body.append(f"{indent * 4}if T.cast({target_mask_ref}, \"float32\") != 0.0:")
    body.append(
        f"{indent * 5}train_step_suffix_row_sum_sq[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):")
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr="token_row",
        hidden_expr="suffix_hidden_col",
        target="train_step_suffix_hidden_value[0]",
        indent=indent * 6,
    )
    body.append(
        f"{indent * 6}train_step_suffix_row_sum_sq[0] = "
        "train_step_suffix_row_sum_sq[0] + "
        "(train_step_suffix_hidden_value[0] * "
        "train_step_suffix_hidden_value[0])"
    )
    body.append(
        f"{indent * 5}train_step_suffix_inv_rms[0] = "
        f"T.rsqrt((train_step_suffix_row_sum_sq[0] / {float(hidden_size)}) "
        "+ 0.00001)"
    )
    body.append(
        f"{indent * 5}train_step_suffix_max_logit[0] = "
        "T.float32(-3.4028234663852886e38)"
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_param_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}if train_step_suffix_logit[0] > "
        "train_step_suffix_max_logit[0]:"
    )
    body.append(
        f"{indent * 7}train_step_suffix_max_logit[0] = "
        "train_step_suffix_logit[0]"
    )
    body.append(
        f"{indent * 5}train_step_suffix_sum_exp[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr="token_row",
        vocab_expr="vocab_col",
        indent=indent * 6,
        hidden_loop_name="suffix_param_logit_hidden_col",
    )
    body.append(
        f"{indent * 6}train_step_suffix_sum_exp[0] = "
        "train_step_suffix_sum_exp[0] + "
        "T.exp(train_step_suffix_logit[0] - "
        "train_step_suffix_max_logit[0])"
    )
    if "lm_head_weight_grad" in buffer_set:
        body.append(f"{indent * 5}for vocab_col in T.serial(0, {vocab_size}):")
        _append_train_step_suffix_param_class_grad(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            row_expr="token_row",
            vocab_expr="vocab_col",
            target_id_ref=target_id_ref,
            target_mask_value=target_mask_value,
            ntokens_ref=ntokens_ref,
            indent=indent * 6,
        )
        body.append(f"{indent * 6}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        _append_suffix_hidden_value(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target="train_step_suffix_hidden_value[0]",
            indent=indent * 7,
        )
        final_norm = _physical_logical_buffer_value_expr(
            physical_abi_plan,
            "final_norm_weight",
            "suffix_hidden_col",
        )
        lm_head_grad_ref = _physical_logical_buffer_ref_2d(
            physical_abi_plan,
            "lm_head_weight_grad",
            "vocab_col",
            "suffix_hidden_col",
        )
        body.append(
            f"{indent * 7}{lm_head_grad_ref} = {lm_head_grad_ref} + "
            "(train_step_suffix_param_class_grad[0] * "
            "train_step_suffix_hidden_value[0] * "
            f"train_step_suffix_inv_rms[0] * {final_norm})"
        )
    if "final_norm_weight_grad" in buffer_set:
        body.append(f"{indent * 5}for suffix_hidden_col in T.serial(0, {hidden_size}):")
        _append_train_step_suffix_param_grad_norm(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target_id_ref=target_id_ref,
            target_mask_value=target_mask_value,
            ntokens_ref=ntokens_ref,
            indent=indent * 6,
        )
        _append_suffix_hidden_value(
            body,
            physical_abi_plan=physical_abi_plan,
            loss_source_buffers=loss_source_buffers,
            row_expr="token_row",
            hidden_expr="suffix_hidden_col",
            target="train_step_suffix_hidden_value[0]",
            indent=indent * 6,
        )
        final_norm_grad_ref = _physical_logical_buffer_ref(
            physical_abi_plan,
            "final_norm_weight_grad",
            "suffix_hidden_col",
        )
        body.append(
            f"{indent * 6}{final_norm_grad_ref} = {final_norm_grad_ref} + "
            "(train_step_suffix_param_grad_norm[0] * "
            "train_step_suffix_hidden_value[0] * "
            "train_step_suffix_inv_rms[0])"
        )


def _append_train_step_suffix_param_grad_norm(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    vocab_size: int,
    row_expr: str,
    hidden_expr: str,
    target_id_ref: str,
    target_mask_value: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    body.append(
        f"{indent}train_step_suffix_param_grad_norm[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent}for vocab_col in T.serial(0, {vocab_size}):")
    _append_train_step_suffix_param_class_grad(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr=row_expr,
        vocab_expr="vocab_col",
        target_id_ref=target_id_ref,
        target_mask_value=target_mask_value,
        ntokens_ref=ntokens_ref,
        indent=f"{indent}    ",
    )
    lm_head = _physical_logical_buffer_value_expr_2d(
        physical_abi_plan,
        "lm_head_weight",
        "vocab_col",
        hidden_expr,
    )
    body.append(
        f"{indent}    train_step_suffix_param_grad_norm[0] = "
        "train_step_suffix_param_grad_norm[0] + "
        f"(train_step_suffix_param_class_grad[0] * {lm_head})"
    )


def _append_train_step_suffix_param_class_grad(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    row_expr: str,
    vocab_expr: str,
    target_id_ref: str,
    target_mask_value: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr=row_expr,
        vocab_expr=vocab_expr,
        indent=indent,
        hidden_loop_name="suffix_param_logit_hidden_col",
    )
    body.append(
        f"{indent}train_step_suffix_param_class_grad[0] = "
        "T.exp(train_step_suffix_logit[0] - train_step_suffix_max_logit[0]) / "
        "train_step_suffix_sum_exp[0]"
    )
    body.append(f"{indent}if {vocab_expr} == {target_id_ref}:")
    body.append(
        f"{indent}    train_step_suffix_param_class_grad[0] = "
        "train_step_suffix_param_class_grad[0] - T.cast(1.0, \"float32\")"
    )
    body.append(
        f"{indent}train_step_suffix_param_class_grad[0] = "
        "train_step_suffix_param_class_grad[0] * "
        f"T.cast({target_mask_value}, \"float32\") / "
        f'T.max(T.cast({ntokens_ref}, "float32"), T.cast(1.0, "float32"))'
    )


def _append_train_step_suffix_seed_grad_norm(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    vocab_size: int,
    row_expr: str,
    hidden_expr: str,
    target_id_ref: str,
    target_mask_value: str,
    ntokens_ref: str,
    indent: str,
) -> None:
    body.append(
        f"{indent}train_step_suffix_seed_grad_norm[0] = "
        'T.cast(0.0, "float32")'
    )
    body.append(f"{indent}for vocab_col in T.serial(0, {vocab_size}):")
    _append_suffix_logit(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        hidden_size=hidden_size,
        row_expr=row_expr,
        vocab_expr="vocab_col",
        indent=f"{indent}    ",
        hidden_loop_name="suffix_seed_logit_hidden_col",
    )
    body.append(
        f"{indent}    train_step_suffix_seed_softmax[0] = "
        "T.exp(train_step_suffix_logit[0] - train_step_suffix_max_logit[0]) / "
        "train_step_suffix_sum_exp[0]"
    )
    body.append(
        f"{indent}    train_step_suffix_seed_class_grad[0] = "
        "train_step_suffix_seed_softmax[0]"
    )
    body.append(f"{indent}    if vocab_col == {target_id_ref}:")
    body.append(
        f"{indent}        train_step_suffix_seed_class_grad[0] = "
        "train_step_suffix_seed_class_grad[0] - T.cast(1.0, \"float32\")"
    )
    body.append(
        f"{indent}    train_step_suffix_seed_class_grad[0] = "
        "train_step_suffix_seed_class_grad[0] * "
        f"T.cast({target_mask_value}, \"float32\") / "
        f'T.max(T.cast({ntokens_ref}, "float32"), T.cast(1.0, "float32"))'
    )
    lm_head = _physical_logical_buffer_value_expr_2d(
        physical_abi_plan,
        "lm_head_weight",
        "vocab_col",
        hidden_expr,
    )
    body.append(
        f"{indent}    train_step_suffix_seed_grad_norm[0] = "
        "train_step_suffix_seed_grad_norm[0] + "
        f"(train_step_suffix_seed_class_grad[0] * {lm_head})"
    )


def _append_suffix_hidden_value(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    row_expr: str,
    hidden_expr: str,
    target: str,
    indent: str,
) -> None:
    body.append(f'{indent}{target} = T.cast(0.0, "float32")')
    for source_name in loss_source_buffers:
        source_ref = _physical_logical_buffer_value_expr_2d(
            physical_abi_plan,
            str(source_name),
            row_expr,
            hidden_expr,
        )
        body.append(f"{indent}{target} = {target} + {source_ref}")


def _append_suffix_logit(
    body: list[str],
    *,
    physical_abi_plan: _PhysicalAbiPlan,
    loss_source_buffers: Sequence[str],
    hidden_size: int,
    row_expr: str,
    vocab_expr: str,
    indent: str,
    hidden_loop_name: str = "suffix_hidden_col",
) -> None:
    body.append(f'{indent}train_step_suffix_logit[0] = T.cast(0.0, "float32")')
    body.append(f"{indent}for {hidden_loop_name} in T.serial(0, {hidden_size}):")
    _append_suffix_hidden_value(
        body,
        physical_abi_plan=physical_abi_plan,
        loss_source_buffers=loss_source_buffers,
        row_expr=row_expr,
        hidden_expr=hidden_loop_name,
        target="train_step_suffix_hidden_value[0]",
        indent=f"{indent}    ",
    )
    final_norm = _physical_logical_buffer_value_expr(
        physical_abi_plan,
        "final_norm_weight",
        hidden_loop_name,
    )
    lm_head = _physical_logical_buffer_value_expr_2d(
        physical_abi_plan,
        "lm_head_weight",
        vocab_expr,
        hidden_loop_name,
    )
    body.append(
        f"{indent}    train_step_suffix_logit[0] = "
        "train_step_suffix_logit[0] + "
        "(train_step_suffix_hidden_value[0] * "
        f"train_step_suffix_inv_rms[0] * {final_norm} * {lm_head})"
    )


def _physical_logical_buffer_ref(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    index_expr: str,
) -> str:
    info = physical_abi_plan.logical_to_physical.get(logical_name)
    if info is None:
        return f"{_safe_identifier(logical_name)}[{index_expr}]"
    size = int(info.get("size", 1) or 1)
    expr = "0" if size <= 1 else index_expr
    bank_name = str(info["bank"])
    offset = int(info.get("offset", 0) or 0)
    if offset == 0:
        return f"{bank_name}[{expr}]"
    return f"{bank_name}[{offset} + ({expr})]"


def _physical_logical_buffer_ref_2d(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    row_expr: str,
    col_expr: str,
) -> str:
    info = physical_abi_plan.logical_to_physical.get(logical_name)
    if info is None:
        return f"{_safe_identifier(logical_name)}[{row_expr}, {col_expr}]"
    shape = tuple(int(dim) for dim in info.get("shape", ()) or ())
    logical_shape = tuple(
        int(dim)
        for dim in (info.get("logical_shape", ()) or shape)
    )
    bank_name = str(info["bank"])
    offset = int(info.get("offset", 0) or 0)
    if offset == 0 and bank_name == logical_name and len(logical_shape) == 2:
        return f"{bank_name}[{row_expr}, {col_expr}]"
    if (
        offset == 0
        and bank_name == logical_name
        and len(logical_shape) == 3
        and logical_shape[0] == 1
    ):
        return f"{bank_name}[0, {row_expr}, {col_expr}]"
    if len(logical_shape) == 2:
        stride = logical_shape[1]
    elif len(logical_shape) == 3 and logical_shape[0] == 1:
        stride = logical_shape[2]
    else:
        stride = shape[1] if len(shape) == 2 else 1
    return _physical_logical_buffer_ref(
        physical_abi_plan,
        logical_name,
        f"({row_expr}) * {stride} + ({col_expr})",
    )


def _physical_logical_buffer_value_expr(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    index_expr: str,
) -> str:
    ref = _physical_logical_buffer_ref(physical_abi_plan, logical_name, index_expr)
    return _physical_logical_buffer_value_from_ref(
        physical_abi_plan,
        logical_name,
        ref,
    )


def _physical_logical_buffer_value_expr_2d(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    row_expr: str,
    col_expr: str,
) -> str:
    ref = _physical_logical_buffer_ref_2d(
        physical_abi_plan,
        logical_name,
        row_expr,
        col_expr,
    )
    return _physical_logical_buffer_value_from_ref(
        physical_abi_plan,
        logical_name,
        ref,
    )


def _physical_logical_buffer_value_from_ref(
    physical_abi_plan: _PhysicalAbiPlan,
    logical_name: str,
    ref: str,
) -> str:
    info = physical_abi_plan.logical_to_physical.get(logical_name, {})
    dtype = str(info.get("dtype", "float32"))
    if dtype in {"bfloat16", "float16", "int32"}:
        return f'T.cast({ref}, "float32")'
    if dtype == "uint8":
        return f"fp8_e4m3fn_to_float({ref})"
    return ref


_ALLOC_SHARED_LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*) = '
    r'T\.alloc_shared\((?P<shape>\([^)]*\)), "(?P<dtype>[^"]+)"\)$'
)


def _coalesced_scratch_bank_name(dtype: str) -> str:
    return _safe_identifier(f"path_c_{dtype}_scratch_bank")


def _root_kernel_buffer_region(buffer_name: str, shape: Sequence[int]) -> str:
    extents = ", ".join(f"0:{int(dim)}" for dim in shape)
    return f"{_safe_identifier(buffer_name)}[{extents}]"


def _root_kernel_buffer_regions(
    shapes_by_buffer: Mapping[str, Sequence[int]],
) -> list[str]:
    return [
        _root_kernel_buffer_region(buffer_name, shape)
        for buffer_name, shape in shapes_by_buffer.items()
    ]


def _append_regions_to_root_annotation(
    line: str,
    *,
    marker: str,
    extra_regions: Sequence[str],
) -> str:
    if marker not in line or not extra_regions:
        return line
    match = re.match(r"(\s*T\.(?:reads|writes)\()(.+?)(\)\s*# .*)", line)
    if match is None:
        return line
    existing = match.group(2).strip()
    regions = [existing] if existing else []
    regions.extend(extra_regions)
    return f"{match.group(1)}{', '.join(regions)}{match.group(3)}"


def _can_coalesce_spilled_scratch(candidate: Mapping[str, Any]) -> bool:
    return len(tuple(candidate["shape"])) == 1 and str(candidate["dtype"]) in {
        "float32",
        "int32",
    }


def _coalesced_scratch_parameter_count(
    candidates: Sequence[Mapping[str, Any]],
) -> int:
    coalesced_banks = {
        _coalesced_scratch_bank_name(str(candidate["dtype"]))
        for candidate in candidates
        if _can_coalesce_spilled_scratch(candidate)
    }
    standalone = sum(
        1 for candidate in candidates if not _can_coalesce_spilled_scratch(candidate)
    )
    return len(coalesced_banks) + standalone


def _replace_one_dimensional_buffer_refs(
    line: str,
    *,
    source_name: str,
    target_name: str,
    target_offset: int,
) -> str:
    if f"{source_name}[" not in line:
        return line
    pieces: list[str] = []
    cursor = 0
    needle = f"{source_name}["
    while True:
        start = line.find(needle, cursor)
        if start < 0:
            pieces.append(line[cursor:])
            break
        if start > 0 and (line[start - 1].isalnum() or line[start - 1] == "_"):
            pieces.append(line[cursor : start + len(needle)])
            cursor = start + len(needle)
            continue
        depth = 1
        index_start = start + len(needle)
        end = index_start
        while end < len(line) and depth:
            char = line[end]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            end += 1
        if depth:
            pieces.append(line[cursor:])
            break
        index_expr = line[index_start : end - 1]
        if target_offset == 0:
            replacement = f"{target_name}[{index_expr}]"
        else:
            replacement = f"{target_name}[{target_offset} + ({index_expr})]"
        pieces.append(line[cursor:start])
        pieces.append(replacement)
        cursor = end
    return "".join(pieces)


def _spill_large_shared_scratch_to_abi(
    source_lines: Sequence[str],
    *,
    existing_parameter_count: int,
    internal_buffer_names: frozenset[str] = frozenset(),
) -> tuple[str, Mapping[str, Mapping[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    total_shared_bytes = 0
    for line in source_lines:
        match = _ALLOC_SHARED_LINE_RE.match(line)
        if match is None:
            continue
        shape_value = ast.literal_eval(match.group("shape"))
        shape = (
            (int(shape_value),)
            if isinstance(shape_value, int)
            else tuple(int(dim) for dim in shape_value)
        )
        dtype = match.group("dtype")
        byte_count = _flattened_extent(shape) * _DTYPE_NBYTES[dtype]
        total_shared_bytes += byte_count
        scratch_name = match.group("name")
        force_scratch_abi = scratch_name in DESCRIPTOR_ROW_PHASED_BWD_SCRATCH_ABI_BUFFERS
        if (
            byte_count > DESCRIPTOR_SHARED_SCRATCH_SPILL_THRESHOLD_BYTES
            or force_scratch_abi
        ):
            candidates.append(
                {
                    "name": scratch_name,
                    "param_name": scratch_name,
                    "shape": shape,
                    "dtype": dtype,
                    "bytes": byte_count,
                    "force_scratch_abi": force_scratch_abi,
                    "internal_scratch_abi": scratch_name in internal_buffer_names,
                }
            )

    available_parameters = max(
        0,
        DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT - existing_parameter_count,
    )
    remaining_shared_bytes = total_shared_bytes
    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("force_scratch_abi")),
            -int(item["bytes"]),
        ),
    ):
        if (
            _coalesced_scratch_parameter_count((*selected, candidate))
            > available_parameters
        ):
            break
        if (
            not bool(candidate.get("force_scratch_abi"))
            and
            remaining_shared_bytes <= DESCRIPTOR_SHARED_SCRATCH_BUDGET_BYTES
            and int(candidate["bytes"])
            <= DESCRIPTOR_SHARED_SCRATCH_SPILL_THRESHOLD_BYTES
        ):
            break
        selected.append(candidate)
        remaining_shared_bytes -= int(candidate["bytes"])
        if (
            remaining_shared_bytes <= DESCRIPTOR_SHARED_SCRATCH_BUDGET_BYTES
            and not bool(candidate.get("force_scratch_abi"))
        ):
            break

    coalesced_offsets: dict[str, tuple[str, int]] = {}
    coalesced_shapes: dict[str, int] = {}
    for candidate in selected:
        if not _can_coalesce_spilled_scratch(candidate):
            continue
        bank_name = _coalesced_scratch_bank_name(str(candidate["dtype"]))
        offset = coalesced_shapes.get(bank_name, 0)
        coalesced_offsets[str(candidate["name"])] = (bank_name, offset)
        coalesced_shapes[bank_name] = offset + _flattened_extent(candidate["shape"])

    spilled = {
        str(candidate["name"]): {
            "dtype": str(candidate["dtype"]),
            "param_name": coalesced_offsets.get(str(candidate["name"]), ("", 0))[0]
            or str(candidate["param_name"]),
            "shape": tuple(candidate["shape"]),
            "bytes": int(candidate["bytes"]),
            "internal_scratch_abi": bool(candidate["internal_scratch_abi"]),
            "coalesced_scratch_bank": str(candidate["name"]) in coalesced_offsets,
            "bank": coalesced_offsets.get(str(candidate["name"]), ("", 0))[0],
            "offset": coalesced_offsets.get(str(candidate["name"]), ("", 0))[1],
        }
        for candidate in selected
    }
    if not spilled:
        return "\n".join(source_lines) + "\n", {}

    extra_root_regions = [
        _root_kernel_buffer_region(bank_name, (extent,))
        for bank_name, extent in coalesced_shapes.items()
    ]
    extra_root_regions.extend(
        _root_kernel_buffer_region(str(info["param_name"]), info["shape"])
        for info in spilled.values()
        if not bool(info.get("coalesced_scratch_bank"))
    )

    def rewrite_scratch_refs(line: str) -> str:
        rewritten_line = line
        for scratch_name, (bank_name, offset) in coalesced_offsets.items():
            rewritten_line = _replace_one_dimensional_buffer_refs(
                rewritten_line,
                source_name=scratch_name,
                target_name=bank_name,
                target_offset=offset,
            )
        return rewritten_line

    rewritten: list[str] = []
    signature_close_index: int | None = None
    for line in source_lines:
        match = _ALLOC_SHARED_LINE_RE.match(line)
        if match is not None and match.group("name") in spilled:
            continue
        if signature_close_index is None and line == "):":
            signature_close_index = len(rewritten)
        rewritten_line = rewrite_scratch_refs(line)
        rewritten_line = _append_regions_to_root_annotation(
            rewritten_line,
            marker=_DESCRIPTOR_ROOT_READS_MARKER,
            extra_regions=extra_root_regions,
        )
        rewritten_line = _append_regions_to_root_annotation(
            rewritten_line,
            marker=_DESCRIPTOR_ROOT_WRITES_MARKER,
            extra_regions=extra_root_regions,
        )
        rewritten.append(rewritten_line)

    if signature_close_index is None:
        raise ValueError("descriptor source did not contain a function signature close")
    param_indent = " " * 4
    coalesced_bank_dtypes = {
        bank_name: bank_name.removeprefix("path_c_").removesuffix("_scratch_bank")
        for bank_name in coalesced_shapes
    }
    coalesced_param_lines = [
        f'{param_indent}{bank_name}: T.Tensor(({extent},), '
        f'"{coalesced_bank_dtypes[bank_name]}"),'
        for bank_name, extent in coalesced_shapes.items()
    ]
    spill_param_lines = [
        f'{param_indent}{info["param_name"]}: T.Tensor({_shape_literal(info["shape"])}, '
        f'"{info["dtype"]}"),'
        for name, info in spilled.items()
        if not bool(info.get("coalesced_scratch_bank"))
    ]
    rewritten[signature_close_index:signature_close_index] = [
        *coalesced_param_lines,
        *spill_param_lines,
    ]
    return "\n".join(rewritten) + "\n", spilled


def _validated_internal_buffer_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in {
        DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
        DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
    }:
        raise ValueError(
            "internal_buffer_policy must be one of "
            f"{DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL!r}, "
            f"{DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN!r}; "
            f"got {policy!r}"
        )
    return normalized


def _validated_loop_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in {
        DESCRIPTOR_LOOP_POLICY_FLAT,
        DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
    }:
        raise ValueError(
            "loop_policy must be one of "
            f"{DESCRIPTOR_LOOP_POLICY_FLAT!r}, "
            f"{DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN!r}; "
            f"got {policy!r}"
        )
    return normalized


def _validated_physical_abi_policy(policy: str) -> str:
    normalized = str(policy)
    if normalized not in {
        DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
        DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_DTYPE,
        DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE,
    }:
        raise ValueError(
            "physical_abi_policy must be one of "
            f"{DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT!r}, "
            f"{DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_DTYPE!r}, "
            f"{DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE!r}; "
            f"got {policy!r}"
        )
    return normalized


def _physical_abi_plan(
    *,
    external_buffers: Sequence[str],
    shape_by_buffer: Mapping[str, tuple[int, ...]],
    dtype_by_buffer: Mapping[str, str],
    buffer_extent: int,
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
    physical_abi_policy: str,
) -> _PhysicalAbiPlan:
    indent = " " * 4
    if physical_abi_policy == DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT:
        return _direct_physical_abi_plan(
            external_buffers=external_buffers,
            shape_by_buffer=shape_by_buffer,
            dtype_by_buffer=dtype_by_buffer,
            loop_extent=loop_extent,
            shape_env=shape_env,
            indent=indent,
        )
    if not external_buffers:
        return _PhysicalAbiPlan((), {}, {}, {})

    bank_order: list[str] = []
    bank_totals: dict[str, int] = {}
    bank_dtypes: dict[str, str] = {}
    access_by_buffer: dict[str, str] = {}
    logical_to_physical: dict[str, Mapping[str, Any]] = {}
    for buffer_name in external_buffers:
        dtype = dtype_by_buffer[buffer_name]
        bank_name = _physical_abi_bank_name_for_buffer(
            dtype,
            buffer_name,
            physical_abi_policy=physical_abi_policy,
        )
        if bank_name not in bank_totals:
            bank_order.append(bank_name)
            bank_totals[bank_name] = 0
            bank_dtypes[bank_name] = dtype
        offset = bank_totals[bank_name]
        shape = shape_by_buffer[buffer_name]
        size = max(1, _flattened_extent(shape))
        bank_totals[bank_name] += size
        logical_ref = _loop_indexed_buffer_ref(
            buffer_name,
            shape,
            loop_extent,
            shape_env,
        )
        access_by_buffer[buffer_name] = _banked_buffer_ref(
            logical_ref,
            bank_name=bank_name,
            offset=offset,
        )
        logical_shape = _direct_logical_buffer_shape(
            buffer_name,
            shape,
            shape_env,
        )
        logical_to_physical[buffer_name] = {
            "bank": bank_name,
            "dtype": dtype,
            "offset": offset,
            "shape": shape,
            "logical_shape": logical_shape,
            "size": size,
        }

    physical_shapes = {bank_name: (bank_totals[bank_name],) for bank_name in bank_order}
    param_lines = tuple(
        f"{indent}{bank_name}: "
        f"T.Tensor({_shape_literal(physical_shapes[bank_name])}, "
        f"\"{bank_dtypes[bank_name]}\"),"
        for bank_name in bank_order
    )
    return _PhysicalAbiPlan(
        param_lines=param_lines,
        external_access_by_buffer=access_by_buffer,
        physical_buffer_shapes=physical_shapes,
        logical_to_physical=logical_to_physical,
    )


def _direct_physical_abi_plan(
    *,
    external_buffers: Sequence[str],
    shape_by_buffer: Mapping[str, tuple[int, ...]],
    dtype_by_buffer: Mapping[str, str],
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
    indent: str,
) -> _PhysicalAbiPlan:
    direct_shape_by_buffer = {
        buffer_name: _direct_logical_buffer_shape(
            buffer_name,
            shape_by_buffer[buffer_name],
            shape_env,
        )
        for buffer_name in external_buffers
    }
    access_by_buffer = {
        buffer_name: _direct_loop_indexed_buffer_ref(
            buffer_name,
            direct_shape_by_buffer[buffer_name],
            loop_extent,
            shape_env,
        )
        for buffer_name in external_buffers
    }
    physical_shapes = {
        buffer_name: direct_shape_by_buffer[buffer_name]
        for buffer_name in external_buffers
    }
    logical_to_physical = {
        buffer_name: {
            "bank": buffer_name,
            "dtype": dtype_by_buffer[buffer_name],
            "offset": 0,
            "shape": direct_shape_by_buffer[buffer_name],
            "logical_shape": direct_shape_by_buffer[buffer_name],
            "size": _flattened_extent(direct_shape_by_buffer[buffer_name]),
        }
        for buffer_name in external_buffers
    }
    param_lines = tuple(
        f"{indent}{name}: T.Tensor({_shape_literal(direct_shape_by_buffer[name])}, "
        f"\"{dtype_by_buffer[name]}\"),"
        for name in external_buffers
    )
    return _PhysicalAbiPlan(
        param_lines=param_lines,
        external_access_by_buffer=access_by_buffer,
        physical_buffer_shapes=physical_shapes,
        logical_to_physical=logical_to_physical,
    )


def _direct_logical_buffer_shape(
    buffer_name: str,
    flat_shape: tuple[int, ...],
    shape_env: PathCModelShapeEnv | None,
) -> tuple[int, ...]:
    if shape_env is None:
        return flat_shape
    hidden = shape_env.hidden_size
    q_dim = shape_env.attention_num_q_heads * shape_env.attention_head_dim
    kv_dim = shape_env.attention_num_kv_heads * shape_env.attention_head_dim
    canonical_name = _canonical_buffer_name(buffer_name)
    if canonical_name == "mamba3_in_proj_weight":
        return (shape_env.mamba_in_proj_dim, hidden)
    if canonical_name == "mamba3_out_proj_weight":
        return (hidden, shape_env.mamba_inner_dim)
    if canonical_name == "mamba3_conv_weight":
        return (shape_env.mamba_conv_channels, shape_env.mamba_conv_kernel, 1)
    if canonical_name in {
        "mamba3_B_norm_weight",
        "mamba3_B_bias",
        "mamba3_C_norm_weight",
        "mamba3_C_bias",
    }:
        return (
            shape_env.mamba_effective_mimo_rank,
            shape_env.mamba_groups,
            shape_env.mamba_state_dim,
        )
    if canonical_name == "m2rnn_in_proj_weight":
        return (shape_env.m2rnn_in_proj_dim, hidden)
    if canonical_name == "m2rnn_conv_weight":
        return (shape_env.m2rnn_conv_dim, shape_env.m2rnn_conv_kernel, 1)
    if canonical_name == "m2rnn_state_weight":
        return (
            shape_env.m2rnn_num_weight_heads,
            shape_env.m2rnn_v_head_dim,
            shape_env.m2rnn_v_head_dim,
        )
    if canonical_name == "m2rnn_D":
        return (shape_env.m2rnn_num_heads, shape_env.m2rnn_v_head_dim)
    if canonical_name == "m2rnn_out_proj_weight":
        return (
            hidden,
            shape_env.m2rnn_num_heads * shape_env.m2rnn_v_head_dim,
        )
    if canonical_name == "attention_q_proj_weight":
        return (q_dim, hidden)
    if canonical_name == "attention_sparse_kv_proj_weight":
        return (kv_dim, hidden)
    if canonical_name == "attention_out_proj_weight":
        return (hidden, q_dim)
    if canonical_name == "final_norm_weight":
        return (hidden,)
    if canonical_name == "lm_head_weight":
        vocab = max(1, int(getattr(shape_env, "vocab_size", 0) or 0))
        return (vocab, hidden)
    sequence_hidden_names = {
        "hidden",
        "mamba3_delta",
        "m2rnn_hidden",
        "m2rnn_delta",
        "attention_hidden",
        "hidden_after_mamba3",
        "hidden_after_m2rnn",
        "attention_out",
    }
    ungrad_name = (
        str(buffer_name)[: -len("_grad")]
        if str(buffer_name).endswith("_grad")
        else str(buffer_name)
    )
    if (
        canonical_name in sequence_hidden_names
        or ungrad_name.endswith("_hidden")
        or ungrad_name.endswith("_hidden_after")
        or ungrad_name.endswith("_delta")
        or ungrad_name.endswith("_out")
    ):
        return (1, shape_env.sequence_length, hidden)
    return flat_shape


def _direct_loop_indexed_buffer_ref(
    buffer_name: str,
    shape: Sequence[int],
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> str:
    if len(tuple(shape)) > 1:
        return _row_major_buffer_ref(_safe_identifier(buffer_name), shape)
    return _loop_indexed_buffer_ref(buffer_name, shape, loop_extent, shape_env)


def _row_major_buffer_ref(name: str, shape: Sequence[int]) -> str:
    dims = tuple(int(dim) for dim in shape)
    flat_extent = _flattened_extent(dims)
    if len(dims) == 2:
        return (
            f"{name}[(i % {flat_extent}) // {dims[1]}, "
            f"(i % {flat_extent}) % {dims[1]}]"
        )
    if len(dims) == 3:
        if dims[0] == 1:
            return f"{name}[0, i // {dims[2]}, i % {dims[2]}]"
        inner = dims[1] * dims[2]
        return (
            f"{name}[(i % {flat_extent}) // {inner}, "
            f"((i % {flat_extent}) // {dims[2]}) % {dims[1]}, "
            f"(i % {flat_extent}) % {dims[2]}]"
        )
    return f"{name}[i % {flat_extent}]"


def _physical_abi_bank_name(dtype: str) -> str:
    return _safe_identifier(f"path_c_{dtype}_abi_bank")


def _physical_abi_bank_name_for_buffer(
    dtype: str,
    buffer_name: str,
    *,
    physical_abi_policy: str,
) -> str:
    if (
        physical_abi_policy != DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE
        or dtype != "float32"
    ):
        return _physical_abi_bank_name(dtype)
    return _safe_identifier(f"path_c_{dtype}_{_physical_abi_role(buffer_name)}_abi_bank")


def _physical_abi_role(buffer_name: str) -> str:
    name = str(buffer_name)
    if name.endswith("_grad"):
        base = name[: -len("_grad")]
        canonical_base = _canonical_buffer_name(base)
        if _physical_abi_activation_like(base, canonical_base):
            return "activation_gradient"
        return "parameter_gradient"
    canonical = _canonical_buffer_name(buffer_name)
    if (
        canonical.endswith("_weight")
        or canonical.endswith("_bias")
        or canonical in {
            "mamba3_D",
            "mamba3_h0",
            "m2rnn_A_log",
            "m2rnn_D",
            "m2rnn_h0",
            "sparse_mla_sinks",
            "sparse_mla_sm_scale",
        }
    ):
        return "parameter"
    if canonical in {
        "mamba_state",
        "scan_state",
        "mamba3_angle_state",
        "mamba3_conv_state",
        "m2rnn_conv_state",
        "m2rnn_h_state",
    }:
        return "state"
    if canonical in {
        "q_scale",
        "kv_scale",
        "lse",
    }:
        return "attention"
    return "activation"


def _physical_abi_activation_like(buffer_name: str, canonical_name: str) -> bool:
    name = str(buffer_name)
    return (
        canonical_name
        in {
            "hidden",
            "hidden_after_m2rnn",
            "attention_out",
            "mamba3_delta",
            "m2rnn_delta",
        }
        or name.endswith("_hidden")
        or name.endswith("_delta")
        or name.endswith("_out")
    )


def _banked_buffer_ref(
    logical_ref: str,
    *,
    bank_name: str,
    offset: int,
) -> str:
    match = re.fullmatch(r"[A-Za-z_]\w*\[(.+)\]", logical_ref)
    expr = match.group(1) if match is not None else "0"
    if offset == 0:
        return f"{bank_name}[{expr}]"
    return f"{bank_name}[{offset} + ({expr})]"


def _internal_buffer_shapes(
    internal_buffers: Sequence[str],
    internal_buffer_policy: str,
    shape_env: PathCModelShapeEnv | None,
) -> dict[str, tuple[int, ...]]:
    if (
        internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        and shape_env is None
        and internal_buffers
    ):
        raise ValueError(
            "row_local_hidden internal buffer policy requires a model shape_env"
        )
    if (
        internal_buffer_policy
        == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        and shape_env is not None
    ):
        return {
            buffer_name: _row_local_internal_buffer_shape(buffer_name, shape_env)
            for buffer_name in internal_buffers
        }
    return {buffer_name: (1,) for buffer_name in internal_buffers}


def _row_local_internal_buffer_shape(
    buffer_name: str,
    shape_env: PathCModelShapeEnv,
) -> tuple[int, ...]:
    canonical_name = _canonical_buffer_name(buffer_name)
    if canonical_name in {
        "mamba3_delta",
        "m2rnn_hidden",
        "m2rnn_delta",
        "attention_hidden",
        "hidden_after_mamba3",
    }:
        return (shape_env.sequence_length * shape_env.hidden_size,)
    if canonical_name == "q_fp8":
        return (
            shape_env.sequence_length
            * shape_env.attention_num_q_heads
            * shape_env.attention_head_dim,
        )
    if canonical_name == "q_scale":
        return (shape_env.sequence_length * shape_env.attention_num_q_heads,)
    if canonical_name == "indices":
        return (
            shape_env.sequence_length
            * shape_env.attention_num_kv_heads
            * shape_env.attention_sparse_topk,
        )
    if str(buffer_name).endswith("_grad"):
        if canonical_name == "q_fp8":
            return (
                shape_env.sequence_length
                * shape_env.attention_num_q_heads
                * shape_env.attention_head_dim,
            )
        if canonical_name == "q_scale":
            return (shape_env.sequence_length * shape_env.attention_num_q_heads,)
        if canonical_name == "kv_fp8":
            return (
                shape_env.sequence_length
                * shape_env.attention_num_kv_heads
                * shape_env.attention_head_dim,
            )
        if canonical_name == "kv_scale":
            return (shape_env.sequence_length * shape_env.attention_num_kv_heads,)
    if canonical_name == "kv_fp8":
        return (shape_env.attention_num_kv_heads * shape_env.attention_head_dim,)
    if canonical_name == "kv_scale":
        return (shape_env.attention_num_kv_heads,)
    if canonical_name == "lse":
        return (shape_env.attention_num_q_heads,)
    # Block A: the entry RMSNorm output (and its bwd grad slot) must be
    # full-sequence because the first in-region brick's row-phased body
    # reads the input at ``(row + 1) * H + ...`` (cross-row look-ahead)
    # during the current row's iteration. A row-local scratch would
    # silently alias all rows to the same H slots and corrupt the
    # state recurrence.
    if _is_entry_rmsnorm_output_buffer(buffer_name):
        return (shape_env.sequence_length * shape_env.hidden_size,)
    return (shape_env.hidden_size,)


def _is_entry_rmsnorm_output_buffer(buffer_name: str) -> bool:
    """Return True for the entry RMSNorm forward output / its bwd grad slot."""

    name = str(buffer_name)
    if name.endswith("_grad"):
        name = name[: -len("_grad")]
    return name.endswith("_entry_rmsnorm_hidden")


def _internal_buffer_ref(
    buffer_name: str,
    shape: Sequence[int],
    shape_env: PathCModelShapeEnv | None,
) -> str:
    name = _safe_identifier(buffer_name)
    if shape_env is not None:
        canonical_name = _canonical_buffer_name(buffer_name)
        if str(buffer_name).endswith("_grad"):
            if canonical_name == "q_fp8":
                q_width = shape_env.attention_num_q_heads * shape_env.attention_head_dim
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * {q_width} + "
                    f"(i % {q_width})]"
                )
            if canonical_name == "q_scale":
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * "
                    f"{shape_env.attention_num_q_heads} + "
                    f"((i % {shape_env.hidden_size}) // "
                    f"{shape_env.attention_head_dim})]"
                )
            if canonical_name == "kv_fp8":
                kv_width = (
                    shape_env.attention_num_kv_heads * shape_env.attention_head_dim
                )
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * {kv_width} + "
                    f"(i % {kv_width})]"
                )
            if canonical_name == "kv_scale":
                return (
                    f"{name}[(i // {shape_env.hidden_size}) * "
                    f"{shape_env.attention_num_kv_heads} + "
                    f"(((i % {shape_env.hidden_size}) // "
                    f"{shape_env.attention_head_dim}) % "
                    f"{shape_env.attention_num_kv_heads})]"
                )
        if canonical_name == "q_fp8":
            q_width = shape_env.attention_num_q_heads * shape_env.attention_head_dim
            return (
                f"{name}[(i // {shape_env.hidden_size}) * {q_width} + "
                f"(i % {q_width})]"
            )
        if canonical_name == "q_scale":
            return (
                f"{name}[(i // {shape_env.hidden_size}) * "
                f"{shape_env.attention_num_q_heads} + "
                f"((i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim})]"
            )
        if canonical_name == "indices":
            row_width = (
                shape_env.attention_num_kv_heads
                * shape_env.attention_sparse_topk
            )
            return f"{name}[(i // {shape_env.hidden_size}) * {row_width} + (i % {row_width})]"
        if canonical_name == "q_scale":
            return (
                f"{name}[(i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim}]"
            )
        if canonical_name == "kv_scale":
            return (
                f"{name}[((i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim}) % "
                f"{shape_env.attention_num_kv_heads}]"
            )
        if canonical_name == "kv_fp8":
            return (
                f"{name}[i % "
                f"{shape_env.attention_num_kv_heads * shape_env.attention_head_dim}]"
            )
        if canonical_name == "lse":
            return f"{name}[i % {shape_env.attention_num_q_heads}]"
    if (
        shape_env is not None
        and _flattened_extent(shape)
        == shape_env.sequence_length * shape_env.hidden_size
    ):
        # Block A: full-sequence internal buffer (e.g. entry RMSNorm
        # output). The consumer schedule indexes by absolute ``i``, so
        # the access pattern must NOT wrap modulo H.
        return f"{name}[i]"
    if shape_env is not None and _flattened_extent(shape) == shape_env.hidden_size:
        return f"{name}[i % {shape_env.hidden_size}]"
    return f"{name}[0]"


def _append_descriptor_node_comments(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    indent: str,
) -> None:
    body.append(f"{indent}# {node.name}: {node.op_name}")
    body.append(
        f"{indent}# {node.name} production_fragment_status: "
        f"{descriptor.production_fragment_status}"
    )
    if descriptor.production_fragment_reason:
        body.append(
            f"{indent}# {node.name} production_fragment_reason: "
            f"{descriptor.production_fragment_reason}"
        )


def _append_row_phased_hidden_body(
    body: list[str],
    *,
    nodes: Sequence[_ScheduleNodeView],
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    fragments: Sequence[PathCBrickScheduleFragment],
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv | None,
    train_step_computed_output_buffers: Sequence[str],
    train_step_loss_source_buffers: Sequence[str],
    physical_abi_plan: _PhysicalAbiPlan,
    train_step_loss_cotangent_buffers: Sequence[str],
    train_step_loss_parameter_grad_buffers: Sequence[str],
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
    indent: str,
) -> None:
    if shape_env is None:
        raise ValueError("row_phased_hidden loop policy requires a model shape_env")
    hidden_size = int(shape_env.hidden_size)
    sequence_length = int(shape_env.sequence_length)
    thread_count = min(DESCRIPTOR_DEFAULT_THREADS, max(1, sequence_length * hidden_size))
    chunked_rows = max_rows_per_launch is not None
    row_loop = (
        "for row in T.serial(row_chunk_start, row_chunk_stop):"
        if chunked_rows
        else f"for row in T.serial(0, {sequence_length}):"
    )
    entry_row_loop = (
        "for row in T.serial(row_chunk_start, entry_row_chunk_stop):"
        if chunked_rows
        else row_loop
    )
    row_chunk_expr = (
        DESCRIPTOR_ROW_CHUNK_INDEX_PARAM
        if row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        else "chunk"
    )
    launcher_subchunked_rows = (
        chunked_rows
        and row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    row_chunk_assumptions: tuple[str, ...] = ()
    if launcher_subchunked_rows:
        chunk_count = max(
            1,
            (sequence_length + int(max_rows_per_launch) - 1)
            // int(max_rows_per_launch),
        )
        subchunk_count = max(
            1,
            (int(max_rows_per_launch) + DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH - 1)
            // DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH,
        )
        row_chunk_assumptions = (
            f"{indent * 2}T.assume({row_chunk_expr} >= 0)",
            f"{indent * 2}T.assume({row_chunk_expr} < {chunk_count})",
            f"{indent * 2}T.assume({DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} >= 0)",
            f"{indent * 2}T.assume({DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} < {subchunk_count})",
        )
        body.append(
            f"{indent * 2}path_c_first_row_launch = T.if_then_else("
            f"{row_chunk_expr} == 0, T.if_then_else("
            f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} == 0, 1, 0), 0)"
        )
    fwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in zip(
            nodes, descriptors, fragments, strict=True
        )
        if not node.op_name.endswith("_bwd")
    )
    bwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in zip(
            nodes, descriptors, fragments, strict=True
        )
        if node.op_name.endswith("_bwd")
    )
    for node, _descriptor, _fragment in fwd_items:
        if node.op_name != "residual_rmsnorm":
            continue
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
    for node, _descriptor, _fragment in fwd_items:
        if node.op_name != "entry_rmsnorm":
            continue
        # Block A: entry RMSNorm reuses the same row-phased reduction
        # scratch layout as residual_rmsnorm. The only difference is the
        # forward math operates on a single ``hidden`` input rather than
        # ``hidden + delta``.
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
    for node, descriptor, _fragment in fwd_items:
        if _is_row_phased_mamba3(node, descriptor, shape_env):
            _append_row_phased_mamba3_init(
                body,
                node=node,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
                launcher_chunked_rows=launcher_subchunked_rows,
            )
    for node, descriptor, _fragment in fwd_items:
        if _is_row_phased_m2rnn(node, descriptor, shape_env):
            _append_row_phased_m2rnn_init(
                body,
                node=node,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
                launcher_chunked_rows=launcher_subchunked_rows,
            )
    for node, _descriptor, _fragment in fwd_items:
        if _is_row_phased_sparse_mla_fp8_apply(node, shape_env):
            q_dim = (
                int(shape_env.attention_num_q_heads)
                * int(shape_env.attention_head_dim)
            )
            body.append(
                f"{indent * 2}{_scratch_name(node, 'context_values')} = "
                f"T.alloc_shared(({q_dim},), \"float32\")"
            )
    for node, _descriptor, _fragment in bwd_items:
        if not _is_row_phased_residual_rmsnorm_bwd(node, _descriptor):
            continue
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_norm_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_total_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
    for node, _descriptor, _fragment in bwd_items:
        if not _is_row_phased_entry_rmsnorm_bwd(node, _descriptor):
            continue
        # Block A: entry RMSNorm bwd reuses the same per-row scratch
        # layout as residual_rmsnorm_bwd. The recompute math is slightly
        # cheaper because the forward sum-of-squares is taken over
        # ``hidden`` alone (no ``+ delta`` term), but the reduction
        # bookkeeping is identical.
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot_partial')} = "
            f"T.alloc_shared(({thread_count},), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot')} = "
            "T.alloc_shared((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_norm_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_total_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
    # Block A: entry RMSNorm produces a full-sequence internal buffer that
    # downstream bricks (mamba3 in particular) read with cross-row
    # look-ahead during their row-phased pass. To make those look-aheads
    # well-defined we materialize the full-sequence entry RMSNorm output
    # in its OWN pre-pass row loop BEFORE the main brick row loop starts.
    entry_rmsnorm_fwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in fwd_items
        if node.op_name == "entry_rmsnorm"
    )
    other_fwd_items = tuple(
        (node, descriptor, fragment)
        for node, descriptor, fragment in fwd_items
        if node.op_name != "entry_rmsnorm"
    )

    def append_forward_phase(lines: Sequence[str]) -> None:
        if not lines:
            return
        if bwd_items:
            body.append(f"{indent * 2}if {DESCRIPTOR_BACKWARD_GATE_PARAM} != 1:")
            body.extend(f"{indent}{line}" for line in lines)
            return
        body.extend(lines)

    if entry_rmsnorm_fwd_items:
        body.append(
            f"{indent * 2}# entry_rmsnorm_policy: full_sequence_prepass"
        )
        if chunked_rows:
            body.append(f"{indent * 2}# row_dispatch_policy: chunked_rows")
            body.extend(row_chunk_assumptions)
            if launcher_subchunked_rows:
                body.append(
                    f"{indent * 2}logical_row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}logical_row_chunk_stop = "
                    f"T.min(logical_row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_start = "
                    f"T.min(logical_row_chunk_start + "
                    f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} * "
                    f"{DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH}, "
                    "logical_row_chunk_stop)"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_stop = "
                    f"T.min(subchunk_row_chunk_start + "
                    f"{DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH}, "
                    "logical_row_chunk_stop)"
                )
                if bwd_items:
                    body.append(
                        f"{indent * 2}row_chunk_start = T.if_then_else("
                        f"{DESCRIPTOR_BACKWARD_GATE_PARAM} == 1, 0, "
                        f"T.if_then_else({DESCRIPTOR_BACKWARD_GATE_PARAM} != 0, "
                        "logical_row_chunk_start, subchunk_row_chunk_start))"
                    )
                    body.append(
                        f"{indent * 2}row_chunk_stop = T.if_then_else("
                        f"{DESCRIPTOR_BACKWARD_GATE_PARAM} == 1, "
                        f"{sequence_length}, "
                        f"T.if_then_else({DESCRIPTOR_BACKWARD_GATE_PARAM} != 0, "
                        "logical_row_chunk_stop, subchunk_row_chunk_stop))"
                    )
                else:
                    body.append(f"{indent * 2}row_chunk_start = subchunk_row_chunk_start")
                    body.append(f"{indent * 2}row_chunk_stop = subchunk_row_chunk_stop")
            else:
                body.append(
                    f"{indent * 2}row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}row_chunk_stop = T.min(row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
            body.append(
                f"{indent * 2}entry_row_chunk_stop = "
                f"T.min(row_chunk_stop + 1, {sequence_length})"
            )
        entry_forward_body: list[str] = []
        entry_forward_body.append(
            f"{indent * 2}{entry_row_loop}"
        )
        for node, descriptor, _fragment in entry_rmsnorm_fwd_items:
            _append_row_phased_entry_rmsnorm_body(
                entry_forward_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
        append_forward_phase(entry_forward_body)
    if other_fwd_items:
        if chunked_rows and not entry_rmsnorm_fwd_items:
            body.append(f"{indent * 2}# row_dispatch_policy: chunked_rows")
            body.extend(row_chunk_assumptions)
            if launcher_subchunked_rows:
                body.append(
                    f"{indent * 2}logical_row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}logical_row_chunk_stop = "
                    f"T.min(logical_row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_start = "
                    f"T.min(logical_row_chunk_start + "
                    f"{DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM} * "
                    f"{DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH}, "
                    "logical_row_chunk_stop)"
                )
                body.append(
                    f"{indent * 2}subchunk_row_chunk_stop = "
                    f"T.min(subchunk_row_chunk_start + "
                    f"{DESCRIPTOR_ROWS_PER_KERNEL_LAUNCH}, "
                    "logical_row_chunk_stop)"
                )
                if bwd_items:
                    body.append(
                        f"{indent * 2}row_chunk_start = T.if_then_else("
                        f"{DESCRIPTOR_BACKWARD_GATE_PARAM} == 1, 0, "
                        f"T.if_then_else({DESCRIPTOR_BACKWARD_GATE_PARAM} != 0, "
                        "logical_row_chunk_start, subchunk_row_chunk_start))"
                    )
                    body.append(
                        f"{indent * 2}row_chunk_stop = T.if_then_else("
                        f"{DESCRIPTOR_BACKWARD_GATE_PARAM} == 1, "
                        f"{sequence_length}, "
                        f"T.if_then_else({DESCRIPTOR_BACKWARD_GATE_PARAM} != 0, "
                        "logical_row_chunk_stop, subchunk_row_chunk_stop))"
                    )
                else:
                    body.append(f"{indent * 2}row_chunk_start = subchunk_row_chunk_start")
                    body.append(f"{indent * 2}row_chunk_stop = subchunk_row_chunk_stop")
            else:
                body.append(
                    f"{indent * 2}row_chunk_start = "
                    f"{row_chunk_expr} * {max_rows_per_launch}"
                )
                body.append(
                    f"{indent * 2}row_chunk_stop = T.min(row_chunk_start + "
                    f"{max_rows_per_launch}, {sequence_length})"
                )
        other_forward_body: list[str] = []
        other_forward_body.append(f"{indent * 2}{row_loop}")
        for node, descriptor, fragment in other_fwd_items:
            if _is_row_phased_mamba3(node, descriptor, shape_env):
                _append_row_phased_mamba3_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                    launcher_chunked_rows=launcher_subchunked_rows,
                )
                continue
            if node.op_name == "residual_rmsnorm":
                _append_row_phased_residual_rmsnorm_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    hidden_size=hidden_size,
                    thread_count=thread_count,
                    indent=indent,
                )
                continue
            if _is_row_phased_attention_qkv_projection(node, shape_env):
                _append_row_phased_attention_qkv_projection_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                )
                continue
            if _is_row_phased_m2rnn(node, descriptor, shape_env):
                _append_row_phased_m2rnn_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                    launcher_chunked_rows=launcher_subchunked_rows,
                )
                continue
            if _is_row_phased_sparse_mla_fp8_apply(node, shape_env):
                _append_row_phased_sparse_mla_fp8_apply_body(
                    other_forward_body,
                    node=node,
                    descriptor=descriptor,
                    dtype_by_buffer=dtype_by_buffer,
                    access_by_buffer=access_by_buffer,
                    shape_env=shape_env,
                    thread_count=thread_count,
                    indent=indent,
                )
                continue
            other_forward_body.append(
                f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
                f"(row + 1) * {hidden_size}, step={thread_count}):"
            )
            _append_descriptor_node_comments(
                other_forward_body,
                node=node,
                descriptor=descriptor,
                indent=indent * 4,
            )
            for statement in fragment.statements:
                other_forward_body.append(f"{indent * 4}{statement}")
            other_forward_body.append(f"{indent * 3}T.sync_threads()")
        append_forward_phase(other_forward_body)
    if not bwd_items:
        _append_train_step_suffix_scalar_outputs(
            body,
            computed_outputs=train_step_computed_output_buffers,
            loss_source_buffers=train_step_loss_source_buffers,
            train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
            train_step_loss_parameter_grad_buffers=(
                train_step_loss_parameter_grad_buffers
            ),
            physical_abi_plan=physical_abi_plan,
            loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
            shape_env=shape_env,
            indent=indent,
        )
        return
    bwd_body: list[str] = []
    bwd_once_body: list[str] = []
    _append_train_step_suffix_scalar_outputs(
        bwd_once_body,
        computed_outputs=train_step_computed_output_buffers,
        loss_source_buffers=train_step_loss_source_buffers,
        train_step_loss_cotangent_buffers=train_step_loss_cotangent_buffers,
        train_step_loss_parameter_grad_buffers=(
            train_step_loss_parameter_grad_buffers
        ),
        physical_abi_plan=physical_abi_plan,
        loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        shape_env=shape_env,
        indent=indent,
    )
    row_phased_bwd_items = tuple(
        (node, descriptor)
        for node, descriptor, _fragment in bwd_items
        if _is_row_phased_bwd_descriptor(node, descriptor, shape_env)
    )
    if not row_phased_bwd_items:
        _reject_proxy_backward_fragments(bwd_items)
        if bwd_once_body:
            if launcher_subchunked_rows:
                bwd_body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
                bwd_body.extend(f"{indent}{line}" for line in bwd_once_body)
            else:
                bwd_body.extend(bwd_once_body)
        bwd_body.append(
            f"{indent * 2}# backward_policy: flat_after_row_phased_forward"
        )
        bwd_body.append(
            f"{indent * 2}for i in T.serial(lane, "
            f"{sequence_length * hidden_size}, step={thread_count}):"
        )
        for node, descriptor, fragment in bwd_items:
            _append_descriptor_node_comments(
                bwd_body,
                node=node,
                descriptor=descriptor,
                indent=indent * 3,
            )
            for statement in fragment.statements:
                bwd_body.append(f"{indent * 3}{statement}")
            bwd_body.append(f"{indent * 3}T.sync_threads()")
        body.append(f"{indent * 2}if {DESCRIPTOR_BACKWARD_GATE_PARAM} == 1:")
        body.extend(f"{indent}{line}" for line in bwd_body)
        return
    for node, _descriptor, _fragment in bwd_items:
        if _is_row_phased_residual_rmsnorm_bwd(node, _descriptor):
            _append_row_phased_residual_rmsnorm_bwd_init(
                bwd_once_body,
                node=node,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
    for node, _descriptor, _fragment in bwd_items:
        if _is_row_phased_entry_rmsnorm_bwd(node, _descriptor):
            _append_row_phased_entry_rmsnorm_bwd_init(
                bwd_once_body,
                node=node,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                thread_count=thread_count,
                indent=indent,
            )
    if bwd_once_body:
        if launcher_subchunked_rows:
            bwd_body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
            bwd_body.extend(f"{indent}{line}" for line in bwd_once_body)
        else:
            bwd_body.extend(bwd_once_body)
    def append_row_phased_bwd_item(
        node: _ScheduleNodeView,
        descriptor: PathCBrickScheduleDescriptor,
        fragment: Any,
    ) -> None:
        if _is_row_phased_residual_rmsnorm_bwd(node, descriptor):
            _append_row_phased_residual_rmsnorm_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
            return
        if _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env):
            _append_row_phased_sparse_mla_fp8_apply_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                chunked_rows=chunked_rows,
                indent=indent,
            )
            return
        if _is_row_phased_attention_qkv_projection_bwd(node, descriptor, shape_env):
            _append_row_phased_attention_qkv_projection_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
            )
            return
        if _is_row_phased_m2rnn_bwd(node, descriptor, shape_env):
            _append_row_phased_m2rnn_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
            )
            return
        if _is_row_phased_mamba3_bwd(node, descriptor, shape_env):
            _append_row_phased_mamba3_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                shape_env=shape_env,
                thread_count=thread_count,
                indent=indent,
            )
            return
        if _is_row_phased_entry_rmsnorm_bwd(node, descriptor):
            _append_row_phased_entry_rmsnorm_bwd_body(
                bwd_body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                thread_count=thread_count,
                indent=indent,
            )
            return
        _reject_proxy_backward_fragments(((node, descriptor, fragment),))
        bwd_body.append(
            f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
            f"(row + 1) * {hidden_size}, step={thread_count}):"
        )
        _append_descriptor_node_comments(
            bwd_body,
            node=node,
            descriptor=descriptor,
            indent=indent * 4,
        )
        for statement in fragment.statements:
            bwd_body.append(f"{indent * 4}{statement}")
        bwd_body.append(f"{indent * 3}T.sync_threads()")

    bwd_row_loop = (
        f"for row in T.serial(0, {sequence_length}):"
        if launcher_subchunked_rows
        else row_loop
    )
    bwd_body.append(f"{indent * 2}# backward_policy: row_phased_hidden_recompute")
    bwd_body.append(f"{indent * 2}# backward_phase: full_sparse_mla_before_projection")
    for node, descriptor, fragment in bwd_items:
        if not _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env):
            continue
        bwd_body.append(f"{indent * 2}{bwd_row_loop}")
        append_row_phased_bwd_item(node, descriptor, fragment)
    bwd_body.append(f"{indent * 2}# backward_phase: projection_and_upstream")
    for node, descriptor, fragment in bwd_items:
        if _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env):
            continue
        bwd_body.append(f"{indent * 2}{bwd_row_loop}")
        append_row_phased_bwd_item(node, descriptor, fragment)
    body.append(f"{indent * 2}if {DESCRIPTOR_BACKWARD_GATE_PARAM} == 1:")
    body.extend(f"{indent}{line}" for line in bwd_body)


def _append_lane0_row_phase(
    body: list[str],
    *,
    indent: str,
    append_fn: Callable[..., None],
    **kwargs: Any,
) -> None:
    lane0_body: list[str] = []
    append_fn(lane0_body, indent=indent, **kwargs)
    body.append(f"{indent * 3}if lane == 0:")
    body.extend(f"{indent}{line}" for line in lane0_body)
    body.append(f"{indent * 3}T.sync_threads()")


def _is_row_phased_residual_rmsnorm_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
) -> bool:
    return (
        node.op_name == "residual_rmsnorm_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_entry_rmsnorm(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "entry_rmsnorm"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_entry_rmsnorm_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
) -> bool:
    return (
        node.op_name == "entry_rmsnorm_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_attention_qkv_projection_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "attention_qkv_projection_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_sparse_mla_fp8_apply_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    if (
        shape_env is None
        or node.op_name != "sparse_mla_fp8_apply_bwd"
        or descriptor.production_fragment_status != "production_region_inlined"
    ):
        return False
    input_canonicals = {_canonical_buffer_name(input_name) for input_name in node.inputs}
    output_canonicals = {_canonical_buffer_name(output) for output in node.outputs}
    return {"q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"}.issubset(
        input_canonicals
    ) and {"q_fp8", "kv_fp8"}.issubset(output_canonicals)


def _is_row_phased_m2rnn_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "m2rnn_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_mamba3_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "mamba3_mimo_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_bwd_descriptor(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        _is_row_phased_residual_rmsnorm_bwd(node, descriptor)
        or _is_row_phased_sparse_mla_fp8_apply_bwd(node, descriptor, shape_env)
        or _is_row_phased_attention_qkv_projection_bwd(node, descriptor, shape_env)
        or _is_row_phased_m2rnn_bwd(node, descriptor, shape_env)
        or _is_row_phased_mamba3_bwd(node, descriptor, shape_env)
        or _is_row_phased_entry_rmsnorm_bwd(node, descriptor)
    )


def _reject_proxy_backward_fragments(
    items: Sequence[
        tuple[
            _ScheduleNodeView,
            PathCBrickScheduleDescriptor,
            PathCBrickScheduleFragment,
        ]
    ],
) -> None:
    for node, descriptor, _fragment in items:
        if node.op_name not in _EXACT_ROW_PHASED_BACKWARD_OPS:
            continue
        raise ValueError(
            f"{node.op_name} requires the row-phased exact backward generator; "
            "refusing to emit a scalar proxy backward fragment"
        )


def _is_row_phased_attention_qkv_projection(
    node: _ScheduleNodeView,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    if shape_env is None or node.op_name != "attention_qkv_projection":
        return False
    output_canonicals = {_canonical_buffer_name(output) for output in node.outputs}
    return {"q_fp8", "q_scale", "kv_fp8", "kv_scale"}.issubset(output_canonicals)


def _is_row_phased_sparse_mla_fp8_apply(
    node: _ScheduleNodeView,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    if shape_env is None or node.op_name != "sparse_mla_fp8_apply":
        return False
    input_canonicals = {_canonical_buffer_name(input_name) for input_name in node.inputs}
    return {"q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"}.issubset(
        input_canonicals
    )


def _is_row_phased_mamba3(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "mamba3_mimo"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _is_row_phased_m2rnn(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    shape_env: PathCModelShapeEnv | None,
) -> bool:
    return (
        shape_env is not None
        and node.op_name == "m2rnn"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _node_indexed_canonical_input_expr(
    node: _ScheduleNodeView,
    canonical_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index_expr: str,
    *,
    default: str = "0.0",
) -> str:
    input_name = _node_input_for_canonical(node, canonical_name)
    if input_name is None:
        return default
    return _indexed_buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        index_expr,
    )


def _node_indexed_canonical_or_positional_input_expr(
    node: _ScheduleNodeView,
    canonical_names: Sequence[str],
    positional_index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index_expr: str,
    *,
    default: str = "0.0",
) -> str:
    input_name: str | None = None
    for canonical_name in canonical_names:
        input_name = _node_input_for_canonical(node, canonical_name)
        if input_name is not None:
            break
    if input_name is None:
        if positional_index >= len(node.inputs):
            return default
        input_name = node.inputs[positional_index]
    return _indexed_buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        index_expr,
    )


def _node_indexed_positional_input_expr(
    node: _ScheduleNodeView,
    input_index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index_expr: str,
    *,
    default: str = "0.0",
) -> str:
    if input_index >= len(node.inputs):
        return default
    input_name = node.inputs[input_index]
    return _indexed_buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        index_expr,
    )


def _mamba3_state_output(node: _ScheduleNodeView) -> str | None:
    output = _node_output_for_canonical(node, "scan_state")
    if output is not None:
        return output
    output = _node_output_for_canonical(node, "mamba_state")
    if output is not None:
        return output
    for output_name in node.outputs:
        if output_name.endswith("_state") or output_name.endswith("_state_out"):
            return output_name
    return node.outputs[1] if len(node.outputs) > 1 else None


def _append_row_phased_mamba3_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    projected = _scratch_name(node, "mamba3_projected_vec")
    conv_history = _scratch_name(node, "mamba3_conv_history")
    conv = _scratch_name(node, "mamba3_conv_vec")
    b_inv = _scratch_name(node, "mamba3_b_inv_rms")
    c_inv = _scratch_name(node, "mamba3_c_inv_rms")
    b_group = _scratch_name(node, "mamba3_b_group")
    c_group = _scratch_name(node, "mamba3_c_group")
    b_raw = _scratch_name(node, "mamba3_b_raw")
    c_raw = _scratch_name(node, "mamba3_c_raw")
    dt_vec = _scratch_name(node, "mamba3_dt_vec")
    a_vec = _scratch_name(node, "mamba3_a_vec")
    trap_group = _scratch_name(node, "mamba3_trap_group")
    next_dt = _scratch_name(node, "mamba3_next_dt")
    next_trap = _scratch_name(node, "mamba3_next_trap")
    angle_cumsum = _scratch_name(node, "mamba3_angle_cumsum")
    out_inner = _scratch_name(node, "mamba3_out_inner")
    accum = _scratch_name(node, "mamba3_accum")
    state_value = _scratch_name(node, "mamba3_state_value")
    inner_dim = int(shape_env.mamba_inner_dim)
    in_proj_dim = int(shape_env.mamba_in_proj_dim)
    conv_channels = int(shape_env.mamba_conv_channels)
    history_len = max(0, int(shape_env.mamba_conv_kernel) - 1)
    heads = int(shape_env.mamba_num_heads)
    head_dim = int(shape_env.mamba_head_dim)
    state_dim = int(shape_env.mamba_state_dim)
    groups = int(shape_env.mamba_groups)
    rank = int(shape_env.mamba_effective_mimo_rank)
    rope_angles = int(shape_env.mamba_num_rope_angles)
    state_output = _mamba3_state_output(node)
    body.append(
        f"{indent * 2}{projected} = T.alloc_shared(({in_proj_dim},), \"float32\")"
    )
    body.append(
        f"{indent * 2}{conv} = T.alloc_shared(({conv_channels},), \"float32\")"
    )
    body.append(f"{indent * 2}{out_inner} = T.alloc_shared(({inner_dim},), \"float32\")")
    body.append(
        f"{indent * 2}{b_inv} = T.alloc_shared(({rank}, {groups}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{c_inv} = T.alloc_shared(({rank}, {groups}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{b_raw} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{c_raw} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{b_group} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{c_group} = T.alloc_shared(({groups}, {state_dim}), \"float32\")"
    )
    body.append(f"{indent * 2}{dt_vec} = T.alloc_shared(({heads},), \"float32\")")
    body.append(f"{indent * 2}{a_vec} = T.alloc_shared(({heads},), \"float32\")")
    body.append(f"{indent * 2}{trap_group} = T.alloc_shared(({groups},), \"float32\")")
    body.append(f"{indent * 2}{next_dt} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{next_trap} = T.alloc_local((1,), \"float32\")")
    body.append(
        f"{indent * 2}{angle_cumsum} = "
        f"T.alloc_shared(({heads}, {rope_angles}), \"float32\")"
    )
    if history_len > 0:
        body.append(
            f"{indent * 2}{conv_history} = "
            f"T.alloc_shared(({history_len}, {conv_channels}), \"float32\")"
        )
    body.append(f"{indent * 2}{accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{state_value} = T.alloc_local((1,), \"float32\")")
    head = _scratch_name(node, "head_init")
    dim = _scratch_name(node, "dim_init")
    state = _scratch_name(node, "state_idx_init")
    angle = _scratch_name(node, "angle_init")
    hist = _scratch_name(node, "hist_init")
    ch = _scratch_name(node, "conv_ch_init")
    state_flat = _scratch_name(node, "state_flat_init")
    angle_flat = _scratch_name(node, "angle_flat_init")
    history_flat = _scratch_name(node, "history_flat_init")
    body.append(f"{indent * 2}# {node.name}: mamba3_state_policy: external_scan_state")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        body.append(
            f"{indent * 3}for {angle_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{head} = {angle_flat} // {rope_angles}")
        body.append(f"{indent * 4}{angle} = {angle_flat} % {rope_angles}")
        body.append(f"{indent * 4}{angle_cumsum}[{head}, {angle}] = 0.0")
        body.append(f"{indent * 2}else:")
        body.append(
            f"{indent * 3}for {angle_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{head} = {angle_flat} // {rope_angles}")
        body.append(f"{indent * 4}{angle} = {angle_flat} % {rope_angles}")
        angle_state_ref = _buffer_ref(
            "mamba3_angle_state",
            access_by_buffer,
            f"{head} * {rope_angles} + {angle}",
        )
        body.append(
            f"{indent * 4}{angle_cumsum}[{head}, {angle}] = {angle_state_ref}"
        )
    else:
        body.append(
            f"{indent * 2}for {angle_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 3}{head} = {angle_flat} // {rope_angles}")
        body.append(f"{indent * 3}{angle} = {angle_flat} % {rope_angles}")
        body.append(f"{indent * 3}{angle_cumsum}[{head}, {angle}] = 0.0")
    if state_output is not None:
        if launcher_chunked_rows:
            body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
            state_loop_indent = indent * 3
            state_body_indent = indent * 4
        else:
            state_loop_indent = indent * 2
            state_body_indent = indent * 3
        body.append(
            f"{state_loop_indent}for {state_flat} in T.serial(lane, {heads * head_dim * state_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{state_body_indent}{head} = {state_flat} // {head_dim * state_dim}")
        body.append(f"{state_body_indent}{dim} = ({state_flat} // {state_dim}) % {head_dim}")
        body.append(f"{state_body_indent}{state} = {state_flat} % {state_dim}")
        state_idx = f"{head} * {head_dim * state_dim} + {dim} * {state_dim} + {state}"
        h0_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_h0",
            dtype_by_buffer,
            access_by_buffer,
            state_idx,
            default=_node_indexed_positional_input_expr(
                node,
                1,
                dtype_by_buffer,
                access_by_buffer,
                state_idx,
            ),
        )
        body.append(
            f"{state_body_indent}{_buffer_ref(state_output, access_by_buffer, state_idx)} = "
            f"{h0_expr}"
        )
    body.append(f"{indent * 2}T.sync_threads()")
    if history_len <= 0:
        return
    body.append(f"{indent * 2}# {node.name}: mamba3_conv_policy: zero_padded_ring_history")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {history_flat} // {conv_channels}")
        body.append(f"{indent * 4}{ch} = {history_flat} % {conv_channels}")
        body.append(f"{indent * 4}{conv_history}[{hist}, {ch}] = 0.0")
        body.append(f"{indent * 2}else:")
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {history_flat} // {conv_channels}")
        body.append(f"{indent * 4}{ch} = {history_flat} % {conv_channels}")
        conv_state_ref = _buffer_ref(
            "mamba3_conv_state",
            access_by_buffer,
            f"{hist} * {conv_channels} + {ch}",
        )
        body.append(
            f"{indent * 4}{conv_history}[{hist}, {ch}] = {conv_state_ref}"
        )
    else:
        body.append(
            f"{indent * 2}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        body.append(f"{indent * 3}{hist} = {history_flat} // {conv_channels}")
        body.append(f"{indent * 3}{ch} = {history_flat} % {conv_channels}")
        body.append(f"{indent * 3}{conv_history}[{hist}, {ch}] = 0.0")
    body.append(f"{indent * 2}T.sync_threads()")


def _append_row_phased_mamba3_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    projected = _scratch_name(node, "mamba3_projected_vec")
    conv_history = _scratch_name(node, "mamba3_conv_history")
    conv = _scratch_name(node, "mamba3_conv_vec")
    b_inv = _scratch_name(node, "mamba3_b_inv_rms")
    c_inv = _scratch_name(node, "mamba3_c_inv_rms")
    b_group = _scratch_name(node, "mamba3_b_group")
    c_group = _scratch_name(node, "mamba3_c_group")
    b_raw = _scratch_name(node, "mamba3_b_raw")
    c_raw = _scratch_name(node, "mamba3_c_raw")
    dt_vec = _scratch_name(node, "mamba3_dt_vec")
    a_vec = _scratch_name(node, "mamba3_a_vec")
    trap_group = _scratch_name(node, "mamba3_trap_group")
    next_dt = _scratch_name(node, "mamba3_next_dt")
    next_trap = _scratch_name(node, "mamba3_next_trap")
    angle_cumsum = _scratch_name(node, "mamba3_angle_cumsum")
    out_inner = _scratch_name(node, "mamba3_out_inner")
    accum = _scratch_name(node, "mamba3_accum")
    state_value = _scratch_name(node, "mamba3_state_value")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim_loop = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    kernel_pos = _scratch_name(node, "kernel_pos")
    head = _scratch_name(node, "head")
    state = _scratch_name(node, "state_idx")
    rank = _scratch_name(node, "rank")
    angle = _scratch_name(node, "angle")
    feature = _scratch_name(node, "feature")
    out_dim = _scratch_name(node, "out_dim")
    rank_group_flat = _scratch_name(node, "rank_group_flat")
    group_state_flat = _scratch_name(node, "group_state_flat")
    history_flat = _scratch_name(node, "history_flat")
    angle_carry_flat = _scratch_name(node, "angle_carry_flat")
    angle_carry_head = _scratch_name(node, "angle_carry_head")
    angle_carry_idx = _scratch_name(node, "angle_carry_idx")
    trap_group_loop = _scratch_name(node, "trap_group_loop")
    hidden_size = int(shape_env.hidden_size)
    inner_dim = int(shape_env.mamba_inner_dim)
    in_proj_dim = int(shape_env.mamba_in_proj_dim)
    conv_channels = int(shape_env.mamba_conv_channels)
    kernel = int(shape_env.mamba_conv_kernel)
    history_len = max(0, kernel - 1)
    heads = int(shape_env.mamba_num_heads)
    head_dim = int(shape_env.mamba_head_dim)
    state_dim = int(shape_env.mamba_state_dim)
    groups = int(shape_env.mamba_groups)
    mimo_rank = int(shape_env.mamba_effective_mimo_rank)
    rope_angles = int(shape_env.mamba_num_rope_angles)
    bc_dim = int(shape_env.mamba_bc_dim)
    heads_per_group = heads // groups
    z_offset = 0
    x_offset = inner_dim
    b_offset = 2 * inner_dim
    c_offset = b_offset + bc_dim
    dt_offset = c_offset + bc_dim
    a_offset = dt_offset + heads
    trap_offset = a_offset + heads
    angle_offset = trap_offset + heads
    conv_b_offset = inner_dim
    conv_c_offset = inner_dim + bc_dim
    rot_dim = min(state_dim, 2 * rope_angles)
    delta = _output_with_suffix(node, "_delta") or (node.outputs[0] if node.outputs else "")
    state_output = _mamba3_state_output(node)
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(f"{indent * 3}# mamba3_projection_policy: dense_row_local")
    body.append(
        f"{indent * 3}for {proj_dim} in T.serial(lane, {in_proj_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{accum}[0] = 0.0")
    body.append(f"{indent * 4}for {hidden_dim_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_positional_input_expr(
        node,
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {hidden_dim_loop}",
    )
    in_proj_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{proj_dim} * {hidden_size} + {hidden_dim_loop}",
    )
    body.append(
        f"{indent * 5}{accum}[0] = {accum}[0] + "
        f"({hidden_expr} * {in_proj_weight_expr})"
    )
    body.append(f"{indent * 4}{projected}[{proj_dim}] = {accum}[0]")
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# mamba3_conv_policy: causal_depthwise_ring_history")
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_channels}, "
        f"step={thread_count}):"
    )
    conv_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        conv_ch,
    )
    body.append(f"{indent * 4}{conv}[{conv_ch}] = {conv_bias_expr}")
    if history_len > 0:
        body.append(f"{indent * 4}for {kernel_pos} in T.serial(0, {history_len}):")
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
            f"({conv_history}[{kernel_pos}, {conv_ch}] * {conv_weight_expr})"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
        f"({projected}[{x_offset} + {conv_ch}] * {current_conv_weight_expr})"
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] * "
        f"(1.0 / (1.0 + T.exp(-{conv}[{conv_ch}])))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# mamba3_dt_policy: softplus_A_trapezoid")
    body.append(
        f"{indent * 3}for {head} in T.serial(lane, {heads}, step={thread_count}):"
    )
    dt_bias = _node_indexed_canonical_input_expr(
        node,
        "mamba3_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
    )
    body.append(
        f"{indent * 4}{dt_vec}[{head}] = T.log(1.0 + "
        f"T.exp({projected}[{dt_offset} + {head}] + {dt_bias}))"
    )
    body.append(
        f"{indent * 4}{a_vec}[{head}] = T.min(-T.log(1.0 + "
        f"T.exp({projected}[{a_offset} + {head}])), -0.01)"
    )
    body.append(f"{indent * 4}for {angle} in T.serial(0, {rope_angles}):")
    body.append(
        f"{indent * 5}{angle_cumsum}[{head}, {angle}] = "
        f"{angle_cumsum}[{head}, {angle}] + "
        f"({projected}[{angle_offset} + {angle}] * {dt_vec}[{head}])"
    )
    if launcher_chunked_rows:
        body.append(
            f"{indent * 3}for {angle_carry_flat} in T.serial(lane, {heads * rope_angles}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{angle_carry_head} = {angle_carry_flat} // {rope_angles}")
        body.append(f"{indent * 4}{angle_carry_idx} = {angle_carry_flat} % {rope_angles}")
        angle_state_ref = _buffer_ref(
            "mamba3_angle_state",
            access_by_buffer,
            f"{angle_carry_head} * {rope_angles} + {angle_carry_idx}",
        )
        body.append(
            f"{indent * 4}{angle_state_ref} = "
            f"{angle_cumsum}[{angle_carry_head}, {angle_carry_idx}]"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {trap_group_loop} in T.serial(lane, {groups}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{trap_group}[{trap_group_loop}] = 0.0")
    body.append(f"{indent * 4}for {head} in T.serial(0, {heads_per_group}):")
    body.append(
        f"{indent * 5}{accum}[0] = "
        f"{trap_group_loop} * {heads_per_group} + {head}"
    )
    body.append(f"{indent * 5}{next_dt}[0] = 0.0")
    body.append(f"{indent * 5}{next_trap}[0] = 0.0")
    body.append(f"{indent * 5}if row + 1 < {int(shape_env.sequence_length)}:")
    body.append(f"{indent * 6}for {hidden_dim_loop} in T.serial(0, {hidden_size}):")
    next_hidden_expr = _node_indexed_positional_input_expr(
        node,
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"(row + 1) * {hidden_size} + {hidden_dim_loop}",
    )
    next_dt_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({dt_offset} + T.cast({accum}[0], \"int32\")) * {hidden_size} + "
        f"{hidden_dim_loop}",
    )
    next_trap_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({trap_offset} + T.cast({accum}[0], \"int32\")) * {hidden_size} + "
        f"{hidden_dim_loop}",
    )
    body.append(
        f"{indent * 7}{next_dt}[0] = {next_dt}[0] + "
        f"({next_hidden_expr} * {next_dt_weight_expr})"
    )
    body.append(
        f"{indent * 7}{next_trap}[0] = {next_trap}[0] + "
        f"({next_hidden_expr} * {next_trap_weight_expr})"
    )
    next_dt_bias = _node_indexed_canonical_input_expr(
        node,
        "mamba3_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        f'T.cast({accum}[0], "int32")',
    )
    body.append(
        f"{indent * 6}{next_dt}[0] = T.log(1.0 + "
        f"T.exp({next_dt}[0] + {next_dt_bias}))"
    )
    body.append(
        f"{indent * 5}{trap_group}[{trap_group_loop}] = "
        f"{trap_group}[{trap_group_loop}] + "
        f"(({next_dt}[0] * (1.0 - (1.0 / (1.0 + T.exp(-{next_trap}[0]))))) + "
        f"({dt_vec}[T.cast({accum}[0], \"int32\")] * "
        f"(1.0 / (1.0 + T.exp(-{projected}[{trap_offset} + "
        f"T.cast({accum}[0], \"int32\")])))))"
    )
    body.append(
        f"{indent * 4}{trap_group}[{trap_group_loop}] = "
        f"{trap_group}[{trap_group_loop}] / "
        f"{float(heads_per_group):.1f}"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# mamba3_bc_policy: rank_group_rmsnorm_rope")
    body.append(
        f"{indent * 3}for {rank_group_flat} in T.serial(lane, {mimo_rank * groups}, "
        f"step={thread_count}):"
    )
    rank_expr = f"({rank_group_flat} // {groups})"
    group_expr = f"({rank_group_flat} % {groups})"
    body.append(f"{indent * 4}{b_inv}[{rank_expr}, {group_expr}] = 0.0")
    body.append(f"{indent * 4}{c_inv}[{rank_expr}, {group_expr}] = 0.0")
    body.append(f"{indent * 4}for {state} in T.serial(0, {state_dim}):")
    bc_index = f"(({rank_expr} * {groups} + {group_expr}) * {state_dim} + {state})"
    body.append(
        f"{indent * 5}{b_inv}[{rank_expr}, {group_expr}] = "
        f"{b_inv}[{rank_expr}, {group_expr}] + "
        f"({conv}[{inner_dim} + {bc_index}] * {conv}[{inner_dim} + {bc_index}])"
    )
    body.append(
        f"{indent * 5}{c_inv}[{rank_expr}, {group_expr}] = "
        f"{c_inv}[{rank_expr}, {group_expr}] + "
        f"({conv}[{inner_dim + bc_dim} + {bc_index}] * "
        f"{conv}[{inner_dim + bc_dim} + {bc_index}])"
    )
    body.append(
        f"{indent * 4}{b_inv}[{rank_expr}, {group_expr}] = "
        f"T.rsqrt(({b_inv}[{rank_expr}, {group_expr}] / "
        f"{float(state_dim):.1f}) + 0.00001)"
    )
    body.append(
        f"{indent * 4}{c_inv}[{rank_expr}, {group_expr}] = "
        f"T.rsqrt(({c_inv}[{rank_expr}, {group_expr}] / "
        f"{float(state_dim):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {group_state_flat} in T.serial(lane, {groups * state_dim}, "
        f"step={thread_count}):"
    )
    group_expr = f"({group_state_flat} // {state_dim})"
    state_expr = f"({group_state_flat} % {state_dim})"
    body.append(f"{indent * 4}{b_raw}[{group_expr}, {state_expr}] = 0.0")
    body.append(f"{indent * 4}{c_raw}[{group_expr}, {state_expr}] = 0.0")
    body.append(f"{indent * 4}for {rank} in T.serial(0, {mimo_rank}):")
    bc_index = f"(({rank} * {groups} + {group_expr}) * {state_dim} + {state_expr})"
    b_norm_weight = _node_indexed_canonical_input_expr(
        node,
        "mamba3_B_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
        default="1.0",
    )
    b_bias = _node_indexed_canonical_input_expr(
        node,
        "mamba3_B_bias",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
    )
    c_norm_weight = _node_indexed_canonical_input_expr(
        node,
        "mamba3_C_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
        default="1.0",
    )
    c_bias = _node_indexed_canonical_input_expr(
        node,
        "mamba3_C_bias",
        dtype_by_buffer,
        access_by_buffer,
        bc_index,
    )
    body.append(
        f"{indent * 5}{b_raw}[{group_expr}, {state_expr}] = "
        f"{b_raw}[{group_expr}, {state_expr}] + "
        f"(({conv}[{inner_dim} + {bc_index}] * {b_inv}[{rank}, {group_expr}] * "
        f"{b_norm_weight}) + {b_bias})"
    )
    body.append(
        f"{indent * 5}{c_raw}[{group_expr}, {state_expr}] = "
        f"{c_raw}[{group_expr}, {state_expr}] + "
        f"(({conv}[{inner_dim + bc_dim} + {bc_index}] * "
        f"{c_inv}[{rank}, {group_expr}] * {c_norm_weight}) + {c_bias})"
    )
    body.append(
        f"{indent * 4}{b_raw}[{group_expr}, {state_expr}] = "
        f"({b_raw}[{group_expr}, {state_expr}] / {float(mimo_rank):.1f}) * "
        f"{trap_group}[{group_expr}]"
    )
    body.append(
        f"{indent * 4}{c_raw}[{group_expr}, {state_expr}] = "
        f"{c_raw}[{group_expr}, {state_expr}] / {float(mimo_rank):.1f}"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {group_state_flat} in T.serial(lane, {groups * state_dim}, "
        f"step={thread_count}):"
    )
    group_expr = f"({group_state_flat} // {state_dim})"
    state_expr = f"({group_state_flat} % {state_dim})"
    angle_expr = f"({state_expr} // 2)"
    body.append(f"{indent * 4}if {state_expr} < {rot_dim}:")
    body.append(f"{indent * 5}if ({state_expr} % 2) == 0:")
    body.append(
        f"{indent * 6}{b_group}[{group_expr}, {state_expr}] = "
        f"({b_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}])) - "
        f"({b_raw}[{group_expr}, {state_expr} + 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(
        f"{indent * 6}{c_group}[{group_expr}, {state_expr}] = "
        f"({c_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}])) - "
        f"({c_raw}[{group_expr}, {state_expr} + 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(f"{indent * 5}else:")
    body.append(
        f"{indent * 6}{b_group}[{group_expr}, {state_expr}] = "
        f"({b_raw}[{group_expr}, {state_expr} - 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}])) + "
        f"({b_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(
        f"{indent * 6}{c_group}[{group_expr}, {state_expr}] = "
        f"({c_raw}[{group_expr}, {state_expr} - 1] * "
        f"T.sin({angle_cumsum}[{group_expr}, {angle_expr}])) + "
        f"({c_raw}[{group_expr}, {state_expr}] * "
        f"T.cos({angle_cumsum}[{group_expr}, {angle_expr}]))"
    )
    body.append(f"{indent * 4}else:")
    body.append(
        f"{indent * 5}{b_group}[{group_expr}, {state_expr}] = "
        f"{b_raw}[{group_expr}, {state_expr}]"
    )
    body.append(
        f"{indent * 5}{c_group}[{group_expr}, {state_expr}] = "
        f"{c_raw}[{group_expr}, {state_expr}]"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    if state_output is not None:
        body.append(f"{indent * 3}# mamba3_scan_policy: external_state_recurrence")
        body.append(
            f"{indent * 3}for {feature} in T.serial(lane, {inner_dim}, "
            f"step={thread_count}):"
        )
        head_expr = f"({feature} // {head_dim})"
        dim_expr = f"({feature} % {head_dim})"
        group_expr = f"({head_expr} // {heads_per_group})"
        body.append(f"{indent * 4}{out_inner}[{feature}] = 0.0")
        body.append(f"{indent * 4}for {state} in T.serial(0, {state_dim}):")
        state_idx = f"{head_expr} * {head_dim * state_dim} + {dim_expr} * {state_dim} + {state}"
        state_ref = _buffer_ref(state_output, access_by_buffer, state_idx)
        body.append(
            f"{indent * 5}{state_value}[0] = "
            f"(T.exp({a_vec}[{head_expr}] * {dt_vec}[{head_expr}]) * "
            f"{state_ref}) + ({conv}[{feature}] * {b_group}[{group_expr}, {state}])"
        )
        body.append(f"{indent * 5}{state_ref} = {state_value}[0]")
        body.append(
            f"{indent * 5}{out_inner}[{feature}] = {out_inner}[{feature}] + "
            f"({state_value}[0] * {c_group}[{group_expr}, {state}])"
        )
        d_skip = _node_indexed_canonical_input_expr(
            node,
            "mamba3_D",
            dtype_by_buffer,
            access_by_buffer,
            head_expr,
            default="1.0",
        )
        z_val = f"{projected}[{z_offset} + {feature}]"
        x_val = f"{conv}[{feature}]"
        body.append(
            f"{indent * 4}{out_inner}[{feature}] = "
            f"({out_inner}[{feature}] + ({d_skip} * {x_val})) * "
            f"{z_val} * (1.0 / (1.0 + T.exp(-{z_val})))"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    if delta:
        body.append(f"{indent * 3}# mamba3_output_policy: dense_out_projection")
        body.append(
            f"{indent * 3}for {out_dim} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{accum}[0] = 0.0")
        body.append(f"{indent * 4}for {feature} in T.serial(0, {inner_dim}):")
        out_proj_weight = _node_indexed_canonical_input_expr(
            node,
            "mamba3_out_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{out_dim} * {inner_dim} + {feature}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{accum}[0] = {accum}[0] + "
            f"({out_inner}[{feature}] * {out_proj_weight})"
        )
        body.append(
            f"{indent * 4}{_buffer_ref(delta, access_by_buffer, f'row * {hidden_size} + {out_dim}')} = "
            f"{accum}[0]"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    if history_len <= 0:
        return
    if history_len > 1:
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{(history_len - 1) * conv_channels}, step={thread_count}):"
        )
        hist_expr = f"({history_flat} // {conv_channels})"
        conv_ch_expr = f"({history_flat} % {conv_channels})"
        body.append(
            f"{indent * 4}{conv_history}[{hist_expr}, {conv_ch_expr}] = "
            f"{conv_history}[{hist_expr} + 1, {conv_ch_expr}]"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_channels}, "
        f"step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{conv_history}[{history_len - 1}, {conv_ch}] = "
        f"{projected}[{x_offset} + {conv_ch}]"
    )
    if launcher_chunked_rows:
        body.append(f"{indent * 3}T.sync_threads()")
        body.append(
            f"{indent * 3}for {history_flat} in T.serial(lane, "
            f"{history_len * conv_channels}, step={thread_count}):"
        )
        hist_expr = f"({history_flat} // {conv_channels})"
        conv_ch_expr = f"({history_flat} % {conv_channels})"
        conv_state_ref = _buffer_ref(
            "mamba3_conv_state",
            access_by_buffer,
            f"{hist_expr} * {conv_channels} + {conv_ch_expr}",
        )
        body.append(
            f"{indent * 4}{conv_state_ref} = {conv_history}[{hist_expr}, {conv_ch_expr}]"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_m2rnn_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    conv_history = _scratch_name(node, "m2rnn_conv_history")
    h_state = _scratch_name(node, "m2rnn_h_state")
    h_next = _scratch_name(node, "m2rnn_h_next")
    projected = _scratch_name(node, "m2rnn_projected_vec")
    conv = _scratch_name(node, "m2rnn_conv_vec")
    post = _scratch_name(node, "m2rnn_post_vec")
    accum = _scratch_name(node, "m2rnn_accum")
    decay = _scratch_name(node, "m2rnn_decay")
    sum_sq = _scratch_name(node, "m2rnn_sum_sq")
    sum_sq_partial = _scratch_name(node, "m2rnn_sum_sq_partial")
    inv_rms = _scratch_name(node, "m2rnn_inv_rms")
    conv_dim = int(shape_env.m2rnn_conv_dim)
    in_proj_dim = int(shape_env.m2rnn_in_proj_dim)
    total_heads = int(shape_env.m2rnn_num_heads)
    k_dim = int(shape_env.m2rnn_k_head_dim)
    v_dim = int(shape_env.m2rnn_v_head_dim)
    features = total_heads * v_dim
    history_len = max(0, int(shape_env.m2rnn_conv_kernel) - 1)
    body.append(
        f"{indent * 2}{projected} = T.alloc_shared(({in_proj_dim},), \"float32\")"
    )
    body.append(f"{indent * 2}{conv} = T.alloc_shared(({conv_dim},), \"float32\")")
    body.append(f"{indent * 2}{post} = T.alloc_shared(({features},), \"float32\")")
    body.append(
        f"{indent * 2}{h_state} = "
        f"T.alloc_shared(({total_heads}, {k_dim}, {v_dim}), \"float32\")"
    )
    body.append(
        f"{indent * 2}{h_next} = "
        f"T.alloc_shared(({total_heads}, {k_dim}, {v_dim}), \"float32\")"
    )
    if history_len > 0:
        body.append(
            f"{indent * 2}{conv_history} = "
            f"T.alloc_shared(({history_len}, {conv_dim}), \"float32\")"
        )
    body.append(f"{indent * 2}{accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{decay} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 2}{sum_sq} = T.alloc_shared((1,), \"float32\")")
    body.append(
        f"{indent * 2}{sum_sq_partial} = T.alloc_shared(({thread_count},), \"float32\")"
    )
    body.append(f"{indent * 2}{inv_rms} = T.alloc_shared((1,), \"float32\")")
    head = _scratch_name(node, "head_init")
    kk = _scratch_name(node, "kk_init")
    vv = _scratch_name(node, "vv_init")
    state_idx = _scratch_name(node, "state_idx_init")
    hist = _scratch_name(node, "hist_init")
    ch = _scratch_name(node, "conv_ch_init")
    history_idx = _scratch_name(node, "history_idx_init")
    body.append(f"{indent * 2}# {node.name}: m2rnn_state_policy: row_carried")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}if path_c_first_row_launch != 0:")
        state_loop_indent = indent * 3
        state_body_indent = indent * 4
    else:
        state_loop_indent = indent * 2
        state_body_indent = indent * 3
    body.append(
        f"{state_loop_indent}for {state_idx} in T.serial(lane, {total_heads * k_dim * v_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{state_body_indent}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{state_body_indent}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{state_body_indent}{vv} = {state_idx} % {v_dim}")
    h0_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_h0",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
    )
    body.append(f"{state_body_indent}{h_state}[{head}, {kk}, {vv}] = {h0_expr}")
    body.append(f"{state_body_indent}{h_next}[{head}, {kk}, {vv}] = {h0_expr}")
    if launcher_chunked_rows:
        body.append(f"{indent * 2}else:")
        body.append(
            f"{indent * 3}for {state_idx} in T.serial(lane, {total_heads * k_dim * v_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{head} = {state_idx} // {k_dim * v_dim}")
        body.append(f"{indent * 4}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
        body.append(f"{indent * 4}{vv} = {state_idx} % {v_dim}")
        h_state_ref = _buffer_ref(
            "m2rnn_h_state",
            access_by_buffer,
            f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
        )
        body.append(f"{indent * 4}{h_state}[{head}, {kk}, {vv}] = {h_state_ref}")
        body.append(f"{indent * 4}{h_next}[{head}, {kk}, {vv}] = {h_state_ref}")
    body.append(f"{indent * 2}T.sync_threads()")
    if history_len <= 0:
        return
    body.append(f"{indent * 2}# {node.name}: m2rnn_conv_policy: ring_history")
    body.append(
        f"{indent * 2}for {history_idx} in T.serial(lane, {history_len * conv_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 3}{hist} = {history_idx} // {conv_dim}")
    body.append(f"{indent * 3}{ch} = {history_idx} % {conv_dim}")
    conv_state_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_state",
        dtype_by_buffer,
        access_by_buffer,
        f"{hist} * {conv_dim} + {ch}",
    )
    body.append(f"{indent * 3}{conv_history}[{hist}, {ch}] = {conv_state_expr}")
    body.append(f"{indent * 2}T.sync_threads()")


def _append_row_phased_m2rnn_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
    launcher_chunked_rows: bool = False,
) -> None:
    conv_history = _scratch_name(node, "m2rnn_conv_history")
    h_state = _scratch_name(node, "m2rnn_h_state")
    h_next = _scratch_name(node, "m2rnn_h_next")
    projected = _scratch_name(node, "m2rnn_projected_vec")
    conv = _scratch_name(node, "m2rnn_conv_vec")
    post = _scratch_name(node, "m2rnn_post_vec")
    accum = _scratch_name(node, "m2rnn_accum")
    decay = _scratch_name(node, "m2rnn_decay")
    sum_sq = _scratch_name(node, "m2rnn_sum_sq")
    sum_sq_partial = _scratch_name(node, "m2rnn_sum_sq_partial")
    inv_rms = _scratch_name(node, "m2rnn_inv_rms")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim_loop = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    kernel_pos = _scratch_name(node, "kernel_pos")
    head = _scratch_name(node, "head")
    kk = _scratch_name(node, "kk")
    vv = _scratch_name(node, "vv")
    vv_inner = _scratch_name(node, "vv_inner")
    feature = _scratch_name(node, "feature")
    out_dim = _scratch_name(node, "out_dim")
    hist = _scratch_name(node, "hist")
    state_idx = _scratch_name(node, "state_idx")
    partial_lane = _scratch_name(node, "partial_lane")
    hidden_size = int(shape_env.hidden_size)
    conv_dim = int(shape_env.m2rnn_conv_dim)
    in_proj_dim = int(shape_env.m2rnn_in_proj_dim)
    total_heads = int(shape_env.m2rnn_num_heads)
    q_heads = int(shape_env.m2rnn_num_q_heads)
    k_heads = int(shape_env.m2rnn_num_k_heads)
    v_heads = int(shape_env.m2rnn_num_v_heads)
    f_heads = int(shape_env.m2rnn_num_f_heads)
    g_heads = int(shape_env.m2rnn_num_g_heads)
    w_heads = int(shape_env.m2rnn_num_weight_heads)
    k_dim = int(shape_env.m2rnn_k_head_dim)
    v_dim = int(shape_env.m2rnn_v_head_dim)
    kernel = int(shape_env.m2rnn_conv_kernel)
    history_len = max(0, kernel - 1)
    q_offset = 0
    k_offset = int(shape_env.m2rnn_q_dim)
    v_offset = k_offset + int(shape_env.m2rnn_k_dim)
    f_offset = conv_dim
    g_offset = conv_dim + f_heads
    features = total_heads * v_dim
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    f_group = total_heads // f_heads
    g_repeat = total_heads // g_heads
    w_group = total_heads // w_heads
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(f"{indent * 3}# m2rnn_projection_policy: lane_strided_dense_row_local")
    body.append(
        f"{indent * 3}for {proj_dim} in T.serial(lane, {in_proj_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{accum}[0] = 0.0")
    body.append(f"{indent * 4}for {hidden_dim_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_positional_input_expr(
        node,
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {hidden_dim_loop}",
    )
    in_proj_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{proj_dim} * {hidden_size} + {hidden_dim_loop}",
    )
    body.append(
        f"{indent * 5}{accum}[0] = {accum}[0] + "
        f"({hidden_expr} * {in_proj_weight_expr})"
    )
    body.append(f"{indent * 4}{projected}[{proj_dim}] = {accum}[0]")
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# m2rnn_conv_policy: lane_strided_causal_depthwise_ring_history")
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_dim}, step={thread_count}):"
    )
    conv_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        conv_ch,
    )
    body.append(f"{indent * 4}{conv}[{conv_ch}] = {conv_bias_expr}")
    if history_len > 0:
        body.append(f"{indent * 4}for {kernel_pos} in T.serial(0, {history_len}):")
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
            f"({conv_history}[{kernel_pos}, {conv_ch}] * {conv_weight_expr})"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] + "
        f"({projected}[{conv_ch}] * {current_conv_weight_expr})"
    )
    body.append(
        f"{indent * 4}{conv}[{conv_ch}] = {conv}[{conv_ch}] * "
        f"(1.0 / (1.0 + T.exp(-{conv}[{conv_ch}])))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# m2rnn_recurrence_policy: lane_strided_mapped_state_update")
    body.append(
        f"{indent * 3}for {head} in T.serial(lane, {total_heads}, step={thread_count}):"
    )
    f_src = f"({head} // {f_group})"
    f_input = f"{projected}[{f_offset} + {f_src}]"
    a_log_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        head,
    )
    dt_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
    )
    body.append(
        f"{indent * 4}{decay}[0] = T.exp(-T.exp({a_log_expr}) * "
        f"T.log(1.0 + T.exp({f_input} + {dt_bias_expr})))"
    )
    body.append(f"{indent * 4}for {kk} in T.serial(0, {k_dim}):")
    body.append(f"{indent * 5}for {vv} in T.serial(0, {v_dim}):")
    body.append(f"{indent * 6}{accum}[0] = 0.0")
    body.append(f"{indent * 6}for {vv_inner} in T.serial(0, {v_dim}):")
    w_src = f"({head} // {w_group})"
    state_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{w_src} * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 7}{accum}[0] = {accum}[0] + "
        f"({h_state}[{head}, {kk}, {vv_inner}] * {state_weight_expr})"
    )
    k_src = f"({head} // {k_group})"
    v_src = f"({head} // {v_group})"
    k_val = f"{conv}[{k_offset} + ({k_src} * {k_dim}) + {kk}]"
    v_val = f"{conv}[{v_offset} + ({v_src} * {v_dim}) + {vv}]"
    body.append(
        f"{indent * 6}{h_next}[{head}, {kk}, {vv}] = "
        f"({decay}[0] * {h_state}[{head}, {kk}, {vv}]) + "
        f"((1.0 - {decay}[0]) * T.tanh({accum}[0] + ({k_val} * {v_val})))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}# m2rnn_post_policy: lane_strided_residual_gate_norm_out_proj")
    body.append(
        f"{indent * 3}for {feature} in T.serial(lane, {features}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {feature} // {v_dim}")
    body.append(f"{indent * 4}{vv} = {feature} % {v_dim}")
    body.append(f"{indent * 4}{post}[{feature}] = 0.0")
    body.append(f"{indent * 4}for {kk} in T.serial(0, {k_dim}):")
    q_src = f"({head} // {q_group})"
    q_val = f"{conv}[{q_offset} + ({q_src} * {k_dim}) + {kk}]"
    body.append(
        f"{indent * 5}{post}[{feature}] = {post}[{feature}] + "
        f"({q_val} * {h_next}[{head}, {kk}, {vv}])"
    )
    d_skip_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {v_dim} + {vv}",
        default="0.0",
    )
    g_flat = f"({feature} // {g_repeat})"
    g_val = f"{projected}[{g_offset} + {g_flat}]"
    body.append(
        f"{indent * 4}{post}[{feature}] = "
        f"({post}[{feature}] + ({v_val} * {d_skip_expr})) * "
        f"{g_val} * (1.0 / (1.0 + T.exp(-{g_val})))"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {state_idx} in T.serial(lane, {total_heads * k_dim * v_dim}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{indent * 4}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{indent * 4}{vv} = {state_idx} % {v_dim}")
    body.append(
        f"{indent * 4}{h_state}[{head}, {kk}, {vv}] = "
        f"{h_next}[{head}, {kk}, {vv}]"
    )
    if launcher_chunked_rows:
        h_state_ref = _buffer_ref(
            "m2rnn_h_state",
            access_by_buffer,
            f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
        )
        body.append(
            f"{indent * 4}{h_state_ref} = {h_state}[{head}, {kk}, {vv}]"
        )
    body.append(f"{indent * 3}{sum_sq_partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for {feature} in T.serial(lane, {features}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{sum_sq_partial}[lane] = {sum_sq_partial}[lane] + "
        f"({post}[{feature}] * {post}[{feature}])"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(
        f"{indent * 4}for {partial_lane} in T.serial(0, {thread_count}):"
    )
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + {sum_sq_partial}[{partial_lane}]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(features):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    for output_name in node.outputs:
        body.append(
            f"{indent * 3}for {out_dim} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{accum}[0] = 0.0")
        body.append(f"{indent * 4}for {feature} in T.serial(0, {features}):")
        gate_norm_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_g_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            feature,
            default="1.0",
        )
        out_proj_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_out_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{out_dim} * {features} + {feature}",
            default="1.0",
        )
        body.append(
            f"{indent * 5}{accum}[0] = {accum}[0] + "
            f"({post}[{feature}] * {inv_rms}[0] * "
            f"{gate_norm_expr} * {out_proj_expr})"
        )
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, f'row * {hidden_size} + {out_dim}')} = "
            f"{accum}[0]"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    if history_len <= 0:
        return
    if history_len > 1:
        body.append(
            f"{indent * 3}for {state_idx} in T.serial(lane, {(history_len - 1) * conv_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {state_idx} // {conv_dim}")
        body.append(f"{indent * 4}{conv_ch} = {state_idx} % {conv_dim}")
        body.append(
            f"{indent * 4}{conv_history}[{hist}, {conv_ch}] = "
            f"{conv_history}[{hist} + 1, {conv_ch}]"
        )
    body.append(
        f"{indent * 3}for {conv_ch} in T.serial(lane, {conv_dim}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{conv_history}[{history_len - 1}, {conv_ch}] = "
        f"{projected}[{conv_ch}]"
    )
    if launcher_chunked_rows:
        body.append(f"{indent * 3}T.sync_threads()")
        body.append(
            f"{indent * 3}for {state_idx} in T.serial(lane, {history_len * conv_dim}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{hist} = {state_idx} // {conv_dim}")
        body.append(f"{indent * 4}{conv_ch} = {state_idx} % {conv_dim}")
        conv_state_ref = _buffer_ref(
            "m2rnn_conv_state",
            access_by_buffer,
            f"{hist} * {conv_dim} + {conv_ch}",
        )
        body.append(
            f"{indent * 4}{conv_state_ref} = {conv_history}[{hist}, {conv_ch}]"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_residual_rmsnorm_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    sum_sq = _scratch_name(node, "row_sum_sq")
    partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    lhs = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
    rhs = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
    weight = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
    residual_expr = f"({lhs} + {rhs})"
    body.append(f"{indent * 3}{partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{partial}[lane] = {partial}[lane] + "
        f"({residual_expr} * {residual_expr})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + {partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    if node.outputs:
        body.append(
            f"{indent * 4}{_buffer_ref(node.outputs[0], access_by_buffer, 'i')} = "
            f"{residual_expr}"
        )
    normalized = f"{residual_expr} * {inv_rms}[0] * {weight}"
    for output_name in node.outputs[1:]:
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, 'i')} = "
            f"{normalized}"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_entry_rmsnorm_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    """Emit the row-phased forward body for the entry RMSNorm op.

    Block A: this mirrors :func:`_append_row_phased_residual_rmsnorm_body`
    but skips the residual ``+ delta`` step because the entry RMSNorm
    runs immediately on ``hidden`` before the first in-region brick. The
    forward writes a single normalized output and leaves no residual
    side-channel because downstream bricks consume the normalized
    ``route_hidden`` directly while the inter-brick bridge still
    reduces against the raw entry ``hidden``.
    """

    sum_sq = _scratch_name(node, "row_sum_sq")
    partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
    weight = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
    body.append(f"{indent * 3}{partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{partial}[lane] = {partial}[lane] + "
        f"({hidden} * {hidden})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + {partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    normalized = f"{hidden} * {inv_rms}[0] * {weight}"
    for output_name in node.outputs:
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, 'i')} = "
            f"{normalized}"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_attention_qkv_projection_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    q_projected = _scratch_name(node, "attention_q_projected")
    kv_projected = _scratch_name(node, "attention_kv_projected")
    q_projected_pair = _scratch_name(node, "attention_q_projected_pair")
    kv_projected_pair = _scratch_name(node, "attention_kv_projected_pair")
    q_projected_vec = _scratch_name(node, "attention_q_projected_vec")
    kv_projected_vec = _scratch_name(node, "attention_kv_projected_vec")
    rope_phase = _scratch_name(node, "attention_rope_phase")
    q_prepared = _scratch_name(node, "attention_q_prepared")
    kv_prepared = _scratch_name(node, "attention_kv_prepared")
    q_head = _scratch_name(node, "q_head")
    kv_head = _scratch_name(node, "kv_head")
    indices_flat = _scratch_name(node, "indices_flat")
    k_top = _scratch_name(node, "k_top")
    d = _scratch_name(node, "d")
    h = _scratch_name(node, "h")
    src_i = _scratch_name(node, "src_i")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    topk = int(shape_env.attention_sparse_topk)
    q_scale_output = _node_output_for_canonical(node, "q_scale")
    kv_scale_output = _node_output_for_canonical(node, "kv_scale")
    q_fp8_output = _node_output_for_canonical(node, "q_fp8")
    kv_fp8_output = _node_output_for_canonical(node, "kv_fp8")
    indices_output = _node_output_for_canonical(node, "indices")
    if (
        q_scale_output is None
        or kv_scale_output is None
        or q_fp8_output is None
        or kv_fp8_output is None
    ):
        raise ValueError("row-phased attention projection requires FP8 outputs")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, src_i)
    body.append(
        f"{indent * 3}{q_projected_vec} = T.alloc_local(({head_dim},), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_projected_vec} = T.alloc_local(({head_dim},), \"float32\")"
    )
    body.append(f"{indent * 3}# fp8_prepare_policy: lane_strided_row_head_reduction")
    body.append(
        f"{indent * 3}for {q_head} in T.serial(lane, {q_heads}, step={thread_count}):"
    )
    q_scale_ref = _row_phased_attention_scale_ref(
        q_scale_output,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        q_side=True,
    )
    body.append(f"{indent * 4}{q_scale_ref} = 0.0")
    _append_attention_projection_head_dot_products(
        body,
        projected_vec=q_projected_vec,
        hidden=hidden,
        weight_buffer="attention_q_proj_weight",
        bias_buffer="attention_q_proj_bias",
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        hidden_size=hidden_size,
        head_dim=head_dim,
        head_expr=q_head,
        hidden_loop_var=h,
        src_index_var=src_i,
        dim_loop_var=d,
        indent=indent * 4,
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=q_projected_vec,
        projected=q_projected,
        paired_projected=q_projected_pair,
        rope_phase=rope_phase,
        prepared=q_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{q_scale_ref} = T.max({q_scale_ref}, "
        f"T.abs(T.cast({q_prepared}[0], \"float32\")))"
    )
    body.append(
        f"{indent * 4}{q_scale_ref} = T.max({q_scale_ref} * "
        f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
        f"T.cast(1.0e-12, \"float32\"))"
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=q_projected_vec,
        projected=q_projected,
        paired_projected=q_projected_pair,
        rope_phase=rope_phase,
        prepared=q_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{_row_phased_attention_value_ref(q_fp8_output, access_by_buffer, shape_env, row_expr='row', head_expr=q_head, dim_expr=d, q_side=True)} = "
        f"{_fp8_encode_expr(f'{q_prepared}[0]', q_scale_ref)}"
    )

    body.append(
        f"{indent * 3}for {kv_head} in T.serial(lane, {kv_heads}, step={thread_count}):"
    )
    kv_scale_ref = _row_phased_attention_scale_ref(
        kv_scale_output,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=kv_head,
        q_side=False,
    )
    body.append(f"{indent * 4}{kv_scale_ref} = 0.0")
    _append_attention_projection_head_dot_products(
        body,
        projected_vec=kv_projected_vec,
        hidden=hidden,
        weight_buffer="attention_sparse_kv_proj_weight",
        bias_buffer="attention_sparse_kv_proj_bias",
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        hidden_size=hidden_size,
        head_dim=head_dim,
        head_expr=kv_head,
        hidden_loop_var=h,
        src_index_var=src_i,
        dim_loop_var=d,
        indent=indent * 4,
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=kv_projected_vec,
        projected=kv_projected,
        paired_projected=kv_projected_pair,
        rope_phase=rope_phase,
        prepared=kv_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{kv_scale_ref} = T.max({kv_scale_ref}, "
        f"T.abs(T.cast({kv_prepared}[0], \"float32\")))"
    )
    body.append(
        f"{indent * 4}{kv_scale_ref} = T.max({kv_scale_ref} * "
        f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
        f"T.cast(1.0e-12, \"float32\"))"
    )
    body.append(f"{indent * 4}for {d} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=kv_projected_vec,
        projected=kv_projected,
        paired_projected=kv_projected_pair,
        rope_phase=rope_phase,
        prepared=kv_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=d,
        indent=indent * 5,
    )
    body.append(
        f"{indent * 5}{_row_phased_attention_value_ref(kv_fp8_output, access_by_buffer, shape_env, row_expr='row', head_expr=kv_head, dim_expr=d, q_side=False)} = "
        f"{_fp8_encode_expr(f'{kv_prepared}[0]', kv_scale_ref)}"
    )
    if indices_output is not None:
        body.append(
            f"{indent * 3}for {indices_flat} in T.serial(lane, {kv_heads * topk}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{kv_head} = {indices_flat} // {topk}")
        body.append(f"{indent * 4}{k_top} = {indices_flat} % {topk}")
        indices_ref = _row_phased_attention_indices_ref(
            indices_output,
            access_by_buffer,
            shape_env,
            head_expr=kv_head,
            topk_expr=k_top,
        )
        body.append(f"{indent * 4}if row >= {k_top}:")
        body.append(
            f"{indent * 5}{indices_ref} = row - {k_top}"
        )
        body.append(f"{indent * 4}else:")
        body.append(f"{indent * 5}{indices_ref} = -1")
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_sparse_mla_fp8_apply_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    sink_enabled = _scratch_name(node, "sink_enabled")
    sparse_index = _scratch_name(node, "sparse_index")
    score_accum = _scratch_name(node, "score_accum")
    score_max = _scratch_name(node, "score_max")
    score_weight = _scratch_name(node, "score_weight")
    score_weights = _scratch_name(node, "score_weights")
    sparse_indices = _scratch_name(node, "sparse_indices")
    sumexp = _scratch_name(node, "sumexp")
    value_accum = _scratch_name(node, "value_accum")
    context_accum = _scratch_name(node, "context_accum")
    context_values = _scratch_name(node, "context_values")
    q_head_index = _scratch_name(node, "q_head")
    kv_head_index = _scratch_name(node, "kv_head")
    source_head_index = _scratch_name(node, "source_head")
    source_dim_index = _scratch_name(node, "source_dim")
    out_dim_loop = _scratch_name(node, "out_dim_loop")
    source_head_loop = _scratch_name(node, "source_head_loop")
    source_dim_loop = _scratch_name(node, "source_dim_loop")
    dot_dim_loop = _scratch_name(node, "dot_dim")
    lse_head_loop = _scratch_name(node, "lse_head")
    k_top = _scratch_name(node, "k_top")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    sequence_length = int(shape_env.sequence_length)
    q_dim = q_heads * head_dim
    topk = int(shape_env.attention_sparse_topk)
    q_per_kv = max(1, q_heads // max(1, kv_heads))
    q_head = f"{q_head_index}[0]"
    kv_head = f"{kv_head_index}[0]"
    source_head = f"{source_head_index}[0]"
    source_dim = f"{source_dim_index}[0]"
    dot_dim = dot_dim_loop
    lse_head = lse_head_loop
    q_fp8_input = _node_input_for_canonical(node, "q_fp8")
    q_scale_input = _node_input_for_canonical(node, "q_scale")
    kv_fp8_input = _node_input_for_canonical(node, "kv_fp8")
    kv_scale_input = _node_input_for_canonical(node, "kv_scale")
    indices_input = _node_input_for_canonical(node, "indices")
    if (
        q_fp8_input is None
        or q_scale_input is None
        or kv_fp8_input is None
        or kv_scale_input is None
        or indices_input is None
    ):
        raise ValueError("row-phased sparse MLA apply requires FP8 and index inputs")
    sm_scale = _optional_buffer_expr(
        "sparse_mla_sm_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        "0",
    )
    sinks = _optional_buffer_expr(
        "sparse_mla_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0.0",
        q_head,
    )
    has_sinks = _optional_buffer_expr(
        "sparse_mla_has_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0",
        "0",
    )
    attention_out = (
        _node_output_for_canonical(node, "attention_out")
        or _node_output_with_suffix(node, "_out")
    )
    lse_output = _node_output_for_canonical(node, "lse")

    def append_sparse_index_for_current_head(
        *,
        loop_indent: str,
        kv_head_expr: str,
        use_cached_sparse_index: bool = False,
    ) -> None:
        if use_cached_sparse_index:
            body.append(f"{loop_indent}{sparse_index}[0] = {sparse_indices}[{k_top}]")
            return
        indices_current = _row_phased_attention_indices_ref(
            indices_input,
            access_by_buffer,
            shape_env,
            head_expr=kv_head_expr,
            topk_expr=k_top,
        )
        body.append(f"{loop_indent}{sparse_index}[0] = {indices_current}")

    def append_score_for_current_head(
        *,
        loop_indent: str,
        head_expr: str,
        kv_head_expr: str,
        use_cached_sparse_index: bool = False,
    ) -> None:
        q_scale_current = _row_phased_attention_scale_ref(
            q_scale_input,
            access_by_buffer,
            shape_env,
            row_expr="row",
            head_expr=head_expr,
            q_side=True,
        )
        kv_scale_current = _row_phased_attention_selected_kv_scale_ref(
            kv_scale_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head_expr,
        )
        append_sparse_index_for_current_head(
            loop_indent=loop_indent,
            kv_head_expr=kv_head_expr,
            use_cached_sparse_index=use_cached_sparse_index,
        )
        body.append(
            f"{loop_indent}if {sparse_index}[0] >= 0 and "
            f"{sparse_index}[0] < {sequence_length}:"
        )
        body.append(f"{loop_indent}    {score_accum}[0] = 0.0")
        body.append(f"{loop_indent}    for {dot_dim_loop} in T.serial(0, {head_dim}):")
        q_dot_ref = _row_phased_attention_value_ref(
            q_fp8_input,
            access_by_buffer,
            shape_env,
            row_expr="row",
            head_expr=head_expr,
            dim_expr=dot_dim,
            q_side=True,
        )
        kv_dot_ref = _row_phased_attention_selected_kv_value_ref(
            kv_fp8_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head_expr,
            dim_expr=dot_dim,
        )
        body.append(
            f"{loop_indent}        {score_accum}[0] = {score_accum}[0] + "
            f"(fp8_e4m3fn_to_float({q_dot_ref}) * "
            f"fp8_e4m3fn_to_float({kv_dot_ref}))"
        )
        body.append(
            f"{loop_indent}    {score_accum}[0] = {score_accum}[0] * "
            f"{q_scale_current} * {kv_scale_current} * {sm_scale}"
        )
        body.append(f"{loop_indent}else:")
        body.append(
            f"{loop_indent}    {score_accum}[0] = "
            "T.float32(-3.4028234663852886e38)"
        )

    if attention_out is not None:
        _append_descriptor_node_comments(
            body,
            node=node,
            descriptor=descriptor,
            indent=indent * 3,
        )
        body.append(
            f"{indent * 3}# sparse_mla_fp8_apply_policy: "
            "lane_strided_context_and_out_projection"
        )
        body.append(
            f"{indent * 3}{score_weights} = "
            f"T.alloc_local(({topk},), \"float32\")"
        )
        body.append(
            f"{indent * 3}{sparse_indices} = "
            f"T.alloc_local(({topk},), \"int32\")"
        )
        body.append(
            f"{indent * 3}{sink_enabled}[0] = "
            f"T.cast({has_sinks} != 0, \"float32\")"
        )
        body.append(
            f"{indent * 3}for {source_head_loop} in T.serial(lane, {q_heads}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{source_head_index}[0] = {source_head_loop}")
        body.append(f"{indent * 4}{q_head_index}[0] = {source_head}")
        body.append(
            f"{indent * 4}{kv_head_index}[0] = {source_head} // {q_per_kv}"
        )
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        indices_current = _row_phased_attention_indices_ref(
            indices_input,
            access_by_buffer,
            shape_env,
            head_expr=kv_head,
            topk_expr=k_top,
        )
        body.append(f"{indent * 5}{sparse_indices}[{k_top}] = {indices_current}")
        body.append(
            f"{indent * 4}{score_max}[0] = "
            "T.float32(-3.4028234663852886e38)"
        )
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        body.append(f"{indent * 5}{score_weights}[{k_top}] = 0.0")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=source_head,
            kv_head_expr=kv_head,
            use_cached_sparse_index=True,
        )
        body.append(f"{indent * 5}if {score_accum}[0] > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {score_accum}[0]")
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(f"{indent * 5}if {sinks} > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {sinks}")
        body.append(f"{indent * 4}{sumexp}[0] = 0.0")
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=source_head,
            kv_head_expr=kv_head,
            use_cached_sparse_index=True,
        )
        body.append(
            f"{indent * 5}{score_weight}[0] = "
            f"T.exp({score_accum}[0] - {score_max}[0])"
        )
        body.append(
            f"{indent * 5}{score_weights}[{k_top}] = {score_weight}[0]"
        )
        body.append(f"{indent * 5}{sumexp}[0] = {sumexp}[0] + {score_weight}[0]")
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(
            f"{indent * 5}{sumexp}[0] = {sumexp}[0] + "
            f"T.exp({sinks} - {score_max}[0])"
        )
        if lse_output is not None:
            lse_ref = _row_phased_lse_ref(
                lse_output,
                access_by_buffer,
                shape_env,
                row_expr="row",
                head_expr=source_head,
            )
            body.append(f"{indent * 4}{lse_ref} = 0.0")
            body.append(f"{indent * 4}if {sumexp}[0] > 0.0:")
            body.append(
                f"{indent * 5}{lse_ref} = {score_max}[0] + "
                f"T.log({sumexp}[0])"
            )
        body.append(
            f"{indent * 4}for {source_dim_loop} in T.serial(0, {head_dim}):"
        )
        body.append(f"{indent * 5}{source_dim_index}[0] = {source_dim_loop}")
        body.append(f"{indent * 5}{value_accum}[0] = 0.0")
        body.append(f"{indent * 5}for {k_top} in T.serial(0, {topk}):")
        append_sparse_index_for_current_head(
            loop_indent=indent * 6,
            kv_head_expr=kv_head,
            use_cached_sparse_index=True,
        )
        body.append(
            f"{indent * 6}if {sparse_index}[0] >= 0 and "
            f"{sparse_index}[0] < {sequence_length}:"
        )
        kv_value_ref = _row_phased_attention_selected_kv_value_ref(
            kv_fp8_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head,
            dim_expr=source_dim,
        )
        kv_value_scale_ref = _row_phased_attention_selected_kv_scale_ref(
            kv_scale_input,
            access_by_buffer,
            shape_env,
            selected_row_expr=f"{sparse_index}[0]",
            head_expr=kv_head,
        )
        body.append(
            f"{indent * 7}{value_accum}[0] = {value_accum}[0] + "
            f"({score_weights}[{k_top}] * fp8_e4m3fn_to_float({kv_value_ref}) * "
            f"{kv_value_scale_ref})"
        )
        body.append(f"{indent * 5}{context_accum}[0] = 0.0")
        body.append(f"{indent * 5}if {sumexp}[0] > 0.0:")
        body.append(
            f"{indent * 6}{context_accum}[0] = "
            f"{value_accum}[0] / {sumexp}[0]"
        )
        body.append(
            f"{indent * 5}{context_values}["
            f"{source_head_loop} * {head_dim} + {source_dim_loop}"
            f"] = {context_accum}[0]"
        )
        body.append(f"{indent * 3}T.sync_threads()")
        body.append(
            f"{indent * 3}for {out_dim_loop} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        attention_out_ref = _indexed_buffer_ref(
            attention_out,
            access_by_buffer,
            f"row * {hidden_size} + {out_dim_loop}",
        )
        out_bias = _optional_indexed_buffer_expr(
            "attention_out_proj_bias",
            dtype_by_buffer,
            access_by_buffer,
            default="0.0",
            index_expr=out_dim_loop,
        )
        body.append(f"{indent * 4}{attention_out_ref} = {out_bias}")
        body.append(f"{indent * 4}for {source_dim_loop} in T.serial(0, {q_dim}):")
        out_weight = _optional_indexed_buffer_expr(
            "attention_out_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            default="1.0",
            index_expr=(
                f"({out_dim_loop}) * {q_dim} + "
                f"{source_dim_loop}"
            ),
        )
        body.append(
            f"{indent * 5}{attention_out_ref} = {attention_out_ref} + "
            f"({context_values}[{source_dim_loop}] * {out_weight})"
        )
        body.append(f"{indent * 3}T.sync_threads()")
    if lse_output is not None and attention_out is None:
        body.append(
            f"{indent * 3}for {lse_head_loop} in T.serial(lane, {q_heads}, "
            f"step={thread_count}):"
        )
        body.append(f"{indent * 4}{q_head_index}[0] = {lse_head}")
        body.append(f"{indent * 4}{kv_head_index}[0] = {q_head} // {q_per_kv}")
        body.append(
            f"{indent * 4}{sink_enabled}[0] = "
            f"T.cast({has_sinks} != 0, \"float32\")"
        )
        body.append(
            f"{indent * 4}{score_max}[0] = "
            "T.float32(-3.4028234663852886e38)"
        )
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=q_head,
            kv_head_expr=kv_head,
        )
        body.append(f"{indent * 5}if {score_accum}[0] > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {score_accum}[0]")
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(f"{indent * 5}if {sinks} > {score_max}[0]:")
        body.append(f"{indent * 6}{score_max}[0] = {sinks}")
        body.append(f"{indent * 4}{sumexp}[0] = 0.0")
        body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
        append_score_for_current_head(
            loop_indent=indent * 5,
            head_expr=q_head,
            kv_head_expr=kv_head,
        )
        body.append(
            f"{indent * 5}{sumexp}[0] = {sumexp}[0] + "
            f"T.exp({score_accum}[0] - {score_max}[0])"
        )
        body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
        body.append(
            f"{indent * 5}{sumexp}[0] = {sumexp}[0] + "
            f"T.exp({sinks} - {score_max}[0])"
        )
        lse_ref = _row_phased_lse_ref(
            lse_output,
            access_by_buffer,
            shape_env,
            row_expr="row",
            head_expr=q_head,
        )
        body.append(
            f"{indent * 4}{lse_ref} = "
            "0.0"
        )
        body.append(f"{indent * 4}if {sumexp}[0] > 0.0:")
        body.append(
            f"{indent * 5}{lse_ref} = "
            f"{score_max}[0] + T.log({sumexp}[0])"
        )


def _append_attention_projection_head_dot_products(
    body: list[str],
    *,
    projected_vec: str,
    hidden: str,
    weight_buffer: str,
    bias_buffer: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    head_dim: int,
    head_expr: str,
    hidden_loop_var: str,
    src_index_var: str,
    dim_loop_var: str,
    indent: str,
) -> None:
    body.append(f"{indent}for {dim_loop_var} in T.serial(0, {head_dim}):")
    output_offset = f"{head_expr} * {head_dim} + {dim_loop_var}"
    bias = _optional_indexed_buffer_expr(
        bias_buffer,
        dtype_by_buffer,
        access_by_buffer,
        index_expr=output_offset,
    )
    body.append(f"{indent}    {projected_vec}[{dim_loop_var}] = {bias}")
    body.append(f"{indent}for {hidden_loop_var} in T.serial(0, {hidden_size}):")
    body.append(
        f"{indent}    {src_index_var} = row * {hidden_size} + "
        f"{hidden_loop_var}"
    )
    body.append(f"{indent}    for {dim_loop_var} in T.serial(0, {head_dim}):")
    output_offset = f"{head_expr} * {head_dim} + {dim_loop_var}"
    weight = _optional_indexed_buffer_expr(
        weight_buffer,
        dtype_by_buffer,
        access_by_buffer,
        index_expr=f"({output_offset}) * {hidden_size} + {hidden_loop_var}",
    )
    body.append(
        f"{indent}        {projected_vec}[{dim_loop_var}] = "
        f"{projected_vec}[{dim_loop_var}] + ({hidden} * {weight})"
    )


def _append_attention_projection_prepare_from_vector(
    body: list[str],
    *,
    projected_vec: str,
    projected: str,
    paired_projected: str,
    rope_phase: str,
    prepared: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    head_dim: int,
    dim_expr: str,
    indent: str,
) -> None:
    rope_half = max(1, head_dim // 2)
    body.append(f"{indent}{projected}[0] = {projected_vec}[{dim_expr}]")
    body.append(f"{indent}if {dim_expr} < {rope_half}:")
    body.append(
        f"{indent}    {paired_projected}[0] = "
        f"{projected_vec}[{dim_expr} + {rope_half}]"
    )
    rope_first = _optional_indexed_buffer_expr(
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        index_expr=dim_expr,
    )
    body.append(
        f"{indent}    {rope_phase}[0] = "
        f'T.cast(row, "float32") * {rope_first}'
    )
    body.append(
        f"{indent}    {prepared}[0] = "
        f"({projected}[0] * T.cos({rope_phase}[0])) + "
        f"({paired_projected}[0] * T.sin({rope_phase}[0]))"
    )
    body.append(f"{indent}else:")
    body.append(
        f"{indent}    {paired_projected}[0] = "
        f"{projected_vec}[{dim_expr} - {rope_half}]"
    )
    rope_second = _optional_indexed_buffer_expr(
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        index_expr=f"{dim_expr} - {rope_half}",
    )
    body.append(
        f"{indent}    {rope_phase}[0] = "
        f'T.cast(row, "float32") * {rope_second}'
    )
    body.append(
        f"{indent}    {prepared}[0] = "
        f"({projected}[0] * T.cos({rope_phase}[0])) - "
        f"({paired_projected}[0] * T.sin({rope_phase}[0]))"
    )


def _attention_qkv_projection_prepare_statements(
    *,
    projected: str,
    rope_phase: str,
    prepared: str,
    hidden: str,
    weight: str,
    bias: str,
    rope: str,
    indent: str,
) -> list[str]:
    return [
        f"{indent}{projected}[0] = {hidden} + {weight} + {bias}",
        f"{indent}{rope_phase}[0] = {rope}",
        f"{indent}{prepared}[0] = {projected}[0] + {rope_phase}[0]",
    ]


def _row_phased_attention_scale_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
    q_side: bool,
) -> str:
    name = _safe_identifier(buffer_name)
    hidden = int(shape_env.hidden_size)
    head_dim = int(shape_env.attention_head_dim)
    heads = (
        int(shape_env.attention_num_q_heads)
        if q_side
        else int(shape_env.attention_num_kv_heads)
    )
    internal_ref = (
        f"{name}[(i % {hidden}) // {head_dim}]"
        if q_side
        else f"{name}[((i % {hidden}) // {head_dim}) % {heads}]"
    )
    if access_by_buffer.get(buffer_name) == internal_ref:
        return f"{name}[{head_expr}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{row_expr} * {heads} + {head_expr}",
    )


def _row_phased_attention_value_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
    dim_expr: str,
    q_side: bool,
) -> str:
    name = _safe_identifier(buffer_name)
    head_dim = int(shape_env.attention_head_dim)
    heads = (
        int(shape_env.attention_num_q_heads)
        if q_side
        else int(shape_env.attention_num_kv_heads)
    )
    row_width = heads * head_dim
    offset = f"{head_expr} * {head_dim} + {dim_expr}"
    if access_by_buffer.get(buffer_name) == f"{name}[i % {row_width}]":
        return f"{name}[{offset}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{row_expr} * {row_width} + {offset}",
    )


def _row_phased_attention_selected_kv_value_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    selected_row_expr: str,
    head_expr: str,
    dim_expr: str,
) -> str:
    name = _safe_identifier(buffer_name)
    head_dim = int(shape_env.attention_head_dim)
    heads = int(shape_env.attention_num_kv_heads)
    row_width = heads * head_dim
    offset = f"{head_expr} * {head_dim} + {dim_expr}"
    if access_by_buffer.get(buffer_name) == f"{name}[i % {row_width}]":
        return f"{name}[{offset}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{selected_row_expr} * {row_width} + {offset}",
    )


def _row_phased_attention_selected_kv_scale_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    selected_row_expr: str,
    head_expr: str,
) -> str:
    name = _safe_identifier(buffer_name)
    heads = int(shape_env.attention_num_kv_heads)
    internal_ref = (
        f"{name}[((i % {shape_env.hidden_size}) // "
        f"{shape_env.attention_head_dim}) % {heads}]"
    )
    if access_by_buffer.get(buffer_name) == internal_ref:
        return f"{name}[{head_expr}]"
    return _row_phased_direct_flat_ref(
        buffer_name,
        access_by_buffer,
        f"{selected_row_expr} * {heads} + {head_expr}",
    )


def _row_phased_direct_flat_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    flat_index_expr: str,
) -> str:
    """Return a direct flat reference for row-phased full-sequence buffers.

    Full-sequence attention scale buffers use hidden-indexed loop refs such as
    ``(i // H) * heads + ...``.  Substituting a scale-space index back into
    that expression collapses ``row * heads + head`` into ``row + head // d``.
    Emit the already-flat row-major index directly instead.
    """

    ref = access_by_buffer.get(buffer_name)
    name = _safe_identifier(buffer_name)
    if ref is not None:
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            return f"{match.group(1)}[{match.group(2)} + {flat_index_expr}]"
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[{flat_index_expr}]"
        match = re.fullmatch(r"([A-Za-z_]\w*)\[i\]", ref)
        if match is not None:
            return f"{match.group(1)}[{flat_index_expr}]"
    return f"{name}[{flat_index_expr}]"


def _row_phased_attention_indices_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    head_expr: str,
    topk_expr: str,
) -> str:
    name = _safe_identifier(buffer_name)
    kv_heads = int(shape_env.attention_num_kv_heads)
    topk = int(shape_env.attention_sparse_topk)
    row_width = kv_heads * topk
    offset = f"{head_expr} * {topk} + {topk_expr}"
    ref = access_by_buffer.get(buffer_name)
    full_sequence_ref = (
        f"{name}[(i // {shape_env.hidden_size}) * {row_width} + "
        f"(i % {row_width})]"
    )
    if ref == full_sequence_ref:
        return f"{name}[row * {row_width} + {offset}]"
    if ref == f"{name}[i % {row_width}]":
        # Row-local internal buffer — offset within single row.
        return f"{name}[{offset}]"
    if ref is not None and f"% {row_width}" not in ref:
        # Full-sequence buffer — bypass _indexed_buffer_ref to avoid
        # broken i-substitution in complex access expressions.
        # Handle banked buffers (e.g. path_c_float32_abi_bank[OFFSET + ...]).
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            bank_name = match.group(1)
            bank_offset = match.group(2)
            return f"{bank_name}[{bank_offset} + row * {row_width} + {offset}]"
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[row * {row_width} + {offset}]"
        return f"{name}[row * {row_width} + {offset}]"
    return _indexed_buffer_ref(
        buffer_name,
        access_by_buffer,
        f"row * {row_width} + {offset}",
    )


def _row_phased_attention_bwd_indices_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
    topk_expr: str,
) -> str:
    """Return row-specific Sparse-MLA indices for the recompute bwd loop.

    The generated attention projection keeps causal sparse indices row-local
    during forward.  The backward row loop runs after the forward row loop, so
    reading that scratch would reuse the final row's indices for every row.
    Recompute the descriptor's causal index formula instead.
    """

    name = _safe_identifier(buffer_name)
    kv_heads = int(shape_env.attention_num_kv_heads)
    topk = int(shape_env.attention_sparse_topk)
    row_width = kv_heads * topk
    ref = access_by_buffer.get(buffer_name)
    offset = f"{head_expr} * {topk} + {topk_expr}"
    full_expr = f"(i // {shape_env.hidden_size}) * {row_width} + (i % {row_width})"
    row_full_expr = f"{row_expr} * {row_width} + {offset}"
    if ref == f"{name}[{full_expr}]":
        return f"{name}[{row_full_expr}]"
    if ref is not None:
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None:
            bank_name = bank_match.group(1)
            bank_expr = bank_match.group(2)
            if bank_expr == full_expr:
                return f"{bank_name}[{row_full_expr}]"
            offset_match = re.fullmatch(
                rf"(\d+) \+ \({re.escape(full_expr)}\)",
                bank_expr,
            )
            if offset_match is not None:
                return f"{bank_name}[{offset_match.group(1)} + ({row_full_expr})]"
    if ref == f"{name}[i % {row_width}]":
        return f"T.if_then_else({row_expr} >= {topk_expr}, {row_expr} - {topk_expr}, -1)"
    if ref is not None:
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            return (
                f"{match.group(1)}[{match.group(2)} + "
                f"{row_expr} * {row_width} + {offset}]"
            )
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[{row_expr} * {row_width} + {offset}]"
    return f"{name}[{row_expr} * {row_width} + {offset}]"


def _row_phased_lse_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    *,
    row_expr: str,
    head_expr: str,
) -> str:
    """Return a direct reference for lse in the row-phased emitter.

    Bypasses ``_indexed_buffer_ref`` because the complex access pattern
    generated by ``_loop_indexed_buffer_ref`` for full-sequence lse
    buffers (``name[(i // H) * Q + ((i % H) // d)]``) produces wrong
    addresses when ``i`` is naively substituted with a flat lse index
    like ``row * Q + head``.
    """
    name = _safe_identifier(buffer_name)
    q_heads = int(shape_env.attention_num_q_heads)
    ref = access_by_buffer.get(buffer_name)
    if ref == f"{name}[i % {q_heads}]":
        # Row-local internal buffer — offset within single row.
        return f"{name}[{head_expr}]"
    # Full-sequence buffer.  The access pattern may be banked, e.g.
    #   path_c_float32_abi_bank[OFFSET + ((i // H) * Q + ...)]
    # Extract the bank name and numeric offset so we can emit a
    # direct flat reference without i-substitution.
    if ref is not None:
        match = re.match(r"([A-Za-z_]\w*)\[(\d+) \+ ", ref)
        if match is not None:
            bank_name = match.group(1)
            offset = match.group(2)
            return f"{bank_name}[{offset} + {row_expr} * {q_heads} + {head_expr}]"
        bank_match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
        if bank_match is not None and "_abi_bank" in bank_match.group(1):
            return f"{bank_match.group(1)}[{row_expr} * {q_heads} + {head_expr}]"
    # Unbanked fallback — direct flat index.
    return f"{name}[{row_expr} * {q_heads} + {head_expr}]"


def _is_full_sequence_bank_slot(
    buffer_name: str,
    access_by_buffer: Mapping[str, str],
) -> bool:
    """Return True when ``buffer_name`` resolves to a full-sequence bank slot.

    A "full-sequence bank slot" is a logical buffer whose physical access
    expression resolves to ``path_c_..._abi_bank[OFFSET + i]`` (i.e. the
    index depends on the kernel-wide loop variable ``i`` covering the
    full ``sequence_length * hidden_size`` range), as opposed to a
    per-row scratch buffer that wraps around each row iteration.

    The schedule uses this signal to decide whether the bwd needs a
    one-shot pre-zero (full-sequence bank slot) or whether per-row
    zero-init followed by ``=`` accumulation is sufficient (per-row
    scratch buffer).
    """
    if buffer_name not in access_by_buffer:
        return False
    access = access_by_buffer[buffer_name]
    if "_abi_bank[" not in access:
        return False
    if " % " in access:
        return False
    return True


def _append_row_phased_residual_rmsnorm_bwd_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    if len(node.outputs) < 3:
        return
    weight_grad = node.outputs[2]
    body.append(f"{indent * 2}for h in T.serial(lane, {hidden_size}, step={thread_count}):")
    body.append(
        f"{indent * 3}{_buffer_ref(weight_grad, access_by_buffer, 'h')} = 0.0"
    )
    body.append(f"{indent * 2}T.sync_threads()")


def _append_row_phased_entry_rmsnorm_bwd_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    access_by_buffer: dict[str, str],
    hidden_size: int,
    sequence_length: int,
    thread_count: int,
    indent: str,
) -> None:
    """Zero-init the entry RMSNorm weight grad before the row loop.

    Block A: the per-brick entry RMSNorm weight gradient is accumulated
    across every row during the bwd pass, so it must start at zero.
    The bank ``hidden_grad`` slot is NOT zero-initialised here -- the
    inter-brick ``residual_rmsnorm_bwd`` writes it with ``=`` first (per
    Fix B-1), and entry_rmsnorm_bwd then accumulates onto that value
    with ``+=``.
    """

    # The entry RMSNorm's normalized-hidden output is consumed downstream
    # by the first in-region brick's forward, which means the brick's
    # backward writes its hidden_grad contribution into that same bank
    # slot via ``+=``. Because the slot is a full-sequence bank slot, the
    # block bwd emitter skips its per-row zero-init (Fix B-1) -- so the
    # entry RMSNorm op (the only logical "owner" of that slot) is
    # responsible for one-shot zeroing the buffer before the row loop
    # starts. The slot is the first bwd input of entry_rmsnorm_bwd
    # (``normed_hidden_grad``).
    if len(node.inputs) >= 1:
        normed_hidden_grad = node.inputs[0]
        if _is_full_sequence_bank_slot(normed_hidden_grad, access_by_buffer):
            body.append(
                f"{indent * 2}for i in T.serial(lane, "
                f"{int(sequence_length) * int(hidden_size)}, step={thread_count}):"
            )
            body.append(
                f"{indent * 3}# entry_rmsnorm_bwd: pre-zero normed_hidden_grad bank slot"
            )
            body.append(
                f"{indent * 3}{_buffer_ref(normed_hidden_grad, access_by_buffer, 'i')} = 0.0"
            )
            body.append(f"{indent * 2}T.sync_threads()")

    if len(node.outputs) < 2:
        return
    weight_grad = node.outputs[1]
    body.append(f"{indent * 2}for h in T.serial(lane, {hidden_size}, step={thread_count}):")
    body.append(
        f"{indent * 3}{_buffer_ref(weight_grad, access_by_buffer, 'h')} = 0.0"
    )
    body.append(f"{indent * 2}T.sync_threads()")


def _append_row_phased_residual_rmsnorm_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    sum_sq = _scratch_name(node, "row_sum_sq")
    sum_sq_partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    dot = _scratch_name(node, "row_dot")
    dot_partial = _scratch_name(node, "row_dot_partial")
    norm_grad_scratch = _scratch_name(node, "row_norm_grad")
    total_grad_scratch = _scratch_name(node, "row_total_grad")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    if len(node.inputs) >= 5:
        hidden_after_grad = _node_input_expr(
            node,
            0,
            dtype_by_buffer,
            access_by_buffer,
            "i",
        )
        norm_grad = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
        hidden = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
        delta = _node_input_expr(node, 3, dtype_by_buffer, access_by_buffer, "i")
        weight = _node_input_expr(node, 4, dtype_by_buffer, access_by_buffer, "i")
    else:
        if _node_output_for_canonical(node, "m2rnn_delta") is not None:
            hidden_after_grad_name = "hidden_after_m2rnn_grad"
        elif _node_output_for_canonical(node, "mamba3_delta") is not None:
            hidden_after_grad_name = "hidden_after_mamba3_grad"
        else:
            hidden_after_grad_name = ""
        hidden_after_grad = (
            _optional_indexed_buffer_expr(
                hidden_after_grad_name,
                dtype_by_buffer,
                access_by_buffer,
                default="0.0",
                index_expr="i",
            )
            if hidden_after_grad_name
            else "0.0"
        )
        norm_grad = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
        hidden = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
        delta = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
        weight = _node_input_expr(node, 3, dtype_by_buffer, access_by_buffer, "i")
    residual_expr = f"({hidden} + {delta})"
    body.append(f"{indent * 3}{sum_sq_partial}[lane] = 0.0")
    body.append(f"{indent * 3}{dot_partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{sum_sq_partial}[lane] = {sum_sq_partial}[lane] + "
        f"({residual_expr} * {residual_expr})"
    )
    body.append(
        f"{indent * 4}{dot_partial}[lane] = {dot_partial}[lane] + "
        f"({norm_grad} * {weight} * {residual_expr})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}{dot}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + "
        f"{sum_sq_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 5}{dot}[0] = {dot}[0] + {dot_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{norm_grad_scratch}[0] = {norm_grad} * {weight}")
    body.append(
        f"{indent * 4}{total_grad_scratch}[0] = {hidden_after_grad} + "
        f"({inv_rms}[0] * ({norm_grad_scratch}[0] - "
        f"({residual_expr} * {dot}[0] * {inv_rms}[0] * "
        f"{inv_rms}[0] / {float(hidden_size):.1f})))"
    )
    for output_name in node.outputs[:2]:
        body.append(
            f"{indent * 4}{_buffer_ref(output_name, access_by_buffer, 'i')} = "
            f"{total_grad_scratch}[0]"
        )
    if len(node.outputs) > 2:
        weight_grad = _buffer_ref(node.outputs[2], access_by_buffer, "i")
        body.append(
            f"{indent * 4}{weight_grad} = {weight_grad} + "
            f"({norm_grad} * {residual_expr} * {inv_rms}[0])"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_entry_rmsnorm_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    thread_count: int,
    indent: str,
) -> None:
    """Emit the row-phased backward body for the entry RMSNorm op.

    Block A: this mirrors :func:`_append_row_phased_residual_rmsnorm_bwd_body`
    but skips the residual ``+ delta`` reduction. The forward computed
    ``y[h] = x[h] * inv_rms * w[h]`` with ``inv_rms = rsqrt(mean(x*x) + eps)``.
    The backward recomputes ``inv_rms`` from ``hidden`` alone, then derives
    ``hidden_grad += inv_rms * (norm_grad*w - x * dot * inv_rms^2 / D)`` and
    ``weight_grad[h%H] += norm_grad * x * inv_rms`` row-by-row.

    The bank ``hidden_grad`` slot is written with ``+=`` because the
    inter-brick ``residual_rmsnorm_bwd`` already wrote it with ``=`` in
    the same iteration (Fix B-1 convention).
    """

    sum_sq = _scratch_name(node, "row_sum_sq")
    sum_sq_partial = _scratch_name(node, "row_sum_sq_partial")
    inv_rms = _scratch_name(node, "row_inv_rms")
    dot = _scratch_name(node, "row_dot")
    dot_partial = _scratch_name(node, "row_dot_partial")
    norm_grad_scratch = _scratch_name(node, "row_norm_grad")
    total_grad_scratch = _scratch_name(node, "row_total_grad")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    norm_grad = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
    hidden = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
    weight = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
    body.append(f"{indent * 3}{sum_sq_partial}[lane] = 0.0")
    body.append(f"{indent * 3}{dot_partial}[lane] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(
        f"{indent * 4}{sum_sq_partial}[lane] = {sum_sq_partial}[lane] + "
        f"({hidden} * {hidden})"
    )
    body.append(
        f"{indent * 4}{dot_partial}[lane] = {dot_partial}[lane] + "
        f"({norm_grad} * {weight} * {hidden})"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}if lane == 0:")
    body.append(f"{indent * 4}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}{dot}[0] = 0.0")
    body.append(f"{indent * 4}for partial_lane in T.serial(0, {thread_count}):")
    body.append(
        f"{indent * 5}{sum_sq}[0] = {sum_sq}[0] + "
        f"{sum_sq_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 5}{dot}[0] = {dot}[0] + {dot_partial}[partial_lane]"
    )
    body.append(
        f"{indent * 4}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size} + lane, "
        f"(row + 1) * {hidden_size}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{norm_grad_scratch}[0] = {norm_grad} * {weight}")
    body.append(
        f"{indent * 4}{total_grad_scratch}[0] = "
        f"({inv_rms}[0] * ({norm_grad_scratch}[0] - "
        f"({hidden} * {dot}[0] * {inv_rms}[0] * "
        f"{inv_rms}[0] / {float(hidden_size):.1f})))"
    )
    if node.outputs:
        hidden_grad_ref = _buffer_ref(node.outputs[0], access_by_buffer, "i")
        if _is_full_sequence_bank_slot(node.outputs[0], access_by_buffer):
            # Fix B-1 chain: residual_rmsnorm_bwd already wrote
            # hidden_grad with ``=`` in the same row iteration; entry
            # RMSNorm bwd is the second writer and must accumulate.
            body.append(
                f"{indent * 4}{hidden_grad_ref} = {hidden_grad_ref} + "
                f"{total_grad_scratch}[0]"
            )
        else:
            # Non-bank scratch slot: first writer wins.
            body.append(f"{indent * 4}{hidden_grad_ref} = {total_grad_scratch}[0]")
    if len(node.outputs) > 1:
        weight_grad = _buffer_ref(node.outputs[1], access_by_buffer, "i")
        body.append(
            f"{indent * 4}{weight_grad} = {weight_grad} + "
            f"({norm_grad} * {hidden} * {inv_rms}[0])"
        )
    body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_sparse_mla_fp8_apply_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    chunked_rows: bool,
    indent: str,
) -> None:
    sink_enabled = _scratch_name(node, "sink_enabled")
    sparse_index = _scratch_name(node, "sparse_index")
    sparse_indices = _scratch_name(node, "sparse_indices")
    score_accum = _scratch_name(node, "score_accum")
    score_max = _scratch_name(node, "score_max")
    score_weight = _scratch_name(node, "score_weight")
    score_weights = _scratch_name(node, "score_weights")
    sumexp = _scratch_name(node, "sumexp")
    context_values = _scratch_name(node, "context_values")
    context_grads = _scratch_name(node, "context_grads")
    attention_hidden_values = _scratch_name(node, "attention_hidden_values")
    attention_hidden_sum_sq = _scratch_name(node, "attention_hidden_sum_sq")
    attention_hidden_inv_rms = _scratch_name(node, "attention_hidden_inv_rms")
    q_projected_vec = _scratch_name(node, "attention_q_projected_vec")
    q_prepared_values = _scratch_name(node, "attention_q_prepared_values")
    q_deq_values = _scratch_name(node, "attention_q_deq_values")
    q_projected = _scratch_name(node, "attention_q_projected")
    q_projected_pair = _scratch_name(node, "attention_q_projected_pair")
    q_prepared = _scratch_name(node, "attention_q_prepared")
    q_recomputed_scale = _scratch_name(node, "attention_q_recomputed_scale")
    q_rope_phase = _scratch_name(node, "attention_q_rope_phase")
    value_accum = _scratch_name(node, "value_accum")
    dp_values = _scratch_name(node, "dp_values")
    dp_accum = _scratch_name(node, "dp_accum")
    ds_score = _scratch_name(node, "ds_score")
    dq_accum = _scratch_name(node, "dq_accum")
    dkv_accum = _scratch_name(node, "dkv_accum")
    q_head = _scratch_name(node, "q_head")
    kv_head = _scratch_name(node, "kv_head")
    dim = _scratch_name(node, "dim")
    dot_dim = _scratch_name(node, "dot_dim")
    out_dim = _scratch_name(node, "out_dim")
    k_top = _scratch_name(node, "k_top")
    flat = _scratch_name(node, "flat")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    sequence_length = int(shape_env.sequence_length)
    q_dim = q_heads * head_dim
    topk = int(shape_env.attention_sparse_topk)
    q_per_kv = max(1, q_heads // max(1, kv_heads))
    q_fp8_grad = _node_output_for_canonical(node, "q_fp8")
    q_scale_grad = _node_output_for_canonical(node, "q_scale")
    kv_fp8_grad = _node_output_for_canonical(node, "kv_fp8")
    kv_scale_grad = _node_output_for_canonical(node, "kv_scale")
    out_weight_grad = _node_output_for_canonical(node, "attention_out_proj_weight")
    out_bias_grad = _node_output_for_canonical(node, "attention_out_proj_bias")
    q_fp8_input = _node_input_for_canonical(node, "q_fp8")
    q_scale_input = _node_input_for_canonical(node, "q_scale")
    kv_fp8_input = _node_input_for_canonical(node, "kv_fp8")
    kv_scale_input = _node_input_for_canonical(node, "kv_scale")
    indices_input = _node_input_for_canonical(node, "indices")
    if (
        q_fp8_input is None
        or q_scale_input is None
        or kv_fp8_input is None
        or kv_scale_input is None
        or indices_input is None
        or q_fp8_grad is None
        or kv_fp8_grad is None
    ):
        raise ValueError("row-phased sparse MLA backward requires FP8 inputs and grads")
    sm_scale = _optional_buffer_expr(
        "sparse_mla_sm_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        "0",
    )
    sinks = _optional_buffer_expr(
        "sparse_mla_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0.0",
        q_head,
    )
    has_sinks = _optional_buffer_expr(
        "sparse_mla_has_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0",
        "0",
    )
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(
        f"{indent * 3}# sparse_mla_fp8_apply_bwd_policy: "
        "exact_softmax_vjp_and_out_projection"
    )
    body.append(f"{indent * 3}if row == 0:")
    for output_name, extent in (
        (q_fp8_grad, sequence_length * q_dim),
        (q_scale_grad, sequence_length * q_heads),
        (kv_fp8_grad, sequence_length * kv_heads * head_dim),
        (kv_scale_grad, sequence_length * kv_heads),
    ):
        if output_name is None:
            continue
        body.append(
            f"{indent * 4}for {flat} in T.serial(lane, {extent}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(output_name, access_by_buffer, flat)} = 0.0"
        )
    body.append(f"{indent * 3}if row == 0:")
    for output_name, extent in (
        (out_weight_grad, hidden_size * q_dim),
        (out_bias_grad, hidden_size),
    ):
        if output_name is None:
            continue
        body.append(
            f"{indent * 4}for {flat} in T.serial(lane, {extent}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(output_name, access_by_buffer, flat)} = 0.0"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(f"{indent * 3}{sink_enabled} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{sparse_index} = T.alloc_local((1,), \"int32\")")
    body.append(f"{indent * 3}{score_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{score_max} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{score_weight} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{sumexp} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{value_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dp_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{ds_score} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dq_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dkv_accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{score_weights} = T.alloc_local(({topk},), \"float32\")")
    body.append(f"{indent * 3}{dp_values} = T.alloc_local(({topk},), \"float32\")")
    body.append(f"{indent * 3}{sparse_indices} = T.alloc_local(({topk},), \"int32\")")
    body.append(f"{indent * 3}{context_values} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(f"{indent * 3}{context_grads} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_values} = T.alloc_local(({hidden_size},), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_sum_sq} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_inv_rms} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{q_projected_vec} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(f"{indent * 3}{q_prepared_values} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(f"{indent * 3}{q_deq_values} = T.alloc_local(({head_dim},), \"float32\")")
    body.append(f"{indent * 3}{q_projected} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{q_projected_pair} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{q_prepared} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{q_recomputed_scale} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{q_rope_phase} = T.alloc_local((1,), \"float32\")")
    body.append(
        f"{indent * 3}for {q_head} in T.serial(lane, {q_heads}, step={thread_count}):"
    )
    body.append(f"{indent * 4}{kv_head} = {q_head} // {q_per_kv}")
    body.append(
        f"{indent * 4}{sink_enabled}[0] = T.cast({has_sinks} != 0, \"float32\")"
    )
    body.append(f"{indent * 4}{attention_hidden_sum_sq}[0] = 0.0")
    body.append(f"{indent * 4}for {out_dim} in T.serial(0, {hidden_size}):")
    hidden_after_m2rnn = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("attention_hidden", "hidden"),
        4,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {out_dim}",
    )
    body.append(
        f"{indent * 5}{attention_hidden_values}[{out_dim}] = {hidden_after_m2rnn}"
    )
    body.append(
        f"{indent * 5}{attention_hidden_sum_sq}[0] = {attention_hidden_sum_sq}[0] + "
        f"({attention_hidden_values}[{out_dim}] * {attention_hidden_values}[{out_dim}])"
    )
    body.append(
        f"{indent * 4}{attention_hidden_inv_rms}[0] = T.rsqrt(({attention_hidden_sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 4}for {out_dim} in T.serial(0, {hidden_size}):")
    attention_norm_weight = _optional_indexed_buffer_expr(
        "m2rnn_residual_to_attention_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        default="1.0",
        index_expr=out_dim,
    )
    body.append(
        f"{indent * 5}{attention_hidden_values}[{out_dim}] = "
        f"{attention_hidden_values}[{out_dim}]"
    )
    body.append(f"{indent * 4}# attention_q_prepare_recompute_policy: exact_row_local_fp8")
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    q_bias = _optional_indexed_buffer_expr(
        "attention_q_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        default="0.0",
        index_expr=f"{q_head} * {head_dim} + {dim}",
    )
    body.append(f"{indent * 5}{q_projected_vec}[{dim}] = {q_bias}")
    body.append(f"{indent * 4}for {out_dim} in T.serial(0, {hidden_size}):")
    body.append(f"{indent * 5}for {dim} in T.serial(0, {head_dim}):")
    q_weight = _optional_indexed_buffer_expr(
        "attention_q_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        default="1.0",
        index_expr=f"({q_head} * {head_dim} + {dim}) * {hidden_size} + {out_dim}",
    )
    body.append(
        f"{indent * 6}{q_projected_vec}[{dim}] = {q_projected_vec}[{dim}] + "
        f"({attention_hidden_values}[{out_dim}] * {q_weight})"
    )
    body.append(f"{indent * 4}{q_recomputed_scale}[0] = 0.0")
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    _append_attention_projection_prepare_from_vector(
        body,
        projected_vec=q_projected_vec,
        projected=q_projected,
        paired_projected=q_projected_pair,
        rope_phase=q_rope_phase,
        prepared=q_prepared,
        dtype_by_buffer=dtype_by_buffer,
        access_by_buffer=access_by_buffer,
        head_dim=head_dim,
        dim_expr=dim,
        indent=indent * 5,
    )
    body.append(f"{indent * 5}{q_prepared_values}[{dim}] = {q_prepared}[0]")
    body.append(
        f"{indent * 5}{q_recomputed_scale}[0] = T.max({q_recomputed_scale}[0], "
        f"T.abs(T.cast({q_prepared}[0], \"float32\")))"
    )
    body.append(
        f"{indent * 4}{q_recomputed_scale}[0] = T.max({q_recomputed_scale}[0] * "
        f"T.cast({1.0 / 448.0:.17g}, \"float32\"), T.cast(1.0e-12, \"float32\"))"
    )
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    body.append(
        f"{indent * 5}{q_deq_values}[{dim}] = "
        f"fp8_e4m3fn_to_float(float_to_fp8_e4m3fn_bits(T.min(T.max("
        f"(T.cast({q_prepared_values}[{dim}], \"float32\") / {q_recomputed_scale}[0]), "
        f"T.cast(-448.0, \"float32\")), T.cast(448.0, \"float32\")))) * "
        f"{q_recomputed_scale}[0]"
    )
    body.append(f"{indent * 4}# q_dequant_bwd_policy: saved_prepared_fp8")
    q_scale_current = _row_phased_attention_scale_ref(
        q_scale_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        q_side=True,
    )
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    q_saved_ref = _row_phased_attention_value_ref(
        q_fp8_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        dim_expr=dim,
        q_side=True,
    )
    body.append(
        f"{indent * 5}{q_deq_values}[{dim}] = "
        f"fp8_e4m3fn_to_float({q_saved_ref}) * {q_scale_current}"
    )
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    indices_ref = _row_phased_attention_bwd_indices_ref(
        indices_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=kv_head,
        topk_expr=k_top,
    )
    body.append(f"{indent * 5}{sparse_indices}[{k_top}] = {indices_ref}")
    body.append(f"{indent * 4}{score_max}[0] = T.float32(-3.4028234663852886e38)")
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 5}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(
        f"{indent * 5}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 6}{score_accum}[0] = 0.0")
    body.append(f"{indent * 6}for {dot_dim} in T.serial(0, {head_dim}):")
    q_dot_ref = _row_phased_attention_value_ref(
        q_fp8_input,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        dim_expr=dot_dim,
        q_side=True,
    )
    kv_dot_ref = _row_phased_attention_selected_kv_value_ref(
        kv_fp8_input,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
        dim_expr=dot_dim,
    )
    body.append(
        f"{indent * 7}{score_accum}[0] = {score_accum}[0] + "
        f"({q_deq_values}[{dot_dim}] * fp8_e4m3fn_to_float({kv_dot_ref}))"
    )
    kv_scale_ref = _row_phased_attention_selected_kv_scale_ref(
        kv_scale_input,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
    )
    body.append(
        f"{indent * 6}{score_accum}[0] = {score_accum}[0] * "
        f"{kv_scale_ref} * {sm_scale}"
    )
    body.append(f"{indent * 5}else:")
    body.append(f"{indent * 6}{score_accum}[0] = T.float32(-3.4028234663852886e38)")
    body.append(f"{indent * 5}if {score_accum}[0] > {score_max}[0]:")
    body.append(f"{indent * 6}{score_max}[0] = {score_accum}[0]")
    body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
    body.append(f"{indent * 5}if {sinks} > {score_max}[0]:")
    body.append(f"{indent * 6}{score_max}[0] = {sinks}")
    body.append(f"{indent * 4}{sumexp}[0] = 0.0")
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 5}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(f"{indent * 5}{score_weights}[{k_top}] = 0.0")
    body.append(
        f"{indent * 5}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 6}{score_accum}[0] = 0.0")
    body.append(f"{indent * 6}for {dot_dim} in T.serial(0, {head_dim}):")
    body.append(
        f"{indent * 7}{score_accum}[0] = {score_accum}[0] + "
        f"({q_deq_values}[{dot_dim}] * fp8_e4m3fn_to_float({kv_dot_ref}))"
    )
    body.append(
        f"{indent * 6}{score_accum}[0] = {score_accum}[0] * "
        f"{kv_scale_ref} * {sm_scale}"
    )
    body.append(
        f"{indent * 6}{score_weight}[0] = T.exp({score_accum}[0] - {score_max}[0])"
    )
    body.append(f"{indent * 6}{score_weights}[{k_top}] = {score_weight}[0]")
    body.append(f"{indent * 6}{sumexp}[0] = {sumexp}[0] + {score_weight}[0]")
    body.append(f"{indent * 4}if {sink_enabled}[0] != 0.0:")
    body.append(
        f"{indent * 5}{sumexp}[0] = {sumexp}[0] + T.exp({sinks} - {score_max}[0])"
    )
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    body.append(f"{indent * 5}{value_accum}[0] = 0.0")
    body.append(f"{indent * 5}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 6}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(
        f"{indent * 6}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    kv_val_ref = _row_phased_attention_selected_kv_value_ref(
        kv_fp8_input,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
        dim_expr=dim,
    )
    body.append(
        f"{indent * 7}{value_accum}[0] = {value_accum}[0] + "
        f"({score_weights}[{k_top}] * fp8_e4m3fn_to_float({kv_val_ref}) * "
        f"{kv_scale_ref})"
    )
    body.append(f"{indent * 5}{context_values}[{dim}] = 0.0")
    body.append(f"{indent * 5}if {sumexp}[0] > 0.0:")
    body.append(f"{indent * 6}{context_values}[{dim}] = {value_accum}[0] / {sumexp}[0]")
    body.append(f"{indent * 5}{context_grads}[{dim}] = 0.0")
    body.append(f"{indent * 5}for {out_dim} in T.serial(0, {hidden_size}):")
    out_grad = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("attention_out", "out"),
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {out_dim}",
    )
    out_weight = _node_indexed_canonical_input_expr(
        node,
        "attention_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{out_dim} * {q_dim} + ({q_head} * {head_dim} + {dim})",
        default="1.0",
    )
    body.append(
        f"{indent * 6}{context_grads}[{dim}] = {context_grads}[{dim}] + "
        f"({out_grad} * {out_weight})"
    )
    if out_weight_grad is not None:
        out_weight_ref = _indexed_buffer_ref(
            out_weight_grad,
            access_by_buffer,
            f"{out_dim} * {q_dim} + ({q_head} * {head_dim} + {dim})",
        )
        body.append(
            f"{indent * 6}{out_weight_ref} = {out_weight_ref} + "
            f"({out_grad} * {context_values}[{dim}])"
        )
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 5}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(f"{indent * 5}{dp_accum}[0] = 0.0")
    body.append(
        f"{indent * 5}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 6}for {dim} in T.serial(0, {head_dim}):")
    body.append(
        f"{indent * 7}{dp_accum}[0] = {dp_accum}[0] + "
        f"({context_grads}[{dim}] * fp8_e4m3fn_to_float({kv_val_ref}) * {kv_scale_ref})"
    )
    body.append(f"{indent * 5}{dp_values}[{k_top}] = 0.0")
    body.append(f"{indent * 5}if {sumexp}[0] > 0.0:")
    body.append(
        f"{indent * 6}{dp_values}[{k_top}] = {dp_accum}[0]"
    )
    body.append(f"{indent * 4}{dp_accum}[0] = 0.0")
    body.append(f"{indent * 4}for {k_top} in T.serial(0, {topk}):")
    body.append(
        f"{indent * 5}{dp_accum}[0] = {dp_accum}[0] + "
        f"(({score_weights}[{k_top}] / T.max({sumexp}[0], T.float32(1.0e-20))) * "
        f"{dp_values}[{k_top}])"
    )
    body.append(f"{indent * 4}for {dim} in T.serial(0, {head_dim}):")
    body.append(f"{indent * 5}{dq_accum}[0] = 0.0")
    body.append(f"{indent * 5}for {k_top} in T.serial(0, {topk}):")
    body.append(f"{indent * 6}{sparse_index}[0] = {sparse_indices}[{k_top}]")
    body.append(
        f"{indent * 6}if {sparse_index}[0] >= 0 and "
        f"{sparse_index}[0] < {sequence_length}:"
    )
    body.append(f"{indent * 7}{ds_score}[0] = 0.0")
    body.append(f"{indent * 7}if {sumexp}[0] > 0.0:")
    body.append(
        f"{indent * 8}{ds_score}[0] = "
        f"({score_weights}[{k_top}] / {sumexp}[0]) * "
        f"({dp_values}[{k_top}] - {dp_accum}[0])"
    )
    body.append(
        f"{indent * 7}{dq_accum}[0] = {dq_accum}[0] + "
        f"({ds_score}[0] * fp8_e4m3fn_to_float({kv_val_ref}) * {kv_scale_ref})"
    )
    body.append(f"{indent * 7}{dkv_accum}[0] = {sm_scale} * {ds_score}[0] * {q_deq_values}[{dim}]")
    body.append(
        f"{indent * 7}{dkv_accum}[0] = {dkv_accum}[0] + "
        f"(({score_weights}[{k_top}] / T.max({sumexp}[0], T.float32(1.0e-20))) * "
        f"{context_grads}[{dim}])"
    )
    dkv_ref = _row_phased_attention_selected_kv_value_ref(
        kv_fp8_grad,
        access_by_buffer,
        shape_env,
        selected_row_expr=f"{sparse_index}[0]",
        head_expr=kv_head,
        dim_expr=dim,
    )
    body.append(
        f"{indent * 7}T.atomic_add({dkv_ref}, {dkv_accum}[0], "
        "memory_order=\"relaxed\")"
    )
    dq_ref = _row_phased_attention_value_ref(
        q_fp8_grad,
        access_by_buffer,
        shape_env,
        row_expr="row",
        head_expr=q_head,
        dim_expr=dim,
        q_side=True,
    )
    body.append(f"{indent * 5}{dq_ref} = {dq_accum}[0] * {sm_scale}")
    body.append(f"{indent * 3}T.sync_threads()")
    if out_bias_grad is not None:
        body.append(
            f"{indent * 3}for {out_dim} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        out_bias_ref = _indexed_buffer_ref(out_bias_grad, access_by_buffer, out_dim)
        out_grad = _node_indexed_canonical_or_positional_input_expr(
            node,
            ("attention_out", "out"),
            0,
            dtype_by_buffer,
            access_by_buffer,
            f"row * {hidden_size} + {out_dim}",
        )
        body.append(f"{indent * 4}{out_bias_ref} = {out_bias_ref} + {out_grad}")
        body.append(f"{indent * 3}T.sync_threads()")


def _append_row_phased_attention_qkv_projection_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    q_grad0 = _scratch_name(node, "attention_q_grad0")
    q_grad1 = _scratch_name(node, "attention_q_grad1")
    kv_grad0 = _scratch_name(node, "attention_kv_grad0")
    kv_grad1 = _scratch_name(node, "attention_kv_grad1")
    q_proj0 = _scratch_name(node, "attention_q_projected0")
    q_proj1 = _scratch_name(node, "attention_q_projected1")
    kv_proj0 = _scratch_name(node, "attention_kv_projected0")
    kv_proj1 = _scratch_name(node, "attention_kv_projected1")
    proj_grad0 = _scratch_name(node, "attention_projected_grad0")
    proj_grad1 = _scratch_name(node, "attention_projected_grad1")
    rope_grad_scratch = _scratch_name(node, "attention_rope_grad")
    rope_phase = _scratch_name(node, "attention_rope_phase")
    pair_flat = _scratch_name(node, "pair_flat")
    grad_flat = _scratch_name(node, "grad_flat")
    h = _scratch_name(node, "h")
    rope_d = _scratch_name(node, "rope_d")
    head = _scratch_name(node, "head")
    attention_hidden_values = _scratch_name(node, "attention_hidden_values")
    attention_hidden_sum_sq = _scratch_name(node, "attention_hidden_sum_sq")
    attention_hidden_inv_rms = _scratch_name(node, "attention_hidden_inv_rms")
    hidden_size = int(shape_env.hidden_size)
    q_heads = int(shape_env.attention_num_q_heads)
    kv_heads = int(shape_env.attention_num_kv_heads)
    head_dim = int(shape_env.attention_head_dim)
    rope_half = max(1, head_dim // 2)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    hidden_grad = _node_output_for_canonical_or_index(
        node,
        ("attention_hidden", "hidden"),
        0,
    )
    q_weight_grad = _node_output_for_canonical(node, "attention_q_proj_weight")
    q_bias_grad = _node_output_for_canonical(node, "attention_q_proj_bias")
    kv_weight_grad = _node_output_for_canonical(
        node,
        "attention_sparse_kv_proj_weight",
    )
    kv_bias_grad = _node_output_for_canonical(
        node,
        "attention_sparse_kv_proj_bias",
    )
    rope_grad = _node_output_for_canonical(node, "attention_rope_inv_freq")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(
        f"{indent * 3}# attention_qkv_projection_bwd_policy: "
        "exact_inverse_rope_weight_bias_hidden"
    )
    body.append(f"{indent * 3}if row == 0:")
    if q_weight_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, "
            f"{q_dim * hidden_size}, step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(q_weight_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if q_bias_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, {q_dim}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(q_bias_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if kv_weight_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, "
            f"{kv_dim * hidden_size}, step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(kv_weight_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if kv_bias_grad is not None:
        body.append(
            f"{indent * 4}for {grad_flat} in T.serial(lane, {kv_dim}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(kv_bias_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    if rope_grad is not None:
        body.append(
            f"{indent * 4}for {rope_d} in T.serial(lane, {rope_half}, "
            f"step={thread_count}):"
        )
        body.append(
            f"{indent * 5}{_indexed_buffer_ref(rope_grad, access_by_buffer, rope_d)} = 0.0"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    if hidden_grad is not None:
        body.append(
            f"{indent * 3}for {h} in T.serial(lane, {hidden_size}, "
            f"step={thread_count}):"
        )
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"row * {hidden_size} + {h}",
        )
        body.append(f"{indent * 4}{hidden_grad_ref} = 0.0")
        body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}{q_grad0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{q_grad1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_grad0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_grad1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{q_proj0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{q_proj1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_proj0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{kv_proj1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{proj_grad0} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{proj_grad1} = T.alloc_local((1,), \"float32\")"
    )
    body.append(
        f"{indent * 3}{rope_phase} = T.alloc_local((1,), \"float32\")"
    )
    if rope_grad is not None:
        body.append(
            f"{indent * 3}{rope_grad_scratch} = T.alloc_local((1,), \"float32\")"
        )
    body.append(f"{indent * 3}{attention_hidden_values} = T.alloc_local(({hidden_size},), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_sum_sq} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_inv_rms} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{attention_hidden_sum_sq}[0] = 0.0")
    body.append(f"{indent * 3}for {h} in T.serial(0, {hidden_size}):")
    hidden_after_m2rnn = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("attention_hidden", "hidden"),
        4,
        dtype_by_buffer,
        access_by_buffer,
        f"row * {hidden_size} + {h}",
    )
    body.append(f"{indent * 4}{attention_hidden_values}[{h}] = {hidden_after_m2rnn}")
    body.append(
        f"{indent * 4}{attention_hidden_sum_sq}[0] = {attention_hidden_sum_sq}[0] + "
        f"({attention_hidden_values}[{h}] * {attention_hidden_values}[{h}])"
    )
    body.append(
        f"{indent * 3}{attention_hidden_inv_rms}[0] = T.rsqrt(({attention_hidden_sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}for {h} in T.serial(0, {hidden_size}):")
    body.append(
        f"{indent * 4}{attention_hidden_values}[{h}] = "
        f"{attention_hidden_values}[{h}]"
    )
    body.append(
        f"{indent * 3}for {pair_flat} in T.serial(lane, {q_heads * rope_half}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {pair_flat} // {rope_half}")
    body.append(f"{indent * 4}{rope_d} = {pair_flat} % {rope_half}")
    q_value_index0 = f"row * {q_dim} + {head} * {head_dim} + {rope_d}"
    q_value_index1 = f"row * {q_dim} + {head} * {head_dim} + {rope_d} + {rope_half}"
    q_fp8_grad0 = _node_indexed_canonical_input_expr(
        node, "q_fp8", dtype_by_buffer, access_by_buffer, q_value_index0
    )
    q_fp8_grad1 = _node_indexed_canonical_input_expr(
        node, "q_fp8", dtype_by_buffer, access_by_buffer, q_value_index1
    )
    q_scale_grad_seen = _node_indexed_canonical_input_expr(
        node,
        "q_scale",
        dtype_by_buffer,
        access_by_buffer,
        f"row * {q_heads} + {head}",
    )
    body.append(f"{indent * 4}{q_grad0}[0] = {q_fp8_grad0}")
    body.append(f"{indent * 4}{q_grad1}[0] = {q_fp8_grad1}")
    body.append(f"{indent * 4}{q_grad0}[0] = {q_grad0}[0] + {q_scale_grad_seen}")
    body.append(f"{indent * 4}{q_grad0}[0] = {q_grad0}[0] - {q_scale_grad_seen}")
    q_bias0 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d}",
    )
    q_bias1 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d} + {rope_half}",
    )
    body.append(f"{indent * 4}{q_proj0}[0] = {q_bias0}")
    body.append(f"{indent * 4}{q_proj1}[0] = {q_bias1}")
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    q_weight0 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        default="1.0",
    )
    q_weight1 = _node_indexed_canonical_input_expr(
        node,
        "attention_q_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        default="1.0",
    )
    body.append(
        f"{indent * 5}{q_proj0}[0] = {q_proj0}[0] + ({hidden} * {q_weight0})"
    )
    body.append(
        f"{indent * 5}{q_proj1}[0] = {q_proj1}[0] + ({hidden} * {q_weight1})"
    )
    rope_input = _node_indexed_canonical_input_expr(
        node,
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        rope_d,
    )
    body.append(
        f"{indent * 4}{rope_phase}[0] = T.cast(row, \"float32\") * {rope_input}"
    )
    body.append(
        f"{indent * 4}{proj_grad0}[0] = "
        f"({q_grad0}[0] * T.cos({rope_phase}[0])) - "
        f"({q_grad1}[0] * T.sin({rope_phase}[0]))"
    )
    body.append(
        f"{indent * 4}{proj_grad1}[0] = "
        f"({q_grad0}[0] * T.sin({rope_phase}[0])) + "
        f"({q_grad1}[0] * T.cos({rope_phase}[0]))"
    )
    if q_bias_grad is not None:
        q_bias_ref0 = _indexed_buffer_ref(
            q_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d}",
        )
        q_bias_ref1 = _indexed_buffer_ref(
            q_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d} + {rope_half}",
        )
        body.append(f"{indent * 4}{q_bias_ref0} = {q_bias_ref0} + {proj_grad0}[0]")
        body.append(f"{indent * 4}{q_bias_ref1} = {q_bias_ref1} + {proj_grad1}[0]")
    if rope_grad is not None:
        body.append(
            f"{indent * 4}{rope_grad_scratch}[0] = "
            f"({q_grad0}[0] * ((-{q_proj0}[0] * T.sin({rope_phase}[0])) + "
            f"({q_proj1}[0] * T.cos({rope_phase}[0])))) + "
            f"({q_grad1}[0] * ((-{q_proj1}[0] * T.sin({rope_phase}[0])) - "
            f"({q_proj0}[0] * T.cos({rope_phase}[0]))))"
        )
        rope_ref = _indexed_buffer_ref(rope_grad, access_by_buffer, rope_d)
        body.append(
            f"{indent * 4}T.atomic_add({rope_ref}, "
            f"{rope_grad_scratch}[0] * T.cast(row, \"float32\"), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    if q_weight_grad is not None:
        q_weight_ref0 = _indexed_buffer_ref(
            q_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        )
        q_weight_ref1 = _indexed_buffer_ref(
            q_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}{q_weight_ref0} = {q_weight_ref0} + "
            f"({hidden} * {proj_grad0}[0])"
        )
        body.append(
            f"{indent * 5}{q_weight_ref1} = {q_weight_ref1} + "
            f"({hidden} * {proj_grad1}[0])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"row * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}T.atomic_add({hidden_grad_ref}, "
            f"(({proj_grad0}[0] * {q_weight0}) + ({proj_grad1}[0] * {q_weight1})), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 3}T.sync_threads()")
    body.append(
        f"{indent * 3}for {pair_flat} in T.serial(lane, {kv_heads * rope_half}, "
        f"step={thread_count}):"
    )
    body.append(f"{indent * 4}{head} = {pair_flat} // {rope_half}")
    body.append(f"{indent * 4}{rope_d} = {pair_flat} % {rope_half}")
    kv_value_index0 = f"row * {kv_dim} + {head} * {head_dim} + {rope_d}"
    kv_value_index1 = f"row * {kv_dim} + {head} * {head_dim} + {rope_d} + {rope_half}"
    kv_fp8_grad0 = _node_indexed_canonical_input_expr(
        node, "kv_fp8", dtype_by_buffer, access_by_buffer, kv_value_index0
    )
    kv_fp8_grad1 = _node_indexed_canonical_input_expr(
        node, "kv_fp8", dtype_by_buffer, access_by_buffer, kv_value_index1
    )
    kv_scale_grad_seen = _node_indexed_canonical_input_expr(
        node,
        "kv_scale",
        dtype_by_buffer,
        access_by_buffer,
        f"row * {kv_heads} + {head}",
    )
    body.append(f"{indent * 4}{kv_grad0}[0] = {kv_fp8_grad0}")
    body.append(f"{indent * 4}{kv_grad1}[0] = {kv_fp8_grad1}")
    body.append(f"{indent * 4}{kv_grad0}[0] = {kv_grad0}[0] + {kv_scale_grad_seen}")
    body.append(f"{indent * 4}{kv_grad0}[0] = {kv_grad0}[0] - {kv_scale_grad_seen}")
    kv_bias0 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d}",
    )
    kv_bias1 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim} + {rope_d} + {rope_half}",
    )
    body.append(f"{indent * 4}{kv_proj0}[0] = {kv_bias0}")
    body.append(f"{indent * 4}{kv_proj1}[0] = {kv_bias1}")
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    kv_weight0 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        default="1.0",
    )
    kv_weight1 = _node_indexed_canonical_input_expr(
        node,
        "attention_sparse_kv_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        default="1.0",
    )
    body.append(
        f"{indent * 5}{kv_proj0}[0] = {kv_proj0}[0] + ({hidden} * {kv_weight0})"
    )
    body.append(
        f"{indent * 5}{kv_proj1}[0] = {kv_proj1}[0] + ({hidden} * {kv_weight1})"
    )
    rope_input = _node_indexed_canonical_input_expr(
        node,
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        rope_d,
    )
    body.append(
        f"{indent * 4}{rope_phase}[0] = T.cast(row, \"float32\") * {rope_input}"
    )
    body.append(
        f"{indent * 4}{proj_grad0}[0] = "
        f"({kv_grad0}[0] * T.cos({rope_phase}[0])) - "
        f"({kv_grad1}[0] * T.sin({rope_phase}[0]))"
    )
    body.append(
        f"{indent * 4}{proj_grad1}[0] = "
        f"({kv_grad0}[0] * T.sin({rope_phase}[0])) + "
        f"({kv_grad1}[0] * T.cos({rope_phase}[0]))"
    )
    if kv_bias_grad is not None:
        kv_bias_ref0 = _indexed_buffer_ref(
            kv_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d}",
        )
        kv_bias_ref1 = _indexed_buffer_ref(
            kv_bias_grad,
            access_by_buffer,
            f"{head} * {head_dim} + {rope_d} + {rope_half}",
        )
        body.append(f"{indent * 4}{kv_bias_ref0} = {kv_bias_ref0} + {proj_grad0}[0]")
        body.append(f"{indent * 4}{kv_bias_ref1} = {kv_bias_ref1} + {proj_grad1}[0]")
    if rope_grad is not None:
        body.append(
            f"{indent * 4}{rope_grad_scratch}[0] = "
            f"({kv_grad0}[0] * ((-{kv_proj0}[0] * T.sin({rope_phase}[0])) + "
            f"({kv_proj1}[0] * T.cos({rope_phase}[0])))) + "
            f"({kv_grad1}[0] * ((-{kv_proj1}[0] * T.sin({rope_phase}[0])) - "
            f"({kv_proj0}[0] * T.cos({rope_phase}[0]))))"
        )
        rope_ref = _indexed_buffer_ref(rope_grad, access_by_buffer, rope_d)
        body.append(
            f"{indent * 4}T.atomic_add({rope_ref}, "
            f"{rope_grad_scratch}[0] * T.cast(row, \"float32\"), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 4}for {h} in T.serial(0, {hidden_size}):")
    hidden = f"{attention_hidden_values}[{h}]"
    if kv_weight_grad is not None:
        kv_weight_ref0 = _indexed_buffer_ref(
            kv_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d}) * {hidden_size} + {h}",
        )
        kv_weight_ref1 = _indexed_buffer_ref(
            kv_weight_grad,
            access_by_buffer,
            f"({head} * {head_dim} + {rope_d} + {rope_half}) * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}{kv_weight_ref0} = {kv_weight_ref0} + "
            f"({hidden} * {proj_grad0}[0])"
        )
        body.append(
            f"{indent * 5}{kv_weight_ref1} = {kv_weight_ref1} + "
            f"({hidden} * {proj_grad1}[0])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"row * {hidden_size} + {h}",
        )
        body.append(
            f"{indent * 5}T.atomic_add({hidden_grad_ref}, "
            f"(({proj_grad0}[0] * {kv_weight0}) + ({proj_grad1}[0] * {kv_weight1})), "
            "memory_order=\"relaxed\")"
        )
    body.append(f"{indent * 3}T.sync_threads()")

def _append_row_phased_m2rnn_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    stage_grad = _scratch_name(node, "m2rnn_stage_grad")
    project_grad = _scratch_name(node, "m2rnn_project_grad")
    conv_grad = _scratch_name(node, "m2rnn_conv_grad")
    conv_pre = _scratch_name(node, "m2rnn_conv_pre")
    conv = _scratch_name(node, "m2rnn_conv_vec")
    projected = _scratch_name(node, "m2rnn_projected_vec")
    post = _scratch_name(node, "m2rnn_post_vec")
    post_grad = _scratch_name(node, "m2rnn_post_grad")
    h_prev = _scratch_name(node, "m2rnn_h_prev")
    h_next = _scratch_name(node, "m2rnn_h_next")
    dh = _scratch_name(node, "m2rnn_state_grad")
    dh_prev = _scratch_name(node, "m2rnn_state_prev_grad")
    sum_sq = _scratch_name(node, "m2rnn_sum_sq")
    dot = _scratch_name(node, "m2rnn_norm_dot")
    accum = _scratch_name(node, "m2rnn_accum")
    decay = _scratch_name(node, "m2rnn_decay")
    tanh_val = _scratch_name(node, "m2rnn_tanh")
    scalar0 = _scratch_name(node, "m2rnn_scalar0")
    scalar1 = _scratch_name(node, "m2rnn_scalar1")
    scalar2 = _scratch_name(node, "m2rnn_scalar2")
    time_rev = _scratch_name(node, "time_rev")
    time_idx = _scratch_name(node, "time_idx")
    replay_time = _scratch_name(node, "replay_time")
    src_row = _scratch_name(node, "src_row")
    src_hist = _scratch_name(node, "src_hist")
    hidden_loop = _scratch_name(node, "hidden_loop")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    kernel_pos = _scratch_name(node, "kernel_pos")
    state_idx = _scratch_name(node, "state_idx")
    grad_flat = _scratch_name(node, "grad_flat")
    feature = _scratch_name(node, "feature")
    head = _scratch_name(node, "head")
    kk = _scratch_name(node, "kk")
    vv = _scratch_name(node, "vv")
    vv_inner = _scratch_name(node, "vv_inner")
    out_dim = _scratch_name(node, "out_dim")
    partial = _scratch_name(node, "partial")
    hidden_size = int(shape_env.hidden_size)
    sequence_length = int(shape_env.sequence_length)
    in_proj_dim = int(shape_env.m2rnn_in_proj_dim)
    conv_dim = int(shape_env.m2rnn_conv_dim)
    kernel = int(shape_env.m2rnn_conv_kernel)
    history_len = max(0, kernel - 1)
    total_heads = int(shape_env.m2rnn_num_heads)
    q_heads = int(shape_env.m2rnn_num_q_heads)
    k_heads = int(shape_env.m2rnn_num_k_heads)
    v_heads = int(shape_env.m2rnn_num_v_heads)
    f_heads = int(shape_env.m2rnn_num_f_heads)
    g_heads = int(shape_env.m2rnn_num_g_heads)
    w_heads = int(shape_env.m2rnn_num_weight_heads)
    k_dim = int(shape_env.m2rnn_k_head_dim)
    v_dim = int(shape_env.m2rnn_v_head_dim)
    state_extent = (
        int(shape_env.m2rnn_num_weight_heads)
        * int(shape_env.m2rnn_v_head_dim)
        * int(shape_env.m2rnn_v_head_dim)
    )
    full_state_extent = total_heads * k_dim * v_dim
    features = total_heads * v_dim
    q_offset = 0
    k_offset = int(shape_env.m2rnn_q_dim)
    v_offset = k_offset + int(shape_env.m2rnn_k_dim)
    f_offset = conv_dim
    g_offset = conv_dim + f_heads
    q_group = total_heads // q_heads
    k_group = total_heads // k_heads
    v_group = total_heads // v_heads
    f_group = total_heads // f_heads
    g_repeat = total_heads // g_heads
    w_group = total_heads // w_heads
    hidden_grad = _node_output_for_canonical_or_index(
        node,
        ("m2rnn_hidden", "hidden"),
        0,
    )
    in_proj_weight_grad = _node_output_for_canonical(node, "m2rnn_in_proj_weight")
    conv_weight_grad = _node_output_for_canonical(node, "m2rnn_conv_weight")
    conv_bias_grad = _node_output_for_canonical(node, "m2rnn_conv_bias")
    state_weight_grad = _node_output_for_canonical(node, "m2rnn_state_weight")
    a_log_grad = _node_output_for_canonical(node, "m2rnn_A_log")
    dt_bias_grad = _node_output_for_canonical(node, "m2rnn_dt_bias")
    d_grad = _node_output_for_canonical(node, "m2rnn_D")
    g_norm_weight_grad = _node_output_for_canonical(node, "m2rnn_g_norm_weight")
    out_proj_weight_grad = _node_output_for_canonical(node, "m2rnn_out_proj_weight")
    h0_grad = _node_output_for_canonical(node, "m2rnn_h0")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    body.append(f"{indent * 3}# m2rnn_bwd_policy: exact_reverse_recompute")
    body.append(f"{indent * 3}{stage_grad} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{project_grad} = T.alloc_local(({in_proj_dim},), \"float32\")")
    body.append(f"{indent * 3}{conv_grad} = T.alloc_local(({conv_dim},), \"float32\")")
    body.append(f"{indent * 3}{conv_pre} = T.alloc_local(({conv_dim},), \"float32\")")
    body.append(f"{indent * 3}{conv} = T.alloc_local(({conv_dim},), \"float32\")")
    body.append(f"{indent * 3}{projected} = T.alloc_local(({in_proj_dim},), \"float32\")")
    body.append(f"{indent * 3}{post} = T.alloc_local(({features},), \"float32\")")
    body.append(f"{indent * 3}{post_grad} = T.alloc_local(({features},), \"float32\")")
    body.append(f"{indent * 3}{h_prev} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{h_next} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{dh} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{dh_prev} = T.alloc_local(({full_state_extent},), \"float32\")")
    body.append(f"{indent * 3}{sum_sq} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{dot} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{accum} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{decay} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{tanh_val} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{scalar0} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{scalar1} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}{scalar2} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}if row == 0:")
    body.append(f"{indent * 4}if lane == 0:")
    for output_name, extent in (
        (in_proj_weight_grad, in_proj_dim * hidden_size),
        (conv_weight_grad, conv_dim * kernel),
        (conv_bias_grad, conv_dim),
        (state_weight_grad, state_extent),
        (a_log_grad, int(shape_env.m2rnn_num_heads)),
        (dt_bias_grad, int(shape_env.m2rnn_num_heads)),
        (d_grad, features),
        (g_norm_weight_grad, features),
        (out_proj_weight_grad, hidden_size * features),
        (h0_grad, full_state_extent),
    ):
        if output_name is None:
            continue
        body.append(
            f"{indent * 5}for {state_idx} in T.serial(0, {extent}):"
        )
        body.append(
            f"{indent * 6}{_indexed_buffer_ref(output_name, access_by_buffer, state_idx)} = 0.0"
        )
    if hidden_grad is not None:
        body.append(
            f"{indent * 5}for {grad_flat} in T.serial(0, "
            f"{sequence_length * hidden_size}):"
        )
        body.append(
            f"{indent * 6}{_indexed_buffer_ref(hidden_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    body.append(f"{indent * 5}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 6}{dh}[{state_idx}] = 0.0")
    body.append(f"{indent * 5}for {time_rev} in T.serial(0, {sequence_length}):")
    body.append(f"{indent * 6}{time_idx} = {sequence_length - 1} - {time_rev}")
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 7}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{indent * 7}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{indent * 7}{vv} = {state_idx} % {v_dim}")
    h0_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_h0",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
    )
    body.append(f"{indent * 7}{h_prev}[{state_idx}] = {h0_expr}")
    body.append(f"{indent * 7}{h_next}[{state_idx}] = {h0_expr}")
    body.append(f"{indent * 6}for {replay_time} in T.serial(0, {sequence_length}):")
    body.append(f"{indent * 7}if {replay_time} <= {time_idx}:")
    body.append(f"{indent * 8}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 9}{accum}[0] = 0.0")
    body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("m2rnn_hidden", "hidden"),
        1,
        dtype_by_buffer,
        access_by_buffer,
        f"{replay_time} * {hidden_size} + {hidden_loop}",
    )
    weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{proj_dim} * {hidden_size} + {hidden_loop}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{accum}[0] = {accum}[0] + "
        f"({hidden_expr} * {weight_expr})"
    )
    body.append(f"{indent * 9}{projected}[{proj_dim}] = {accum}[0]")
    body.append(f"{indent * 8}for {conv_ch} in T.serial(0, {conv_dim}):")
    conv_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        conv_ch,
        default="0.0",
    )
    body.append(f"{indent * 9}{conv_pre}[{conv_ch}] = {conv_bias_expr}")
    if history_len > 0:
        body.append(f"{indent * 9}for {kernel_pos} in T.serial(0, {history_len}):")
        body.append(
            f"{indent * 10}{src_row} = {replay_time} - {history_len} + {kernel_pos}"
        )
        body.append(f"{indent * 10}if {src_row} >= 0:")
        body.append(f"{indent * 11}{scalar0}[0] = 0.0")
        body.append(f"{indent * 11}for {hidden_loop} in T.serial(0, {hidden_size}):")
        src_hidden_expr = _node_indexed_canonical_or_positional_input_expr(
            node,
            ("m2rnn_hidden", "hidden"),
            1,
            dtype_by_buffer,
            access_by_buffer,
            f"{src_row} * {hidden_size} + {hidden_loop}",
        )
        src_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {hidden_size} + {hidden_loop}",
            default="1.0",
        )
        body.append(
            f"{indent * 12}{scalar0}[0] = {scalar0}[0] + "
            f"({src_hidden_expr} * {src_weight_expr})"
        )
        body.append(f"{indent * 10}else:")
        body.append(f"{indent * 11}{src_hist} = {kernel_pos} + {replay_time}")
        conv_state_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_state",
            dtype_by_buffer,
            access_by_buffer,
            f"{src_hist} * {conv_dim} + {conv_ch}",
            default="0.0",
        )
        body.append(f"{indent * 11}{scalar0}[0] = {conv_state_expr}")
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(
            f"{indent * 10}{conv_pre}[{conv_ch}] = {conv_pre}[{conv_ch}] + "
            f"({scalar0}[0] * {conv_weight_expr})"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 9}{conv_pre}[{conv_ch}] = {conv_pre}[{conv_ch}] + "
        f"({projected}[{conv_ch}] * {current_conv_weight_expr})"
    )
    body.append(f"{indent * 9}{scalar0}[0] = 1.0 / (1.0 + T.exp(-{conv_pre}[{conv_ch}]))")
    body.append(f"{indent * 9}{conv}[{conv_ch}] = {conv_pre}[{conv_ch}] * {scalar0}[0]")
    body.append(f"{indent * 8}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 9}{head} = {state_idx} // {k_dim * v_dim}")
    body.append(f"{indent * 9}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
    body.append(f"{indent * 9}{vv} = {state_idx} % {v_dim}")
    body.append(f"{indent * 9}{accum}[0] = 0.0")
    body.append(f"{indent * 9}for {vv_inner} in T.serial(0, {v_dim}):")
    state_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{accum}[0] = {accum}[0] + "
        f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] * "
        f"{state_weight_expr})"
    )
    a_log_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    dt_bias_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    f_src = f"({head} // {f_group})"
    body.append(
        f"{indent * 9}{decay}[0] = T.exp(-T.exp({a_log_expr}) * "
        f"T.log(1.0 + T.exp({projected}[{f_offset} + {f_src}] + {dt_bias_expr})))"
    )
    k_val = f"{conv}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}]"
    v_val = f"{conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}]"
    body.append(f"{indent * 9}{tanh_val}[0] = T.tanh({accum}[0] + ({k_val} * {v_val}))")
    body.append(
        f"{indent * 9}{h_next}[{state_idx}] = "
        f"({decay}[0] * {h_prev}[{state_idx}]) + "
        f"((1.0 - {decay}[0]) * {tanh_val}[0])"
    )
    body.append(f"{indent * 8}if {replay_time} < {time_idx}:")
    body.append(f"{indent * 9}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 10}{h_prev}[{state_idx}] = {h_next}[{state_idx}]")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{head} = {feature} // {v_dim}")
    body.append(f"{indent * 7}{vv} = {feature} % {v_dim}")
    body.append(f"{indent * 7}{post}[{feature}] = 0.0")
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(
        f"{indent * 8}{post}[{feature}] = {post}[{feature}] + "
        f"({conv}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] * "
        f"{h_next}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}])"
    )
    d_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="0.0",
    )
    body.append(
        f"{indent * 7}{post}[{feature}] = "
        f"({post}[{feature}] + "
        f"({conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] * {d_expr}))"
    )
    g_flat = f"({feature} // {g_repeat})"
    body.append(f"{indent * 7}{scalar0}[0] = {projected}[{g_offset} + {g_flat}]")
    body.append(f"{indent * 7}{scalar1}[0] = 1.0 / (1.0 + T.exp(-{scalar0}[0]))")
    body.append(f"{indent * 7}{post}[{feature}] = {post}[{feature}] * {scalar0}[0] * {scalar1}[0]")
    body.append(f"{indent * 6}{sum_sq}[0] = 0.0")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{sum_sq}[0] = {sum_sq}[0] + ({post}[{feature}] * {post}[{feature}])")
    body.append(f"{indent * 6}{scalar2}[0] = T.rsqrt(({sum_sq}[0] / {float(features):.1f}) + 0.00001)")
    body.append(f"{indent * 6}{dot}[0] = 0.0")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{post_grad}[{feature}] = 0.0")
    body.append(f"{indent * 7}for {out_dim} in T.serial(0, {hidden_size}):")
    delta_grad_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("m2rnn_delta",),
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"{time_idx} * {hidden_size} + {out_dim}",
    )
    out_proj_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{out_dim} * {features} + {feature}",
        default="1.0",
    )
    body.append(f"{indent * 8}{stage_grad}[0] = {delta_grad_expr}")
    body.append(f"{indent * 8}{post_grad}[{feature}] = {post_grad}[{feature}] + ({stage_grad}[0] * {out_proj_weight_expr})")
    if out_proj_weight_grad is not None:
        out_proj_grad_ref = _indexed_buffer_ref(
            out_proj_weight_grad,
            access_by_buffer,
            f"{out_dim} * {features} + {feature}",
        )
        g_norm_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_g_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            feature,
            default="1.0",
        )
        body.append(
            f"{indent * 8}{out_proj_grad_ref} = {out_proj_grad_ref} + "
            f"({stage_grad}[0] * {post}[{feature}] * {scalar2}[0] * {g_norm_weight_expr})"
        )
    g_norm_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_g_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="1.0",
    )
    body.append(
        f"{indent * 7}{dot}[0] = {dot}[0] + "
        f"({post_grad}[{feature}] * {g_norm_weight_expr} * {post}[{feature}])"
    )
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    g_norm_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_g_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="1.0",
    )
    if g_norm_weight_grad is not None:
        g_norm_grad_ref = _indexed_buffer_ref(g_norm_weight_grad, access_by_buffer, feature)
        body.append(
            f"{indent * 7}{g_norm_grad_ref} = {g_norm_grad_ref} + "
            f"({post_grad}[{feature}] * {post}[{feature}] * {scalar2}[0])"
        )
    body.append(
        f"{indent * 7}{post_grad}[{feature}] = {scalar2}[0] * "
        f"(({post_grad}[{feature}] * {g_norm_weight_expr}) - "
        f"({post}[{feature}] * {dot}[0] * {scalar2}[0] * {scalar2}[0] / {float(features):.1f}))"
    )
    body.append(f"{indent * 6}for {conv_ch} in T.serial(0, {conv_dim}):")
    body.append(f"{indent * 7}{conv_grad}[{conv_ch}] = 0.0")
    body.append(f"{indent * 6}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 7}{project_grad}[{proj_dim}] = 0.0")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {features}):")
    body.append(f"{indent * 7}{head} = {feature} // {v_dim}")
    body.append(f"{indent * 7}{vv} = {feature} % {v_dim}")
    body.append(f"{indent * 7}{accum}[0] = 0.0")
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(
        f"{indent * 8}{accum}[0] = {accum}[0] + "
        f"({conv}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] * "
        f"{h_next}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}])"
    )
    d_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        feature,
        default="0.0",
    )
    body.append(
        f"{indent * 7}{accum}[0] = {accum}[0] + "
        f"({conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] * {d_expr})"
    )
    g_flat = f"({feature} // {g_repeat})"
    body.append(f"{indent * 7}{scalar0}[0] = {projected}[{g_offset} + {g_flat}]")
    body.append(f"{indent * 7}{scalar1}[0] = 1.0 / (1.0 + T.exp(-{scalar0}[0]))")
    body.append(f"{indent * 7}{scalar2}[0] = {post_grad}[{feature}] * {scalar0}[0] * {scalar1}[0]")
    body.append(
        f"{indent * 7}{project_grad}[{g_offset} + {g_flat}] = "
        f"{project_grad}[{g_offset} + {g_flat}] + "
        f"({post_grad}[{feature}] * {accum}[0] * {scalar1}[0] * "
        f"(1.0 + ({scalar0}[0] * (1.0 - {scalar1}[0]))))"
    )
    if d_grad is not None:
        d_grad_ref = _indexed_buffer_ref(d_grad, access_by_buffer, feature)
        body.append(
            f"{indent * 7}{d_grad_ref} = {d_grad_ref} + "
            f"({scalar2}[0] * {conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}])"
        )
    body.append(
        f"{indent * 7}{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] = "
        f"{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] + "
        f"({scalar2}[0] * {d_expr})"
    )
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(
        f"{indent * 8}{conv_grad}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] = "
        f"{conv_grad}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}] + "
        f"({scalar2}[0] * {h_next}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}])"
    )
    body.append(
        f"{indent * 8}{dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] = "
        f"{dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] + "
        f"({scalar2}[0] * {conv}[{q_offset} + (({head} // {q_group}) * {k_dim}) + {kk}])"
    )
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 7}{dh_prev}[{state_idx}] = 0.0")
    body.append(f"{indent * 6}for {head} in T.serial(0, {total_heads}):")
    body.append(f"{indent * 7}{scalar0}[0] = 0.0")
    a_log_expr_head = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    dt_bias_expr_head = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    body.append(
        f"{indent * 7}{scalar1}[0] = T.log(1.0 + "
        f"T.exp({projected}[{f_offset} + ({head} // {f_group})] + {dt_bias_expr_head}))"
    )
    body.append(f"{indent * 7}{scalar2}[0] = T.exp({a_log_expr_head})")
    body.append(
        f"{indent * 7}{decay}[0] = T.exp(-{scalar2}[0] * {scalar1}[0])"
    )
    body.append(f"{indent * 7}for {kk} in T.serial(0, {k_dim}):")
    body.append(f"{indent * 8}for {vv} in T.serial(0, {v_dim}):")
    body.append(f"{indent * 9}{accum}[0] = 0.0")
    body.append(f"{indent * 9}for {vv_inner} in T.serial(0, {v_dim}):")
    state_weight_expr_inner = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{accum}[0] = {accum}[0] + "
        f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] * "
        f"{state_weight_expr_inner})"
    )
    body.append(
        f"{indent * 9}{tanh_val}[0] = "
        f"({h_next}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] - "
        f"({decay}[0] * "
        f"{h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}])) / "
        f"(1.0 - {decay}[0])"
    )
    body.append(
        f"{indent * 9}{scalar0}[0] = {scalar0}[0] + "
        f"({dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] * "
        f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] - {tanh_val}[0]))"
    )
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"{dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] * "
        f"(1.0 - {decay}[0]) * (1.0 - ({tanh_val}[0] * {tanh_val}[0]))"
    )
    body.append(
        f"{indent * 9}{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] = "
        f"{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] + "
        f"({dh}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}] * {decay}[0])"
    )
    body.append(f"{indent * 9}for {vv_inner} in T.serial(0, {v_dim}):")
    state_weight_expr_rev = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        default="1.0",
    )
    body.append(
        f"{indent * 10}{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] = "
        f"{dh_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] + "
        f"({scalar2}[0] * {state_weight_expr_rev})"
    )
    if state_weight_grad is not None:
        state_weight_grad_ref = _indexed_buffer_ref(
            state_weight_grad,
            access_by_buffer,
            f"({head} // {w_group}) * {v_dim * v_dim} + {vv_inner} * {v_dim} + {vv}",
        )
        body.append(
            f"{indent * 10}{state_weight_grad_ref} = {state_weight_grad_ref} + "
            f"({h_prev}[{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv_inner}] * {scalar2}[0])"
        )
    body.append(
        f"{indent * 9}{conv_grad}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}] = "
        f"{conv_grad}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}] + "
        f"({scalar2}[0] * {conv}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}])"
    )
    body.append(
        f"{indent * 9}{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] = "
        f"{conv_grad}[{v_offset} + (({head} // {v_group}) * {v_dim}) + {vv}] + "
        f"({scalar2}[0] * {conv}[{k_offset} + (({head} // {k_group}) * {k_dim}) + {kk}])"
    )
    body.append(f"{indent * 7}{scalar0}[0] = {scalar0}[0] * {decay}[0]")
    if a_log_grad is not None:
        a_log_grad_ref = _indexed_buffer_ref(a_log_grad, access_by_buffer, head)
        body.append(
            f"{indent * 7}{a_log_grad_ref} = {a_log_grad_ref} + "
            f"({scalar0}[0] * (-T.exp({a_log_expr_head}) * {scalar1}[0]))"
        )
    body.append(
        f"{indent * 7}{scalar0}[0] = {scalar0}[0] * "
        f"(-T.exp({a_log_expr_head})) * "
        f"(1.0 / (1.0 + T.exp(-({projected}[{f_offset} + ({head} // {f_group})] + {dt_bias_expr_head}))))"
    )
    body.append(
        f"{indent * 7}{project_grad}[{f_offset} + ({head} // {f_group})] = "
        f"{project_grad}[{f_offset} + ({head} // {f_group})] + {scalar0}[0]"
    )
    if dt_bias_grad is not None:
        dt_bias_grad_ref = _indexed_buffer_ref(dt_bias_grad, access_by_buffer, head)
        body.append(f"{indent * 7}{dt_bias_grad_ref} = {dt_bias_grad_ref} + {scalar0}[0]")
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {full_state_extent}):")
    body.append(f"{indent * 7}{dh}[{state_idx}] = {dh_prev}[{state_idx}]")
    body.append(f"{indent * 6}for {conv_ch} in T.serial(0, {conv_dim}):")
    body.append(f"{indent * 7}{scalar0}[0] = 1.0 / (1.0 + T.exp(-{conv_pre}[{conv_ch}]))")
    body.append(
        f"{indent * 7}{scalar1}[0] = {conv_grad}[{conv_ch}] * {scalar0}[0] * "
        f"(1.0 + ({conv_pre}[{conv_ch}] * (1.0 - {scalar0}[0])))"
    )
    if conv_bias_grad is not None:
        conv_bias_grad_ref = _indexed_buffer_ref(conv_bias_grad, access_by_buffer, conv_ch)
        body.append(f"{indent * 7}{conv_bias_grad_ref} = {conv_bias_grad_ref} + {scalar1}[0]")
    if history_len > 0:
        body.append(f"{indent * 7}for {kernel_pos} in T.serial(0, {history_len}):")
        body.append(
            f"{indent * 8}{src_row} = {time_idx} - {history_len} + {kernel_pos}"
        )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        body.append(f"{indent * 9}{scalar2}[0] = 0.0")
        body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
        src_hidden_expr = _node_indexed_canonical_or_positional_input_expr(
            node,
            ("m2rnn_hidden", "hidden"),
            1,
            dtype_by_buffer,
            access_by_buffer,
            f"{src_row} * {hidden_size} + {hidden_loop}",
        )
        src_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {hidden_size} + {hidden_loop}",
            default="1.0",
        )
        body.append(
            f"{indent * 10}{scalar2}[0] = {scalar2}[0] + "
            f"({src_hidden_expr} * {src_weight_expr})"
        )
        body.append(f"{indent * 8}else:")
        body.append(f"{indent * 9}{src_hist} = {kernel_pos} + {time_idx}")
        conv_state_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_state",
            dtype_by_buffer,
            access_by_buffer,
            f"{src_hist} * {conv_dim} + {conv_ch}",
            default="0.0",
        )
        body.append(f"{indent * 9}{scalar2}[0] = {conv_state_expr}")
        if conv_weight_grad is not None:
            conv_weight_grad_ref = _indexed_buffer_ref(
                conv_weight_grad,
                access_by_buffer,
                f"{conv_ch} * {kernel} + {kernel_pos}",
            )
            body.append(
                f"{indent * 8}{conv_weight_grad_ref} = {conv_weight_grad_ref} + "
                f"({scalar1}[0] * {scalar2}[0])"
            )
        conv_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {kernel_pos}",
            default="1.0",
        )
        body.append(f"{indent * 8}if {src_row} == {time_idx}:")
        body.append(
            f"{indent * 9}{project_grad}[{conv_ch}] = "
            f"{project_grad}[{conv_ch}] + ({scalar1}[0] * {conv_weight_expr})"
        )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        if in_proj_weight_grad is not None:
            body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
            src_hidden_expr = _node_indexed_canonical_or_positional_input_expr(
                node,
                ("m2rnn_hidden", "hidden"),
                1,
                dtype_by_buffer,
                access_by_buffer,
                f"{src_row} * {hidden_size} + {hidden_loop}",
            )
            in_proj_grad_ref = _indexed_buffer_ref(
                in_proj_weight_grad,
                access_by_buffer,
                f"{conv_ch} * {hidden_size} + {hidden_loop}",
            )
            body.append(
                f"{indent * 10}{in_proj_grad_ref} = {in_proj_grad_ref} + "
                f"({src_hidden_expr} * {scalar1}[0] * {conv_weight_expr})"
            )
        if hidden_grad is not None:
            body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
            hidden_grad_ref = _indexed_buffer_ref(
                hidden_grad,
                access_by_buffer,
                f"{src_row} * {hidden_size} + {hidden_loop}",
            )
            in_proj_weight_expr = _node_indexed_canonical_input_expr(
                node,
                "m2rnn_in_proj_weight",
                dtype_by_buffer,
                access_by_buffer,
                f"{conv_ch} * {hidden_size} + {hidden_loop}",
                default="1.0",
            )
            body.append(
                f"{indent * 10}{hidden_grad_ref} = {hidden_grad_ref} + "
                f"({scalar1}[0] * {conv_weight_expr} * {in_proj_weight_expr})"
            )
    if conv_weight_grad is not None:
        current_conv_weight_grad_ref = _indexed_buffer_ref(
            conv_weight_grad,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {history_len}",
        )
        body.append(
            f"{indent * 7}{current_conv_weight_grad_ref} = {current_conv_weight_grad_ref} + "
            f"({scalar1}[0] * {projected}[{conv_ch}])"
        )
    current_conv_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{conv_ch} * {kernel} + {history_len}",
        default="1.0",
    )
    body.append(
        f"{indent * 7}{project_grad}[{conv_ch}] = "
        f"{project_grad}[{conv_ch}] + ({scalar1}[0] * {current_conv_weight_expr})"
    )
    body.append(f"{indent * 6}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 7}for {hidden_loop} in T.serial(0, {hidden_size}):")
    hidden_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("m2rnn_hidden", "hidden"),
        1,
        dtype_by_buffer,
        access_by_buffer,
        f"{time_idx} * {hidden_size} + {hidden_loop}",
    )
    if in_proj_weight_grad is not None:
        in_proj_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"{proj_dim} * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 8}{in_proj_grad_ref} = {in_proj_grad_ref} + "
            f"({hidden_expr} * {project_grad}[{proj_dim}])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"{time_idx} * {hidden_size} + {hidden_loop}",
        )
        in_proj_weight_expr = _node_indexed_canonical_input_expr(
            node,
            "m2rnn_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"{proj_dim} * {hidden_size} + {hidden_loop}",
            default="1.0",
        )
        body.append(
            f"{indent * 8}{hidden_grad_ref} = {hidden_grad_ref} + "
            f"({project_grad}[{proj_dim}] * {in_proj_weight_expr})"
        )
    if h0_grad is not None:
        body.append(f"{indent * 5}for {state_idx} in T.serial(0, {full_state_extent}):")
        body.append(f"{indent * 6}{head} = {state_idx} // {k_dim * v_dim}")
        body.append(f"{indent * 6}{kk} = ({state_idx} // {v_dim}) % {k_dim}")
        body.append(f"{indent * 6}{vv} = {state_idx} % {v_dim}")
        h0_grad_ref = _indexed_buffer_ref(
            h0_grad,
            access_by_buffer,
            f"{head} * {k_dim * v_dim} + {kk} * {v_dim} + {vv}",
        )
        body.append(f"{indent * 6}{h0_grad_ref} = {h0_grad_ref} + {dh}[{state_idx}]")
    body.append(f"{indent * 3}T.sync_threads()")
    return


def _append_row_phased_mamba3_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    shape_env: PathCModelShapeEnv,
    thread_count: int,
    indent: str,
) -> None:
    stage_grad = _scratch_name(node, "mamba3_stage_grad")
    proj_dim = _scratch_name(node, "proj_dim")
    hidden_dim = _scratch_name(node, "hidden_dim")
    conv_ch = _scratch_name(node, "conv_ch")
    state_idx = _scratch_name(node, "state_idx")
    grad_flat = _scratch_name(node, "grad_flat")
    hidden_size = int(shape_env.hidden_size)
    sequence_length = int(shape_env.sequence_length)
    in_proj_dim = int(shape_env.mamba_in_proj_dim)
    inner_dim = int(shape_env.mamba_inner_dim)
    conv_channels = int(shape_env.mamba_conv_channels)
    kernel = int(shape_env.mamba_conv_kernel)
    state_extent = (
        int(shape_env.mamba_num_heads)
        * int(shape_env.mamba_head_dim)
        * int(shape_env.mamba_state_dim)
    )
    norm_extent = (
        int(shape_env.mamba_effective_mimo_rank)
        * int(shape_env.mamba_groups)
        * int(shape_env.mamba_state_dim)
    )
    hidden_grad = _node_output_for_canonical(node, "hidden")
    in_proj_weight_grad = _node_output_for_canonical(node, "mamba3_in_proj_weight")
    out_proj_weight_grad = _node_output_for_canonical(node, "mamba3_out_proj_weight")
    conv_weight_grad = _node_output_for_canonical(node, "mamba3_conv_weight")
    conv_bias_grad = _node_output_for_canonical(node, "mamba3_conv_bias")
    dt_bias_grad = _node_output_for_canonical(node, "mamba3_dt_bias")
    b_norm_weight_grad = _node_output_for_canonical(node, "mamba3_B_norm_weight")
    b_bias_grad = _node_output_for_canonical(node, "mamba3_B_bias")
    c_norm_weight_grad = _node_output_for_canonical(node, "mamba3_C_norm_weight")
    c_bias_grad = _node_output_for_canonical(node, "mamba3_C_bias")
    d_grad = _node_output_for_canonical(node, "mamba3_D")
    h0_grad = _node_output_for_canonical(node, "mamba3_h0")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    projected = _scratch_name(node, "mamba3_projected_vec")
    project_grad = _scratch_name(node, "mamba3_project_grad")
    conv_pre = _scratch_name(node, "mamba3_conv_pre")
    conv = _scratch_name(node, "mamba3_conv_vec")
    conv_grad = _scratch_name(node, "mamba3_conv_grad")
    out_inner = _scratch_name(node, "mamba3_out_inner")
    y_skip = _scratch_name(node, "mamba3_y_skip")
    out_inner_grad = _scratch_name(node, "mamba3_out_inner_grad")
    h_checkpoint = _scratch_name(node, "mamba3_h_checkpoint")
    angle_checkpoint = _scratch_name(node, "mamba3_angle_checkpoint")
    h_prev = _scratch_name(node, "mamba3_h_prev")
    h_next = _scratch_name(node, "mamba3_h_next")
    dh = _scratch_name(node, "mamba3_dh")
    dh_prev = _scratch_name(node, "mamba3_dh_prev")
    b_inv = _scratch_name(node, "mamba3_b_inv_rms")
    c_inv = _scratch_name(node, "mamba3_c_inv_rms")
    b_mean = _scratch_name(node, "mamba3_b_mean")
    c_mean = _scratch_name(node, "mamba3_c_mean")
    b_raw = _scratch_name(node, "mamba3_b_raw")
    c_raw = _scratch_name(node, "mamba3_c_raw")
    b_group = _scratch_name(node, "mamba3_b_group")
    c_group = _scratch_name(node, "mamba3_c_group")
    b_raw_grad = _scratch_name(node, "mamba3_b_raw_grad")
    c_raw_grad = _scratch_name(node, "mamba3_c_raw_grad")
    b_group_grad = _scratch_name(node, "mamba3_b_group_grad")
    c_group_grad = _scratch_name(node, "mamba3_c_group_grad")
    dt_vec = _scratch_name(node, "mamba3_dt_vec")
    a_vec = _scratch_name(node, "mamba3_a_vec")
    dt_grad = _scratch_name(node, "mamba3_dt_grad")
    a_grad = _scratch_name(node, "mamba3_a_grad")
    trap_group = _scratch_name(node, "mamba3_trap_group")
    trap_grad = _scratch_name(node, "mamba3_trap_grad")
    next_dt_pre_vec = _scratch_name(node, "mamba3_next_dt_pre_vec")
    next_dt_vec = _scratch_name(node, "mamba3_next_dt_vec")
    next_trap_vec = _scratch_name(node, "mamba3_next_trap_vec")
    angle_cumsum = _scratch_name(node, "mamba3_angle_cumsum")
    angle_grad = _scratch_name(node, "mamba3_angle_grad")
    sum_scratch = _scratch_name(node, "mamba3_sum")
    dot_scratch = _scratch_name(node, "mamba3_dot")
    scalar0 = _scratch_name(node, "mamba3_scalar0")
    scalar1 = _scratch_name(node, "mamba3_scalar1")
    scalar2 = _scratch_name(node, "mamba3_scalar2")
    scalar3 = _scratch_name(node, "mamba3_scalar3")
    scalar4 = _scratch_name(node, "mamba3_scalar4")
    time_rev = _scratch_name(node, "time_rev")
    time_idx = _scratch_name(node, "time_idx")
    replay_time = _scratch_name(node, "replay_time")
    replay_offset = _scratch_name(node, "replay_offset")
    checkpoint_idx = _scratch_name(node, "checkpoint_idx")
    checkpoint_start = _scratch_name(node, "checkpoint_start")
    src_row = _scratch_name(node, "src_row")
    src_hist = _scratch_name(node, "src_hist")
    hidden_loop = _scratch_name(node, "hidden_loop")
    kernel_pos = _scratch_name(node, "kernel_pos")
    rank_loop = _scratch_name(node, "rank_loop")
    group_loop = _scratch_name(node, "group_loop")
    angle_loop = _scratch_name(node, "angle_loop")
    state_loop = _scratch_name(node, "state_loop")
    pair_state = _scratch_name(node, "pair_state")
    head = _scratch_name(node, "head")
    feature = _scratch_name(node, "feature")
    out_dim = _scratch_name(node, "out_dim")
    heads = int(shape_env.mamba_num_heads)
    head_dim = int(shape_env.mamba_head_dim)
    state_dim = int(shape_env.mamba_state_dim)
    groups = int(shape_env.mamba_groups)
    mimo_rank = int(shape_env.mamba_effective_mimo_rank)
    rope_angles = int(shape_env.mamba_num_rope_angles)
    bc_dim = int(shape_env.mamba_bc_dim)
    heads_per_group = max(1, heads // max(1, groups))
    history_len = max(0, kernel - 1)
    z_offset = 0
    x_offset = inner_dim
    b_offset = 2 * inner_dim
    c_offset = b_offset + bc_dim
    dt_offset = c_offset + bc_dim
    a_offset = dt_offset + heads
    trap_offset = a_offset + heads
    angle_offset = trap_offset + heads
    conv_b_offset = inner_dim
    conv_c_offset = inner_dim + bc_dim
    rot_dim = min(state_dim, 2 * rope_angles)
    checkpoint_interval = MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL
    checkpoint_count = (sequence_length + checkpoint_interval - 1) // checkpoint_interval
    angle_extent = heads * rope_angles

    def _mamba3_hidden_expr(time_expr: str, hidden_expr: str) -> str:
        return _node_indexed_canonical_or_positional_input_expr(
            node,
            ("hidden",),
            1,
            dtype_by_buffer,
            access_by_buffer,
            f"{time_expr} * {hidden_size} + {hidden_expr}",
        )

    def _mamba3_project_weight_expr(proj_expr: str, hidden_expr: str) -> str:
        return _node_indexed_canonical_input_expr(
            node,
            "mamba3_in_proj_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({proj_expr}) * {hidden_size} + ({hidden_expr})",
            default="1.0",
        )

    def _mamba3_conv_weight_expr(channel_expr: str, kernel_expr: str) -> str:
        return _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_weight",
            dtype_by_buffer,
            access_by_buffer,
            f"({channel_expr}) * {kernel} + ({kernel_expr})",
            default="1.0",
        )

    def _mamba3_emit_recompute_row(
        time_expr: str,
        *,
        level: int,
        update_angle: bool,
        update_state: bool,
    ) -> None:
        body.append(f"{indent * level}for {proj_dim} in T.serial(0, {in_proj_dim}):")
        body.append(f"{indent * (level + 1)}{sum_scratch}[0] = 0.0")
        body.append(
            f"{indent * (level + 1)}for {hidden_loop} in T.serial(0, {hidden_size}):"
        )
        body.append(
            f"{indent * (level + 2)}{sum_scratch}[0] = {sum_scratch}[0] + "
            f"({_mamba3_hidden_expr(time_expr, hidden_loop)} * "
            f"{_mamba3_project_weight_expr(proj_dim, hidden_loop)})"
        )
        body.append(f"{indent * (level + 1)}{projected}[{proj_dim}] = {sum_scratch}[0]")
        body.append(f"{indent * level}for {conv_ch} in T.serial(0, {conv_channels}):")
        conv_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_bias",
            dtype_by_buffer,
            access_by_buffer,
            conv_ch,
            default="0.0",
        )
        body.append(f"{indent * (level + 1)}{conv_pre}[{conv_ch}] = {conv_bias_expr}")
        if history_len > 0:
            body.append(
                f"{indent * (level + 1)}for {kernel_pos} in T.serial(0, {history_len}):"
            )
            body.append(
                f"{indent * (level + 2)}{src_row} = {time_expr} - {history_len} + {kernel_pos}"
            )
            body.append(f"{indent * (level + 2)}if {src_row} >= 0:")
            body.append(f"{indent * (level + 3)}{scalar0}[0] = 0.0")
            body.append(
                f"{indent * (level + 3)}for {hidden_loop} in T.serial(0, {hidden_size}):"
            )
            body.append(
                f"{indent * (level + 4)}{scalar0}[0] = {scalar0}[0] + "
                f"({_mamba3_hidden_expr(src_row, hidden_loop)} * "
                f"{_mamba3_project_weight_expr(f'{x_offset} + {conv_ch}', hidden_loop)})"
            )
            body.append(f"{indent * (level + 2)}else:")
            body.append(f"{indent * (level + 3)}{src_hist} = {kernel_pos} + {time_expr}")
            conv_state_expr = _node_indexed_canonical_input_expr(
                node,
                "mamba3_conv_state",
                dtype_by_buffer,
                access_by_buffer,
                f"{src_hist} * {conv_channels} + {conv_ch}",
                default="0.0",
            )
            body.append(f"{indent * (level + 3)}{scalar0}[0] = {conv_state_expr}")
            body.append(
                f"{indent * (level + 2)}{conv_pre}[{conv_ch}] = "
                f"{conv_pre}[{conv_ch}] + "
                f"({scalar0}[0] * {_mamba3_conv_weight_expr(conv_ch, kernel_pos)})"
            )
        body.append(
            f"{indent * (level + 1)}{conv_pre}[{conv_ch}] = "
            f"{conv_pre}[{conv_ch}] + "
            f"({projected}[{x_offset} + {conv_ch}] * "
            f"{_mamba3_conv_weight_expr(conv_ch, str(history_len))})"
        )
        body.append(
            f"{indent * (level + 1)}{scalar0}[0] = 1.0 / "
            f"(1.0 + T.exp(-{conv_pre}[{conv_ch}]))"
        )
        body.append(
            f"{indent * (level + 1)}{conv}[{conv_ch}] = "
            f"{conv_pre}[{conv_ch}] * {scalar0}[0]"
        )
        body.append(f"{indent * level}for {head} in T.serial(0, {heads}):")
        dt_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_dt_bias",
            dtype_by_buffer,
            access_by_buffer,
            head,
            default="0.0",
        )
        body.append(
            f"{indent * (level + 1)}{dt_vec}[{head}] = "
            f"T.log(1.0 + T.exp({projected}[{dt_offset} + {head}] + {dt_bias_expr}))"
        )
        body.append(
            f"{indent * (level + 1)}{a_vec}[{head}] = "
            f"T.min(-T.log(1.0 + T.exp({projected}[{a_offset} + {head}])), -0.01)"
        )
        if update_angle:
            body.append(
                f"{indent * (level + 1)}for {angle_loop} in T.serial(0, {rope_angles}):"
            )
            body.append(
                f"{indent * (level + 2)}{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] = "
                f"{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] + "
                f"({projected}[{angle_offset} + {angle_loop}] * {dt_vec}[{head}])"
            )
        body.append(
            f"{indent * (level + 1)}{next_dt_pre_vec}[{head}] = 0.0"
        )
        body.append(f"{indent * (level + 1)}{next_dt_vec}[{head}] = 0.0")
        body.append(f"{indent * (level + 1)}{next_trap_vec}[{head}] = 0.0")
        body.append(f"{indent * (level + 1)}if {time_expr} + 1 < {sequence_length}:")
        body.append(
            f"{indent * (level + 2)}for {hidden_loop} in T.serial(0, {hidden_size}):"
        )
        body.append(
            f"{indent * (level + 3)}{next_dt_pre_vec}[{head}] = "
            f"{next_dt_pre_vec}[{head}] + "
            f"({_mamba3_hidden_expr(f'({time_expr} + 1)', hidden_loop)} * "
            f"{_mamba3_project_weight_expr(f'{dt_offset} + {head}', hidden_loop)})"
        )
        body.append(
            f"{indent * (level + 3)}{next_trap_vec}[{head}] = "
            f"{next_trap_vec}[{head}] + "
            f"({_mamba3_hidden_expr(f'({time_expr} + 1)', hidden_loop)} * "
            f"{_mamba3_project_weight_expr(f'{trap_offset} + {head}', hidden_loop)})"
        )
        body.append(
            f"{indent * (level + 2)}{next_dt_pre_vec}[{head}] = "
            f"{next_dt_pre_vec}[{head}] + {dt_bias_expr}"
        )
        body.append(
            f"{indent * (level + 2)}{next_dt_vec}[{head}] = "
            f"T.log(1.0 + T.exp({next_dt_pre_vec}[{head}]))"
        )
        body.append(f"{indent * level}for {group_loop} in T.serial(0, {groups}):")
        body.append(f"{indent * (level + 1)}{trap_group}[{group_loop}] = 0.0")
        body.append(
            f"{indent * (level + 1)}for {head} in T.serial(0, {heads_per_group}):"
        )
        body.append(
            f"{indent * (level + 2)}{hidden_dim} = {group_loop} * {heads_per_group} + {head}"
        )
        body.append(
            f"{indent * (level + 2)}{scalar0}[0] = "
            f"1.0 / (1.0 + T.exp(-{projected}[{trap_offset} + {hidden_dim}]))"
        )
        body.append(
            f"{indent * (level + 2)}{scalar1}[0] = "
            f"1.0 / (1.0 + T.exp(-{next_trap_vec}[{hidden_dim}]))"
        )
        body.append(
            f"{indent * (level + 2)}{trap_group}[{group_loop}] = "
            f"{trap_group}[{group_loop}] + "
            f"({next_dt_vec}[{hidden_dim}] * (1.0 - {scalar1}[0]) + "
            f"{dt_vec}[{hidden_dim}] * {scalar0}[0])"
        )
        body.append(
            f"{indent * (level + 1)}{trap_group}[{group_loop}] = "
            f"{trap_group}[{group_loop}] / {float(heads_per_group):.1f}"
        )
        body.append(f"{indent * level}for {rank_loop} in T.serial(0, {mimo_rank}):")
        body.append(f"{indent * (level + 1)}for {group_loop} in T.serial(0, {groups}):")
        body.append(f"{indent * (level + 2)}{sum_scratch}[0] = 0.0")
        body.append(f"{indent * (level + 2)}{dot_scratch}[0] = 0.0")
        body.append(
            f"{indent * (level + 2)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(
            f"{indent * (level + 3)}{grad_flat} = "
            f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
        )
        body.append(
            f"{indent * (level + 3)}{sum_scratch}[0] = {sum_scratch}[0] + "
            f"({conv}[{conv_b_offset} + {grad_flat}] * {conv}[{conv_b_offset} + {grad_flat}])"
        )
        body.append(
            f"{indent * (level + 3)}{dot_scratch}[0] = {dot_scratch}[0] + "
            f"({conv}[{conv_c_offset} + {grad_flat}] * {conv}[{conv_c_offset} + {grad_flat}])"
        )
        body.append(
            f"{indent * (level + 2)}{b_inv}[{rank_loop} * {groups} + {group_loop}] = "
            f"T.rsqrt(({sum_scratch}[0] / {float(state_dim):.1f}) + 0.00001)"
        )
        body.append(
            f"{indent * (level + 2)}{c_inv}[{rank_loop} * {groups} + {group_loop}] = "
            f"T.rsqrt(({dot_scratch}[0] / {float(state_dim):.1f}) + 0.00001)"
        )
        body.append(f"{indent * level}for {group_loop} in T.serial(0, {groups}):")
        body.append(
            f"{indent * (level + 1)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(
            f"{indent * (level + 2)}{b_mean}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
        )
        body.append(
            f"{indent * (level + 2)}{c_mean}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
        )
        body.append(f"{indent * (level + 2)}for {rank_loop} in T.serial(0, {mimo_rank}):")
        body.append(
            f"{indent * (level + 3)}{grad_flat} = "
            f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
        )
        b_norm_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_B_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="1.0",
        )
        b_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_B_bias",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="0.0",
        )
        c_norm_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_C_norm_weight",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="1.0",
        )
        c_bias_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_C_bias",
            dtype_by_buffer,
            access_by_buffer,
            grad_flat,
            default="0.0",
        )
        body.append(
            f"{indent * (level + 3)}{b_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_mean}[{group_loop} * {state_dim} + {state_loop}] + "
            f"({conv}[{conv_b_offset} + {grad_flat}] * "
            f"{b_inv}[{rank_loop} * {groups} + {group_loop}] * {b_norm_expr} + {b_bias_expr})"
        )
        body.append(
            f"{indent * (level + 3)}{c_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_mean}[{group_loop} * {state_dim} + {state_loop}] + "
            f"({conv}[{conv_c_offset} + {grad_flat}] * "
            f"{c_inv}[{rank_loop} * {groups} + {group_loop}] * {c_norm_expr} + {c_bias_expr})"
        )
        body.append(
            f"{indent * (level + 2)}{b_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_mean}[{group_loop} * {state_dim} + {state_loop}] / {float(mimo_rank):.1f}"
        )
        body.append(
            f"{indent * (level + 2)}{c_mean}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_mean}[{group_loop} * {state_dim} + {state_loop}] / {float(mimo_rank):.1f}"
        )
        body.append(
            f"{indent * (level + 2)}{b_raw}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_mean}[{group_loop} * {state_dim} + {state_loop}] * {trap_group}[{group_loop}]"
        )
        body.append(
            f"{indent * (level + 2)}{c_raw}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_mean}[{group_loop} * {state_dim} + {state_loop}]"
        )
        body.append(f"{indent * level}for {group_loop} in T.serial(0, {groups}):")
        body.append(
            f"{indent * (level + 1)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(f"{indent * (level + 2)}if {state_loop} < {rot_dim}:")
        body.append(f"{indent * (level + 3)}if ({state_loop} % 2) == 0:")
        body.append(
            f"{indent * (level + 4)}{b_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) - "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop} + 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(
            f"{indent * (level + 4)}{c_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) - "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop} + 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(f"{indent * (level + 3)}else:")
        body.append(
            f"{indent * (level + 4)}{b_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop} - 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) + "
            f"({b_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(
            f"{indent * (level + 4)}{c_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop} - 1] * "
            f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)])) + "
            f"({c_raw}[{group_loop} * {state_dim} + {state_loop}] * "
            f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + ({state_loop} // 2)]))"
        )
        body.append(f"{indent * (level + 2)}else:")
        body.append(
            f"{indent * (level + 3)}{b_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{b_raw}[{group_loop} * {state_dim} + {state_loop}]"
        )
        body.append(
            f"{indent * (level + 3)}{c_group}[{group_loop} * {state_dim} + {state_loop}] = "
            f"{c_raw}[{group_loop} * {state_dim} + {state_loop}]"
        )
        body.append(f"{indent * level}for {feature} in T.serial(0, {inner_dim}):")
        body.append(f"{indent * (level + 1)}{head} = {feature} // {head_dim}")
        body.append(f"{indent * (level + 1)}{hidden_dim} = {feature} % {head_dim}")
        body.append(f"{indent * (level + 1)}{group_loop} = {head} // {heads_per_group}")
        body.append(f"{indent * (level + 1)}{y_skip}[{feature}] = 0.0")
        body.append(
            f"{indent * (level + 1)}for {state_loop} in T.serial(0, {state_dim}):"
        )
        body.append(
            f"{indent * (level + 2)}{state_idx} = "
            f"{head} * {head_dim * state_dim} + {hidden_dim} * {state_dim} + {state_loop}"
        )
        if update_state:
            body.append(
                f"{indent * (level + 2)}{h_next}[{state_idx}] = "
                f"(T.exp({a_vec}[{head}] * {dt_vec}[{head}]) * {h_next}[{state_idx}]) + "
                f"({conv}[{feature}] * {b_group}[{group_loop} * {state_dim} + {state_loop}])"
            )
        body.append(
            f"{indent * (level + 2)}{y_skip}[{feature}] = {y_skip}[{feature}] + "
            f"({h_next}[{state_idx}] * {c_group}[{group_loop} * {state_dim} + {state_loop}])"
        )
        d_skip_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_D",
            dtype_by_buffer,
            access_by_buffer,
            head,
            default="1.0",
        )
        body.append(
            f"{indent * (level + 1)}{y_skip}[{feature}] = "
            f"{y_skip}[{feature}] + ({d_skip_expr} * {conv}[{feature}])"
        )
        body.append(
            f"{indent * (level + 1)}{scalar0}[0] = "
            f"1.0 / (1.0 + T.exp(-{projected}[{z_offset} + {feature}]))"
        )
        body.append(
            f"{indent * (level + 1)}{out_inner}[{feature}] = "
            f"{y_skip}[{feature}] * {projected}[{z_offset} + {feature}] * {scalar0}[0]"
        )

    body.append(f"{indent * 3}# mamba3_mimo_bwd_policy: exact_reverse_checkpoint_replay")
    for name, extent in (
        (projected, in_proj_dim),
        (project_grad, in_proj_dim),
        (conv_pre, conv_channels),
        (conv, conv_channels),
        (conv_grad, conv_channels),
        (out_inner, inner_dim),
        (y_skip, inner_dim),
        (out_inner_grad, inner_dim),
        (h_checkpoint, (checkpoint_count + 1) * state_extent),
        (angle_checkpoint, (checkpoint_count + 1) * angle_extent),
        (h_prev, state_extent),
        (h_next, state_extent),
        (dh, state_extent),
        (dh_prev, state_extent),
        (b_inv, mimo_rank * groups),
        (c_inv, mimo_rank * groups),
        (b_mean, groups * state_dim),
        (c_mean, groups * state_dim),
        (b_raw, groups * state_dim),
        (c_raw, groups * state_dim),
        (b_group, groups * state_dim),
        (c_group, groups * state_dim),
        (b_raw_grad, groups * state_dim),
        (c_raw_grad, groups * state_dim),
        (b_group_grad, groups * state_dim),
        (c_group_grad, groups * state_dim),
        (dt_vec, heads),
        (a_vec, heads),
        (dt_grad, heads),
        (a_grad, heads),
        (trap_group, groups),
        (trap_grad, groups),
        (next_dt_pre_vec, heads),
        (next_dt_vec, heads),
        (next_trap_vec, heads),
        (angle_cumsum, heads * rope_angles),
        (angle_grad, heads * rope_angles),
    ):
        alloc = (
            "T.alloc_shared"
            if name in {h_checkpoint, angle_checkpoint}
            else "T.alloc_local"
        )
        body.append(f"{indent * 3}{name} = {alloc}(({extent},), \"float32\")")
    for scalar in (
        stage_grad,
        sum_scratch,
        dot_scratch,
        scalar0,
        scalar1,
        scalar2,
        scalar3,
        scalar4,
    ):
        body.append(f"{indent * 3}{scalar} = T.alloc_local((1,), \"float32\")")
    body.append(f"{indent * 3}if row == 0:")
    body.append(f"{indent * 4}if lane == 0:")
    for output_name, extent in (
        (in_proj_weight_grad, in_proj_dim * hidden_size),
        (out_proj_weight_grad, hidden_size * inner_dim),
        (conv_weight_grad, conv_channels * kernel),
        (conv_bias_grad, conv_channels),
        (dt_bias_grad, heads),
        (b_norm_weight_grad, norm_extent),
        (b_bias_grad, norm_extent),
        (c_norm_weight_grad, norm_extent),
        (c_bias_grad, norm_extent),
        (d_grad, heads),
        (h0_grad, state_extent),
    ):
        if output_name is None:
            continue
        body.append(f"{indent * 5}for {grad_flat} in T.serial(0, {extent}):")
        body.append(
            f"{indent * 6}{_indexed_buffer_ref(output_name, access_by_buffer, grad_flat)} = 0.0"
        )
    if hidden_grad is not None and not _is_full_sequence_bank_slot(
        hidden_grad,
        access_by_buffer,
    ):
        body.append(
            f"{indent * 5}for {grad_flat} in T.serial(0, {sequence_length * hidden_size}):"
        )
        body.append(
            f"{indent * 6}{_indexed_buffer_ref(hidden_grad, access_by_buffer, grad_flat)} = 0.0"
        )
    body.append(f"{indent * 5}for {state_idx} in T.serial(0, {state_extent}):")
    body.append(f"{indent * 6}{head} = {state_idx} // {head_dim * state_dim}")
    body.append(f"{indent * 6}{hidden_dim} = ({state_idx} // {state_dim}) % {head_dim}")
    body.append(f"{indent * 6}{state_loop} = {state_idx} % {state_dim}")
    h0_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_h0",
        dtype_by_buffer,
        access_by_buffer,
        f"{head} * {head_dim * state_dim} + {hidden_dim} * {state_dim} + {state_loop}",
        default="0.0",
    )
    body.append(f"{indent * 6}{h_next}[{state_idx}] = {h0_expr}")
    body.append(f"{indent * 6}{h_checkpoint}[{state_idx}] = {h0_expr}")
    body.append(f"{indent * 6}{dh}[{state_idx}] = 0.0")
    body.append(f"{indent * 5}for {grad_flat} in T.serial(0, {heads * rope_angles}):")
    body.append(f"{indent * 6}{angle_cumsum}[{grad_flat}] = 0.0")
    body.append(f"{indent * 6}{angle_checkpoint}[{grad_flat}] = 0.0")
    body.append(f"{indent * 6}{angle_grad}[{grad_flat}] = 0.0")
    body.append(f"{indent * 5}for {replay_time} in T.serial(0, {sequence_length}):")
    _mamba3_emit_recompute_row(
        replay_time,
        level=6,
        update_angle=True,
        update_state=True,
    )
    body.append(f"{indent * 6}if (({replay_time} + 1) % {checkpoint_interval}) == 0:")
    body.append(
        f"{indent * 7}{checkpoint_idx} = ({replay_time} + 1) // {checkpoint_interval}"
    )
    body.append(f"{indent * 7}for {state_idx} in T.serial(0, {state_extent}):")
    body.append(
        f"{indent * 8}{h_checkpoint}[{checkpoint_idx} * {state_extent} + {state_idx}] = "
        f"{h_next}[{state_idx}]"
    )
    body.append(f"{indent * 7}for {grad_flat} in T.serial(0, {angle_extent}):")
    body.append(
        f"{indent * 8}{angle_checkpoint}[{checkpoint_idx} * {angle_extent} + {grad_flat}] = "
        f"{angle_cumsum}[{grad_flat}]"
    )
    body.append(f"{indent * 5}for {time_rev} in T.serial(0, {sequence_length}):")
    body.append(f"{indent * 6}{time_idx} = {sequence_length - 1} - {time_rev}")
    body.append(f"{indent * 6}{checkpoint_idx} = {time_idx} // {checkpoint_interval}")
    body.append(
        f"{indent * 6}{checkpoint_start} = {checkpoint_idx} * {checkpoint_interval}"
    )
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {state_extent}):")
    body.append(
        f"{indent * 7}{h_next}[{state_idx}] = "
        f"{h_checkpoint}[{checkpoint_idx} * {state_extent} + {state_idx}]"
    )
    body.append(f"{indent * 6}for {grad_flat} in T.serial(0, {angle_extent}):")
    body.append(
        f"{indent * 7}{angle_cumsum}[{grad_flat}] = "
        f"{angle_checkpoint}[{checkpoint_idx} * {angle_extent} + {grad_flat}]"
    )
    body.append(f"{indent * 6}for {replay_offset} in T.serial(0, {checkpoint_interval}):")
    body.append(
        f"{indent * 7}{replay_time} = {checkpoint_start} + {replay_offset}"
    )
    body.append(f"{indent * 7}if {replay_time} < {time_idx}:")
    _mamba3_emit_recompute_row(
        replay_time,
        level=8,
        update_angle=True,
        update_state=True,
    )
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {state_extent}):")
    body.append(f"{indent * 7}{h_prev}[{state_idx}] = {h_next}[{state_idx}]")
    _mamba3_emit_recompute_row(
        time_idx,
        level=6,
        update_angle=True,
        update_state=True,
    )
    body.append(f"{indent * 6}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 7}{project_grad}[{proj_dim}] = 0.0")
    body.append(f"{indent * 6}for {conv_ch} in T.serial(0, {conv_channels}):")
    body.append(f"{indent * 7}{conv_grad}[{conv_ch}] = 0.0")
    body.append(f"{indent * 6}for {feature} in T.serial(0, {inner_dim}):")
    body.append(f"{indent * 7}{out_inner_grad}[{feature}] = 0.0")
    body.append(f"{indent * 7}for {out_dim} in T.serial(0, {hidden_size}):")
    delta_grad_expr = _node_indexed_canonical_or_positional_input_expr(
        node,
        ("mamba3_delta",),
        0,
        dtype_by_buffer,
        access_by_buffer,
        f"{time_idx} * {hidden_size} + {out_dim}",
    )
    out_weight_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        f"{out_dim} * {inner_dim} + {feature}",
        default="1.0",
    )
    body.append(
        f"{indent * 8}{stage_grad}[0] = {delta_grad_expr}"
    )
    body.append(
        f"{indent * 8}{out_inner_grad}[{feature}] = "
        f"{out_inner_grad}[{feature}] + ({stage_grad}[0] * {out_weight_expr})"
    )
    if out_proj_weight_grad is not None:
        out_grad_ref = _indexed_buffer_ref(
            out_proj_weight_grad,
            access_by_buffer,
            f"{out_dim} * {inner_dim} + {feature}",
        )
        body.append(
            f"{indent * 8}{out_grad_ref} = {out_grad_ref} + "
            f"({stage_grad}[0] * {out_inner}[{feature}])"
        )
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {state_extent}):")
    body.append(f"{indent * 7}{dh_prev}[{state_idx}] = 0.0")
    body.append(f"{indent * 6}for {head} in T.serial(0, {heads}):")
    body.append(f"{indent * 7}{dt_grad}[{head}] = 0.0")
    body.append(f"{indent * 7}{a_grad}[{head}] = 0.0")
    body.append(f"{indent * 6}for {group_loop} in T.serial(0, {groups}):")
    body.append(f"{indent * 7}{trap_grad}[{group_loop}] = 0.0")
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{b_group_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(
        f"{indent * 8}{c_group_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(f"{indent * 6}for {feature} in T.serial(0, {inner_dim}):")
    body.append(f"{indent * 7}{head} = {feature} // {head_dim}")
    body.append(f"{indent * 7}{hidden_dim} = {feature} % {head_dim}")
    body.append(f"{indent * 7}{group_loop} = {head} // {heads_per_group}")
    body.append(
        f"{indent * 7}{scalar0}[0] = "
        f"1.0 / (1.0 + T.exp(-{projected}[{z_offset} + {feature}]))"
    )
    body.append(
        f"{indent * 7}{scalar1}[0] = {out_inner_grad}[{feature}] * {y_skip}[{feature}]"
    )
    body.append(
        f"{indent * 7}{scalar2}[0] = "
        f"{out_inner_grad}[{feature}] * {projected}[{z_offset} + {feature}] * {scalar0}[0]"
    )
    body.append(
        f"{indent * 7}{project_grad}[{z_offset} + {feature}] = "
        f"{project_grad}[{z_offset} + {feature}] + "
        f"({scalar1}[0] * {scalar0}[0] * "
        f"(1.0 + ({projected}[{z_offset} + {feature}] * (1.0 - {scalar0}[0]))))"
    )
    d_skip_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_D",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="1.0",
    )
    if d_grad is not None:
        d_ref = _indexed_buffer_ref(d_grad, access_by_buffer, head)
        body.append(
            f"{indent * 7}{d_ref} = {d_ref} + ({scalar2}[0] * {conv}[{feature}])"
        )
    body.append(
        f"{indent * 7}{conv_grad}[{feature}] = {conv_grad}[{feature}] + "
        f"({scalar2}[0] * {d_skip_expr})"
    )
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{state_idx} = "
        f"{head} * {head_dim * state_dim} + {hidden_dim} * {state_dim} + {state_loop}"
    )
    body.append(
        f"{indent * 8}{scalar3}[0] = {dh}[{state_idx}] + "
        f"({scalar2}[0] * {c_group}[{group_loop} * {state_dim} + {state_loop}])"
    )
    body.append(
        f"{indent * 8}{c_group_grad}[{group_loop} * {state_dim} + {state_loop}] = "
        f"{c_group_grad}[{group_loop} * {state_dim} + {state_loop}] + "
        f"({scalar2}[0] * {h_next}[{state_idx}])"
    )
    body.append(
        f"{indent * 8}{conv_grad}[{feature}] = {conv_grad}[{feature}] + "
        f"({scalar3}[0] * {b_group}[{group_loop} * {state_dim} + {state_loop}])"
    )
    body.append(
        f"{indent * 8}{b_group_grad}[{group_loop} * {state_dim} + {state_loop}] = "
        f"{b_group_grad}[{group_loop} * {state_dim} + {state_loop}] + "
        f"({scalar3}[0] * {conv}[{feature}])"
    )
    body.append(
        f"{indent * 8}{a_grad}[{head}] = {a_grad}[{head}] + "
        f"({scalar3}[0] * {h_prev}[{state_idx}] * "
        f"T.exp({a_vec}[{head}] * {dt_vec}[{head}]) * {dt_vec}[{head}])"
    )
    body.append(
        f"{indent * 8}{dt_grad}[{head}] = {dt_grad}[{head}] + "
        f"({scalar3}[0] * {h_prev}[{state_idx}] * "
        f"T.exp({a_vec}[{head}] * {dt_vec}[{head}]) * {a_vec}[{head}])"
    )
    body.append(
        f"{indent * 8}{dh_prev}[{state_idx}] = "
        f"{scalar3}[0] * T.exp({a_vec}[{head}] * {dt_vec}[{head}])"
    )
    body.append(f"{indent * 6}for {group_loop} in T.serial(0, {groups}):")
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(
        f"{indent * 8}{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = 0.0"
    )
    body.append(f"{indent * 7}for {angle_loop} in T.serial(0, {rope_angles}):")
    body.append(f"{indent * 8}{pair_state} = {angle_loop} * 2")
    body.append(f"{indent * 8}if {pair_state} + 1 < {rot_dim}:")
    body.append(
        f"{indent * 9}{scalar0}[0] = "
        f"T.cos({angle_cumsum}[{group_loop} * {rope_angles} + {angle_loop}])"
    )
    body.append(
        f"{indent * 9}{scalar1}[0] = "
        f"T.sin({angle_cumsum}[{group_loop} * {rope_angles} + {angle_loop}])"
    )
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"{b_group_grad}[{group_loop} * {state_dim} + {pair_state}]"
    )
    body.append(
        f"{indent * 9}{scalar3}[0] = "
        f"{b_group_grad}[{group_loop} * {state_dim} + {pair_state} + 1]"
    )
    body.append(
        f"{indent * 9}{b_raw_grad}[{group_loop} * {state_dim} + {pair_state}] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {pair_state}] + "
        f"({scalar2}[0] * {scalar0}[0] + {scalar3}[0] * {scalar1}[0])"
    )
    body.append(
        f"{indent * 9}{b_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] + "
        f"((-{scalar2}[0] * {scalar1}[0]) + ({scalar3}[0] * {scalar0}[0]))"
    )
    body.append(
        f"{indent * 9}{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] = "
        f"{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] + "
        f"({scalar2}[0] * ((-{b_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar1}[0]) - "
        f"({b_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar0}[0])) + "
        f"{scalar3}[0] * ({b_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar0}[0] - "
        f"{b_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar1}[0]))"
    )
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"{c_group_grad}[{group_loop} * {state_dim} + {pair_state}]"
    )
    body.append(
        f"{indent * 9}{scalar3}[0] = "
        f"{c_group_grad}[{group_loop} * {state_dim} + {pair_state} + 1]"
    )
    body.append(
        f"{indent * 9}{c_raw_grad}[{group_loop} * {state_dim} + {pair_state}] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {pair_state}] + "
        f"({scalar2}[0] * {scalar0}[0] + {scalar3}[0] * {scalar1}[0])"
    )
    body.append(
        f"{indent * 9}{c_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {pair_state} + 1] + "
        f"((-{scalar2}[0] * {scalar1}[0]) + ({scalar3}[0] * {scalar0}[0]))"
    )
    body.append(
        f"{indent * 9}{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] = "
        f"{angle_grad}[{group_loop} * {rope_angles} + {angle_loop}] + "
        f"({scalar2}[0] * ((-{c_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar1}[0]) - "
        f"({c_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar0}[0])) + "
        f"{scalar3}[0] * ({c_raw}[{group_loop} * {state_dim} + {pair_state}] * {scalar0}[0] - "
        f"{c_raw}[{group_loop} * {state_dim} + {pair_state} + 1] * {scalar1}[0]))"
    )
    body.append(f"{indent * 7}for {state_loop} in T.serial({rot_dim}, {state_dim}):")
    body.append(
        f"{indent * 8}{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] + "
        f"{b_group_grad}[{group_loop} * {state_dim} + {state_loop}]"
    )
    body.append(
        f"{indent * 8}{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] + "
        f"{c_group_grad}[{group_loop} * {state_dim} + {state_loop}]"
    )
    body.append(f"{indent * 6}for {group_loop} in T.serial(0, {groups}):")
    body.append(f"{indent * 7}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 8}{trap_grad}[{group_loop}] = {trap_grad}[{group_loop}] + "
        f"({b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] * "
        f"{b_mean}[{group_loop} * {state_dim} + {state_loop}])"
    )
    body.append(f"{indent * 6}for {rank_loop} in T.serial(0, {mimo_rank}):")
    body.append(f"{indent * 7}for {group_loop} in T.serial(0, {groups}):")
    body.append(f"{indent * 8}{sum_scratch}[0] = 0.0")
    body.append(f"{indent * 8}{dot_scratch}[0] = 0.0")
    body.append(f"{indent * 8}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 9}{grad_flat} = "
        f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
    )
    b_norm_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_B_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        grad_flat,
        default="1.0",
    )
    c_norm_expr = _node_indexed_canonical_input_expr(
        node,
        "mamba3_C_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        grad_flat,
        default="1.0",
    )
    body.append(
        f"{indent * 9}{sum_scratch}[0] = {sum_scratch}[0] + "
        f"(({b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] * "
        f"{trap_group}[{group_loop}] / {float(mimo_rank):.1f}) * {b_norm_expr} * "
        f"{conv}[{conv_b_offset} + {grad_flat}])"
    )
    body.append(
        f"{indent * 9}{dot_scratch}[0] = {dot_scratch}[0] + "
        f"(({c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] / "
        f"{float(mimo_rank):.1f}) * {c_norm_expr} * {conv}[{conv_c_offset} + {grad_flat}])"
    )
    body.append(f"{indent * 8}for {state_loop} in T.serial(0, {state_dim}):")
    body.append(
        f"{indent * 9}{grad_flat} = "
        f"({rank_loop} * {groups} + {group_loop}) * {state_dim} + {state_loop}"
    )
    body.append(
        f"{indent * 9}{scalar0}[0] = "
        f"{b_raw_grad}[{group_loop} * {state_dim} + {state_loop}] * "
        f"{trap_group}[{group_loop}] / {float(mimo_rank):.1f}"
    )
    if b_norm_weight_grad is not None:
        b_norm_grad_ref = _indexed_buffer_ref(
            b_norm_weight_grad,
            access_by_buffer,
            grad_flat,
        )
        body.append(
            f"{indent * 9}{b_norm_grad_ref} = {b_norm_grad_ref} + "
            f"({scalar0}[0] * {conv}[{conv_b_offset} + {grad_flat}] * "
            f"{b_inv}[{rank_loop} * {groups} + {group_loop}])"
        )
    if b_bias_grad is not None:
        b_bias_grad_ref = _indexed_buffer_ref(b_bias_grad, access_by_buffer, grad_flat)
        body.append(f"{indent * 9}{b_bias_grad_ref} = {b_bias_grad_ref} + {scalar0}[0]")
    body.append(
        f"{indent * 9}{conv_grad}[{conv_b_offset} + {grad_flat}] = "
        f"{conv_grad}[{conv_b_offset} + {grad_flat}] + "
        f"({b_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"(({scalar0}[0] * {b_norm_expr}) - "
        f"({conv}[{conv_b_offset} + {grad_flat}] * {sum_scratch}[0] * "
        f"{b_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"{b_inv}[{rank_loop} * {groups} + {group_loop}] / {float(state_dim):.1f})))"
    )
    body.append(
        f"{indent * 9}{scalar1}[0] = "
        f"{c_raw_grad}[{group_loop} * {state_dim} + {state_loop}] / {float(mimo_rank):.1f}"
    )
    if c_norm_weight_grad is not None:
        c_norm_grad_ref = _indexed_buffer_ref(
            c_norm_weight_grad,
            access_by_buffer,
            grad_flat,
        )
        body.append(
            f"{indent * 9}{c_norm_grad_ref} = {c_norm_grad_ref} + "
            f"({scalar1}[0] * {conv}[{conv_c_offset} + {grad_flat}] * "
            f"{c_inv}[{rank_loop} * {groups} + {group_loop}])"
        )
    if c_bias_grad is not None:
        c_bias_grad_ref = _indexed_buffer_ref(c_bias_grad, access_by_buffer, grad_flat)
        body.append(f"{indent * 9}{c_bias_grad_ref} = {c_bias_grad_ref} + {scalar1}[0]")
    body.append(
        f"{indent * 9}{conv_grad}[{conv_c_offset} + {grad_flat}] = "
        f"{conv_grad}[{conv_c_offset} + {grad_flat}] + "
        f"({c_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"(({scalar1}[0] * {c_norm_expr}) - "
        f"({conv}[{conv_c_offset} + {grad_flat}] * {dot_scratch}[0] * "
        f"{c_inv}[{rank_loop} * {groups} + {group_loop}] * "
        f"{c_inv}[{rank_loop} * {groups} + {group_loop}] / {float(state_dim):.1f})))"
    )
    body.append(f"{indent * 6}for {group_loop} in T.serial(0, {groups}):")
    body.append(f"{indent * 7}for {head} in T.serial(0, {heads_per_group}):")
    body.append(f"{indent * 8}{hidden_dim} = {group_loop} * {heads_per_group} + {head}")
    body.append(f"{indent * 8}{scalar0}[0] = {trap_grad}[{group_loop}] / {float(heads_per_group):.1f}")
    body.append(
        f"{indent * 8}{scalar1}[0] = "
        f"1.0 / (1.0 + T.exp(-{projected}[{trap_offset} + {hidden_dim}]))"
    )
    body.append(
        f"{indent * 8}{dt_grad}[{hidden_dim}] = {dt_grad}[{hidden_dim}] + "
        f"({scalar0}[0] * {scalar1}[0])"
    )
    body.append(
        f"{indent * 8}{project_grad}[{trap_offset} + {hidden_dim}] = "
        f"{project_grad}[{trap_offset} + {hidden_dim}] + "
        f"({scalar0}[0] * {dt_vec}[{hidden_dim}] * {scalar1}[0] * (1.0 - {scalar1}[0]))"
    )
    body.append(f"{indent * 8}if {time_idx} + 1 < {sequence_length}:")
    body.append(
        f"{indent * 9}{scalar2}[0] = "
        f"1.0 / (1.0 + T.exp(-{next_trap_vec}[{hidden_dim}]))"
    )
    body.append(
        f"{indent * 9}{scalar3}[0] = "
        f"{scalar0}[0] * (1.0 - {scalar2}[0]) * "
        f"(1.0 / (1.0 + T.exp(-{next_dt_pre_vec}[{hidden_dim}])))"
    )
    body.append(
        f"{indent * 9}{scalar4}[0] = "
        f"{scalar0}[0] * {next_dt_vec}[{hidden_dim}] * "
        f"(-{scalar2}[0] * (1.0 - {scalar2}[0]))"
    )
    if dt_bias_grad is not None:
        dt_bias_grad_ref = _indexed_buffer_ref(dt_bias_grad, access_by_buffer, hidden_dim)
        body.append(f"{indent * 9}{dt_bias_grad_ref} = {dt_bias_grad_ref} + {scalar3}[0]")
    body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
    if in_proj_weight_grad is not None:
        next_dt_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"({dt_offset} + {hidden_dim}) * {hidden_size} + {hidden_loop}",
        )
        next_trap_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"({trap_offset} + {hidden_dim}) * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 10}{next_dt_grad_ref} = {next_dt_grad_ref} + "
            f"({_mamba3_hidden_expr(f'({time_idx} + 1)', hidden_loop)} * {scalar3}[0])"
        )
        body.append(
            f"{indent * 10}{next_trap_grad_ref} = {next_trap_grad_ref} + "
            f"({_mamba3_hidden_expr(f'({time_idx} + 1)', hidden_loop)} * {scalar4}[0])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"({time_idx} + 1) * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 10}{hidden_grad_ref} = {hidden_grad_ref} + "
            f"({scalar3}[0] * {_mamba3_project_weight_expr(f'{dt_offset} + {hidden_dim}', hidden_loop)} + "
            f"{scalar4}[0] * {_mamba3_project_weight_expr(f'{trap_offset} + {hidden_dim}', hidden_loop)})"
        )
    body.append(f"{indent * 6}for {head} in T.serial(0, {heads}):")
    dt_bias_expr_head = _node_indexed_canonical_input_expr(
        node,
        "mamba3_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        head,
        default="0.0",
    )
    body.append(f"{indent * 7}for {angle_loop} in T.serial(0, {rope_angles}):")
    body.append(
        f"{indent * 8}{project_grad}[{angle_offset} + {angle_loop}] = "
        f"{project_grad}[{angle_offset} + {angle_loop}] + "
        f"({angle_grad}[{head} * {rope_angles} + {angle_loop}] * {dt_vec}[{head}])"
    )
    body.append(
        f"{indent * 8}{dt_grad}[{head}] = {dt_grad}[{head}] + "
        f"({angle_grad}[{head} * {rope_angles} + {angle_loop}] * "
        f"{projected}[{angle_offset} + {angle_loop}])"
    )
    body.append(
        f"{indent * 8}{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] = "
        f"{angle_cumsum}[{head} * {rope_angles} + {angle_loop}] - "
        f"({projected}[{angle_offset} + {angle_loop}] * {dt_vec}[{head}])"
    )
    body.append(
        f"{indent * 7}{scalar0}[0] = "
        f"1.0 / (1.0 + T.exp(-({projected}[{dt_offset} + {head}] + {dt_bias_expr_head})))"
    )
    body.append(
        f"{indent * 7}{project_grad}[{dt_offset} + {head}] = "
        f"{project_grad}[{dt_offset} + {head}] + ({dt_grad}[{head}] * {scalar0}[0])"
    )
    if dt_bias_grad is not None:
        dt_bias_grad_ref = _indexed_buffer_ref(dt_bias_grad, access_by_buffer, head)
        body.append(
            f"{indent * 7}{dt_bias_grad_ref} = {dt_bias_grad_ref} + "
            f"({dt_grad}[{head}] * {scalar0}[0])"
        )
    body.append(
        f"{indent * 7}if -T.log(1.0 + T.exp({projected}[{a_offset} + {head}])) < -0.01:"
    )
    body.append(
        f"{indent * 8}{project_grad}[{a_offset} + {head}] = "
        f"{project_grad}[{a_offset} + {head}] + "
        f"({a_grad}[{head}] * (-(1.0 / (1.0 + T.exp(-{projected}[{a_offset} + {head}])))))"
    )
    body.append(f"{indent * 6}for {conv_ch} in T.serial(0, {conv_channels}):")
    body.append(
        f"{indent * 7}{scalar0}[0] = 1.0 / (1.0 + T.exp(-{conv_pre}[{conv_ch}]))"
    )
    body.append(
        f"{indent * 7}{scalar1}[0] = {conv_grad}[{conv_ch}] * {scalar0}[0] * "
        f"(1.0 + ({conv_pre}[{conv_ch}] * (1.0 - {scalar0}[0])))"
    )
    if conv_bias_grad is not None:
        conv_bias_ref = _indexed_buffer_ref(conv_bias_grad, access_by_buffer, conv_ch)
        body.append(f"{indent * 7}{conv_bias_ref} = {conv_bias_ref} + {scalar1}[0]")
    if history_len > 0:
        body.append(f"{indent * 7}for {kernel_pos} in T.serial(0, {history_len}):")
        body.append(
            f"{indent * 8}{src_row} = {time_idx} - {history_len} + {kernel_pos}"
        )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        body.append(f"{indent * 9}{scalar2}[0] = 0.0")
        body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
        body.append(
            f"{indent * 10}{scalar2}[0] = {scalar2}[0] + "
            f"({_mamba3_hidden_expr(src_row, hidden_loop)} * "
            f"{_mamba3_project_weight_expr(f'{x_offset} + {conv_ch}', hidden_loop)})"
        )
        body.append(f"{indent * 8}else:")
        body.append(f"{indent * 9}{src_hist} = {kernel_pos} + {time_idx}")
        conv_state_expr = _node_indexed_canonical_input_expr(
            node,
            "mamba3_conv_state",
            dtype_by_buffer,
            access_by_buffer,
            f"{src_hist} * {conv_channels} + {conv_ch}",
            default="0.0",
        )
        body.append(f"{indent * 9}{scalar2}[0] = {conv_state_expr}")
        if conv_weight_grad is not None:
            conv_weight_grad_ref = _indexed_buffer_ref(
                conv_weight_grad,
                access_by_buffer,
                f"{conv_ch} * {kernel} + {kernel_pos}",
            )
            body.append(
                f"{indent * 8}{conv_weight_grad_ref} = {conv_weight_grad_ref} + "
                f"({scalar1}[0] * {scalar2}[0])"
            )
        body.append(f"{indent * 8}if {src_row} >= 0:")
        body.append(f"{indent * 9}for {hidden_loop} in T.serial(0, {hidden_size}):")
        if in_proj_weight_grad is not None:
            in_proj_grad_ref = _indexed_buffer_ref(
                in_proj_weight_grad,
                access_by_buffer,
                f"({x_offset} + {conv_ch}) * {hidden_size} + {hidden_loop}",
            )
            body.append(
                f"{indent * 10}{in_proj_grad_ref} = {in_proj_grad_ref} + "
                f"({_mamba3_hidden_expr(src_row, hidden_loop)} * {scalar1}[0] * "
                f"{_mamba3_conv_weight_expr(conv_ch, kernel_pos)})"
            )
        if hidden_grad is not None:
            hidden_grad_ref = _indexed_buffer_ref(
                hidden_grad,
                access_by_buffer,
                f"{src_row} * {hidden_size} + {hidden_loop}",
            )
            body.append(
                f"{indent * 10}{hidden_grad_ref} = {hidden_grad_ref} + "
                f"({scalar1}[0] * {_mamba3_conv_weight_expr(conv_ch, kernel_pos)} * "
                f"{_mamba3_project_weight_expr(f'{x_offset} + {conv_ch}', hidden_loop)})"
            )
    if conv_weight_grad is not None:
        current_conv_weight_grad_ref = _indexed_buffer_ref(
            conv_weight_grad,
            access_by_buffer,
            f"{conv_ch} * {kernel} + {history_len}",
        )
        body.append(
            f"{indent * 7}{current_conv_weight_grad_ref} = "
            f"{current_conv_weight_grad_ref} + "
            f"({scalar1}[0] * {projected}[{x_offset} + {conv_ch}])"
        )
    body.append(
        f"{indent * 7}{project_grad}[{x_offset} + {conv_ch}] = "
        f"{project_grad}[{x_offset} + {conv_ch}] + "
        f"({scalar1}[0] * {_mamba3_conv_weight_expr(conv_ch, str(history_len))})"
    )
    body.append(f"{indent * 6}for {proj_dim} in T.serial(0, {in_proj_dim}):")
    body.append(f"{indent * 7}for {hidden_loop} in T.serial(0, {hidden_size}):")
    if in_proj_weight_grad is not None:
        in_proj_grad_ref = _indexed_buffer_ref(
            in_proj_weight_grad,
            access_by_buffer,
            f"{proj_dim} * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 8}{in_proj_grad_ref} = {in_proj_grad_ref} + "
            f"({_mamba3_hidden_expr(time_idx, hidden_loop)} * {project_grad}[{proj_dim}])"
        )
    if hidden_grad is not None:
        hidden_grad_ref = _indexed_buffer_ref(
            hidden_grad,
            access_by_buffer,
            f"{time_idx} * {hidden_size} + {hidden_loop}",
        )
        body.append(
            f"{indent * 8}{hidden_grad_ref} = {hidden_grad_ref} + "
            f"({project_grad}[{proj_dim}] * {_mamba3_project_weight_expr(proj_dim, hidden_loop)})"
        )
    body.append(f"{indent * 6}for {state_idx} in T.serial(0, {state_extent}):")
    body.append(f"{indent * 7}{dh}[{state_idx}] = {dh_prev}[{state_idx}]")
    body.append(f"{indent * 7}{h_next}[{state_idx}] = {h_prev}[{state_idx}]")
    if h0_grad is not None:
        body.append(f"{indent * 5}for {state_idx} in T.serial(0, {state_extent}):")
        h0_grad_ref = _indexed_buffer_ref(h0_grad, access_by_buffer, state_idx)
        body.append(f"{indent * 6}{h0_grad_ref} = {h0_grad_ref} + {dh}[{state_idx}]")
    body.append(f"{indent * 3}T.sync_threads()")
    return


def _descriptor_node_source(
    *,
    node: _ScheduleNodeView,
    node_index: int,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    if node.op_name in _EXACT_ROW_PHASED_BACKWARD_OPS:
        return _ScheduleNodeFragment(allocations=(), statements=())
    if descriptor.fragment_emitter is not None:
        fragment = descriptor.fragment_emitter(
            node=node,
            dtype_by_buffer=dtype_by_buffer,
            access_by_buffer=access_by_buffer,
        )
        if not isinstance(fragment, _ScheduleNodeFragment):
            raise TypeError(
                f"fragment emitter for {descriptor.op_name!r} must return "
                "_ScheduleNodeFragment"
            )
        return fragment
    if node.op_name.endswith("_bwd"):
        return _emit_owner_output_backward_source(
            node,
            dtype_by_buffer,
            access_by_buffer,
        )
    return _emit_generic_descriptor_source(
        node,
        node_index,
        dtype_by_buffer,
        access_by_buffer,
    )


def _validated_buffer_extent(buffer_extent: int) -> int:
    extent = int(buffer_extent)
    if extent <= 0:
        raise ValueError("descriptor schedule buffer_extent must be positive")
    return extent


def _validated_max_rows_per_launch(max_rows_per_launch: int | None) -> int | None:
    if max_rows_per_launch is None:
        return None
    rows = int(max_rows_per_launch)
    if rows <= 0:
        raise ValueError("max_rows_per_launch must be positive when provided")
    return rows


def _validated_row_dispatch_mode(row_dispatch_mode: str) -> str:
    mode = str(row_dispatch_mode)
    if mode not in {
        DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS,
        DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    }:
        raise ValueError(
            "row_dispatch_mode must be one of "
            f"{DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS!r}, "
            f"{DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS!r}"
        )
    return mode


def _row_phased_chunk_count(
    *,
    loop_policy: str,
    shape_env: PathCModelShapeEnv | None,
    max_rows_per_launch: int | None,
) -> int | None:
    if (
        loop_policy != DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        or shape_env is None
        or max_rows_per_launch is None
    ):
        return None
    sequence_length = int(shape_env.sequence_length)
    rows_per_launch = _validated_max_rows_per_launch(max_rows_per_launch)
    assert rows_per_launch is not None
    return max(1, (sequence_length + rows_per_launch - 1) // rows_per_launch)


def _descriptor_loop_extent(
    external_buffers: Sequence[str],
    buffer_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> int:
    if shape_env is None:
        return buffer_extent
    if any(_canonical_buffer_name(name) == "hidden" for name in external_buffers):
        return shape_env.sequence_length * shape_env.hidden_size
    return buffer_extent


def _loop_indexed_buffer_ref(
    buffer_name: str,
    shape: Sequence[int],
    loop_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> str:
    name = _safe_identifier(buffer_name)
    flat_extent = _flattened_extent(shape)
    if flat_extent <= 1:
        return f"{name}[0]"
    if flat_extent >= loop_extent:
        return f"{name}[i]"
    if shape_env is None:
        return f"{name}[0]"
    canonical_name = _canonical_buffer_name(buffer_name)
    if canonical_name == "q_scale":
        return f"{name}[(i % {shape_env.hidden_size}) // {shape_env.attention_head_dim}]"
    if canonical_name == "kv_scale":
        if flat_extent == shape_env.sequence_length * shape_env.attention_num_kv_heads:
            return (
                f"{name}[(i // {shape_env.hidden_size}) * "
                f"{shape_env.attention_num_kv_heads} + "
                f"(((i % {shape_env.hidden_size}) // "
                f"{shape_env.attention_head_dim}) % "
                f"{shape_env.attention_num_kv_heads})]"
            )
        return (
            f"{name}[((i % {shape_env.hidden_size}) // "
            f"{shape_env.attention_head_dim}) % {shape_env.attention_num_kv_heads}]"
        )
    if canonical_name == "kv_fp8":
        kv_width = shape_env.attention_num_kv_heads * shape_env.attention_head_dim
        if flat_extent == shape_env.sequence_length * kv_width:
            return (
                f"{name}[(i // {shape_env.hidden_size}) * "
                f"{kv_width} + (i % {kv_width})]"
            )
        return (
            f"{name}[i % "
            f"{kv_width}]"
        )
    if canonical_name == "indices":
        row_width = shape_env.attention_num_kv_heads * shape_env.attention_sparse_topk
        if flat_extent == shape_env.sequence_length * row_width:
            return (
                f"{name}[(i // {shape_env.hidden_size}) * {row_width} + "
                f"(i % {row_width})]"
            )
        return (
            f"{name}[i % "
            f"{row_width}]"
        )
    if canonical_name == "lse":
        return (
            f"{name}[(i // {shape_env.hidden_size}) * "
            f"{shape_env.attention_num_q_heads} + "
            f"((i % {shape_env.hidden_size}) // {shape_env.attention_head_dim})]"
        )
    if canonical_name in {"target_ids", "target_mask"}:
        return f"{name}[i // {shape_env.hidden_size}]"
    if flat_extent == shape_env.hidden_size or canonical_name in {
        "attention_q_proj_bias",
        "attention_out_proj_bias",
        "mamba3_residual_to_m2rnn_norm_weight",
        "m2rnn_residual_to_attention_norm_weight",
        "residual_norm_weight",
        "final_norm_weight",
        # Block A: per-brick entry RMSNorm weight is also a length-H vector
        # whose values index over the hidden dimension on every row.
        "entry_rmsnorm_weight",
    }:
        return f"{name}[i % {shape_env.hidden_size}]"
    if canonical_name == "sparse_mla_sinks":
        return f"{name}[i % {shape_env.attention_num_q_heads}]"
    return f"{name}[i % {flat_extent}]"


def _emit_mamba3_mimo_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    proj = _scratch_name(node, "mamba3_proj")
    conv = _scratch_name(node, "mamba3_conv")
    dt = _scratch_name(node, "mamba3_dt")
    state = _scratch_name(node, "mamba3_state")
    out = _scratch_name(node, "mamba3_out")
    delta = _output_with_suffix(node, "_delta") or node.outputs[0]
    index = "i"
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    in_proj_weight = _optional_buffer_expr(
        "mamba3_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    conv_weight = _optional_buffer_expr(
        "mamba3_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    conv_bias = _optional_buffer_expr(
        "mamba3_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    dt_bias = _optional_buffer_expr(
        "mamba3_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    d_skip = _optional_buffer_expr(
        "mamba3_D",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    h0 = _optional_buffer_expr(
        "mamba3_h0",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    b_norm_weight = _optional_buffer_expr(
        "mamba3_B_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    b_bias = _optional_buffer_expr(
        "mamba3_B_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    c_norm_weight = _optional_buffer_expr(
        "mamba3_C_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    c_bias = _optional_buffer_expr(
        "mamba3_C_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    out_proj_weight = _optional_buffer_expr(
        "mamba3_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    inner = [
        f"{proj}[0] = {hidden} * {in_proj_weight}",
        f"{conv}[0] = ({proj}[0] * {conv_weight}) + {conv_bias}",
        f"{dt}[0] = T.log(1.0 + T.exp({proj}[0] + {dt_bias}))",
        f"{state}[0] = ({h0} * T.exp(-{dt}[0])) + "
        f"(({conv}[0] * {d_skip}) * {b_norm_weight}) + {b_bias}",
        f"{out}[0] = (({state}[0] * {c_norm_weight}) + {c_bias}) * "
        f"{out_proj_weight}",
        f"{_buffer_ref(delta, access_by_buffer, index)} = {out}[0]",
    ]
    for output_name in node.outputs:
        if output_name == delta:
            continue
        source = state if _canonical_buffer_name(output_name) == "scan_state" else out
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{source}[0]"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{proj} = T.alloc_local((1,), \"float32\")",
            f"{conv} = T.alloc_local((1,), \"float32\")",
            f"{dt} = T.alloc_local((1,), \"float32\")",
            f"{state} = T.alloc_local((1,), \"float32\")",
            f"{out} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_residual_rmsnorm_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    residual = _scratch_name(node, "residual")
    inv_rms = _scratch_name(node, "inv_rms")
    index = "i"
    lhs = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    rhs = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, index)
    weight = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, index)
    inner = [
        f"{residual}[0] = {lhs} + {rhs}",
        f"{inv_rms}[0] = T.rsqrt(({residual}[0] * {residual}[0]) + 0.00001)",
    ]
    if node.outputs:
        inner.append(
            f"{_buffer_ref(node.outputs[0], access_by_buffer, index)} = "
            f"{residual}[0]"
        )
    if len(node.outputs) > 1:
        inner.append(
            f"{_buffer_ref(node.outputs[1], access_by_buffer, index)} = "
            f"{residual}[0] * {inv_rms}[0] * {weight}"
        )
    for output_name in node.outputs[2:]:
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{residual}[0] * {inv_rms}[0] * {weight}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{residual} = T.alloc_local((1,), \"float32\")",
            f"{inv_rms} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_entry_rmsnorm_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    """Flat fallback fragment for entry RMSNorm (descriptor only).

    Block A: the row-phased dispatch in
    :func:`_append_row_phased_hidden_body` handles the production path.
    This fragment is the scalar descriptor source emitted when the
    schedule is forced onto the flat per-cell loop (used only by the
    descriptor-status diagnostics and tests).
    """

    inv_rms = _scratch_name(node, "inv_rms")
    index = "i"
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    weight = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, index)
    inner = [
        f"{inv_rms}[0] = T.rsqrt(({hidden} * {hidden}) + 0.00001)",
    ]
    for output_name in node.outputs:
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{hidden} * {inv_rms}[0] * {weight}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{inv_rms} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_m2rnn_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    projected = _scratch_name(node, "m2rnn_projected")
    conv = _scratch_name(node, "m2rnn_conv")
    xf = _scratch_name(node, "m2rnn_xf")
    recurrent = _scratch_name(node, "m2rnn_recurrent")
    post = _scratch_name(node, "m2rnn_post")
    index = "i"
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    in_proj_weight = _optional_buffer_expr(
        "m2rnn_in_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    conv_weight = _optional_buffer_expr(
        "m2rnn_conv_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    conv_bias = _optional_buffer_expr(
        "m2rnn_conv_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    conv_state = _optional_buffer_expr(
        "m2rnn_conv_state",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    a_log = _optional_buffer_expr(
        "m2rnn_A_log",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    dt_bias = _optional_buffer_expr(
        "m2rnn_dt_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    state_weight = _optional_buffer_expr(
        "m2rnn_state_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    h0 = _optional_buffer_expr(
        "m2rnn_h0",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    d_skip = _optional_buffer_expr(
        "m2rnn_D",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    gate_norm_weight = _optional_buffer_expr(
        "m2rnn_g_norm_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    out_proj_weight = _optional_buffer_expr(
        "m2rnn_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    inner = [
        f"{projected}[0] = {hidden} * {in_proj_weight}",
        f"{conv}[0] = ({projected}[0] * {conv_weight}) + "
        f"{conv_bias} + {conv_state}",
        f"{xf}[0] = T.log(1.0 + T.exp({projected}[0] + {a_log} + "
        f"{dt_bias}))",
        f"{recurrent}[0] = ({conv}[0] * {state_weight}) + "
        f"({h0} * T.exp(-{xf}[0]))",
        f"{post}[0] = ({recurrent}[0] + ({conv}[0] * {d_skip})) * "
        f"(1.0 / (1.0 + T.exp(-{projected}[0])))",
    ]
    for output_name in node.outputs:
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{post}[0] * {gate_norm_weight} * {out_proj_weight}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{projected} = T.alloc_local((1,), \"float32\")",
            f"{conv} = T.alloc_local((1,), \"float32\")",
            f"{xf} = T.alloc_local((1,), \"float32\")",
            f"{recurrent} = T.alloc_local((1,), \"float32\")",
            f"{post} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_attention_qkv_projection_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    q_projected = _scratch_name(node, "attention_q_projected")
    kv_projected = _scratch_name(node, "attention_kv_projected")
    q_projected_pair = _scratch_name(node, "attention_q_projected_pair")
    kv_projected_pair = _scratch_name(node, "attention_kv_projected_pair")
    rope_phase = _scratch_name(node, "attention_rope_phase")
    q_prepared = _scratch_name(node, "attention_q_prepared")
    kv_prepared = _scratch_name(node, "attention_kv_prepared")
    assigned: set[str] = set()
    index = "i"
    q_scale_output = _node_output_for_canonical(node, "q_scale")
    kv_scale_output = _node_output_for_canonical(node, "kv_scale")
    q_fp8_output = _node_output_for_canonical(node, "q_fp8")
    kv_fp8_output = _node_output_for_canonical(node, "kv_fp8")
    indices_output = _node_output_for_canonical(node, "indices")
    hidden = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    q_weight = _optional_buffer_expr(
        "attention_q_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    q_bias = _optional_buffer_expr(
        "attention_q_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    kv_weight = _optional_buffer_expr(
        "attention_sparse_kv_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    kv_bias = _optional_buffer_expr(
        "attention_sparse_kv_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    rope = _optional_buffer_expr(
        "attention_rope_inv_freq",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    inner = [
        f"{q_projected}[0] = {hidden} + {q_weight} + {q_bias}",
        f"{kv_projected}[0] = {hidden} + {kv_weight} + {kv_bias}",
        f"{rope_phase}[0] = {rope}",
        f"{q_prepared}[0] = {q_projected}[0] + {rope_phase}[0]",
        f"{kv_prepared}[0] = {kv_projected}[0] + {rope_phase}[0]",
    ]
    if q_scale_output is not None:
        inner.append(
            f"{_buffer_ref(q_scale_output, access_by_buffer, index)} = "
            f"T.max(T.abs(T.cast({q_prepared}[0], \"float32\")) * "
            f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
            f"T.cast(1.0e-12, \"float32\"))"
        )
        assigned.add(q_scale_output)
    if kv_scale_output is not None:
        inner.append(
            f"{_buffer_ref(kv_scale_output, access_by_buffer, index)} = "
            f"T.max(T.abs(T.cast({kv_prepared}[0], \"float32\")) * "
            f"T.cast({1.0 / 448.0:.17g}, \"float32\"), "
            f"T.cast(1.0e-12, \"float32\"))"
        )
        assigned.add(kv_scale_output)
    if q_fp8_output is not None:
        denominator = (
            _buffer_ref(q_scale_output, access_by_buffer, index)
            if q_scale_output is not None
            else "1.0"
        )
        inner.append(
            f"{_buffer_ref(q_fp8_output, access_by_buffer, index)} = "
            f"{_fp8_encode_expr(f'{q_prepared}[0]', denominator)}"
        )
        assigned.add(q_fp8_output)
    if kv_fp8_output is not None:
        denominator = (
            _buffer_ref(kv_scale_output, access_by_buffer, index)
            if kv_scale_output is not None
            else "1.0"
        )
        inner.append(
            f"{_buffer_ref(kv_fp8_output, access_by_buffer, index)} = "
            f"{_fp8_encode_expr(f'{kv_prepared}[0]', denominator)}"
        )
        assigned.add(kv_fp8_output)
    if indices_output is not None:
        inner.append(f"{_buffer_ref(indices_output, access_by_buffer, index)} = i")
        assigned.add(indices_output)
    for output_name in node.outputs:
        if output_name in assigned:
            continue
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{q_prepared}[0] + {kv_prepared}[0]"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{q_projected} = T.alloc_local((1,), \"float32\")",
            f"{kv_projected} = T.alloc_local((1,), \"float32\")",
            f"{q_projected_pair} = T.alloc_local((1,), \"float32\")",
            f"{kv_projected_pair} = T.alloc_local((1,), \"float32\")",
            f"{rope_phase} = T.alloc_local((1,), \"float32\")",
            f"{q_prepared} = T.alloc_local((1,), \"float32\")",
            f"{kv_prepared} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_sparse_mla_fp8_apply_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    sink_enabled = _scratch_name(node, "sink_enabled")
    sparse_index = _scratch_name(node, "sparse_index")
    score_accum = _scratch_name(node, "score_accum")
    score_max = _scratch_name(node, "score_max")
    score_weight = _scratch_name(node, "score_weight")
    sumexp = _scratch_name(node, "sumexp")
    value_accum = _scratch_name(node, "value_accum")
    context_accum = _scratch_name(node, "context_accum")
    q_head_index = _scratch_name(node, "q_head")
    kv_head_index = _scratch_name(node, "kv_head")
    source_head_index = _scratch_name(node, "source_head")
    source_dim_index = _scratch_name(node, "source_dim")
    index = "i"
    q_fp8 = _optional_buffer_expr("q_fp8", dtype_by_buffer, access_by_buffer, index=index)
    q_scale = _optional_buffer_expr(
        "q_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    indices_ref = _optional_buffer_raw_ref(
        "indices",
        dtype_by_buffer,
        access_by_buffer,
        default="0",
        index=index,
    )
    kv_fp8 = _optional_indexed_buffer_expr(
        "kv_fp8",
        dtype_by_buffer,
        access_by_buffer,
        index_expr=f"{sparse_index}[0]",
    )
    kv_scale = _optional_buffer_expr(
        "kv_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    selected_kv_scale = _optional_indexed_buffer_expr(
        "kv_scale",
        dtype_by_buffer,
        access_by_buffer,
        default=kv_scale,
        index_expr=f"{sparse_index}[0]",
    )
    sm_scale = _optional_buffer_expr(
        "sparse_mla_sm_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    sinks = _optional_buffer_expr(
        "sparse_mla_sinks",
        dtype_by_buffer,
        access_by_buffer,
        index=index,
    )
    has_sinks = _optional_buffer_expr(
        "sparse_mla_has_sinks",
        dtype_by_buffer,
        access_by_buffer,
        "0",
        index,
    )
    out_proj_weight = _optional_buffer_expr(
        "attention_out_proj_weight",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    out_proj_bias = _optional_buffer_expr(
        "attention_out_proj_bias",
        dtype_by_buffer,
        access_by_buffer,
        "0.0",
        index,
    )
    inner = [
        f"{sink_enabled}[0] = T.cast({has_sinks} != 0, \"float32\")",
        f"{sparse_index}[0] = T.cast({indices_ref}, \"int32\")",
    ]
    assigned: set[str] = set()
    attention_out = (
        _node_output_for_canonical(node, "attention_out")
        or _node_output_with_suffix(node, "_out")
    )
    lse_output = _node_output_for_canonical(node, "lse")
    if attention_out is not None:
        inner.append(
            f"{_buffer_ref(attention_out, access_by_buffer, index)} = "
            f"(((({q_fp8} * {q_scale}) + "
            f"({kv_fp8} * {selected_kv_scale})) * {sm_scale} + "
            f"({sinks} * {sink_enabled}[0])) * "
            f"{out_proj_weight}) + {out_proj_bias}"
        )
        assigned.add(attention_out)
    if lse_output is not None:
        inner.append(f"{_buffer_ref(lse_output, access_by_buffer, index)} = 0.0")
        assigned.add(lse_output)
    for output_name in node.outputs:
        if output_name in assigned:
            continue
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{q_fp8} + {kv_fp8}"
        )
    return _ScheduleNodeFragment(
        allocations=(
            f"{sink_enabled} = T.alloc_local((1,), \"float32\")",
            f"{sparse_index} = T.alloc_local((1,), \"int32\")",
            f"{score_accum} = T.alloc_local((1,), \"float32\")",
            f"{score_max} = T.alloc_local((1,), \"float32\")",
            f"{score_weight} = T.alloc_local((1,), \"float32\")",
            f"{sumexp} = T.alloc_local((1,), \"float32\")",
            f"{value_accum} = T.alloc_local((1,), \"float32\")",
            f"{context_accum} = T.alloc_local((1,), \"float32\")",
            f"{q_head_index} = T.alloc_local((1,), \"int32\")",
            f"{kv_head_index} = T.alloc_local((1,), \"int32\")",
            f"{source_head_index} = T.alloc_local((1,), \"int32\")",
            f"{source_dim_index} = T.alloc_local((1,), \"int32\")",
        ),
        statements=tuple(inner),
    )


def _emit_owner_output_backward_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    accum = _scratch_name(node, "grad_accum")
    index = "i"
    inner = [
        f"{accum}[0] = {_sum_buffer_expr(node.inputs, dtype_by_buffer, access_by_buffer, index)}",
    ]
    for output_index, output_name in enumerate(node.outputs):
        inner.append(
            f"{_buffer_ref(output_name, access_by_buffer, index)} = "
            f"{accum}[0] + "
            f"{float(output_index):.1f}"
        )
    return _ScheduleNodeFragment(
        allocations=(f"{accum} = T.alloc_local((1,), \"float32\")",),
        statements=tuple(inner),
    )


def _emit_generic_descriptor_source(
    node: _ScheduleNodeView,
    node_index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    index = "i"
    inner: list[str] = []
    for output_index, output_name in enumerate(node.outputs):
        dtype = dtype_by_buffer[output_name]
        if dtype == "int32":
            inner.append(
                f"{_buffer_ref(output_name, access_by_buffer, index)} = "
                f"{index} + {output_index}"
            )
            continue
        expr = _sum_buffer_expr(node.inputs, dtype_by_buffer, access_by_buffer, index)
        if node_index or output_index:
            expr = f"({expr}) + {float(node_index + output_index):.1f}"
        inner.append(f"{_buffer_ref(output_name, access_by_buffer, index)} = {expr}")
    return _ScheduleNodeFragment(allocations=(), statements=tuple(inner))


def _mamba3_projection_inputs(node: _ScheduleNodeView) -> tuple[str, ...]:
    return _projection_inputs_from_node(
        node,
        leading_input_count=2,
        canonical_names=(
            "mamba3_in_proj_weight",
            "mamba3_out_proj_weight",
            "mamba3_conv_weight",
            "mamba3_conv_bias",
            "mamba3_dt_bias",
            "mamba3_h0",
        ),
    )


def _m2rnn_projection_inputs(node: _ScheduleNodeView) -> tuple[str, ...]:
    return _projection_inputs_from_node(
        node,
        leading_input_count=1,
        canonical_names=(
            "m2rnn_in_proj_weight",
            "m2rnn_conv_weight",
            "m2rnn_conv_bias",
            "m2rnn_state_weight",
            "m2rnn_A_log",
            "m2rnn_dt_bias",
            "m2rnn_h0",
            "m2rnn_conv_state",
        ),
    )


def _attention_projection_inputs(node: _ScheduleNodeView) -> tuple[str, ...]:
    return _projection_inputs_from_node(
        node,
        leading_input_count=1,
        canonical_names=(
            "attention_q_proj_weight",
            "attention_q_proj_bias",
            "attention_sparse_kv_proj_weight",
            "attention_sparse_kv_proj_bias",
            "attention_rope_inv_freq",
        ),
    )


def _projection_inputs_from_node(
    node: _ScheduleNodeView,
    *,
    leading_input_count: int,
    canonical_names: Sequence[str],
) -> tuple[str, ...]:
    selected: list[str] = []
    seen = set()
    for input_name in node.inputs[:leading_input_count]:
        if input_name in seen:
            continue
        seen.add(input_name)
        selected.append(input_name)
    wanted = set(canonical_names)
    for input_name in node.inputs[leading_input_count:]:
        canonical_name = _canonical_buffer_name(input_name)
        if canonical_name not in wanted or canonical_name in seen:
            continue
        seen.add(canonical_name)
        selected.append(input_name)
    return tuple(selected)


def _scratch_name(node: _ScheduleNodeView, suffix: str) -> str:
    return _safe_identifier(f"{node.name}_{suffix}")


def _output_with_suffix(node: _ScheduleNodeView, suffix: str) -> str | None:
    for output_name in node.outputs:
        if output_name.endswith(suffix):
            return output_name
    return None


def _node_output_for_canonical(
    node: _ScheduleNodeView,
    canonical_name: str,
) -> str | None:
    for output_name in node.outputs:
        if _canonical_buffer_name(output_name) == canonical_name:
            return output_name
    return None


def _node_output_for_canonical_or_index(
    node: _ScheduleNodeView,
    canonical_names: Sequence[str],
    positional_index: int,
) -> str | None:
    for canonical_name in canonical_names:
        output_name = _node_output_for_canonical(node, canonical_name)
        if output_name is not None:
            return output_name
    if positional_index >= len(node.outputs):
        return None
    return node.outputs[positional_index]


def _node_input_for_canonical(
    node: _ScheduleNodeView,
    canonical_name: str,
) -> str | None:
    for input_name in node.inputs:
        if _canonical_buffer_name(input_name) == canonical_name:
            return input_name
    return None


def _node_output_with_suffix(
    node: _ScheduleNodeView,
    suffix: str,
) -> str | None:
    for output_name in node.outputs:
        if output_name.endswith(suffix):
            return output_name
    return None


def _node_input_expr(
    node: _ScheduleNodeView,
    index: int,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    loop_index: str,
) -> str:
    if index >= len(node.inputs):
        return "0.0"
    input_name = node.inputs[index]
    return _buffer_value_expr(
        input_name,
        dtype_by_buffer[input_name],
        access_by_buffer,
        loop_index,
    )


def _sum_buffer_expr(
    buffer_names: Sequence[str],
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    index: str,
) -> str:
    terms = [
        _buffer_value_expr(
            buffer_name,
            dtype_by_buffer[buffer_name],
            access_by_buffer,
            index,
        )
        for buffer_name in buffer_names
        if buffer_name in dtype_by_buffer
    ]
    return " + ".join(terms) if terms else "0.0"


def _optional_buffer_expr(
    buffer_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    default: str = "0.0",
    index: str = "0",
) -> str:
    resolved_name = _buffer_name_for_canonical_or_exact(buffer_name, dtype_by_buffer)
    if resolved_name is None:
        return default
    return _buffer_value_expr(
        resolved_name,
        dtype_by_buffer[resolved_name],
        access_by_buffer,
        index,
    )


def _optional_buffer_raw_ref(
    buffer_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    *,
    default: str,
    index: str,
) -> str:
    resolved_name = _buffer_name_for_canonical_or_exact(buffer_name, dtype_by_buffer)
    if resolved_name is None:
        return default
    return _buffer_ref(resolved_name, access_by_buffer, index)


def _optional_indexed_buffer_expr(
    buffer_name: str,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    *,
    default: str = "0.0",
    index_expr: str,
) -> str:
    resolved_name = _buffer_name_for_canonical_or_exact(buffer_name, dtype_by_buffer)
    if resolved_name is None:
        return default
    return _indexed_buffer_value_expr(
        resolved_name,
        dtype_by_buffer[resolved_name],
        access_by_buffer,
        index_expr,
    )


def _buffer_name_for_canonical_or_exact(
    buffer_name: str,
    dtype_by_buffer: Mapping[str, str],
) -> str | None:
    if buffer_name in dtype_by_buffer:
        return buffer_name
    canonical_name = _canonical_buffer_name(buffer_name)
    for candidate_name in dtype_by_buffer:
        if _canonical_buffer_name(candidate_name) == canonical_name:
            return candidate_name
    return None


def _buffer_value_expr(
    buffer_name: str,
    dtype: str,
    access_by_buffer: dict[str, str],
    index: str,
) -> str:
    ref = _buffer_ref(buffer_name, access_by_buffer, index)
    if dtype in {"bfloat16", "float16", "int32"}:
        return f'T.cast({ref}, "float32")'
    if dtype == "uint8":
        return f"fp8_e4m3fn_to_float({ref})"
    return ref


def _fp8_encode_expr(value_expr: str, scale_expr: str) -> str:
    normalized = f'(T.cast({value_expr}, "float32") / {scale_expr})'
    clamped = (
        f'T.min(T.max({normalized}, T.cast(-448.0, "float32")), '
        f'T.cast(448.0, "float32"))'
    )
    return f"float_to_fp8_e4m3fn_bits({clamped})"


def _indexed_buffer_value_expr(
    buffer_name: str,
    dtype: str,
    access_by_buffer: dict[str, str],
    index_expr: str,
) -> str:
    ref = _indexed_buffer_ref(buffer_name, access_by_buffer, index_expr)
    if dtype in {"bfloat16", "float16", "int32"}:
        return f'T.cast({ref}, "float32")'
    if dtype == "uint8":
        return f"fp8_e4m3fn_to_float({ref})"
    return ref


def _buffer_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    index: str,
) -> str:
    ref = access_by_buffer.get(buffer_name)
    if ref is None:
        return f"{buffer_name}[{index}]"
    if index == "i":
        return ref
    return _indexed_buffer_ref(buffer_name, access_by_buffer, index)


def _indexed_buffer_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    index_expr: str,
) -> str:
    ref = access_by_buffer.get(buffer_name)
    if ref is None:
        return f"{_safe_identifier(buffer_name)}[{index_expr}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[0\]", ref)
    if match is not None:
        return ref
    match = re.fullmatch(r"([A-Za-z_]\w*)\[i\]", ref)
    if match is not None:
        return f"{match.group(1)}[{index_expr}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[i % (\d+)\]", ref)
    if match is not None:
        return f"{match.group(1)}[({index_expr}) % {match.group(2)}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[\(i % (\d+)\) // (\d+)\]", ref)
    if match is not None:
        return (
            f"{match.group(1)}[(({index_expr}) % {match.group(2)}) // "
            f"{match.group(3)}]"
        )
    match = re.fullmatch(
        r"([A-Za-z_]\w*)\[\(\(i % (\d+)\) // (\d+)\) % (\d+)\]",
        ref,
    )
    if match is not None:
        return (
            f"{match.group(1)}[((({index_expr}) % {match.group(2)}) // "
            f"{match.group(3)}) % {match.group(4)}]"
        )
    match = re.fullmatch(r"([A-Za-z_]\w*)\[i // (\d+)\]", ref)
    if match is not None:
        return f"{match.group(1)}[({index_expr}) // {match.group(2)}]"
    match = re.fullmatch(r"([A-Za-z_]\w*)\[(.+)\]", ref)
    if match is not None:
        if re.search(r"\bi\b", match.group(2)):
            replaced = re.sub(r"\bi\b", f"({index_expr})", match.group(2))
            return f"{match.group(1)}[{replaced}]"
        return ref
    return f"{_safe_identifier(buffer_name)}[{index_expr}]"


def _node_views_for_region(region: Any) -> tuple[_ScheduleNodeView, ...]:
    nodes = getattr(region, "nodes", None)
    if nodes is None:
        raise TypeError("descriptor schedule generation requires a region with nodes")
    views: list[_ScheduleNodeView] = []
    for node in nodes:
        op_name = getattr(node, "op_name", None)
        if op_name is None:
            op_name = getattr(node, "op", None)
        if op_name is None:
            raise TypeError("region node must expose op_name or op")
        views.append(
            _ScheduleNodeView(
                name=str(getattr(node, "name")),
                op_name=str(op_name),
                inputs=tuple(str(name) for name in getattr(node, "inputs", ())),
                outputs=tuple(str(name) for name in getattr(node, "outputs", ())),
                backward=str(getattr(node, "backward", "")),
            )
        )
    if not views:
        raise ValueError("descriptor schedule generation requires at least one node")
    return tuple(views)


def _internal_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
    *,
    shape_env: PathCModelShapeEnv | None = None,
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT,
) -> tuple[str, ...]:
    input_names = {name for node in nodes for name in node.inputs}
    internal: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for output_name in node.outputs:
            if output_name not in input_names or output_name in seen:
                continue
            if _is_attention_kv_history_workspace_output(nodes, node, output_name):
                continue
            if _uses_external_kv_history_workspace(
                output_name,
                shape_env=shape_env,
                internal_buffer_policy=internal_buffer_policy,
                loop_policy=loop_policy,
            ):
                continue
            seen.add(output_name)
            internal.append(output_name)
    return tuple(internal)


def _descriptor_chain_uses_kv_history_workspace(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> bool:
    op_names = {descriptor.op_name for descriptor in descriptors}
    return {
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    }.issubset(op_names)


def _is_attention_kv_history_workspace_output(
    nodes: Sequence[_ScheduleNodeView],
    producer: _ScheduleNodeView,
    output_name: str,
) -> bool:
    if producer.op_name != "attention_qkv_projection":
        return False
    if str(output_name).endswith("_grad"):
        return False
    if _canonical_buffer_name(output_name) not in {
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    }:
        return False
    return any(
        consumer.op_name in {"sparse_mla_fp8_apply", "sparse_mla_fp8_apply_bwd"}
        and output_name in consumer.inputs
        for consumer in nodes
    )


def _uses_external_kv_history_workspace(
    buffer_name: str,
    *,
    shape_env: PathCModelShapeEnv | None,
    internal_buffer_policy: str,
    loop_policy: str,
) -> bool:
    if str(buffer_name).endswith("_grad"):
        return False
    if shape_env is None:
        return False
    if internal_buffer_policy != DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN:
        return False
    if loop_policy != DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
        return False
    return _canonical_buffer_name(buffer_name) in {
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
    }


def _external_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
    internal_buffers: Sequence[str],
) -> tuple[str, ...]:
    internal_set = set(internal_buffers)
    external: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for buffer_name in (*node.inputs, *node.outputs):
            if buffer_name in internal_set or buffer_name in seen:
                continue
            seen.add(buffer_name)
            external.append(buffer_name)
    return tuple(external)


def _buffer_dtype(
    buffer_name: str,
    *,
    shape_env: PathCModelShapeEnv | None = None,
) -> str:
    if str(buffer_name).endswith("_grad"):
        return "float32"
    canonical = _canonical_buffer_name(buffer_name)
    if canonical in {"q_fp8", "kv_fp8"}:
        return "uint8"
    if canonical == "target_ids":
        return "int32"
    if buffer_name == "indices" or buffer_name.endswith("_indices"):
        return "int32"
    if buffer_name.endswith("_has_sinks"):
        return "int32"
    if canonical in {
        "mamba3_angle_state",
        "mamba3_conv_state",
        "m2rnn_h_state",
        "q_scale",
        "kv_scale",
        "lse",
    } or buffer_name.endswith(
        ("_scale", "_sm_scale", "_lse")
    ):
        return "float32"
    return (
        str(shape_env.model_value_dtype)
        if shape_env is not None and str(shape_env.model_value_dtype)
        else "float32"
    )


def _shape_env_for_region(region: Any) -> PathCModelShapeEnv | None:
    metadata = getattr(region, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    shape_env = metadata.get("path_c_model_shape_env")
    return shape_env if isinstance(shape_env, PathCModelShapeEnv) else None


_KNOWN_BUFFER_SUFFIXES = tuple(
    sorted(
        {
            *MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
            *_GENERIC_MODEL_REAL_ABI_INPUT_SUFFIXES,
            *_TRAIN_STEP_SUFFIX_LOSS_INPUT_ABI_NAMES,
            "hidden",
            "mamba_state",
            "scan_state",
            "mamba3_angle_state",
            "mamba3_conv_state",
            "mamba3_delta",
            "m2rnn_hidden",
            "m2rnn_delta",
            "m2rnn_h_state",
            "attention_hidden",
            "hidden_after_mamba3",
            "hidden_after_m2rnn",
            "attention_out",
            "lse",
            "q_fp8",
            "q_scale",
            "kv_fp8",
            "kv_scale",
            "indices",
        },
        key=len,
        reverse=True,
    )
)


def _canonical_buffer_name(buffer_name: str) -> str:
    name = str(buffer_name)
    if name.endswith("_grad"):
        name = name[: -len("_grad")]
    for suffix in _KNOWN_BUFFER_SUFFIXES:
        if name == suffix or name.endswith(f"_{suffix}"):
            return suffix
    return name


def _is_mamba3_state_like_buffer(buffer_name: str, canonical_name: str) -> bool:
    if canonical_name in {"mamba_state", "scan_state", "mamba3_h0"}:
        return True
    if canonical_name in {
        "mamba3_angle_state",
        "mamba3_conv_state",
        "m2rnn_conv_state",
        "m2rnn_h_state",
    }:
        return False
    name = str(buffer_name)
    if name.endswith("_grad"):
        name = name[: -len("_grad")]
    return (
        (name.endswith("_state") or name.endswith("_state_in") or name.endswith("_state_out"))
        and "m2rnn" not in name
    )


def _buffer_shape(
    buffer_name: str,
    buffer_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> tuple[int, ...]:
    if shape_env is None:
        return (buffer_extent,)
    name = _canonical_buffer_name(buffer_name)
    seq = shape_env.sequence_length
    hidden = shape_env.hidden_size
    q_heads = shape_env.attention_num_q_heads
    kv_heads = shape_env.attention_num_kv_heads
    head_dim = shape_env.attention_head_dim
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    topk = shape_env.attention_sparse_topk
    if name in {
        "hidden",
        "mamba3_delta",
        "m2rnn_hidden",
        "m2rnn_delta",
        "attention_hidden",
        "hidden_after_mamba3",
        "hidden_after_m2rnn",
        "attention_out",
    }:
        return (seq * hidden,)
    if _is_mamba3_state_like_buffer(buffer_name, name):
        return (
            shape_env.mamba_num_heads
            * shape_env.mamba_head_dim
            * shape_env.mamba_state_dim,
        )
    if name == "mamba3_angle_state":
        return (
            max(1, shape_env.mamba_num_heads * shape_env.mamba_num_rope_angles),
        )
    if name == "mamba3_conv_state":
        return (
            max(
                1,
                (shape_env.mamba_conv_kernel - 1) * shape_env.mamba_conv_channels,
            ),
        )
    if name == "mamba3_in_proj_weight":
        return (shape_env.mamba_in_proj_dim * hidden,)
    if name == "mamba3_out_proj_weight":
        return (hidden * shape_env.mamba_inner_dim,)
    if name == "mamba3_conv_weight":
        return (shape_env.mamba_conv_channels * shape_env.mamba_conv_kernel,)
    if name == "mamba3_conv_bias":
        return (shape_env.mamba_conv_channels,)
    if name in {"mamba3_dt_bias", "mamba3_D"}:
        return (shape_env.mamba_num_heads,)
    if name in {
        "mamba3_B_norm_weight",
        "mamba3_B_bias",
        "mamba3_C_norm_weight",
        "mamba3_C_bias",
    }:
        return (
            shape_env.mamba_effective_mimo_rank
            * shape_env.mamba_groups
            * shape_env.mamba_state_dim,
        )
    if name in {
        "mamba3_residual_to_m2rnn_norm_weight",
        "m2rnn_residual_to_attention_norm_weight",
        "residual_norm_weight",
        "final_norm_weight",
    }:
        return (hidden,)
    if name == "entry_rmsnorm_weight" or name.endswith("entry_rmsnorm_weight"):
        # Block A: the entry RMSNorm weight is a single ``hidden``-sized
        # vector applied to every row of the entry hidden state, mirroring
        # the eager ``layers.{first_in_region}.norm.weight`` parameter.
        return (hidden,)
    if name in {"target_ids", "target_mask"}:
        return (seq,)
    if name == "lm_head_weight":
        vocab = max(1, int(getattr(shape_env, "vocab_size", 0) or 0))
        return (vocab * hidden,)
    if name == "m2rnn_in_proj_weight":
        return (shape_env.m2rnn_in_proj_dim * hidden,)
    if name == "m2rnn_conv_weight":
        return (shape_env.m2rnn_conv_dim * shape_env.m2rnn_conv_kernel,)
    if name == "m2rnn_conv_bias":
        return (shape_env.m2rnn_conv_dim,)
    if name == "m2rnn_state_weight":
        return (
            shape_env.m2rnn_num_weight_heads
            * shape_env.m2rnn_v_head_dim
            * shape_env.m2rnn_v_head_dim,
        )
    if name in {"m2rnn_A_log", "m2rnn_dt_bias"}:
        return (shape_env.m2rnn_num_heads,)
    if name in {"m2rnn_D", "m2rnn_g_norm_weight"}:
        return (shape_env.m2rnn_num_heads * shape_env.m2rnn_v_head_dim,)
    if name == "m2rnn_out_proj_weight":
        return (hidden * shape_env.m2rnn_num_heads * shape_env.m2rnn_v_head_dim,)
    if name in {"m2rnn_h0", "m2rnn_h_state"}:
        return (
            shape_env.m2rnn_num_heads
            * shape_env.m2rnn_k_head_dim
            * shape_env.m2rnn_v_head_dim,
        )
    if name == "m2rnn_conv_state":
        return ((shape_env.m2rnn_conv_kernel - 1) * shape_env.m2rnn_conv_dim,)
    if name == "attention_q_proj_weight":
        return (q_dim * hidden,)
    if name == "attention_q_proj_bias":
        return (q_dim,)
    if name == "attention_sparse_kv_proj_weight":
        return (kv_dim * hidden,)
    if name == "attention_sparse_kv_proj_bias":
        return (kv_dim,)
    if name == "attention_rope_inv_freq":
        return (head_dim // 2,)
    if name == "attention_out_proj_weight":
        return (hidden * q_dim,)
    if name == "attention_out_proj_bias":
        return (hidden,)
    if name == "q_fp8":
        return (seq * q_heads * head_dim,)
    if name == "q_scale":
        return (seq * q_heads,)
    if name == "kv_fp8":
        return (seq * kv_heads * head_dim,)
    if name == "kv_scale":
        return (seq * kv_heads,)
    if name == "indices":
        return (seq * kv_heads * topk,)
    if name == "lse":
        return (seq * q_heads,)
    if name == "sparse_mla_sm_scale":
        return (1,)
    if name == "sparse_mla_sinks":
        return (q_heads,)
    if name == "sparse_mla_has_sinks":
        return (1,)
    if (
        name.endswith("_hidden")
        or name.endswith("_hidden_after")
        or name.endswith("_delta")
        or name.endswith("_out")
    ):
        return (seq * hidden,)
    return (buffer_extent,)


def _shape_literal(shape: Sequence[int]) -> str:
    if len(shape) == 1:
        return f"({int(shape[0])},)"
    return repr(tuple(int(dim) for dim in shape))


def _flattened_extent(shape: Sequence[int]) -> int:
    extent = 1
    for dim in shape:
        extent *= int(dim)
    return extent


_IDENTIFIER_RE = re.compile(r"\W+")


def _safe_identifier(name: object) -> str:
    identifier = _IDENTIFIER_RE.sub("_", str(name)).strip("_")
    if not identifier:
        identifier = "path_c_descriptor_region"
    if identifier[0].isdigit() or keyword.iskeyword(identifier):
        identifier = f"path_c_{identifier}"
    return identifier


def _validate_descriptors_match_nodes(
    nodes: Sequence[_ScheduleNodeView],
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> None:
    for node, descriptor in zip(nodes, descriptors, strict=True):
        if node.op_name != descriptor.op_name:
            raise ValueError(
                f"descriptor op {descriptor.op_name!r} does not match "
                f"region node {node.name!r} op {node.op_name!r}"
            )


def _descriptor_chain_for_region_or_signature(
    region: Any,
    fallback_signature: Sequence[str],
) -> tuple[PathCBrickScheduleDescriptor, ...]:
    if getattr(region, "nodes", None) is not None:
        signature = tuple(node.op_name for node in _node_views_for_region(region))
    else:
        signature = tuple(fallback_signature)
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(signature)
    )
    if descriptors is None:
        raise RuntimeError(
            f"no Path C brick descriptors registered for op signature {signature!r}"
        )
    return descriptors


def _require_path_c_region_graph(
    region: Any,
    *,
    function_name: str,
) -> Any:
    if getattr(region, "nodes", None) is not None:
        return region
    raise ValueError(
        f"{function_name} requires an explicit discovered Path C region; "
        "use build_mamba3_fp8_train_acceptance_fixture_region() only for the "
        "named acceptance fixture, or path_c_fusion_schedule_template(region) "
        "for model-derived brick chains"
    )


def _mamba3_fp8_train_acceptance_region(
    *,
    include_backward: bool = True,
    model_config: Any | None = None,
) -> PathCFusionRegion:
    return build_mamba3_fp8_train_acceptance_fixture_region(
        include_backward=include_backward,
        model_config=model_config,
    )


def _mamba3_fp8_train_acceptance_buffer_extent(
    model_config: Any | None = None,
) -> int:
    if model_config is None:
        return MAMBA3_FP8_TRAIN_BUFFER_EXTENT
    return int(
        getattr(
            model_config,
            "max_seq_length",
            getattr(model_config, "sequence_length", MAMBA3_FP8_TRAIN_BUFFER_EXTENT),
        )
    )


def _mamba3_fp8_train_acceptance_profile(
    model_config: Any | None = None,
) -> PathCFusionScheduleAcceptanceProfile:
    return PathCFusionScheduleAcceptanceProfile(
        op_signature=_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE,
        schedule_id=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID,
        schedule_name=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME,
        schedule_status=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS,
        implementation_kind="production",
        missing_reason=_PRODUCTION_SCHEDULE_REASON,
        required_codegen_steps=(
            "route_symbol_brick_chain_region",
            "real_model_parameter_abi_contract",
            "z3_sync_async_schedule_points",
            "cache_key_shape_specialization_audit",
        ),
        entry_symbol="mamba3_m2rnn_attention_fp8_train_block",
        required_real_abi_inputs=MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
        buffer_extent=_mamba3_fp8_train_acceptance_buffer_extent(model_config),
        required_region_tags=("mamba3_fp8_train_acceptance",),
    )


def _mamba3_fp8_train_prototype_profile() -> PathCFusionScheduleAcceptanceProfile:
    return PathCFusionScheduleAcceptanceProfile(
        op_signature=_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE,
        schedule_id="mamba3_m2rnn_attention_fp8_train_block_prototype_fwd_bwd",
        schedule_name=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME,
        schedule_status=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS,
        implementation_kind="prototype",
        missing_reason="prototype schedule scaffold is not a production implementation",
        required_codegen_steps=(
            "route_symbol_brick_chain_region",
            "real_model_parameter_abi_contract",
        ),
        entry_symbol="mamba3_m2rnn_attention_fp8_train_block",
        required_real_abi_inputs=MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
        buffer_extent=MAMBA3_FP8_TRAIN_BUFFER_EXTENT,
        required_region_tags=("mamba3_fp8_train_acceptance",),
    )


def _profiled_descriptor_target_for_region(
    region: Any,
    profile: PathCFusionScheduleAcceptanceProfile,
) -> PathCFusionScheduleTarget:
    signature = tuple(node.op_name for node in _node_views_for_region(region))
    if signature != profile.op_signature:
        raise RuntimeError(
            f"acceptance profile {profile.schedule_id!r} does not match "
            f"region op signature {signature!r}"
        )
    if not _acceptance_profile_matches_region(profile, region):
        raise RuntimeError(
            f"acceptance profile {profile.schedule_id!r} requires region tags "
            f"{profile.required_region_tags!r}"
        )
    descriptors = _required_descriptors_for_signature(signature)
    return _dynamic_descriptor_target_for_region(
        region,
        descriptors,
        acceptance_profile=profile,
    )


def _acceptance_profile_matches_region(
    profile: PathCFusionScheduleAcceptanceProfile,
    region: Any,
) -> bool:
    required_tags = set(profile.required_region_tags)
    if not required_tags:
        return True
    metadata = getattr(region, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    region_tags = set(metadata.get("path_c_acceptance_tags", ()))
    return required_tags.issubset(region_tags)


def _required_codegen_steps_from_descriptors(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    steps: list[str] = [
        "dynamic_region_graph_walk",
        "brick_descriptor_chain_resolution",
        "single_entry_tilelang_region",
    ]
    seen = set(steps)
    for descriptor in descriptors:
        for step in descriptor.required_codegen_steps:
            if step in seen:
                continue
            seen.add(step)
            steps.append(step)
    return tuple(steps)


def _brick_descriptor_statuses(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        f"{descriptor.op_name}:{descriptor.implementation_status}"
        for descriptor in descriptors
    )


def _brick_production_fragment_statuses(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        ":".join(
            (
                descriptor.op_name,
                descriptor.production_fragment_status,
                descriptor.production_source or "missing_source",
            )
        )
        for descriptor in descriptors
    )


def _brick_production_fragment_reasons(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        ":".join(
            (
                descriptor.op_name,
                descriptor.production_fragment_status,
                descriptor.production_fragment_reason or "no_reason",
            )
        )
        for descriptor in descriptors
    )


def _brick_production_fragment_blockers(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> tuple[str, ...]:
    return tuple(
        ":".join(
            (
                descriptor.op_name,
                descriptor.production_fragment_status,
                descriptor.production_fragment_reason or "no_reason",
            )
        )
        for descriptor in descriptors
        if descriptor.production_fragment_status != "production_region_inlined"
    )


def _production_fragments_complete(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> bool:
    return all(
        descriptor.production_fragment_status == "production_region_inlined"
        for descriptor in descriptors
    )


def _schedule_generator_status(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> str:
    if _production_fragments_complete(descriptors):
        return "production_region_fragments"
    families = {descriptor.schedule_family for descriptor in descriptors}
    if families == {"loop_descriptor_dataflow"}:
        return "loop_per_brick_descriptor_fragments"
    return "mixed_descriptor_schedule_fragments"


def _effective_implementation_kind(
    acceptance_profile: PathCFusionScheduleAcceptanceProfile | None,
    descriptors: Sequence[PathCBrickScheduleDescriptor],
) -> str:
    if acceptance_profile is None:
        if _production_fragments_complete(descriptors):
            return "production"
        return "prototype"
    if (
        acceptance_profile.implementation_kind == "production"
        and not _production_fragments_complete(descriptors)
    ):
        return "scaffold"
    return acceptance_profile.implementation_kind


class PathCFusionScheduleRegistry:
    """Pattern registry that selects fused schedules from a Path C region graph."""

    def __init__(
        self,
        targets: tuple[PathCFusionScheduleTarget, ...] = (),
        *,
        brick_registry: PathCBrickScheduleDescriptorRegistry | None = None,
        acceptance_profiles: tuple[
            PathCFusionScheduleAcceptanceProfile,
            ...,
        ] = (),
        enable_dynamic_descriptor_targets: bool = True,
    ) -> None:
        self._targets: dict[tuple[str, ...], PathCFusionScheduleTarget] = {}
        self._brick_registry = (
            brick_registry or default_path_c_brick_schedule_descriptor_registry()
        )
        self._acceptance_profiles = {
            profile.op_signature: profile for profile in acceptance_profiles
        }
        self._enable_dynamic_descriptor_targets = enable_dynamic_descriptor_targets
        for target in targets:
            self.register(target)

    def register(
        self,
        target: PathCFusionScheduleTarget,
    ) -> "PathCFusionScheduleRegistry":
        if not isinstance(target, PathCFusionScheduleTarget):
            raise TypeError("target must be PathCFusionScheduleTarget")
        if not target.op_signature:
            raise ValueError("Path C fusion schedule target op_signature must not be empty")
        self._targets[target.op_signature] = target
        return self

    def select(self, region: PathCFusionRegion) -> PathCFusionScheduleTarget | None:
        if not isinstance(region, PathCFusionRegion):
            raise TypeError("region must be PathCFusionRegion")
        signature = tuple(node.op_name for node in region.nodes)
        target = self._targets.get(signature)
        if target is not None:
            return target
        if not self._enable_dynamic_descriptor_targets:
            return None
        descriptors = self._brick_registry.descriptors_for_signature(signature)
        if descriptors is None:
            return None
        acceptance_profile = self._acceptance_profiles.get(signature)
        if (
            acceptance_profile is not None
            and not _acceptance_profile_matches_region(acceptance_profile, region)
        ):
            acceptance_profile = None
        return _dynamic_descriptor_target_for_region(
            region,
            descriptors,
            acceptance_profile=acceptance_profile,
        )


class PathCFusionScheduleOptimizer:
    """FX-like Path C optimizer facade over graph build, AOTAutograd, and schedules."""

    def __init__(
        self,
        region_name: str,
        *,
        registry: PathCFusionScheduleRegistry | None = None,
        z3_sync: Z3SyncSpec | None = None,
        enable_aot_autograd: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._region_name = region_name
        self._registry = registry or default_path_c_fusion_schedule_registry()
        self._z3_sync = z3_sync or Z3SyncSpec.minimize_sync_async()
        self._enable_aot_autograd = enable_aot_autograd
        self._metadata = dict(metadata or {})
        self._surfaces: list[FusionKernelSurface] = []

    def add_kernel(
        self,
        surface: FusionKernelSurface,
    ) -> "PathCFusionScheduleOptimizer":
        if not isinstance(surface, FusionKernelSurface):
            raise TypeError("surface must be FusionKernelSurface")
        self._surfaces.append(surface)
        return self

    def add_kernels(
        self,
        surfaces: Sequence[FusionKernelSurface],
    ) -> "PathCFusionScheduleOptimizer":
        for surface in surfaces:
            self.add_kernel(surface)
        return self

    def enable_aot_autograd(self) -> "PathCFusionScheduleOptimizer":
        self._enable_aot_autograd = True
        return self

    def build_region(self) -> PathCFusionRegion:
        region = build_path_c_fusion_region(
            region_name=self._region_name,
            surfaces=tuple(self._surfaces),
            z3_sync=self._z3_sync,
            metadata=self._metadata,
        )
        if self._enable_aot_autograd:
            return build_path_c_aot_autograd_region(region)
        return region

    def select_schedule_target(
        self,
        region: PathCFusionRegion | None = None,
    ) -> PathCFusionScheduleTarget | None:
        return self._registry.select(region or self.build_region())

    def plan(self) -> PathCFusionScheduleOptimizerPlan:
        region = self.build_region()
        target = self.select_schedule_target(region)
        schedule_template = (
            _attested_schedule_template_for_target(target, region)
            if target is not None
            else None
        )
        plan = compile_path_c_region(
            region,
            schedule_template=schedule_template,
            schedule_name=target.schedule_name if target is not None else None,
            schedule_status=target.schedule_status
            if target is not None
            else MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS,
        )
        if not isinstance(plan, FusionCompilePlan):
            raise TypeError("compile_path_c_region unexpectedly returned an artifact")
        return PathCFusionScheduleOptimizerPlan(
            region=region,
            plan=plan,
            schedule_target=target,
        )

    def compile(
        self,
        *,
        tilelang_lowerer: Callable[..., Any],
        target_name: str = "metal",
    ) -> CompiledPathCRegion:
        """Compile the selected target through its descriptor schedule template."""

        region = self.build_region()
        target = self.select_schedule_target(region)
        if target is None:
            raise RuntimeError(
                f"no Path C fusion schedule target registered for op signature "
                f"{tuple(node.op_name for node in region.nodes)!r}"
            )
        schedule_template = _attested_schedule_template_for_target(target, region)
        compiled = compile_path_c_region(
            region,
            schedule_template=schedule_template,
            schedule_name=target.schedule_name,
            schedule_status=target.schedule_status,
            tilelang_lowerer=tilelang_lowerer,
            target=target_name,
        )
        if not isinstance(compiled, CompiledPathCRegion):
            raise TypeError("compile_path_c_region unexpectedly returned a plan")
        return compiled


def mamba3_fp8_train_fusion_schedule_target() -> PathCFusionScheduleTarget:
    """Return the explicit Mamba3 acceptance target built from its fixture graph."""

    return _profiled_descriptor_target_for_region(
        _mamba3_fp8_train_acceptance_region(include_backward=True),
        _mamba3_fp8_train_acceptance_profile(),
    )


def mamba3_fp8_train_prototype_schedule_target() -> PathCFusionScheduleTarget:
    """Return the explicit prototype target built from its fixture graph."""

    return _profiled_descriptor_target_for_region(
        _mamba3_fp8_train_acceptance_region(include_backward=True),
        _mamba3_fp8_train_prototype_profile(),
    )


def _required_descriptors_for_signature(
    op_signature: Sequence[str],
) -> tuple[PathCBrickScheduleDescriptor, ...]:
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(op_signature)
    )
    if descriptors is None:
        raise RuntimeError(
            f"no Path C brick descriptors registered for op signature {tuple(op_signature)!r}"
        )
    return descriptors


def _dynamic_descriptor_target_for_region(
    region: Any,
    descriptors: tuple[PathCBrickScheduleDescriptor, ...],
    *,
    acceptance_profile: PathCFusionScheduleAcceptanceProfile | None = None,
) -> PathCFusionScheduleTarget:
    nodes = _node_views_for_region(region)
    signature = tuple(node.op_name for node in nodes)
    digest = sha256("|".join(signature).encode()).hexdigest()[:12]
    region_name = _safe_identifier(getattr(region, "name", "path_c_descriptor_region"))
    shape_env = _shape_env_for_region(region)
    buffer_extent = (
        shape_env.sequence_length
        if shape_env is not None
        else (
            acceptance_profile.buffer_extent
            if acceptance_profile is not None
            else DESCRIPTOR_DEFAULT_BUFFER_EXTENT
        )
    )
    internal_buffer_policy, loop_policy = _descriptor_codegen_policies_for_region(
        descriptors,
        shape_env,
    )
    descriptors = _descriptors_with_policy_fragment_statuses(
        descriptors,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
        shape_env=shape_env,
    )
    implementation_kind = _effective_implementation_kind(
        acceptance_profile,
        descriptors,
    )
    physical_abi_policy = _physical_abi_policy_for_region(
        nodes,
        shape_env=shape_env,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
    )
    train_step_output_abi = _region_requires_train_step_output_abi(region, nodes)
    schedule_name = (
        acceptance_profile.schedule_name
        if acceptance_profile is not None
        else f"{region_name}:descriptor_generated_fwd_bwd"
    )
    required_codegen_steps = _required_codegen_steps_from_descriptors(descriptors)
    required_real_abi_inputs = (
        acceptance_profile.required_real_abi_inputs
        if acceptance_profile is not None
        else _real_abi_inputs_for_nodes(nodes)
    )
    extra_steps: list[str] = []
    if required_real_abi_inputs:
        extra_steps.append("real_model_parameter_abi_contract")
    if getattr(getattr(region, "z3_sync", None), "enabled", False):
        extra_steps.append("z3_sync_async_schedule_points")
    if shape_env is not None:
        extra_steps.append("cache_key_shape_specialization_audit")
    if train_step_output_abi:
        extra_steps.append("train_step_scalar_output_abi")
        extra_steps.append("train_step_suffix_loss_input_abi")
    if extra_steps:
        seen = set(required_codegen_steps)
        required_codegen_steps = (
            *required_codegen_steps,
            *(step for step in extra_steps if step not in seen),
        )
    if acceptance_profile is not None:
        seen = set(required_codegen_steps)
        required_codegen_steps = (
            *required_codegen_steps,
            *(
                step
                for step in acceptance_profile.required_codegen_steps
                if step not in seen
            ),
        )
    schedule_id = (
        acceptance_profile.schedule_id
        if acceptance_profile is not None
        else f"path_c_descriptor_chain_{digest}"
    )
    dynamic_schedule_status = (
        MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS
        if implementation_kind == "production"
        else "descriptor_codegen_scaffold"
    )
    dynamic_missing_reason = (
        (
            "descriptor-generated Path C schedule is selected from the model "
            "route graph with all brick fragments production-inlined; it remains "
            "untrusted until compile, benchmark, profiling, memory, and cache "
            "receipts pass"
        )
        if implementation_kind == "production"
        else (
            "descriptor-generated Path C schedule was selected from the "
            "region graph, but it is not a named production acceptance target"
        )
    )
    return PathCFusionScheduleTarget(
        schedule_id=schedule_id,
        schedule_name=schedule_name,
        op_signature=signature,
        schedule_status=acceptance_profile.schedule_status
        if acceptance_profile is not None
        else dynamic_schedule_status,
        implementation_kind=implementation_kind,
        missing_reason=(
            acceptance_profile.missing_reason
            if acceptance_profile is not None
            else dynamic_missing_reason
        ),
        required_codegen_steps=required_codegen_steps,
        schedule_template=make_path_c_descriptor_schedule_template(
            descriptors,
            entry_symbol=acceptance_profile.entry_symbol
            if acceptance_profile is not None
            else region_name,
            buffer_extent=buffer_extent,
            shape_env=shape_env,
            internal_buffer_policy=internal_buffer_policy,
            loop_policy=loop_policy,
            physical_abi_policy=physical_abi_policy,
            train_step_output_abi=train_step_output_abi,
        ),
        required_real_abi_inputs=required_real_abi_inputs,
        brick_descriptors=descriptors,
        buffer_extent=buffer_extent,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
        physical_abi_policy=physical_abi_policy,
    )


def _region_requires_train_step_output_abi(
    region: Any,
    nodes: Sequence[_ScheduleNodeView],
) -> bool:
    metadata = getattr(region, "metadata", {}) or {}
    if bool(metadata.get("path_c_acceptance_fixture_abi", False)):
        return False
    if metadata.get("path_c_chain_source_region"):
        return False
    if not metadata.get("path_c_bricks"):
        return False
    return any(str(node.op_name).endswith("_bwd") for node in nodes)


def _physical_abi_policy_for_region(
    nodes: Sequence[_ScheduleNodeView],
    *,
    shape_env: PathCModelShapeEnv | None,
    internal_buffer_policy: str,
    loop_policy: str,
) -> str:
    internal_buffers = _internal_buffers_for_nodes(
        nodes,
        shape_env=shape_env,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
    )
    external_buffer_count = len(_external_buffers_for_nodes(nodes, internal_buffers))
    if external_buffer_count > DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT:
        return DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE
    return DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT


def _descriptor_codegen_policies_for_region(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    shape_env: PathCModelShapeEnv | None,
) -> tuple[str, str]:
    if shape_env is None:
        return (
            DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
            DESCRIPTOR_LOOP_POLICY_FLAT,
        )
    internal_buffer_policy = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL
    loop_policy = DESCRIPTOR_LOOP_POLICY_FLAT
    for descriptor in descriptors:
        preferred_internal = _validated_internal_buffer_policy(
            descriptor.preferred_internal_buffer_policy
        )
        preferred_loop = _validated_loop_policy(descriptor.preferred_loop_policy)
        if preferred_internal == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN:
            internal_buffer_policy = preferred_internal
        if preferred_loop == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN:
            loop_policy = preferred_loop
    return internal_buffer_policy, loop_policy


def _descriptors_with_policy_fragment_statuses(
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    internal_buffer_policy: str,
    loop_policy: str,
    shape_env: PathCModelShapeEnv | None,
) -> tuple[PathCBrickScheduleDescriptor, ...]:
    if not (
        shape_env is not None
        and internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
        and loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    ):
        return tuple(descriptors)

    adjusted: list[PathCBrickScheduleDescriptor] = []
    op_occurrences = Counter(descriptor.op_name for descriptor in descriptors)
    for descriptor in descriptors:
        if not _descriptor_production_policy_matches(
            descriptor,
            internal_buffer_policy=internal_buffer_policy,
            loop_policy=loop_policy,
            shape_env=shape_env,
            op_occurrences=op_occurrences,
        ):
            adjusted.append(descriptor)
            continue
        adjusted.append(
            replace(
                descriptor,
                required_codegen_steps=_append_unique_codegen_step(
                    descriptor.required_codegen_steps,
                    descriptor.production_fragment_codegen_step,
                ),
                production_fragment_status="production_region_inlined",
                production_fragment_reason=descriptor.production_fragment_inlined_reason,
            )
        )
    return tuple(adjusted)


def _descriptor_production_policy_matches(
    descriptor: PathCBrickScheduleDescriptor,
    *,
    internal_buffer_policy: str,
    loop_policy: str,
    shape_env: PathCModelShapeEnv,
    op_occurrences: Mapping[str, int],
) -> bool:
    if not descriptor.production_fragment_policy:
        return False
    if not descriptor.production_fragment_codegen_step:
        return False
    if not descriptor.production_fragment_inlined_reason:
        return False
    if (
        descriptor.max_production_hidden_size is not None
        and shape_env.hidden_size > descriptor.max_production_hidden_size
    ):
        return False
    if (
        descriptor.max_production_op_occurrences is not None
        and op_occurrences.get(descriptor.op_name, 0)
        > descriptor.max_production_op_occurrences
    ):
        min_hidden_size = descriptor.max_production_op_occurrences_min_hidden_size
        if min_hidden_size is None or shape_env.hidden_size >= min_hidden_size:
            return False
    return (
        descriptor.production_fragment_policy == loop_policy
        and descriptor.preferred_internal_buffer_policy == internal_buffer_policy
        and descriptor.preferred_loop_policy == loop_policy
    )


def _append_unique_codegen_step(
    steps: Sequence[str],
    step: str,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in steps)
    if step in values:
        return values
    return (*values, step)


def _real_abi_inputs_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
) -> tuple[str, ...]:
    internal_buffers = _internal_buffers_for_nodes(nodes)
    external_buffers = _external_buffers_for_nodes(nodes, internal_buffers)
    return tuple(
        name
        for name in external_buffers
        if _canonical_buffer_name(name)
        in (
            *MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
            *_GENERIC_MODEL_REAL_ABI_INPUT_SUFFIXES,
        )
    )


def default_path_c_fusion_schedule_registry() -> PathCFusionScheduleRegistry:
    """Return the default descriptor-backed registry for Path C fusion."""

    return PathCFusionScheduleRegistry()


def prototype_path_c_fusion_schedule_registry() -> PathCFusionScheduleRegistry:
    """Return a registry that selects descriptor-generated prototype schedules."""

    return PathCFusionScheduleRegistry(
        acceptance_profiles=(_mamba3_fp8_train_prototype_profile(),),
    )


def select_path_c_fusion_schedule_target(
    region: PathCFusionRegion,
    *,
    registry: PathCFusionScheduleRegistry | None = None,
) -> PathCFusionScheduleTarget | None:
    """Select the fused schedule target matching ``region``'s op signature."""

    return (registry or default_path_c_fusion_schedule_registry()).select(region)


def path_c_fusion_schedule_template(
    region: PathCFusionRegion,
    *,
    registry: PathCFusionScheduleRegistry | None = None,
) -> Any:
    """Generate a descriptor schedule from a discovered Path C model region."""

    target = select_path_c_fusion_schedule_target(region, registry=registry)
    if target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in region.nodes)!r}"
        )
    return target.schedule_template(region)


def plan_path_c_fusion_schedule_for_region(
    region: PathCFusionRegion,
    *,
    include_backward: bool = True,
    registry: PathCFusionScheduleRegistry | None = None,
) -> PathCFusionScheduleOptimizerPlan:
    """Plan a Path C fused schedule from an already-discovered model region."""

    if not isinstance(region, PathCFusionRegion):
        raise TypeError("region must be PathCFusionRegion")
    has_backward = any(node.op_name.endswith("_bwd") for node in region.nodes)
    return (
        PathCFusionScheduleOptimizer(
            region.name,
            registry=registry,
            metadata=region.metadata,
            enable_aot_autograd=include_backward and not has_backward,
        )
        .add_kernels(_surfaces_from_region(region))
        .plan()
    )


def plan_path_c_direct_fusion_chain_for_region(
    region: PathCFusionRegion,
    *,
    include_backward: bool = True,
    max_kernel_buffers: int = DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT,
    max_segment_nodes: int | None = None,
    registry: PathCFusionScheduleRegistry | None = None,
) -> PathCFusionScheduleChainPlan:
    """Greedily split a Path C region into direct-buffer fused segments.

    This is the generic escape hatch when a single direct-buffer train-block
    would exceed Metal's portable buffer slot limit.  It never falls back to
    dtype-bank packing; segments that cannot be expressed with direct buffers
    under ``max_kernel_buffers`` are reported as blocked.
    """

    if not isinstance(region, PathCFusionRegion):
        raise TypeError("region must be PathCFusionRegion")
    if max_kernel_buffers <= 0:
        raise ValueError("max_kernel_buffers must be positive")
    if max_segment_nodes is not None and max_segment_nodes <= 0:
        raise ValueError("max_segment_nodes must be positive when provided")
    working_region = (
        build_path_c_aot_autograd_region(region)
        if include_backward
        and not any(node.op_name.endswith("_bwd") for node in region.nodes)
        else region
    )
    nodes = tuple(working_region.nodes)
    segments: list[PathCFusionScheduleChainSegment] = []
    start = 0
    selector = registry or default_path_c_fusion_schedule_registry()
    while start < len(nodes):
        best: PathCFusionScheduleChainSegment | None = None
        first_failure: str | None = None
        for end in range(start + 1, len(nodes) + 1):
            if (
                max_segment_nodes is not None
                and end - start > max_segment_nodes
            ):
                first_failure = (
                    f"direct-buffer segment reached max_segment_nodes="
                    f"{max_segment_nodes}"
                )
                break
            candidate_region = _subregion_from_nodes(
                working_region,
                start=start,
                end=end,
                name=f"{working_region.name}_chain_{start}_{end}",
            )
            execution_phase = _path_c_schedule_segment_execution_phase(
                candidate_region.nodes
            )
            if execution_phase == "mixed":
                first_failure = (
                    "direct-chain segment would cross the forward/backward "
                    "execution boundary required by the loss cotangent bridge"
                )
                break
            target = selector.select(candidate_region)
            if target is None:
                first_failure = (
                    f"no descriptor target for op signature "
                    f"{tuple(node.op_name for node in candidate_region.nodes)!r}"
                )
                break
            direct_target = _target_with_physical_abi_policy(
                target,
                candidate_region,
                DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
            )
            try:
                parameter_count = _kernel_parameter_count_for_target(
                    candidate_region,
                    direct_target,
                )
            except Exception as exc:
                first_failure = str(exc)
                if best is None:
                    break
                continue
            if parameter_count > max_kernel_buffers:
                first_failure = (
                    f"direct-buffer segment needs {parameter_count} kernel "
                    f"buffers, above limit {max_kernel_buffers}"
                )
                break
            plan = compile_path_c_region(
                candidate_region,
                schedule_template=_attested_schedule_template_for_target(
                    direct_target,
                    candidate_region,
                ),
                schedule_name=direct_target.schedule_name,
                schedule_status=direct_target.schedule_status,
            )
            if not isinstance(plan, FusionCompilePlan):
                raise TypeError("compile_path_c_region unexpectedly returned an artifact")
            best = PathCFusionScheduleChainSegment(
                index=len(segments),
                node_start=start,
                node_end=end,
                region=candidate_region,
                plan=plan,
                schedule_target=direct_target,
                kernel_parameter_count=parameter_count,
                physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
                status="ok",
                reason="direct-buffer segment fits the portable Metal buffer limit",
                execution_phase=execution_phase,
            )
        if best is not None:
            segments.append(best)
            start = best.node_end
            continue
        blocked_region = _subregion_from_nodes(
            working_region,
            start=start,
            end=start + 1,
            name=f"{working_region.name}_chain_{start}_{start + 1}",
        )
        segments.append(
            PathCFusionScheduleChainSegment(
                index=len(segments),
                node_start=start,
                node_end=start + 1,
                region=blocked_region,
                plan=None,
                schedule_target=None,
                kernel_parameter_count=None,
                physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_DIRECT,
                status="blocked",
                reason=first_failure or "direct-buffer segment planning failed",
                execution_phase=_path_c_schedule_segment_execution_phase(
                    blocked_region.nodes
                ),
            )
        )
        start += 1
    blocked = tuple(segment for segment in segments if segment.status != "ok")
    return PathCFusionScheduleChainPlan(
        source_region=working_region,
        max_kernel_buffers=max_kernel_buffers,
        segments=tuple(segments),
        status="ready" if not blocked else "blocked",
        reason=(
            "all chain segments fit direct-buffer portable Metal limits"
            if not blocked
            else "at least one chain segment cannot be expressed as direct buffers"
        ),
    )


def plan_path_c_direct_fusion_chains_for_model(
    model: Any,
    *,
    region_prefix: str | None = None,
    include_backward: bool = True,
    min_route_bricks: int = 2,
    max_kernel_buffers: int = DESCRIPTOR_PORTABLE_KERNEL_BUFFER_LIMIT,
    max_segment_nodes: int | None = None,
    registry: PathCFusionScheduleRegistry | None = None,
    sequence_length: int | None = None,
) -> tuple[PathCFusionScheduleChainPlan, ...]:
    """Plan direct-buffer fused schedule chains for every supported model region.

    This is the direct-buffer sibling of
    ``plan_path_c_fusion_schedules_for_model``: discover regions from the
    model's brick graph first, then split each discovered region into generic
    direct-buffer segments.  It does not consult named acceptance fixtures.
    """

    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=region_prefix,
        include_backward=False,
        min_route_bricks=min_route_bricks,
        sequence_length=sequence_length,
    )
    return tuple(
        plan_path_c_direct_fusion_chain_for_region(
            region,
            include_backward=include_backward,
            max_kernel_buffers=max_kernel_buffers,
            max_segment_nodes=max_segment_nodes,
            registry=registry,
        )
        for region in regions
    )


def plan_path_c_fusion_schedules_for_model(
    model: Any,
    *,
    region_prefix: str | None = None,
    include_backward: bool = True,
    min_route_bricks: int = 2,
    registry: PathCFusionScheduleRegistry | None = None,
    sequence_length: int | None = None,
) -> tuple[PathCFusionScheduleOptimizerPlan, ...]:
    """Plan Path C fused schedules for every supported region in ``model``.

    This is the production-oriented entrypoint: discover regions from the
    model's bricks first, then resolve schedule descriptors per region.  Named
    acceptance fixtures are not consulted.
    """

    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix=region_prefix,
        include_backward=False,
        min_route_bricks=min_route_bricks,
        sequence_length=sequence_length,
    )
    return tuple(
        plan_path_c_fusion_schedule_for_region(
            region,
            include_backward=include_backward,
            registry=registry,
        )
        for region in regions
    )


def _subregion_from_nodes(
    region: PathCFusionRegion,
    *,
    start: int,
    end: int,
    name: str,
) -> PathCFusionRegion:
    selected_nodes = tuple(region.nodes[start:end])
    if not selected_nodes:
        raise ValueError("subregion node slice must not be empty")
    metadata = {
        **dict(region.metadata),
        "path_c_chain_source_region": region.name,
        "path_c_chain_node_start": start,
        "path_c_chain_node_end": end,
    }
    return build_path_c_fusion_region(
        region_name=name,
        surfaces=tuple(
            FusionKernelSurface.path_c(
                name=node.name,
                op_name=node.op_name,
                inputs=node.inputs,
                outputs=node.outputs,
                backward=node.backward,
                backend=node.backend,
            )
            for node in selected_nodes
        ),
        z3_sync=region.z3_sync,
        metadata=metadata,
    )


def _target_with_physical_abi_policy(
    target: PathCFusionScheduleTarget,
    region: PathCFusionRegion,
    physical_abi_policy: str,
) -> PathCFusionScheduleTarget:
    validated_policy = _validated_physical_abi_policy(physical_abi_policy)
    if target.physical_abi_policy == validated_policy:
        return target
    shape_env = _shape_env_for_region(region)
    return replace(
        target,
        schedule_template=make_path_c_descriptor_schedule_template(
            target.brick_descriptors,
            entry_symbol=getattr(region, "entry_symbol", None)
            or getattr(region, "name", None),
            buffer_extent=target.buffer_extent,
            shape_env=shape_env,
            internal_buffer_policy=target.internal_buffer_policy,
            loop_policy=target.loop_policy,
            physical_abi_policy=validated_policy,
            max_rows_per_launch=target.max_rows_per_launch,
            row_dispatch_mode=target.row_dispatch_mode,
        ),
        physical_abi_policy=validated_policy,
    )


def _target_with_max_rows_per_launch(
    target: PathCFusionScheduleTarget,
    region: PathCFusionRegion,
    max_rows_per_launch: int | None,
    row_dispatch_mode: str,
) -> PathCFusionScheduleTarget:
    validated_rows = _validated_max_rows_per_launch(max_rows_per_launch)
    validated_mode = _validated_row_dispatch_mode(row_dispatch_mode)
    if (
        target.max_rows_per_launch == validated_rows
        and target.row_dispatch_mode == validated_mode
    ):
        return target
    shape_env = _shape_env_for_region(region)
    return replace(
        target,
        schedule_template=make_path_c_descriptor_schedule_template(
            target.brick_descriptors,
            entry_symbol=getattr(region, "entry_symbol", None)
            or getattr(region, "name", None),
            buffer_extent=target.buffer_extent,
            shape_env=shape_env,
            internal_buffer_policy=target.internal_buffer_policy,
            loop_policy=target.loop_policy,
            physical_abi_policy=target.physical_abi_policy,
            max_rows_per_launch=validated_rows,
            row_dispatch_mode=validated_mode,
        ),
        max_rows_per_launch=validated_rows,
        row_dispatch_mode=validated_mode,
    )


def _kernel_parameter_count_for_target(
    region: PathCFusionRegion,
    target: PathCFusionScheduleTarget,
) -> int:
    return len(tuple(target.schedule_template(region).params))


def _attested_schedule_template_for_target(
    target: PathCFusionScheduleTarget,
    region: PathCFusionRegion,
) -> Callable[[Any], Any]:
    kwargs = {
        "implementation_kind": target.implementation_kind,
        "required_real_abi_inputs": target.required_real_abi_inputs,
    }
    if target.implementation_kind == "production":
        kwargs["production_schedule_id"] = target.schedule_id
    return mark_path_c_schedule_template_for_region(
        target.schedule_template,
        region,
        **kwargs,
    )


def path_c_fusion_schedule_spec(
    region: PathCFusionRegion | None = None,
    *,
    contract: FusionScheduleContractStatus | None = None,
    target: PathCFusionScheduleTarget | None = None,
) -> PathCFusionScheduleSpec:
    """Return the schedule contract selected from ``region``."""

    if region is None:
        raise ValueError(
            "path_c_fusion_schedule_spec requires a discovered Path C region; "
            "use mamba3_fp8_train_fusion_schedule_spec() only for the explicit "
            "Mamba3 acceptance fixture"
        )
    resolved_region = region
    resolved_contract = contract or _contract_for_region(resolved_region)
    resolved_target = target or select_path_c_fusion_schedule_target(resolved_region)
    if resolved_target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in resolved_region.nodes)!r}"
        )
    missing_real_abi_inputs = _missing_real_abi_inputs(
        resolved_contract,
        resolved_target,
    )
    return PathCFusionScheduleSpec(
        schedule_id=resolved_target.schedule_id,
        schedule_name=resolved_target.schedule_name,
        region_name=resolved_region.name,
        implementation_kind=resolved_target.implementation_kind,
        implementation_status=resolved_target.schedule_status,
        missing_reason=resolved_target.missing_reason,
        trusted_by_default=(
            resolved_target.implementation_kind == "production"
            and resolved_target.schedule_id
            in trusted_path_c_production_schedule_ids()
        ),
        contract_name=resolved_contract.name,
        contract_key=resolved_contract.key,
        shape_env_key=resolved_contract.shape_env_key,
        op_signature=resolved_contract.op_signature,
        required_internal_buffers=resolved_contract.required_internal_buffers,
        required_external_buffers=resolved_contract.required_external_buffers,
        required_real_abi_inputs=resolved_target.required_real_abi_inputs,
        required_real_abi_input_shapes=_required_real_abi_input_shapes(
            resolved_region,
            resolved_target,
        ),
        missing_real_abi_inputs=missing_real_abi_inputs,
        real_abi_contract_complete=not missing_real_abi_inputs,
        required_codegen_steps=resolved_target.required_codegen_steps,
        schedule_generator=resolved_target.schedule_generator,
        schedule_generator_status=_schedule_generator_status(
            resolved_target.brick_descriptors
        ),
        internal_buffer_policy=resolved_target.internal_buffer_policy,
        loop_policy=resolved_target.loop_policy,
        buffer_extent=resolved_target.buffer_extent,
        loop_extent=_descriptor_loop_extent(
            resolved_contract.required_external_buffers,
            resolved_target.buffer_extent,
            _shape_env_for_region(resolved_region),
        ),
        brick_ops=tuple(
            descriptor.op_name for descriptor in resolved_target.brick_descriptors
        ),
        brick_schedule_families=tuple(
            descriptor.schedule_family
            for descriptor in resolved_target.brick_descriptors
        ),
        brick_descriptor_statuses=_brick_descriptor_statuses(
            resolved_target.brick_descriptors
        ),
        brick_production_fragment_statuses=(
            _brick_production_fragment_statuses(
                resolved_target.brick_descriptors
            )
        ),
        brick_production_fragment_reasons=(
            _brick_production_fragment_reasons(
                resolved_target.brick_descriptors
            )
        ),
        brick_production_fragment_blockers=(
            _brick_production_fragment_blockers(
                resolved_target.brick_descriptors
            )
        ),
        production_fragments_complete=_production_fragments_complete(
            resolved_target.brick_descriptors
        ),
    )


def mamba3_fp8_train_fusion_schedule_spec(
    region: PathCFusionRegion | None = None,
    *,
    contract: FusionScheduleContractStatus | None = None,
    target: PathCFusionScheduleTarget | None = None,
) -> Mamba3Fp8TrainFusionScheduleSpec:
    """Return the named Mamba3 acceptance schedule target for ``region``."""

    resolved_region = _require_path_c_region_graph(
        region,
        function_name="mamba3_fp8_train_fusion_schedule_spec",
    )
    resolved_target = target or mamba3_fp8_train_fusion_schedule_target()
    generic = path_c_fusion_schedule_spec(
        resolved_region,
        contract=contract,
        target=resolved_target,
    )
    return Mamba3Fp8TrainFusionScheduleSpec(**generic.__dict__)


def _missing_real_abi_inputs(
    contract: FusionScheduleContractStatus,
    target: PathCFusionScheduleTarget,
) -> tuple[str, ...]:
    external_buffers = set(contract.required_external_buffers)
    return tuple(
        name
        for name in target.required_real_abi_inputs
        if name not in external_buffers
    )


def _required_real_abi_input_shapes(
    region: PathCFusionRegion,
    target: PathCFusionScheduleTarget,
) -> tuple[str, ...]:
    shape_env = _shape_env_for_region(region)
    return tuple(
        f"{name}:{_shape_literal(_buffer_shape(name, target.buffer_extent, shape_env))}"
        for name in target.required_real_abi_inputs
    )


def plan_mamba3_fp8_train_fusion_schedule(
    *,
    include_backward: bool = True,
    model_config: Any | None = None,
    max_rows_per_launch: int | None = DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
) -> Mamba3Fp8TrainFusionSchedulePlan:
    """Build and plan the named Mamba3 FP8 train-block acceptance schedule."""

    fwd_region = _mamba3_fp8_train_acceptance_region(
        include_backward=False,
        model_config=model_config,
    )
    acceptance_registry = PathCFusionScheduleRegistry(
        acceptance_profiles=(_mamba3_fp8_train_acceptance_profile(model_config),),
    )
    optimized = plan_path_c_fusion_schedule_for_region(
        fwd_region,
        include_backward=include_backward,
        registry=acceptance_registry,
    )
    target = optimized.schedule_target
    if target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in optimized.region.nodes)!r}"
        )
    target = _target_with_max_rows_per_launch(
        target,
        optimized.region,
        max_rows_per_launch,
        row_dispatch_mode,
    )
    return Mamba3Fp8TrainFusionSchedulePlan(
        region=optimized.region,
        plan=optimized.plan,
        schedule_spec=mamba3_fp8_train_fusion_schedule_spec(
            optimized.region,
            contract=optimized.plan.schedule_contract,
            target=target,
        ),
    )


def compile_mamba3_fp8_train_fusion_schedule(
    *,
    tilelang_lowerer: Callable[..., Any] | None = None,
    target_name: str = "metal",
    include_backward: bool = True,
    model_config: Any | None = None,
    max_rows_per_launch: int | None = DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH,
    row_dispatch_mode: str = DESCRIPTOR_DEFAULT_ROW_DISPATCH_MODE,
) -> CompiledMamba3Fp8TrainFusionSchedule:
    """Lower the named Mamba3 FP8 train-block acceptance schedule.

    This is the compiled counterpart of
    :func:`plan_mamba3_fp8_train_fusion_schedule`: it selects the named
    acceptance profile from the region graph, attests the generated descriptor
    schedule for that exact contract, and invokes the supplied TileLang lowerer.
    The schedule is still not trusted-by-default; callers must inspect the
    returned contract and external receipts before enabling it as a default.
    """

    fwd_region = _mamba3_fp8_train_acceptance_region(
        include_backward=False,
        model_config=model_config,
    )
    acceptance_registry = PathCFusionScheduleRegistry(
        acceptance_profiles=(_mamba3_fp8_train_acceptance_profile(model_config),),
    )
    optimizer = PathCFusionScheduleOptimizer(
        fwd_region.name,
        registry=acceptance_registry,
        metadata=fwd_region.metadata,
        enable_aot_autograd=include_backward,
    ).add_kernels(_surfaces_from_region(fwd_region))
    region = optimizer.build_region()
    target = optimizer.select_schedule_target(region)
    if target is None:
        raise RuntimeError(
            f"no Path C fusion schedule target registered for op signature "
            f"{tuple(node.op_name for node in region.nodes)!r}"
        )
    target = _target_with_max_rows_per_launch(
        target,
        region,
        max_rows_per_launch,
        row_dispatch_mode,
    )
    lowerer = tilelang_lowerer or tilelang_single_entry_lowerer
    schedule_template = _attested_schedule_template_for_target(target, region)
    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name=target.schedule_name,
        schedule_status=target.schedule_status,
        tilelang_lowerer=lowerer,
        target=target_name,
    )
    if not isinstance(compiled, CompiledPathCRegion):
        raise TypeError("compile_path_c_region unexpectedly returned a plan")
    return CompiledMamba3Fp8TrainFusionSchedule(
        region=region,
        compiled=compiled,
        schedule_spec=mamba3_fp8_train_fusion_schedule_spec(
            region,
            contract=compiled.plan.schedule_contract,
            target=target,
        ),
    )


def _surfaces_from_region(region: PathCFusionRegion) -> tuple[FusionKernelSurface, ...]:
    return tuple(
        FusionKernelSurface.path_c(
            name=node.name,
            op_name=node.op_name,
            inputs=node.inputs,
            outputs=node.outputs,
            backward=node.backward,
            backend=node.backend,
        )
        for node in region.nodes
    )


def _contract_for_region(
    region: PathCFusionRegion,
) -> FusionScheduleContractStatus:
    plan = compile_path_c_region(region)
    if not isinstance(plan, FusionCompilePlan):
        raise TypeError("compile_path_c_region unexpectedly returned an artifact")
    if plan.schedule_contract is None:
        raise RuntimeError("Path C fusion region did not produce a schedule contract")
    return plan.schedule_contract
