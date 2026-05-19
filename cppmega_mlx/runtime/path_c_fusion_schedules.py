"""Descriptor-driven schedule planning for Path C fusion regions.

The named Mamba3 FP8 train-block target is an acceptance preset, not the source
of truth.  Schedule construction starts from a region graph, resolves each
known op through a brick descriptor, and then emits a single-entry TileLang
template for that chain.  The production schedule ID remains untrusted by
default until compile, profile, memory, and 1B matrix receipts prove it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import keyword
import linecache
import re
from typing import Any

from cppmega_mlx.runtime.path_c_fusion import (
    CompiledPathCRegion,
    FusionCompilePlan,
    FusionKernelSurface,
    FusionScheduleContractStatus,
    MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
    PathCFusionRegion,
    PathCModelShapeEnv,
    Z3SyncSpec,
    build_path_c_aot_autograd_region,
    build_path_c_fusion_region,
    build_path_c_model_regions_from_model,
    build_mamba3_fp8_train_acceptance_fixture_region,
    compile_path_c_region,
    mark_path_c_schedule_template_for_region,
    trusted_path_c_production_schedule_ids,
)


__all__ = [
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID",
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME",
    "MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS",
    "MAMBA3_FP8_TRAIN_BUFFER_EXTENT",
    "MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME",
    "MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS",
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
MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS = "descriptor_scaffold_untrusted"
PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR = "dynamic_brick_descriptor_generator"
DESCRIPTOR_DEFAULT_BUFFER_EXTENT = 4
DESCRIPTOR_DEFAULT_THREADS = 256
DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL = "scalar_local"
DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN = "row_local_hidden"
DESCRIPTOR_LOOP_POLICY_FLAT = "flat"
DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN = "row_phased_hidden"
_GENERIC_MODEL_REAL_ABI_INPUT_SUFFIXES = ("residual_norm_weight",)


def _mamba3_fp8_train_buffer_extent() -> int:
    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile

    return int(local_gb10_quarter_profile().max_seq_length)


MAMBA3_FP8_TRAIN_BUFFER_EXTENT = _mamba3_fp8_train_buffer_extent()

_PRODUCTION_SCHEDULE_REASON = (
    "dynamic brick descriptors can construct a single-entry TileLang/TIR "
    "schedule for the model-semantic mamba3 + residual/RMSNorm + m2rnn + "
    "attention_qkv_projection + sparse_mla_fp8_apply fwd/bwd region, but the "
    "schedule body still uses scaffold descriptor fragments; it is not a ready "
    "production fused kernel until all brick fragments are production-inlined "
    "and 1B matrix, profiling, memory, and cache receipts pass"
)
_MAMBA3_FP8_TRAIN_FWD_BWD_OP_SIGNATURE = (
    "mamba3_mimo",
    "residual_rmsnorm",
    "m2rnn",
    "residual_rmsnorm",
    "attention_qkv_projection",
    "sparse_mla_fp8_apply",
    "attention_qkv_projection_bwd",
    "residual_rmsnorm_bwd",
    "m2rnn_bwd",
    "residual_rmsnorm_bwd",
    "mamba3_mimo_bwd",
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
class Mamba3Fp8TrainFusionSchedulePlan:
    """High-level planner output for the Mamba3 FP8 train-block target."""

    region: PathCFusionRegion
    plan: FusionCompilePlan
    schedule_spec: Mamba3Fp8TrainFusionScheduleSpec


@dataclass(frozen=True)
class _ScheduleNodeView:
    name: str
    op_name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class PathCBrickScheduleFragment:
    allocations: tuple[str, ...]
    statements: tuple[str, ...]


_ScheduleNodeFragment = PathCBrickScheduleFragment


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
                op_name="mamba3_mimo",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "mamba3_scan_descriptor",
                    "mamba3_scan_fwd_internal_buffers",
                ),
                description="Mamba3 scan brick descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.mamba3_path_c:"
                    "mamba3_mimo_fwd_path_c/mamba3_mimo_bwd_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits a Mamba3 descriptor "
                    "fragment, but it is still scaffold code rather than the "
                    "production shape-specialized scan schedule; the existing "
                    "Path C scan kernel expects pre-projected x/B/C/z/A/dt "
                    "buffers, while this model-region node still starts from "
                    "block-level in_proj/conv/norm parameters"
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
                    "Path C backward source, but the fused region still emits a "
                    "placeholder gradient fragment; final-gradient owner-output "
                    "lowering is not implemented in the train-block body, and "
                    "the existing Path C backward kernel consumes scan-level "
                    "dy/x/B/C/z/A/dt/D/h0 tensors rather than block-level "
                    "model weights"
                ),
                fragment_emitter=_emit_owner_output_backward_source,
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
                    "residual/RMSNorm backward has an explicit descriptor, "
                    "and one full-hidden bridge is emitted as row-phased "
                    "recompute; full 1B train-blocks with multiple "
                    "residual/RMSNorm backward bridges stay fail-closed because "
                    "the native TileLang lowerer timed out when that occurrence "
                    "gate was removed"
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
                max_production_op_occurrences=1,
                max_production_op_occurrences_min_hidden_size=257,
                fragment_emitter=None,
            ),
            PathCBrickScheduleDescriptor(
                op_name="m2rnn",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "m2rnn_descriptor",
                    "m2rnn_packed_post_internal_buffers",
                ),
                description="M2RNN packed post descriptor",
                production_source=(
                    "cppmega_mlx.nn._tilelang.m2rnn_path_c:"
                    "m2rnn_apply_mapped_packed_post_with_state_path_c"
                ),
                production_fragment_status="region_fragment_inlined_unoptimized",
                production_fragment_reason=(
                    "the fused train-block region emits an M2RNN descriptor "
                    "fragment, but it is still scaffold code rather than the "
                    "production recurrence and post-residual schedule; the "
                    "existing mapped-packed Path C kernels expect projected "
                    "conv_input/xf/projected tensors, while this model-region "
                    "node still starts from block-level projection/conv/state "
                    "parameters"
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
                fragment_emitter=_emit_m2rnn_bwd_source,
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
                    "descriptor fragment with the real ABI, but it is still "
                    "scaffold code rather than the production q/kv projection, "
                    "RoPE, and FP8 prepare schedule"
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
                fragment_emitter=_emit_attention_qkv_projection_bwd_source,
            ),
            PathCBrickScheduleDescriptor(
                op_name="sparse_mla_fp8_apply",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=(
                    "sparse_mla_fp8_apply_descriptor",
                    "sparse_mla_fp8_apply_owner_output",
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
                    "descriptor fragment, but it is still scaffold code rather "
                    "than the production prepared apply schedule"
                ),
                fragment_emitter=_emit_sparse_mla_fp8_apply_source,
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

    def descriptor_schedule_template(template_region: Any) -> Any:
        return build_path_c_descriptor_prim_func(
            template_region,
            descriptors,
            entry_symbol=entry_symbol,
            buffer_extent=extent,
            shape_env=shape_env,
            internal_buffer_policy=validated_internal_buffer_policy,
            loop_policy=validated_loop_policy,
        )

    descriptor_schedule_template._cppmega_path_c_schedule_generator = (
        PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    )
    descriptor_schedule_template._cppmega_path_c_brick_ops = tuple(
        descriptor.op_name for descriptor in descriptors
    )
    descriptor_schedule_template._cppmega_path_c_buffer_extent = extent
    descriptor_schedule_template._cppmega_path_c_shape_env = shape_env
    descriptor_schedule_template._cppmega_path_c_internal_buffer_policy = (
        validated_internal_buffer_policy
    )
    descriptor_schedule_template._cppmega_path_c_loop_policy = validated_loop_policy
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


def build_path_c_descriptor_prim_func(
    region: Any,
    brick_descriptors: Sequence[PathCBrickScheduleDescriptor],
    *,
    entry_symbol: str | None = None,
    buffer_extent: int = DESCRIPTOR_DEFAULT_BUFFER_EXTENT,
    shape_env: PathCModelShapeEnv | None = None,
    internal_buffer_policy: str = DESCRIPTOR_INTERNAL_BUFFER_POLICY_SCALAR_LOCAL,
    loop_policy: str = DESCRIPTOR_LOOP_POLICY_FLAT,
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
    internal_buffers = _internal_buffers_for_nodes(nodes)
    dtype_by_buffer = {
        name: _buffer_dtype(name)
        for node in nodes
        for name in (*node.inputs, *node.outputs)
    }
    external_buffers = _external_buffers_for_nodes(nodes, internal_buffers)
    extent = _validated_buffer_extent(buffer_extent)
    resolved_shape_env = shape_env or _shape_env_for_region(region)
    validated_internal_buffer_policy = _validated_internal_buffer_policy(
        internal_buffer_policy
    )
    validated_loop_policy = _validated_loop_policy(loop_policy)
    entry_name = _safe_identifier(
        entry_symbol or getattr(region, "entry_symbol", None) or getattr(region, "name", None)
        or "path_c_descriptor_region"
    )
    internal_buffer_shapes = _internal_buffer_shapes(
        internal_buffers,
        validated_internal_buffer_policy,
        resolved_shape_env,
    )
    source = _descriptor_prim_func_source(
        entry_name=entry_name,
        nodes=nodes,
        descriptors=descriptors,
        internal_buffers=internal_buffers,
        internal_buffer_shapes=internal_buffer_shapes,
        internal_buffer_policy=validated_internal_buffer_policy,
        loop_policy=validated_loop_policy,
        external_buffers=external_buffers,
        dtype_by_buffer=dtype_by_buffer,
        buffer_extent=extent,
        shape_env=resolved_shape_env,
    )

    import tilelang.language as T

    filename = f"<path_c_descriptor_schedule:{entry_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, Any] = {"T": T}
    exec(compile(source, filename, "exec"), namespace)
    prim_func = namespace[entry_name]
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
    prim_func._cppmega_path_c_internal_buffer_shapes = internal_buffer_shapes
    prim_func._cppmega_path_c_buffer_abi_shapes = {
        name: _buffer_shape(name, extent, resolved_shape_env)
        for name in external_buffers
    }
    prim_func._cppmega_path_c_loop_extent = _descriptor_loop_extent(
        external_buffers,
        extent,
        resolved_shape_env,
    )
    prim_func._cppmega_path_c_generated_source = source
    return prim_func


def _descriptor_prim_func_source(
    *,
    entry_name: str,
    nodes: Sequence[_ScheduleNodeView],
    descriptors: Sequence[PathCBrickScheduleDescriptor],
    internal_buffers: Sequence[str],
    internal_buffer_shapes: Mapping[str, tuple[int, ...]],
    internal_buffer_policy: str,
    loop_policy: str,
    external_buffers: Sequence[str],
    dtype_by_buffer: dict[str, str],
    buffer_extent: int,
    shape_env: PathCModelShapeEnv | None,
) -> str:
    indent = " " * 4
    shape_by_buffer = {
        name: _buffer_shape(name, buffer_extent, shape_env)
        for name in external_buffers
    }
    loop_extent = _descriptor_loop_extent(
        external_buffers,
        buffer_extent,
        shape_env,
    )
    param_lines = [
        f"{indent}{name}: T.Buffer({_shape_literal(shape_by_buffer[name])}, "
        f"\"{dtype_by_buffer[name]}\"),"
        for name in external_buffers
    ]
    if not param_lines:
        param_lines = [
            f"{indent}_dummy: T.Buffer(({buffer_extent},), \"float32\"),"
        ]
    access_by_buffer = {
        buffer_name: _internal_buffer_ref(
            buffer_name,
            internal_buffer_shapes[buffer_name],
            shape_env,
        )
        for buffer_name in internal_buffers
    }
    for buffer_name, shape in shape_by_buffer.items():
        access_by_buffer[buffer_name] = _loop_indexed_buffer_ref(
            buffer_name,
            shape,
            loop_extent,
            shape_env,
        )
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

    body: list[str] = [
        "@T.prim_func",
        f"def {entry_name}(",
        *param_lines,
        "):",
        (
            f"{indent}with T.Kernel({block_count}, threads={thread_count}) as bx:"
            if loop_policy == DESCRIPTOR_LOOP_POLICY_FLAT
            else f"{indent}with T.Kernel(1, threads=1):"
        ),
        f"{indent * 2}# internal_buffer_policy: {internal_buffer_policy}",
        f"{indent * 2}# loop_policy: {loop_policy}",
    ]
    for buffer_name in internal_buffers:
        body.append(
            f"{indent * 2}{_safe_identifier(buffer_name)} = "
            f"T.alloc_local({_shape_literal(internal_buffer_shapes[buffer_name])}, "
            f"\"{dtype_by_buffer[buffer_name]}\")"
        )
    for node, descriptor, fragment in zip(nodes, descriptors, fragments, strict=True):
        if (
            loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
            and (
                node.op_name == "residual_rmsnorm"
                or _is_row_phased_residual_rmsnorm_bwd(node, descriptor)
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
            indent=indent,
        )
    else:
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
    return "\n".join(body) + "\n"


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
    shape = (
        (shape_env.hidden_size,)
        if (
            internal_buffer_policy
            == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
            and shape_env is not None
        )
        else (1,)
    )
    return {buffer_name: shape for buffer_name in internal_buffers}


def _internal_buffer_ref(
    buffer_name: str,
    shape: Sequence[int],
    shape_env: PathCModelShapeEnv | None,
) -> str:
    name = _safe_identifier(buffer_name)
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
    indent: str,
) -> None:
    if shape_env is None:
        raise ValueError("row_phased_hidden loop policy requires a model shape_env")
    hidden_size = int(shape_env.hidden_size)
    sequence_length = int(shape_env.sequence_length)
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
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_local((1,), \"float32\")"
        )
    for node, _descriptor, _fragment in bwd_items:
        if not _is_row_phased_residual_rmsnorm_bwd(node, _descriptor):
            continue
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_sum_sq')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_inv_rms')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_dot')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_norm_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
        body.append(
            f"{indent * 2}{_scratch_name(node, 'row_total_grad')} = "
            "T.alloc_local((1,), \"float32\")"
        )
    body.append(f"{indent * 2}for row in T.serial(0, {sequence_length}):")
    for node, descriptor, fragment in fwd_items:
        if node.op_name == "residual_rmsnorm":
            _append_row_phased_residual_rmsnorm_body(
                body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                indent=indent,
            )
            continue
        body.append(
            f"{indent * 3}for i in T.serial(row * {hidden_size}, "
            f"(row + 1) * {hidden_size}):"
        )
        _append_descriptor_node_comments(
            body,
            node=node,
            descriptor=descriptor,
            indent=indent * 4,
        )
        for statement in fragment.statements:
            body.append(f"{indent * 4}{statement}")
    if not bwd_items:
        return
    row_phased_bwd_items = tuple(
        (node, descriptor)
        for node, descriptor, _fragment in bwd_items
        if _is_row_phased_residual_rmsnorm_bwd(node, descriptor)
    )
    if not row_phased_bwd_items:
        body.append(
            f"{indent * 2}# backward_policy: flat_after_row_phased_forward"
        )
        body.append(
            f"{indent * 2}for i in T.serial(0, {sequence_length * hidden_size}):"
        )
        for node, descriptor, fragment in bwd_items:
            _append_descriptor_node_comments(
                body,
                node=node,
                descriptor=descriptor,
                indent=indent * 3,
            )
            for statement in fragment.statements:
                body.append(f"{indent * 3}{statement}")
        return
    for node, _descriptor, _fragment in bwd_items:
        if _is_row_phased_residual_rmsnorm_bwd(node, _descriptor):
            _append_row_phased_residual_rmsnorm_bwd_init(
                body,
                node=node,
                hidden_size=hidden_size,
                indent=indent,
            )
    body.append(f"{indent * 2}# backward_policy: row_phased_hidden_recompute")
    body.append(f"{indent * 2}for row in T.serial(0, {sequence_length}):")
    for node, descriptor, fragment in bwd_items:
        if _is_row_phased_residual_rmsnorm_bwd(node, descriptor):
            _append_row_phased_residual_rmsnorm_bwd_body(
                body,
                node=node,
                descriptor=descriptor,
                dtype_by_buffer=dtype_by_buffer,
                access_by_buffer=access_by_buffer,
                hidden_size=hidden_size,
                indent=indent,
            )
            continue
        body.append(
            f"{indent * 3}for i in T.serial(row * {hidden_size}, "
            f"(row + 1) * {hidden_size}):"
        )
        _append_descriptor_node_comments(
            body,
            node=node,
            descriptor=descriptor,
            indent=indent * 4,
        )
        for statement in fragment.statements:
            body.append(f"{indent * 4}{statement}")


def _is_row_phased_residual_rmsnorm_bwd(
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
) -> bool:
    return (
        node.op_name == "residual_rmsnorm_bwd"
        and descriptor.production_fragment_status == "production_region_inlined"
    )


def _append_row_phased_residual_rmsnorm_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    indent: str,
) -> None:
    sum_sq = _scratch_name(node, "row_sum_sq")
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
    body.append(f"{indent * 3}{sum_sq}[0] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size}, "
        f"(row + 1) * {hidden_size}):"
    )
    body.append(
        f"{indent * 4}{sum_sq}[0] = {sum_sq}[0] + "
        f"({residual_expr} * {residual_expr})"
    )
    body.append(
        f"{indent * 3}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size}, "
        f"(row + 1) * {hidden_size}):"
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


def _append_row_phased_residual_rmsnorm_bwd_init(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    hidden_size: int,
    indent: str,
) -> None:
    if len(node.outputs) < 3:
        return
    weight_grad = _safe_identifier(node.outputs[2])
    body.append(f"{indent * 2}for h in T.serial(0, {hidden_size}):")
    body.append(f"{indent * 3}{weight_grad}[h] = 0.0")


def _append_row_phased_residual_rmsnorm_bwd_body(
    body: list[str],
    *,
    node: _ScheduleNodeView,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
    hidden_size: int,
    indent: str,
) -> None:
    sum_sq = _scratch_name(node, "row_sum_sq")
    inv_rms = _scratch_name(node, "row_inv_rms")
    dot = _scratch_name(node, "row_dot")
    norm_grad_scratch = _scratch_name(node, "row_norm_grad")
    total_grad_scratch = _scratch_name(node, "row_total_grad")
    _append_descriptor_node_comments(
        body,
        node=node,
        descriptor=descriptor,
        indent=indent * 3,
    )
    hidden_after_grad = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, "i")
    norm_grad = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, "i")
    hidden = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, "i")
    delta = _node_input_expr(node, 3, dtype_by_buffer, access_by_buffer, "i")
    weight = _node_input_expr(node, 4, dtype_by_buffer, access_by_buffer, "i")
    residual_expr = f"({hidden} + {delta})"
    body.append(f"{indent * 3}{sum_sq}[0] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size}, "
        f"(row + 1) * {hidden_size}):"
    )
    body.append(
        f"{indent * 4}{sum_sq}[0] = {sum_sq}[0] + "
        f"({residual_expr} * {residual_expr})"
    )
    body.append(
        f"{indent * 3}{inv_rms}[0] = T.rsqrt(({sum_sq}[0] / "
        f"{float(hidden_size):.1f}) + 0.00001)"
    )
    body.append(f"{indent * 3}{dot}[0] = 0.0")
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size}, "
        f"(row + 1) * {hidden_size}):"
    )
    body.append(
        f"{indent * 4}{dot}[0] = {dot}[0] + "
        f"({norm_grad} * {weight} * {residual_expr})"
    )
    body.append(
        f"{indent * 3}for i in T.serial(row * {hidden_size}, "
        f"(row + 1) * {hidden_size}):"
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


def _descriptor_node_source(
    *,
    node: _ScheduleNodeView,
    node_index: int,
    descriptor: PathCBrickScheduleDescriptor,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
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
    if flat_extent == shape_env.hidden_size or canonical_name in {
        "attention_q_proj_bias",
        "attention_out_proj_bias",
        "mamba3_residual_to_m2rnn_norm_weight",
        "m2rnn_residual_to_attention_norm_weight",
        "residual_norm_weight",
    }:
        return f"{name}[i % {shape_env.hidden_size}]"
    if canonical_name in {"q_scale", "kv_scale"}:
        return f"{name}[i // {shape_env.attention_head_dim}]"
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
            f"({q_prepared}[0] * {q_prepared}[0]) + 1.0"
        )
        assigned.add(q_scale_output)
    if kv_scale_output is not None:
        inner.append(
            f"{_buffer_ref(kv_scale_output, access_by_buffer, index)} = "
            f"({kv_prepared}[0] * {kv_prepared}[0]) + 1.0"
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
            f"{q_prepared}[0] / {denominator}"
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
            f"{kv_prepared}[0] / {denominator}"
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
    index = "i"
    q_fp8 = _optional_buffer_expr("q_fp8", dtype_by_buffer, access_by_buffer, index=index)
    q_scale = _optional_buffer_expr(
        "q_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
    )
    kv_fp8 = _optional_buffer_expr("kv_fp8", dtype_by_buffer, access_by_buffer, index=index)
    kv_scale = _optional_buffer_expr(
        "kv_scale",
        dtype_by_buffer,
        access_by_buffer,
        "1.0",
        index,
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
    indices = _optional_buffer_expr("indices", dtype_by_buffer, access_by_buffer, index=index)
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
    inner = [f"{sink_enabled}[0] = T.cast({has_sinks} != 0, \"float32\")"]
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
            f"({kv_fp8} * {kv_scale})) * {sm_scale} + "
            f"({sinks} * {sink_enabled}[0]) + {indices}) * "
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
        allocations=(f"{sink_enabled} = T.alloc_local((1,), \"float32\")",),
        statements=tuple(inner),
    )


def _emit_attention_qkv_projection_bwd_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    q_grad = _scratch_name(node, "attention_q_grad")
    kv_grad = _scratch_name(node, "attention_kv_grad")
    rope_grad = _scratch_name(node, "attention_rope_grad")
    index = "i"
    q_fp8_grad = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    q_scale_grad = _node_input_expr(node, 1, dtype_by_buffer, access_by_buffer, index)
    kv_fp8_grad = _node_input_expr(node, 2, dtype_by_buffer, access_by_buffer, index)
    kv_scale_grad = _node_input_expr(node, 3, dtype_by_buffer, access_by_buffer, index)
    inner = [
        f"{q_grad}[0] = {q_fp8_grad} + {q_scale_grad}",
        f"{kv_grad}[0] = {kv_fp8_grad} + {kv_scale_grad}",
        f"{rope_grad}[0] = {q_grad}[0] + {kv_grad}[0]",
    ]
    for output_name in node.outputs:
        canonical_name = _canonical_buffer_name(output_name)
        output_ref = _buffer_ref(output_name, access_by_buffer, index)
        if canonical_name == "attention_hidden":
            inner.append(f"{output_ref} = {q_grad}[0] + {kv_grad}[0]")
        elif canonical_name in {
            "attention_q_proj_weight",
            "attention_q_proj_bias",
        }:
            inner.append(f"{output_ref} = {q_grad}[0]")
        elif canonical_name in {
            "attention_sparse_kv_proj_weight",
            "attention_sparse_kv_proj_bias",
        }:
            inner.append(f"{output_ref} = {kv_grad}[0]")
        elif canonical_name == "attention_rope_inv_freq":
            inner.append(f"{output_ref} = {rope_grad}[0]")
        else:
            inner.append(f"{output_ref} = {q_grad}[0] + {kv_grad}[0]")
    return _ScheduleNodeFragment(
        allocations=(
            f"{q_grad} = T.alloc_local((1,), \"float32\")",
            f"{kv_grad} = T.alloc_local((1,), \"float32\")",
            f"{rope_grad} = T.alloc_local((1,), \"float32\")",
        ),
        statements=tuple(inner),
    )


def _emit_m2rnn_bwd_source(
    node: _ScheduleNodeView,
    dtype_by_buffer: dict[str, str],
    access_by_buffer: dict[str, str],
) -> _ScheduleNodeFragment:
    project_grad = _scratch_name(node, "m2rnn_project_grad")
    conv_grad = _scratch_name(node, "m2rnn_conv_grad")
    recurrent_grad = _scratch_name(node, "m2rnn_recurrent_grad")
    post_grad = _scratch_name(node, "m2rnn_post_grad")
    index = "i"
    delta_grad = _node_input_expr(node, 0, dtype_by_buffer, access_by_buffer, index)
    inner = [
        f"{project_grad}[0] = {delta_grad}",
        f"{conv_grad}[0] = {project_grad}[0]",
        f"{recurrent_grad}[0] = {conv_grad}[0]",
        f"{post_grad}[0] = {recurrent_grad}[0]",
    ]
    for output_name in node.outputs:
        canonical_name = _canonical_buffer_name(output_name)
        output_ref = _buffer_ref(output_name, access_by_buffer, index)
        if canonical_name in {
            "m2rnn_hidden",
            "m2rnn_in_proj_weight",
        }:
            inner.append(f"{output_ref} = {project_grad}[0] + {recurrent_grad}[0]")
        elif canonical_name in {
            "m2rnn_conv_weight",
            "m2rnn_conv_bias",
        }:
            inner.append(f"{output_ref} = {conv_grad}[0]")
        elif canonical_name in {
            "m2rnn_state_weight",
            "m2rnn_A_log",
            "m2rnn_dt_bias",
            "m2rnn_h0",
        }:
            inner.append(f"{output_ref} = {recurrent_grad}[0]")
        elif canonical_name in {
            "m2rnn_D",
            "m2rnn_g_norm_weight",
            "m2rnn_out_proj_weight",
        }:
            inner.append(f"{output_ref} = {post_grad}[0]")
        else:
            inner.append(f"{output_ref} = {post_grad}[0]")
    return _ScheduleNodeFragment(
        allocations=(
            f"{project_grad} = T.alloc_local((1,), \"float32\")",
            f"{conv_grad} = T.alloc_local((1,), \"float32\")",
            f"{recurrent_grad} = T.alloc_local((1,), \"float32\")",
            f"{post_grad} = T.alloc_local((1,), \"float32\")",
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
    if dtype == "int32":
        return f'T.cast({ref}, "float32")'
    return ref


def _buffer_ref(
    buffer_name: str,
    access_by_buffer: dict[str, str],
    index: str,
) -> str:
    return access_by_buffer.get(buffer_name, f"{buffer_name}[{index}]")


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
            )
        )
    if not views:
        raise ValueError("descriptor schedule generation requires at least one node")
    return tuple(views)


def _internal_buffers_for_nodes(
    nodes: Sequence[_ScheduleNodeView],
) -> tuple[str, ...]:
    input_names = {name for node in nodes for name in node.inputs}
    internal: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for output_name in node.outputs:
            if output_name not in input_names or output_name in seen:
                continue
            seen.add(output_name)
            internal.append(output_name)
    return tuple(internal)


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


def _buffer_dtype(buffer_name: str) -> str:
    if buffer_name == "indices" or buffer_name.endswith("_indices"):
        return "int32"
    if buffer_name.endswith("_has_sinks"):
        return "int32"
    return "float32"


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
            "hidden",
            "mamba_state",
            "scan_state",
            "mamba3_delta",
            "m2rnn_hidden",
            "m2rnn_delta",
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
    if name == "mamba_state" or name == "scan_state" or name == "mamba3_h0":
        return (
            shape_env.mamba_num_heads
            * shape_env.mamba_head_dim
            * shape_env.mamba_state_dim,
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
    }:
        return (hidden,)
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
    if name == "m2rnn_h0":
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
) -> PathCFusionRegion:
    return build_mamba3_fp8_train_acceptance_fixture_region(
        include_backward=include_backward,
    )


def _mamba3_fp8_train_acceptance_profile() -> PathCFusionScheduleAcceptanceProfile:
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
        buffer_extent=MAMBA3_FP8_TRAIN_BUFFER_EXTENT,
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
    implementation_kind = _effective_implementation_kind(
        acceptance_profile,
        descriptors,
    )
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
    return PathCFusionScheduleTarget(
        schedule_id=acceptance_profile.schedule_id
        if acceptance_profile is not None
        else f"path_c_descriptor_chain_{digest}",
        schedule_name=schedule_name,
        op_signature=signature,
        schedule_status=acceptance_profile.schedule_status
        if acceptance_profile is not None
        else "descriptor_codegen_scaffold",
        implementation_kind=implementation_kind,
        missing_reason=(
            acceptance_profile.missing_reason
            if acceptance_profile is not None
            else (
                "descriptor-generated Path C schedule was selected from the "
                "region graph, but it is not a named production acceptance target"
            )
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
        ),
        required_real_abi_inputs=required_real_abi_inputs,
        brick_descriptors=descriptors,
        buffer_extent=buffer_extent,
        internal_buffer_policy=internal_buffer_policy,
        loop_policy=loop_policy,
    )


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


def plan_path_c_fusion_schedules_for_model(
    model: Any,
    *,
    region_prefix: str | None = None,
    include_backward: bool = True,
    min_route_bricks: int = 2,
    registry: PathCFusionScheduleRegistry | None = None,
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
    )
    return tuple(
        plan_path_c_fusion_schedule_for_region(
            region,
            include_backward=include_backward,
            registry=registry,
        )
        for region in regions
    )


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
) -> Mamba3Fp8TrainFusionSchedulePlan:
    """Build and plan the named Mamba3 FP8 train-block acceptance schedule."""

    fwd_region = _mamba3_fp8_train_acceptance_region(include_backward=False)
    acceptance_registry = PathCFusionScheduleRegistry(
        acceptance_profiles=(_mamba3_fp8_train_acceptance_profile(),),
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
    return Mamba3Fp8TrainFusionSchedulePlan(
        region=optimized.region,
        plan=optimized.plan,
        schedule_spec=mamba3_fp8_train_fusion_schedule_spec(
            optimized.region,
            contract=optimized.plan.schedule_contract,
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
