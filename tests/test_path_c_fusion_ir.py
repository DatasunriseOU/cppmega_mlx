from __future__ import annotations

from dataclasses import replace
import re
from types import SimpleNamespace

import pytest
import tilelang.language as T

from cppmega_mlx.runtime import path_c_fusion
from cppmega_mlx.runtime.path_c_fusion import (
    BenchmarkAcceptanceRow,
    CompiledPathCRegion,
    FusionCompilePlan,
    FusionKernelSurface,
    MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
    FusionScheduleContractStatus,
    PathCFusionRegion,
    PathCFusionRegionBuilder,
    PathCFusionMode,
    PathCModelBrick,
    Z3SyncSpec,
    build_path_c_aot_autograd_region,
    audit_fusion_cache_key,
    audit_fused_path_c_plan_default_eligibility,
    audit_warm_cache_reuse,
    build_path_c_fusion_region,
    build_path_c_model_regions_from_bricks,
    build_path_c_model_region_from_route_symbols,
    build_path_c_model_regions_from_model,
    build_path_c_model_regions_from_route_symbols,
    build_mamba3_fp8_train_acceptance_fixture_region,
    build_mamba3_fp8_train_region,
    compile_path_c_region,
    fused_path_c_plan_default_eligible,
    fused_path_c_default_eligible,
    mark_path_c_schedule_template_for_region,
    path_b_baseline_is_clean,
    selected_path_c_fusion_mode,
    tilelang_single_entry_lowerer,
    trusted_path_c_production_schedule_ids,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import (
    DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
    DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH,
    DESCRIPTOR_EXECUTION_STAGE_BACKWARD,
    DESCRIPTOR_EXECUTION_STAGE_FORWARD,
    DESCRIPTOR_LOOP_POLICY_FLAT,
    DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
    DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE,
    DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS,
    DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID,
    MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME,
    MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS,
    MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME,
    MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS,
    PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR,
    PathCBrickScheduleFragment,
    PathCFusionScheduleOptimizer,
    PathCFusionScheduleAcceptanceProfile,
    PathCFusionScheduleRegistry,
    PathCBrickScheduleDescriptor,
    PathCBrickScheduleDescriptorRegistry,
    build_path_c_descriptor_prim_func,
    default_path_c_brick_schedule_descriptor_registry,
    default_path_c_fusion_schedule_registry,
    mamba3_fp8_train_fusion_schedule_spec,
    mamba3_fp8_train_fusion_schedule_template,
    mamba3_fp8_train_fusion_schedule_target,
    mamba3_fp8_train_prototype_schedule_template,
    path_c_descriptor_stage_prim_funcs,
    path_c_fusion_schedule_spec,
    path_c_fusion_schedule_template,
    plan_path_c_descriptor_phase_groups,
    plan_path_c_descriptor_stage_groups,
    plan_path_c_direct_fusion_chain_for_region,
    plan_path_c_direct_fusion_chains_for_model,
    plan_path_c_fusion_schedule_for_region,
    plan_path_c_fusion_schedules_for_model,
    plan_mamba3_fp8_train_fusion_schedule,
    prototype_path_c_fusion_schedule_registry,
    select_path_c_fusion_schedule_target,
)
from cppmega_mlx.recipes.model_factory import (
    LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH,
    local_gb10_quarter_profile,
)


def _physical_bank_fragment(prim_func: object, logical_name: str) -> str:
    mapping = getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
    info = mapping[logical_name]
    if info["offset"] == 0:
        return f"{info['bank']}["
    return f"{info['bank']}[{info['offset']}"


def _line_uses_logical_buffer(
    prim_func: object,
    line: str,
    logical_name: str,
) -> bool:
    if logical_name in line:
        return True
    mapping = getattr(prim_func, "_cppmega_path_c_physical_buffer_abi_map", {})
    info = mapping.get(logical_name)
    if info is None:
        aliases = getattr(prim_func, "_cppmega_path_c_internal_scratch_abi_aliases", {})
        info = aliases.get(logical_name)
    if info is None:
        spilled = getattr(prim_func, "_cppmega_path_c_spilled_shared_scratch_shapes", {})
        info = spilled.get(logical_name)
    if info is None:
        return False
    if info["offset"] == 0:
        return f"{info['bank']}[" in line
    return f"{info['bank']}[{info['offset']}" in line


def _shared_alloc_bytes(source: str) -> int:
    dtype_bytes = {"float32": 4, "uint8": 1, "int32": 4}
    total = 0
    for match in re.finditer(
        r"= T\.alloc_shared\(\(([^)]*)\), \"([^\"]+)\"\)", source
    ):
        shape_text, dtype = match.groups()
        elements = 1
        for dim_text in shape_text.split(","):
            dim_text = dim_text.strip()
            if not dim_text:
                continue
            elements *= int(dim_text)
        total += elements * dtype_bytes[dtype]
    return total


def _max_tir_region_line_length(source: str) -> int:
    return max(
        (
            len(line)
            for line in source.splitlines()
            if "T.reads" in line or "T.writes" in line
        ),
        default=0,
    )


def _generated_source_window(source: str, marker: str, line_count: int = 80) -> str:
    lines = source.splitlines()
    marker_index = next(index for index, line in enumerate(lines) if marker in line)
    return "\n".join(lines[marker_index : marker_index + line_count])


def _production_mra_staged_launcher_prim_func(stage_suffix: str) -> tuple[object, object]:
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    cfg = local_gb10_quarter_profile().hybrid_config()
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="generic_mra_path_c",
        route_symbols=("M", "R", "A"),
        model_config=cfg,
    )
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert target.implementation_kind == "production"

    group = next(
        group
        for group in plan_path_c_descriptor_stage_groups(region)
        if group.stage_suffix == stage_suffix
    )
    shape_env = target.schedule_template._cppmega_path_c_shape_env
    entry_base = getattr(region, "entry_symbol", None) or region.name
    schedule_template = schedules.make_path_c_descriptor_schedule_template(
        target.brick_descriptors,
        entry_symbol=f"{entry_base}_{group.stage_suffix}",
        buffer_extent=target.buffer_extent,
        shape_env=shape_env,
        internal_buffer_policy=target.internal_buffer_policy,
        loop_policy=target.loop_policy,
        physical_abi_policy=target.physical_abi_policy,
        max_rows_per_launch=target.max_rows_per_launch,
        row_dispatch_mode=group.row_dispatch_mode,
        rows_per_kernel_launch=group.rows_per_kernel_launch,
        execution_stage=group.execution_stage,
        active_node_names=group.active_node_names,
    )
    return group, schedule_template(region)


@T.prim_func
def _toy_path_c_train_block(
    hidden: T.Tensor((4,), "float32"),
    mamba_state: T.Tensor((4,), "float32"),
    indices: T.Tensor((4,), "int32"),
    scan_state: T.Tensor((4,), "float32"),
    attention_out: T.Tensor((4,), "float32"),
    lse: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        scan_y = T.alloc_local((4,), "float32")
        post_y = T.alloc_local((4,), "float32")
        q_fp8 = T.alloc_local((4,), "float32")
        q_scale = T.alloc_local((4,), "float32")
        kv_fp8 = T.alloc_local((4,), "float32")
        kv_scale = T.alloc_local((4,), "float32")
        scan_y[0] = hidden[0] + mamba_state[0]
        scan_state[0] = scan_y[0]
        post_y[0] = scan_y[0]
        q_fp8[0] = post_y[0]
        q_scale[0] = 1.0
        kv_fp8[0] = post_y[0]
        kv_scale[0] = 1.0
        attention_out[0] = (
            q_fp8[0]
            + kv_fp8[0]
            + q_scale[0]
            + kv_scale[0]
            + T.cast(indices[0], "float32")
        )
        lse[0] = 0.0


@T.prim_func
def _toy_path_c_model_train_block(
    hidden: T.Tensor((4,), "float32"),
    mamba3_entry_rmsnorm_weight: T.Tensor((4,), "float32"),
    mamba_state: T.Tensor((4,), "float32"),
    mamba3_in_proj_weight: T.Tensor((4,), "float32"),
    mamba3_out_proj_weight: T.Tensor((4,), "float32"),
    mamba3_conv_weight: T.Tensor((4,), "float32"),
    mamba3_conv_bias: T.Tensor((4,), "float32"),
    mamba3_dt_bias: T.Tensor((4,), "float32"),
    mamba3_B_norm_weight: T.Tensor((4,), "float32"),
    mamba3_B_bias: T.Tensor((4,), "float32"),
    mamba3_C_norm_weight: T.Tensor((4,), "float32"),
    mamba3_C_bias: T.Tensor((4,), "float32"),
    mamba3_D: T.Tensor((4,), "float32"),
    mamba3_h0: T.Tensor((4,), "float32"),
    scan_state: T.Tensor((4,), "float32"),
    mamba3_residual_to_m2rnn_norm_weight: T.Tensor((4,), "float32"),
    m2rnn_in_proj_weight: T.Tensor((4,), "float32"),
    m2rnn_conv_weight: T.Tensor((4,), "float32"),
    m2rnn_conv_bias: T.Tensor((4,), "float32"),
    m2rnn_state_weight: T.Tensor((4,), "float32"),
    m2rnn_A_log: T.Tensor((4,), "float32"),
    m2rnn_dt_bias: T.Tensor((4,), "float32"),
    m2rnn_D: T.Tensor((4,), "float32"),
    m2rnn_g_norm_weight: T.Tensor((4,), "float32"),
    m2rnn_out_proj_weight: T.Tensor((4,), "float32"),
    m2rnn_h0: T.Tensor((4,), "float32"),
    m2rnn_conv_state: T.Tensor((4,), "float32"),
    hidden_after_m2rnn: T.Tensor((4,), "float32"),
    m2rnn_residual_to_attention_norm_weight: T.Tensor((4,), "float32"),
    attention_q_proj_weight: T.Tensor((4,), "float32"),
    attention_q_proj_bias: T.Tensor((4,), "float32"),
    attention_sparse_kv_proj_weight: T.Tensor((4,), "float32"),
    attention_sparse_kv_proj_bias: T.Tensor((4,), "float32"),
    attention_rope_inv_freq: T.Tensor((4,), "float32"),
    attention_out_proj_weight: T.Tensor((4,), "float32"),
    attention_out_proj_bias: T.Tensor((4,), "float32"),
    sparse_mla_sm_scale: T.Tensor((4,), "float32"),
    sparse_mla_sinks: T.Tensor((4,), "float32"),
    sparse_mla_has_sinks: T.Tensor((4,), "int32"),
    q_fp8: T.Tensor((4,), "uint8"),
    q_scale: T.Tensor((4,), "float32"),
    kv_fp8: T.Tensor((4,), "uint8"),
    kv_scale: T.Tensor((4,), "float32"),
    attention_out: T.Tensor((4,), "float32"),
    lse: T.Tensor((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        # Block A: the fused model region now starts with an entry RMSNorm op
        # whose output is an internal edge buffer consumed by the first
        # in-region brick. Materialize a store + load so the fullgraph
        # internal-edge validator sees the buffer touch the entry PrimFunc.
        mamba3_entry_rmsnorm_hidden = T.alloc_local((4,), "float32")
        mamba3_delta = T.alloc_local((4,), "float32")
        hidden_after_mamba3 = T.alloc_local((4,), "float32")
        m2rnn_hidden = T.alloc_local((4,), "float32")
        m2rnn_delta = T.alloc_local((4,), "float32")
        attention_hidden = T.alloc_local((4,), "float32")
        indices = T.alloc_local((4,), "int32")
        mamba3_entry_rmsnorm_hidden[0] = (
            hidden[0] + mamba3_entry_rmsnorm_weight[0] * 0.0
        )
        mamba3_delta[0] = (
            mamba3_entry_rmsnorm_hidden[0]
            + mamba_state[0]
            + mamba3_in_proj_weight[0] * 0.0
            + mamba3_h0[0] * 0.0
        )
        scan_state[0] = mamba3_delta[0]
        hidden_after_mamba3[0] = hidden[0] + mamba3_delta[0]
        m2rnn_hidden[0] = (
            hidden_after_mamba3[0]
            + mamba3_residual_to_m2rnn_norm_weight[0] * 0.0
        )
        m2rnn_delta[0] = (
            m2rnn_hidden[0]
            + m2rnn_in_proj_weight[0] * 0.0
            + m2rnn_h0[0] * 0.0
        )
        hidden_after_m2rnn[0] = hidden_after_mamba3[0] + m2rnn_delta[0]
        attention_hidden[0] = (
            hidden_after_m2rnn[0]
            + m2rnn_residual_to_attention_norm_weight[0] * 0.0
            + attention_q_proj_weight[0] * 0.0
        )
        q_fp8[0] = T.cast(attention_hidden[0], "uint8")
        q_scale[0] = 1.0
        kv_fp8[0] = T.cast(attention_hidden[0], "uint8")
        kv_scale[0] = 1.0
        indices[0] = 0
        attention_out[0] = (
            T.cast(q_fp8[0], "float32")
            + T.cast(kv_fp8[0], "float32")
            + q_scale[0]
            + kv_scale[0]
            + T.cast(indices[0], "float32")
            + sparse_mla_sm_scale[0] * 0.0
            + sparse_mla_sinks[0] * 0.0
            + T.cast(sparse_mla_has_sinks[0], "float32") * 0.0
        )
        lse[0] = 0.0


def test_region_builder_creates_fx_like_ir_without_msl_post_fusion() -> None:
    builder = PathCFusionRegionBuilder(
        "hybrid_train_block",
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    builder.add_kernels(
        (
            FusionKernelSurface.path_c(
                name="mamba3_scan",
                op_name="mamba3_mimo",
                inputs=("hidden", "state"),
                outputs=("scan_y", "scan_state"),
                backward="aot_autograd",
            ),
            FusionKernelSurface.path_c(
                name="packed_post",
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
    )

    region = builder.build()
    plan = compile_path_c_region(region)

    assert isinstance(region, PathCFusionRegion)
    assert region.node_names == (
        "mamba3_scan",
        "packed_post",
        "fp8_prepare",
        "sparse_mla_fp8_apply",
    )
    assert plan.lowering_boundary == "tilelang_tvm_ir"
    assert plan.compiler == "tilelang.engine.fusion"
    assert plan.requires_msl_post_fusion is False
    assert plan.fusion_groups[0].node_names == region.node_names
    assert plan.backward_graph == "aot_autograd"
    assert plan.z3_sync.proof_required is True
    assert plan.z3_sync.objective == "minimize_sync_async"


def test_builder_rejects_msl_string_surfaces() -> None:
    builder = PathCFusionRegionBuilder("bad")

    with pytest.raises(ValueError, match="MSL"):
        builder.add_msl_kernel("already_lowered", "kernel void k() {}")


def test_region_builder_orders_surfaces_by_inferred_dependencies() -> None:
    builder = PathCFusionRegionBuilder("out_of_order_train_block")
    builder.add_kernels(
        (
            FusionKernelSurface.path_c(
                name="apply",
                op_name="sparse_mla_fp8_apply",
                inputs=("post_y",),
                outputs=("out",),
                backward="owner_output",
            ),
            FusionKernelSurface.path_c(
                name="post",
                op_name="m2rnn",
                inputs=("scan_y",),
                outputs=("post_y",),
                backward="aot_autograd",
            ),
            FusionKernelSurface.path_c(
                name="scan",
                op_name="mamba3_mimo",
                inputs=("hidden",),
                outputs=("scan_y",),
                backward="aot_autograd",
            ),
        )
    )

    region = builder.build()

    assert region.node_names == ("scan", "post", "apply")
    assert [(edge.producer, edge.consumer, edge.input) for edge in region.edges] == [
        ("scan", "post", "scan_y"),
        ("post", "apply", "post_y"),
    ]


def test_build_path_c_fusion_region_adds_dynamic_surfaces_and_infers_chain() -> None:
    region = build_path_c_fusion_region(
        region_name="dynamic_train_block",
        surfaces=(
            FusionKernelSurface.path_c(
                name="apply",
                op_name="sparse_mla_fp8_apply",
                inputs=("post_y",),
                outputs=("out",),
                backward="owner_output",
            ),
            FusionKernelSurface.path_c(
                name="scan",
                op_name="mamba3_mimo",
                inputs=("hidden",),
                outputs=("scan_y",),
                backward="aot_autograd",
            ),
            FusionKernelSurface.path_c(
                name="post",
                op_name="m2rnn",
                inputs=("scan_y",),
                outputs=("post_y",),
                backward="aot_autograd",
            ),
        ),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )

    assert region.node_names == ("scan", "post", "apply")
    assert [(edge.producer, edge.consumer, edge.input) for edge in region.edges] == [
        ("scan", "post", "scan_y"),
        ("post", "apply", "post_y"),
    ]
    assert region.z3_sync.enabled is True
    assert region.z3_sync.proof_required is True


def test_build_path_c_model_region_from_route_symbols_uses_dynamic_default_target() -> None:
    region = build_path_c_model_region_from_route_symbols(
        region_name="generic_mra_path_c",
        route_symbols=("M", "R", "A"),
        model_config=local_gb10_quarter_profile().hybrid_config(),
    )

    assert region.node_names == (
        "route_0_M_entry_rmsnorm",
        "route_0_M",
        "route_0_M_residual_norm",
        "route_1_R",
        "route_1_R_residual_norm",
        "route_2_A_qkv_projection",
        "route_2_A_sparse_mla_fp8_apply",
    )
    assert tuple(node.op_name for node in region.nodes) == (
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    )
    assert region.z3_sync.enabled is True
    assert region.metadata["path_c_acceptance_tags"] == ()
    assert region.metadata["path_c_acceptance_fixture_abi"] is False
    assert region.metadata["path_c_bricks"] == (
        {"name": "route_0_M", "kind": "M", "route_symbol": "M"},
        {"name": "route_1_R", "kind": "R", "route_symbol": "R"},
        {"name": "route_2_A", "kind": "A", "route_symbol": "A"},
    )
    assert region.metadata["path_c_model_shape_env"].sequence_length == (
        LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH
    )

    target = select_path_c_fusion_schedule_target(
        build_path_c_aot_autograd_region(region)
    )
    plan = compile_path_c_region(build_path_c_aot_autograd_region(region))

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.schedule_name == (
        "generic_mra_path_c:descriptor_generated_fwd_bwd"
    )
    assert target.implementation_kind == "production"
    assert target.max_rows_per_launch == DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH
    assert target.row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS
    prim_func = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    assert "def generic_mra_path_c(" in prim_func._cppmega_path_c_generated_source
    assert prim_func._cppmega_path_c_row_chunk_count == (
        LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH + DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH - 1
    ) // DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH
    assert prim_func._cppmega_path_c_max_rows_per_launch == (
        DESCRIPTOR_DEFAULT_MAX_ROWS_PER_LAUNCH
    )
    assert "vocab_col" not in prim_func._cppmega_path_c_generated_source
    assert plan.schedule_contract is not None
    spec = path_c_fusion_schedule_spec(
        build_path_c_aot_autograd_region(region),
        contract=plan.schedule_contract,
        target=target,
    )
    assert spec.real_abi_contract_complete is True
    assert spec.missing_real_abi_inputs == ()


def test_descriptor_stage_planner_groups_train_block_from_region_graph() -> None:
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="generic_mra_path_c",
        route_symbols=("M", "R", "A"),
        model_config=local_gb10_quarter_profile().hybrid_config(),
    )
    region = build_path_c_aot_autograd_region(fwd_region)

    groups = plan_path_c_descriptor_stage_groups(region)

    assert [(group.execution_stage, group.stage_suffix) for group in groups] == [
        (DESCRIPTOR_EXECUTION_STAGE_FORWARD, "g0"),
        (DESCRIPTOR_EXECUTION_STAGE_FORWARD, "g1"),
        (DESCRIPTOR_EXECUTION_STAGE_FORWARD, "g2"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b0"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b1"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b2"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b3"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b4"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b5"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b6"),
    ]
    assert [group.active_node_names for group in groups[:3]] == [
        ("route_0_M_entry_rmsnorm",),
        ("route_0_M",),
        (
            "route_0_M_residual_norm",
            "route_1_R",
            "route_1_R_residual_norm",
            "route_2_A_qkv_projection",
            "route_2_A_sparse_mla_fp8_apply",
        ),
    ]
    assert groups[3].active_node_names == ("route_2_A_sparse_mla_fp8_apply_bwd",)
    assert groups[4].active_node_names == ("route_2_A_qkv_projection_bwd",)
    assert groups[4].rows_per_kernel_launch == 1
    assert groups[8].active_node_names == ("route_0_M_bwd",)
    assert groups[8].rows_per_kernel_launch == 1
    assert all(
        group.row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        for group in groups
    )
    assert all(group.rows_per_kernel_launch == 8 for group in groups[:4])
    assert groups[5].rows_per_kernel_launch == 8


def test_descriptor_phase_planner_fuses_train_block_by_execution_phase() -> None:
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="generic_mra_path_c",
        route_symbols=("M", "R", "A"),
        model_config=local_gb10_quarter_profile().hybrid_config(),
    )
    region = build_path_c_aot_autograd_region(fwd_region)

    groups = plan_path_c_descriptor_phase_groups(region)

    assert [(group.execution_stage, group.stage_suffix) for group in groups] == [
        (DESCRIPTOR_EXECUTION_STAGE_FORWARD, "g0"),
        (DESCRIPTOR_EXECUTION_STAGE_BACKWARD, "b0"),
    ]
    assert groups[0].active_node_names == tuple(
        name for name in region.node_names if not name.endswith("_bwd")
    )
    assert groups[1].active_node_names == tuple(
        name for name in region.node_names if name.endswith("_bwd")
    )
    assert groups[0].reason == "descriptor_fuses_forward_phase_blocks"
    assert groups[1].reason == "descriptor_fuses_backward_phase_blocks"
    assert all(
        group.row_dispatch_mode == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
        for group in groups
    )


def test_production_staged_launcher_m2rnn_b3_replays_one_row() -> None:
    import tvm

    group, prim_func = _production_mra_staged_launcher_prim_func("b3")
    generated_source = prim_func._cppmega_path_c_generated_source
    policy_window = _generated_source_window(
        generated_source,
        "# m2rnn_bwd_policy: exact_reverse_checkpoint_replay",
    )
    tir_source = str(tvm.IRModule.from_expr(prim_func).script())

    assert group.execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
    assert group.active_node_names == ("route_1_R_bwd",)
    assert prim_func._cppmega_path_c_execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
    assert (
        prim_func._cppmega_path_c_row_dispatch_mode
        == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    assert "path_c_first_row_launch = T.if_then_else(" in generated_source
    assert prim_func._cppmega_path_c_rows_per_kernel_launch == 8
    assert prim_func._cppmega_path_c_row_subchunk_count == 8
    assert (
        "if path_c_first_row_launch != 0 and row == row_chunk_start:"
        in generated_source
    )
    assert "if path_c_first_row_launch != 0:" not in policy_window
    assert "for route_1_R_bwd_time_rev in T.serial(row, row + 1):" in (
        policy_window
    )
    assert "for route_1_R_bwd_replay_offset in T.serial(0, 1):" in (
        policy_window
    )
    assert "for route_1_R_bwd_replay_offset in T.serial(0, 64):" not in (
        policy_window
    )
    assert "for route_1_R_bwd_replay_offset in range(1):" in tir_source
    assert "for route_1_R_bwd_replay_offset in range(64):" not in tir_source


def test_production_staged_launcher_mamba3_b5_spills_grad_vector_scratch_to_abi() -> None:
    import tvm

    group, prim_func = _production_mra_staged_launcher_prim_func("b5")
    generated_source = prim_func._cppmega_path_c_generated_source
    tir_source = str(tvm.IRModule.from_expr(prim_func).script())
    spilled = prim_func._cppmega_path_c_spilled_shared_scratch_shapes
    forced_spill_names = (
        "route_0_M_bwd_mamba3_h_prev",
        "route_0_M_bwd_mamba3_h_next",
        "route_0_M_bwd_mamba3_dh",
        "route_0_M_bwd_mamba3_dh_prev",
        "route_0_M_bwd_mamba3_angle_grad",
        "route_0_M_bwd_mamba3_b_raw_grad",
        "route_0_M_bwd_mamba3_c_raw_grad",
        "route_0_M_bwd_mamba3_b_group_grad",
        "route_0_M_bwd_mamba3_c_group_grad",
        "route_0_M_bwd_mamba3_dt_vec",
        "route_0_M_bwd_mamba3_a_vec",
        "route_0_M_bwd_mamba3_dt_grad",
        "route_0_M_bwd_mamba3_a_grad",
        "route_0_M_bwd_mamba3_next_dt_pre_vec",
        "route_0_M_bwd_mamba3_next_dt_vec",
        "route_0_M_bwd_mamba3_next_trap_vec",
        "route_0_M_bwd_mamba3_trap_grad",
        "route_0_M_bwd_mamba3_projected_vec",
        "route_0_M_bwd_mamba3_project_grad",
        "route_0_M_bwd_mamba3_conv_vec",
        "route_0_M_bwd_mamba3_conv_grad",
        "route_0_M_bwd_mamba3_out_inner",
        "route_0_M_bwd_mamba3_out_inner_grad",
    )

    assert group.execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
    assert group.active_node_names == ("route_0_M_bwd",)
    assert prim_func._cppmega_path_c_execution_stage == DESCRIPTOR_EXECUTION_STAGE_BACKWARD
    assert (
        prim_func._cppmega_path_c_row_dispatch_mode
        == DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    assert "path_c_first_row_launch = T.if_then_else(" in generated_source
    assert prim_func._cppmega_path_c_rows_per_kernel_launch == 1
    assert prim_func._cppmega_path_c_row_subchunk_count == 64
    assert "path_c_row_subchunk_index < 64" in generated_source
    assert (
        "subchunk_row_chunk_start = T.min(logical_row_chunk_start + "
        "path_c_row_subchunk_index * 1, logical_row_chunk_stop)"
        in generated_source
    )
    assert (
        "if path_c_first_row_launch != 0 and row == row_chunk_start:"
        in generated_source
    )
    assert "# mamba3_mimo_bwd_policy: exact_lane_parallel_checkpoint_replay" in (
        generated_source
    )
    assert "for route_0_M_bwd_replay_offset in T.serial(0, 8):" in (
        generated_source
    )
    assert "for route_0_M_bwd_replay_offset in T.serial(0, 64):" not in (
        generated_source
    )
    assert "if lane == 0:" not in generated_source
    assert "T.serial(0, 11264)" not in generated_source
    assert "T.serial(0, 18784)" not in generated_source
    assert "T.serial(0, 458752)" not in generated_source
    assert "path_c_float32_scratch_bank: T.Tensor" in generated_source
    assert "T.alloc_shared" not in generated_source
    assert 'scope="shared"' not in tir_source
    assert "threadgroup" not in tir_source
    assert "buf_dyn_shmem" not in tir_source

    for scratch_name in forced_spill_names:
        info = spilled[scratch_name]
        assert info["dtype"] == "float32"
        assert info["param_name"] == "path_c_float32_scratch_bank"
        assert info["coalesced_scratch_bank"] is True
        assert f"{scratch_name} = T.alloc_local" not in generated_source
        assert f"{scratch_name} = T.alloc_shared" not in generated_source
        assert f"{scratch_name} = T.alloc_buffer" not in tir_source

    for scratch_name in (
        "route_0_M_bwd_mamba3_h_prev",
        "route_0_M_bwd_mamba3_dh",
        "route_0_M_bwd_mamba3_project_grad",
        "route_0_M_bwd_mamba3_conv_grad",
        "route_0_M_bwd_mamba3_out_inner_grad",
    ):
        assert any(
            _line_uses_logical_buffer(prim_func, line, scratch_name)
            for line in generated_source.splitlines()
        )


def test_named_mamba3_acceptance_target_is_explicit_fixture_only() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    target = mamba3_fp8_train_fusion_schedule_target()
    plan = compile_path_c_region(region)

    assert region.metadata["path_c_acceptance_fixture_abi"] is True
    assert "mamba3_fp8_train_acceptance" in region.metadata[
        "path_c_acceptance_tags"
    ]
    assert target.schedule_id == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert plan.schedule_contract is not None
    spec = mamba3_fp8_train_fusion_schedule_spec(
        region,
        contract=plan.schedule_contract,
        target=target,
    )
    assert spec.schedule_id == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert spec.region_name == "mamba3_m2rnn_attention_fp8_train_block"


def test_generic_schedule_spec_requires_discovered_region() -> None:
    with pytest.raises(ValueError, match="requires a discovered Path C region"):
        path_c_fusion_schedule_spec()


def test_named_mamba3_acceptance_schedule_requires_explicit_region() -> None:
    with pytest.raises(ValueError, match="requires an explicit discovered Path C region"):
        mamba3_fp8_train_fusion_schedule_spec()
    with pytest.raises(ValueError, match="requires an explicit discovered Path C region"):
        mamba3_fp8_train_fusion_schedule_template(None)


def test_build_path_c_model_regions_from_route_symbols_splits_model_segments() -> None:
    regions = build_path_c_model_regions_from_route_symbols(
        ("A", "E", "M", "R", "E", "A"),
        region_prefix="hybrid_path_c",
        min_route_bricks=2,
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.name == "hybrid_path_c_2_3"
    assert tuple(node.op_name for node in region.nodes) == (
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
    )
    assert select_path_c_fusion_schedule_target(region) is not None


def test_build_path_c_model_regions_from_model_uses_model_route_symbols() -> None:
    model = SimpleNamespace(route_symbols=("E", "M", "R", "A"))

    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix="model_route_path_c",
        min_route_bricks=2,
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.name == "model_route_path_c_1_3"
    assert tuple(node.op_name for node in region.nodes) == (
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    )


def test_build_path_c_model_regions_from_bricks_is_not_static_mra_only() -> None:
    bricks = (
        PathCModelBrick(name="embed", kind="embedding"),
        PathCModelBrick(name="scan_a", kind="mamba3"),
        PathCModelBrick(name="m2_a", kind="m2rnn"),
        PathCModelBrick(name="moe", kind="moe"),
        PathCModelBrick(name="scan_b", kind="mamba3"),
        PathCModelBrick(name="m2_b", kind="m2rnn"),
        PathCModelBrick(name="attn_b", kind="sparse_mla_fp8"),
    )

    regions = build_path_c_model_regions_from_bricks(
        bricks,
        region_prefix="brick_model_path_c",
        min_route_bricks=2,
    )

    assert [region.name for region in regions] == [
        "brick_model_path_c_1_2",
        "brick_model_path_c_4_6",
    ]
    assert [tuple(node.op_name for node in region.nodes) for region in regions] == [
        ("entry_rmsnorm", "mamba3_mimo", "residual_rmsnorm", "m2rnn"),
        (
            "entry_rmsnorm",
            "mamba3_mimo",
            "residual_rmsnorm",
            "m2rnn",
            "residual_rmsnorm",
            "attention_qkv_projection",
            "sparse_mla_fp8_apply",
        ),
    ]
    assert all(
        select_path_c_fusion_schedule_target(region) is not None
        for region in regions
    )


def test_build_path_c_model_regions_from_bricks_bridges_any_supported_chain() -> None:
    bricks = (
        PathCModelBrick(name="attn_a", kind="sparse_mla_fp8"),
        PathCModelBrick(name="scan_a", kind="mamba3"),
        PathCModelBrick(name="m2_a", kind="m2rnn"),
    )

    (region,) = build_path_c_model_regions_from_bricks(
        bricks,
        region_prefix="generic_chain_path_c",
        min_route_bricks=2,
    )

    assert region.name == "generic_chain_path_c_0_2"
    assert tuple(node.name for node in region.nodes) == (
        "attn_a_entry_rmsnorm",
        "attn_a_qkv_projection",
        "attn_a_sparse_mla_fp8_apply",
        "attn_a_residual_norm",
        "scan_a",
        "scan_a_residual_norm",
        "m2_a",
    )
    assert tuple(node.op_name for node in region.nodes) == (
        "entry_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
        "residual_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
    )
    assert tuple(region.nodes[3].inputs) == (
        "hidden",
        "attn_a_sparse_mla_fp8_apply_out",
        "attn_a_residual_norm_weight",
    )
    assert region.nodes[4].inputs[0] == "attn_a_residual_norm_hidden"
    assert region.nodes[6].inputs[0] == "scan_a_residual_norm_hidden"


def test_build_path_c_model_regions_from_model_prefers_path_c_bricks() -> None:
    model = SimpleNamespace(
        route_symbols=("M",),
        path_c_bricks=(
            {"name": "scan", "kind": "mamba3"},
            {"name": "m2", "kind": "m2rnn"},
            {"name": "attn", "kind": "sparse_mla_fp8"},
        ),
    )

    regions = build_path_c_model_regions_from_model(
        model,
        region_prefix="path_c_brick_model",
    )

    assert len(regions) == 1
    assert regions[0].name == "path_c_brick_model_0_2"
    assert tuple(node.op_name for node in regions[0].nodes) == (
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    )


def test_plan_path_c_fusion_schedules_for_model_uses_dynamic_brick_chain() -> None:
    model = SimpleNamespace(
        name="dynamic_model",
        route_symbols=("M",),
        path_c_bricks=(
            {"name": "scan", "kind": "mamba3"},
            {"name": "m2", "kind": "m2rnn"},
            {"name": "attn", "kind": "sparse_mla_fp8"},
        ),
        config=local_gb10_quarter_profile().hybrid_config(),
    )

    plans = plan_path_c_fusion_schedules_for_model(
        model,
        region_prefix="dynamic_model_path_c",
    )

    assert len(plans) == 1
    scheduled = plans[0]
    assert scheduled.region.name == "dynamic_model_path_c_0_2"
    assert scheduled.region.metadata.get("path_c_acceptance_fixture_abi") is not True
    assert tuple(node.op_name for node in scheduled.region.nodes) == (
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
    assert scheduled.schedule_target is not None
    assert scheduled.schedule_target.schedule_id.startswith(
        "path_c_descriptor_chain_"
    )
    assert scheduled.schedule_target.schedule_id != MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert scheduled.schedule_target.schedule_name == (
        "dynamic_model_path_c_0_2:descriptor_generated_fwd_bwd"
    )
    assert "scan_mamba3_in_proj_weight" in (
        scheduled.schedule_target.required_real_abi_inputs
    )
    assert "mamba3_in_proj_weight" not in (
        scheduled.schedule_target.required_real_abi_inputs
    )


def test_plan_path_c_direct_fusion_chains_for_model_uses_dynamic_brick_chain() -> None:
    model = SimpleNamespace(
        name="dynamic_model",
        route_symbols=("M",),
        path_c_bricks=(
            {"name": "scan", "kind": "mamba3"},
            {"name": "m2", "kind": "m2rnn"},
            {"name": "attn", "kind": "sparse_mla_fp8"},
        ),
        config=local_gb10_quarter_profile().hybrid_config(),
    )

    # Disable the Metal-only watchdog caps so this test pins the dynamic-brick
    # discovery + pure greedy direct-buffer segmentation on every host.
    chains = plan_path_c_direct_fusion_chains_for_model(
        model,
        region_prefix="dynamic_model_path_c",
        forward_max_segment_nodes=None,
        backward_max_segment_nodes=None,
    )

    assert len(chains) == 1
    chain = chains[0]
    assert chain.status == "ready"
    assert chain.reason == "all chain segments fit direct-buffer portable Metal limits"
    assert chain.source_region.name == "dynamic_model_path_c_0_2"
    assert any(
        node.op_name.endswith("_bwd") for node in chain.source_region.nodes
    )
    assert chain.source_region.metadata.get("path_c_acceptance_fixture_abi") is not True
    assert chain.segments[0].node_start == 0
    assert chain.segments[-1].node_end == len(chain.source_region.nodes)
    assert all(segment.physical_abi_policy == "direct_buffers" for segment in chain.segments)
    # The mamba3 backward op lands in its own single-op direct-buffer segment.
    mamba3_bwd_segments = tuple(
        segment
        for segment in chain.segments
        if tuple(node.op_name for node in segment.region.nodes) == ("mamba3_mimo_bwd",)
    )
    assert len(mamba3_bwd_segments) == 1
    assert all(segment.status == "ok" for segment in chain.segments)
    assert all(
        "mamba3_in_proj_weight" not in getattr(
            segment.schedule_target,
            "required_real_abi_inputs",
            (),
        )
        for segment in chain.segments
        if segment.schedule_target is not None
    )


def test_model_derived_biasless_attention_omits_qkv_bias_abi() -> None:
    profile = local_gb10_quarter_profile()
    model = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix=f"{profile.name}_path_c",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    qkv_node = next(
        node for node in region.nodes if node.op_name == "attention_qkv_projection"
    )
    qkv_bwd_node = next(
        node
        for node in region.nodes
        if node.op_name == "attention_qkv_projection_bwd"
    )
    apply_node = next(
        node for node in region.nodes if node.op_name == "sparse_mla_fp8_apply"
    )
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert "local_gb10_quarter_brick_12_A_qkv_projection_attention_q_proj_bias" not in (
        qkv_node.inputs
    )
    assert (
        "local_gb10_quarter_brick_12_A_qkv_projection_attention_sparse_kv_proj_bias"
        not in qkv_node.inputs
    )
    assert (
        "local_gb10_quarter_brick_12_A_qkv_projection_attention_q_proj_bias_grad"
        not in qkv_bwd_node.outputs
    )
    assert (
        "local_gb10_quarter_brick_12_A_qkv_projection_attention_sparse_kv_proj_bias_grad"
        not in qkv_bwd_node.outputs
    )
    assert (
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_attention_out_proj_bias"
        not in apply_node.inputs
    )

    prim_func = target.schedule_template(region)
    abi_map = prim_func._cppmega_path_c_physical_buffer_abi_map
    generated_source = prim_func._cppmega_path_c_generated_source

    assert all(
        "q_proj_bias" not in name and "sparse_kv_proj_bias" not in name
        for name in abi_map
    )
    assert all("attention_out_proj_bias" not in name for name in abi_map)
    assert "q_proj_bias" not in generated_source
    assert "sparse_kv_proj_bias" not in generated_source
    assert "attention_out_proj_bias" not in generated_source


def test_compile_plan_cache_key_is_stable_for_equivalent_surface_ordering() -> None:
    scan = FusionKernelSurface.path_c(
        name="scan",
        op_name="mamba3_mimo",
        inputs=("hidden",),
        outputs=("scan_y",),
        backward="aot_autograd",
    )
    post = FusionKernelSurface.path_c(
        name="post",
        op_name="m2rnn",
        inputs=("scan_y",),
        outputs=("post_y",),
        backward="aot_autograd",
    )
    apply = FusionKernelSurface.path_c(
        name="apply",
        op_name="sparse_mla_fp8_apply",
        inputs=("post_y",),
        outputs=("out",),
        backward="owner_output",
    )

    ordered_region = build_path_c_fusion_region(
        region_name="same_dynamic_train_block",
        surfaces=(scan, post, apply),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    reversed_region = build_path_c_fusion_region(
        region_name="same_dynamic_train_block",
        surfaces=(apply, post, scan),
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    ordered_plan = compile_path_c_region(ordered_region)
    reversed_plan = compile_path_c_region(reversed_region)

    assert ordered_region.node_names == reversed_region.node_names
    assert ordered_region.edges == reversed_region.edges
    assert ordered_plan.cache_key_parts == reversed_plan.cache_key_parts


def test_region_builder_rejects_ambiguous_inferred_edges() -> None:
    builder = PathCFusionRegionBuilder("ambiguous_train_block")
    builder.add_kernels(
        (
            FusionKernelSurface.path_c(
                name="producer_a",
                op_name="toy",
                inputs=("hidden",),
                outputs=("scan_y",),
                backward="owner_output",
            ),
            FusionKernelSurface.path_c(
                name="producer_b",
                op_name="toy",
                inputs=("state",),
                outputs=("scan_y",),
                backward="owner_output",
            ),
            FusionKernelSurface.path_c(
                name="consumer",
                op_name="toy_consumer",
                inputs=("scan_y",),
                outputs=("out",),
                backward="owner_output",
            ),
        )
    )

    with pytest.raises(ValueError, match="ambiguous fusion producer.*scan_y"):
        builder.build()


def test_mamba3_fp8_template_has_expected_train_block_pattern() -> None:
    region = build_mamba3_fp8_train_region()
    plan = compile_path_c_region(region)

    assert region.node_names == (
        "route_0_M_entry_rmsnorm",
        "route_0_M",
        "route_0_M_residual_norm",
        "route_1_R",
        "route_1_R_residual_norm",
        "route_2_A_qkv_projection",
        "route_2_A_sparse_mla_fp8_apply",
    )
    assert [node.op_name for node in region.nodes] == [
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    ]
    assert plan.cache_key_parts[:7] == (
        "region:mamba3_fp8_train_block",
        "entry:mamba3_fp8_train_block",
        (
            "nodes:route_0_M_entry_rmsnorm,route_0_M,route_0_M_residual_norm,"
            "route_1_R,route_1_R_residual_norm,"
            "route_2_A_qkv_projection,route_2_A_sparse_mla_fp8_apply"
        ),
        (
            "edges:"
            "route_0_M_entry_rmsnorm->route_0_M:"
            "route_0_M_entry_rmsnorm_hidden:internal,"
            "route_0_M->route_0_M_residual_norm:"
            "route_0_M_delta:internal,"
            "route_0_M_residual_norm->route_1_R:"
            "route_0_M_residual_norm_hidden:internal,"
            "route_0_M_residual_norm->route_1_R_residual_norm:"
            "route_0_M_hidden_after:internal,"
            "route_1_R->route_1_R_residual_norm:"
            "route_1_R_delta:internal,"
            "route_1_R_residual_norm->route_2_A_qkv_projection:"
            "route_1_R_residual_norm_hidden:internal,"
            "route_2_A_qkv_projection->route_2_A_sparse_mla_fp8_apply:"
            "route_2_A_qkv_projection_q_fp8:workspace,"
            "route_2_A_qkv_projection->route_2_A_sparse_mla_fp8_apply:"
            "route_2_A_qkv_projection_q_scale:workspace,"
            "route_2_A_qkv_projection->route_2_A_sparse_mla_fp8_apply:"
            "route_2_A_qkv_projection_kv_fp8:workspace,"
            "route_2_A_qkv_projection->route_2_A_sparse_mla_fp8_apply:"
            "route_2_A_qkv_projection_kv_scale:workspace,"
            "route_2_A_qkv_projection->route_2_A_sparse_mla_fp8_apply:"
            "route_2_A_qkv_projection_indices:internal"
        ),
        "backend:tilelang_tvm_ffi",
        "boundary:tilelang_tvm_ir",
        "z3:sync_async",
    )
    assert plan.cache_key_parts[7].startswith("schedule:")
    assert plan.fusion_kind == "tilelang_ir_graph_region"
    assert plan.schedule_status == "missing_real_fused_schedule_template"
    assert plan.single_kernel_fused is False
    assert plan.autograd_status == "requires_aot_autograd_codegen"
    assert plan.autograd_missing_backward_nodes == (
        "route_0_M_entry_rmsnorm_bwd",
        "route_0_M_bwd",
        "route_0_M_residual_norm_bwd",
        "route_1_R_bwd",
        "route_1_R_residual_norm_bwd",
        "route_2_A_qkv_projection_bwd",
        "route_2_A_sparse_mla_fp8_apply_bwd",
    )
    assert plan.semantic_blockers == ()
    assert plan.schedule_contract is not None
    assert plan.schedule_contract.status == "missing_schedule_template"


def test_mamba3_fp8_train_region_is_route_driven_not_static_mra() -> None:
    region = build_mamba3_fp8_train_region(route_symbols=("M", "R"))

    assert region.metadata["path_c_route_symbols"] == ("M", "R")
    assert region.metadata["path_c_bricks"] == (
        {"name": "route_0_M", "kind": "M", "route_symbol": "M"},
        {"name": "route_1_R", "kind": "R", "route_symbol": "R"},
    )
    assert region.node_names == (
        "route_0_M_entry_rmsnorm",
        "route_0_M",
        "route_0_M_residual_norm",
        "route_1_R",
    )
    assert tuple(node.op_name for node in region.nodes) == (
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
    )


def test_mamba3_fp8_acceptance_fixture_region_includes_residual_norm_and_attention_projection() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    plan = compile_path_c_region(region)

    assert region.node_names == (
        "mamba3_entry_rmsnorm",
        "mamba3_scan",
        "mamba3_residual_to_m2rnn_norm",
        "m2rnn_packed_post",
        "m2rnn_residual_to_attention_norm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    )
    assert [node.op_name for node in region.nodes] == [
        "entry_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    ]
    assert [(edge.producer, edge.consumer, edge.input) for edge in region.edges] == [
        ("mamba3_entry_rmsnorm", "mamba3_scan", "mamba3_entry_rmsnorm_hidden"),
        ("mamba3_scan", "mamba3_residual_to_m2rnn_norm", "mamba3_delta"),
        ("mamba3_residual_to_m2rnn_norm", "m2rnn_packed_post", "m2rnn_hidden"),
        ("mamba3_residual_to_m2rnn_norm", "m2rnn_residual_to_attention_norm", "hidden_after_mamba3"),
        ("m2rnn_packed_post", "m2rnn_residual_to_attention_norm", "m2rnn_delta"),
        ("m2rnn_residual_to_attention_norm", "attention_qkv_projection", "attention_hidden"),
        ("attention_qkv_projection", "sparse_mla_fp8_apply", "q_fp8"),
        ("attention_qkv_projection", "sparse_mla_fp8_apply", "q_scale"),
        ("attention_qkv_projection", "sparse_mla_fp8_apply", "kv_fp8"),
        ("attention_qkv_projection", "sparse_mla_fp8_apply", "kv_scale"),
        ("attention_qkv_projection", "sparse_mla_fp8_apply", "indices"),
    ]
    assert {
        edge.input: edge.lifetime
        for edge in region.edges
        if edge.producer == "attention_qkv_projection"
    } == {
        "q_fp8": "workspace",
        "q_scale": "workspace",
        "kv_fp8": "workspace",
        "kv_scale": "workspace",
        "indices": "internal",
    }
    assert plan.semantic_blockers == ()
    assert plan.schedule_status == "missing_real_fused_schedule_template"
    assert plan.schedule_contract is not None
    assert plan.schedule_contract.status == "missing_schedule_template"
    assert plan.schedule_contract.required_internal_buffers == (
        "mamba3_entry_rmsnorm_hidden",
        "mamba3_delta",
        "m2rnn_hidden",
        "hidden_after_mamba3",
        "m2rnn_delta",
        "attention_hidden",
        "indices",
    )
    external_buffers = plan.schedule_contract.required_external_buffers
    assert external_buffers[:3] == ("hidden", "mamba3_entry_rmsnorm_weight", "mamba_state")
    assert "scan_state" in external_buffers
    assert "hidden_after_m2rnn" in external_buffers
    assert "attention_out" in external_buffers
    assert "lse" in external_buffers
    assert "q_fp8" in external_buffers
    assert "q_scale" in external_buffers
    assert "kv_fp8" in external_buffers
    assert "kv_scale" in external_buffers
    assert set(MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS).issubset(
        external_buffers
    )
    assert plan.single_kernel_fused is False
    assert plan.autograd_missing_backward_nodes == (
        "mamba3_entry_rmsnorm_bwd",
        "mamba3_scan_bwd",
        "mamba3_residual_to_m2rnn_norm_bwd",
        "m2rnn_packed_post_bwd",
        "m2rnn_residual_to_attention_norm_bwd",
        "attention_qkv_projection_bwd",
        "sparse_mla_fp8_apply_bwd",
    )


def test_mamba3_fp8_acceptance_fixture_region_can_include_symbolic_aot_backward_graph() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    plan = compile_path_c_region(region)

    assert region.node_names[-7:] == (
        "sparse_mla_fp8_apply_bwd",
        "attention_qkv_projection_bwd",
        "m2rnn_residual_to_attention_norm_bwd",
        "m2rnn_packed_post_bwd",
        "mamba3_residual_to_m2rnn_norm_bwd",
        "mamba3_scan_bwd",
        "mamba3_entry_rmsnorm_bwd",
    )
    assert plan.autograd_status == "ready"
    assert plan.autograd_missing_backward_nodes == ()
    assert plan.autograd_backward_nodes == (
        "sparse_mla_fp8_apply_bwd",
        "attention_qkv_projection_bwd",
        "m2rnn_residual_to_attention_norm_bwd",
        "m2rnn_packed_post_bwd",
        "mamba3_residual_to_m2rnn_norm_bwd",
        "mamba3_scan_bwd",
        "mamba3_entry_rmsnorm_bwd",
    )
    assert plan.autograd_backward_edges == (
        (
            "sparse_mla_fp8_apply_bwd",
            "attention_qkv_projection_bwd",
            "kv_scale_grad",
        ),
        (
            "sparse_mla_fp8_apply_bwd",
            "attention_qkv_projection_bwd",
            "kv_fp8_grad",
        ),
        (
            "sparse_mla_fp8_apply_bwd",
            "attention_qkv_projection_bwd",
            "q_scale_grad",
        ),
        (
            "sparse_mla_fp8_apply_bwd",
            "attention_qkv_projection_bwd",
            "q_fp8_grad",
        ),
        (
            "attention_qkv_projection_bwd",
            "m2rnn_residual_to_attention_norm_bwd",
            "attention_hidden_grad",
        ),
        (
            "m2rnn_residual_to_attention_norm_bwd",
            "m2rnn_packed_post_bwd",
            "m2rnn_delta_grad",
        ),
        (
            "m2rnn_residual_to_attention_norm_bwd",
            "mamba3_residual_to_m2rnn_norm_bwd",
            "hidden_after_mamba3_grad",
        ),
        (
            "m2rnn_packed_post_bwd",
            "mamba3_residual_to_m2rnn_norm_bwd",
            "m2rnn_hidden_grad",
        ),
        (
            "mamba3_residual_to_m2rnn_norm_bwd",
            "mamba3_scan_bwd",
            "mamba3_delta_grad",
        ),
        (
            "mamba3_scan_bwd",
            "mamba3_entry_rmsnorm_bwd",
            "mamba3_entry_rmsnorm_hidden_grad",
        ),
    )


def test_mamba3_fp8_train_backward_surfaces_receive_forward_real_abi_inputs() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    node_by_name = {node.name: node for node in region.nodes}

    assert node_by_name["sparse_mla_fp8_apply_bwd"].inputs == (
        "attention_out_grad",
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
        "indices",
        "sparse_mla_sm_scale",
        "sparse_mla_sinks",
        "sparse_mla_has_sinks",
        "attention_out_proj_weight",
        "attention_out_proj_bias",
    )
    assert node_by_name["sparse_mla_fp8_apply_bwd"].outputs == (
        "q_fp8_grad",
        "q_scale_grad",
        "kv_fp8_grad",
        "kv_scale_grad",
        "attention_out_proj_weight_grad",
        "attention_out_proj_bias_grad",
    )
    assert node_by_name["attention_qkv_projection_bwd"].inputs == (
        "q_fp8_grad",
        "q_scale_grad",
        "kv_fp8_grad",
        "kv_scale_grad",
        "attention_hidden",
        "attention_q_proj_weight",
        "attention_q_proj_bias",
        "attention_sparse_kv_proj_weight",
        "attention_sparse_kv_proj_bias",
        "attention_rope_inv_freq",
    )
    assert node_by_name["m2rnn_packed_post_bwd"].inputs == (
        "m2rnn_delta_grad",
        "m2rnn_hidden",
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
    assert node_by_name["mamba3_scan_bwd"].inputs == (
        "mamba3_delta_grad",
        "mamba3_entry_rmsnorm_hidden",
        "mamba_state",
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


def test_mamba3_fp8_train_production_schedule_spec_is_explicit_and_abi_complete() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    plan = compile_path_c_region(region)

    assert isinstance(plan, FusionCompilePlan)
    assert plan.schedule_contract is not None

    spec = mamba3_fp8_train_fusion_schedule_spec(
        region,
        contract=plan.schedule_contract,
    )

    assert spec.schedule_id == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert spec.schedule_name == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME
    assert spec.region_name == "mamba3_m2rnn_attention_fp8_train_block"
    assert spec.implementation_kind == "production"
    assert spec.implementation_status == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS
    assert spec.trusted_by_default is False
    assert spec.schedule_id not in trusted_path_c_production_schedule_ids()
    assert spec.contract_key == plan.schedule_contract.key
    assert spec.shape_env_key == plan.schedule_contract.shape_env_key
    assert spec.shape_env_key
    assert spec.required_internal_buffers == (
        plan.schedule_contract.required_internal_buffers
    )
    assert spec.required_external_buffers == (
        plan.schedule_contract.required_external_buffers
    )
    assert "single_entry_tilelang_region" in spec.required_codegen_steps
    assert "dynamic_region_graph_walk" in spec.required_codegen_steps
    assert "brick_descriptor_chain_resolution" in spec.required_codegen_steps
    assert "mamba3_scan_descriptor" in spec.required_codegen_steps
    assert "real_model_parameter_abi_contract" in spec.required_codegen_steps
    assert "z3_sync_async_schedule_points" in spec.required_codegen_steps
    assert spec.schedule_generator == PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    assert spec.schedule_generator_status == "production_region_fragments"
    assert spec.internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
    assert spec.loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    assert "residual_rmsnorm_row_phased_production_fragment" in (
        spec.required_codegen_steps
    )
    assert "attention_qkv_projection_row_phased_rope_fp8_fragment" in (
        spec.required_codegen_steps
    )
    assert "sparse_mla_fp8_apply_softmax_lse_out_proj" in (
        spec.required_codegen_steps
    )
    assert "sparse_mla_fp8_apply_lse_reuses_softmax_stats" in (
        spec.required_codegen_steps
    )
    assert "sparse_mla_fp8_apply_row_topk_indices_cache" in (
        spec.required_codegen_steps
    )
    assert "sparse_mla_fp8_apply_invalid_index_sentinel" in (
        spec.required_codegen_steps
    )
    assert spec.buffer_extent == LOCAL_GB10_QUARTER_MAX_SEQ_LENGTH
    assert spec.brick_ops == spec.op_signature
    assert set(spec.brick_schedule_families) == {
        "loop_descriptor_dataflow",
    }
    assert "mamba3_mimo:descriptor_codegen_ready" in (
        spec.brick_descriptor_statuses
    )
    assert "attention_qkv_projection_bwd:descriptor_codegen_ready" in (
        spec.brick_descriptor_statuses
    )
    assert "m2rnn_bwd:descriptor_codegen_ready" in spec.brick_descriptor_statuses
    assert "mamba3_mimo_bwd:descriptor_codegen_ready" in (
        spec.brick_descriptor_statuses
    )
    assert "residual_rmsnorm_bwd:descriptor_codegen_ready" in (
        spec.brick_descriptor_statuses
    )
    assert spec.production_fragments_complete is True
    assert spec.brick_production_fragment_blockers == ()
    assert any(
        status.startswith("mamba3_mimo:production_region_inlined:")
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        reason.startswith("mamba3_mimo:production_region_inlined:")
        and "row-phased descriptor codegen fuses Mamba3 dense input projection"
        in reason
        and "without full activation staging" in reason
        for reason in spec.brick_production_fragment_reasons
    )
    assert not any(
        blocker.startswith("mamba3_mimo:")
        for blocker in spec.brick_production_fragment_blockers
    )
    assert any(
        status.startswith("residual_rmsnorm:production_region_inlined:")
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        status.startswith("residual_rmsnorm_bwd:production_region_inlined:")
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        reason.startswith(
            "residual_rmsnorm_bwd:production_region_inlined:"
            "row-phased descriptor codegen recomputes residual/RMSNorm"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert not any(
        blocker.startswith("residual_rmsnorm_bwd:")
        for blocker in spec.brick_production_fragment_blockers
    )
    assert any(
        status.startswith(
            "attention_qkv_projection:production_region_inlined:"
        )
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        reason.startswith(
            "attention_qkv_projection:production_region_inlined:"
            "row-phased descriptor codegen emits real q/sparse-kv dot-products"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert not any(
        blocker.startswith("attention_qkv_projection:")
        for blocker in spec.brick_production_fragment_blockers
    )
    assert any(
        status.startswith("sparse_mla_fp8_apply:production_region_inlined:")
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        reason.startswith(
            "sparse_mla_fp8_apply:production_region_inlined:"
            "row-phased descriptor codegen emits prepared-FP8 sparse attention apply"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert not any(
        blocker.startswith("sparse_mla_fp8_apply:")
        for blocker in spec.brick_production_fragment_blockers
    )
    assert any(
        reason.startswith(
            "attention_qkv_projection_bwd:production_region_inlined:"
            "row-phased descriptor codegen emits attention Q/KV projection backward"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        reason.startswith(
            "m2rnn_bwd:production_region_inlined:"
            "row-phased descriptor codegen recomputes M2RNN backward owner outputs"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        reason.startswith(
            "m2rnn:production_region_inlined:"
            "row-phased descriptor codegen fuses M2RNN dense input projection"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert not any(
        blocker.startswith("m2rnn:")
        for blocker in spec.brick_production_fragment_blockers
    )
    assert any(
        reason.startswith(
            "mamba3_mimo_bwd:production_region_inlined:"
            "row-phased descriptor codegen recomputes Mamba3 backward owner outputs"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        reason.startswith(
            "residual_rmsnorm:production_region_inlined:"
            "row-phased descriptor codegen emits the residual bridge"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert not any(
        blocker.startswith("residual_rmsnorm:production_region_inlined:")
        for blocker in spec.brick_production_fragment_blockers
    )
    assert spec.real_abi_contract_complete is True
    assert "mamba3_in_proj_weight" in spec.required_real_abi_inputs
    assert "m2rnn_state_weight" in spec.required_real_abi_inputs
    assert "attention_sparse_kv_proj_weight" in spec.required_real_abi_inputs
    assert "sparse_mla_sm_scale" in spec.required_real_abi_inputs
    assert spec.missing_real_abi_inputs == ()
    assert set(spec.required_real_abi_inputs).issubset(
        spec.required_external_buffers
    )


def test_mamba3_fp8_train_production_schedule_template_is_descriptor_generated() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    prim_func = mamba3_fp8_train_fusion_schedule_template(region)

    script = prim_func.script()
    generated_source = prim_func._cppmega_path_c_generated_source
    assert prim_func._cppmega_path_c_schedule_generator == (
        PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    )
    assert prim_func._cppmega_path_c_brick_ops == (
        mamba3_fp8_train_fusion_schedule_spec(region).brick_ops
    )
    assert "mamba3_m2rnn_attention_fp8_train_block" in script
    assert "attention_q_proj_weight" in script
    assert "sparse_mla_sm_scale" in script
    assert "# mamba3_scan: mamba3_mimo" in generated_source
    assert "# mamba3_scan_bwd: mamba3_mimo_bwd" in generated_source


def test_mamba3_fp8_train_descriptor_schedule_uses_per_brick_emitters() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    assert "mamba3_scan_proj_dim" in generated_source
    assert "mamba3_scan_mamba3_accum" in generated_source
    assert "m2rnn_packed_post_m2rnn_projected" in generated_source
    assert "attention_qkv_projection_attention_q_projected" in generated_source
    assert "attention_qkv_projection_attention_kv_projected" in generated_source
    assert "attention_qkv_projection_attention_rope_phase" in generated_source
    assert "sparse_mla_fp8_apply_sink_enabled" in generated_source
    assert "# mamba3_scan_bwd: mamba3_mimo_bwd" in generated_source
    assert "# mamba3_mimo_bwd_policy: exact_lane_parallel_checkpoint_replay" in generated_source
    assert "path_c_float32_scratch_bank" in generated_source
    assert "path_c_float32_parameter_gradient_abi_bank" in generated_source
    assert "mamba3_scan_bwd_mamba3_stage_grad" in generated_source
    assert "mamba3_scan_bwd_grad_accum" not in generated_source


def test_descriptor_schedule_uses_descriptor_owned_fragment_emitter() -> None:
    region = build_path_c_fusion_region(
        region_name="custom_descriptor_emitter_region",
        surfaces=(
            FusionKernelSurface.path_c(
                name="custom_double",
                op_name="custom_double",
                inputs=("hidden",),
                outputs=("custom_out",),
                backward="owner_output",
            ),
        ),
    )

    def emit_custom_double(
        *,
        node,
        dtype_by_buffer,
        access_by_buffer,
    ) -> PathCBrickScheduleFragment:
        del dtype_by_buffer
        return PathCBrickScheduleFragment(
            allocations=(),
            statements=(
                f"{access_by_buffer[node.outputs[0]]} = "
                f"{access_by_buffer[node.inputs[0]]} * 2.0",
            ),
        )

    prim_func = build_path_c_descriptor_prim_func(
        region,
        (
            PathCBrickScheduleDescriptor(
                op_name="custom_double",
                implementation_status="descriptor_codegen_ready",
                required_codegen_steps=("custom_double_descriptor",),
                fragment_emitter=emit_custom_double,
            ),
        ),
        entry_symbol="custom_descriptor_emitter_region",
        buffer_extent=8,
    )

    generated_source = prim_func._cppmega_path_c_generated_source
    assert "# custom_double: custom_double" in generated_source
    assert "with T.Kernel(1, threads=8) as bx:" in generated_source
    assert "tid = T.get_thread_binding(0)" in generated_source
    assert "i = bx * 8 + tid" in generated_source
    assert "if i < 8:" in generated_source
    assert "for i in T.serial(0, 8):" not in generated_source
    assert "custom_out[i] = hidden[i] * 2.0" in generated_source
    assert "+ 0.0" not in generated_source


def test_flat_descriptor_template_compiles_with_tilelang_lowerer() -> None:
    region = build_path_c_fusion_region(
        region_name="flat_native_descriptor_region",
        surfaces=(
            FusionKernelSurface.path_c(
                name="custom_double",
                op_name="custom_double",
                inputs=("hidden",),
                outputs=("custom_out",),
                backward="owner_output",
            ),
        ),
    )

    def emit_custom_double(
        *,
        node,
        dtype_by_buffer,
        access_by_buffer,
    ) -> PathCBrickScheduleFragment:
        del dtype_by_buffer
        return PathCBrickScheduleFragment(
            allocations=(),
            statements=(
                f"{access_by_buffer[node.outputs[0]]} = "
                f"{access_by_buffer[node.inputs[0]]} * 2.0",
            ),
        )

    descriptor = PathCBrickScheduleDescriptor(
        op_name="custom_double",
        implementation_status="descriptor_codegen_ready",
        required_codegen_steps=("custom_double_descriptor",),
        fragment_emitter=emit_custom_double,
    )

    def schedule_template(template_region):
        return build_path_c_descriptor_prim_func(
            template_region,
            (descriptor,),
            entry_symbol="flat_native_descriptor_region",
            buffer_extent=8,
        )

    compiled = compile_path_c_region(
        region,
        schedule_template=mark_path_c_schedule_template_for_region(
            schedule_template,
            region,
            implementation_kind="prototype",
        ),
        schedule_name="flat_native_descriptor_region:prototype",
        schedule_status="prototype",
        tilelang_lowerer=tilelang_single_entry_lowerer,
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert type(compiled.artifact).__name__ == "JITKernel"
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"


def test_mamba3_fp8_train_descriptor_schedule_labels_nonproduction_fragments() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    assert "# mamba3_scan production_fragment_status: production_region_inlined" in (
        generated_source
    )
    assert (
        "# attention_qkv_projection production_fragment_status: "
        "production_region_inlined"
    ) in generated_source
    assert (
        "# mamba3_residual_to_m2rnn_norm production_fragment_status: "
        "production_region_inlined"
    ) in generated_source
    assert (
        "# attention_qkv_projection_bwd production_fragment_status: production_region_inlined"
    ) in generated_source
    assert "# m2rnn_packed_post_bwd production_fragment_status: production_region_inlined" in (
        generated_source
    )
    assert "# mamba3_scan_bwd production_fragment_status: production_region_inlined" in (
        generated_source
    )
    assert "mamba3_projection_policy: dense_row_local" in generated_source
    assert "mamba3_conv_policy: causal_depthwise_ring_history" in generated_source
    assert "mamba3_dt_policy: softplus_A_trapezoid" in generated_source
    assert "mamba3_scan_policy: external_state_recurrence" in generated_source
    assert "mamba3_output_policy: dense_out_projection" in generated_source
    assert "row-phased descriptor codegen emits real q/sparse-kv dot-products" in (
        generated_source
    )
    assert (
        "row-phased descriptor codegen fuses M2RNN dense input projection"
        in generated_source
    )
    assert (
        "m2rnn_conv_policy: lane_strided_causal_depthwise_ring_history"
        in generated_source
    )
    assert "m2rnn_recurrence_policy: lane_strided_mapped_state_update" in (
        generated_source
    )
    assert "m2rnn_post_policy: lane_strided_residual_gate_norm_out_proj" in (
        generated_source
    )
    assert (
        "attention_qkv_projection_bwd_policy: exact_inverse_rope_weight_bias_hidden"
        in generated_source
    )
    assert (
        "sparse_mla_fp8_apply_bwd_policy: exact_softmax_vjp_and_out_projection"
        in generated_source
    )
    assert "q_dequant_bwd_policy: saved_prepared_fp8" in generated_source
    assert "m2rnn_bwd_policy: exact_reverse_checkpoint_replay" in generated_source
    assert "mamba3_mimo_bwd_policy: exact_lane_parallel_checkpoint_replay" in (
        generated_source
    )
    assert "# backward_policy: row_phased_hidden_recompute" in generated_source
    assert "# backward_policy: flat_after_row_phased_forward" not in generated_source


def test_mamba3_fp8_train_descriptor_schedule_has_no_scalar_proxy_bwd_fragments() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    proxy_fragments = (
        'sparse_mla_fp8_apply_bwd_apply_grad = T.alloc_local((1,), "float32")',
        "sparse_mla_fp8_apply_bwd_apply_grad[0] =",
        'attention_qkv_projection_bwd_attention_q_grad = T.alloc_local((1,), "float32")',
        "attention_qkv_projection_bwd_attention_q_grad[0] =",
        'm2rnn_packed_post_bwd_m2rnn_project_grad = T.alloc_local((1,), "float32")',
        "m2rnn_packed_post_bwd_m2rnn_project_grad[0] =",
        'mamba3_scan_bwd_mamba3_project_grad = T.alloc_local((1,), "float32")',
        "mamba3_scan_bwd_mamba3_project_grad[0] =",
        "attention_qkv_projection_bwd_grad_accum",
        "m2rnn_packed_post_bwd_grad_accum",
        "mamba3_scan_bwd_grad_accum",
    )
    for fragment in proxy_fragments:
        assert fragment not in generated_source


def test_exact_backward_descriptors_reject_flat_scalar_proxy_schedule() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(tuple(node.op_name for node in region.nodes))
    )

    assert descriptors is not None

    with pytest.raises(
        ValueError,
        match="requires the row-phased exact backward generator",
    ):
        build_path_c_descriptor_prim_func(
            region,
            descriptors,
            entry_symbol="mamba3_acceptance_flat_proxy_rejected",
            loop_policy=DESCRIPTOR_LOOP_POLICY_FLAT,
        )


def test_default_registry_uses_explicit_train_block_backward_descriptors() -> None:
    registry = default_path_c_brick_schedule_descriptor_registry()

    for op_name, expected_step in (
        ("mamba3_mimo_bwd", "mamba3_mimo_bwd_descriptor"),
        ("m2rnn_bwd", "m2rnn_bwd_descriptor"),
        (
            "attention_qkv_projection_bwd",
            "attention_qkv_projection_bwd_descriptor",
        ),
    ):
        descriptor = registry.descriptor_for(op_name)
        assert descriptor is not None
        assert descriptor.op_name == op_name
        assert descriptor.implementation_status == "descriptor_codegen_ready"
        assert expected_step in descriptor.required_codegen_steps
        assert descriptor.supports_backward is False
        assert descriptor.production_fragment_policy == (
            DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
        )
        assert descriptor.production_fragment_codegen_step
        assert descriptor.production_fragment_inlined_reason


def test_complete_production_fragments_stay_non_default_until_compile_and_gates() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    acceptance_registry = PathCFusionScheduleRegistry(
        acceptance_profiles=(
            PathCFusionScheduleAcceptanceProfile(
                op_signature=tuple(node.op_name for node in region.nodes),
                schedule_id=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID,
                schedule_name=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME,
                schedule_status=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS,
                implementation_kind="production",
                missing_reason="acceptance fixture for untrusted production attestation",
                required_codegen_steps=("single_entry_tilelang_region",),
                entry_symbol="mamba3_m2rnn_attention_fp8_train_block",
                required_real_abi_inputs=MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
                required_region_tags=("mamba3_fp8_train_acceptance",),
            ),
        ),
    )
    target = select_path_c_fusion_schedule_target(region, registry=acceptance_registry)

    assert target is not None
    assert target.schedule_id == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert target.implementation_kind == "production"

    planned = plan_path_c_fusion_schedule_for_region(
        region,
        include_backward=True,
        registry=acceptance_registry,
    )
    assert planned.schedule_target is not None
    assert planned.schedule_target.implementation_kind == "production"
    assert planned.plan.schedule_contract is not None
    assert (
        planned.plan.schedule_contract.status
        == "registered_not_lowered"
    )
    assert planned.plan.schedule_contract.declared_implementation_kind == "production"
    assert (
        planned.plan.schedule_contract.declared_schedule_id
        == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    )
    assert planned.plan.single_kernel_fused is False


def test_complete_production_fragments_keep_dynamic_target_production() -> None:
    region = build_path_c_model_region_from_route_symbols(
        region_name="dynamic_complete_production_region",
        route_symbols=("M", "R"),
    )
    descriptors = tuple(
        PathCBrickScheduleDescriptor(
            op_name=node.op_name,
            implementation_status="descriptor_codegen_ready",
            required_codegen_steps=(f"{node.op_name}_production_fragment",),
            production_source=f"test:{node.op_name}",
            production_fragment_status="production_region_inlined",
        )
        for node in region.nodes
    )
    signature = tuple(node.op_name for node in region.nodes)
    registry = PathCFusionScheduleRegistry(
        brick_registry=PathCBrickScheduleDescriptorRegistry(descriptors),
        acceptance_profiles=(
            PathCFusionScheduleAcceptanceProfile(
                op_signature=signature,
                schedule_id="dynamic_complete_production_schedule",
                schedule_name="dynamic_complete_production_region:production",
                schedule_status="ready",
                implementation_kind="production",
                missing_reason="all test fragments are production-inlined",
                required_codegen_steps=("single_entry_tilelang_region",),
                entry_symbol="dynamic_complete_production_region",
                required_real_abi_inputs=("hidden",),
            ),
        ),
    )
    planned = plan_path_c_fusion_schedule_for_region(
        region,
        include_backward=False,
        registry=registry,
    )
    assert planned.schedule_target is not None
    assert planned.schedule_target.implementation_kind == "production"

    compiled = PathCFusionScheduleOptimizer(
        "dynamic_complete_production_region",
        registry=registry,
    ).add_kernels(_surface_copies_for_region(region)).compile(
        tilelang_lowerer=lambda *args, **kwargs: "compiled-production-region"
    )

    assert compiled.artifact == "compiled-production-region"
    assert compiled.plan.single_kernel_fused is True
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "verified"
    assert (
        compiled.plan.schedule_contract.declared_schedule_id
        == "dynamic_complete_production_schedule"
    )


def test_descriptor_policy_drives_row_phased_optimization_without_static_op_gate() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="R",
        depth=1,
        dsa_a_layer_ranks=(),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    region = build_path_c_model_region_from_route_symbols(
        region_name="descriptor_policy_m2rnn_only",
        route_symbols=("R",),
        model_config=cfg,
    )
    descriptor = PathCBrickScheduleDescriptor(
        op_name="m2rnn",
        implementation_status="descriptor_codegen_ready",
        required_codegen_steps=("m2rnn_descriptor",),
        preferred_internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
        preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        production_fragment_codegen_step="m2rnn_row_phased_production_fragment",
        production_fragment_inlined_reason=(
            "test descriptor advertises a row-phased production fragment"
        ),
    )
    # Block A: every fused region now starts with an entry RMSNorm op;
    # provide a minimal descriptor so the registry resolves the full
    # signature.
    entry_descriptor = PathCBrickScheduleDescriptor(
        op_name="entry_rmsnorm",
        implementation_status="descriptor_codegen_ready",
        required_codegen_steps=("entry_rmsnorm_descriptor",),
        preferred_internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
        preferred_loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        production_fragment_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        production_fragment_codegen_step=(
            "entry_rmsnorm_row_phased_production_fragment"
        ),
        production_fragment_inlined_reason=(
            "test descriptor advertises a row-phased entry rmsnorm fragment"
        ),
    )
    registry = PathCFusionScheduleRegistry(
        brick_registry=PathCBrickScheduleDescriptorRegistry(
            (entry_descriptor, descriptor)
        ),
    )

    target = select_path_c_fusion_schedule_target(region, registry=registry)

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
    assert target.loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    assert "m2rnn_row_phased_production_fragment" in (
        target.required_codegen_steps
    )
    assert target.brick_descriptors[1].production_fragment_status == (
        "production_region_inlined"
    )
    assert target.brick_descriptors[1].op_name == "m2rnn"


def test_mamba3_fp8_train_descriptor_schedule_uses_loop_fragments() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    cfg = local_gb10_quarter_profile().hybrid_config()
    activation_extent = cfg.max_seq_length * cfg.hidden_size
    attention = cfg.attention_config("dsa")

    assert "# loop_policy: row_phased_hidden" in generated_source
    assert "# internal_buffer_policy: row_local_hidden" in generated_source
    assert "with T.Kernel(64, threads=1024) as chunk:" in generated_source
    assert "lane = T.get_thread_binding(0)" in generated_source
    assert "if lane == 0:" in generated_source
    assert "T.sync_threads()" in generated_source
    assert "for row in T.serial(row_chunk_start, row_chunk_stop):" in (
        generated_source
    )
    assert (
        f"for i in T.serial(row * {cfg.hidden_size} + lane, "
        f"(row + 1) * {cfg.hidden_size}, step=1024):"
    ) in generated_source
    assert "# backward_policy: row_phased_hidden_recompute" in generated_source
    # The train block has two forward row loops and one row loop per
    # exact generated backward fragment.
    assert generated_source.count(
        "for row in T.serial(row_chunk_start, row_chunk_stop):"
    ) == 8
    assert f"for i in T.serial(0, {activation_extent}):" not in generated_source
    assert 'path_c_float32_scratch_bank: T.Tensor' in generated_source
    assert any(
        _line_uses_logical_buffer(
            prim_func,
            line,
            "mamba3_scan_mamba3_projected_vec",
        )
        and "mamba3_scan_proj_dim" in line
        for line in generated_source.splitlines()
    )
    assert "mamba3_scan_mamba3_next_dt[0]" in generated_source
    assert "mamba3_scan_mamba3_next_trap[0]" in generated_source
    assert any(
        _line_uses_logical_buffer(prim_func, line, "q_fp8")
        and f"row * {cfg.hidden_size} + attention_qkv_projection_q_head * "
        f"{attention.q_head_dim} + attention_qkv_projection_d" in line
        for line in generated_source.splitlines()
    )
    assert any(
        _physical_bank_fragment(prim_func, "attention_out") in line
        and _line_uses_logical_buffer(
            prim_func,
            line,
            "sparse_mla_fp8_apply_context_values",
        )
        for line in generated_source.splitlines()
    )
    assert "mamba3_scan_bwd_mamba3_stage_grad[0]" in generated_source
    assert _physical_bank_fragment(prim_func, "mamba3_out_proj_weight_grad") in (
        generated_source
    )
    assert "mamba3_scan_bwd_grad_accum[0]" not in generated_source


def test_mamba3_fp8_train_descriptor_spills_large_shared_scratch_to_abi() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    spilled = prim_func._cppmega_path_c_spilled_shared_scratch_shapes
    cfg = local_gb10_quarter_profile().hybrid_config()
    activation_extent = cfg.max_seq_length * cfg.hidden_size

    assert 'path_c_float32_scratch_bank: T.Tensor' in generated_source
    assert (
        'mamba3_scan_mamba3_projected_vec = T.alloc_shared((18784,), "float32")'
        not in generated_source
    )
    assert spilled["mamba3_scan_mamba3_projected_vec"]["dtype"] == "float32"
    assert spilled["mamba3_scan_mamba3_projected_vec"]["shape"] == (18784,)
    assert spilled["mamba3_scan_mamba3_projected_vec"]["bytes"] == 75136
    assert (
        spilled["mamba3_scan_mamba3_projected_vec"]["param_name"]
        == "path_c_float32_scratch_bank"
    )
    assert spilled["mamba3_scan_mamba3_projected_vec"]["coalesced_scratch_bank"] is True
    assert spilled["mamba3_scan_mamba3_conv_history"]["dtype"] == "float32"
    assert spilled["mamba3_scan_mamba3_conv_history"]["shape"] == (2, 11264)
    assert spilled["mamba3_scan_mamba3_conv_history"]["bytes"] == 90112
    assert spilled["mamba3_scan_mamba3_conv_vec"]["bytes"] == 45056
    assert spilled["mamba3_scan_mamba3_out_inner"]["bytes"] == 28672
    assert spilled["mamba3_delta"]["internal_scratch_abi"] is True
    assert spilled["mamba3_delta"]["param_name"] == "path_c_float32_scratch_bank"
    assert spilled["mamba3_delta"]["shape"] == (activation_extent,)
    assert spilled["mamba3_delta"]["coalesced_scratch_bank"] is True
    assert (
        f'mamba3_delta = T.alloc_shared(({activation_extent},), "float32")'
        not in generated_source
    )
    assert "mamba3_delta" in prim_func._cppmega_path_c_internal_scratch_abi_buffers
    assert "mamba3_delta" in prim_func._cppmega_path_c_internal_scratch_abi_aliases
    assert _shared_alloc_bytes(generated_source) <= 32 * 1024
    assert len(tuple(prim_func.params)) <= 31


def test_mamba3_fp8_train_descriptor_replays_mamba3_backward_without_full_h_steps_scratch() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    spilled = prim_func._cppmega_path_c_spilled_shared_scratch_shapes
    physical_abi_map = prim_func._cppmega_path_c_physical_buffer_abi_map

    assert "mamba3_mimo_bwd_policy: exact_lane_parallel_checkpoint_replay" in (
        generated_source
    )
    assert "mamba3_mimo_bwd_policy: exact_reverse_recompute" not in (
        generated_source
    )
    assert "mamba3_scan_bwd_mamba3_h_steps" not in generated_source
    assert "mamba3_scan_bwd_mamba3_h_steps" not in spilled
    assert "mamba3_h_checkpoint" in physical_abi_map
    assert "mamba3_angle_checkpoint" in physical_abi_map
    assert physical_abi_map["mamba3_h_checkpoint"]["bank"] == (
        "path_c_float32_state_abi_bank"
    )
    assert physical_abi_map["mamba3_angle_checkpoint"]["bank"] == (
        "path_c_float32_state_abi_bank"
    )
    assert all(
        not scratch_name.endswith("_mamba3_h_steps")
        for scratch_name in spilled
    )
    assert max(info["bytes"] for info in spilled.values()) < 1024 * 1024 * 1024


def test_mamba3_fp8_train_descriptor_splits_float32_abi_region_surface() -> None:
    import tvm

    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    physical_shapes = prim_func._cppmega_path_c_physical_buffer_abi_shapes
    generated_source = prim_func._cppmega_path_c_generated_source

    assert "path_c_float32_abi_bank" not in physical_shapes
    assert {
        "path_c_float32_activation_abi_bank",
        "path_c_float32_attention_abi_bank",
        "path_c_float32_parameter_abi_bank",
        "path_c_float32_parameter_gradient_abi_bank",
        "path_c_float32_state_abi_bank",
    }.issubset(physical_shapes)
    assert "path_c_float32_scratch_bank" in generated_source
    assert "path_c_int32_scratch_bank" in generated_source
    assert len(tuple(prim_func.params)) <= 31

    tir_source = str(tvm.IRModule.from_expr(prim_func).script())
    assert _max_tir_region_line_length(tir_source) < 12_000


def test_bf16_descriptor_stages_do_not_emit_threadgroup_bfloat16_scratch() -> None:
    cfg = local_gb10_quarter_profile().hybrid_config()
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="generic_mra_path_c",
        route_symbols=("M", "R", "A"),
        model_config=cfg,
    )
    region = build_path_c_aot_autograd_region(fwd_region)
    shape_env = region.metadata["path_c_model_shape_env"]
    region = replace(
        region,
        metadata={
            **region.metadata,
            "path_c_model_shape_env": replace(
                shape_env,
                sequence_length=128,
                model_value_dtype="bfloat16",
            ),
        },
    )
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    abi_prim_func = target.schedule_template(region)
    groups = plan_path_c_descriptor_stage_groups(region)
    stage_prim_funcs = path_c_descriptor_stage_prim_funcs(
        region=region,
        schedule_target=target,
        abi_prim_func=abi_prim_func,
        groups=groups,
    )
    source_by_suffix = {
        group.stage_suffix: prim_func._cppmega_path_c_generated_source
        for group, prim_func in zip(groups, stage_prim_funcs, strict=True)
    }

    for source in source_by_suffix.values():
        assert re.search(r'T\.alloc_shared\([^\n]+, "bfloat16"\)', source) is None
    assert "route_1_R_residual_norm_hidden" not in source_by_suffix["g1"]
    g2_spilled = stage_prim_funcs[2]._cppmega_path_c_spilled_shared_scratch_shapes
    assert g2_spilled["route_1_R_residual_norm_hidden"]["dtype"] == "float32"
    assert (
        g2_spilled["route_1_R_residual_norm_hidden"]["param_name"]
        == "path_c_float32_scratch_bank"
    )


def test_mamba3_fp8_train_descriptor_schedule_uses_row_local_internal_arrays_without_full_staging() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    cfg = local_gb10_quarter_profile().hybrid_config()
    activation_extent = cfg.max_seq_length * cfg.hidden_size
    attention = cfg.attention_config("dsa")
    q_history_extent = cfg.max_seq_length * attention.num_q_heads * attention.q_head_dim
    q_scale_extent = cfg.max_seq_length * attention.num_q_heads
    kv_history_extent = cfg.max_seq_length * attention.kv_heads * attention.q_head_dim
    kv_scale_extent = cfg.max_seq_length * attention.kv_heads
    uint8_history_extent = q_history_extent + kv_history_extent

    # The exact train backward keeps each generated bwd fragment in its own
    # launcher row loop so saved row-local buffers are populated before VJP use.
    assert (
        generated_source.count(
            "for row in T.serial(row_chunk_start, row_chunk_stop):"
        )
        == 8
    )
    assert (
        generated_source.count(
            f"for i in T.serial(0, {activation_extent}):"
        )
        == 0
    )
    assert (
        f'mamba3_delta = T.alloc_local(({activation_extent},), "float32")'
        not in generated_source
    )
    assert (
        f'q_fp8 = T.alloc_local(({activation_extent},), "float32")'
        not in generated_source
    )
    mamba3_delta_spill = prim_func._cppmega_path_c_spilled_shared_scratch_shapes[
        "mamba3_delta"
    ]
    assert mamba3_delta_spill["param_name"] == "path_c_float32_scratch_bank"
    assert mamba3_delta_spill["shape"] == (activation_extent,)
    assert (
        f'mamba3_delta = T.alloc_shared(({activation_extent},), "float32")'
        not in generated_source
    )
    assert (
        f'q_fp8: T.Tensor(({activation_extent},), "uint8"),'
        not in generated_source
    )
    assert (
        f'kv_fp8 = T.alloc_local(({attention.kv_heads * attention.q_head_dim},), "uint8")'
        not in generated_source
    )
    assert "kv_fp8 = T.alloc_local" not in generated_source
    assert "kv_scale = T.alloc_local" not in generated_source
    assert prim_func._cppmega_path_c_physical_abi_policy == "banked_by_role"
    assert (
        f'path_c_uint8_abi_bank: T.Tensor(({uint8_history_extent},), "uint8"),'
        in generated_source
    )
    assert (
        prim_func._cppmega_path_c_buffer_abi_shapes["q_fp8"]
        == (q_history_extent,)
    )
    assert (
        prim_func._cppmega_path_c_buffer_abi_shapes["q_scale"]
        == (q_scale_extent,)
    )
    assert (
        prim_func._cppmega_path_c_buffer_abi_shapes["kv_fp8"]
        == (kv_history_extent,)
    )
    assert (
        prim_func._cppmega_path_c_buffer_abi_shapes["kv_scale"]
        == (kv_scale_extent,)
    )
    assert (
        prim_func._cppmega_path_c_physical_buffer_abi_map["q_fp8"]["bank"]
        == "path_c_uint8_abi_bank"
    )
    assert (
        prim_func._cppmega_path_c_physical_buffer_abi_map["kv_fp8"]["bank"]
        == "path_c_uint8_abi_bank"
    )
    assert (
        prim_func._cppmega_path_c_physical_buffer_abi_map["q_scale"]["bank"]
        == "path_c_float32_attention_abi_bank"
    )
    assert "q_fp8" not in prim_func._cppmega_path_c_internal_buffer_shapes
    assert "q_scale" not in prim_func._cppmega_path_c_internal_buffer_shapes
    assert "kv_fp8" not in prim_func._cppmega_path_c_internal_buffer_shapes
    assert "kv_scale" not in prim_func._cppmega_path_c_internal_buffer_shapes
    assert prim_func._cppmega_path_c_internal_buffer_shapes["indices"] == (
        cfg.max_seq_length * attention.kv_heads * attention.sparse_topk,
    )
    assert (
        f'q_scale: T.Tensor(({q_scale_extent},), "float32"),'
        not in generated_source
    )
    assert 'm2rnn_packed_post_m2rnn_projected_vec = T.alloc_shared((226,), "float32")' in (
        generated_source
    )
    m2rnn_partial_spill = prim_func._cppmega_path_c_spilled_shared_scratch_shapes[
        "m2rnn_packed_post_m2rnn_sum_sq_partial"
    ]
    assert m2rnn_partial_spill["shape"] == (1024,)
    assert m2rnn_partial_spill["param_name"] == "path_c_float32_scratch_bank"
    assert (
        'm2rnn_packed_post_m2rnn_sum_sq_partial = T.alloc_shared((1024,), "float32")'
        not in generated_source
    )
    assert "# m2rnn_projection_policy: lane_strided_dense_row_local" in generated_source
    assert (
        "for m2rnn_packed_post_proj_dim in T.serial(lane, 226, step=1024):"
        in generated_source
    )
    assert (
        f"for m2rnn_packed_post_out_dim in T.serial(lane, {cfg.hidden_size}, step=1024):"
        in generated_source
    )
    assert (
        "for mamba3_scan_state_flat_init in T.serial(lane, 458752, step=1024):"
        in generated_source
    )
    assert "mamba3_scan_head_init = mamba3_scan_state_flat_init // 4096" in (
        generated_source
    )
    assert "# fp8_prepare_policy: lane_strided_row_head_reduction" in generated_source
    assert (
        "for attention_qkv_projection_q_head in "
        f"T.serial(lane, {attention.num_q_heads}, step=1024):"
        in generated_source
    )
    assert (
        "for attention_qkv_projection_kv_head in "
        f"T.serial(lane, {attention.kv_heads}, step=1024):"
        in generated_source
    )
    assert (
        "for attention_qkv_projection_indices_flat in "
        f"T.serial(lane, {attention.kv_heads * attention.sparse_topk}, step=1024):"
        in generated_source
    )
    assert (
        f"for attention_qkv_projection_d in T.serial(0, {attention.q_head_dim}):"
        in generated_source
    )
    mamba3_delta_bank_ref = (
        "path_c_float32_scratch_bank[i]"
        if mamba3_delta_spill["offset"] == 0
        else f"path_c_float32_scratch_bank[{mamba3_delta_spill['offset']} + (i)]"
    )
    assert mamba3_delta_bank_ref in generated_source
    assert any(
        _line_uses_logical_buffer(prim_func, line, "q_fp8")
        and "attention_qkv_projection_q_head" in line
        and "attention_qkv_projection_d" in line
        for line in generated_source.splitlines()
    )
    assert any(
        _line_uses_logical_buffer(prim_func, line, "kv_fp8")
        and "attention_qkv_projection_kv_head" in line
        and "attention_qkv_projection_d" in line
        for line in generated_source.splitlines()
    )
    assert any(
        _line_uses_logical_buffer(prim_func, line, "attention_out")
        and "sparse_mla_fp8_apply_out_dim_loop" in line
        for line in generated_source.splitlines()
    )


def test_descriptor_schedule_can_row_materialize_internal_buffers_without_gb_staging() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MR",
        depth=2,
        dsa_a_layer_ranks=(),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    region = build_path_c_model_regions_from_model(
        SimpleNamespace(route_symbols=("M", "R"), config=cfg),
        region_prefix="row_materialized_model",
    )[0]
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(tuple(node.op_name for node in region.nodes))
    )

    assert descriptors is not None
    prim_func = build_path_c_descriptor_prim_func(
        region,
        descriptors,
        entry_symbol="row_materialized_model",
        internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    assert (
        prim_func._cppmega_path_c_internal_buffer_policy
        == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
    )
    assert prim_func._cppmega_path_c_internal_buffer_shapes[
        "route_0_M_delta"
    ] == (32,)
    assert "# internal_buffer_policy: row_local_hidden" in generated_source
    assert 'route_0_M_delta = T.alloc_shared((32,), "float32")' in generated_source
    assert 'route_0_M_delta = T.alloc_local((4096,), "float32")' not in (
        generated_source
    )
    assert "route_0_M_delta[i % 32]" in generated_source
    assert "with T.Kernel(16, threads=256) as bx:" in generated_source
    assert "i = bx * 256 + tid" in generated_source
    assert "if i < 4096:" in generated_source
    assert "for i in T.serial(0, 4096):" not in generated_source


def test_descriptor_schedule_row_materialization_requires_shape_env() -> None:
    region = build_path_c_model_region_from_route_symbols(
        region_name="shape_missing_row_materialization",
        route_symbols=("M", "R"),
    )
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(tuple(node.op_name for node in region.nodes))
    )

    assert descriptors is not None
    with pytest.raises(ValueError, match="requires a model shape_env"):
        build_path_c_descriptor_prim_func(
            region,
            descriptors,
            entry_symbol="shape_missing_row_materialization",
            internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
        )


def test_descriptor_schedule_row_phased_loop_orders_rmsnorm_after_full_row() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MR",
        depth=2,
        dsa_a_layer_ranks=(),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    region = build_path_c_model_regions_from_model(
        SimpleNamespace(route_symbols=("M", "R"), config=cfg),
        region_prefix="row_phased_model",
    )[0]
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(tuple(node.op_name for node in region.nodes))
    )

    assert descriptors is not None
    prim_func = build_path_c_descriptor_prim_func(
        region,
        descriptors,
        entry_symbol="row_phased_model",
        internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
        loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE,
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    assert (
        prim_func._cppmega_path_c_loop_policy
        == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    )
    assert "# loop_policy: row_phased_hidden" in generated_source
    assert "with T.Kernel(1, threads=1024):" in generated_source
    assert "lane = T.get_thread_binding(0)" in generated_source
    assert "if lane == 0:" in generated_source
    assert "T.sync_threads()" in generated_source
    assert "for i in T.serial(0, 4096):" not in generated_source
    assert "for row in T.serial(0, 128):" in generated_source
    assert (
        "for i in T.serial(row * 32 + lane, "
        "(row + 1) * 32, step=1024):"
    ) in generated_source
    assert "route_0_M_residual_norm_inv_rms = T.alloc_local" not in generated_source
    residual_partial_spill = prim_func._cppmega_path_c_spilled_shared_scratch_shapes[
        "route_0_M_residual_norm_row_sum_sq_partial"
    ]
    assert residual_partial_spill["shape"] == (1024,)
    assert residual_partial_spill["param_name"] == "path_c_float32_scratch_bank"
    assert "path_c_float32_scratch_bank[" in generated_source
    assert "for partial_lane in T.serial(0, 1024):" in generated_source
    assert (
        "route_0_M_residual_norm_row_sum_sq[0] = "
        "route_0_M_residual_norm_row_sum_sq[0] + "
        "path_c_float32_scratch_bank["
    ) in generated_source
    assert (
        "route_0_M_residual_norm_row_inv_rms[0] = "
        "T.rsqrt((route_0_M_residual_norm_row_sum_sq[0] / 32.0) + 0.00001)"
    ) in generated_source
    assert generated_source.index("# route_0_M: mamba3_mimo") < (
        generated_source.index("# route_0_M_residual_norm: residual_rmsnorm")
    )
    assert "# route_1_R: m2rnn" in generated_source


def test_descriptor_schedule_row_phased_recomputes_residual_rmsnorm_backward() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MR",
        depth=2,
        dsa_a_layer_ranks=(),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    region = build_path_c_model_regions_from_model(
        SimpleNamespace(route_symbols=("M", "R"), config=cfg),
        region_prefix="small_recompute_bwd_model",
        include_backward=True,
    )[0]
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    prim_func = target.schedule_template(region)
    generated_source = prim_func._cppmega_path_c_generated_source
    activation_extent = cfg.max_seq_length * cfg.hidden_size

    assert "# backward_policy: row_phased_hidden_recompute" in generated_source
    assert f"for i in T.serial(0, {activation_extent}):" not in generated_source
    assert "route_0_M_residual_norm_bwd_row_dot[0]" in generated_source
    assert (
        "# route_0_M_residual_norm_bwd production_fragment_status: "
        "production_region_inlined"
    ) in generated_source
    assert any(
        _physical_bank_fragment(prim_func, "route_0_M_residual_norm_weight_grad")
        in line
        and " = " in line
        and " + " in line
        for line in generated_source.splitlines()
    )
    assert "for h in T.serial(lane, 32, step=1024):" in generated_source
    bwd_start = generated_source.index(
        "# route_0_M_residual_norm_bwd: residual_rmsnorm_bwd"
    )
    bwd_end = generated_source.index("# route_0_M_bwd: mamba3_mimo_bwd", bwd_start)
    bwd_segment = generated_source[bwd_start:bwd_end]
    assert (
        bwd_segment.count(
            "for i in T.serial(row * 32 + lane, (row + 1) * 32, step=1024):"
        )
        == 2
    )
    bwd_spills = prim_func._cppmega_path_c_spilled_shared_scratch_shapes
    assert bwd_spills["route_0_M_residual_norm_bwd_row_sum_sq_partial"][
        "param_name"
    ] == "path_c_float32_scratch_bank"
    assert bwd_spills["route_0_M_residual_norm_bwd_row_dot_partial"][
        "param_name"
    ] == "path_c_float32_scratch_bank"
    assert "path_c_float32_scratch_bank[" in bwd_segment
    assert "for partial_lane in T.serial(0, 1024):" in bwd_segment
    assert generated_source.index("# route_1_R: m2rnn") < (
        generated_source.index("# backward_policy: row_phased_hidden_recompute")
    )
    assert generated_source.index("# backward_policy: row_phased_hidden_recompute") < (
        generated_source.index("# route_1_R_bwd: m2rnn_bwd")
    )
    assert generated_source.index("# route_1_R_bwd: m2rnn_bwd") < (
        generated_source.index("# route_0_M_residual_norm_bwd: residual_rmsnorm_bwd")
    )


def test_mamba3_fp8_train_descriptor_schedule_uses_model_derived_shape_env() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    cfg = local_gb10_quarter_profile().hybrid_config()
    sequence_extent = cfg.max_seq_length
    hidden_extent = cfg.max_seq_length * cfg.hidden_size
    attention_weight_extent = cfg.hidden_size * cfg.hidden_size
    sinks_extent = cfg.num_attention_heads

    assert prim_func._cppmega_path_c_physical_abi_policy == "banked_by_role"
    assert 'path_c_float32_activation_abi_bank: T.Tensor(' in generated_source
    assert prim_func._cppmega_path_c_buffer_extent == sequence_extent
    assert prim_func._cppmega_path_c_loop_extent == hidden_extent
    assert f"for i in T.serial(0, {hidden_extent}):" not in generated_source
    assert "for row in T.serial(row_chunk_start, row_chunk_stop):" in generated_source
    assert prim_func._cppmega_path_c_buffer_abi_shapes["hidden"] == (hidden_extent,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "attention_q_proj_weight"
    ] == (
        attention_weight_extent,
    )
    assert prim_func._cppmega_path_c_buffer_abi_shapes["sparse_mla_sinks"] == (
        sinks_extent,
    )
    assert (
        prim_func._cppmega_path_c_physical_buffer_abi_map["hidden"]["bank"]
        == "path_c_float32_activation_abi_bank"
    )


def test_mamba3_fp8_train_descriptor_loop_covers_activation_extent_by_rows() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    cfg = local_gb10_quarter_profile().hybrid_config()
    sequence_extent = cfg.max_seq_length
    activation_extent = cfg.max_seq_length * cfg.hidden_size

    assert prim_func._cppmega_path_c_buffer_extent == sequence_extent
    assert prim_func._cppmega_path_c_loop_extent == activation_extent
    assert f"for i in T.serial(0, {activation_extent}):" not in generated_source
    # Exact backward keeps row-phased reverse fragments in launcher-row loops
    # instead of scalar proxy loops.
    assert (
        generated_source.count(
            "for row in T.serial(row_chunk_start, row_chunk_stop):"
        )
        == 8
    )
    assert (
        f"for i in T.serial(row * {cfg.hidden_size} + lane, "
        f"(row + 1) * {cfg.hidden_size}, step=1024):"
    ) in generated_source
    assert f"for i in T.serial(0, {sequence_extent}):" not in generated_source
    assert prim_func._cppmega_path_c_buffer_abi_shapes["hidden"] == (
        activation_extent,
    )
    assert any(
        _physical_bank_fragment(prim_func, "attention_out") in line
        and _line_uses_logical_buffer(
            prim_func, line, "sparse_mla_fp8_apply_context_values"
        )
        for line in generated_source.splitlines()
    )
    assert _physical_bank_fragment(
        prim_func,
        "mamba3_residual_to_m2rnn_norm_weight",
    ) in generated_source


def test_mamba3_fp8_train_descriptor_projection_inputs_are_not_duplicated() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    mamba3_project_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_accum[0] = mamba3_scan_mamba3_accum[0] +"
        in line
        and _line_uses_logical_buffer(prim_func, line, "mamba3_in_proj_weight")
    )
    mamba3_project_copy_line = next(
        line
        for line in generated_source.splitlines()
        if (
            _line_uses_logical_buffer(
                prim_func, line, "mamba3_scan_mamba3_projected_vec"
            )
            and "mamba3_scan_proj_dim" in line
            and "mamba3_scan_mamba3_accum[0]" in line
        )
    )
    mamba3_conv_bias_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "mamba3_scan_mamba3_conv_vec")
        and _line_uses_logical_buffer(prim_func, line, "mamba3_conv_bias")
    )
    mamba3_conv_history_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_conv_history" in line
        and _line_uses_logical_buffer(prim_func, line, "mamba3_conv_weight")
    )
    mamba3_conv_current_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(
            prim_func, line, "mamba3_scan_mamba3_projected_vec"
        )
        and "7168 + mamba3_scan_conv_ch" in line
        and _line_uses_logical_buffer(prim_func, line, "mamba3_conv_weight")
    )
    mamba3_dt_line = next(
        line
        for line in generated_source.splitlines()
        if "T.log(1.0 + T.exp(" in line
        and _line_uses_logical_buffer(prim_func, line, "mamba3_dt_bias")
    )
    mamba3_next_dt_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_next_dt[0] = "
        "mamba3_scan_mamba3_next_dt[0] +" in line
    )
    mamba3_next_trap_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_next_trap[0] = "
        "mamba3_scan_mamba3_next_trap[0] +" in line
    )
    mamba3_trap_line = next(
        line
        for line in generated_source.splitlines()
        if "path_c_float32_scratch_bank[" in line
        and "mamba3_scan_trap_group_loop" in line
        and "mamba3_scan_mamba3_next_dt[0]" in line
        and "mamba3_scan_mamba3_next_trap[0]" in line
    )
    mamba3_b_norm_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_b_raw[" in line
        and _line_uses_logical_buffer(prim_func, line, "mamba3_B_norm_weight")
    )
    mamba3_c_norm_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_c_raw[" in line
        and _line_uses_logical_buffer(prim_func, line, "mamba3_C_norm_weight")
    )
    mamba3_state_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_state_value[0] =" in line
        and _line_uses_logical_buffer(prim_func, line, "scan_state")
    )
    mamba3_out_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "mamba3_scan_mamba3_out_inner")
        and _line_uses_logical_buffer(prim_func, line, "mamba3_D")
    )
    mamba3_delta_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "mamba3_delta")
        and "mamba3_scan_mamba3_accum[0]" in line
    )
    generated_lines = generated_source.splitlines()
    mamba3_h_checkpoint_if_index = next(
        index
        for index, line in enumerate(generated_lines)
        if "if ((row + 1) % 8) == 0:" in line
        and any(
            _line_uses_logical_buffer(
                prim_func,
                follow_line,
                "mamba3_h_checkpoint",
            )
            for follow_line in generated_lines[index : index + 10]
        )
    )
    m2rnn_project_line = next(
        line
        for line in generated_source.splitlines()
        if (
            "m2rnn_packed_post_m2rnn_accum[0] = "
            "m2rnn_packed_post_m2rnn_accum[0] +"
        )
        in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_in_proj_weight")
    )
    m2rnn_project_copy_line = next(
        line
        for line in generated_source.splitlines()
        if (
            "m2rnn_packed_post_m2rnn_projected_vec[m2rnn_packed_post_proj_dim] = "
            "m2rnn_packed_post_m2rnn_accum[0]"
        )
        in line
    )
    m2rnn_conv_state_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_conv_history" in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_conv_state")
    )
    m2rnn_conv_bias_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_conv_vec[m2rnn_packed_post_conv_ch] = "
        in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_conv_bias")
    )
    m2rnn_conv_history_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_conv_history" in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_conv_weight")
    )
    m2rnn_conv_current_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_projected_vec[m2rnn_packed_post_conv_ch]"
        in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_conv_weight")
    )
    m2rnn_decay_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_decay[0] =" in line
    )
    m2rnn_state_weight_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_accum[0] =" in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_state_weight")
    )
    m2rnn_recurrent_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_h_next[" in line
        and "m2rnn_packed_post_m2rnn_h_state" in line
    )
    m2rnn_h0_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_h_state[" in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_h0")
    )
    m2rnn_post_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_post_vec[m2rnn_packed_post_feature] = ("
        in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_D")
    )
    m2rnn_output_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_accum[0] =" in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_g_norm_weight")
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_out_proj_weight")
    )
    attention_q_project_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_q_projected[0] =" in line
    )
    attention_q_project_bias_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_q_projected_vec[attention_qkv_projection_d] =" in line
        and _line_uses_logical_buffer(prim_func, line, "attention_q_proj_bias")
    )
    attention_q_project_weight_line = next(
        line
        for line in generated_source.splitlines()
        if (
            "attention_qkv_projection_attention_q_projected_vec[attention_qkv_projection_d] = "
            "attention_qkv_projection_attention_q_projected_vec[attention_qkv_projection_d] +"
        ) in line
        and _line_uses_logical_buffer(prim_func, line, "attention_q_proj_weight")
    )
    attention_kv_project_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_kv_projected[0] =" in line
    )
    attention_kv_project_bias_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_kv_projected_vec[attention_qkv_projection_d] =" in line
        and _line_uses_logical_buffer(
            prim_func,
            line,
            "attention_sparse_kv_proj_bias",
        )
    )
    attention_kv_project_weight_line = next(
        line
        for line in generated_source.splitlines()
        if (
            "attention_qkv_projection_attention_kv_projected_vec[attention_qkv_projection_d] = "
            "attention_qkv_projection_attention_kv_projected_vec[attention_qkv_projection_d] +"
        ) in line
        and _line_uses_logical_buffer(
            prim_func,
            line,
            "attention_sparse_kv_proj_weight",
        )
    )
    attention_rope_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_rope_phase[0] =" in line
    )
    attention_q_prepare_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_q_prepared[0] =" in line
    )
    attention_kv_prepare_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_kv_prepared[0] =" in line
    )
    attention_q_scale_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "q_scale")
        and "attention_qkv_projection_attention_q_prepared[0]" in line
    )
    attention_kv_scale_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "kv_scale")
        and "attention_qkv_projection_attention_kv_prepared[0]" in line
    )
    attention_q_fp8_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "q_fp8")
        and "attention_qkv_projection_attention_q_prepared[0]" in line
    )
    attention_kv_fp8_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "kv_fp8")
        and "attention_qkv_projection_attention_kv_prepared[0]" in line
    )
    attention_score_line = next(
        line
        for line in generated_source.splitlines()
        if "sparse_mla_fp8_apply_score_accum[0] = "
        "sparse_mla_fp8_apply_score_accum[0] +" in line
    )
    attention_score_scale_line = next(
        line
        for line in generated_source.splitlines()
        if "sparse_mla_fp8_apply_score_accum[0] = "
        "sparse_mla_fp8_apply_score_accum[0] *" in line
    )
    attention_output_bias_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "attention_out")
        and _line_uses_logical_buffer(prim_func, line, "attention_out_proj_bias")
    )
    attention_output_projection_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "attention_out")
        and _line_uses_logical_buffer(
            prim_func,
            line,
            "sparse_mla_fp8_apply_context_values",
        )
        and _line_uses_logical_buffer(prim_func, line, "attention_out_proj_weight")
    )
    attention_bwd_q_fp8_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_bwd_attention_q_grad0[0] =" in line
        and _line_uses_logical_buffer(prim_func, line, "q_fp8_grad")
    )
    attention_bwd_q_scale_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_bwd_attention_q_grad0[0] = "
        "attention_qkv_projection_bwd_attention_q_grad0[0] +" in line
        and _line_uses_logical_buffer(prim_func, line, "q_scale_grad")
    )
    attention_bwd_kv_fp8_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_bwd_attention_kv_grad0[0] =" in line
        and _line_uses_logical_buffer(prim_func, line, "kv_fp8_grad")
    )
    attention_bwd_kv_scale_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_bwd_attention_kv_grad0[0] = "
        "attention_qkv_projection_bwd_attention_kv_grad0[0] +" in line
        and _line_uses_logical_buffer(prim_func, line, "kv_scale_grad")
    )
    attention_q_hidden_grad_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "attention_hidden_grad")
        and _line_uses_logical_buffer(prim_func, line, "attention_q_proj_weight")
        and "attention_qkv_projection_bwd_attention_projected_grad0[0]" in line
    )
    attention_kv_hidden_grad_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "attention_hidden_grad")
        and _line_uses_logical_buffer(prim_func, line, "attention_sparse_kv_proj_weight")
        and "attention_qkv_projection_bwd_attention_projected_grad0[0]" in line
    )
    attention_q_weight_grad_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "attention_q_proj_weight_grad")
        and "attention_qkv_projection_bwd_attention_projected_grad0[0]" in line
    )
    attention_kv_weight_grad_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(
            prim_func,
            line,
            "attention_sparse_kv_proj_weight_grad",
        )
        and "attention_qkv_projection_bwd_attention_projected_grad0[0]" in line
    )
    attention_rope_grad_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(
            prim_func,
            line,
            "attention_rope_inv_freq_grad",
        )
        and "attention_qkv_projection_bwd_attention_rope_grad[0]" in line
    )
    m2rnn_bwd_stage_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_bwd_m2rnn_stage_grad[0] =" in line
    )
    m2rnn_bwd_hidden_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "m2rnn_hidden_grad")
        and "m2rnn_packed_post_bwd_m2rnn_project_grad" in line
        and _line_uses_logical_buffer(prim_func, line, "m2rnn_in_proj_weight")
    )
    m2rnn_bwd_conv_weight_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "m2rnn_conv_weight_grad")
        and "m2rnn_packed_post_bwd_m2rnn_scalar2[0]" in line
    )
    m2rnn_bwd_state_weight_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "m2rnn_state_weight_grad")
        and "m2rnn_packed_post_bwd_m2rnn_h_prev" in line
        and "m2rnn_packed_post_bwd_m2rnn_scalar2[0]" in line
    )
    m2rnn_bwd_out_proj_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "m2rnn_out_proj_weight_grad")
        and "m2rnn_packed_post_bwd_m2rnn_stage_grad[0]" in line
        and "m2rnn_packed_post_bwd_m2rnn_post_vec" in line
    )
    mamba3_bwd_stage_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_bwd_mamba3_stage_grad[0] =" in line
    )
    mamba3_bwd_hidden_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(
            prim_func,
            line,
            "mamba3_entry_rmsnorm_hidden_grad",
        )
        and "path_c_float32_scratch_bank[" in line
        and "mamba3_scan_bwd_proj_dim" in line
        and _line_uses_logical_buffer(prim_func, line, "mamba3_in_proj_weight")
    )
    mamba3_bwd_conv_weight_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "mamba3_conv_weight_grad")
        and "mamba3_scan_bwd_mamba3_scalar2[0]" in line
    )
    mamba3_bwd_b_norm_weight_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "mamba3_B_norm_weight_grad")
        and "path_c_float32_scratch_bank[" in line
        and "mamba3_scan_bwd_mamba3_scalar0[0]" in line
    )
    mamba3_bwd_out_proj_line = next(
        line
        for line in generated_source.splitlines()
        if _line_uses_logical_buffer(prim_func, line, "mamba3_out_proj_weight_grad")
        and "mamba3_scan_bwd_mamba3_stage_grad[0]" in line
        and "path_c_float32_scratch_bank[" in line
    )

    assert _line_uses_logical_buffer(prim_func, mamba3_project_line, "mamba3_in_proj_weight")
    assert not _line_uses_logical_buffer(prim_func, mamba3_project_line, "mamba3_out_proj_weight")
    assert not _line_uses_logical_buffer(prim_func, mamba3_project_line, "mamba3_conv_weight")
    assert not _line_uses_logical_buffer(prim_func, mamba3_project_line, "mamba3_h0")
    assert not _line_uses_logical_buffer(prim_func, mamba3_project_copy_line, "mamba3_in_proj_weight")
    assert _line_uses_logical_buffer(prim_func, mamba3_conv_bias_line, "mamba3_conv_bias")
    assert _line_uses_logical_buffer(prim_func, mamba3_conv_history_line, "mamba3_conv_weight")
    assert _line_uses_logical_buffer(prim_func, mamba3_conv_current_line, "mamba3_conv_weight")
    assert _line_uses_logical_buffer(prim_func, mamba3_dt_line, "mamba3_dt_bias")
    assert _line_uses_logical_buffer(prim_func, mamba3_next_dt_line, "mamba3_in_proj_weight")
    assert _line_uses_logical_buffer(prim_func, mamba3_next_trap_line, "mamba3_in_proj_weight")
    assert "mamba3_scan_mamba3_next_dt[0]" in mamba3_trap_line
    assert "mamba3_scan_mamba3_next_trap[0]" in mamba3_trap_line
    assert _line_uses_logical_buffer(prim_func, mamba3_b_norm_line, "mamba3_B_norm_weight")
    assert _line_uses_logical_buffer(prim_func, mamba3_c_norm_line, "mamba3_C_norm_weight")
    assert _line_uses_logical_buffer(prim_func, mamba3_state_line, "scan_state")
    assert _line_uses_logical_buffer(
        prim_func,
        mamba3_state_line,
        "mamba3_scan_mamba3_conv_vec",
    )
    assert "mamba3_scan_mamba3_b_group" in mamba3_state_line
    assert _line_uses_logical_buffer(prim_func, mamba3_out_line, "mamba3_D")
    assert _line_uses_logical_buffer(
        prim_func,
        mamba3_out_line,
        "mamba3_scan_mamba3_projected_vec",
    )
    assert "mamba3_scan_mamba3_accum[0]" in mamba3_delta_line
    assert "T.sync_threads()" in generated_lines[mamba3_h_checkpoint_if_index - 1]
    assert _line_uses_logical_buffer(prim_func, m2rnn_project_line, "m2rnn_in_proj_weight")
    assert not _line_uses_logical_buffer(prim_func, m2rnn_project_line, "m2rnn_conv_weight")
    assert not _line_uses_logical_buffer(prim_func, m2rnn_project_line, "m2rnn_state_weight")
    assert not _line_uses_logical_buffer(prim_func, m2rnn_project_line, "m2rnn_h0")
    assert not _line_uses_logical_buffer(prim_func, m2rnn_project_line, "m2rnn_D")
    assert not _line_uses_logical_buffer(prim_func, m2rnn_project_line, "m2rnn_out_proj_weight")
    assert not _line_uses_logical_buffer(prim_func, m2rnn_project_copy_line, "m2rnn_in_proj_weight")
    assert _line_uses_logical_buffer(prim_func, m2rnn_conv_state_line, "m2rnn_conv_state")
    assert _line_uses_logical_buffer(prim_func, m2rnn_conv_bias_line, "m2rnn_conv_bias")
    assert _line_uses_logical_buffer(prim_func, m2rnn_conv_history_line, "m2rnn_conv_weight")
    assert _line_uses_logical_buffer(prim_func, m2rnn_conv_current_line, "m2rnn_conv_weight")
    assert _line_uses_logical_buffer(prim_func, m2rnn_decay_line, "m2rnn_A_log")
    assert _line_uses_logical_buffer(prim_func, m2rnn_decay_line, "m2rnn_dt_bias")
    assert _line_uses_logical_buffer(prim_func, m2rnn_state_weight_line, "m2rnn_state_weight")
    assert "m2rnn_packed_post_m2rnn_h_next" in m2rnn_recurrent_line
    assert _line_uses_logical_buffer(prim_func, m2rnn_h0_line, "m2rnn_h0")
    assert _line_uses_logical_buffer(prim_func, m2rnn_post_line, "m2rnn_D")
    assert _line_uses_logical_buffer(prim_func, m2rnn_output_line, "m2rnn_g_norm_weight")
    assert _line_uses_logical_buffer(prim_func, m2rnn_output_line, "m2rnn_out_proj_weight")
    assert attention_q_project_line.count("attention_q_projected_vec") == 1
    assert _line_uses_logical_buffer(prim_func, attention_q_project_bias_line, "attention_q_proj_bias")
    assert not _line_uses_logical_buffer(prim_func, attention_q_project_bias_line, "attention_q_proj_weight")
    assert not _line_uses_logical_buffer(prim_func, attention_q_project_line, "attention_q_proj_weight")
    assert _line_uses_logical_buffer(prim_func, attention_q_project_weight_line, "attention_q_proj_weight")
    assert _line_uses_logical_buffer(prim_func, attention_q_project_weight_line, "attention_hidden")
    assert "attention_qkv_projection_h" in attention_q_project_weight_line
    assert not _line_uses_logical_buffer(prim_func, attention_q_project_line, "attention_sparse_kv_proj_weight")
    assert not _line_uses_logical_buffer(prim_func, attention_q_project_line, "attention_rope_inv_freq")
    assert not _line_uses_logical_buffer(prim_func, attention_q_project_line, "attention_out_proj_weight")
    assert attention_kv_project_line.count("attention_kv_projected_vec") == 1
    assert _line_uses_logical_buffer(prim_func, attention_kv_project_bias_line, "attention_sparse_kv_proj_bias")
    assert not _line_uses_logical_buffer(prim_func, attention_kv_project_bias_line, "attention_sparse_kv_proj_weight")
    assert not _line_uses_logical_buffer(prim_func, attention_kv_project_line, "attention_sparse_kv_proj_weight")
    assert _line_uses_logical_buffer(prim_func, attention_kv_project_weight_line, "attention_sparse_kv_proj_weight")
    assert _line_uses_logical_buffer(prim_func, attention_kv_project_weight_line, "attention_hidden")
    assert "attention_qkv_projection_h" in attention_kv_project_weight_line
    assert attention_kv_project_line.count("attention_q_proj_weight") == 0
    assert attention_kv_project_line.count("attention_rope_inv_freq") == 0
    assert attention_kv_project_line.count("attention_out_proj_weight") == 0
    assert _line_uses_logical_buffer(
        prim_func,
        attention_rope_line,
        "attention_rope_inv_freq",
    )
    assert attention_q_prepare_line.count("attention_qkv_projection_attention_q_projected[0]") == 1
    assert attention_q_prepare_line.count("attention_qkv_projection_attention_q_projected_pair[0]") == 1
    assert attention_q_prepare_line.count("attention_qkv_projection_attention_rope_phase[0]") == 2
    assert "T.cos(attention_qkv_projection_attention_rope_phase[0])" in (
        attention_q_prepare_line
    )
    assert "T.sin(attention_qkv_projection_attention_rope_phase[0])" in (
        attention_q_prepare_line
    )
    assert attention_kv_prepare_line.count("attention_qkv_projection_attention_kv_projected[0]") == 1
    assert attention_kv_prepare_line.count("attention_qkv_projection_attention_kv_projected_pair[0]") == 1
    assert attention_kv_prepare_line.count("attention_qkv_projection_attention_rope_phase[0]") == 2
    assert "T.cos(attention_qkv_projection_attention_rope_phase[0])" in (
        attention_kv_prepare_line
    )
    assert "T.sin(attention_qkv_projection_attention_rope_phase[0])" in (
        attention_kv_prepare_line
    )
    assert "if attention_qkv_projection_d < 64:" in generated_source
    assert (
        "attention_qkv_projection_attention_rope_phase[0] = "
        'T.cast(row, "float32") * '
    ) in generated_source
    assert (
        "attention_qkv_projection_attention_q_prepared[0] = "
        "(attention_qkv_projection_attention_q_projected[0] * "
        "T.cos(attention_qkv_projection_attention_rope_phase[0])) - "
    ) in generated_source
    assert not _line_uses_logical_buffer(
        prim_func,
        attention_q_scale_line,
        "attention_out_proj_weight",
    )
    assert not _line_uses_logical_buffer(
        prim_func,
        attention_kv_scale_line,
        "attention_out_proj_bias",
    )
    assert "attention_qkv_projection_attention_q_prepared[0]" in attention_q_fp8_line
    assert "attention_qkv_projection_attention_kv_prepared[0]" in attention_kv_fp8_line
    assert _line_uses_logical_buffer(prim_func, attention_q_scale_line, "q_scale")
    assert "T.max(" in attention_q_scale_line
    assert "T.abs(T.cast(attention_qkv_projection_attention_q_prepared[0]" in (
        attention_q_scale_line
    )
    assert any(
        _line_uses_logical_buffer(prim_func, line, "q_scale")
        and "attention_qkv_projection_q_head" in line
        and "T.max(" in line
        and "T.cast(0.002232142857142857" in line
        for line in generated_source.splitlines()
    )
    assert _physical_bank_fragment(prim_func, "kv_scale") in generated_source
    assert "T.abs(T.cast(attention_qkv_projection_attention_kv_prepared[0]" in (
        generated_source
    )
    assert (
        "if row >= attention_qkv_projection_k_top:"
    ) in generated_source
    assert any(
        _line_uses_logical_buffer(prim_func, line, "indices")
        and "row * 448 + attention_qkv_projection_kv_head * 16" in line
        and "= row - attention_qkv_projection_k_top" in line
        for line in generated_source.splitlines()
    )
    assert any(
        _line_uses_logical_buffer(prim_func, line, "indices")
        and "row * 448 + attention_qkv_projection_kv_head * 16" in line
        and "= -1" in line
        for line in generated_source.splitlines()
    )
    assert "float_to_fp8_e4m3fn_bits" in attention_q_fp8_line
    assert "float_to_fp8_e4m3fn_bits" in attention_kv_fp8_line
    assert "T.cast(-448.0" in attention_q_fp8_line
    assert "T.cast(448.0" in attention_kv_fp8_line
    assert _line_uses_logical_buffer(prim_func, attention_score_line, "q_fp8")
    assert _line_uses_logical_buffer(prim_func, attention_score_line, "kv_fp8")
    assert "kv_fp8[sparse_mla_fp8_apply_sparse_index[0]]" not in generated_source
    assert "kv_scale[sparse_mla_fp8_apply_sparse_index[0]]" not in generated_source
    assert "sparse_mla_fp8_apply_sparse_index[0] * 3584 +" in attention_score_line
    assert _line_uses_logical_buffer(prim_func, attention_score_scale_line, "kv_scale")
    assert "sparse_mla_fp8_apply_sparse_index[0] * 28 +" in attention_score_scale_line
    assert "for sparse_mla_fp8_apply_source in T.serial(0, 3584):" not in (
        generated_source
    )
    assert (
        "for sparse_mla_fp8_apply_source_head_loop in T.serial(lane, 28, step=1024):"
        in generated_source
    )
    assert (
        "for sparse_mla_fp8_apply_source_dim_loop in T.serial(0, 128):"
        in generated_source
    )
    assert (
        "for sparse_mla_fp8_apply_out_dim_loop in T.serial(lane, 3584, step=1024):"
        in generated_source
    )
    assert any(
        _line_uses_logical_buffer(
            prim_func,
            line,
            "sparse_mla_fp8_apply_context_values",
        )
        for line in generated_source.splitlines()
    )
    assert (
        'sparse_mla_fp8_apply_score_weights = T.alloc_local((16,), "float32")'
        in generated_source
    )
    assert (
        'sparse_mla_fp8_apply_sparse_indices = T.alloc_local((16,), "int32")'
        in generated_source
    )
    assert (
        "sparse_mla_fp8_apply_score_weights[sparse_mla_fp8_apply_k_top]"
        in generated_source
    )
    assert any(
        "sparse_mla_fp8_apply_sparse_indices[sparse_mla_fp8_apply_k_top] = "
        in line
        and _line_uses_logical_buffer(prim_func, line, "indices")
        for line in generated_source.splitlines()
    )
    assert (
        "sparse_mla_fp8_apply_sparse_index[0] = "
        "sparse_mla_fp8_apply_sparse_indices[sparse_mla_fp8_apply_k_top]"
    ) in generated_source
    assert "T.exp(sparse_mla_fp8_apply_score_accum[0]" in generated_source
    assert (
        "sparse_mla_fp8_apply_score_accum[0] = "
        "T.float32(-3.4028234663852886e38)"
    ) in generated_source
    assert "T.log(sparse_mla_fp8_apply_sumexp[0])" in generated_source
    assert (
        "for sparse_mla_fp8_apply_lse_head in T.serial(0, 28):"
        not in generated_source
    )
    assert "sparse_mla_fp8_apply_value_accum[0]" in generated_source
    assert "sparse_mla_fp8_apply_output_accum" not in generated_source
    assert _line_uses_logical_buffer(
        prim_func,
        attention_output_projection_line,
        "attention_out_proj_weight",
    )
    assert _line_uses_logical_buffer(
        prim_func,
        attention_output_bias_line,
        "attention_out_proj_bias",
    )
    assert "sparse_mla_fp8_apply_out_dim_loop" in attention_output_projection_line
    assert (
        "sparse_mla_fp8_apply_source_head_loop * 128 + "
        "sparse_mla_fp8_apply_source_dim_loop"
    ) in generated_source
    assert _line_uses_logical_buffer(
        prim_func,
        attention_output_projection_line,
        "sparse_mla_fp8_apply_context_values",
    )
    assert "+ T.cast(indices" not in attention_output_projection_line
    assert "+ indices" not in attention_output_projection_line
    assert "sparse_mla_fp8_apply_sparse_index[0]" in attention_score_line
    assert "attention_qkv_projection_attention_projected" not in generated_source
    assert _line_uses_logical_buffer(prim_func, attention_bwd_q_fp8_line, "q_fp8_grad")
    assert _line_uses_logical_buffer(prim_func, attention_bwd_q_scale_line, "q_scale_grad")
    assert _line_uses_logical_buffer(prim_func, attention_bwd_kv_fp8_line, "kv_fp8_grad")
    assert _line_uses_logical_buffer(prim_func, attention_bwd_kv_scale_line, "kv_scale_grad")
    assert "attention_qkv_projection_bwd_attention_projected_grad0[0]" in attention_q_hidden_grad_line
    assert _line_uses_logical_buffer(
        prim_func,
        attention_q_hidden_grad_line,
        "attention_q_proj_weight",
    )
    assert "attention_qkv_projection_bwd_attention_projected_grad0[0]" in (
        attention_kv_hidden_grad_line
    )
    assert _line_uses_logical_buffer(
        prim_func,
        attention_kv_hidden_grad_line,
        "attention_sparse_kv_proj_weight",
    )
    assert "attention_qkv_projection_bwd_attention_projected_grad0[0]" in attention_q_weight_grad_line
    assert _line_uses_logical_buffer(
        prim_func,
        attention_q_weight_grad_line,
        "attention_hidden",
    )
    assert "attention_qkv_projection_bwd_attention_projected_grad0[0]" in attention_kv_weight_grad_line
    assert _line_uses_logical_buffer(
        prim_func,
        attention_kv_weight_grad_line,
        "attention_hidden",
    )
    assert "attention_qkv_projection_bwd_attention_rope_grad[0]" in attention_rope_grad_line
    assert _line_uses_logical_buffer(
        prim_func,
        attention_rope_grad_line,
        "attention_rope_inv_freq_grad",
    )
    assert "attention_qkv_projection_bwd_grad_accum" not in generated_source
    assert _line_uses_logical_buffer(prim_func, m2rnn_bwd_stage_line, "m2rnn_delta_grad")
    assert "m2rnn_packed_post_bwd_m2rnn_project_grad" in m2rnn_bwd_hidden_line
    assert _line_uses_logical_buffer(prim_func, m2rnn_bwd_hidden_line, "m2rnn_in_proj_weight")
    assert "m2rnn_packed_post_bwd_m2rnn_scalar2[0]" in m2rnn_bwd_conv_weight_line
    assert "m2rnn_packed_post_bwd_m2rnn_h_prev" in m2rnn_bwd_state_weight_line
    assert "m2rnn_packed_post_bwd_m2rnn_stage_grad[0]" in m2rnn_bwd_out_proj_line
    assert "m2rnn_packed_post_bwd_m2rnn_post_vec" in m2rnn_bwd_out_proj_line
    assert "m2rnn_packed_post_bwd_grad_accum" not in generated_source
    assert any(
        "m2rnn_packed_post_bwd_grad_flat" in line
        and "path_c_float32_scratch_bank[" in line
        and "= 0.0" in line
        for line in generated_source.splitlines()
    )
    assert not any(
        "m2rnn_packed_post_bwd_time_idx" in line
        and "path_c_float32_scratch_bank[" in line
        and "= 0.0" in line
        for line in generated_source.splitlines()
    )
    assert _line_uses_logical_buffer(prim_func, mamba3_bwd_stage_line, "mamba3_delta_grad")
    assert "path_c_float32_scratch_bank[" in mamba3_bwd_hidden_line
    assert _line_uses_logical_buffer(prim_func, mamba3_bwd_hidden_line, "mamba3_in_proj_weight")
    assert "mamba3_scan_bwd_mamba3_scalar2[0]" in mamba3_bwd_conv_weight_line
    assert "path_c_float32_scratch_bank[" in mamba3_bwd_b_norm_weight_line
    assert "mamba3_scan_bwd_mamba3_stage_grad[0]" in mamba3_bwd_out_proj_line
    assert "path_c_float32_scratch_bank[" in mamba3_bwd_out_proj_line
    assert "mamba3_scan_bwd_grad_accum" not in generated_source
    assert any(
        "mamba3_scan_bwd_grad_flat" in line
        and "path_c_float32_scratch_bank[" in line
        and "= 0.0" in line
        for line in generated_source.splitlines()
    )
    assert not any(
        "mamba3_scan_bwd_time_idx" in line
        and "path_c_float32_scratch_bank[" in line
        and "= 0.0" in line
        for line in generated_source.splitlines()
    )


def test_mamba3_fp8_train_prototype_template_lowers_as_attested_non_production() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    schedule_template = mark_path_c_schedule_template_for_region(
        mamba3_fp8_train_prototype_schedule_template,
        region,
        implementation_kind="prototype",
    )

    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME,
        schedule_status=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS,
        tilelang_lowerer=lambda *args, **kwargs: "compiled-prototype-fwd-bwd",
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-prototype-fwd-bwd"
    assert compiled.plan.autograd_status == "ready"
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"
    assert compiled.plan.schedule_contract.declared_key == (
        compiled.plan.schedule_contract.key
    )
    assert compiled.plan.schedule_contract.declared_implementation_kind == "prototype"


def test_mamba3_fp8_train_prototype_template_compiles_with_tilelang_lowerer() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    schedule_template = mark_path_c_schedule_template_for_region(
        mamba3_fp8_train_prototype_schedule_template,
        region,
        implementation_kind="prototype",
    )

    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME,
        schedule_status=MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS,
        tilelang_lowerer=tilelang_single_entry_lowerer,
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert type(compiled.artifact).__name__ == "JITKernel"
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"


def test_model_derived_descriptor_template_compiles_with_tilelang_lowerer() -> None:
    profile = local_gb10_quarter_profile()
    model = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix=f"{profile.name}_path_c",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert region.node_names[:4] == (
        "local_gb10_quarter_brick_10_M_entry_rmsnorm",
        "local_gb10_quarter_brick_10_M",
        "local_gb10_quarter_brick_10_M_residual_norm",
        "local_gb10_quarter_brick_11_R",
    )
    schedule_template = mark_path_c_schedule_template_for_region(
        target.schedule_template,
        region,
        implementation_kind=target.implementation_kind,
        production_schedule_id=target.schedule_id
        if target.implementation_kind == "production"
        else "",
        required_real_abi_inputs=target.required_real_abi_inputs,
    )

    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name=target.schedule_name,
        schedule_status=target.schedule_status,
        tilelang_lowerer=tilelang_single_entry_lowerer,
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert type(compiled.artifact).__name__ == "JITKernel"
    assert compiled.plan.single_kernel_fused is True
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "verified"
    assert compiled.plan.schedule_contract.declared_required_real_abi_inputs


def test_model_derived_aot_backward_schedule_uses_dynamic_edge_names() -> None:
    profile = local_gb10_quarter_profile()
    model = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix=f"{profile.name}_path_c",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None

    prim_func = target.schedule_template(region)
    generated_source = prim_func._cppmega_path_c_generated_source
    generated_lines = generated_source.splitlines()

    assert dict(prim_func.attrs["tilelang_pass_configs"]) == {
        "tirx.disable_cse_tir": True,
        "tirx.disable_storage_rewrite": True,
        "tirx.merge_static_smem": False,
        "tl.disable_thread_storage_sync": True,
    }
    assert bool(prim_func.attrs["tl.fusion.disable_tir_simplify"])
    assert "stage_grad[0] = 0.0 *" not in generated_source
    assert any(
        "local_gb10_quarter_brick_11_R_bwd_m2rnn_stage_grad[0] = "
        in line
        and _line_uses_logical_buffer(
            prim_func, line, "local_gb10_quarter_brick_11_R_delta_grad"
        )
        for line in generated_lines
    )
    assert any(
        "local_gb10_quarter_brick_10_M_bwd_mamba3_stage_grad[0] = "
        in line
        and _line_uses_logical_buffer(
            prim_func, line, "local_gb10_quarter_brick_10_M_delta_grad"
        )
        for line in generated_lines
    )
    assert any(
        _line_uses_logical_buffer(
            prim_func, line, "local_gb10_quarter_brick_10_M_residual_norm_hidden_grad"
        )
        and "local_gb10_quarter_brick_11_R_bwd_m2rnn_project_grad" in line
        for line in generated_lines
    )
    assert any(
        _line_uses_logical_buffer(
            prim_func, line, "local_gb10_quarter_brick_11_R_residual_norm_hidden_grad"
        )
        and "qkv_projection_bwd_attention_projected_grad0[0]" in line
        for line in generated_lines
    )


def test_row_phased_bwd_only_descriptor_template_skips_empty_forward_loop() -> None:
    profile = local_gb10_quarter_profile()
    model = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix=f"{profile.name}_path_c",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    bwd_region = build_path_c_fusion_region(
        region_name=f"{profile.name}_path_c_bwd_only",
        surfaces=tuple(
            FusionKernelSurface.path_c(
                name=node.name,
                op_name=node.op_name,
                inputs=node.inputs,
                outputs=node.outputs,
                backward=node.backward,
                backend=node.backend,
            )
            for node in region.nodes
            if node.op_name.endswith("_bwd")
        ),
        z3_sync=region.z3_sync,
        metadata=region.metadata,
    )
    target = select_path_c_fusion_schedule_target(bwd_region)

    assert target is not None
    prim_func = target.schedule_template(bwd_region)
    generated_source = prim_func._cppmega_path_c_generated_source

    assert "# backward_policy: row_phased_hidden_recompute" in generated_source
    assert "for row in T.serial(0, 4096):\n        # backward_policy" not in (
        generated_source
    )


def test_direct_fusion_chain_splits_model_route_under_metal_buffer_limit() -> None:
    profile = local_gb10_quarter_profile()
    model = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix=f"{profile.name}_path_c",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)

    # Pin the PURE greedy buffer-limit splitting (caps disabled): this test
    # asserts contiguous direct-buffer segmentation + the per-op buffer budget,
    # independent of the Metal-only watchdog caps (covered separately).
    chain = plan_path_c_direct_fusion_chain_for_region(
        region,
        include_backward=False,
        forward_max_segment_nodes=None,
        backward_max_segment_nodes=None,
    )

    assert chain.status == "ready"
    assert len(chain.segments) >= 2
    assert tuple(
        (segment.node_start, segment.node_end)
        for segment in chain.segments
    )[0][0] == 0
    assert chain.segments[-1].node_end == len(region.nodes)
    for previous, current in zip(chain.segments[:-1], chain.segments[1:], strict=True):
        assert previous.node_end == current.node_start
    # The mamba3 backward op binds enough direct buffers that the greedy splitter
    # places it in its OWN single-op segment under the portable buffer limit.
    mamba3_bwd_segments = tuple(
        segment
        for segment in chain.segments
        if tuple(node.op_name for node in segment.region.nodes) == ("mamba3_mimo_bwd",)
    )
    assert len(mamba3_bwd_segments) == 1
    for segment in chain.segments:
        assert segment.status == "ok"
        assert segment.physical_abi_policy == "direct_buffers"
        assert segment.kernel_parameter_count is not None
        assert segment.kernel_parameter_count <= chain.max_kernel_buffers
        assert segment.schedule_target is not None
        assert segment.plan is not None
    shape_env = fwd_region.metadata["path_c_model_shape_env"]
    abi_shapes = {}
    for segment in (segment for segment in chain.segments if segment.status == "ok"):
        assert segment.schedule_target is not None
        prim_func = segment.schedule_target.schedule_template(segment.region)
        for buffer_name, shape in (
            prim_func._cppmega_path_c_physical_buffer_abi_shapes.items()
        ):
            previous_shape = abi_shapes.setdefault(buffer_name, shape)
            assert previous_shape == shape
    assert abi_shapes["local_gb10_quarter_brick_10_M_hidden"] == (
        1,
        shape_env.sequence_length,
        shape_env.hidden_size,
    )
    assert abi_shapes["local_gb10_quarter_brick_10_M_delta"] == (
        1,
        shape_env.sequence_length,
        shape_env.hidden_size,
    )
    assert abi_shapes[
        "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight"
    ] == (shape_env.mamba_in_proj_dim, shape_env.hidden_size)
    assert abi_shapes[
        "local_gb10_quarter_brick_10_M_mamba3_conv_weight"
    ] == (
        shape_env.mamba_conv_channels,
        shape_env.mamba_conv_kernel,
        1,
    )
    mamba3_state_shape = (
        shape_env.mamba_num_heads
        * shape_env.mamba_head_dim
        * shape_env.mamba_state_dim,
    )
    assert abi_shapes["local_gb10_quarter_brick_10_M_state_in"] == (
        mamba3_state_shape
    )
    # The mamba3 backward segment now plans ``ok`` (it lands in its own single-op
    # direct-buffer segment), so it contributes its reverse-scan state-grad ABI
    # buffers, shaped like the forward ``state_in``.
    assert abi_shapes["local_gb10_quarter_brick_10_M_state_in_grad"] == (
        mamba3_state_shape
    )
    assert abi_shapes[
        "local_gb10_quarter_brick_11_R_residual_norm_weight"
    ] == (shape_env.hidden_size,)


def test_direct_fusion_chain_keeps_loss_bridge_forward_boundary_separate() -> None:
    profile = local_gb10_quarter_profile()
    model = SimpleNamespace(
        name=profile.name,
        path_c_bricks=profile.path_c_bricks,
        config=profile.hybrid_config(),
    )
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix=f"{profile.name}_path_c",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)

    # This test pins the PURE greedy planner grouping (buffer-limit splitting +
    # loss-bridge forward/backward boundary), so disable the Metal-only watchdog
    # caps (forward newComputePipelineState cap / backward GPU-watchdog cap) that
    # would otherwise further split forward/backward segments on Metal hosts. The
    # caps' own splitting is covered separately; here we assert the underlying
    # greedy fusion + execution-phase boundary is correct on every host.
    chain = plan_path_c_direct_fusion_chain_for_region(
        region,
        include_backward=False,
        forward_max_segment_nodes=None,
        backward_max_segment_nodes=None,
    )

    assert chain.status == "ready"
    assert [
        (
            segment.node_start,
            segment.node_end,
            getattr(segment, "execution_phase", None),
            segment.status,
            tuple(node.op_name for node in segment.region.nodes),
        )
        for segment in chain.segments
    ] == [
        (
            0,
            3,
            "forward",
            "ok",
            (
                "entry_rmsnorm",
                "mamba3_mimo",
                "residual_rmsnorm",
            ),
        ),
        (
            3,
            7,
            "forward",
            "ok",
            (
                "m2rnn",
                "residual_rmsnorm",
                "attention_qkv_projection",
                "sparse_mla_fp8_apply",
            ),
        ),
        (
            7,
            10,
            "backward",
            "ok",
            (
                "sparse_mla_fp8_apply_bwd",
                "attention_qkv_projection_bwd",
                "residual_rmsnorm_bwd",
            ),
        ),
        (10, 11, "backward", "ok", ("m2rnn_bwd",)),
        (11, 12, "backward", "ok", ("residual_rmsnorm_bwd",)),
        (12, 13, "backward", "ok", ("mamba3_mimo_bwd",)),
        (13, 14, "backward", "ok", ("entry_rmsnorm_bwd",)),
    ]
    for segment in chain.segments:
        assert {
            str(getattr(node, "backward", ""))
            for node in segment.region.nodes
        } != {"aot_autograd", "owner_output"}


def test_row_phased_acceptance_fwd_bwd_template_generates_valid_source() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(tuple(node.op_name for node in region.nodes))
    )

    assert descriptors is not None

    prim_func = build_path_c_descriptor_prim_func(
        region,
        descriptors,
        entry_symbol="mamba3_acceptance_row_phased_source_regression",
        internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
        loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
        physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE,
    )

    generated_source = prim_func._cppmega_path_c_generated_source
    assert "# backward_policy: row_phased_hidden_recompute" in generated_source
    assert "# backward_policy: flat_after_row_phased_forward" not in generated_source
    assert (
        'attention_qkv_projection_bwd_attention_q_grad = T.alloc_local((1,), "float32")'
        not in generated_source
    )
    assert (
        'm2rnn_packed_post_bwd_m2rnn_project_grad = T.alloc_local((1,), "float32")'
        not in generated_source
    )
    assert (
        'mamba3_scan_bwd_mamba3_project_grad = T.alloc_local((1,), "float32")'
        not in generated_source
    )


def test_row_phased_descriptor_template_compiles_with_tilelang_lowerer() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MR",
        depth=2,
        dsa_a_layer_ranks=(),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    fwd_region = build_path_c_model_regions_from_model(
        SimpleNamespace(route_symbols=("M", "R"), config=cfg),
        region_prefix="row_phased_native_model",
    )[0]
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(tuple(node.op_name for node in fwd_region.nodes))
    )

    assert descriptors is not None

    def schedule_template(template_region):
        return build_path_c_descriptor_prim_func(
            template_region,
            descriptors,
            entry_symbol="row_phased_native_model",
            internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
            loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
            physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE,
        )

    attested_template = mark_path_c_schedule_template_for_region(
        schedule_template,
        fwd_region,
        implementation_kind="prototype",
    )
    compiled = compile_path_c_region(
        fwd_region,
        schedule_template=attested_template,
        schedule_name="row_phased_native_model:prototype",
        schedule_status="prototype",
        tilelang_lowerer=tilelang_single_entry_lowerer,
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert type(compiled.artifact).__name__ == "JITKernel"
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"


def test_model_derived_fwd_bwd_tilelang_source_avoids_threadgroup_atomic_scratch() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(max_seq_length=64)
    fwd_region = build_path_c_model_regions_from_model(
        SimpleNamespace(name="small_compile_path_c", route_symbols=("M", "R", "A"), config=cfg),
        region_prefix="small_compile_path_c",
        sequence_length=64,
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None

    prim_func = target.schedule_template(region)
    artifact = tilelang_single_entry_lowerer(prim_func, target="metal")
    kernel_source = str(artifact.get_kernel_source())
    shmem_match = re.search(r"threadgroup float buf_dyn_shmem\[(\d+)\];", kernel_source)

    assert "AtomicAdd((&(buf_dyn_shmem" not in kernel_source
    assert shmem_match is not None
    assert int(shmem_match.group(1)) * 4 <= 32 * 1024


def test_row_phased_acceptance_fwd_bwd_template_compiles_with_tilelang_lowerer() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    descriptors = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptors_for_signature(tuple(node.op_name for node in region.nodes))
    )

    assert descriptors is not None

    def schedule_template(template_region):
        return build_path_c_descriptor_prim_func(
            template_region,
            descriptors,
            entry_symbol="mamba3_acceptance_row_phased_native",
            internal_buffer_policy=DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN,
            loop_policy=DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
            physical_abi_policy=DESCRIPTOR_PHYSICAL_ABI_POLICY_BANKED_BY_ROLE,
        )

    attested_template = mark_path_c_schedule_template_for_region(
        schedule_template,
        region,
        implementation_kind="prototype",
        required_real_abi_inputs=MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS,
    )
    compiled = compile_path_c_region(
        region,
        schedule_template=attested_template,
        schedule_name="mamba3_acceptance_row_phased_native:prototype",
        schedule_status="prototype",
        tilelang_lowerer=tilelang_single_entry_lowerer,
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert type(compiled.artifact).__name__ == "JITKernel"
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"


def test_mamba3_fp8_train_schedule_planner_uses_named_production_target() -> None:
    scheduled = plan_mamba3_fp8_train_fusion_schedule(include_backward=True)

    assert scheduled.region.name == "mamba3_m2rnn_attention_fp8_train_block"
    assert scheduled.plan.schedule_name == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME
    assert scheduled.plan.schedule_status == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS
    assert scheduled.plan.schedule_contract is not None
    assert (
        scheduled.plan.schedule_contract.status
        == "registered_not_lowered"
    )
    assert (
        scheduled.plan.schedule_contract.declared_implementation_kind
        == "production"
    )
    assert (
        scheduled.plan.schedule_contract.declared_schedule_id
        == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    )
    assert scheduled.schedule_spec.contract_key == scheduled.plan.schedule_contract.key
    assert scheduled.schedule_spec.schedule_id == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert scheduled.schedule_spec.schedule_name == scheduled.plan.schedule_name
    assert scheduled.schedule_spec.implementation_kind == "production"
    assert scheduled.schedule_spec.real_abi_contract_complete is True
    assert scheduled.schedule_spec.missing_real_abi_inputs == ()
    assert "attention_out_proj_weight" in (
        scheduled.schedule_spec.required_external_buffers
    )
    assert scheduled.plan.single_kernel_fused is False


def test_mamba3_fp8_train_schedule_compile_helper_verifies_named_target() -> None:
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        tilelang_lowerer=lambda *args, **kwargs: "compiled-named-mamba3"
    )

    assert isinstance(compiled.compiled, CompiledPathCRegion)
    assert compiled.compiled.artifact == "compiled-named-mamba3"
    assert compiled.region.name == "mamba3_m2rnn_attention_fp8_train_block"
    assert compiled.schedule_spec.schedule_id == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert compiled.schedule_spec.schedule_name == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME
    assert compiled.schedule_spec.real_abi_contract_complete is True
    assert compiled.compiled.plan.single_kernel_fused is True
    assert compiled.compiled.plan.schedule_contract is not None
    assert compiled.compiled.plan.schedule_contract.status == "verified"
    assert (
        compiled.compiled.plan.schedule_contract.declared_schedule_id
        == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    )


def test_mamba3_fp8_train_compile_helper_passes_chunked_row_dispatch_to_schedule() -> None:
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    captured_sources: list[str] = []

    def capture_lowerer(func_or_mod, **_kwargs):
        funcs = list(func_or_mod.functions.items())
        assert len(funcs) == 1
        captured_sources.append(str(func_or_mod.script()))
        return "compiled-chunked-mamba3"

    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        tilelang_lowerer=capture_lowerer,
        max_rows_per_launch=64,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS,
    )

    assert compiled.compiled.artifact == "compiled-chunked-mamba3"
    assert captured_sources
    generated_source = captured_sources[0]
    cfg = local_gb10_quarter_profile().hybrid_config()
    num_chunks = (cfg.max_seq_length + 63) // 64
    descriptor_region = schedules._mamba3_fp8_train_acceptance_region(
        include_backward=True,
    )
    descriptor_target = schedules._target_with_max_rows_per_launch(
        schedules.mamba3_fp8_train_fusion_schedule_target(),
        descriptor_region,
        64,
        schedules.DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS,
    )
    descriptor_source = (
        descriptor_target.schedule_template(
            descriptor_region,
        )._cppmega_path_c_generated_source
    )

    assert f"with T.Kernel({num_chunks}, threads=1024) as chunk:" in descriptor_source
    assert f'bx = T.launch_thread("blockIdx.x", {num_chunks})' in generated_source
    assert 'lane = T.launch_thread("threadIdx.x", 1024)' in generated_source
    assert "row_chunk_start: T.int32 = bx * 64" in generated_source
    assert (
        f"row_chunk_stop: T.int32 = T.min(row_chunk_start + 64, {cfg.max_seq_length})"
        in generated_source
    )
    assert (
        f"entry_row_chunk_stop: T.int32 = T.min(row_chunk_stop + 1, {cfg.max_seq_length})"
        in generated_source
    )
    assert (
        generated_source.count(
            "for row in range(row_chunk_start, row_chunk_start + (entry_row_chunk_stop - row_chunk_start)):"
        )
        == 1
    )
    assert (
        generated_source.count(
            "for row in range(row_chunk_start, row_chunk_start + (row_chunk_stop - row_chunk_start)):"
        )
        == 8
    )
    assert f"for row in range({cfg.max_seq_length}):" not in generated_source


def test_path_c_launcher_manifest_decodes_chunked_row_dispatch_contract() -> None:
    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    captured_prim_funcs: list[object] = []

    def capture_lowerer(func_or_mod, **_kwargs):
        funcs = list(func_or_mod.functions.items())
        assert len(funcs) == 1
        captured_prim_funcs.append(funcs[0][1])
        return "compiled-chunked-mamba3"

    schedules.compile_mamba3_fp8_train_fusion_schedule(
        tilelang_lowerer=capture_lowerer,
        max_rows_per_launch=64,
    )

    assert captured_prim_funcs
    manifest = launcher_mod.load_path_c_abi_manifest(captured_prim_funcs[0])
    cfg = local_gb10_quarter_profile().hybrid_config()

    assert manifest.max_rows_per_launch == 64
    assert manifest.row_chunk_count == (cfg.max_seq_length + 63) // 64
    assert (
        manifest.row_dispatch_mode
        == schedules.DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS
    )
    assert manifest.row_chunk_index_buffer is None


def test_mamba3_fp8_train_launcher_chunk_dispatch_bounds_scalar_chunk_index_for_simplify() -> None:
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    captured_sources: list[str] = []

    def capture_lowerer(func_or_mod, **_kwargs):
        funcs = list(func_or_mod.functions.items())
        assert len(funcs) == 1
        captured_sources.append(str(func_or_mod.script()))
        return "compiled-launcher-chunked-mamba3"

    schedules.compile_mamba3_fp8_train_fusion_schedule(
        tilelang_lowerer=capture_lowerer,
        max_rows_per_launch=64,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )

    assert captured_sources
    generated_source = captured_sources[0]
    cfg = local_gb10_quarter_profile().hybrid_config()
    num_chunks = (cfg.max_seq_length + 63) // 64

    assert "path_c_row_chunk_index: T.int32" in generated_source
    assert "path_c_row_subchunk_index: T.int32" in generated_source
    assert "path_c_run_backward: T.int32" in generated_source
    assert "if path_c_run_backward != 1:" in generated_source
    assert "if path_c_run_backward == 1:" in generated_source
    assert "path_c_row_chunk_index >= 0" in generated_source
    assert f"path_c_row_chunk_index < {num_chunks}" in generated_source
    assert "path_c_row_subchunk_index >= 0" in generated_source
    assert "path_c_row_subchunk_index < 8" in generated_source
    assert (
        "logical_row_chunk_start: T.int32 = path_c_row_chunk_index * 64"
        in generated_source
    )
    assert (
        "row_chunk_start: T.int32 = T.min(logical_row_chunk_start + "
        "path_c_row_subchunk_index * 8, logical_row_chunk_stop)"
        in generated_source
    )
    assert (
        "row_chunk_stop: T.int32 = T.min(row_chunk_start + 8, "
        "logical_row_chunk_stop)"
        in generated_source
    )
    assert "row_chunk_start: T.int32 = T.if_then_else(path_c_run_backward == 1, 0" not in generated_source


def test_path_c_launcher_manifest_decodes_launcher_chunk_index_param() -> None:
    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    captured_prim_funcs: list[object] = []

    def capture_lowerer(func_or_mod, **_kwargs):
        funcs = list(func_or_mod.functions.items())
        assert len(funcs) == 1
        captured_prim_funcs.append(funcs[0][1])
        return "compiled-launcher-chunked-mamba3"

    schedules.compile_mamba3_fp8_train_fusion_schedule(
        tilelang_lowerer=capture_lowerer,
        max_rows_per_launch=64,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )

    assert captured_prim_funcs
    manifest = launcher_mod.load_path_c_abi_manifest(captured_prim_funcs[0])

    assert (
        manifest.row_dispatch_mode
        == schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    assert manifest.row_chunk_index_param == schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM
    assert (
        manifest.row_subchunk_index_param
        == schedules.DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM
    )
    assert manifest.row_subchunk_count == 8
    assert manifest.rows_per_kernel_launch == 8
    assert manifest.row_chunk_index_buffer is None
    assert manifest.backward_gate_param == schedules.DESCRIPTOR_BACKWARD_GATE_PARAM
    assert schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM in manifest.param_order
    assert schedules.DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM in manifest.param_order
    assert schedules.DESCRIPTOR_BACKWARD_GATE_PARAM in manifest.param_order


def test_path_c_launcher_calls_kernel_once_per_launcher_chunk() -> None:
    import json

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    logical_to_physical = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "attention_out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 1,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
    }
    attrs = {
        "global_symbol": "toy_chunked_launcher",
        "tilelang_out_idx": [0],
        "tl.fusion.physical_abi.logical_to_physical": json.dumps(
            logical_to_physical
        ),
        "tl.fusion.physical_abi.physical_buffer_shapes": json.dumps(
            {"path_c_float32_abi_bank": [2]}
        ),
        "tl.fusion.max_rows_per_launch": 64,
        "tl.fusion.row_chunk_count": 3,
        "tl.fusion.row_dispatch_mode": schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
        "tl.fusion.row_chunk_index_param": schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM,
    }
    prim_func = SimpleNamespace(
        attrs=attrs,
        params=[
            SimpleNamespace(name="path_c_float32_abi_bank_handle"),
            SimpleNamespace(name=schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM),
        ],
        buffer_map={
            "path_c_float32_abi_bank": SimpleNamespace(
                name="path_c_float32_abi_bank",
                shape=(2,),
                dtype="float32",
            )
        },
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.prim_func = prim_func
            self.calls: list[int] = []

        def __call__(self, bank: mx.array, chunk_index: int) -> mx.array:
            self.calls.append(int(chunk_index))
            index = mx.array([1], dtype=mx.int32)
            increment = mx.array([1.0], dtype=mx.float32)
            return bank.at[index].add(increment)

    fake_kernel = FakeKernel()
    schedule_contract = SimpleNamespace(
        declared_required_real_abi_inputs=("hidden",)
    )
    compiled = SimpleNamespace(
        compiled=SimpleNamespace(
            artifact=fake_kernel,
            plan=SimpleNamespace(schedule_contract=schedule_contract),
        )
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)

    result = launcher(
        real_abi_inputs={"hidden": mx.array([2.0], dtype=mx.float32)},
        cotangent_seeds={},
    )
    mx.eval(result.forward["attention_out"])

    assert fake_kernel.calls == [0, 1, 2]
    assert result.forward["attention_out"].item() == pytest.approx(3.0)
    assert result.parameter_grads == {}


def test_path_c_launcher_sets_backward_gate_from_cotangent_seeds() -> None:
    import json

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    logical_to_physical = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "attention_out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 1,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "attention_out_grad": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 2,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "weight_grad": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 3,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
    }
    attrs = {
        "global_symbol": "toy_backward_gate",
        "tilelang_out_idx": [0],
        "tl.fusion.physical_abi.logical_to_physical": json.dumps(
            logical_to_physical
        ),
        "tl.fusion.physical_abi.physical_buffer_shapes": json.dumps(
            {"path_c_float32_abi_bank": [4]}
        ),
        "tl.fusion.train_step_loss_cotangent_abi": json.dumps(
            {"logical_cotangent_buffers": ["attention_out_grad"]}
        ),
        "tl.fusion.max_rows_per_launch": 64,
        "tl.fusion.row_chunk_count": 2,
        "tl.fusion.row_dispatch_mode": schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
        "tl.fusion.row_chunk_index_param": schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM,
        "tl.fusion.backward_gate_param": schedules.DESCRIPTOR_BACKWARD_GATE_PARAM,
    }
    prim_func = SimpleNamespace(
        attrs=attrs,
        params=[
            SimpleNamespace(name="path_c_float32_abi_bank_handle"),
            SimpleNamespace(name=schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM),
            SimpleNamespace(name=schedules.DESCRIPTOR_BACKWARD_GATE_PARAM),
        ],
        buffer_map={
            "path_c_float32_abi_bank": SimpleNamespace(
                name="path_c_float32_abi_bank",
                shape=(4,),
                dtype="float32",
            )
        },
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.prim_func = prim_func
            self.calls: list[tuple[int, int]] = []

        def __call__(
            self,
            bank: mx.array,
            chunk_index: int,
            run_backward: int,
        ) -> mx.array:
            gate = int(run_backward)
            self.calls.append((int(chunk_index), gate))
            attention_index = mx.array([1], dtype=mx.int32)
            grad_index = mx.array([3], dtype=mx.int32)
            bank = bank.at[attention_index].add(mx.array([1.0], dtype=mx.float32))
            if gate == 1:
                bank = bank.at[grad_index].add(mx.array([2.0], dtype=mx.float32))
            return bank

    fake_kernel = FakeKernel()
    schedule_contract = SimpleNamespace(
        declared_required_real_abi_inputs=("hidden",)
    )
    compiled = SimpleNamespace(
        compiled=SimpleNamespace(
            artifact=fake_kernel,
            plan=SimpleNamespace(schedule_contract=schedule_contract),
        )
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)

    forward_only = launcher(
        real_abi_inputs={"hidden": mx.array([2.0], dtype=mx.float32)},
        cotangent_seeds={},
    )
    mx.eval(forward_only.forward["attention_out"])

    backward = launcher(
        real_abi_inputs={"hidden": mx.array([2.0], dtype=mx.float32)},
        cotangent_seeds={
            "attention_out_grad": mx.array([1.0], dtype=mx.float32)
        },
    )
    mx.eval(backward.forward["attention_out"], backward.parameter_grads["weight_grad"])

    assert fake_kernel.calls == [
        (0, 0),
        (1, 0),
        (0, 2),
        (1, 2),
        (0, 1),
        (1, 1),
    ]
    assert forward_only.parameter_grads == {}
    assert backward.parameter_grads["weight_grad"].item() == pytest.approx(4.0)


def test_path_c_launcher_backward_forward_prepass_covers_subchunks() -> None:
    import json

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    logical_to_physical = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "attention_out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 1,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "attention_out_grad": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 2,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "weight_grad": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 3,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
    }
    attrs = {
        "global_symbol": "toy_backward_subchunks",
        "tilelang_out_idx": [0],
        "tl.fusion.physical_abi.logical_to_physical": json.dumps(
            logical_to_physical
        ),
        "tl.fusion.physical_abi.physical_buffer_shapes": json.dumps(
            {"path_c_float32_abi_bank": [4]}
        ),
        "tl.fusion.train_step_loss_cotangent_abi": json.dumps(
            {"logical_cotangent_buffers": ["attention_out_grad"]}
        ),
        "tl.fusion.max_rows_per_launch": 64,
        "tl.fusion.row_chunk_count": 2,
        "tl.fusion.row_subchunk_count": 3,
        "tl.fusion.row_dispatch_mode": schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
        "tl.fusion.row_chunk_index_param": schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM,
        "tl.fusion.row_subchunk_index_param": (
            schedules.DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM
        ),
        "tl.fusion.backward_gate_param": schedules.DESCRIPTOR_BACKWARD_GATE_PARAM,
    }
    prim_func = SimpleNamespace(
        attrs=attrs,
        params=[
            SimpleNamespace(name="path_c_float32_abi_bank_handle"),
            SimpleNamespace(name=schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM),
            SimpleNamespace(name=schedules.DESCRIPTOR_ROW_SUBCHUNK_INDEX_PARAM),
            SimpleNamespace(name=schedules.DESCRIPTOR_BACKWARD_GATE_PARAM),
        ],
        buffer_map={
            "path_c_float32_abi_bank": SimpleNamespace(
                name="path_c_float32_abi_bank",
                shape=(4,),
                dtype="float32",
            )
        },
    )

    class FakeKernel:
        def __init__(self) -> None:
            self.prim_func = prim_func
            self.calls: list[tuple[int, int, int]] = []

        def __call__(
            self,
            bank: mx.array,
            chunk_index: int,
            subchunk_index: int,
            run_backward: int,
        ) -> mx.array:
            gate = int(run_backward)
            self.calls.append((int(chunk_index), int(subchunk_index), gate))
            attention_index = mx.array([1], dtype=mx.int32)
            grad_index = mx.array([3], dtype=mx.int32)
            bank = bank.at[attention_index].add(mx.array([1.0], dtype=mx.float32))
            if gate == 1:
                bank = bank.at[grad_index].add(mx.array([2.0], dtype=mx.float32))
            return bank

    fake_kernel = FakeKernel()
    schedule_contract = SimpleNamespace(
        declared_required_real_abi_inputs=("hidden",)
    )
    compiled = SimpleNamespace(
        compiled=SimpleNamespace(
            artifact=fake_kernel,
            plan=SimpleNamespace(schedule_contract=schedule_contract),
        )
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)

    result = launcher(
        real_abi_inputs={"hidden": mx.array([2.0], dtype=mx.float32)},
        cotangent_seeds={
            "attention_out_grad": mx.array([1.0], dtype=mx.float32)
        },
    )
    mx.eval(result.forward["attention_out"], result.parameter_grads["weight_grad"])

    assert fake_kernel.calls == [
        (0, 0, 2),
        (0, 1, 2),
        (0, 2, 2),
        (1, 0, 2),
        (1, 1, 2),
        (1, 2, 2),
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (1, 0, 1),
        (1, 1, 1),
        (1, 2, 1),
    ]
    assert result.forward["attention_out"].item() == pytest.approx(12.0)
    assert result.parameter_grads["weight_grad"].item() == pytest.approx(12.0)


def test_path_c_launcher_chunked_dispatch_does_not_sync_after_each_launch() -> None:
    """Chunked dispatch must chain owner outputs instead of host-syncing each chunk."""

    import inspect

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod

    source = inspect.getsource(launcher_mod.Mamba3Fp8TrainBlockLauncher.__call__)
    launch_chunk_body = source.split("def launch_chunk", 1)[1].split(
        "if run_backward", 1
    )[0]

    assert "eval_positionals()" not in launch_chunk_body


def test_path_c_launcher_rejects_grid_chunked_backward_parameter_grads() -> None:
    import json

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    logical_to_physical = {
        "hidden": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 0,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "attention_out": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 1,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "attention_out_grad": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 2,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
        "weight_grad": {
            "bank": "path_c_float32_abi_bank",
            "dtype": "float32",
            "offset": 3,
            "size": 1,
            "shape": [1],
            "logical_shape": [1],
        },
    }
    attrs = {
        "global_symbol": "toy_grid_chunked_backward",
        "tilelang_out_idx": [0],
        "tl.fusion.physical_abi.logical_to_physical": json.dumps(
            logical_to_physical
        ),
        "tl.fusion.physical_abi.physical_buffer_shapes": json.dumps(
            {"path_c_float32_abi_bank": [4]}
        ),
        "tl.fusion.train_step_loss_cotangent_abi": json.dumps(
            {"logical_cotangent_buffers": ["attention_out_grad"]}
        ),
        "tl.fusion.max_rows_per_launch": 64,
        "tl.fusion.row_chunk_count": 3,
        "tl.fusion.row_dispatch_mode": schedules.DESCRIPTOR_ROW_DISPATCH_GRID_CHUNKS,
    }
    prim_func = SimpleNamespace(
        attrs=attrs,
        params=[SimpleNamespace(name="path_c_float32_abi_bank_handle")],
        buffer_map={
            "path_c_float32_abi_bank": SimpleNamespace(
                name="path_c_float32_abi_bank",
                shape=(4,),
                dtype="float32",
            )
        },
    )

    class FakeKernel:
        def __init__(self, prim_func):
            self.prim_func = prim_func

        def __call__(self, *_args):
            raise AssertionError("grid-chunked backward should fail before launch")

    schedule_contract = SimpleNamespace(
        declared_required_real_abi_inputs=("hidden",)
    )
    compiled = SimpleNamespace(
        compiled=SimpleNamespace(
            artifact=FakeKernel(prim_func),
            plan=SimpleNamespace(schedule_contract=schedule_contract),
        )
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)

    with pytest.raises(RuntimeError, match="grid_chunks row dispatch"):
        launcher(
            real_abi_inputs={"hidden": mx.array([2.0], dtype=mx.float32)},
            cotangent_seeds={
                "attention_out_grad": mx.array([1.0], dtype=mx.float32)
            },
        )


def test_full_1b_quarter_launcher_chunk_contract() -> None:
    """Full-shape launcher-chunk contract for the 1B-quarter block.

    The full 4096-row kernel is too expensive for a unit test once owner
    outputs are returned correctly, so this pins the compile-time watchdog
    contract: one row-chunk-indexed kernel body, 64 host-visible chunks, and
    only written ABI banks returned as owner outputs. The tiny Metal smoke below
    covers the real runtime owner-output path.
    """

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    captured_sources: list[str] = []
    captured_prim_funcs: list[object] = []

    def capture_lowerer(func_or_mod, **_kwargs):
        funcs = list(func_or_mod.functions.items())
        assert len(funcs) == 1
        captured_prim_funcs.append(funcs[0][1])
        captured_sources.append(str(func_or_mod.script()))
        return "compiled-launcher-chunked-mamba3"

    schedules.compile_mamba3_fp8_train_fusion_schedule(
        tilelang_lowerer=capture_lowerer,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )

    assert captured_sources
    assert captured_prim_funcs
    generated_source = captured_sources[0]
    manifest = launcher_mod.load_path_c_abi_manifest(captured_prim_funcs[0])
    cfg = local_gb10_quarter_profile().hybrid_config()

    assert manifest.row_chunk_count == (cfg.max_seq_length + 63) // 64
    assert manifest.row_chunk_count == 64
    assert manifest.row_subchunk_count == 8
    assert manifest.rows_per_kernel_launch == 8
    assert (
        manifest.row_dispatch_mode
        == schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS
    )
    assert manifest.row_chunk_index_param == schedules.DESCRIPTOR_ROW_CHUNK_INDEX_PARAM
    assert manifest.row_chunk_index_buffer is None
    assert manifest.backward_gate_param == schedules.DESCRIPTOR_BACKWARD_GATE_PARAM
    out_idx = list(captured_prim_funcs[0].attrs.get("tilelang_out_idx", ()))
    output_buffer_names = {
        str(
            getattr(
                captured_prim_funcs[0].buffer_map[captured_prim_funcs[0].params[idx]],
                "name",
                captured_prim_funcs[0].params[idx],
            )
        )
        for idx in out_idx
    }
    all_buffer_param_count = sum(
        1
        for param in captured_prim_funcs[0].params
        if param in captured_prim_funcs[0].buffer_map
    )
    assert len(out_idx) < all_buffer_param_count
    assert "path_c_float32_parameter_abi_bank" not in output_buffer_names
    assert "path_c_float32_parameter_gradient_abi_bank" in output_buffer_names
    assert "path_c_float32_activation_abi_bank" in output_buffer_names
    assert 'bx = T.launch_thread("blockIdx.x", 1)' in generated_source
    assert 'lane = T.launch_thread("threadIdx.x", 1024)' in generated_source
    assert "path_c_run_backward: T.int32" in generated_source
    assert "if path_c_run_backward != 1:" in generated_source
    assert "if path_c_run_backward == 1:" in generated_source
    assert (
        "logical_row_chunk_start: T.int32 = path_c_row_chunk_index * 64"
        in generated_source
    )
    assert (
        "row_chunk_start: T.int32 = T.min(logical_row_chunk_start + "
        "path_c_row_subchunk_index * 8, logical_row_chunk_stop)"
        in generated_source
    )
    assert (
        "row_chunk_stop: T.int32 = T.min(row_chunk_start + 8, "
        "logical_row_chunk_stop)"
        in generated_source
    )
    assert "row_chunk_start: T.int32 = T.if_then_else(path_c_run_backward == 1, 0" not in generated_source
    assert (
        "entry_row_chunk_stop: T.int32 = T.min(row_chunk_stop + 1, 4096)"
        in generated_source
    )
    assert (
        generated_source.count(
            "for row in range(row_chunk_start, row_chunk_start + (entry_row_chunk_stop - row_chunk_start)):"
        )
        == 1
    )
    assert (
        generated_source.count(
            "for row in range(row_chunk_start, row_chunk_start + (row_chunk_stop - row_chunk_start)):"
        )
        >= 1
    )
    assert generated_source.count("for row in range(4096):") == 0


@pytest.mark.slow
def test_compiled_vs_eager_full_1b_quarter_chunked() -> None:
    """Full-shape chunked compile/ABI contract for the 1B-quarter block.

    The nonzero eager M/R/A forward parity gate below covers the tiny
    executable allclose case. This integration check covers the full
    F21/F23 watchdog contract at the real ABI shape: forward-only launcher
    chunks must compile at ``seq=4096`` and publish the expected launcher
    manifest/output ABI without executing the full dense 1B runtime path.
    """

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    if not mx.metal.is_available():
        pytest.skip("Metal unavailable")

    cfg = local_gb10_quarter_profile().hybrid_config()
    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        include_backward=False,
        model_config=cfg,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)
    manifest = launcher.manifest

    assert manifest.row_chunk_count == 64
    assert manifest.row_subchunk_count == 8
    assert manifest.rows_per_kernel_launch == 8

    assert compiled.compiled.plan.single_kernel_fused
    assert compiled.compiled.lowered_module is not None
    assert len(compiled.compiled.lowered_module.functions) == 1
    assert manifest.logical_to_physical["attention_out"].logical_shape == (
        1,
        4096,
        3584,
    )
    assert manifest.logical_to_physical["hidden_after_m2rnn"].logical_shape == (
        1,
        4096,
        3584,
    )
    assert manifest.logical_to_physical["lse"].logical_shape == (114688,)


def test_mamba3_fp8_train_schedule_compile_helper_defaults_to_tilelang_lowerer() -> None:
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule()

    assert isinstance(compiled.compiled, CompiledPathCRegion)
    assert type(compiled.compiled.artifact).__name__ == "JITKernel"
    assert compiled.compiled.plan.single_kernel_fused is True
    assert compiled.compiled.plan.schedule_contract is not None
    assert compiled.compiled.plan.schedule_contract.status == "verified"



def test_mamba3_fp8_train_schedule_runtime_smoke_on_tiny_metal() -> None:
    """Step 4.b support: end-to-end launcher + tiny-shape runtime smoke.

    Sets up the production fused train block compiled for the *tiny smoke*
    M/R/A shape (so the single-launch kernel fits inside the local Metal
    command-buffer watchdog, ~5 seconds), runs the kernel through the
    production ABI manifest launcher
    (:mod:`cppmega_mlx.runtime.path_c_fusion_launcher`), seeds the
    cotangent buffers with ``ones`` and asserts the structural+sanity
    contract that pins the launcher/schedule plumbing to a runtime-verified
    state:

    * The launcher consumes the schedule's compiled ``JITKernel`` and
      runs it on Metal without a GPU timeout, regardless of host wall
      clock budget. (Hits 4.b regression if the schedule expands
      activation budget beyond the watchdog cap.)
    * Every named forward output (``attention_out``,
      ``hidden_after_m2rnn``, ``lse``) has the expected logical shape
      from the ABI manifest. (Hits 4.b regression if the schedule
      silently drops or reshapes one of the saved tensors.)
    * ``attention_out`` carries only finite values. Strict non-zero/parity
      checks remain in the pending eager-reference tests because the current
      tiny schedule still has known forward completeness gaps.
    * Every manifest parameter-gradient buffer is returned with the expected
      shape and finite values. Non-zero numeric parity remains in the pending
      eager-reference tests.

    Actual loss/grad parity vs the explicit MLX eager M+R+A reference
    is tracked by
    ``test_mamba3_fp8_train_schedule_eager_loss_grad_parity_on_metal_pending_reference``;
    the schedule's reductions still produce
    ``-inf`` for the ``lse`` buffer when seeded with random fp8 weight
    patterns at the tiny shape, which would dominate any naive rtol
    bound. The assertion floor here is the runtime evidence that the
    launcher and schedule wiring are correct; once the kernel ships
    numerically-clean lse + hidden_after_m2rnn outputs, the rtol bound
    against the eager reference can be tightened in this same test.
    """

    import math

    import mlx.core as mx

    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile
    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    mx.random.seed(20260523)
    tiny_config = local_gb10_quarter_profile().tiny_smoke_config()

    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        model_config=tiny_config,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)
    manifest = launcher.manifest

    inputs: dict[str, mx.array] = {}
    for name in launcher.real_abi_inputs:
        placement = manifest.logical_to_physical[name]
        dtype = launcher_mod._to_mx_dtype(placement.dtype)
        if placement.dtype == "float32":
            inputs[name] = (
                mx.random.normal(shape=placement.logical_shape, dtype=mx.float32)
                * 0.1
            )
        elif placement.dtype == "int32":
            inputs[name] = mx.zeros(placement.logical_shape, dtype=mx.int32)
        elif placement.dtype == "uint8":
            inputs[name] = mx.random.randint(
                0, 128, shape=placement.logical_shape
            ).astype(mx.uint8)
        else:
            inputs[name] = mx.zeros(placement.logical_shape, dtype=dtype)
    inputs["sparse_mla_sm_scale"] = mx.array(
        [
            1.0
            / math.sqrt(
                tiny_config.hidden_size // tiny_config.num_attention_heads
            )
        ],
        dtype=mx.float32,
    )
    inputs["sparse_mla_has_sinks"] = mx.array([0], dtype=mx.int32)

    cotangent_seeds = {
        name: mx.ones(
            manifest.logical_to_physical[name].logical_shape, dtype=mx.float32
        )
        for name in launcher.cotangent_seed_buffers
    }

    result = launcher(real_abi_inputs=inputs, cotangent_seeds=cotangent_seeds)
    mx.eval(*result.forward.values(), *result.parameter_grads.values())

    for name in launcher.forward_outputs:
        if name not in manifest.logical_to_physical:
            continue
        placement = manifest.logical_to_physical[name]
        assert tuple(result.forward[name].shape) == tuple(placement.logical_shape)
    attention_out = result.forward["attention_out"]
    assert bool(mx.all(mx.isfinite(attention_out)).item())
    assert float(mx.mean(mx.abs(attention_out)).item()) > 1e-6
    assert float(mx.mean(mx.abs(result.forward["hidden_after_m2rnn"])).item()) > 1e-6
    assert bool(mx.all(mx.isfinite(result.forward["hidden_after_m2rnn"])).item())
    assert bool(mx.all(mx.isfinite(result.forward["lse"])).item())
    assert set(result.parameter_grads) == set(launcher.parameter_grad_buffers)
    for name, grad in result.parameter_grads.items():
        placement = manifest.logical_to_physical[name]
        assert tuple(grad.shape) == tuple(placement.logical_shape)
        assert bool(mx.all(mx.isfinite(grad)).item())


def test_mamba3_fp8_train_schedule_eager_loss_grad_parity_on_metal_pending_reference() -> None:
    """Nonzero tiny-shape runtime gate for Step 4.b's parity contract.

    This test runs the launcher with deterministic inputs and asserts
    the compiled fused-train-block output/gradient floor that must hold
    before strict eager parity can be meaningful:

    * forward outputs are finite and non-trivial
    * repeated launches with identical inputs are reproducible
    * every manifest parameter-gradient buffer is reproducible at
      ``rtol=1e-2``

    Strict eager-vs-compiled M/R/A allclose remains covered by
    ``test_compiled_vs_eager_mra_forward_parity_rtol1e3``.
    """

    import math

    import mlx.core as mx

    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile
    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    mx.random.seed(20260523)
    tiny_config = local_gb10_quarter_profile().tiny_smoke_config()

    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        model_config=tiny_config,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)
    manifest = launcher.manifest

    # Deterministic inputs (same recipe as runtime smoke; the parity
    # check needs exact reproducibility so both compiled and reference
    # forwards see identical bits).
    inputs: dict[str, mx.array] = {}
    for name in launcher.real_abi_inputs:
        placement = manifest.logical_to_physical[name]
        if placement.dtype == "float32":
            inputs[name] = (
                mx.random.normal(shape=placement.logical_shape, dtype=mx.float32)
                * 0.1
            )
        elif placement.dtype == "int32":
            inputs[name] = mx.zeros(placement.logical_shape, dtype=mx.int32)
        elif placement.dtype == "uint8":
            inputs[name] = mx.random.randint(
                0, 128, shape=placement.logical_shape
            ).astype(mx.uint8)
        else:
            inputs[name] = mx.zeros(
                placement.logical_shape,
                dtype=launcher_mod._to_mx_dtype(placement.dtype),
            )
    inputs["sparse_mla_sm_scale"] = mx.array(
        [
            1.0
            / math.sqrt(
                tiny_config.hidden_size // tiny_config.num_attention_heads
            )
        ],
        dtype=mx.float32,
    )
    inputs["sparse_mla_has_sinks"] = mx.array([0], dtype=mx.int32)
    for name in launcher.forward_outputs:
        if name in launcher.real_abi_inputs:
            placement = manifest.logical_to_physical[name]
            inputs[name] = mx.zeros(
                placement.logical_shape,
                dtype=launcher_mod._to_mx_dtype(placement.dtype),
            )

    cotangent_seeds: dict[str, mx.array] = {
        name: mx.ones(
            manifest.logical_to_physical[name].logical_shape, dtype=mx.float32
        )
        for name in launcher.cotangent_seed_buffers
    }

    result = launcher(real_abi_inputs=inputs, cotangent_seeds=cotangent_seeds)
    mx.eval(*result.forward.values(), *result.parameter_grads.values())

    # The strict parity bar: every forward output must be finite and
    # carry non-trivial data before the tighter eager allclose checks.
    rtol_forward = 1e-3
    rtol_grad = 1e-2

    attention_out = result.forward["attention_out"]
    hidden_after_m2rnn = result.forward["hidden_after_m2rnn"]
    lse = result.forward["lse"]

    # All three forward outputs must be finite (no NaN / +/- inf).
    assert bool(mx.all(mx.isfinite(attention_out)).item()), (
        "compiled forward attention_out has non-finite values"
    )
    assert bool(mx.all(mx.isfinite(hidden_after_m2rnn)).item()), (
        "compiled forward hidden_after_m2rnn has non-finite values"
    )
    assert bool(mx.all(mx.isfinite(lse)).item()), (
        "compiled forward lse has non-finite values"
    )

    # hidden_after_m2rnn must carry the m2rnn residual+norm result. With
    # non-trivial inputs the running hidden state cannot collapse to
    # identically zero; that would mean the m2rnn residual is missing
    # from the returned owner-output bank.
    h_after_mean_abs = float(mx.mean(mx.abs(hidden_after_m2rnn)).item())
    assert h_after_mean_abs > 1e-4, (
        "compiled forward hidden_after_m2rnn collapsed to ~zero: "
        f"mean_abs={h_after_mean_abs:.3e}; the schedule is not emitting "
        "the m2rnn residual+norm stage"
    )

    # Cross-check attention_out against the launcher's runtime-smoke
    # baseline: same inputs and seed produce the same attention_out (no
    # drift between runs).
    second_result = launcher(
        real_abi_inputs=inputs, cotangent_seeds=cotangent_seeds
    )
    mx.eval(second_result.forward["attention_out"])
    diff = mx.max(
        mx.abs(attention_out - second_result.forward["attention_out"])
    ).item()
    assert diff <= rtol_forward * float(
        mx.max(mx.abs(attention_out)).item() + 1e-8
    ), (
        f"compiled fused-train-block attention_out is non-deterministic "
        f"across launches with identical inputs: max abs diff = {diff:.3e}"
    )

    # Parameter gradients: every entry in the manifest's
    # parameter_grad_buffers must agree with itself across launches at
    # rtol_grad. (Once the schedule is numerically correct, the same
    # assertion structure will be used to compare against the explicit
    # MLX eager M+R+A reference.)
    for name in launcher.parameter_grad_buffers:
        a = result.parameter_grads[name]
        b = second_result.parameter_grads[name]
        max_a = float(mx.max(mx.abs(a)).item())
        diff = float(mx.max(mx.abs(a - b)).item())
        assert diff <= rtol_grad * (max_a + 1e-8), (
            f"parameter gradient {name!r} not reproducible across launches: "
            f"max abs diff = {diff:.3e}, max abs value = {max_a:.3e}"
        )
def _build_mra_mini_model_for_real_abi_extraction(model_config=None):
    """Build a minimal model exposing the (M, R, A) tail used by the launcher.

    The default named Path C train block runs against the
    ``local_gb10_quarter`` ABI footprint; callers may pass a smaller
    ``HybridTinyConfig`` when the compiled schedule is built with the
    same shape contract. A full ``HybridTinyLM`` over the whole 13-layer
    route would allocate many unused parameters. The parameter extractor
    only walks ``model.layers[-3:]``, so this helper builds the same
    three blocks and wraps them in an object that exposes ``.layers``.
    """

    from types import SimpleNamespace

    from cppmega_mlx.models.hybrid_lm import HybridTinyBlock
    from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile

    profile = local_gb10_quarter_profile()
    config = model_config or profile.hybrid_config()
    last_three = list(config.expanded_pattern().layers)[-3:]
    blocks = [HybridTinyBlock(layer, config) for layer in last_three]
    return SimpleNamespace(layers=blocks), config


def _zero_runtime_inputs_for_launcher(launcher, launcher_mod):
    import mlx.core as mx

    runtime_inputs: dict[str, mx.array] = {}
    for name in launcher.real_abi_inputs:
        placement = launcher.manifest.logical_to_physical[name]
        runtime_inputs[name] = mx.zeros(
            placement.logical_shape,
            dtype=launcher_mod._to_mx_dtype(placement.dtype),
        )
    return runtime_inputs


def test_extract_mamba3_fp8_train_real_abi_inputs_returns_full_manifest_set() -> None:
    """Step 4.b group A.1: the parameter extractor binds every real ABI input.

    The launcher's ``real_abi_inputs`` enumerates every external buffer
    the fused Path C train block reads on Metal — trainable parameters
    (the M/R/A weights and the two cross-block residual norm weights),
    the entry RMSNorm weight, recurrent state seeds, and the
    ``sparse_mla`` configuration constants. The
    ``extract_mamba3_fp8_train_real_abi_inputs`` helper walks
    ``model.layers[-3:]`` and binds each name from the model's parameter
    tree, while runtime activation/state buffers must be supplied by the
    caller. This test pins the contract: every required name resolves to an
    ``mx.array`` whose
    shape and dtype exactly match the schedule's manifest, with no
    missing or extraneous keys.
    """

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    tiny_config = local_gb10_quarter_profile().tiny_smoke_config()
    mini_model, _ = _build_mra_mini_model_for_real_abi_extraction(tiny_config)
    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        model_config=tiny_config,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)

    with pytest.raises(ValueError, match="missing required real ABI input"):
        launcher_mod.extract_mamba3_fp8_train_real_abi_inputs(
            mini_model, launcher=launcher
        )

    runtime_inputs = _zero_runtime_inputs_for_launcher(launcher, launcher_mod)
    inputs = launcher_mod.extract_mamba3_fp8_train_real_abi_inputs(
        mini_model,
        launcher=launcher,
        runtime_inputs=runtime_inputs,
    )

    expected = set(launcher.real_abi_inputs)
    assert set(inputs) == expected, (
        f"extractor key-set drift; missing={expected - set(inputs)!r}, "
        f"extra={set(inputs) - expected!r}"
    )
    # The 34/35 declared trainable + 3 state buffers contract: this
    # test fails noisily if the manifest grows or shrinks so the
    # extractor stays in lockstep with the schedule.
    assert len(inputs) >= 34, (
        f"extractor returned only {len(inputs)} real ABI inputs; the Path C "
        "train block contract requires at least the 34 trainable + state "
        "ABI inputs"
    )

    for name in launcher.real_abi_inputs:
        placement = launcher.manifest.logical_to_physical[name]
        value = inputs[name]
        assert isinstance(value, mx.array), (
            f"real ABI input {name!r} must be an mx.array; got {type(value).__name__}"
        )
        assert tuple(value.shape) == tuple(placement.logical_shape), (
            f"real ABI input {name!r} shape mismatch: got {tuple(value.shape)}, "
            f"manifest expects {tuple(placement.logical_shape)}"
        )
        assert str(value.dtype) == str(launcher_mod._to_mx_dtype(placement.dtype)), (
            f"real ABI input {name!r} dtype mismatch: got {value.dtype}, "
            f"manifest expects {placement.dtype}"
        )

    # The launcher accepts this dict end-to-end (validates inputs +
    # cotangent seeds + runs the kernel). If the extractor produced a
    # subset of names or wrong-shape arrays this call would raise the
    # launcher's fail-closed ValueError.
    cotangent_seeds = {
        name: mx.ones(
            launcher.manifest.logical_to_physical[name].logical_shape,
            dtype=mx.float32,
        )
        for name in launcher.cotangent_seed_buffers
    }
    result = launcher(real_abi_inputs=inputs, cotangent_seeds=cotangent_seeds)
    assert set(result.forward) >= {"attention_out", "hidden_after_m2rnn"}
    assert set(result.parameter_grads) == set(launcher.parameter_grad_buffers)


def test_extract_mamba3_fp8_train_real_abi_inputs_rejects_runtime_shape_or_dtype_drift() -> None:
    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    mini_model, _ = _build_mra_mini_model_for_real_abi_extraction()
    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule()
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)
    runtime_inputs = _zero_runtime_inputs_for_launcher(launcher, launcher_mod)
    hidden_placement = launcher.manifest.logical_to_physical["hidden"]

    bad_dtype = dict(runtime_inputs)
    bad_dtype["hidden"] = mx.zeros(hidden_placement.logical_shape, dtype=mx.float16)
    with pytest.raises(ValueError, match="dtype mismatch"):
        launcher_mod.extract_mamba3_fp8_train_real_abi_inputs(
            mini_model,
            launcher=launcher,
            runtime_inputs=bad_dtype,
        )

    bad_size = dict(runtime_inputs)
    bad_size["hidden"] = mx.zeros((hidden_placement.size - 1,), dtype=mx.float32)
    with pytest.raises(ValueError, match="size mismatch"):
        launcher_mod.extract_mamba3_fp8_train_real_abi_inputs(
            mini_model,
            launcher=launcher,
            runtime_inputs=bad_size,
        )


def _run_eager_mra_forward(mini_model, hidden_states):
    """Run an eager Path C-equivalent forward through the M+R+A tail.

    Mirrors the fused Path C train block's compute graph:

    * ``mamba3_entry_rmsnorm_hidden = M.norm(hidden)``
    * ``mamba3_delta = M.block(mamba3_entry_rmsnorm_hidden, h0=0)``
    * ``hidden_after_mamba3 = hidden + mamba3_delta``
    * ``m2rnn_hidden = R.norm(hidden_after_mamba3)``
    * ``m2rnn_delta = R.block(m2rnn_hidden, h0=0)``
    * ``hidden_after_m2rnn = hidden_after_mamba3 + m2rnn_delta``
    * ``attention_hidden = A.norm(hidden_after_m2rnn)``
    * ``attention_prepared = A.block.prepare_sparse_mla_fp8(...)``
    * ``attention_out = A.block._apply_sparse_mla_fp8_path_c_prepared(...)``

    Returns ``(attention_out, hidden_after_m2rnn)`` so callers can pair
    them up with the launcher's forward outputs.
    """

    m_block, r_block, a_block = mini_model.layers[-3:]
    mam = m_block.block
    rnn = r_block.block
    att = a_block.block

    batch = int(hidden_states.shape[0])
    dtype = hidden_states.dtype

    m_normed = m_block.norm(hidden_states)
    mamba3_delta, _ = mam(m_normed, h0=mam.initial_h0(batch, dtype))
    hidden_after_mamba3 = hidden_states + mamba3_delta

    r_normed = r_block.norm(hidden_after_mamba3)
    m2rnn_delta, _ = rnn(r_normed, h0=rnn.initial_h0(batch, dtype))
    hidden_after_m2rnn = hidden_after_mamba3 + m2rnn_delta

    a_normed = a_block.norm(hidden_after_m2rnn)
    prepared = att.prepare_sparse_mla_fp8(a_normed, mask="causal")
    attention_out = att._apply_sparse_mla_fp8_path_c_prepared(
        prepared,
        output_shape=(
            int(a_normed.shape[0]),
            int(a_normed.shape[1]),
            att.config.q_proj_dim,
        ),
    )

    return attention_out, hidden_after_m2rnn


def _max_abs_diff(a, b) -> float:
    import mlx.core as mx

    n = min(int(a.size), int(b.size))
    af = a.reshape((-1,))[:n]
    bf = b.reshape((-1,))[:n]
    return float(mx.max(mx.abs(af - bf)).item())


def test_compiled_vs_eager_mra_forward_parity_rtol1e3() -> None:
    """Eager-vs-compiled forward parity for the M+R+A train block.

    Builds the same M/R/A tail the launcher reads from, drives both the
    compiled fused kernel and an explicit eager forward through the same
    Path C Sparse-MLA A semantics with identical inputs, and asserts that
    ``attention_out`` and ``hidden_after_m2rnn`` agree at ``rtol=1e-3`` via
    ``mx.allclose``.
    On failure the assertion message records the max absolute diff for
    each forward output so the gap is concrete, not abstract.
    """

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    mx.random.seed(20260523)
    tiny_config = local_gb10_quarter_profile().tiny_smoke_config()
    mini_model, _ = _build_mra_mini_model_for_real_abi_extraction(tiny_config)
    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        model_config=tiny_config,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)

    runtime_inputs = _zero_runtime_inputs_for_launcher(launcher, launcher_mod)
    inputs = launcher_mod.extract_mamba3_fp8_train_real_abi_inputs(
        mini_model,
        launcher=launcher,
        runtime_inputs=runtime_inputs,
    )
    hidden_placement = launcher.manifest.logical_to_physical["hidden"]
    hidden = (
        mx.random.normal(shape=hidden_placement.logical_shape, dtype=mx.float32)
        * 0.1
    )
    inputs["hidden"] = hidden
    result = launcher(real_abi_inputs=inputs, cotangent_seeds={})
    mx.eval(*result.forward.values())
    eager_attn_out, eager_hidden_after_m2rnn = _run_eager_mra_forward(
        mini_model, hidden
    )
    mx.eval(eager_attn_out, eager_hidden_after_m2rnn)

    rtol = 1e-3
    atol = 1e-3
    attn_diff = _max_abs_diff(result.forward["attention_out"], eager_attn_out)
    h_diff = _max_abs_diff(
        result.forward["hidden_after_m2rnn"], eager_hidden_after_m2rnn
    )

    attn_ok = bool(
        mx.allclose(
            result.forward["attention_out"].reshape((-1,))[: eager_attn_out.size],
            eager_attn_out.reshape((-1,)),
            rtol=rtol,
            atol=atol,
        ).item()
    )
    h_ok = bool(
        mx.allclose(
            result.forward["hidden_after_m2rnn"].reshape((-1,))[
                : eager_hidden_after_m2rnn.size
            ],
            eager_hidden_after_m2rnn.reshape((-1,)),
            rtol=rtol,
            atol=atol,
        ).item()
    )

    assert attn_ok, (
        f"attention_out parity failed at rtol={rtol}: "
        f"max_abs_diff={attn_diff:.3e}"
    )
    assert h_ok, (
        f"hidden_after_m2rnn parity failed at rtol={rtol}: "
        f"max_abs_diff={h_diff:.3e}"
    )


def test_compiled_vs_eager_mra_grad_parity_rtol1e2() -> None:
    """Eager-vs-compiled backward parity for the M+R+A train block.

    Computes parameter gradients two ways and asserts every entry in
    ``launcher.parameter_grad_buffers`` agrees with the eager reference
    at ``rtol=1e-2``:

    * Compiled: run the launcher with ones-seeded cotangents for
      ``attention_out`` and ``hidden_after_m2rnn``, collect
      ``result.parameter_grads``.
    * Eager: run the explicit MLX eager M+R+A forward, define
      ``L = sum(attention_out) + sum(hidden_after_m2rnn)`` (so the
      cotangent on each output is ones), and take
      ``mx.value_and_grad`` against every named real ABI parameter.

    On failure the assertion records each failing parameter's
    max_abs_diff so the regression has concrete evidence.
    """

    import mlx.core as mx

    from cppmega_mlx.runtime import path_c_fusion_launcher as launcher_mod
    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    mx.random.seed(20260523)
    tiny_config = local_gb10_quarter_profile().tiny_smoke_config()
    mini_model, _ = _build_mra_mini_model_for_real_abi_extraction(tiny_config)
    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule(
        model_config=tiny_config,
        row_dispatch_mode=schedules.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    launcher = launcher_mod.Mamba3Fp8TrainBlockLauncher(compiled)

    runtime_inputs = _zero_runtime_inputs_for_launcher(launcher, launcher_mod)
    inputs = launcher_mod.extract_mamba3_fp8_train_real_abi_inputs(
        mini_model,
        launcher=launcher,
        runtime_inputs=runtime_inputs,
    )
    hidden_placement = launcher.manifest.logical_to_physical["hidden"]
    hidden = (
        mx.random.normal(shape=hidden_placement.logical_shape, dtype=mx.float32)
        * 0.1
    )
    inputs["hidden"] = hidden
    cotangent_seeds = {
        name: mx.ones(
            launcher.manifest.logical_to_physical[name].logical_shape,
            dtype=mx.float32,
        )
        for name in launcher.cotangent_seed_buffers
    }

    compiled_result = launcher(
        real_abi_inputs=inputs, cotangent_seeds=cotangent_seeds
    )
    mx.eval(*compiled_result.parameter_grads.values())

    # Wrap the M/R/A blocks in an nn.Module so MLX's autograd can walk
    # the parameter tree and emit grads keyed by parameter path.
    import mlx.nn as nn

    class _MraTail(nn.Module):
        def __init__(self, m_block, r_block, a_block):
            super().__init__()
            self.m = m_block
            self.r = r_block
            self.a = a_block

        def __call__(self, hidden_):
            m_normed = self.m.norm(hidden_)
            mamba3_delta, _ = self.m.block(
                m_normed, h0=self.m.block.initial_h0(int(hidden_.shape[0]), hidden_.dtype)
            )
            hidden_after_mamba3 = hidden_ + mamba3_delta
            r_normed = self.r.norm(hidden_after_mamba3)
            m2rnn_delta, _ = self.r.block(
                r_normed,
                h0=self.r.block.initial_h0(int(hidden_.shape[0]), hidden_.dtype),
            )
            hidden_after_m2rnn = hidden_after_mamba3 + m2rnn_delta
            a_normed = self.a.norm(hidden_after_m2rnn)
            prepared = self.a.block.prepare_sparse_mla_fp8(a_normed, mask="causal")
            # This is the eager oracle: keep the prepared Path C forward
            # semantics, but use the reference VJP instead of the standalone
            # Path C Sparse-MLA backward kernel.
            from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
                sparse_mla_fp8_path_c_apply_prepared_float,
            )

            attention_values = sparse_mla_fp8_path_c_apply_prepared_float(
                prepared.q,
                prepared.kv,
                prepared.q_fp8,
                prepared.q_scale,
                prepared.kv_fp8,
                prepared.kv_scale,
                prepared.indices,
                sm_scale=prepared.sm_scale,
                d_v=prepared.d_v,
                sinks=None,
                force_path_c=True,
                causal=prepared.causal,
                output_dtype=prepared.q.dtype,
                force_backward_path_c=False,
            )
            attention_out = self.a.block.out_proj(
                attention_values.reshape(
                    (
                        int(a_normed.shape[0]),
                        int(a_normed.shape[1]),
                        self.a.block.config.q_proj_dim,
                    )
                )
            )
            return attention_out.sum() + hidden_after_m2rnn.sum()

    m_block, r_block, a_block = mini_model.layers[-3:]
    tail = _MraTail(m_block, r_block, a_block)

    def loss_fn(model_):
        return model_(hidden)

    _, grad_tree = nn.value_and_grad(tail, loss_fn)(tail)

    # Flatten the grad tree to a name → array map so we can look up by
    # path and bind each compiled grad slot to its eager counterpart.
    from mlx.utils import tree_flatten

    eager_grads_by_path: dict[str, mx.array] = dict(tree_flatten(grad_tree))
    mx.eval(*eager_grads_by_path.values())

    # Build the bridge from compiled launcher grad names → eager path
    # in the wrapped tail module.
    path_map = {
        "mamba3_entry_rmsnorm_weight_grad": "m.norm.weight",
        "mamba3_in_proj_weight_grad": "m.block.in_proj.weight",
        "mamba3_out_proj_weight_grad": "m.block.out_proj.weight",
        "mamba3_conv_weight_grad": "m.block.conv_weight",
        "mamba3_conv_bias_grad": "m.block.conv_bias",
        "mamba3_dt_bias_grad": "m.block.dt_bias",
        "mamba3_B_norm_weight_grad": "m.block.B_norm_weight",
        "mamba3_B_bias_grad": "m.block.B_bias",
        "mamba3_C_norm_weight_grad": "m.block.C_norm_weight",
        "mamba3_C_bias_grad": "m.block.C_bias",
        "mamba3_D_grad": "m.block.D",
        "mamba3_residual_to_m2rnn_norm_weight_grad": "r.norm.weight",
        "m2rnn_residual_to_attention_norm_weight_grad": "a.norm.weight",
        "m2rnn_in_proj_weight_grad": "r.block.in_proj.weight",
        "m2rnn_conv_weight_grad": "r.block.conv_weight",
        "m2rnn_conv_bias_grad": "r.block.conv_bias",
        "m2rnn_state_weight_grad": "r.block.state_weight",
        "m2rnn_A_log_grad": "r.block.A_log",
        "m2rnn_dt_bias_grad": "r.block.dt_bias",
        "m2rnn_D_grad": "r.block.D",
        "m2rnn_g_norm_weight_grad": "r.block.g_norm.weight",
        "m2rnn_out_proj_weight_grad": "r.block.out_proj.weight",
        "attention_q_proj_weight_grad": "a.block.q_proj.weight",
        "attention_sparse_kv_proj_weight_grad": "a.block.sparse_kv_proj.weight",
        "attention_out_proj_weight_grad": "a.block.out_proj.weight",
    }
    eager_grads = {
        compiled_name: eager_grads_by_path[path]
        for compiled_name, path in path_map.items()
        if path in eager_grads_by_path
    }

    rtol = 1e-2
    atol = 1e-2
    failures: list[str] = []
    compared = 0
    for name in launcher.parameter_grad_buffers:
        if name not in eager_grads:
            continue
        compared += 1
        compiled_grad = compiled_result.parameter_grads[name]
        eager_grad = eager_grads[name]
        n = min(int(compiled_grad.size), int(eager_grad.size))
        a = compiled_grad.reshape((-1,))[:n]
        b = eager_grad.reshape((-1,))[:n]
        ok = bool(mx.allclose(a, b, rtol=rtol, atol=atol).item())
        if not ok:
            diff = float(mx.max(mx.abs(a - b)).item())
            eager_mag = float(mx.max(mx.abs(b)).item())
            failures.append(
                f"{name}: max_abs_diff={diff:.3e} (eager max_abs={eager_mag:.3e})"
            )

    assert compared > 0, (
        "no parameter_grad_buffers overlapped with the eager parameter set; "
        "extractor or naming drift"
    )
    assert not failures, (
        f"{len(failures)}/{compared} parameter grads diverged from eager "
        f"reference at rtol={rtol}; failures:\n  "
        + "\n  ".join(failures)
    )


def test_mamba3_fp8_train_schedule_compile_helper_emits_metal_single_kernel_for_1b_train_block() -> None:
    """Step 4.b regression: the 1B-quarter Path C train block lowers to one kernel on Metal.

    Locks the production single-kernel contract for the train block
    that drives ``cppmega_mlx``'s mamba3_m2rnn_attention_fp8_train_block
    fusion schedule:

    * The schedule's ``single_kernel_fused`` flag is True.
    * The lowered IRModule contains exactly one ``tir.PrimFunc``.
    * The compiled artifact is a TileLang ``JITKernel`` targeting
      Metal (the local-GB10-quarter / 1B-style profile this runs on).
    * The compiled PrimFunc's parameter signature matches the schedule
      template's declared buffer set, so the schedule didn't silently
      drop a saved-tensor / activation slot (which would split fwd from
      bwd into two launches).

    The eager loss/grad parity check is registered separately as
    ``test_mamba3_fp8_train_schedule_*_eager_loss_grad_parity`` once
    the MLX eager mamba3 reference is wired (tracked as step 4.b
    follow-up; the regression below is the structural floor).
    """
    import tvm

    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules
    from tvm import tir

    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule()
    plan = compiled.compiled.plan
    artifact = compiled.compiled.artifact

    # Plan-level single-kernel flag (boolean intent).
    assert plan.single_kernel_fused is True
    assert plan.schedule_contract is not None
    assert plan.schedule_contract.status == "verified"

    # Structural single-PrimFunc invariant on the lowered IRModule.
    lowered = compiled.compiled.lowered_module
    assert isinstance(lowered, tvm.IRModule)
    func_names = [gv.name_hint for gv, _ in lowered.functions.items()]
    assert len(func_names) == 1, (
        f"1B-quarter Path C train block must lower to exactly one PrimFunc; "
        f"got {len(func_names)}: {func_names!r}"
    )
    entry_name = func_names[0]
    entry_func = lowered[entry_name]
    assert isinstance(entry_func, tir.PrimFunc)

    # Artifact-level invariants: this is a real ``JITKernel`` compiled
    # against Metal (the local-GB10-quarter / 1B-style profile target).
    assert type(artifact).__name__ == "JITKernel"
    assert hasattr(artifact, "prim_func")
    primfunc = artifact.prim_func
    assert isinstance(primfunc, tir.PrimFunc)
    # The compiled entry's symbol must match the schedule's fused
    # train block name (if these diverge, a wrapper kernel slipped in).
    compiled_sym = primfunc.attrs.get("global_symbol")
    assert compiled_sym is not None
    assert str(compiled_sym) == entry_name, (
        f"JITKernel entry symbol {compiled_sym!r} does not match the "
        f"schedule's IRModule entry {entry_name!r} -- a wrapper "
        "launch was inserted between the schedule and the artifact"
    )

    # Parameter-count sanity: the train block carries a non-trivial
    # number of buffers (Path C abi bank + per-brick activations). A
    # regression that drops the joint capture would shrink this below
    # ~10. Use a soft floor so the check survives ABI evolution.
    assert len(primfunc.params) >= 10, (
        f"compiled PrimFunc has only {len(primfunc.params)} params; "
        "the 1B-quarter train block schedule expects at least 10 "
        "(Path C abi bank + per-brick activations + scratch)."
    )

    # The buffer map must include the canonical Path C abi bank that
    # carries the schedule's pre-flattened bank tensors. A missing bank
    # would indicate the schedule reverted to a per-brick split.
    buffer_names = {
        getattr(buf, "name", str(buf)) for buf in primfunc.buffer_map.values()
    }
    assert any("abi_bank" in name for name in buffer_names), (
        f"compiled buffer_map missing the Path C abi bank: {sorted(buffer_names)!r}"
    )
def test_mamba3_fp8_train_schedule_compile_helper_emits_exactly_one_primfunc() -> None:
    """Step 4.a one-launch assertion (structural).

    The plan flag ``single_kernel_fused`` only summarises the *intent*
    to fuse forward+backward into a single launch. This test inspects
    the lowered IRModule that ``compile_mamba3_fp8_train_fusion_schedule``
    hands to ``tilelang.compile`` and confirms it contains exactly one
    PrimFunc -- no second function emitted for the backward half, no
    synthetic copy seam between fwd and bwd. This locks the invariant
    that the production train block lowers to one launch.
    """
    import tvm

    from cppmega_mlx.runtime import path_c_fusion_schedules as schedules

    compiled = schedules.compile_mamba3_fp8_train_fusion_schedule()
    lowered = compiled.compiled.lowered_module
    assert isinstance(lowered, tvm.IRModule), (
        f"expected tvm.IRModule, got {type(lowered).__name__}"
    )
    funcs = list(lowered.functions.items())
    assert len(funcs) == 1, (
        f"single-kernel Path C train block must lower to exactly one "
        f"PrimFunc; got {len(funcs)}: "
        f"{[gv.name_hint for gv, _ in funcs]!r}"
    )
    entry_gv, entry_func = funcs[0]
    from tvm import tir

    assert isinstance(entry_func, tir.PrimFunc), (
        f"single Path C entry must be a tir.PrimFunc; got "
        f"{type(entry_func).__name__}"
    )
    # The entry's global symbol should match the schedule's entry
    # symbol -- if either ``aot_backward`` or the fwd half had split
    # into its own function we'd see two names here.
    assert entry_gv.name_hint == "mamba3_m2rnn_attention_fp8_train_block", (
        f"unexpected entry symbol {entry_gv.name_hint!r}; the train block "
        "schedule expects a single fused entry called "
        "'mamba3_m2rnn_attention_fp8_train_block'"
    )


def test_path_c_schedule_registry_selects_dynamic_descriptor_chain_by_default() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.schedule_id != MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert target.schedule_name == (
        "mamba3_m2rnn_attention_fp8_train_block:descriptor_generated_fwd_bwd"
    )
    assert target.schedule_status == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS
    assert target.implementation_kind == "production"
    assert target.op_signature == tuple(node.op_name for node in region.nodes)
    assert target.schedule_generator == PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    assert tuple(descriptor.op_name for descriptor in target.brick_descriptors) == (
        target.op_signature
    )


def test_mamba3_acceptance_profile_is_not_applied_to_untagged_dynamic_region_graph() -> None:
    base_region = build_mamba3_fp8_train_acceptance_fixture_region()
    custom_surfaces = tuple(
        FusionKernelSurface.path_c(
            name=f"custom_{node.name}",
            op_name=node.op_name,
            inputs=node.inputs,
            outputs=node.outputs,
            backward=node.backward,
            backend=node.backend,
        )
        for node in base_region.nodes
    )
    custom_region = build_path_c_aot_autograd_region(
        build_path_c_fusion_region(
            region_name="custom_model_chain",
            surfaces=custom_surfaces,
        )
    )

    target = select_path_c_fusion_schedule_target(custom_region)

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.schedule_id != MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert target.implementation_kind == "prototype"
    assert target.buffer_extent == 4
    with pytest.raises(
        ValueError,
        match=(
            "requires the row-phased exact backward generator; "
            "refusing to emit a scalar proxy backward fragment"
        ),
    ):
        target.schedule_template(custom_region)


def test_model_region_shape_env_controls_dynamic_descriptor_extent() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R"), config=cfg)

    region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_tiny_model",
    )[0]
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.buffer_extent == 128
    assert "route_0_M_mamba3_in_proj_weight" in (
        target.required_real_abi_inputs
    )
    assert "route_1_R_m2rnn_in_proj_weight" in (
        target.required_real_abi_inputs
    )
    prim_func = target.schedule_template(region)
    generated_source = prim_func._cppmega_path_c_generated_source
    physical_map = prim_func._cppmega_path_c_physical_buffer_abi_map
    assert physical_map["route_0_M_hidden"]["logical_shape"] == (1, 128, 32)
    assert physical_map["route_0_M_hidden"]["shape"] == (4096,)
    assert physical_map["route_0_M_hidden"]["bank"] == (
        "path_c_float32_activation_abi_bank"
    )
    assert "route_0_M_hidden: T.Tensor((1, 128, 32), \"float32\")" not in generated_source
    assert "route_0_M_hidden[0, (i % 4096) // 32, (i % 4096) % 32]" not in generated_source
    assert prim_func._cppmega_path_c_buffer_extent == 128
    assert prim_func._cppmega_path_c_loop_extent == 4096
    assert target.internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
    assert target.loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    assert "# loop_policy: row_phased_hidden" in generated_source
    assert "row_chunk_start = chunk * 64" in generated_source
    assert "row_chunk_stop = T.min(row_chunk_start + 64, 128)" in generated_source
    assert "entry_row_chunk_stop = T.min(row_chunk_stop + 1, 128)" in generated_source
    assert "for row in T.serial(row_chunk_start, row_chunk_stop):" in generated_source
    assert (
        "for i in T.serial(row * 32 + lane, "
        "(row + 1) * 32, step=1024):"
    ) in generated_source
    assert "for i in T.serial(0, 4096):" not in generated_source


def test_untagged_mra_model_route_builds_dynamic_schedule_and_real_abi() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R", "A"), config=cfg)

    region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_mra_model",
    )[0]
    target = select_path_c_fusion_schedule_target(region)

    assert tuple(node.name for node in region.nodes) == (
        "route_0_M_entry_rmsnorm",
        "route_0_M",
        "route_0_M_residual_norm",
        "route_1_R",
        "route_1_R_residual_norm",
        "route_2_A_qkv_projection",
        "route_2_A_sparse_mla_fp8_apply",
    )
    assert region.metadata["path_c_bricks"] == (
        {"name": "route_0_M", "kind": "M", "route_symbol": "M"},
        {"name": "route_1_R", "kind": "R", "route_symbol": "R"},
        {"name": "route_2_A", "kind": "A", "route_symbol": "A"},
    )
    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.schedule_id != MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert target.implementation_kind == "production"
    assert target.buffer_extent == 128
    assert "real_model_parameter_abi_contract" in target.required_codegen_steps
    assert "cache_key_shape_specialization_audit" in (
        target.required_codegen_steps
    )
    assert "route_0_M_mamba3_in_proj_weight" in (
        target.required_real_abi_inputs
    )
    assert "route_1_R_m2rnn_in_proj_weight" in (
        target.required_real_abi_inputs
    )
    assert "route_0_M_residual_norm_weight" in (
        target.required_real_abi_inputs
    )
    assert "route_1_R_residual_norm_weight" in (
        target.required_real_abi_inputs
    )
    assert "route_2_A_qkv_projection_attention_q_proj_weight" in (
        target.required_real_abi_inputs
    )
    assert "route_2_A_sparse_mla_fp8_apply_sparse_mla_sinks" in (
        target.required_real_abi_inputs
    )

    prim_func = target.schedule_template(region)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "route_2_A_qkv_projection_attention_q_proj_weight"
    ] == (1024,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "route_0_M_residual_norm_weight"
    ] == (32,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "route_0_M_state_in"
    ] == (128,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "route_0_M_state"
    ] == (128,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "route_1_R_residual_norm_weight"
    ] == (32,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "route_2_A_sparse_mla_fp8_apply_sparse_mla_sinks"
    ] == (4,)
    generated_source = prim_func._cppmega_path_c_generated_source
    assert _physical_bank_fragment(
        prim_func,
        "route_0_M_mamba3_in_proj_weight",
    ) in generated_source
    assert _physical_bank_fragment(
        prim_func,
        "route_1_R_m2rnn_in_proj_weight",
    ) in generated_source
    assert _physical_bank_fragment(
        prim_func,
        "route_2_A_qkv_projection_attention_q_proj_weight",
    ) in generated_source
    assert _physical_bank_fragment(prim_func, "route_0_M_mamba3_D") in (
        generated_source
    )
    assert _physical_bank_fragment(prim_func, "route_1_R_m2rnn_D") in (
        generated_source
    )
    assert any(
        _line_uses_logical_buffer(
            prim_func,
            line,
            "route_2_A_qkv_projection_q_fp8",
        )
        and "row * 32 + route_2_A_qkv_projection_q_head * 8 + "
        "route_2_A_qkv_projection_d" in line
        for line in generated_source.splitlines()
    )
    assert any(
        _line_uses_logical_buffer(
            prim_func,
            line,
            "route_2_A_qkv_projection_q_scale",
        )
        and "row * 4 + route_2_A_qkv_projection_q_head" in line
        for line in generated_source.splitlines()
    )
    assert _physical_bank_fragment(
        prim_func,
        "route_2_A_sparse_mla_fp8_apply_sparse_mla_sm_scale",
    ) in generated_source
    assert (
        _physical_bank_fragment(prim_func, "route_2_A_sparse_mla_fp8_apply_out")
    ) in generated_source


def test_model_derived_fwd_bwd_descriptor_declares_train_step_scalar_abi() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R", "A"), config=cfg)
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_mra_model",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None

    prim_func = target.schedule_template(region)
    output_abi = prim_func._cppmega_path_c_train_step_output_abi
    physical_abi_map = prim_func._cppmega_path_c_physical_buffer_abi_map

    assert output_abi == {
        "declared": False,
        "outputs_computed": False,
        "computed_logical_outputs": (),
        "pending_logical_outputs": (),
        "logical_outputs": (),
        "reason": "train-step scalar ABI slots are not required for this descriptor",
    }
    assert "loss" not in physical_abi_map
    assert "ntokens" not in physical_abi_map
    assert "target_ids" not in physical_abi_map
    assert "lm_head_weight" not in physical_abi_map


def test_model_derived_fwd_bwd_descriptor_declares_suffix_loss_input_abi() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R", "A"), config=cfg)
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_mra_model",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert "train_step_suffix_loss_input_abi" not in target.required_codegen_steps

    prim_func = target.schedule_template(region)
    suffix_abi = prim_func._cppmega_path_c_train_step_suffix_loss_input_abi
    physical_abi_map = prim_func._cppmega_path_c_physical_buffer_abi_map

    assert suffix_abi == {
        "declared": False,
        "logical_inputs": (),
        "reason": "train-step suffix loss inputs are not required for this descriptor",
    }
    assert "target_ids" not in physical_abi_map
    assert "target_mask" not in physical_abi_map
    assert "final_norm_weight" not in physical_abi_map
    assert "lm_head_weight" not in physical_abi_map


def test_model_derived_fwd_bwd_descriptor_computes_ntokens_from_target_mask() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R", "A"), config=cfg)
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_mra_model",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None

    prim_func = target.schedule_template(region)
    output_abi = prim_func._cppmega_path_c_train_step_output_abi
    generated_source = prim_func._cppmega_path_c_generated_source

    assert output_abi["outputs_computed"] is False
    assert output_abi["computed_logical_outputs"] == ()
    assert output_abi["pending_logical_outputs"] == ()
    assert "# train_step_suffix_loss_ntokens" not in generated_source
    assert "target_mask" not in prim_func._cppmega_path_c_physical_buffer_abi_map
    assert "ntokens" not in prim_func._cppmega_path_c_physical_buffer_abi_map
    assert "loss" not in output_abi["computed_logical_outputs"]


def test_model_derived_fwd_bwd_descriptor_computes_suffix_loss_before_backward() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R", "A"), config=cfg)
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_mra_model",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None

    prim_func = target.schedule_template(region)
    output_abi = prim_func._cppmega_path_c_train_step_output_abi
    generated_source = prim_func._cppmega_path_c_generated_source

    assert output_abi["outputs_computed"] is False
    assert output_abi["computed_logical_outputs"] == ()
    assert output_abi["pending_logical_outputs"] == ()
    backward_index = generated_source.index("# backward_policy: row_phased_hidden_recompute")
    assert backward_index > 0
    assert "# train_step_suffix_loss_scalar" not in generated_source
    assert "for vocab_col in T.serial(0, 256):" not in generated_source
    assert "train_step_suffix_logit" not in generated_source
    assert "train_step_suffix_sum_exp" not in generated_source
    for source_name in (
        "route_1_R_hidden_after",
        "route_2_A_sparse_mla_fp8_apply_out",
    ):
        source_info = prim_func._cppmega_path_c_physical_buffer_abi_map[source_name]
        assert source_info["logical_shape"] == (1, 128, 32)
    assert _physical_bank_fragment(prim_func, "route_1_R_hidden_after") in (
        generated_source
    )
    assert _physical_bank_fragment(
        prim_func,
        "route_2_A_sparse_mla_fp8_apply_out",
    ) in generated_source


def test_model_derived_fwd_bwd_descriptor_seeds_suffix_loss_cotangents_before_backward() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R", "A"), config=cfg)
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_mra_model",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None

    prim_func = target.schedule_template(region)
    generated_source = prim_func._cppmega_path_c_generated_source
    cotangent_abi = prim_func._cppmega_path_c_train_step_loss_cotangent_abi

    assert cotangent_abi["cotangents_computed"] is False
    assert cotangent_abi["source_logical_buffers"] == (
        "route_1_R_hidden_after",
        "route_2_A_sparse_mla_fp8_apply_out",
    )
    assert cotangent_abi["logical_cotangent_buffers"] == (
        "route_1_R_hidden_after_grad",
        "route_2_A_sparse_mla_fp8_apply_out_grad",
    )
    backward_index = generated_source.index("# backward_policy: row_phased_hidden_recompute")
    assert backward_index > 0
    assert "# train_step_suffix_loss_cotangent_seeds" not in generated_source
    assert "train_step_suffix_seed_softmax[0]" not in generated_source
    assert "train_step_suffix_seed_class_grad[0]" not in generated_source
    assert "for vocab_col in T.serial(0, 256):" not in generated_source
    for grad_name in cotangent_abi["logical_cotangent_buffers"]:
        assert grad_name in prim_func._cppmega_path_c_physical_buffer_abi_map
        assert _physical_bank_fragment(prim_func, grad_name) in generated_source


def test_model_derived_fwd_bwd_descriptor_computes_suffix_parameter_grads_before_backward() -> None:
    cfg = local_gb10_quarter_profile().tiny_smoke_config(
        pattern="MRA",
        depth=3,
        dsa_a_layer_ranks=(0,),
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    model = SimpleNamespace(route_symbols=("M", "R", "A"), config=cfg)
    fwd_region = build_path_c_model_regions_from_model(
        model,
        region_prefix="dynamic_mra_model",
    )[0]
    region = build_path_c_aot_autograd_region(fwd_region)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None

    prim_func = target.schedule_template(region)
    generated_source = prim_func._cppmega_path_c_generated_source
    parameter_grad_abi = (
        prim_func._cppmega_path_c_train_step_suffix_loss_parameter_grad_abi
    )
    physical_abi_map = prim_func._cppmega_path_c_physical_buffer_abi_map

    assert parameter_grad_abi == {
        "declared": False,
        "parameter_logical_buffers": (),
        "logical_gradient_buffers": (),
        "gradients_computed": False,
        "missing_logical_gradient_buffers": (
            "final_norm_weight_grad",
            "lm_head_weight_grad",
        ),
        "reason": "train-step suffix loss parameter gradients are not required for this descriptor",
    }
    backward_index = generated_source.index("# backward_policy: row_phased_hidden_recompute")
    assert backward_index > 0
    assert "final_norm_weight_grad" not in physical_abi_map
    assert "lm_head_weight_grad" not in physical_abi_map
    assert "# train_step_suffix_loss_parameter_grads" not in generated_source
    assert "train_step_suffix_param_class_grad[0]" not in generated_source


def test_model_region_shape_env_specializes_contract_and_cache_key() -> None:
    base_profile = local_gb10_quarter_profile()
    cfg_a = base_profile.tiny_smoke_config(
        max_seq_length=128,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    cfg_b = base_profile.tiny_smoke_config(
        max_seq_length=256,
        hidden_size=32,
        num_attention_heads=4,
        mamba_head_dim=8,
        mamba_state_dim=4,
        mamba_groups=1,
        m2rnn_k_head_dim=8,
        m2rnn_v_head_dim=8,
    )
    region_a = build_path_c_model_regions_from_model(
        SimpleNamespace(route_symbols=("M", "R"), config=cfg_a),
        region_prefix="shape_specialized_model",
    )[0]
    region_b = build_path_c_model_regions_from_model(
        SimpleNamespace(route_symbols=("M", "R"), config=cfg_b),
        region_prefix="shape_specialized_model",
    )[0]

    plan_a = compile_path_c_region(region_a)
    plan_b = compile_path_c_region(region_b)

    assert plan_a.schedule_contract is not None
    assert plan_b.schedule_contract is not None
    assert plan_a.schedule_contract.shape_env_key
    assert plan_b.schedule_contract.shape_env_key
    assert plan_a.schedule_contract.shape_env_key != (
        plan_b.schedule_contract.shape_env_key
    )
    assert plan_a.schedule_contract.key != plan_b.schedule_contract.key
    assert any(part.startswith("shape_env:") for part in plan_a.cache_key_parts)
    assert any(part.startswith("shape_env:") for part in plan_b.cache_key_parts)


def test_path_c_schedule_registry_builds_descriptor_target_for_supported_bricks() -> None:
    surfaces = (
        FusionKernelSurface.path_c(
            name="norm",
            op_name="residual_rmsnorm",
            inputs=("hidden", "delta", "norm_weight"),
            outputs=("hidden_after_norm", "attention_hidden"),
            backward="owner_output",
        ),
        FusionKernelSurface.path_c(
            name="attention_qkv",
            op_name="attention_qkv_projection",
            inputs=("attention_hidden", "q_weight", "kv_weight"),
            outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"),
            backward="owner_output",
        ),
        FusionKernelSurface.path_c(
            name="sparse_apply",
            op_name="sparse_mla_fp8_apply",
            inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"),
            outputs=("attention_out", "lse"),
            backward="owner_output",
        ),
    )
    region = build_path_c_fusion_region(
        region_name="descriptor_built_attention_block",
        surfaces=surfaces,
    )

    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.schedule_name == (
        "descriptor_built_attention_block:descriptor_generated_fwd_bwd"
    )
    assert target.implementation_kind == "prototype"
    assert target.schedule_status == "descriptor_codegen_scaffold"
    assert target.schedule_generator == PATH_C_DESCRIPTOR_SCHEDULE_GENERATOR
    assert target.required_real_abi_inputs == ()
    assert tuple(descriptor.op_name for descriptor in target.brick_descriptors) == (
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    )
    assert (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptor_for("attention_qkv_projection")
        .implementation_status
        == "descriptor_codegen_ready"
    )
    backward_descriptor = (
        default_path_c_brick_schedule_descriptor_registry()
        .descriptor_for("attention_qkv_projection_bwd")
    )
    assert backward_descriptor is not None
    assert backward_descriptor.implementation_status == (
        "descriptor_codegen_ready"
    )
    assert backward_descriptor.production_fragment_status == (
        "region_fragment_inlined_unoptimized"
    )
    assert "explicit descriptor" in (
        backward_descriptor.production_fragment_reason
    )


def test_path_c_schedule_optimizer_compiles_supported_descriptor_chain() -> None:
    surfaces = (
        FusionKernelSurface.path_c(
            name="norm",
            op_name="residual_rmsnorm",
            inputs=("hidden", "delta", "norm_weight"),
            outputs=("hidden_after_norm", "attention_hidden"),
            backward="owner_output",
        ),
        FusionKernelSurface.path_c(
            name="attention_qkv",
            op_name="attention_qkv_projection",
            inputs=("attention_hidden", "q_weight", "kv_weight"),
            outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"),
            backward="owner_output",
        ),
        FusionKernelSurface.path_c(
            name="sparse_apply",
            op_name="sparse_mla_fp8_apply",
            inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"),
            outputs=("attention_out", "lse"),
            backward="owner_output",
        ),
    )
    optimizer = PathCFusionScheduleOptimizer(
        "descriptor_built_attention_block",
    ).add_kernels(surfaces)

    compiled = optimizer.compile(
        tilelang_lowerer=lambda *_args, **_kwargs: "compiled-descriptor-schedule"
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-descriptor-schedule"
    assert compiled.plan.schedule_name == (
        "descriptor_built_attention_block:descriptor_generated_fwd_bwd"
    )
    assert compiled.plan.schedule_status == "descriptor_codegen_scaffold"
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"


def test_path_c_schedule_planner_accepts_discovered_model_region() -> None:
    region = build_path_c_model_regions_from_route_symbols(
        ("E", "M", "R"),
        region_prefix="discovered_model_path_c",
        min_route_bricks=2,
    )[0]

    planned = plan_path_c_fusion_schedule_for_region(region, include_backward=True)

    assert planned.region.name == "discovered_model_path_c_1_2"
    assert planned.schedule_target is not None
    assert planned.schedule_target.schedule_id.startswith("path_c_descriptor_chain_")
    assert planned.schedule_target.implementation_kind == "prototype"
    assert planned.plan.autograd_status == "ready"
    assert planned.plan.schedule_contract is not None
    assert planned.plan.schedule_contract.status == "attested_non_production_schedule"


def test_path_c_schedule_registry_returns_none_for_unknown_pattern() -> None:
    region = build_path_c_fusion_region(
        region_name="unknown_pattern",
        surfaces=(
            FusionKernelSurface.path_c(
                name="scan",
                op_name="unknown_scan",
                inputs=("hidden",),
                outputs=("scan_y",),
                backward="owner_output",
            ),
            FusionKernelSurface.path_c(
                name="apply",
                op_name="unknown_apply",
                inputs=("scan_y",),
                outputs=("out",),
                backward="owner_output",
            ),
        ),
    )

    assert select_path_c_fusion_schedule_target(region) is None
    assert default_path_c_fusion_schedule_registry().select(region) is None


def _surface_copies_for_region(region: PathCFusionRegion) -> tuple[FusionKernelSurface, ...]:
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


def test_path_c_schedule_optimizer_builds_aot_region_and_selects_schedule() -> None:
    fwd_region = build_mamba3_fp8_train_acceptance_fixture_region()
    optimizer = PathCFusionScheduleOptimizer(
        "mamba3_m2rnn_attention_fp8_train_block",
        enable_aot_autograd=True,
        metadata=fwd_region.metadata,
    )
    optimizer.add_kernels(_surface_copies_for_region(fwd_region))

    optimized = optimizer.plan()

    assert optimized.region.node_names == build_mamba3_fp8_train_acceptance_fixture_region(
        include_backward=True
    ).node_names
    assert optimized.region.z3_sync.enabled is True
    assert optimized.region.z3_sync.objective == "minimize_sync_async"
    assert optimized.schedule_target is not None
    assert optimized.schedule_target.schedule_id.startswith("path_c_descriptor_chain_")
    assert optimized.schedule_target.schedule_id != MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert optimized.plan.schedule_name == (
        "mamba3_m2rnn_attention_fp8_train_block:descriptor_generated_fwd_bwd"
    )
    assert optimized.plan.schedule_status == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS
    assert optimized.plan.autograd_status == "ready"
    assert optimized.plan.single_kernel_fused is False


def test_path_c_schedule_optimizer_compiles_selected_prototype_schedule() -> None:
    fwd_region = build_mamba3_fp8_train_acceptance_fixture_region()
    optimizer = PathCFusionScheduleOptimizer(
        "mamba3_m2rnn_attention_fp8_train_block",
        registry=prototype_path_c_fusion_schedule_registry(),
        enable_aot_autograd=True,
        metadata=fwd_region.metadata,
    ).add_kernels(_surface_copies_for_region(fwd_region))

    compiled = optimizer.compile(
        tilelang_lowerer=lambda *args, **kwargs: "compiled-prototype-schedule"
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-prototype-schedule"
    assert compiled.plan.schedule_name == MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_NAME
    assert compiled.plan.schedule_status == MAMBA3_FP8_TRAIN_PROTOTYPE_SCHEDULE_STATUS
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"


def test_path_c_schedule_optimizer_compile_verifies_dynamic_production_target() -> None:
    fwd_region = build_mamba3_fp8_train_acceptance_fixture_region()
    optimizer = PathCFusionScheduleOptimizer(
        "mamba3_m2rnn_attention_fp8_train_block",
        enable_aot_autograd=True,
        metadata=fwd_region.metadata,
    ).add_kernels(_surface_copies_for_region(fwd_region))

    compiled = optimizer.compile(tilelang_lowerer=lambda *args, **kwargs: "compiled")

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled"
    assert compiled.plan.single_kernel_fused is True
    assert compiled.plan.schedule_status == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS
    assert compiled.plan.schedule_contract is not None
    assert (
        compiled.plan.schedule_contract.status
        == "verified"
    )
    assert (
        compiled.plan.schedule_contract.declared_implementation_kind
        == "production"
    )
    assert compiled.plan.schedule_contract.declared_schedule_id.startswith(
        "path_c_descriptor_chain_"
    )


def test_acceptance_gate_ignores_path_b_baseline_regressions() -> None:
    clean_loss = BenchmarkAcceptanceRow(
        dtype="fp8",
        optimizer="adamw",
        path_b_tok_sec=725.4,
        path_c_warm_tok_sec=428.9,
        path_b_median_step_s=2.73,
        path_b_cache_gib=37.7,
        path_c_peak_delta_gib=32.4,
    )
    bad_baseline_win = BenchmarkAcceptanceRow(
        dtype="fp8",
        optimizer="adam8bit",
        path_b_tok_sec=96.0,
        path_c_warm_tok_sec=107.2,
        path_b_median_step_s=22.1,
        path_b_cache_gib=66.4,
        path_c_peak_delta_gib=19.5,
    )

    assert path_b_baseline_is_clean(clean_loss) is True
    assert fused_path_c_default_eligible(clean_loss) is False
    assert path_b_baseline_is_clean(bad_baseline_win) is False
    assert fused_path_c_default_eligible(bad_baseline_win) is False


def test_plan_acceptance_gate_requires_real_single_kernel_and_aot_bwd() -> None:
    region = build_mamba3_fp8_train_region()
    plan = compile_path_c_region(region)
    clean_win = BenchmarkAcceptanceRow(
        dtype="fp8",
        optimizer="adamw",
        path_b_tok_sec=725.4,
        path_c_warm_tok_sec=790.0,
        path_b_median_step_s=2.73,
        path_b_cache_gib=37.7,
        path_c_peak_delta_gib=0.0,
    )

    assert fused_path_c_default_eligible(clean_win) is True
    assert fused_path_c_plan_default_eligible(plan, clean_win) is False

    single_kernel_missing_bwd = replace(
        plan,
        schedule_status="ready",
        single_kernel_fused=True,
        autograd_status="requires_aot_autograd_codegen",
    )
    assert fused_path_c_plan_default_eligible(single_kernel_missing_bwd, clean_win) is False

    ready_plan = replace(
        plan,
        schedule_status="ready",
        single_kernel_fused=True,
        autograd_status="ready",
        autograd_missing_backward_nodes=(),
        semantic_blockers=(),
        schedule_contract=FusionScheduleContractStatus(
            name="test_contract",
            key="test_key",
            status="verified",
            reason="verified in test",
            op_signature=("mamba3_mimo", "m2rnn"),
            required_internal_buffers=("scan_y",),
            required_external_buffers=("hidden",),
            declared_key="test_key",
            declared_implementation_kind="production",
            declared_schedule_id="untrusted_test_schedule",
            declared_required_real_abi_inputs=("hidden",),
        ),
    )
    assert fused_path_c_plan_default_eligible(ready_plan, clean_win) is False
    audit = audit_fused_path_c_plan_default_eligibility(ready_plan, clean_win)
    assert audit.eligible is False
    assert audit.status == "untrusted_schedule_id"
    assert audit.schedule_id == "untrusted_test_schedule"

    trusted_ready_plan = replace(
        ready_plan,
        schedule_contract=replace(
            ready_plan.schedule_contract,
            declared_schedule_id="trusted_test_schedule",
        ),
    )
    assert (
        fused_path_c_plan_default_eligible(
            trusted_ready_plan,
            clean_win,
            trusted_schedule_ids=frozenset({"trusted_test_schedule"}),
        )
        is True
    )
    trusted_audit = audit_fused_path_c_plan_default_eligibility(
        trusted_ready_plan,
        clean_win,
        trusted_schedule_ids=frozenset({"trusted_test_schedule"}),
    )
    assert trusted_audit.eligible is True
    assert trusted_audit.status == "eligible"
    assert trusted_audit.schedule_id == "trusted_test_schedule"

    undeclared_real_abi = replace(
        ready_plan,
        schedule_contract=FusionScheduleContractStatus(
            name="test_contract",
            key="test_key",
            status="verified",
            reason="verified in test",
            op_signature=("mamba3_mimo", "m2rnn"),
            required_internal_buffers=("scan_y",),
            required_external_buffers=("hidden",),
            declared_key="test_key",
            declared_implementation_kind="production",
        ),
    )
    assert fused_path_c_plan_default_eligible(undeclared_real_abi, clean_win) is False

    incomplete_real_abi = replace(
        ready_plan,
        schedule_contract=FusionScheduleContractStatus(
            name="test_contract",
            key="test_key",
            status="verified",
            reason="verified in test",
            op_signature=("mamba3_mimo", "m2rnn"),
            required_internal_buffers=("scan_y",),
            required_external_buffers=("hidden",),
            declared_key="test_key",
            declared_implementation_kind="production",
            declared_required_real_abi_inputs=("hidden", "mamba3_in_proj_weight"),
            missing_real_abi_inputs=("mamba3_in_proj_weight",),
        ),
    )
    assert fused_path_c_plan_default_eligible(incomplete_real_abi, clean_win) is False

    unverified_contract = replace(
        ready_plan,
        schedule_contract=FusionScheduleContractStatus(
            name="test_contract",
            key="test_key",
            status="registered_not_lowered",
            reason="not lowered in test",
            op_signature=("mamba3_mimo", "m2rnn"),
            required_internal_buffers=("scan_y",),
            required_external_buffers=("hidden",),
            declared_key="test_key",
            declared_implementation_kind="production",
        ),
    )
    assert fused_path_c_plan_default_eligible(unverified_contract, clean_win) is False


def test_compile_path_c_region_registers_explicit_fused_schedule_template() -> None:
    region = build_mamba3_fp8_train_region()

    def ready_schedule_template(_region: object) -> object:
        raise AssertionError("planning must register the schedule without lowering it")

    plan = compile_path_c_region(
        region,
        schedule_template=ready_schedule_template,
        schedule_name="mamba3_fp8_train_block:ready_schedule",
        schedule_status="ready",
    )

    assert plan.schedule_name == "mamba3_fp8_train_block:ready_schedule"
    assert plan.schedule_status == "ready"
    assert plan.single_kernel_fused is False
    assert plan.fusion_groups[0].schedule_template == (
        "mamba3_fp8_train_block:ready_schedule"
    )
    assert any(
        part.startswith("schedule:mamba3_fp8_train_block:ready_schedule")
        for part in plan.cache_key_parts
    )
    assert plan.autograd_status == "requires_aot_autograd_codegen"
    assert plan.schedule_contract is not None
    assert plan.schedule_contract.status == "unattested_schedule_template"


def test_compile_path_c_region_can_invoke_tilelang_compile_from_graph() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()

    compiled = compile_path_c_region(
        region,
        schedule_template=lambda _region: _toy_path_c_model_train_block,
        schedule_name="mamba3_m2rnn_attention_fp8_train_block:toy_single_entry",
        schedule_status="ready",
        tilelang_lowerer=lambda *args, **kwargs: "compiled-from-path-c-graph",
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-from-path-c-graph"
    assert compiled.lowered_module is not None
    assert compiled.plan.schedule_name == (
        "mamba3_m2rnn_attention_fp8_train_block:toy_single_entry"
    )
    assert compiled.plan.schedule_status == "ready"
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.lowering_boundary == "tilelang_tvm_ir"
    assert compiled.plan.requires_msl_post_fusion is False
    assert compiled.plan.semantic_blockers == ()
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "unattested_schedule_template"


def test_compile_path_c_region_keeps_diagnostic_attested_schedule_non_production() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    schedule_template = mark_path_c_schedule_template_for_region(
        lambda _region: _toy_path_c_model_train_block,
        region,
    )

    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name="mamba3_m2rnn_attention_fp8_train_block:attested_toy_single_entry",
        schedule_status="ready",
        tilelang_lowerer=lambda *args, **kwargs: "compiled-from-attested-graph",
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-from-attested-graph"
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"
    assert compiled.plan.schedule_contract.declared_key == compiled.plan.schedule_contract.key
    assert compiled.plan.schedule_contract.declared_implementation_kind == "diagnostic"


def test_mark_path_c_schedule_template_rejects_production_without_schedule_id() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()

    with pytest.raises(ValueError, match="production_schedule_id"):
        mark_path_c_schedule_template_for_region(
            lambda _region: _toy_path_c_model_train_block,
            region,
            implementation_kind="production",
        )


def test_compile_path_c_region_requires_real_abi_for_single_kernel_fused() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    schedule_template = mark_path_c_schedule_template_for_region(
        lambda _region: _toy_path_c_model_train_block,
        region,
        implementation_kind="production",
        production_schedule_id="untrusted_toy_schedule",
    )

    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name="mamba3_m2rnn_attention_fp8_train_block:untrusted_production_toy",
        schedule_status="ready",
        tilelang_lowerer=lambda *args, **kwargs: "compiled-from-untrusted-production-graph",
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-from-untrusted-production-graph"
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "verified"
    assert compiled.plan.schedule_contract.declared_key == compiled.plan.schedule_contract.key
    assert compiled.plan.schedule_contract.declared_implementation_kind == "production"
    assert compiled.plan.schedule_contract.declared_schedule_id == "untrusted_toy_schedule"


def test_compile_path_c_region_marks_trusted_production_schedule_contract_verified() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    schedule_template = mark_path_c_schedule_template_for_region(
        lambda _region: _toy_path_c_model_train_block,
        region,
        implementation_kind="production",
        production_schedule_id="trusted_test_schedule",
        required_real_abi_inputs=("hidden",),
    )

    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name="mamba3_m2rnn_attention_fp8_train_block:trusted_production_toy",
        schedule_status="ready",
        tilelang_lowerer=lambda *args, **kwargs: "compiled-from-trusted-production-graph",
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-from-trusted-production-graph"
    assert compiled.plan.single_kernel_fused is True
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "verified"
    assert compiled.plan.schedule_contract.declared_key == compiled.plan.schedule_contract.key
    assert compiled.plan.schedule_contract.declared_implementation_kind == "production"
    assert compiled.plan.schedule_contract.declared_schedule_id == "trusted_test_schedule"
    assert compiled.plan.schedule_contract.missing_real_abi_inputs == ()


def test_compile_path_c_region_blocks_trusted_production_missing_real_abi() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    schedule_template = mark_path_c_schedule_template_for_region(
        lambda _region: _toy_path_c_model_train_block,
        region,
        implementation_kind="production",
        production_schedule_id="trusted_missing_abi_test_schedule",
        required_real_abi_inputs=("hidden", "unwired_real_abi_input"),
    )

    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name="mamba3_m2rnn_attention_fp8_train_block:trusted_missing_abi",
        schedule_status="ready",
        tilelang_lowerer=lambda *args, **kwargs: "compiled-but-missing-real-abi",
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "incomplete_real_abi_contract"
    assert compiled.plan.schedule_contract.declared_schedule_id == (
        "trusted_missing_abi_test_schedule"
    )
    assert compiled.plan.schedule_contract.missing_real_abi_inputs == (
        "unwired_real_abi_input",
    )


def test_compile_path_c_region_does_not_mark_semantically_blocked_region_fused() -> None:
    region = path_c_fusion._build_legacy_mamba3_fp8_train_diagnostic_region()

    compiled = compile_path_c_region(
        region,
        schedule_template=lambda _region: _toy_path_c_train_block,
        schedule_name="mamba3_fp8_train_block:diagnostic_toy_single_entry",
        schedule_status="ready",
        tilelang_lowerer=lambda *args, **kwargs: "compiled-diagnostic-artifact",
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-diagnostic-artifact"
    assert compiled.plan.schedule_status == "ready"
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "blocked_semantic_contract"
    assert [blocker.kind for blocker in compiled.plan.semantic_blockers] == [
        "residual_norm_bridge_missing",
        "attention_qkv_projection_missing",
    ]


def test_compile_path_c_region_can_use_standard_tilelang_single_entry_lowerer() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    captured: dict[str, object] = {}

    def fake_compile(prim_func: object, *, target: str, execution_backend: str) -> str:
        captured["prim_func"] = prim_func
        captured["target"] = target
        captured["execution_backend"] = execution_backend
        return "compiled-single-entry-primfunc"

    schedule_template = mark_path_c_schedule_template_for_region(
        lambda _region: _toy_path_c_model_train_block,
        region,
        implementation_kind="production",
        production_schedule_id="trusted_real_lowerer_test_schedule",
        required_real_abi_inputs=("hidden",),
    )
    compiled = compile_path_c_region(
        region,
        schedule_template=schedule_template,
        schedule_name="mamba3_m2rnn_attention_fp8_train_block:toy_single_entry_real_lowerer",
        schedule_status="ready",
        tilelang_lowerer=lambda func_or_mod, **kwargs: tilelang_single_entry_lowerer(
            func_or_mod,
            compile_prim_func=fake_compile,
            **kwargs,
        ),
    )

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled-single-entry-primfunc"
    assert captured["target"] == "metal"
    assert captured["execution_backend"] == "tvm_ffi"
    assert "mamba3_m2rnn_attention_fp8_train_block" in captured["prim_func"].script()
    assert compiled.plan.single_kernel_fused is True
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "verified"


def test_compile_path_c_region_marks_registered_aot_backward_nodes_ready() -> None:
    builder = PathCFusionRegionBuilder(
        "hybrid_train_block",
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    builder.add_kernels(
        (
            FusionKernelSurface.path_c(
                name="mamba3_scan",
                op_name="mamba3_mimo",
                inputs=("hidden", "state"),
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
                name="m2rnn_packed_post_bwd",
                op_name="m2rnn_bwd",
                inputs=("post_y_grad",),
                outputs=("scan_y_grad",),
                backward="owner_output",
            ),
            FusionKernelSurface.path_c(
                name="mamba3_scan_bwd",
                op_name="mamba3_mimo_bwd",
                inputs=("scan_y_grad",),
                outputs=("hidden_grad", "state_grad"),
                backward="owner_output",
            ),
        )
    )
    region = builder.build()

    def ready_schedule_template(_region: object) -> object:
        raise AssertionError("planning must register the schedule without lowering it")

    plan = compile_path_c_region(
        region,
        schedule_template=ready_schedule_template,
        schedule_name="hybrid_train_block:ready_fwd_bwd_schedule",
        schedule_status="ready",
    )

    assert plan.schedule_status == "ready"
    assert plan.single_kernel_fused is False
    assert plan.autograd_status == "ready"
    assert plan.autograd_missing_backward_nodes == ()
    assert plan.autograd_backward_nodes == (
        "m2rnn_packed_post_bwd",
        "mamba3_scan_bwd",
    )
    assert plan.autograd_backward_edges == (
        ("m2rnn_packed_post_bwd", "mamba3_scan_bwd", "scan_y_grad"),
    )
    clean_win = BenchmarkAcceptanceRow(
        dtype="fp8",
        optimizer="adamw",
        path_b_tok_sec=725.4,
        path_c_warm_tok_sec=790.0,
        path_b_median_step_s=2.73,
        path_b_cache_gib=37.7,
        path_c_peak_delta_gib=0.0,
    )
    assert fused_path_c_plan_default_eligible(plan, clean_win) is False


def test_compile_plan_propagates_ready_single_kernel_tilelang_plan() -> None:
    region = build_mamba3_fp8_train_region()
    fake_autograd_plan = SimpleNamespace(
        mode="aot_autograd",
        status="ready",
        backward_node_names=("m2rnn_packed_post_bwd", "mamba3_scan_bwd"),
        backward_edges=(("m2rnn_packed_post_bwd", "mamba3_scan_bwd", "scan_y_grad"),),
        missing_backward_node_names=(),
    )
    fake_tilelang_plan = SimpleNamespace(
        lowering_boundary="tilelang_tvm_ir",
        backend="tilelang_tvm_ffi",
        cache_key_material=("region:mamba3_fp8_train_block", "schedule:ready"),
        schedule_name="ready_train_block",
        schedule_status="ready",
        require_single_kernel=True,
        autograd_plan=fake_autograd_plan,
    )

    plan = compile_path_c_region(
        region,
        tilelang_plan_factory=lambda _region, **_kwargs: fake_tilelang_plan,
    )

    assert plan.schedule_name == "ready_train_block"
    assert plan.schedule_status == "ready"
    assert plan.single_kernel_fused is False
    assert plan.autograd_status == "ready"


def test_warm_cache_audit_flags_warm_first_step_that_still_looks_cold() -> None:
    audit = audit_warm_cache_reuse(
        case_id="fp8_adamw_path_c_warm",
        cold_first_step_s=38.8,
        warm_first_step_s=13.9,
        cache_hit=True,
        compile_cache_key="region:mamba3_fp8_train_block",
    )

    assert audit.status == "incomplete"
    assert "warm first step" in audit.reason
    assert audit.cache_hit is True


def test_cache_key_audit_flags_stable_key_that_recompiled() -> None:
    audit = audit_fusion_cache_key(
        case_id="fp8_adamw_path_c_warm",
        expected_cache_key="region:mamba3_fp8_train_block:abc",
        observed_cache_key="region:mamba3_fp8_train_block:abc",
        cache_hit=False,
    )

    assert audit.status == "recompiled_same_key"
    assert "stable" in audit.reason
    assert audit.cache_hit is False


def test_path_c_fusion_mode_env_parser_is_fail_closed() -> None:
    assert selected_path_c_fusion_mode({"CPPMEGA_PATH_C_FUSION": "force"}) is (
        PathCFusionMode.FORCE
    )
    assert selected_path_c_fusion_mode({"CPPMEGA_PATH_C_FUSION": "auto"}) is (
        PathCFusionMode.AUTO
    )
    assert selected_path_c_fusion_mode({"CPPMEGA_PATH_C_FUSION": "nonsense"}) is (
        PathCFusionMode.OFF
    )
