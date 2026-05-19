from __future__ import annotations

from dataclasses import replace
import os
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
    DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN,
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
    path_c_fusion_schedule_spec,
    path_c_fusion_schedule_template,
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


@T.prim_func
def _toy_path_c_train_block(
    hidden: T.Buffer((4,), "float32"),
    mamba_state: T.Buffer((4,), "float32"),
    indices: T.Buffer((4,), "int32"),
    scan_state: T.Buffer((4,), "float32"),
    attention_out: T.Buffer((4,), "float32"),
    lse: T.Buffer((4,), "float32"),
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
    hidden: T.Buffer((4,), "float32"),
    mamba_state: T.Buffer((4,), "float32"),
    mamba3_in_proj_weight: T.Buffer((4,), "float32"),
    mamba3_out_proj_weight: T.Buffer((4,), "float32"),
    mamba3_conv_weight: T.Buffer((4,), "float32"),
    mamba3_conv_bias: T.Buffer((4,), "float32"),
    mamba3_dt_bias: T.Buffer((4,), "float32"),
    mamba3_B_norm_weight: T.Buffer((4,), "float32"),
    mamba3_B_bias: T.Buffer((4,), "float32"),
    mamba3_C_norm_weight: T.Buffer((4,), "float32"),
    mamba3_C_bias: T.Buffer((4,), "float32"),
    mamba3_D: T.Buffer((4,), "float32"),
    mamba3_h0: T.Buffer((4,), "float32"),
    scan_state: T.Buffer((4,), "float32"),
    mamba3_residual_to_m2rnn_norm_weight: T.Buffer((4,), "float32"),
    m2rnn_in_proj_weight: T.Buffer((4,), "float32"),
    m2rnn_conv_weight: T.Buffer((4,), "float32"),
    m2rnn_conv_bias: T.Buffer((4,), "float32"),
    m2rnn_state_weight: T.Buffer((4,), "float32"),
    m2rnn_A_log: T.Buffer((4,), "float32"),
    m2rnn_dt_bias: T.Buffer((4,), "float32"),
    m2rnn_D: T.Buffer((4,), "float32"),
    m2rnn_g_norm_weight: T.Buffer((4,), "float32"),
    m2rnn_out_proj_weight: T.Buffer((4,), "float32"),
    m2rnn_h0: T.Buffer((4,), "float32"),
    m2rnn_conv_state: T.Buffer((4,), "float32"),
    hidden_after_m2rnn: T.Buffer((4,), "float32"),
    m2rnn_residual_to_attention_norm_weight: T.Buffer((4,), "float32"),
    attention_q_proj_weight: T.Buffer((4,), "float32"),
    attention_q_proj_bias: T.Buffer((4,), "float32"),
    attention_sparse_kv_proj_weight: T.Buffer((4,), "float32"),
    attention_sparse_kv_proj_bias: T.Buffer((4,), "float32"),
    attention_rope_inv_freq: T.Buffer((4,), "float32"),
    attention_out_proj_weight: T.Buffer((4,), "float32"),
    attention_out_proj_bias: T.Buffer((4,), "float32"),
    sparse_mla_sm_scale: T.Buffer((4,), "float32"),
    sparse_mla_sinks: T.Buffer((4,), "float32"),
    sparse_mla_has_sinks: T.Buffer((4,), "int32"),
    attention_out: T.Buffer((4,), "float32"),
    lse: T.Buffer((4,), "float32"),
):
    with T.Kernel(1, threads=1):
        mamba3_delta = T.alloc_local((4,), "float32")
        hidden_after_mamba3 = T.alloc_local((4,), "float32")
        m2rnn_hidden = T.alloc_local((4,), "float32")
        m2rnn_delta = T.alloc_local((4,), "float32")
        attention_hidden = T.alloc_local((4,), "float32")
        q_fp8 = T.alloc_local((4,), "float32")
        q_scale = T.alloc_local((4,), "float32")
        kv_fp8 = T.alloc_local((4,), "float32")
        kv_scale = T.alloc_local((4,), "float32")
        indices = T.alloc_local((4,), "int32")
        mamba3_delta[0] = (
            hidden[0]
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
        q_fp8[0] = attention_hidden[0]
        q_scale[0] = 1.0
        kv_fp8[0] = attention_hidden[0]
        kv_scale[0] = 1.0
        indices[0] = 0
        attention_out[0] = (
            q_fp8[0]
            + kv_fp8[0]
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
        "mamba3_scan_1",
        "mamba3_scan_1_residual_norm",
        "m2rnn_packed_post_1",
        "m2rnn_packed_post_1_residual_norm",
        "attention_qkv_projection_1",
        "sparse_mla_fp8_apply_1",
    )
    assert tuple(node.op_name for node in region.nodes) == (
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
    assert target.implementation_kind == "prototype"
    prim_func = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    assert "def generic_mra_path_c(" in prim_func._cppmega_path_c_generated_source
    assert plan.schedule_contract is not None
    spec = path_c_fusion_schedule_spec(
        build_path_c_aot_autograd_region(region),
        contract=plan.schedule_contract,
        target=target,
    )
    assert spec.real_abi_contract_complete is True
    assert spec.missing_real_abi_inputs == ()


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
        ("mamba3_mimo", "residual_rmsnorm", "m2rnn"),
        (
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
        "attn_a_qkv_projection",
        "attn_a_sparse_mla_fp8_apply",
        "attn_a_residual_norm",
        "scan_a",
        "scan_a_residual_norm",
        "m2_a",
    )
    assert tuple(node.op_name for node in region.nodes) == (
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
        "residual_rmsnorm",
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
    )
    assert tuple(region.nodes[2].inputs) == (
        "hidden",
        "attn_a_sparse_mla_fp8_apply_out",
        "attn_a_residual_norm_weight",
    )
    assert region.nodes[3].inputs[0] == "attn_a_residual_norm_hidden"
    assert region.nodes[5].inputs[0] == "scan_a_residual_norm_hidden"


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
        "mamba3_scan",
        "m2rnn_packed_post",
        "fp8_prepare",
        "sparse_mla_fp8_apply",
    )
    assert [node.op_name for node in region.nodes] == [
        "mamba3_mimo",
        "m2rnn",
        "sparse_mla_fp8_prepare",
        "sparse_mla_fp8_apply",
    ]
    assert plan.cache_key_parts[:7] == (
        "region:mamba3_fp8_train_block",
        "entry:mamba3_fp8_train_block",
        "nodes:mamba3_scan,m2rnn_packed_post,fp8_prepare,sparse_mla_fp8_apply",
        (
            "edges:"
            "mamba3_scan->m2rnn_packed_post:scan_y:internal,"
            "m2rnn_packed_post->fp8_prepare:post_y:internal,"
            "fp8_prepare->sparse_mla_fp8_apply:q_fp8:internal,"
            "fp8_prepare->sparse_mla_fp8_apply:q_scale:internal,"
            "fp8_prepare->sparse_mla_fp8_apply:kv_fp8:internal,"
            "fp8_prepare->sparse_mla_fp8_apply:kv_scale:internal"
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
        "mamba3_scan_bwd",
        "m2rnn_packed_post_bwd",
    )
    assert [blocker.kind for blocker in plan.semantic_blockers] == [
        "residual_norm_bridge_missing",
        "attention_qkv_projection_missing",
    ]
    assert plan.semantic_blockers[0].required_node == "mamba3_residual_to_m2rnn_norm"
    assert plan.semantic_blockers[1].required_node == "attention_qkv_projection"
    assert plan.schedule_contract is not None
    assert plan.schedule_contract.status == "blocked_semantic_contract"


def test_mamba3_fp8_acceptance_fixture_region_includes_residual_norm_and_attention_projection() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    plan = compile_path_c_region(region)

    assert region.node_names == (
        "mamba3_scan",
        "mamba3_residual_to_m2rnn_norm",
        "m2rnn_packed_post",
        "m2rnn_residual_to_attention_norm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    )
    assert [node.op_name for node in region.nodes] == [
        "mamba3_mimo",
        "residual_rmsnorm",
        "m2rnn",
        "residual_rmsnorm",
        "attention_qkv_projection",
        "sparse_mla_fp8_apply",
    ]
    assert [(edge.producer, edge.consumer, edge.input) for edge in region.edges] == [
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
    assert plan.semantic_blockers == ()
    assert plan.schedule_status == "missing_real_fused_schedule_template"
    assert plan.schedule_contract is not None
    assert plan.schedule_contract.status == "missing_schedule_template"
    assert plan.schedule_contract.required_internal_buffers == (
        "mamba3_delta",
        "m2rnn_hidden",
        "hidden_after_mamba3",
        "m2rnn_delta",
        "attention_hidden",
        "q_fp8",
        "q_scale",
        "kv_fp8",
        "kv_scale",
        "indices",
    )
    external_buffers = plan.schedule_contract.required_external_buffers
    assert external_buffers[:2] == ("hidden", "mamba_state")
    assert "scan_state" in external_buffers
    assert "hidden_after_m2rnn" in external_buffers
    assert "attention_out" in external_buffers
    assert "lse" in external_buffers
    assert set(MAMBA3_FP8_TRAIN_REQUIRED_REAL_ABI_INPUTS).issubset(
        external_buffers
    )
    assert plan.single_kernel_fused is False
    assert plan.autograd_missing_backward_nodes == (
        "mamba3_scan_bwd",
        "mamba3_residual_to_m2rnn_norm_bwd",
        "m2rnn_packed_post_bwd",
        "m2rnn_residual_to_attention_norm_bwd",
        "attention_qkv_projection_bwd",
    )


def test_mamba3_fp8_acceptance_fixture_region_can_include_symbolic_aot_backward_graph() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    plan = compile_path_c_region(region)

    assert region.node_names[-5:] == (
        "attention_qkv_projection_bwd",
        "m2rnn_residual_to_attention_norm_bwd",
        "m2rnn_packed_post_bwd",
        "mamba3_residual_to_m2rnn_norm_bwd",
        "mamba3_scan_bwd",
    )
    assert plan.autograd_status == "ready"
    assert plan.autograd_missing_backward_nodes == ()
    assert plan.autograd_backward_nodes == (
        "attention_qkv_projection_bwd",
        "m2rnn_residual_to_attention_norm_bwd",
        "m2rnn_packed_post_bwd",
        "mamba3_residual_to_m2rnn_norm_bwd",
        "mamba3_scan_bwd",
    )
    assert plan.autograd_backward_edges == (
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
    assert spec.implementation_kind == "scaffold"
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
    assert (
        spec.schedule_generator_status
        == "loop_per_brick_descriptor_fragments"
    )
    assert spec.internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
    assert spec.loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    assert "residual_rmsnorm_row_phased_production_fragment" in (
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
    assert spec.production_fragments_complete is False
    assert len(spec.brick_production_fragment_blockers) == 9
    assert any(
        status.startswith("mamba3_mimo:region_fragment_inlined_unoptimized:")
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        "mamba3_mimo:region_fragment_inlined_unoptimized:" in reason
        and "Path C scan kernel expects pre-projected x/B/C/z/A/dt" in reason
        and "block-level in_proj/conv/norm parameters" in reason
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        status.startswith("residual_rmsnorm:production_region_inlined:")
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        status.startswith("residual_rmsnorm_bwd:region_fragment_inlined_unoptimized:")
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        reason.startswith(
            "residual_rmsnorm_bwd:region_fragment_inlined_unoptimized:"
            "residual/RMSNorm backward has an explicit descriptor"
        )
        and "native TileLang lowerer timed out" in reason
        and "occurrence gate was removed" in reason
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        blocker.startswith(
            "residual_rmsnorm_bwd:region_fragment_inlined_unoptimized:"
            "residual/RMSNorm backward has an explicit descriptor"
        )
        for blocker in spec.brick_production_fragment_blockers
    )
    assert any(
        status.startswith(
            "attention_qkv_projection:region_fragment_inlined_unoptimized:"
        )
        for status in spec.brick_production_fragment_statuses
    )
    assert any(
        reason.startswith(
            "attention_qkv_projection_bwd:region_fragment_inlined_unoptimized:"
            "attention Q/KV projection backward now has an explicit descriptor"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        reason.startswith(
            "m2rnn_bwd:region_fragment_inlined_unoptimized:"
            "M2RNN backward now has an explicit descriptor"
        )
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        "m2rnn:region_fragment_inlined_unoptimized:" in reason
        and "mapped-packed Path C kernels expect projected conv_input/xf/projected"
        in reason
        and "block-level projection/conv/state parameters" in reason
        for reason in spec.brick_production_fragment_reasons
    )
    assert any(
        reason.startswith(
            "mamba3_mimo_bwd:region_fragment_inlined_unoptimized:"
            "Mamba3 backward now has an explicit descriptor"
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

    assert "mamba3_scan_mamba3_proj" in generated_source
    assert "m2rnn_packed_post_m2rnn_projected" in generated_source
    assert "attention_qkv_projection_attention_q_projected" in generated_source
    assert "attention_qkv_projection_attention_kv_projected" in generated_source
    assert "attention_qkv_projection_attention_rope_phase" in generated_source
    assert "sparse_mla_fp8_apply_sink_enabled" in generated_source
    assert "mamba3_scan_bwd_grad_accum" in generated_source


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


@pytest.mark.skipif(
    os.environ.get("CPPMEGA_RUN_NATIVE_TILELANG_COMPILE_SMOKE") != "1",
    reason=(
        "native tilelang.compile smoke is opt-in because it loads the native "
        "Triton frontend and should run only when explicitly requested"
    ),
)
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

    assert (
        "# mamba3_scan production_fragment_status: "
        "region_fragment_inlined_unoptimized"
    ) in generated_source
    assert (
        "# attention_qkv_projection production_fragment_status: "
        "region_fragment_inlined_unoptimized"
    ) in generated_source
    assert (
        "# mamba3_residual_to_m2rnn_norm production_fragment_status: "
        "production_region_inlined"
    ) in generated_source
    assert (
        "# attention_qkv_projection_bwd production_fragment_status: "
        "region_fragment_inlined_unoptimized"
    ) in generated_source
    assert "production shape-specialized scan schedule" in generated_source
    assert "Path C scan kernel expects pre-projected x/B/C/z/A/dt" in generated_source
    assert (
        "mapped-packed Path C kernels expect projected conv_input/xf/projected"
        in generated_source
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
        assert descriptor.production_fragment_status == (
            "region_fragment_inlined_unoptimized"
        )
        assert "explicit descriptor" in descriptor.production_fragment_reason


def test_incomplete_production_fragments_are_attested_as_scaffold() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    acceptance_registry = PathCFusionScheduleRegistry(
        acceptance_profiles=(
            PathCFusionScheduleAcceptanceProfile(
                op_signature=tuple(node.op_name for node in region.nodes),
                schedule_id=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID,
                schedule_name=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_NAME,
                schedule_status=MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_STATUS,
                implementation_kind="production",
                missing_reason="acceptance fixture for scaffold attestation",
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
    assert target.implementation_kind == "scaffold"

    planned = plan_path_c_fusion_schedule_for_region(
        region,
        include_backward=True,
        registry=acceptance_registry,
    )
    assert planned.schedule_target is not None
    assert planned.schedule_target.implementation_kind == "scaffold"
    assert planned.plan.schedule_contract is not None
    assert (
        planned.plan.schedule_contract.status
        == "attested_non_production_schedule"
    )
    assert planned.plan.schedule_contract.declared_implementation_kind == "scaffold"
    assert planned.plan.schedule_contract.declared_schedule_id == ""


def test_complete_production_fragments_keep_dynamic_target_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        path_c_fusion,
        "_TRUSTED_PRODUCTION_SCHEDULE_IDS",
        frozenset({"dynamic_complete_production_schedule"}),
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
    registry = PathCFusionScheduleRegistry(
        brick_registry=PathCBrickScheduleDescriptorRegistry((descriptor,)),
    )

    target = select_path_c_fusion_schedule_target(region, registry=registry)

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
    assert target.loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    assert "m2rnn_row_phased_production_fragment" in (
        target.required_codegen_steps
    )
    assert target.brick_descriptors[0].production_fragment_status == (
        "production_region_inlined"
    )
    assert target.brick_descriptors[0].op_name == "m2rnn"


def test_mamba3_fp8_train_descriptor_schedule_uses_loop_fragments() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    cfg = local_gb10_quarter_profile().hybrid_config()
    activation_extent = cfg.max_seq_length * cfg.hidden_size

    assert "# loop_policy: row_phased_hidden" in generated_source
    assert "# internal_buffer_policy: row_local_hidden" in generated_source
    assert f"for row in T.serial(0, {cfg.max_seq_length}):" in generated_source
    assert (
        f"for i in T.serial(row * {cfg.hidden_size}, "
        f"(row + 1) * {cfg.hidden_size}):"
    ) in generated_source
    assert "# backward_policy: flat_after_row_phased_forward" in generated_source
    assert generated_source.count(f"for row in T.serial(0, {cfg.max_seq_length}):") == 1
    assert f"for i in T.serial(0, {activation_extent}):" in generated_source
    assert 'mamba3_scan_mamba3_proj = T.alloc_local((1,), "float32")' in (
        generated_source
    )
    assert "mamba3_scan_mamba3_proj[0]" in generated_source
    assert f"q_fp8[i % {cfg.hidden_size}]" in generated_source
    assert "attention_out[i]" in generated_source
    assert "mamba3_scan_bwd_grad_accum[0]" in generated_source


def test_mamba3_fp8_train_descriptor_schedule_uses_row_local_internal_arrays_without_full_staging() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    cfg = local_gb10_quarter_profile().hybrid_config()
    activation_extent = cfg.max_seq_length * cfg.hidden_size

    assert generated_source.count(f"for row in T.serial(0, {cfg.max_seq_length}):") == 1
    assert (
        generated_source.count(
            f"for i in T.serial(0, {activation_extent}):"
        )
        == 1
    )
    assert (
        f'mamba3_delta = T.alloc_local(({activation_extent},), "float32")'
        not in generated_source
    )
    assert (
        f'q_fp8 = T.alloc_local(({activation_extent},), "float32")'
        not in generated_source
    )
    assert (
        f'mamba3_delta = T.alloc_local(({cfg.hidden_size},), "float32")'
        in generated_source
    )
    assert (
        f'q_fp8 = T.alloc_local(({cfg.hidden_size},), "float32")'
        in generated_source
    )
    assert f"mamba3_delta[i % {cfg.hidden_size}]" in generated_source
    assert f"q_fp8[i % {cfg.hidden_size}]" in generated_source
    assert "attention_out[i]" in generated_source


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
    assert 'route_0_M_delta = T.alloc_local((32,), "float32")' in generated_source
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
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    assert (
        prim_func._cppmega_path_c_loop_policy
        == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    )
    assert "# loop_policy: row_phased_hidden" in generated_source
    assert "for i in T.serial(0, 4096):" not in generated_source
    assert "for row in T.serial(0, 128):" in generated_source
    assert "for i in T.serial(row * 32, (row + 1) * 32):" in generated_source
    assert "route_0_M_residual_norm_inv_rms = T.alloc_local" not in generated_source
    assert (
        "route_0_M_residual_norm_row_sum_sq[0] = "
        "route_0_M_residual_norm_row_sum_sq[0] + "
    ) in generated_source
    assert (
        "route_0_M_residual_norm_row_inv_rms[0] = "
        "T.rsqrt((route_0_M_residual_norm_row_sum_sq[0] / 32.0) + 0.00001)"
    ) in generated_source
    assert generated_source.index("# route_0_M: mamba3_mimo") < (
        generated_source.index("# route_0_M_residual_norm: residual_rmsnorm")
    )
    assert generated_source.index("# route_0_M_residual_norm: residual_rmsnorm") < (
        generated_source.index("# route_1_R: m2rnn")
    )


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
    assert (
        "route_0_M_residual_norm_weight_grad[i % 32] = "
        "route_0_M_residual_norm_weight_grad[i % 32] + "
    ) in generated_source
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

    assert f"hidden: T.Buffer(({hidden_extent},), \"float32\")" in generated_source
    assert (
        "attention_q_proj_weight: "
        f"T.Buffer(({attention_weight_extent},), \"float32\")"
    ) in generated_source
    assert f"sparse_mla_sinks: T.Buffer(({sinks_extent},), \"float32\")" in (
        generated_source
    )
    assert prim_func._cppmega_path_c_buffer_extent == sequence_extent
    assert prim_func._cppmega_path_c_loop_extent == hidden_extent
    assert f"for i in T.serial(0, {hidden_extent}):" in generated_source
    assert prim_func._cppmega_path_c_buffer_abi_shapes["hidden"] == (hidden_extent,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "attention_q_proj_weight"
    ] == (
        attention_weight_extent,
    )


def test_mamba3_fp8_train_descriptor_loop_covers_flat_activation_extent() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source
    cfg = local_gb10_quarter_profile().hybrid_config()
    sequence_extent = cfg.max_seq_length
    activation_extent = cfg.max_seq_length * cfg.hidden_size

    assert prim_func._cppmega_path_c_buffer_extent == sequence_extent
    assert prim_func._cppmega_path_c_loop_extent == activation_extent
    assert f"for i in T.serial(0, {activation_extent}):" in generated_source
    assert f"for i in T.serial(0, {sequence_extent}):" not in generated_source
    assert "hidden[i]" in generated_source
    assert "attention_out[i]" in generated_source
    assert "mamba3_residual_to_m2rnn_norm_weight[i % 3584]" in generated_source


def test_mamba3_fp8_train_descriptor_projection_inputs_are_not_duplicated() -> None:
    prim_func = mamba3_fp8_train_fusion_schedule_template(
        build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    )
    generated_source = prim_func._cppmega_path_c_generated_source

    mamba3_project_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_proj[0] =" in line
    )
    mamba3_conv_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_conv[0] =" in line
    )
    mamba3_dt_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_dt[0] =" in line
    )
    mamba3_state_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_state[0] =" in line
    )
    mamba3_out_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_scan_mamba3_out[0] =" in line
    )
    mamba3_delta_line = next(
        line
        for line in generated_source.splitlines()
        if "mamba3_delta[" in line
        and "mamba3_scan_mamba3_out[0]" in line
    )
    m2rnn_project_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_projected[0] =" in line
    )
    m2rnn_conv_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_conv[0] =" in line
    )
    m2rnn_decay_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_xf[0] =" in line
    )
    m2rnn_recurrent_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_recurrent[0] =" in line
    )
    m2rnn_post_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_m2rnn_post[0] =" in line
    )
    m2rnn_output_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_delta[" in line
        and "m2rnn_packed_post_m2rnn_post[0]" in line
    )
    attention_q_project_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_q_projected[0] =" in line
    )
    attention_kv_project_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_attention_kv_projected[0] =" in line
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
        if "q_scale[" in line
        and "attention_qkv_projection_attention_q_prepared[0]" in line
    )
    attention_kv_scale_line = next(
        line
        for line in generated_source.splitlines()
        if "kv_scale[" in line
        and "attention_qkv_projection_attention_kv_prepared[0]" in line
    )
    attention_q_fp8_line = next(
        line
        for line in generated_source.splitlines()
        if "q_fp8[" in line
        and "attention_qkv_projection_attention_q_prepared[0]" in line
    )
    attention_kv_fp8_line = next(
        line
        for line in generated_source.splitlines()
        if "kv_fp8[" in line
        and "attention_qkv_projection_attention_kv_prepared[0]" in line
    )
    attention_output_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_out[" in line
        and "sparse_mla_fp8_apply_sink_enabled" in line
    )
    attention_bwd_q_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_bwd_attention_q_grad[0] =" in line
    )
    attention_bwd_kv_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_qkv_projection_bwd_attention_kv_grad[0] =" in line
    )
    attention_hidden_grad_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_hidden_grad[" in line
        and "attention_qkv_projection_bwd_attention_q_grad[0]" in line
    )
    attention_q_weight_grad_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_q_proj_weight_grad[" in line
        and "attention_qkv_projection_bwd_attention_q_grad[0]" in line
    )
    attention_kv_weight_grad_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_sparse_kv_proj_weight_grad[" in line
        and "attention_qkv_projection_bwd_attention_kv_grad[0]" in line
    )
    attention_rope_grad_line = next(
        line
        for line in generated_source.splitlines()
        if "attention_rope_inv_freq_grad[" in line
        and "attention_qkv_projection_bwd_attention_rope_grad[0]" in line
    )
    m2rnn_bwd_project_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_bwd_m2rnn_project_grad[0] =" in line
    )
    m2rnn_bwd_conv_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_bwd_m2rnn_conv_grad[0] =" in line
    )
    m2rnn_bwd_recurrent_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_bwd_m2rnn_recurrent_grad[0] =" in line
    )
    m2rnn_bwd_post_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_packed_post_bwd_m2rnn_post_grad[0] =" in line
    )
    m2rnn_bwd_hidden_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_hidden_grad[" in line
        and "m2rnn_packed_post_bwd_m2rnn_project_grad[0]" in line
    )
    m2rnn_bwd_conv_weight_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_conv_weight_grad[" in line
        and "m2rnn_packed_post_bwd_m2rnn_conv_grad[0]" in line
    )
    m2rnn_bwd_state_weight_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_state_weight_grad[" in line
        and "m2rnn_packed_post_bwd_m2rnn_recurrent_grad[0]" in line
    )
    m2rnn_bwd_out_proj_line = next(
        line
        for line in generated_source.splitlines()
        if "m2rnn_out_proj_weight_grad[" in line
        and "m2rnn_packed_post_bwd_m2rnn_post_grad[0]" in line
    )

    assert mamba3_project_line.count("mamba3_in_proj_weight") == 1
    assert mamba3_project_line.count("mamba3_out_proj_weight") == 0
    assert mamba3_project_line.count("mamba3_conv_weight") == 0
    assert mamba3_project_line.count("mamba3_h0") == 0
    assert "mamba3_conv_weight" in mamba3_conv_line
    assert "mamba3_conv_bias" in mamba3_conv_line
    assert "mamba3_dt_bias" in mamba3_dt_line
    assert "mamba3_D" in mamba3_state_line
    assert "mamba3_h0" in mamba3_state_line
    assert "mamba3_B_norm_weight" in mamba3_state_line
    assert "mamba3_C_norm_weight" in mamba3_out_line
    assert "mamba3_out_proj_weight" in mamba3_out_line
    assert "mamba3_scan_mamba3_out[0]" in mamba3_delta_line
    assert m2rnn_project_line.count("m2rnn_in_proj_weight") == 1
    assert m2rnn_project_line.count("m2rnn_conv_weight") == 0
    assert m2rnn_project_line.count("m2rnn_state_weight") == 0
    assert m2rnn_project_line.count("m2rnn_h0") == 0
    assert m2rnn_project_line.count("m2rnn_D") == 0
    assert m2rnn_project_line.count("m2rnn_out_proj_weight") == 0
    assert "m2rnn_conv_weight" in m2rnn_conv_line
    assert "m2rnn_conv_bias" in m2rnn_conv_line
    assert "m2rnn_conv_state" in m2rnn_conv_line
    assert "m2rnn_A_log" in m2rnn_decay_line
    assert "m2rnn_dt_bias" in m2rnn_decay_line
    assert "m2rnn_state_weight" in m2rnn_recurrent_line
    assert "m2rnn_h0" in m2rnn_recurrent_line
    assert "m2rnn_D" in m2rnn_post_line
    assert "m2rnn_g_norm_weight" in m2rnn_output_line
    assert "m2rnn_out_proj_weight" in m2rnn_output_line
    assert attention_q_project_line.count("attention_q_proj_weight") == 1
    assert attention_q_project_line.count("attention_q_proj_bias") == 1
    assert attention_q_project_line.count("attention_sparse_kv_proj_weight") == 0
    assert attention_q_project_line.count("attention_rope_inv_freq") == 0
    assert attention_q_project_line.count("attention_out_proj_weight") == 0
    assert attention_kv_project_line.count("attention_sparse_kv_proj_weight") == 1
    assert attention_kv_project_line.count("attention_sparse_kv_proj_bias") == 1
    assert attention_kv_project_line.count("attention_q_proj_weight") == 0
    assert attention_kv_project_line.count("attention_rope_inv_freq") == 0
    assert attention_kv_project_line.count("attention_out_proj_weight") == 0
    assert attention_rope_line.count("attention_rope_inv_freq") == 1
    assert attention_q_prepare_line.count("attention_qkv_projection_attention_q_projected[0]") == 1
    assert attention_q_prepare_line.count("attention_qkv_projection_attention_rope_phase[0]") == 1
    assert attention_kv_prepare_line.count("attention_qkv_projection_attention_kv_projected[0]") == 1
    assert attention_kv_prepare_line.count("attention_qkv_projection_attention_rope_phase[0]") == 1
    assert "attention_out_proj_weight" not in attention_q_scale_line
    assert "attention_out_proj_bias" not in attention_kv_scale_line
    assert "attention_qkv_projection_attention_q_prepared[0]" in attention_q_fp8_line
    assert "attention_qkv_projection_attention_kv_prepared[0]" in attention_kv_fp8_line
    assert "attention_out_proj_weight" in attention_output_line
    assert "attention_out_proj_bias" in attention_output_line
    assert "attention_qkv_projection_attention_projected" not in generated_source
    assert "q_fp8_grad" in attention_bwd_q_line
    assert "q_scale_grad" in attention_bwd_q_line
    assert "kv_fp8_grad" in attention_bwd_kv_line
    assert "kv_scale_grad" in attention_bwd_kv_line
    assert "attention_qkv_projection_bwd_attention_kv_grad[0]" in attention_hidden_grad_line
    assert "attention_qkv_projection_bwd_attention_q_grad[0]" in attention_q_weight_grad_line
    assert "attention_qkv_projection_bwd_attention_kv_grad[0]" in attention_kv_weight_grad_line
    assert "attention_qkv_projection_bwd_attention_rope_grad[0]" in attention_rope_grad_line
    assert "attention_qkv_projection_bwd_grad_accum" not in generated_source
    assert "m2rnn_delta_grad" in m2rnn_bwd_project_line
    assert "m2rnn_packed_post_bwd_m2rnn_project_grad[0]" in m2rnn_bwd_conv_line
    assert "m2rnn_packed_post_bwd_m2rnn_conv_grad[0]" in m2rnn_bwd_recurrent_line
    assert "m2rnn_packed_post_bwd_m2rnn_recurrent_grad[0]" in m2rnn_bwd_post_line
    assert "m2rnn_packed_post_bwd_m2rnn_recurrent_grad[0]" in m2rnn_bwd_hidden_line
    assert "m2rnn_packed_post_bwd_m2rnn_conv_grad[0]" in m2rnn_bwd_conv_weight_line
    assert "m2rnn_packed_post_bwd_m2rnn_recurrent_grad[0]" in m2rnn_bwd_state_weight_line
    assert "m2rnn_packed_post_bwd_m2rnn_post_grad[0]" in m2rnn_bwd_out_proj_line
    assert "m2rnn_packed_post_bwd_grad_accum" not in generated_source


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


@pytest.mark.skipif(
    os.environ.get("CPPMEGA_RUN_NATIVE_TILELANG_COMPILE_SMOKE") != "1",
    reason=(
        "native tilelang.compile smoke is opt-in because it loads the native "
        "Triton frontend and should run only when explicitly requested"
    ),
)
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


@pytest.mark.skipif(
    os.environ.get("CPPMEGA_RUN_NATIVE_TILELANG_COMPILE_SMOKE") != "1",
    reason=(
        "native tilelang.compile smoke is opt-in because it loads the native "
        "Triton frontend and should run only when explicitly requested"
    ),
)
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
    assert region.node_names[:3] == (
        "local_gb10_quarter_brick_10_M",
        "local_gb10_quarter_brick_10_M_residual_norm",
        "local_gb10_quarter_brick_11_R",
    )
    schedule_template = mark_path_c_schedule_template_for_region(
        target.schedule_template,
        region,
        implementation_kind=target.implementation_kind,
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
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_contract is not None
    assert compiled.plan.schedule_contract.status == "attested_non_production_schedule"
    assert compiled.plan.schedule_contract.declared_required_real_abi_inputs


@pytest.mark.skipif(
    os.environ.get("CPPMEGA_RUN_NATIVE_TILELANG_COMPILE_SMOKE") != "1",
    reason=(
        "native tilelang.compile smoke is opt-in because it loads the native "
        "Triton frontend and should run only when explicitly requested"
    ),
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


@pytest.mark.skipif(
    os.environ.get("CPPMEGA_RUN_NATIVE_TILELANG_COMPILE_SMOKE") != "1",
    reason=(
        "native tilelang.compile smoke is opt-in because it loads the native "
        "Triton frontend and should run only when explicitly requested"
    ),
)
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
        == "attested_non_production_schedule"
    )
    assert (
        scheduled.plan.schedule_contract.declared_implementation_kind
        == "scaffold"
    )
    assert scheduled.plan.schedule_contract.declared_schedule_id == ""
    assert scheduled.schedule_spec.contract_key == scheduled.plan.schedule_contract.key
    assert scheduled.schedule_spec.schedule_id == MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert scheduled.schedule_spec.schedule_name == scheduled.plan.schedule_name
    assert scheduled.schedule_spec.implementation_kind == "scaffold"
    assert scheduled.schedule_spec.real_abi_contract_complete is True
    assert scheduled.schedule_spec.missing_real_abi_inputs == ()
    assert "attention_out_proj_weight" in (
        scheduled.schedule_spec.required_external_buffers
    )
    assert scheduled.plan.single_kernel_fused is False


def test_path_c_schedule_registry_selects_dynamic_descriptor_chain_by_default() -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region(include_backward=True)
    target = select_path_c_fusion_schedule_target(region)

    assert target is not None
    assert target.schedule_id.startswith("path_c_descriptor_chain_")
    assert target.schedule_id != MAMBA3_FP8_TRAIN_FUSION_SCHEDULE_ID
    assert target.schedule_name == (
        "mamba3_m2rnn_attention_fp8_train_block:descriptor_generated_fwd_bwd"
    )
    assert target.schedule_status == "descriptor_codegen_scaffold"
    assert target.implementation_kind == "prototype"
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
    prim_func = target.schedule_template(custom_region)
    generated_source = prim_func._cppmega_path_c_generated_source
    assert "# custom_mamba3_scan: mamba3_mimo" in generated_source
    assert "custom_mamba3_scan_mamba3_proj" in generated_source
    assert "# custom_mamba3_scan_bwd: mamba3_mimo_bwd" in generated_source


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
    assert "hidden: T.Buffer((4096,), \"float32\")" in generated_source
    assert prim_func._cppmega_path_c_buffer_extent == 128
    assert prim_func._cppmega_path_c_loop_extent == 4096
    assert target.internal_buffer_policy == DESCRIPTOR_INTERNAL_BUFFER_POLICY_ROW_LOCAL_HIDDEN
    assert target.loop_policy == DESCRIPTOR_LOOP_POLICY_ROW_PHASED_HIDDEN
    assert "# loop_policy: row_phased_hidden" in generated_source
    assert "for row in T.serial(0, 128):" in generated_source
    assert "for i in T.serial(row * 32, (row + 1) * 32):" in generated_source
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
    assert target.implementation_kind == "prototype"
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
        "route_1_R_residual_norm_weight"
    ] == (32,)
    assert prim_func._cppmega_path_c_buffer_abi_shapes[
        "route_2_A_sparse_mla_fp8_apply_sparse_mla_sinks"
    ] == (4,)
    generated_source = prim_func._cppmega_path_c_generated_source
    assert "route_0_M_mamba3_in_proj_weight" in generated_source
    assert "route_1_R_m2rnn_in_proj_weight" in generated_source
    assert "route_2_A_qkv_projection_attention_q_proj_weight" in generated_source
    assert "route_0_M_mamba3_D[i % 4]" in generated_source
    assert "route_1_R_m2rnn_D[i % 32]" in generated_source
    assert "route_2_A_qkv_projection_q_fp8[i % 32]" in generated_source
    assert "route_2_A_qkv_projection_q_scale[i % 32]" in generated_source
    assert "route_2_A_sparse_mla_fp8_apply_sparse_mla_sm_scale[0]" in generated_source
    assert "route_2_A_sparse_mla_fp8_apply_out[i]" in generated_source


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
    assert optimized.plan.schedule_status == "descriptor_codegen_scaffold"
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


def test_path_c_schedule_optimizer_compile_keeps_dynamic_target_nonproduction() -> None:
    fwd_region = build_mamba3_fp8_train_acceptance_fixture_region()
    optimizer = PathCFusionScheduleOptimizer(
        "mamba3_m2rnn_attention_fp8_train_block",
        enable_aot_autograd=True,
        metadata=fwd_region.metadata,
    ).add_kernels(_surface_copies_for_region(fwd_region))

    compiled = optimizer.compile(tilelang_lowerer=lambda *args, **kwargs: "compiled")

    assert isinstance(compiled, CompiledPathCRegion)
    assert compiled.artifact == "compiled"
    assert compiled.plan.single_kernel_fused is False
    assert compiled.plan.schedule_status == "descriptor_codegen_scaffold"
    assert compiled.plan.schedule_contract is not None
    assert (
        compiled.plan.schedule_contract.status
        == "attested_non_production_schedule"
    )
    assert (
        compiled.plan.schedule_contract.declared_implementation_kind
        == "prototype"
    )
    assert compiled.plan.schedule_contract.declared_schedule_id == ""


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
            declared_required_real_abi_inputs=("hidden",),
        ),
    )
    assert fused_path_c_plan_default_eligible(ready_plan, clean_win) is True

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
    assert plan.schedule_contract.status == "blocked_semantic_contract"


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


def test_compile_path_c_region_keeps_untrusted_production_schedule_non_production() -> None:
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
    assert compiled.plan.schedule_contract.status == "untrusted_production_schedule"
    assert compiled.plan.schedule_contract.declared_key == compiled.plan.schedule_contract.key
    assert compiled.plan.schedule_contract.declared_implementation_kind == "production"
    assert compiled.plan.schedule_contract.declared_schedule_id == "untrusted_toy_schedule"


def test_compile_path_c_region_marks_trusted_production_schedule_contract_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    monkeypatch.setattr(
        path_c_fusion,
        "_TRUSTED_PRODUCTION_SCHEDULE_IDS",
        frozenset({"trusted_test_schedule"}),
    )
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


def test_compile_path_c_region_blocks_trusted_production_missing_real_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    monkeypatch.setattr(
        path_c_fusion,
        "_TRUSTED_PRODUCTION_SCHEDULE_IDS",
        frozenset({"trusted_missing_abi_test_schedule"}),
    )
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
    region = build_mamba3_fp8_train_region()

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


def test_compile_path_c_region_can_use_standard_tilelang_single_entry_lowerer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = build_mamba3_fp8_train_acceptance_fixture_region()
    captured: dict[str, object] = {}

    def fake_compile(prim_func: object, *, target: str, execution_backend: str) -> str:
        captured["prim_func"] = prim_func
        captured["target"] = target
        captured["execution_backend"] = execution_backend
        return "compiled-single-entry-primfunc"

    monkeypatch.setattr(path_c_fusion, "_compile_tilelang_prim_func", fake_compile)
    monkeypatch.setattr(
        path_c_fusion,
        "_TRUSTED_PRODUCTION_SCHEDULE_IDS",
        frozenset({"trusted_real_lowerer_test_schedule"}),
    )

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
        tilelang_lowerer=tilelang_single_entry_lowerer,
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


def test_compile_plan_propagates_ready_single_kernel_tilelang_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        path_c_fusion,
        "_tilelang_compile_plan_for",
        lambda _region, **_kwargs: fake_tilelang_plan,
    )

    plan = compile_path_c_region(region)

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
