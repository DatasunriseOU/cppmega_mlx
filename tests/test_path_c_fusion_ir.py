from __future__ import annotations

from dataclasses import replace

import pytest

from cppmega_mlx.runtime.path_c_fusion import (
    BenchmarkAcceptanceRow,
    FusionKernelSurface,
    PathCFusionRegion,
    PathCFusionRegionBuilder,
    PathCFusionMode,
    Z3SyncSpec,
    audit_fusion_cache_key,
    audit_warm_cache_reuse,
    build_mamba3_fp8_train_region,
    compile_path_c_region,
    fused_path_c_plan_default_eligible,
    fused_path_c_default_eligible,
    path_b_baseline_is_clean,
    selected_path_c_fusion_mode,
)


def test_region_builder_creates_fx_like_ir_without_msl_post_fusion() -> None:
    builder = PathCFusionRegionBuilder(
        "hybrid_train_block",
        z3_sync=Z3SyncSpec.minimize_sync_async(),
    )
    builder.add_kernel(
        FusionKernelSurface.path_c(
            name="mamba3_scan",
            op_name="mamba3_mimo",
            inputs=("hidden", "state"),
            outputs=("scan_y", "scan_state"),
            backward="aot_autograd",
        )
    )
    builder.add_kernel(
        FusionKernelSurface.path_c(
            name="packed_post",
            op_name="m2rnn",
            inputs=("scan_y",),
            outputs=("post_y",),
            backward="aot_autograd",
        )
    )
    builder.add_kernel(
        FusionKernelSurface.path_c(
            name="fp8_prepare",
            op_name="sparse_mla_fp8_prepare",
            inputs=("post_y",),
            outputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"),
            backward="owner_output",
        )
    )
    builder.add_kernel(
        FusionKernelSurface.path_c(
            name="sparse_mla_fp8_apply",
            op_name="sparse_mla_fp8_apply",
            inputs=("q_fp8", "q_scale", "kv_fp8", "kv_scale", "indices"),
            outputs=("attention_out", "lse"),
            backward="owner_output",
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
    assert plan.cache_key_parts == (
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
            "fp8_prepare->sparse_mla_fp8_apply:kv_scale:internal,"
            "fp8_prepare->sparse_mla_fp8_apply:indices:internal"
        ),
        "backend:tilelang_tvm_ffi",
        "boundary:tilelang_tvm_ir",
        "z3:sync_async",
    )
    assert plan.fusion_kind == "tilelang_ir_graph_region"
    assert plan.schedule_status == "missing_real_fused_schedule_template"
    assert plan.single_kernel_fused is False
    assert plan.autograd_status == "requires_aot_autograd_codegen"
    assert plan.autograd_missing_backward_nodes == (
        "mamba3_scan_bwd",
        "m2rnn_packed_post_bwd",
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
    )
    assert fused_path_c_plan_default_eligible(ready_plan, clean_win) is True


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
