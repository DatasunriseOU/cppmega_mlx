from __future__ import annotations

from cppmega_mlx.nn._tilelang._async_barrier_plan import (
    MetalReductionSyncPlan,
    plan_metal_path_c_reduction_sync,
)
from cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c import (
    fp8_sparse_mla_qk_reduce_sync_plan,
)


def test_metal_path_c_reduction_plan_prefers_simdgroup_async_for_full_wave() -> None:
    plan = plan_metal_path_c_reduction_sync(
        outputs_per_block=16,
        reduce_threads=32,
        vec=4,
        k_extent=64,
    )

    assert isinstance(plan, MetalReductionSyncPlan)
    assert plan.strategy == "simdgroup_async"
    assert plan.is_async is True
    assert plan.barrier_count == 0
    assert plan.wait_count == 0
    assert plan.async_stages == 1
    assert plan.reduction_isolated is True
    assert plan.z3_used is True
    assert plan.z3_proved is True


def test_metal_path_c_reduction_plan_keeps_subsimdgroup_schedule_synchronous() -> None:
    plan = plan_metal_path_c_reduction_sync(
        outputs_per_block=4,
        reduce_threads=4,
        vec=4,
        k_extent=64,
    )

    assert plan.strategy == "threadgroup_sync"
    assert plan.is_async is False
    assert plan.barrier_count >= 1
    assert plan.wait_count == plan.barrier_count
    assert plan.async_stages == 0
    assert plan.reduction_isolated is False
    assert "sub-simdgroup" in plan.reason


def test_fp8_sparse_mla_qk_reduce_sync_plan_uses_tuned_default_schedule() -> None:
    plan = fp8_sparse_mla_qk_reduce_sync_plan(N=16, K=64)

    assert plan.outputs_per_block == 16
    assert plan.reduce_threads == 32
    assert plan.vec == 4
    assert plan.strategy == "simdgroup_async"
    assert plan.as_feature_dict()["sync_plan_strategy"] == "simdgroup_async"


def test_fp8_sparse_mla_qk_reduce_sync_plan_preserves_explicit_sync_schedule() -> None:
    plan = fp8_sparse_mla_qk_reduce_sync_plan(
        N=16,
        K=64,
        outputs_per_block=4,
        reduce_threads=4,
        vec=4,
    )

    assert plan.outputs_per_block == 4
    assert plan.reduce_threads == 4
    assert plan.strategy == "threadgroup_sync"
