"""Step 7: static per-segment estimator + derived chunking + bisection."""

from __future__ import annotations

import math

import pytest

from cppmega_mlx.runtime import path_c_fusion_schedules as sched
from cppmega_mlx.runtime import path_c_segment_estimator as est_mod
from cppmega_mlx.runtime.path_c_device_caps import device_caps
from cppmega_mlx.runtime.path_c_fusion import PathCModelShapeEnv


def _local_gb10_quarter_env() -> PathCModelShapeEnv:
    return PathCModelShapeEnv(
        sequence_length=4096,
        hidden_size=3584,
        attention_num_q_heads=28,
        attention_num_kv_heads=4,
        attention_head_dim=128,
        attention_sparse_topk=64,
        mamba_expand=2,
        mamba_head_dim=128,
        mamba_state_dim=128,
        mamba_groups=1,
        mamba_mimo_rank=1,
        mamba_is_mimo=True,
        mamba_conv_kernel=4,
        mamba_rope_fraction=0.5,
        m2rnn_k_head_dim=128,
        m2rnn_v_head_dim=128,
        m2rnn_num_q_heads=8,
        m2rnn_num_k_heads=8,
        m2rnn_num_v_heads=8,
        m2rnn_num_f_heads=8,
        m2rnn_num_g_heads=8,
        m2rnn_num_weight_heads=8,
        m2rnn_conv_kernel=4,
    )


def test_shared_bytes_physical_uses_margin():
    caps = device_caps()
    src = (
        '    a = T.alloc_shared((1000,), "float32")\n'
        '    b = T.alloc_shared((500,), "float32")\n'
    )
    logical, physical = est_mod.shared_bytes_from_source(
        src,
        caps,
        alloc_shared_re=sched._ALLOC_SHARED_LINE_RE,
        dtype_nbytes=sched._DTYPE_NBYTES,
        flattened_extent=sched._flattened_extent,
    )
    assert logical == (1000 + 500) * 4
    assert physical == int(math.ceil(logical * caps.logical_to_physical_shared_margin))


def test_per_op_time_direct_coefficient():
    caps = device_caps()
    if caps.backend != "metal":
        pytest.skip("preset coefficients are M4 Max specific")
    env = _local_gb10_quarter_env()
    # direct coefficient 10/4096 * 4096 == 10.0
    t = est_mod.est_op_gpu_time_s("attention_qkv_projection_bwd", env, caps)
    assert abs(t - 10.0) < 1e-6
    # light op under budget
    t_light = est_mod.est_op_gpu_time_s("residual_rmsnorm_bwd", env, caps)
    assert t_light < caps.watchdog_window_s * caps.safety_margin


def test_derived_max_rows_lands_on_64_at_calibration_scale():
    caps = device_caps()
    if caps.backend != "metal":
        pytest.skip("watchdog row-window is Metal-only")
    env = _local_gb10_quarter_env()
    t = est_mod.est_op_gpu_time_s("attention_qkv_projection_bwd", env, caps)
    est = est_mod.SegmentEstimate(
        logical_shared_bytes=0,
        physical_shared_bytes=0,
        buffer_arg_count=1,
        est_gpu_time_s=t,
        is_recurrent=False,
        msl_source_bytes=1000,
        per_row_time_s=t / env.sequence_length,
    )
    # the heavy op IS over the watchdog budget -> must chunk
    assert est.est_gpu_time_s > caps.watchdog_window_s * caps.safety_margin
    # the watchdog-safe bound is ~1024; capped by the descriptor default 64
    safe = est_mod.watchdog_safe_max_rows_per_launch(est, caps)
    assert safe is not None and safe >= 64
    assert est_mod.derived_max_rows_per_launch(est, caps) == 64


def test_derived_time_chunk_count_for_recurrent():
    caps = device_caps()
    if caps.backend != "metal":
        pytest.skip("time-chunking is Metal-only")
    # a recurrent op whose monolithic time is 10 s with a 2.5 s budget -> 4 chunks
    est = est_mod.SegmentEstimate(0, 0, 1, 10.0, True, 1000, 0.0)
    assert est_mod.derived_time_chunk_count(est, caps) == math.ceil(
        10.0 / (caps.watchdog_window_s * caps.safety_margin)
    )


def test_no_throughput_and_no_coeff_returns_zero_not_guess():
    # RULE #1: a missing coefficient + zero throughput must NOT fabricate a time.
    caps = device_caps()
    env = _local_gb10_quarter_env()
    import dataclasses

    caps_no_roof = dataclasses.replace(
        caps, per_op_time_per_row_s={}, effective_flop_s=0.0, effective_bytes_s=0.0
    )
    assert est_mod.est_op_gpu_time_s("attention_qkv_projection_bwd", env, caps_no_roof) == 0.0


def test_bisection_finds_largest_survivor():
    from scripts.calibrate_path_c_device import bisect_threshold

    # survives(x) True for x <= 5.0
    result = bisect_threshold(lambda x: x <= 5.0, lo=1.0, hi=10.0, rel_tol=0.01)
    assert 4.9 <= result <= 5.0


def test_bisection_precondition_raises():
    from scripts.calibrate_path_c_device import bisect_threshold

    with pytest.raises(RuntimeError):
        bisect_threshold(lambda x: False, lo=1.0, hi=10.0)  # lo doesn't survive
    with pytest.raises(RuntimeError):
        bisect_threshold(lambda x: True, lo=1.0, hi=10.0)  # hi survives
