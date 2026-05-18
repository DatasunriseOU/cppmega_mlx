"""Tests for the Path C TileLang DSL Mamba3 MIMO port.

Path C is the @T.prim_func DSL counterpart to the hand-written MSL kernels in
:mod:`cppmega_mlx.nn._tilelang.mamba3` (Path B). Both paths must be numerically
equivalent at FP32 within atol=1e-4 / rtol=1e-3 and the Path C kernel must
lower cleanly through the patched Apple-head TileLang Metal backend
(``tilelang.engine.lower.lower(target="metal")``).

Coverage:
  - lowering smoke test (cppmega Path C must lower without raising);
  - forward parity vs Path B at FP32 (bit-exact on this hardware);
  - backward parity vs Path B Metal kernel (bit-exact at FP32);
  - VJP-through-mx.custom_function parity vs autograd-through-Path-B-reference
    at FP32 / bench shape;
  - small bf16 carrier shape (FP32 internal accumulator preserves precision).

The "bit-exact" expectation is a property of the M4 Max instance running this
test; the conservative atol/rtol budget is what we ship as the contract.
"""

import json
import re
from typing import cast

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

import cppmega_mlx.nn._tilelang.mamba3_path_c as mamba3_path_c
from cppmega_mlx.nn._tilelang import (
    _msl_transform,
    mamba3_mimo_bwd_metal,
    mamba3_mimo_fwd_metal,
    mamba3_mimo_reference,
)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    Mamba3PathCSchedulePlan,
    Mamba3PathCStatus,
    dump_lowered_bwd_msl,
    dump_lowered_fwd_msl,
    mamba3_mimo_apply_path_c,
    mamba3_mimo_apply_training_path_c,
    mamba3_mimo_apply_with_state_path_c,
    mamba3_mimo_apply_with_state_training_path_c,
    mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd,
    mamba3_mimo_bwd_path_c,
    mamba3_mimo_fwd_path_c,
    mamba3_mimo_path_c_status,
    mamba3_path_c_receipt_allows_auto_promotion,
    mamba3_path_c_receipt_auto_mode,
    mamba3_path_c_schedule_plan,
)


_BWD_PARTIAL_OUTPUT_TOKENS = (
    "dA" + "_partial",
    "dB" + "_partial",
    "dC" + "_partial",
    "dD" + "_partial",
    "ddt" + "_partial",
)


def _np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.asarray(x)


def _make_inputs(
    *,
    batch: int,
    seq: int,
    heads: int,
    headdim: int,
    state: int,
    dtype: mx.Dtype,
    seed: int = 17,
) -> tuple[mx.array, ...]:
    mx.random.seed(seed)
    x = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    B = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    C = (mx.random.normal((batch, seq, heads, state)) * 0.1).astype(dtype)
    z = (mx.random.normal((batch, seq, heads, headdim)) * 0.1).astype(dtype)
    A = (-mx.random.uniform(0.01, 0.5, (batch, seq, heads))).astype(dtype)
    dt = (mx.random.uniform(0.001, 0.05, (batch, seq, heads))).astype(dtype)
    D = mx.ones((heads,), dtype=dtype)
    h0 = mx.zeros((batch, heads, headdim, state), dtype=dtype)
    mx.eval(x, B, C, z, A, dt, D, h0)
    return x, B, C, z, A, dt, D, h0


def _make_mamba3_projection_inputs(
    *,
    d_model: int,
    seq: int,
    dtype: mx.Dtype,
) -> tuple[mx.array, ...]:
    """Build post-projection tensors at the Mamba3 scan contract."""

    from cppmega_mlx.nn.mamba3 import (
        Mamba3Config,
        Mamba3ReferenceBlock,
        _apply_rope_on_state_dim,
        _broadcast_groups_to_heads,
        _compute_trapezoidal_scale,
        _heads_to_group_scale,
        _split_by_sizes,
        causal_depthwise_conv1d,
    )

    cfg = Mamba3Config(
        d_model=d_model,
        expand=2,
        headdim=64,
        d_state=16,
        ngroups=1,
        chunk_size=128,
    )
    block = Mamba3ReferenceBlock(cfg)
    block.update(
        tree_map(
            lambda value: value.astype(dtype) if isinstance(value, mx.array) else value,
            block.parameters(),
        )
    )

    mx.random.seed(0)
    hidden = (mx.random.normal((1, seq, cfg.d_model)) * 0.02).astype(dtype)
    z, x, B, C, dd_dt, dd_A, trap, angles = block.split_in_proj(
        block.in_proj(hidden)
    )

    xBC = mx.concatenate([x, B, C], axis=-1)
    xBC = causal_depthwise_conv1d(
        xBC,
        block.conv_weight.astype(xBC.dtype),
        block.conv_bias.astype(xBC.dtype),
    )
    x, B, C = _split_by_sizes(
        nn.silu(xBC),
        [cfg.d_inner, block.dims.d_bc, block.dims.d_bc],
    )
    x = x.reshape(1, seq, cfg.nheads, cfg.headdim)
    z = z.reshape(1, seq, cfg.nheads, cfg.headdim)

    B_mimo = B.reshape(1, seq, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
    C_mimo = C.reshape(1, seq, cfg.effective_mimo_rank, cfg.ngroups, cfg.d_state)
    B_mimo, C_mimo = block.transform_bc(B_mimo, C_mimo)
    B = mx.mean(B_mimo, axis=2)
    C = mx.mean(C_mimo, axis=2)

    dt = nn.softplus(dd_dt + block.dt_bias.astype(dd_dt.dtype))
    trap_scale = _compute_trapezoidal_scale(dt, trap)
    B = B * _heads_to_group_scale(trap_scale, cfg.ngroups)[:, :, :, None]

    angles = mx.broadcast_to(
        angles[:, :, None, :],
        (1, seq, cfg.nheads, block.dims.num_rope_angles),
    )
    angles_cumsum = mx.cumsum(angles * dt[:, :, :, None], axis=1)
    B = _apply_rope_on_state_dim(B, angles_cumsum)
    C = _apply_rope_on_state_dim(C, angles_cumsum)
    B = _broadcast_groups_to_heads(B, cfg.nheads, "B")
    C = _broadcast_groups_to_heads(C, cfg.nheads, "C")
    A = mx.minimum(-nn.softplus(dd_A), -cfg.A_floor)
    h0 = mx.zeros((1, cfg.nheads, cfg.headdim, cfg.d_state), dtype=dtype)
    return x, B, C, z, A, dt, block.D.astype(dtype), h0


# ---------------------------------------------------------------------------
# Status & lowering smoke tests
# ---------------------------------------------------------------------------


def test_status_reports_available_or_explains_why() -> None:
    status = mamba3_mimo_path_c_status()
    assert isinstance(status, Mamba3PathCStatus)
    assert isinstance(status.available, bool)
    assert isinstance(status.reason, str) and status.reason


def test_status_no_longer_requires_debug_fast_kernel_wrapper_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "CPPMEGA_ENABLE_MAMBA3_PATH_C_FAST_KERNEL_WRAPPER"
    monkeypatch.delenv(
        env_name,
        raising=False,
    )
    status = mamba3_mimo_path_c_status()
    assert env_name not in status.reason


def _require_mamba3_path_c() -> None:
    status = mamba3_mimo_path_c_status()
    if not status.available:
        pytest.skip(f"mamba3 Path C unavailable on this host: {status.reason}")


def _assert_no_bwd_partial_outputs(msl: str) -> None:
    for name in _BWD_PARTIAL_OUTPUT_TOKENS:
        assert name not in msl


def _assert_bwd_partial_outputs(msl: str) -> None:
    for name in _BWD_PARTIAL_OUTPUT_TOKENS:
        assert name in msl


def _kernel_body(msl: str) -> str:
    _prelude, _signature, body = _msl_transform._split_kernel_msl(msl)
    return body


def test_lowered_fwd_msl_contains_kernel_void() -> None:
    """Lowering emits a self-contained MSL kernel string."""

    msl = dump_lowered_fwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert "kernel void" in msl
    # The lowered MSL references each of the alphabetically-ordered buffers.
    for name in ("A", "B", "C", "D", "dt", "h0", "h_last", "x", "y", "z"):
        assert name in msl, f"buffer {name!r} missing from lowered MSL"


def test_fwd_snapshot_route_lowers_h_snap_as_output() -> None:
    """Training fwd can materialize bwd snapshots without a separate kernel."""

    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._fwd_with_snapshots_kernel_for(
        1, 4, 1, 2, 4
    )

    assert "h_snap" in lowering.buffer_param_names
    assert "device float* h_snap" in lowering.msl_text
    assert "h_snap[" in lowering.msl_text


def test_lowered_bwd_msl_contains_kernel_void() -> None:
    msl = dump_lowered_bwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert "kernel void" in msl
    # Production bwd emits partial outputs and consumes h snapshots.
    for name in ("A", "B", "C", "D", "dt", "dy", "h_snap", "x", "z"):
        assert name in msl, f"input buffer {name!r} missing from lowered MSL"
    assert "h_steps" not in msl
    _assert_bwd_partial_outputs(msl)
    for name in ("dh0", "dx", "dz"):
        assert name in msl, f"output buffer {name!r} missing from lowered MSL"


def test_bwd_scratch_route_lowering_owns_scratch_without_h_snap_input() -> None:
    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._bwd_scratch_partial_kernel_for(
        1, 4, 1, 2, 4
    )

    assert "h_steps_scratch" in lowering.buffer_param_names
    assert "h_snap" not in lowering.buffer_param_names
    assert "device float* h_steps_scratch" in lowering.msl_text
    assert "device float* h_snap" not in lowering.msl_text
    _assert_bwd_partial_outputs(lowering.msl_text)


def test_bwd_scratch_route_lowering_compacts_bf16_partials() -> None:
    _require_mamba3_path_c()

    dtype_args = (
        "bfloat16",  # dy
        "bfloat16",  # x
        "bfloat16",  # B
        "bfloat16",  # C
        "bfloat16",  # z
        "bfloat16",  # A
        "bfloat16",  # dt
        "bfloat16",  # D
        "bfloat16",  # h0
        "bfloat16",  # dx
        "bfloat16",  # dz
        "bfloat16",  # dB_partial
        "bfloat16",  # dC_partial
        "float32",  # dA_partial
        "float32",  # ddt_partial
        "float32",  # dD_partial
        "bfloat16",  # dh0
        "bfloat16",  # h_steps_scratch
    )
    _kernel, lowering = mamba3_path_c._bwd_scratch_partial_kernel_for(
        1, 4, 1, 4, 4, *dtype_args
    )

    assert "device tvm_bfloat16* h_steps_scratch" in lowering.msl_text
    assert "device tvm_bfloat16* dB_partial" in lowering.msl_text
    assert "device tvm_bfloat16* dC_partial" in lowering.msl_text
    assert "device float* dA_partial" in lowering.msl_text
    assert "device float* ddt_partial" in lowering.msl_text
    assert "device float* dD_partial" in lowering.msl_text


def test_dump_lowered_bwd_msl_matches_bf16_partial_dtype_policy() -> None:
    _require_mamba3_path_c()

    msl = dump_lowered_bwd_msl(
        batch=1,
        seq=4,
        heads=1,
        headdim=4,
        state=4,
        dtype="bfloat16",
    )

    assert "device tvm_bfloat16* dB_partial" in msl
    assert "device tvm_bfloat16* dC_partial" in msl
    assert "device float* dA_partial" in msl
    assert "device float* ddt_partial" in msl
    assert "device float* dD_partial" in msl


def test_bwd_partial_reduce_kernel_matches_mlx_sum_small_shape() -> None:
    """Standalone TileLang P-reducer matches the MLX reductions it may replace."""

    _require_mamba3_path_c()
    batch, seq, heads, headdim, state = 1, 3, 2, 4, 5
    mx.random.seed(71)
    dB_partial = mx.random.normal((batch, seq, heads, state, headdim)).astype(mx.float32)
    dC_partial = mx.random.normal((batch, seq, heads, state, headdim)).astype(mx.float32)
    dA_partial = mx.random.normal((batch, seq, heads, headdim)).astype(mx.float32)
    ddt_partial = mx.random.normal((batch, seq, heads, headdim)).astype(mx.float32)
    dD_partial = mx.random.normal((batch, heads, headdim)).astype(mx.float32)
    mx.eval(dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial)

    reduced = mamba3_path_c._reduce_bwd_partials_path_c_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
    )
    expected = (
        mx.sum(dB_partial, axis=4),
        mx.sum(dC_partial, axis=4),
        mx.sum(dA_partial, axis=3),
        mx.sum(ddt_partial, axis=3),
        mx.sum(dD_partial, axis=(0, 2)),
    )
    mx.eval(*reduced, *expected)
    for name, got, want in zip(("dB", "dC", "dA", "ddt", "dD"), reduced, expected, strict=True):
        np.testing.assert_allclose(
            _np(got),
            _np(want),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"partial reducer mismatch on {name}",
        )


def test_bwd_partial_reduce_kernel_can_write_bf16_outputs() -> None:
    """Diagnostic reducer can write final BF16 grads without hidden MLX casts."""

    _require_mamba3_path_c()
    batch, seq, heads, headdim, state = 1, 3, 2, 4, 5
    mx.random.seed(74)
    dB_partial = mx.random.normal((batch, seq, heads, state, headdim)).astype(mx.float32)
    dC_partial = mx.random.normal((batch, seq, heads, state, headdim)).astype(mx.float32)
    dA_partial = mx.random.normal((batch, seq, heads, headdim)).astype(mx.float32)
    ddt_partial = mx.random.normal((batch, seq, heads, headdim)).astype(mx.float32)
    dD_partial = mx.random.normal((batch, heads, headdim)).astype(mx.float32)
    mx.eval(dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial)

    reduced = mamba3_path_c._reduce_bwd_partials_path_c_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        output_dtypes=("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"),
    )
    expected = (
        mx.sum(dB_partial, axis=4).astype(mx.bfloat16),
        mx.sum(dC_partial, axis=4).astype(mx.bfloat16),
        mx.sum(dA_partial, axis=3).astype(mx.bfloat16),
        mx.sum(ddt_partial, axis=3).astype(mx.bfloat16),
        mx.sum(dD_partial, axis=(0, 2)).astype(mx.bfloat16),
    )
    mx.eval(*reduced, *expected)

    for got in reduced:
        assert got.dtype == mx.bfloat16
    for name, got, want in zip(("dB", "dC", "dA", "ddt", "dD"), reduced, expected, strict=True):
        np.testing.assert_allclose(
            _np(got),
            _np(want),
            rtol=0,
            atol=0,
            err_msg=f"bf16 partial reducer mismatch on {name}",
        )


def test_bwd_threaded_partial_reduce_kernel_matches_mlx_sum_small_shape() -> None:
    """Threaded P-reducer is a candidate for replacing serial post-reduce."""

    _require_mamba3_path_c()
    batch, seq, heads, headdim, state = 1, 3, 2, 4, 5
    mx.random.seed(75)
    dB_partial = mx.random.normal((batch, seq, heads, state, headdim)).astype(mx.float32)
    dC_partial = mx.random.normal((batch, seq, heads, state, headdim)).astype(mx.float32)
    dA_partial = mx.random.normal((batch, seq, heads, headdim)).astype(mx.float32)
    ddt_partial = mx.random.normal((batch, seq, heads, headdim)).astype(mx.float32)
    dD_partial = mx.random.normal((batch, heads, headdim)).astype(mx.float32)
    mx.eval(dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial)

    reduced = mamba3_path_c._reduce_bwd_partials_path_c_threaded_kernel(
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
    )
    expected = (
        mx.sum(dB_partial, axis=4),
        mx.sum(dC_partial, axis=4),
        mx.sum(dA_partial, axis=3),
        mx.sum(ddt_partial, axis=3),
        mx.sum(dD_partial, axis=(0, 2)),
    )
    mx.eval(*reduced, *expected)
    for name, got, want in zip(("dB", "dC", "dA", "ddt", "dD"), reduced, expected, strict=True):
        np.testing.assert_allclose(
            _np(got),
            _np(want),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"threaded partial reducer mismatch on {name}",
        )


def test_bwd_threaded_partial_reduce_kernel_lowers_thread_reduce() -> None:
    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._bwd_partial_reduce_threaded_kernel_for(
        1, 3, 2, 4, 5
    )

    assert lowering.threadgroup == (4, 256, 1)
    assert "thread_position_in_threadgroup" in lowering.msl_text
    assert "simd_sum" in lowering.msl_text


def test_bwd_partial_outputs_compact_large_reduction_partials_for_bf16() -> None:
    """BF16 bwd stores bandwidth-dominant dB/dC partials as BF16."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=4, heads=2, headdim=4, state=8, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(72)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    (
        dx_pc,
        dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        dh0_pc,
    ) = mamba3_path_c._mamba3_mimo_bwd_path_c_partial_outputs(
        dy, x, B, C, z, A, dt, D, h0
    )
    mx.eval(dx_pc, dz_pc, dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial, dh0_pc)

    assert dx_pc.dtype == mx.bfloat16
    assert dz_pc.dtype == mx.bfloat16
    assert dh0_pc.dtype == mx.bfloat16
    assert dB_partial.dtype == mx.bfloat16
    assert dC_partial.dtype == mx.bfloat16
    assert dA_partial.dtype == mx.float32
    assert ddt_partial.dtype == mx.float32
    assert dD_partial.dtype == mx.float32


def test_bwd_partial_outputs_store_bc_reduction_axis_innermost() -> None:
    """dB/dC partials should make the P reduction contiguous for post-reduce."""

    _require_mamba3_path_c()
    batch, seq, heads, headdim, state = 1, 4, 2, 4, 8
    inputs = _make_inputs(
        batch=batch,
        seq=seq,
        heads=heads,
        headdim=headdim,
        state=state,
        dtype=mx.float32,
    )
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(73)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    (
        _dx_pc,
        _dz_pc,
        dB_partial,
        dC_partial,
        dA_partial,
        ddt_partial,
        dD_partial,
        _dh0_pc,
    ) = mamba3_path_c._mamba3_mimo_bwd_path_c_partial_outputs(
        dy, x, B, C, z, A, dt, D, h0
    )
    mx.eval(dB_partial, dC_partial, dA_partial, ddt_partial, dD_partial)

    assert tuple(dB_partial.shape) == (batch, seq, heads, state, headdim)
    assert tuple(dC_partial.shape) == (batch, seq, heads, state, headdim)
    assert tuple(dA_partial.shape) == (batch, seq, heads, headdim)
    assert tuple(ddt_partial.shape) == (batch, seq, heads, headdim)
    assert tuple(dD_partial.shape) == (batch, heads, headdim)


def test_lowered_bwd_bench_shape_uses_partial_outputs_not_hot_allreduce() -> None:
    msl = dump_lowered_bwd_msl(batch=1, seq=4, heads=1, headdim=32, state=4)
    body = _kernel_body(msl)
    assert "kernel void" in msl
    assert "simd_sum(" not in body
    assert "simd_shuffle_down" not in body
    assert "threadgroup_barrier(" not in body
    assert "red_buf_staging" not in body
    _assert_bwd_partial_outputs(msl)
    assert "dD_partial" in msl


@pytest.mark.parametrize("headdim", [64, 96, 128, 256, 512, 1024])
def test_lowered_bwd_aligned_headdims_avoid_split_thread_allreduce(
    headdim: int,
) -> None:
    msl = dump_lowered_bwd_msl(batch=1, seq=2, heads=1, headdim=headdim, state=2)
    body = _kernel_body(msl)
    assert "kernel void" in msl
    assert "simd_sum(" not in body
    assert "threadgroup_barrier(" not in body
    assert "red_buf_staging" not in body
    _assert_bwd_partial_outputs(msl)
    assert "dD_partial" in msl


@pytest.mark.parametrize(
    ("batch", "heads", "headdim", "threads"),
    [
        (1, 1, 256, 256),
        (1, 1, 512, 512),
        (1, 1, 1024, 1024),
        (1, 4, 64, 256),
    ],
)
def test_bwd_p_reduction_threads_cover_large_aligned_rows(
    batch: int,
    heads: int,
    headdim: int,
    threads: int,
) -> None:
    lanes = batch * heads * headdim
    assert mamba3_path_c._bwd_threads_for(lanes, headdim) == threads
    assert mamba3_path_c._bwd_simd_p_reduction_supported(
        batch=batch,
        heads=heads,
        headdim=headdim,
    )


def test_direct_bwd_simd_lowering_rejects_long_sequences_without_snapshots() -> None:
    with pytest.raises(
        _msl_transform.MSLDispatchUnsupported,
        match="explicit state snapshots",
    ):
        mamba3_path_c._bwd_simd_reduce_kernel_for(1, 2, 1, 64, 2)


def test_bwd_scan_plan_selects_state_snapshots_for_long_reverse_recurrence() -> None:
    plan = mamba3_path_c._bwd_scan_plan_for(
        batch=1,
        seq=8,
        heads=2,
        headdim=64,
        state=16,
    )

    assert plan.direction == "reverse"
    assert plan.snapshot_plan.policy == "state-boundary-cache"
    assert plan.snapshot_plan.snapshot_count == 9
    assert plan.rematerialization_policy == "reuse-forward-state-snapshots"
    assert plan.alias_plan.in_place_allowed is False
    assert plan.host_sync_required is False
    assert plan.device_event_required is False
    assert plan.fused_post_ops == ("skip_D", "silu_gate")


def test_bwd_scan_plan_keeps_single_step_direct_recompute() -> None:
    plan = mamba3_path_c._bwd_scan_plan_for(
        batch=1,
        seq=1,
        heads=1,
        headdim=32,
        state=8,
    )

    assert plan.snapshot_plan.policy == "none"
    assert plan.rematerialization_policy == "direct-recompute"


def test_metal_simd_sum_op_has_tscript_printer_name() -> None:
    from tilelang.language.fp8_op import assert_metal_fp8_intrinsics_registered

    assert_metal_fp8_intrinsics_registered(["tirx.metal.simd_sum"])
    try:
        from tilelang.tvm.ir import Op  # type: ignore
    except Exception:
        try:
            from tvm.ir import Op  # type: ignore
        except Exception as exc:
            pytest.skip(f"TVM unavailable: {exc}")

    op = Op.get("tirx.metal.simd_sum")
    assert op.has_attr("TScriptPrinterName")
    assert str(op.get_attr("TScriptPrinterName")) == "metal_simd_sum"


def test_lowered_msl_reuses_hot_scalar_temporaries() -> None:
    """TileLang CSE plus scalar binding reuse avoids hot exp/sigmoid recompute."""

    fwd = dump_lowered_fwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert "float y_acc = " in fwd
    assert "thread float y_acc[1]" not in fwd
    assert "gridThreadIdx [[thread_position_in_grid]]" in fwd
    assert "blockIdx.x) * 256) + (((int)threadIdx.x)" not in fwd
    assert len(re.findall(r"\bdecay = exp\(", fwd)) == 1
    assert re.search(r"exp\([^;\n]+\) \* h_state", fwd) is None
    assert fwd.index("float D_h =") < fwd.index("for (int t =")
    assert "if (0.000000e+00f <= z_val)" not in fwd
    assert "sig_z = exp(z_val)" not in fwd
    assert "1.000000e+00f + exp(" in fwd
    assert re.search(r"z_val \* \([^;\n]+exp\(", fwd) is None

    bwd = dump_lowered_bwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert "gridThreadIdx [[thread_position_in_grid]]" in bwd
    assert "blockIdx.x) * 256) + (((int)threadIdx.x)" not in bwd
    assert len(re.findall(r"\bdecay = exp\(", bwd)) == 1
    assert "h_snap" in bwd
    assert "1.000000e+00 / decay" not in bwd
    assert re.search(r"d_decay\[0\] \* exp\(", bwd) is None
    assert re.search(r"dh\[n_\d+\] = \(dh\[n_\d+\] \* exp\(", bwd) is None
    assert "if (0.000000e+00f <= z_val)" not in bwd
    assert "sig_z = exp(z_val)" not in bwd
    assert "1.000000e+00f + exp(" in bwd
    assert re.search(r"dY \* \(z_val \* \([^;\n]+exp\(", bwd) is None
    assert "y_state = sig_z" not in bwd
    assert "dx_inp = sig_z" not in bwd
    assert "d_decay = sig_z" not in bwd
    assert "sig_z = sig_z" not in bwd


def test_runtime_fwd_msl_reuses_decay_scalar_without_dump_postprocess() -> None:
    """The tvm-ffi fwd runtime source must not rely on report-only MSL cleanup."""

    _require_mamba3_path_c()
    _fwd_kernel, fwd_lowering = mamba3_path_c._fwd_kernel_for(1, 4, 1, 2, 4)

    assert re.search(r"\bdecay = exp\(", fwd_lowering.msl_text)
    assert re.search(r"new_h = [^;\n]*exp\(", fwd_lowering.msl_text) is None


def test_runtime_fwd_msl_omits_signed_grid_mod_corrections_for_full_lane_shape() -> None:
    """Full-shape lane decomposition must use unsigned Metal thread ids."""

    _require_mamba3_path_c()
    _fwd_kernel, fwd_lowering = mamba3_path_c._fwd_kernel_for(2, 2048, 112, 64, 64)
    body = fwd_lowering.msl_text.split("kernel void", 1)[1]

    assert "gridThreadIdx.x % 7168" in body
    assert "gridThreadIdx.x / 7168" in body
    assert ">> 31" not in body
    assert ">>31" not in body
    assert "7168 &" not in body


def test_runtime_fwd_msl_specializes_batch_one_index_base_for_full_lane_shape() -> None:
    """Batch-one forward should not recompute a zero batch index in the T loop."""

    _require_mamba3_path_c()
    _fwd_kernel, fwd_lowering = mamba3_path_c._fwd_kernel_for(1, 2048, 112, 64, 64)
    body = fwd_lowering.msl_text.split("kernel void", 1)[1]
    t_loop = body.split("for (int t = 0; t < 2048; ++t)", 1)[1]

    assert "gridThreadIdx.x % 7168" not in body
    assert "gridThreadIdx.x / 7168" not in t_loop
    assert "* 14680064" not in t_loop


def test_runtime_bwd_msl_reuses_decay_scalar_without_dump_postprocess() -> None:
    """The tvm-ffi bwd runtime source must not rely on report-only MSL cleanup."""

    _require_mamba3_path_c()
    _bwd_kernel, bwd_lowering = mamba3_path_c._bwd_partial_kernel_for_state_snapshots(
        1, 4, 1, 2, 4
    )

    assert re.search(r"\bdecay = exp\(", bwd_lowering.msl_text)
    assert re.search(r"dh\[[^\]]+\] = [^;\n]*exp\(", bwd_lowering.msl_text) is None
    assert re.search(r"d_logdecay = [^;\n]*exp\(", bwd_lowering.msl_text) is None


def test_runtime_bwd_msl_specializes_batch_one_index_base_for_full_lane_shape() -> None:
    """Batch-one backward should not recompute a zero batch index in the T loop."""

    _require_mamba3_path_c()
    dtype_args = ("bfloat16",) * 18
    _bwd_kernel, bwd_lowering = mamba3_path_c._bwd_partial_kernel_for_state_snapshots(
        1,
        2048,
        112,
        64,
        64,
        *dtype_args,
    )
    body = bwd_lowering.msl_text.split("kernel void", 1)[1]
    rt_loop = body.split("for (int rt = 0; rt < 2048; ++rt)", 1)[1]

    assert "gridThreadIdx.x % 7168" not in body
    assert "gridThreadIdx.x / 7168" not in rt_loop
    assert "* 14680064" not in rt_loop


def test_runtime_bwd_msl_elides_full_lane_guard_for_aligned_model_shape() -> None:
    """Aligned 1B bwd kernels should not branch around every recurrent step."""

    _require_mamba3_path_c()
    _snap_kernel, snap_lowering = mamba3_path_c._bwd_state_snapshots_kernel_for(
        1, 2048, 112, 64, 64
    )
    _partial_kernel, partial_lowering = mamba3_path_c._bwd_partial_kernel_for_state_snapshots(
        1, 2048, 112, 64, 64
    )
    _scratch_kernel, scratch_lowering = mamba3_path_c._bwd_scratch_partial_kernel_for(
        1, 2048, 112, 64, 64
    )

    for lowering in (snap_lowering, partial_lowering, scratch_lowering):
        body = lowering.msl_text.split("kernel void", 1)[1]
        assert "gridThreadIdx.x < 7168" not in body

    for lowering in (partial_lowering, scratch_lowering):
        hot_loop = lowering.msl_text.split("for (int rt = 0; rt < 2048; ++rt)", 1)[1]
        assert "gridThreadIdx.x % 64" not in hot_loop


def test_runtime_bwd_msl_keeps_lane_guard_for_unaligned_shape() -> None:
    """The exact-lane cleanup must not remove OOB protection on ragged grids."""

    _require_mamba3_path_c()
    _partial_kernel, partial_lowering = mamba3_path_c._bwd_partial_kernel_for_state_snapshots(
        1, 4, 5, 65, 4
    )
    body = partial_lowering.msl_text.split("kernel void", 1)[1]

    assert "gridThreadIdx.x < 325" in body


def test_raw_lowering_uses_tilelang_metal_scalar_pipeline() -> None:
    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._bwd_simd_reduce_kernel_for_state_snapshots(
        1, 4, 1, 64, 4
    )
    assert lowering.grid == (1, 1, 1)
    assert lowering.threadgroup == (64, 1, 1)
    assert "h_snap" in lowering.body
    assert re.search(r"\bdecay = exp\(", lowering.body)
    assert "1.000000e+00 / decay" not in lowering.body
    assert re.search(r"dh\[[^\]]+\] = [^;\n]*exp\(", lowering.body) is None
    assert re.search(r"d_logdecay = [^;\n]*exp\(", lowering.body) is None
    assert "T.metal_thread_position_in_grid_x() % 64" not in lowering.body


def test_path_c_launch_geometry_comes_from_tilelang_lowering() -> None:
    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._fwd_kernel_for(
        1, 4, 56, 64, 4, return_msl=True
    )
    assert lowering is not None
    assert lowering.grid == (14, 1, 1)
    assert lowering.threadgroup == (256, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(lowering) == (3584, 1, 1)


def test_mamba3_path_c_schedule_plan_uses_rule_and_z3_for_spec_shape() -> None:
    pytest.importorskip("z3")
    mamba3_path_c_schedule_plan.cache_clear()

    plan = mamba3_path_c_schedule_plan(
        batch=2,
        seq=512,
        heads=4,
        headdim=32,
        state=64,
        dtype="float32",
        z3_policy="enabled",
    )

    assert isinstance(plan, Mamba3PathCSchedulePlan)
    assert plan.threads == 256
    assert plan.grid_blocks == 1
    assert plan.fwd_path_c_candidate is True
    assert plan.bwd_path_c_candidate is True
    assert plan.mode == "path_c_fwd_bwd"
    assert plan.z3_used is True
    assert plan.z3_proved is True

    bf16_plan = mamba3_path_c_schedule_plan(
        batch=2,
        seq=512,
        heads=4,
        headdim=32,
        state=64,
        dtype="bfloat16",
        z3_policy="enabled",
    )
    assert bf16_plan.fwd_path_c_candidate is True
    assert bf16_plan.bwd_path_c_candidate is True
    assert bf16_plan.mode == "path_c_fwd_bwd"

    wide_p_plan = mamba3_path_c_schedule_plan(
        batch=1,
        seq=512,
        heads=1,
        headdim=128,
        state=64,
        dtype="float32",
        z3_policy="enabled",
    )
    assert wide_p_plan.fwd_path_c_candidate is True
    assert wide_p_plan.bwd_path_c_candidate is True
    assert wide_p_plan.mode == "path_c_fwd_bwd"


def test_mamba3_path_c_schedule_plan_accepts_explicit_z3_policy() -> None:
    mamba3_path_c_schedule_plan.cache_clear()

    plan = mamba3_path_c_schedule_plan(
        batch=2,
        seq=512,
        heads=4,
        headdim=32,
        state=64,
        dtype="float32",
        z3_policy="disabled",
    )

    assert plan.z3_used is False
    assert plan.z3_proved is False
    assert plan.fwd_path_c_candidate is False
    assert plan.bwd_path_c_candidate is False
    assert "z3 disabled by policy" in plan.reason


def test_mamba3_path_c_receipt_gate_requires_matching_shape_and_fwd_win(
    tmp_path,
) -> None:
    pytest.importorskip("z3")
    mamba3_path_c_schedule_plan.cache_clear()

    receipt = {
        "kernel": "mamba3_mimo_path_c_vs_path_b",
        "shape": {
            "batch": 2,
            "seq": 512,
            "heads": 4,
            "headdim": 32,
            "state": 64,
            "dtype": "float32",
        },
        "strict_policy": {
            "phase": "fwd",
            "requires_path_b_and_path_c": True,
            "path_c_fwd_over_path_b_max_ratio": 1.0,
        },
        "scheduler_decision": {
            "mode": "path_c_fwd_path_b_bwd",
            "selected_forward_kernel": "path_c_tilelang_dsl",
            "selected_backward_kernel": "metal_kernel_bwd_v1",
        },
        "timings": {
            "fwd_path_b": {"median_ms": 2.0},
            "fwd_path_c": {"median_ms": 1.5},
        },
    }
    receipt_path = tmp_path / "mamba3_path_c.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert mamba3_path_c_receipt_allows_auto_promotion(
        receipt_path,
        batch=2,
        seq=512,
        heads=4,
        headdim=32,
        state=64,
        dtype="float32",
        z3_policy="enabled",
    )
    assert (
        mamba3_path_c_receipt_auto_mode(
            receipt_path,
            batch=2,
            seq=512,
            heads=4,
            headdim=32,
            state=64,
            dtype="float32",
            z3_policy="enabled",
        )
        == "path_c_fwd_path_b_bwd"
    )

    full_path_c = json.loads(json.dumps(receipt))
    full_path_c["scheduler_decision"] = {
        "mode": "path_c_fwd_bwd",
        "selected_forward_kernel": "path_c_tilelang_dsl",
        "selected_backward_kernel": "path_c_tilelang_dsl",
        "ratios": {
            "bwd_path_c_over_path_b": 0.9,
            "fwd_bwd_path_c_over_path_b": 0.95,
        },
    }
    full_path_c["strict_policy"]["path_c_bwd_over_path_b_max_ratio"] = 1.0
    full_path_c["strict_policy"]["path_c_fwd_bwd_over_path_b_max_ratio"] = 1.0
    receipt_path.write_text(json.dumps(full_path_c), encoding="utf-8")
    assert (
        mamba3_path_c_receipt_auto_mode(
            receipt_path,
            batch=2,
            seq=512,
            heads=4,
            headdim=32,
            state=64,
            dtype="float32",
            z3_policy="enabled",
        )
        == "path_c_fwd_bwd"
    )

    wrong_shape = json.loads(json.dumps(receipt))
    wrong_shape["shape"]["seq"] = 256
    receipt_path.write_text(json.dumps(wrong_shape), encoding="utf-8")
    assert not mamba3_path_c_receipt_allows_auto_promotion(
        receipt_path,
        batch=2,
        seq=512,
        heads=4,
        headdim=32,
        state=64,
        dtype="float32",
        z3_policy="enabled",
    )

    slow_path_c = json.loads(json.dumps(receipt))
    slow_path_c["timings"]["fwd_path_c"]["median_ms"] = 2.1
    receipt_path.write_text(json.dumps(slow_path_c), encoding="utf-8")
    assert not mamba3_path_c_receipt_allows_auto_promotion(
        receipt_path,
        batch=2,
        seq=512,
        heads=4,
        headdim=32,
        state=64,
        dtype="float32",
        z3_policy="enabled",
    )


def test_fwd_path_c_dispatch_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Path C dispatch failure must not hide behind the reference path."""

    _require_mamba3_path_c()

    class FailingKernel:
        def __call__(self, *_args: object, **_kwargs: object) -> list[mx.array]:
            raise _msl_transform.MSLDispatchUnsupported("forced dispatch failure")

    monkeypatch.setattr(
        mamba3_path_c,
        "mamba3_mimo_path_c_status",
        lambda: Mamba3PathCStatus(True, "available"),
    )
    monkeypatch.setattr(
        mamba3_path_c,
        "_fwd_kernel_for",
        lambda *_args, **_kwargs: (FailingKernel(), object()),
    )
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    with pytest.raises(RuntimeError, match="dispatch failed"):
        mamba3_mimo_fwd_path_c(*inputs)


def test_path_c_forward_backward_use_tvm_ffi_not_mx_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_mamba3_path_c()
    mamba3_path_c._clear_mamba3_path_c_caches()

    def fail_mx_fast_wrapper(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Mamba3 Path C production path must not build mx.fast wrapper")

    monkeypatch.setattr(mx.fast, "metal_kernel", fail_mx_fast_wrapper)

    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    y, h_last = mamba3_mimo_fwd_path_c(*inputs)
    dy = mx.ones(y.shape, dtype=mx.float32)
    grads = mamba3_mimo_bwd_path_c(dy, *inputs)
    mx.eval(y, h_last, *grads)
    assert y.shape == (1, 6, 2, 4)
    assert h_last.shape == (1, 2, 4, 4)
    assert grads[0].shape == inputs[0].shape


def test_fwd_path_c_owner_outputs_avoid_hidden_zero_alloc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_out = mx.zeros((1, 6, 2, 4), dtype=mx.float32)
    h_out = mx.zeros((1, 2, 4, 4), dtype=mx.float32)
    mx.eval(y_out, h_out)

    def fail_zero_alloc(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("owner-output fwd route must not allocate mx.zeros")

    monkeypatch.setattr(mamba3_path_c.mx, "zeros", fail_zero_alloc)

    y, h_last = mamba3_mimo_fwd_path_c(*inputs, out=(y_out, h_out))
    mx.eval(y, h_last)
    assert y is y_out
    assert h_last is h_out


def test_fwd_path_c_default_outputs_use_tilelang_write_only_alloc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)

    def fail_zero_alloc(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("default fwd route must not allocate mx.zeros")

    monkeypatch.setattr(mamba3_path_c.mx, "zeros", fail_zero_alloc)

    y, h_last = mamba3_mimo_fwd_path_c(*inputs)
    mx.eval(y, h_last)
    assert y.shape == (1, 6, 2, 4)
    assert h_last.shape == (1, 2, 4, 4)


def test_bwd_path_c_rejects_public_partial_owner_outputs() -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=5, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    dy = mx.ones(x.shape, dtype=mx.float32)
    owner_outputs = (
        mx.zeros(x.shape, dtype=mx.float32),
        mx.zeros(z.shape, dtype=mx.float32),
        mx.zeros((1, 6, 2, 5, 4), dtype=mx.float32),
        mx.zeros((1, 6, 2, 5, 4), dtype=mx.float32),
        mx.zeros((1, 6, 2, 4), dtype=mx.float32),
        mx.zeros((1, 6, 2, 4), dtype=mx.float32),
        mx.zeros((1, 2, 4), dtype=mx.float32),
        mx.zeros(h0.shape, dtype=mx.float32),
    )
    mx.eval(dy, *owner_outputs)

    with pytest.raises(RuntimeError, match="does not expose public owner-output"):
        mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0, out=owner_outputs)


def test_bwd_path_c_unaligned_p_shape_uses_generic_partial_route() -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=2, heads=1, headdim=33, state=2, dtype=mx.float32)
    dy = mx.ones(inputs[0].shape, dtype=mx.float32)
    mx.eval(dy)

    grads = mamba3_mimo_bwd_path_c(dy, *inputs)
    mx.eval(*grads)
    assert grads[0].shape == inputs[0].shape
    assert grads[1].shape == inputs[1].shape


def test_bwd_path_c_default_outputs_use_tilelang_write_only_alloc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    dy = mx.ones(x.shape, dtype=mx.float32)
    mx.eval(dy)

    def fail_zero_alloc(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("default bwd route must not allocate mx.zeros")

    monkeypatch.setattr(mamba3_path_c.mx, "zeros", fail_zero_alloc)

    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*grads)
    assert grads[0].shape == x.shape
    assert grads[7].shape == h0.shape


def test_fwd_path_c_unavailable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mamba3_path_c,
        "mamba3_mimo_path_c_status",
        lambda: Mamba3PathCStatus(False, "forced unavailable"),
    )
    inputs = _make_inputs(batch=1, seq=1, heads=1, headdim=2, state=2, dtype=mx.float32)
    with pytest.raises(RuntimeError, match="forced unavailable"):
        mamba3_mimo_fwd_path_c(*inputs)


# ---------------------------------------------------------------------------
# Forward parity tests
# ---------------------------------------------------------------------------


def test_fwd_path_c_matches_path_b_fp32_small_shape() -> None:
    """Path C fwd matches Path B Metal fwd within FP32 tolerance."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    y_pb, h_pb = mamba3_mimo_fwd_metal(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_pb), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(_np(h_pc), _np(h_pb), rtol=1e-3, atol=1e-4)


def test_fwd_path_c_matches_path_b_bf16_small_shape() -> None:
    """Path C fwd consumes bf16 owner buffers directly with FP32 accumulators."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.bfloat16)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    y_pb, h_pb = mamba3_mimo_fwd_metal(*inputs)
    mx.eval(y_pc, h_pc, y_pb, h_pb)
    assert y_pc.dtype == mx.bfloat16
    assert h_pc.dtype == mx.bfloat16
    np.testing.assert_allclose(_np(y_pc), _np(y_pb), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(_np(h_pc), _np(h_pb), rtol=1e-3, atol=1e-4)


def test_fwd_path_c_matches_reference_fp32_small_shape() -> None:
    """Path C fwd also matches the pure-MLX reference."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    y_ref, h_ref = mamba3_mimo_reference(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_ref), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_np(h_pc), _np(h_ref), rtol=1e-4, atol=1e-5)


def test_fwd_path_c_stable_silu_handles_large_negative_gate() -> None:
    """Path C follows Path B SiLU and remains finite for saturated gates."""

    _require_mamba3_path_c()
    inputs = list(_make_inputs(batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float32))
    inputs[3] = mx.full(inputs[3].shape, -200.0, dtype=mx.float32)
    mx.eval(*inputs)

    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    y_ref, h_ref = mamba3_mimo_reference(*inputs)
    y_pc_np = _np(y_pc)
    h_pc_np = _np(h_pc)
    assert np.isfinite(y_pc_np).all()
    assert np.isfinite(h_pc_np).all()
    np.testing.assert_allclose(y_pc_np, _np(y_ref), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(h_pc_np, _np(h_ref), rtol=1e-4, atol=1e-5)


def test_fwd_path_c_matches_path_b_at_bench_shape_fp32() -> None:
    """At the spec bench shape (B=2,T=512,H=4,P=32,N=64) Path C matches Path B."""

    _require_mamba3_path_c()
    inputs = _make_inputs(
        batch=2, seq=512, heads=4, headdim=32, state=64, dtype=mx.float32
    )
    y_pc, _ = mamba3_mimo_fwd_path_c(*inputs)
    y_pb, _ = mamba3_mimo_fwd_metal(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_pb), rtol=1e-3, atol=1e-4)


def test_fwd_snapshot_route_matches_fwd_and_snapshot_kernel_fp32() -> None:
    """Training fwd snapshots match the standalone snapshot kernel exactly enough."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs

    y_snap, h_last_snap, h_snap = mamba3_path_c._mamba3_mimo_fwd_path_c_with_snapshots(
        x, B, C, z, A, dt, D, h0
    )
    y_ref, h_last_ref = mamba3_mimo_fwd_path_c(x, B, C, z, A, dt, D, h0)
    snapshot_kernel, _lowering = mamba3_path_c._bwd_state_snapshots_kernel_for(
        1, 5, 2, 4, 8
    )
    snapshot_out = snapshot_kernel(x, B, A, dt, h0)
    h_snap_ref = snapshot_out[0] if isinstance(snapshot_out, (list, tuple)) else snapshot_out
    mx.eval(y_snap, h_last_snap, h_snap, y_ref, h_last_ref, h_snap_ref)

    np.testing.assert_allclose(_np(y_snap), _np(y_ref), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(_np(h_last_snap), _np(h_last_ref), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_np(h_snap), _np(h_snap_ref), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_np(h_snap[:, 0]), _np(h0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_np(h_snap[:, -1]), _np(h_last_snap), rtol=1e-4, atol=1e-5)


def test_fwd_path_c_rejects_fp16_without_hidden_casts() -> None:
    """Path C must not materialize large hidden cast buffers for unsupported inputs."""

    _require_mamba3_path_c()
    inputs = _make_inputs(
        batch=1, seq=64, heads=2, headdim=8, state=16, dtype=mx.float16
    )
    with pytest.raises(RuntimeError, match="without hidden casts"):
        mamba3_mimo_fwd_path_c(*inputs)


def test_bwd_path_c_rejects_fp16_without_hidden_casts() -> None:
    """Backward follows the same FP32/BF16 no-hidden-cast ABI as forward."""

    _require_mamba3_path_c()
    inputs = _make_inputs(
        batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float16
    )
    dy = mx.zeros_like(inputs[0])
    with pytest.raises(RuntimeError, match="without hidden casts"):
        mamba3_mimo_bwd_path_c(dy, *inputs)


# ---------------------------------------------------------------------------
# Backward parity tests
# ---------------------------------------------------------------------------


def test_bwd_path_c_matches_path_b_fp32_small_shape() -> None:
    """Path C bwd final-gradient route matches Path B on a small shape."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(123)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    g_pc = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb):
        np.testing.assert_allclose(_np(gpc), _np(gpb), rtol=1e-3, atol=1e-4,
                                    err_msg=f"grad mismatch on {name}")


def test_bwd_scratch_route_matches_path_b_fp32_small_shape() -> None:
    """Single-kernel scratch bwd route matches Path B before it is promoted."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(321)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_scratch_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"scratch bwd grad mismatch on {name}",
        )


def test_bwd_scratch_route_matches_path_b_bf16_small_shape() -> None:
    """BF16 scratch route keeps final grads in BF16 and matches Path B tolerance."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(327)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_scratch_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert gpc.dtype == gpb.dtype == mx.bfloat16
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=8e-3,
            atol=2e-3,
            err_msg=f"bf16 scratch bwd grad mismatch on {name}",
        )


def test_bwd_scratch_mlx_reduce_route_matches_path_b_bf16_small_shape() -> None:
    """BF16 scratch+MLX-reduce route matches Path B before default promotion."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(330)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_scratch_mlx_reduce_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert gpc.dtype == gpb.dtype == mx.bfloat16
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=8e-3,
            atol=2e-3,
            err_msg=f"bf16 scratch+mlx-reduce bwd grad mismatch on {name}",
        )


def test_bwd_scratch_route_uses_tilelang_post_reduce(monkeypatch) -> None:
    """Scratch route must use the same fast TileLang reduction tail."""

    _require_mamba3_path_c()
    calls: list[tuple[str, ...] | None] = []
    original_reduce = mamba3_path_c._reduce_bwd_partials_path_c_threaded_kernel

    def wrapped_reduce(*args, **kwargs):
        output_dtypes = kwargs.get("output_dtypes")
        calls.append(output_dtypes)
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(
        mamba3_path_c,
        "_reduce_bwd_partials_path_c_threaded_kernel",
        wrapped_reduce,
    )

    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(326)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    grads = mamba3_path_c._mamba3_mimo_bwd_path_c_scratch_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    mx.eval(*grads)

    assert calls == [None]


def test_bwd_scratch_route_uses_compact_partials_for_bf16(monkeypatch) -> None:
    """BF16 scratch diagnostic should compare against production dtype policy."""

    _require_mamba3_path_c()
    seen_partial_dtypes: list[tuple[str, str, str, str, str]] = []
    original_kernel_for = mamba3_path_c._bwd_scratch_partial_kernel_for

    def wrapped_kernel_for(*args, **kwargs):
        del kwargs
        seen_partial_dtypes.append(tuple(args[16:21]))
        return original_kernel_for(*args)

    monkeypatch.setattr(
        mamba3_path_c,
        "_bwd_scratch_partial_kernel_for",
        wrapped_kernel_for,
    )

    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(329)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    grads = mamba3_path_c._mamba3_mimo_bwd_path_c_scratch_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    mx.eval(*grads)

    assert seen_partial_dtypes == [
        ("bfloat16", "bfloat16", "float32", "float32", "float32")
    ]


def test_bwd_scratch_bf16_uses_serial_tilelang_post_reduce(monkeypatch) -> None:
    """BF16 compact partials should not take the slower threaded reducer."""

    _require_mamba3_path_c()
    serial_calls: list[tuple[str, ...] | None] = []
    threaded_calls = 0
    original_serial = mamba3_path_c._reduce_bwd_partials_path_c_kernel

    def wrapped_serial(*args, **kwargs):
        output_dtypes = kwargs.get("output_dtypes")
        serial_calls.append(output_dtypes)
        return original_serial(*args, **kwargs)

    def fail_threaded(*_args, **_kwargs):
        nonlocal threaded_calls
        threaded_calls += 1
        raise AssertionError("BF16 compact partials should use the serial reducer")

    monkeypatch.setattr(
        mamba3_path_c,
        "_reduce_bwd_partials_path_c_kernel",
        wrapped_serial,
    )
    monkeypatch.setattr(
        mamba3_path_c,
        "_reduce_bwd_partials_path_c_threaded_kernel",
        fail_threaded,
    )

    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(330)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    grads = mamba3_path_c._mamba3_mimo_bwd_path_c_scratch_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    mx.eval(*grads)

    assert serial_calls == [
        ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16")
    ]
    assert threaded_calls == 0


def test_bwd_from_training_fwd_snapshots_matches_path_b_fp32(monkeypatch) -> None:
    """Bwd can consume fwd-produced snapshots without launching snapshot kernel."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(327)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    _y, _h_last, h_snap = mamba3_path_c._mamba3_mimo_fwd_path_c_with_snapshots(
        x, B, C, z, A, dt, D, h0
    )
    mx.eval(h_snap)

    def fail_snapshot_builder(*_args, **_kwargs):
        raise AssertionError("snapshot kernel must not be built for snapshot-reuse bwd")

    monkeypatch.setattr(
        mamba3_path_c,
        "_bwd_state_snapshots_kernel_for",
        fail_snapshot_builder,
    )

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_from_snapshots_kernel(
        dy, x, B, C, z, A, dt, D, h0, h_snap
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"snapshot-reuse bwd grad mismatch on {name}",
        )


def test_bwd_tilelang_post_reduce_route_matches_path_b_fp32_small_shape() -> None:
    """TileLang post-reduce bwd diagnostic route matches Path B before promotion."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(322)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_partial_tl_reduce_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"tilelang post-reduce bwd grad mismatch on {name}",
        )


def test_bwd_threaded_post_reduce_route_matches_path_b_fp32_small_shape() -> None:
    """Threaded TileLang post-reduce route matches Path B before promotion."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(328)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_partial_threaded_reduce_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"threaded post-reduce bwd grad mismatch on {name}",
        )


def test_bwd_tilelang_post_reduce_route_matches_path_b_bf16_small_shape() -> None:
    """BF16 TileLang post-reduce route can write final reducer outputs as BF16."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(323)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_partial_tl_reduce_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert gpc.dtype == gpb.dtype == mx.bfloat16
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=2e-2,
            atol=2e-3,
            err_msg=f"bf16 tilelang post-reduce bwd grad mismatch on {name}",
        )


def test_bwd_path_c_bf16_production_uses_scratch_tilelang_reduce(monkeypatch) -> None:
    """BF16 production bwd uses the promoted scratch producer plus TileLang reducer."""

    _require_mamba3_path_c()
    calls = 0
    original_reduce = mamba3_path_c._mamba3_mimo_bwd_path_c_scratch_kernel

    def wrapped_reduce(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(
        mamba3_path_c,
        "_mamba3_mimo_bwd_path_c_scratch_kernel",
        wrapped_reduce,
    )

    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(324)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*grads)

    assert calls == 1


def test_bwd_path_c_fp32_production_uses_tilelang_post_reduce(monkeypatch) -> None:
    """FP32 production bwd uses the profiled threaded TileLang post-reduce policy."""

    _require_mamba3_path_c()
    calls: list[tuple[str, ...] | None] = []
    original_reduce = mamba3_path_c._reduce_bwd_partials_path_c_threaded_kernel

    def wrapped_reduce(*args, **kwargs):
        output_dtypes = kwargs.get("output_dtypes")
        calls.append(output_dtypes)
        return original_reduce(*args, **kwargs)

    monkeypatch.setattr(
        mamba3_path_c,
        "_reduce_bwd_partials_path_c_threaded_kernel",
        wrapped_reduce,
    )

    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(325)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.float32)
    mx.eval(dy)

    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*grads)

    assert calls == [None]


def test_bwd_path_c_matches_path_b_bf16_small_shape() -> None:
    """Path C bwd consumes/returns bf16 owner buffers without hidden casts."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(123)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    g_pc = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb):
        assert gpc.dtype == gpb.dtype == mx.bfloat16
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-2,
            atol=1e-5,
            err_msg=f"bf16 grad mismatch on {name}",
        )


def test_bwd_bf16_snapshot_route_matches_path_b_bf16_small_shape() -> None:
    """Diagnostic BF16 snapshot route must stay within the BF16 bwd contract."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.bfloat16)
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(124)
    dy = (mx.random.normal(x.shape) * 0.1).astype(mx.bfloat16)
    mx.eval(dy)

    g_pc = mamba3_path_c._mamba3_mimo_bwd_path_c_bf16_snapshot_kernel(
        dy, x, B, C, z, A, dt, D, h0
    )
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert gpc.dtype == gpb.dtype == mx.bfloat16
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-2,
            atol=1e-5,
            err_msg=f"bf16 snapshot route grad mismatch on {name}",
        )


def test_bwd_path_c_headdim64_bf16_uses_skip_D_tensor() -> None:
    """Regression for full-tensor bf16 bwd lowering at the model headdim."""

    _require_mamba3_path_c()
    batch, seq, heads, headdim, state = 1, 8, 2, 64, 64
    x = mx.full((batch, seq, heads, headdim), 0.125, dtype=mx.bfloat16)
    B = mx.zeros((batch, seq, heads, state), dtype=mx.bfloat16)
    C = mx.zeros((batch, seq, heads, state), dtype=mx.bfloat16)
    z = mx.full((batch, seq, heads, headdim), 0.25, dtype=mx.bfloat16)
    A = mx.full((batch, seq, heads), -0.1, dtype=mx.bfloat16)
    dt = mx.full((batch, seq, heads), 0.01, dtype=mx.bfloat16)
    D = mx.array([1.0, 2.0], dtype=mx.bfloat16)
    h0 = mx.zeros((batch, heads, headdim, state), dtype=mx.bfloat16)
    dy = mx.ones_like(x)

    g_pc = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert bool(mx.all(mx.isfinite(gpc.astype(mx.float32))).item())
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-2,
            atol=1e-5,
            err_msg=f"headdim64 bf16 D-path grad mismatch on {name}",
        )


def test_bwd_path_c_snapshot_route_keeps_long_headdim64_bf16_finite() -> None:
    """Long bf16 reverse recurrence uses tensor snapshots instead of T-wide inversion."""

    _require_mamba3_path_c()
    seq = mamba3_path_c._BWD_SNAPSHOT_BLOCK + 32
    inputs = _make_inputs(
        batch=1,
        seq=seq,
        heads=1,
        headdim=64,
        state=64,
        dtype=mx.bfloat16,
        seed=29,
    )
    x, B, C, z, A, dt, D, h0 = inputs
    dy = mx.ones_like(x)

    g_pc = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert bool(mx.all(mx.isfinite(gpc.astype(mx.float32))).item())
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-1,
            atol=5e-3,
            err_msg=f"snapshot bwd grad mismatch on {name}",
        )


def test_bwd_path_c_snapshot_route_survives_decay_underflow() -> None:
    """Snapshot bwd must not reconstruct h_prev via 1 / decay when decay underflows."""

    _require_mamba3_path_c()
    batch, heads, headdim, state = 1, 1, 4, 4
    seq = mamba3_path_c._BWD_SNAPSHOT_BLOCK + 3
    x = mx.full((batch, seq, heads, headdim), 0.125, dtype=mx.float32)
    B = mx.full((batch, seq, heads, state), 0.5, dtype=mx.float32)
    C = mx.full((batch, seq, heads, state), 0.25, dtype=mx.float32)
    z = mx.full((batch, seq, heads, headdim), 0.1, dtype=mx.float32)
    A = mx.full((batch, seq, heads), -200.0, dtype=mx.float32)
    dt = mx.full((batch, seq, heads), 5.0, dtype=mx.float32)
    D = mx.ones((heads,), dtype=mx.float32)
    h0 = mx.zeros((batch, heads, headdim, state), dtype=mx.float32)
    dy = mx.ones_like(x)

    g_pc = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert bool(mx.all(mx.isfinite(gpc.astype(mx.float32))).item())
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"underflow snapshot bwd grad mismatch on {name}",
        )


def test_bwd_path_c_long_model_bf16_uses_stable_snapshot_simd() -> None:
    """Long model-shaped bf16 bwd must not use inverse-state SIMD recurrence."""

    _require_mamba3_path_c()
    inputs = _make_mamba3_projection_inputs(
        d_model=1024,
        seq=2048,
        dtype=mx.bfloat16,
    )
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(123)
    dy = (mx.random.normal(x.shape) * 0.01).astype(mx.bfloat16)
    mx.eval(dy, *inputs)

    g_pc = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    g_pb = mamba3_mimo_bwd_metal(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*g_pc, *g_pb)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb, strict=True):
        assert bool(mx.all(mx.isfinite(gpc.astype(mx.float32))).item())
        np.testing.assert_allclose(
            _np(gpc),
            _np(gpb),
            rtol=1e-1,
            atol=5e-3,
            err_msg=f"long model bf16 grad mismatch on {name}",
        )


@pytest.mark.kernel
def test_bwd_path_c_full_1b_model_bf16_snapshot_simd_stays_finite() -> None:
    """1B Mamba3 shape guards the HEADDIM=64 SIMD allreduce regression."""

    _require_mamba3_path_c()
    inputs = _make_mamba3_projection_inputs(
        d_model=3584,
        seq=2048,
        dtype=mx.bfloat16,
    )
    x, B, C, z, A, dt, D, h0 = inputs
    mx.random.seed(123)
    dy = (mx.random.normal(x.shape) * 0.01).astype(mx.bfloat16)
    mx.eval(dy, *inputs)

    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*grads)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, grad in zip(names, grads, strict=True):
        assert bool(mx.all(mx.isfinite(grad.astype(mx.float32))).item()), name


@pytest.mark.kernel
@pytest.mark.parametrize("dy_scale", [1.0, 1e-4])
def test_bwd_path_c_full_1b_model_bf16_snapshot_simd_extreme_dy_stays_finite(
    dy_scale: float,
) -> None:
    """Production projection tensors stay finite for direct extreme upstream grads."""

    _require_mamba3_path_c()
    inputs = _make_mamba3_projection_inputs(
        d_model=3584,
        seq=2048,
        dtype=mx.bfloat16,
    )
    x, B, C, z, A, dt, D, h0 = inputs
    dy = (mx.ones_like(x) * mx.array(dy_scale, dtype=x.dtype)).astype(x.dtype)
    mx.eval(dy, *inputs)

    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    mx.eval(*grads)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, grad in zip(names, grads, strict=True):
        assert bool(mx.all(mx.isfinite(grad.astype(mx.float32))).item()), name


@pytest.mark.kernel
def test_apply_path_c_full_1b_model_bf16_graph_vjp_stays_finite() -> None:
    """1B graph VJP must route long-sequence bwd through state snapshots."""

    _require_mamba3_path_c()
    inputs = _make_mamba3_projection_inputs(
        d_model=3584,
        seq=2048,
        dtype=mx.bfloat16,
    )

    def pc_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y = cast(mx.array, mamba3_mimo_apply_path_c(x, B, C, z, A, dt, D, h0))
        y_f32 = y.astype(mx.float32)
        return mx.sum(y_f32 * y_f32) * 0.5

    grads = mx.grad(pc_loss, argnums=tuple(range(8)))(*inputs)
    mx.eval(*grads)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, grad in zip(names, grads, strict=True):
        assert bool(mx.all(mx.isfinite(grad.astype(mx.float32))).item()), name


def test_apply_path_c_vjp_matches_reference_inside_mlx_graph_transform() -> None:
    """TileLang Path C is callable from ``mx.grad`` and uses its custom VJP."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)

    def ref_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y, _ = mamba3_mimo_reference(x, B, C, z, A, dt, D, h0)
        return mx.sum(y * y) * 0.5

    def pc_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y = cast(mx.array, mamba3_mimo_apply_path_c(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    g_ref = mx.grad(ref_loss, argnums=tuple(range(8)))(*inputs)
    g_pc = mx.grad(pc_loss, argnums=tuple(range(8)))(*inputs)
    mx.eval(*g_ref, *g_pc)
    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, got, expected in zip(names, g_pc, g_ref, strict=True):
        np.testing.assert_allclose(
            _np(got),
            _np(expected),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"graph VJP mismatch on {name}",
        )


def test_apply_training_path_c_vjp_reuses_forward_snapshots(monkeypatch) -> None:
    """Training surface VJP must not launch the old standalone snapshot pass."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=8, dtype=mx.float32)

    def fail_snapshot_builder(*_args, **_kwargs):
        raise AssertionError("training VJP must reuse forward snapshots")

    monkeypatch.setattr(
        mamba3_path_c,
        "_bwd_state_snapshots_kernel_for",
        fail_snapshot_builder,
    )

    y_training = mamba3_mimo_apply_training_path_c(*inputs)
    y_state_training, h_state_training = mamba3_mimo_apply_with_state_training_path_c(
        *inputs
    )
    y_fwd, h_fwd = mamba3_mimo_fwd_path_c(*inputs)
    mx.eval(y_training, y_state_training, h_state_training, y_fwd, h_fwd)
    np.testing.assert_allclose(_np(y_training), _np(y_fwd), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(y_state_training), _np(y_fwd), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_state_training), _np(h_fwd), rtol=1e-6, atol=1e-6)

    def training_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y, _h = mamba3_mimo_apply_with_state_training_path_c(
            x,
            B,
            C,
            z,
            A,
            dt,
            D,
            h0,
        )
        return mx.sum(y * y) * 0.5

    g_training = mx.grad(training_loss, argnums=tuple(range(8)))(*inputs)
    g_path_b = mamba3_mimo_bwd_metal(y_training, *inputs)
    mx.eval(*g_training, *g_path_b)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, got, expected in zip(names, g_training, g_path_b, strict=True):
        np.testing.assert_allclose(
            _np(got),
            _np(expected),
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"training snapshot-reuse VJP mismatch on {name}",
        )


def test_apply_training_path_c_bf16_vjp_matches_path_b() -> None:
    """BF16 training surface keeps the faster production Path C VJP route."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=5, heads=2, headdim=4, state=8, dtype=mx.bfloat16)

    y_training, _h_training = mamba3_mimo_apply_with_state_training_path_c(*inputs)
    mx.eval(y_training)

    def training_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y, _h = mamba3_mimo_apply_with_state_training_path_c(
            x,
            B,
            C,
            z,
            A,
            dt,
            D,
            h0,
        )
        y_f32 = y.astype(mx.float32)
        return mx.sum(y_f32 * y_f32) * 0.5

    g_training = mx.grad(training_loss, argnums=tuple(range(8)))(*inputs)
    g_path_b = mamba3_mimo_bwd_metal(y_training, *inputs)
    mx.eval(*g_training, *g_path_b)

    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, got, expected in zip(names, g_training, g_path_b, strict=True):
        assert got.dtype == expected.dtype == mx.bfloat16
        np.testing.assert_allclose(
            _np(got),
            _np(expected),
            rtol=2e-2,
            atol=2e-3,
            err_msg=f"bf16 training snapshot-reuse VJP mismatch on {name}",
        )


def test_apply_with_state_path_c_matches_forward_and_vjp_works() -> None:
    """Tuple Path C surface matches fwd and its VJP uses the TileLang bwd."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_state, h_state = mamba3_mimo_apply_with_state_path_c(*inputs)
    y_fwd, h_fwd = mamba3_mimo_fwd_path_c(*inputs)
    mx.eval(y_state, h_state, y_fwd, h_fwd)
    np.testing.assert_allclose(_np(y_state), _np(y_fwd), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_state), _np(h_fwd), rtol=1e-6, atol=1e-6)

    def state_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y, _ = mamba3_mimo_apply_with_state_path_c(x, B, C, z, A, dt, D, h0)
        return mx.sum(y * y) * 0.5

    def y_only_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y = cast(mx.array, mamba3_mimo_apply_path_c(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    g_state = mx.grad(state_loss, argnums=tuple(range(8)))(*inputs)
    g_y_only = mx.grad(y_only_loss, argnums=tuple(range(8)))(*inputs)
    mx.eval(*g_state, *g_y_only)
    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, got, expected in zip(names, g_state, g_y_only, strict=True):
        np.testing.assert_allclose(
            _np(got),
            _np(expected),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"state/y-only VJP mismatch on {name}",
        )


def test_apply_with_state_path_c_fwd_path_b_bwd_vjp_matches_path_b() -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)

    y_hybrid, h_hybrid = mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd(*inputs)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    mx.eval(y_hybrid, h_hybrid, y_pc, h_pc)
    np.testing.assert_allclose(_np(y_hybrid), _np(y_pc), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_np(h_hybrid), _np(h_pc), rtol=1e-6, atol=1e-6)

    def hybrid_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y, _ = mamba3_mimo_apply_with_state_path_c_fwd_path_b_bwd(
            x, B, C, z, A, dt, D, h0
        )
        return mx.sum(y * y) * 0.5

    g_hybrid = mx.grad(hybrid_loss, argnums=tuple(range(8)))(*inputs)
    g_path_b = mamba3_mimo_bwd_metal(y_hybrid, *inputs)
    mx.eval(*g_hybrid, *g_path_b)
    names = ["dx", "dB", "dC", "dz", "dA", "ddt", "dD", "dh0"]
    for name, got, expected in zip(names, g_hybrid, g_path_b, strict=True):
        np.testing.assert_allclose(
            _np(got),
            _np(expected),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"hybrid VJP mismatch on {name}",
        )


def test_apply_path_c_vjp_runs_at_bench_shape() -> None:
    """Bench-shape VJP runs through the TileLang graph/autograd boundary."""

    _require_mamba3_path_c()
    inputs = _make_inputs(
        batch=2, seq=512, heads=4, headdim=32, state=64, dtype=mx.float32
    )

    def pc_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y = cast(mx.array, mamba3_mimo_apply_path_c(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    grads = mx.grad(pc_loss, argnums=tuple(range(8)))(*inputs)
    mx.eval(*grads)
    assert grads[0].shape == inputs[0].shape
    assert grads[7].shape == inputs[7].shape


def test_bwd_path_c_returns_correct_zero_seq_shapes() -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=0, heads=1, headdim=2, state=2, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    dy = mx.zeros_like(x)
    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    assert grads[0].shape == x.shape
    assert grads[7].shape == h0.shape
