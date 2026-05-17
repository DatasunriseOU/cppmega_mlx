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
    mamba3_mimo_apply_with_state_path_c,
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


def test_lowered_fwd_msl_contains_kernel_void() -> None:
    """Lowering emits a self-contained MSL kernel string."""

    msl = dump_lowered_fwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert "kernel void" in msl
    # The lowered MSL references each of the alphabetically-ordered buffers.
    for name in ("A", "B", "C", "D", "dt", "h0", "h_last", "x", "y", "z"):
        assert name in msl, f"buffer {name!r} missing from lowered MSL"


def test_lowered_bwd_msl_contains_kernel_void() -> None:
    msl = dump_lowered_bwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert "kernel void" in msl
    # Long-sequence bwd emits final-gradient outputs and consumes h snapshots.
    for name in ("A", "B", "C", "D", "dt", "dy", "h_snap", "x", "z"):
        assert name in msl, f"input buffer {name!r} missing from lowered MSL"
    assert "h_steps" not in msl
    _assert_no_bwd_partial_outputs(msl)
    for name in ("dA", "dB", "dC", "dD_batch", "ddt", "dh0", "dx", "dz"):
        assert name in msl, f"output buffer {name!r} missing from lowered MSL"


def test_lowered_bwd_bench_shape_uses_simd_p_reduction() -> None:
    msl = dump_lowered_bwd_msl(batch=1, seq=4, heads=1, headdim=32, state=4)
    assert "kernel void" in msl
    assert "simd_sum(" in msl
    assert "simd_shuffle_down" not in msl
    _assert_no_bwd_partial_outputs(msl)
    assert "dD_batch" in msl


@pytest.mark.parametrize("headdim", [64, 96, 128, 256, 512, 1024])
def test_lowered_bwd_aligned_headdims_use_split_thread_allreduce(
    headdim: int,
) -> None:
    assert mamba3_path_c._bwd_simd_p_reduction_supported(
        batch=1, heads=1, headdim=headdim
    )
    msl = dump_lowered_bwd_msl(batch=1, seq=2, heads=1, headdim=headdim, state=2)
    assert "kernel void" in msl
    assert "simd_sum(" in msl
    assert "red_buf_staging" in msl
    _assert_no_bwd_partial_outputs(msl)
    assert "dD_batch" in msl


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
    assert len(re.findall(r"float decay = exp\(", fwd)) == 1
    assert re.search(r"exp\([^;\n]+\) \* h_state", fwd) is None
    assert len(re.findall(r"sig_z = .*exp\(", fwd)) == 2
    assert "sig_z = exp(z_val)" in fwd
    assert re.search(r"z_val \* \([^;\n]+exp\(", fwd) is None

    bwd = dump_lowered_bwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert len(re.findall(r"float decay = exp\(", bwd)) == 1
    assert "h_snap" in bwd
    assert "1.000000e+00 / decay" not in bwd
    assert re.search(r"d_decay\[0\] \* exp\(", bwd) is None
    assert re.search(r"dh\[n_\d+\] = \(dh\[n_\d+\] \* exp\(", bwd) is None
    assert len(re.findall(r"sig_z = .*exp\(", bwd)) == 2
    assert "sig_z = exp(z_val)" in bwd
    assert re.search(r"dY \* \(z_val \* \([^;\n]+exp\(", bwd) is None
    assert "y_state = sig_z" not in bwd
    assert "dx_inp = sig_z" not in bwd
    assert "d_decay = sig_z" not in bwd
    assert "sig_z = sig_z" not in bwd


def test_raw_lowering_uses_tilelang_metal_scalar_pipeline() -> None:
    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._bwd_simd_reduce_kernel_for_state_snapshots(
        1, 4, 1, 2, 4
    )
    assert lowering.grid == (1, 1, 1)
    assert lowering.threadgroup == (2, 1, 1)
    assert "h_snap" in lowering.body
    assert "float decay = exp(" in lowering.body
    assert "1.000000e+00 / decay" not in lowering.body
    assert "exp((A_val * dt_val))" not in lowering.body


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

    with pytest.raises(RuntimeError, match="does not expose partial owner-output"):
        mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0, out=owner_outputs)


def test_bwd_path_c_unsupported_p_reduction_shape_has_no_partial_fallback() -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=2, heads=1, headdim=33, state=2, dtype=mx.float32)
    dy = mx.ones(inputs[0].shape, dtype=mx.float32)
    mx.eval(dy)

    with pytest.raises(RuntimeError, match="no host-reduced partial fallback"):
        mamba3_mimo_bwd_path_c(dy, *inputs)


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
    """Path C emits a sign-split SiLU and avoids exp(-z) overflow in-kernel."""

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
