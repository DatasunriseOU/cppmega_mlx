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
  - small fp16 carrier shape (FP32 internal accumulator preserves precision).

The "bit-exact" expectation is a property of the M4 Max instance running this
test; the conservative atol/rtol budget is what we ship as the contract.
"""

import json
import re
from typing import cast

import numpy as np
import pytest

import mlx.core as mx

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
    mamba3_path_c_schedule_plan,
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
    # The bwd kernel emits the partials, dh0, dx, dz, etc. plus the scratch.
    for name in ("A", "B", "C", "D", "dt", "dy", "h0", "h_steps", "x", "z"):
        assert name in msl, f"input buffer {name!r} missing from lowered MSL"
    for name in (
        "dA_partial",
        "dB_partial",
        "dC_partial",
        "dD_partial",
        "ddt_partial",
        "dh0",
        "dx",
        "dz",
    ):
        assert name in msl, f"output buffer {name!r} missing from lowered MSL"


def test_lowered_msl_reuses_hot_scalar_temporaries() -> None:
    """TileLang CSE plus scalar binding reuse avoids hot exp/sigmoid recompute."""

    fwd = dump_lowered_fwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert "float y_acc = " in fwd
    assert "thread float y_acc[1]" not in fwd
    assert len(re.findall(r"float decay = exp\(", fwd)) == 1
    assert re.search(r"exp\([^;\n]+\) \* h_state", fwd) is None
    assert len(re.findall(r"float sig_z = .*exp\(", fwd)) == 1
    assert re.search(r"z_val \* \([^;\n]+exp\(", fwd) is None

    bwd = dump_lowered_bwd_msl(batch=1, seq=4, heads=1, headdim=2, state=4)
    assert len(re.findall(r"float decay = exp\(", bwd)) == 1
    assert len(re.findall(r"float decay_1 = exp\(", bwd)) == 1
    assert re.search(r"d_decay\[0\] \* exp\(", bwd) is None
    assert re.search(r"dh\[n_\d+\] = \(dh\[n_\d+\] \* exp\(", bwd) is None
    assert len(re.findall(r"float sig_z = .*exp\(", bwd)) == 1
    assert re.search(r"dY \* \(z_val \* \([^;\n]+exp\(", bwd) is None


def test_raw_lowering_uses_tilelang_metal_scalar_pipeline() -> None:
    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._bwd_kernel_for(
        1, 4, 1, 2, 4, return_msl=True
    )
    assert lowering.grid == (1, 1, 1)
    assert lowering.threadgroup == (2, 1, 1)
    assert "float decay = exp(" in lowering.body
    assert "float decay_1 = exp(" in lowering.body
    assert "exp((A_val * dt_val))" not in lowering.body
    assert "exp((A_val_1 * dt_val_1))" not in lowering.body


def test_path_c_launch_geometry_comes_from_tilelang_lowering() -> None:
    _require_mamba3_path_c()

    _kernel, lowering = mamba3_path_c._fwd_kernel_for(
        1, 4, 56, 64, 4, return_msl=True
    )
    assert lowering is not None
    assert lowering.grid == (14, 1, 1)
    assert lowering.threadgroup == (256, 1, 1)
    assert _msl_transform.metal_grid_for_lowering(lowering) == (3584, 1, 1)


def test_mamba3_path_c_schedule_plan_uses_rule_and_z3_for_spec_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("z3")
    for name in (
        "TILELANG_DISABLE_Z3",
        "CPPMEGA_DISABLE_Z3",
        "CPPMEGA_DISABLE_MAMBA3_PATH_C_Z3",
    ):
        monkeypatch.delenv(name, raising=False)
    mamba3_path_c_schedule_plan.cache_clear()

    plan = mamba3_path_c_schedule_plan(
        batch=2,
        seq=512,
        heads=4,
        headdim=32,
        state=64,
        dtype="float32",
    )

    assert isinstance(plan, Mamba3PathCSchedulePlan)
    assert plan.threads == 256
    assert plan.grid_blocks == 1
    assert plan.fwd_path_c_candidate is True
    assert plan.bwd_path_c_candidate is False
    assert plan.mode == "path_c_fwd_path_b_bwd"
    assert plan.z3_used is True
    assert plan.z3_proved is True


def test_mamba3_path_c_receipt_gate_requires_matching_shape_and_fwd_win(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("z3")
    for name in (
        "TILELANG_DISABLE_Z3",
        "CPPMEGA_DISABLE_Z3",
        "CPPMEGA_DISABLE_MAMBA3_PATH_C_Z3",
    ):
        monkeypatch.delenv(name, raising=False)
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
    mamba3_path_c._fwd_kernel_for.cache_clear()
    mamba3_path_c._bwd_kernel_for.cache_clear()

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


def test_bwd_path_c_owner_outputs_avoid_hidden_zero_alloc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=6, heads=2, headdim=4, state=4, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    dy = mx.ones(x.shape, dtype=mx.float32)
    owner_outputs = (
        mx.zeros((1, 2, 4, 6, 4), dtype=mx.float32),
        mx.zeros(x.shape, dtype=mx.float32),
        mx.zeros(z.shape, dtype=mx.float32),
        mx.zeros((1, 6, 2, 4, 4), dtype=mx.float32),
        mx.zeros((1, 6, 2, 4, 4), dtype=mx.float32),
        mx.zeros((1, 6, 2, 4), dtype=mx.float32),
        mx.zeros((1, 6, 2, 4), dtype=mx.float32),
        mx.zeros((1, 2, 4), dtype=mx.float32),
        mx.zeros(h0.shape, dtype=mx.float32),
    )
    mx.eval(dy, *owner_outputs)

    def fail_zero_alloc(*_args: object, **_kwargs: object) -> mx.array:
        raise AssertionError("owner-output bwd route must not allocate mx.zeros")

    monkeypatch.setattr(mamba3_path_c.mx, "zeros", fail_zero_alloc)

    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0, out=owner_outputs)
    mx.eval(*grads)
    assert grads[0] is owner_outputs[1]
    assert grads[3] is owner_outputs[2]
    assert grads[7] is owner_outputs[8]


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


def test_fwd_path_c_matches_reference_fp32_small_shape() -> None:
    """Path C fwd also matches the pure-MLX reference."""

    _require_mamba3_path_c()
    inputs = _make_inputs(batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    y_ref, h_ref = mamba3_mimo_reference(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_ref), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_np(h_pc), _np(h_ref), rtol=1e-4, atol=1e-5)


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
    """Path C must not materialize large hidden cast buffers for non-FP32 inputs."""

    _require_mamba3_path_c()
    inputs = _make_inputs(
        batch=1, seq=64, heads=2, headdim=8, state=16, dtype=mx.float16
    )
    with pytest.raises(RuntimeError, match="without hidden casts"):
        mamba3_mimo_fwd_path_c(*inputs)


def test_bwd_path_c_rejects_fp16_without_hidden_casts() -> None:
    """Backward follows the same FP32-only no-hidden-cast ABI as forward."""

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
    """Path C bwd kernel emits the same partials as Path B (after host reduction)."""

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
