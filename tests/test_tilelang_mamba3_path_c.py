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

from typing import cast

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang import (
    mamba3_mimo_apply,
    mamba3_mimo_bwd_metal,
    mamba3_mimo_fwd_metal,
    mamba3_mimo_reference,
)
from cppmega_mlx.nn._tilelang.mamba3_path_c import (
    Mamba3PathCStatus,
    dump_lowered_bwd_msl,
    dump_lowered_fwd_msl,
    mamba3_mimo_apply_path_c,
    mamba3_mimo_bwd_path_c,
    mamba3_mimo_fwd_path_c,
    mamba3_mimo_path_c_status,
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


# ---------------------------------------------------------------------------
# Forward parity tests
# ---------------------------------------------------------------------------


def test_fwd_path_c_matches_path_b_fp32_small_shape() -> None:
    """Path C fwd matches Path B Metal fwd within FP32 tolerance."""

    inputs = _make_inputs(batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    y_pb, h_pb = mamba3_mimo_fwd_metal(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_pb), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(_np(h_pc), _np(h_pb), rtol=1e-3, atol=1e-4)


def test_fwd_path_c_matches_reference_fp32_small_shape() -> None:
    """Path C fwd also matches the pure-MLX reference."""

    inputs = _make_inputs(batch=1, seq=8, heads=2, headdim=4, state=4, dtype=mx.float32)
    y_pc, h_pc = mamba3_mimo_fwd_path_c(*inputs)
    y_ref, h_ref = mamba3_mimo_reference(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_ref), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_np(h_pc), _np(h_ref), rtol=1e-4, atol=1e-5)


def test_fwd_path_c_matches_path_b_at_bench_shape_fp32() -> None:
    """At the spec bench shape (B=2,T=512,H=4,P=32,N=64) Path C matches Path B."""

    inputs = _make_inputs(
        batch=2, seq=512, heads=4, headdim=32, state=64, dtype=mx.float32
    )
    y_pc, _ = mamba3_mimo_fwd_path_c(*inputs)
    y_pb, _ = mamba3_mimo_fwd_metal(*inputs)
    np.testing.assert_allclose(_np(y_pc), _np(y_pb), rtol=1e-3, atol=1e-4)


def test_fwd_path_c_matches_path_b_at_fp16() -> None:
    """FP16 callers up-cast internally; outputs match Path B with FP16 carrier
    tolerance."""

    inputs = _make_inputs(
        batch=1, seq=64, heads=2, headdim=8, state=16, dtype=mx.float16
    )
    y_pc, _ = mamba3_mimo_fwd_path_c(*inputs)
    y_pb, _ = mamba3_mimo_fwd_metal(*inputs)
    diff = float(np.max(np.abs(_np(y_pc.astype(mx.float32)) - _np(y_pb.astype(mx.float32)))))
    norm = float(np.sqrt((_np(y_pb.astype(mx.float32)) ** 2).sum()))
    assert diff <= max(1e-3 * norm + 1e-3, 5e-3), f"max_abs={diff}, norm={norm}"


# ---------------------------------------------------------------------------
# Backward parity tests
# ---------------------------------------------------------------------------


def test_bwd_path_c_matches_path_b_fp32_small_shape() -> None:
    """Path C bwd kernel emits the same partials as Path B (after host reduction)."""

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


def test_bwd_path_c_via_vjp_matches_autograd_through_reference_fp32() -> None:
    """Full VJP through mx.custom_function matches autograd-through-reference."""

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
    names = ["x", "B", "C", "z", "A", "dt", "D", "h0"]
    for name, gr, gp in zip(names, g_ref, g_pc):
        gr_np = _np(gr)
        gp_np = _np(gp)
        np.testing.assert_allclose(gp_np, gr_np, rtol=5e-3, atol=1e-5,
                                    err_msg=f"VJP mismatch on {name}")


def test_bwd_path_c_via_vjp_matches_path_b_vjp_at_bench_shape() -> None:
    """At bench shape, Path C's VJP grads match Path B's VJP grads."""

    inputs = _make_inputs(
        batch=2, seq=512, heads=4, headdim=32, state=64, dtype=mx.float32
    )

    def pc_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y = cast(mx.array, mamba3_mimo_apply_path_c(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    def pb_loss(x, B, C, z, A, dt, D, h0):  # type: ignore[no-untyped-def]
        y = cast(mx.array, mamba3_mimo_apply(x, B, C, z, A, dt, D, h0))
        return mx.sum(y * y) * 0.5

    g_pc = mx.grad(pc_loss, argnums=tuple(range(8)))(*inputs)
    g_pb = mx.grad(pb_loss, argnums=tuple(range(8)))(*inputs)
    mx.eval(*g_pc, *g_pb)
    names = ["x", "B", "C", "z", "A", "dt", "D", "h0"]
    for name, gpc, gpb in zip(names, g_pc, g_pb):
        np.testing.assert_allclose(_np(gpc), _np(gpb), rtol=1e-3, atol=1e-4,
                                    err_msg=f"VJP mismatch on {name}")


def test_bwd_path_c_returns_correct_zero_seq_shapes() -> None:
    inputs = _make_inputs(batch=1, seq=0, heads=1, headdim=2, state=2, dtype=mx.float32)
    x, B, C, z, A, dt, D, h0 = inputs
    dy = mx.zeros_like(x)
    grads = mamba3_mimo_bwd_path_c(dy, x, B, C, z, A, dt, D, h0)
    assert grads[0].shape == x.shape
    assert grads[7].shape == h0.shape
