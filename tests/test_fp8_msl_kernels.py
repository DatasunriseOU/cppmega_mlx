"""Tests for the retired FP8 (e4m3fn) direct-MSL compatibility helpers.

The helpers live in ``cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py`` and now use
MLX reference math only. These tests exercise:

1. Status: the direct-MSL surface is explicitly retired.
2. Encode round-trip via the LUT decode is bit-exact against ``mx.from_fp8``
   modulo the ``-0.0`` sign-of-zero corner.
3. ``fp8_scaled_matmul_raw`` matches a "dequant + ``mx.matmul``" oracle
   bit-exactly at fp32 precision (no scale factor).
4. ``fp8_scaled_vecmat`` matches the same oracle at M=1.
5. Special-value handling: zero, near-max (448.0), and the e4m3fn NaN
   encoding all decode without crashing.
6. ``mx.custom_function`` VJP returns finite gradients with the expected
   shapes for the autograd-aware ``fp8_scaled_matmul`` wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from cppmega_mlx.nn._tilelang.fp8_msl_kernels import (
    FP8MSLKernelStatus,
    __license_notice__,
    fp8_msl_status,
    fp8_scaled_matmul,
    fp8_scaled_matmul_raw,
    fp8_scaled_vecmat,
    fp8_to_half,
    half_to_fp8,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Status assertions
# ---------------------------------------------------------------------------


def test_fp8_msl_status_returns_dataclass() -> None:
    status = fp8_msl_status()
    assert isinstance(status, FP8MSLKernelStatus)
    assert status.dispatch_surface == "retired_direct_msl_pure_mlx_reference"
    assert status.available is False
    assert "direct-MSL Path B is retired" in status.reason
    assert status.normal_path_available is False
    assert "fp8_matmul_path_c.py" in status.normal_path_reason


def test_fp8_msl_license_notice_present() -> None:
    assert "Apache 2.0" in __license_notice__
    assert "MIT" in __license_notice__
    assert "AppMana" in __license_notice__
    assert "audiohacking" in __license_notice__


def test_fp8_msl_source_no_longer_constructs_direct_msl() -> None:
    source = (REPO_ROOT / "cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py").read_text()
    assert "make_metal_kernel" not in source
    assert "_msl_transform" not in source
    assert "mx.fast.metal_kernel" not in source
    assert "_FP8_MATMUL_BODY" not in source


def test_checked_in_fp8_helper_receipt_reports_retired_status() -> None:
    receipt = json.loads(
        (REPO_ROOT / "bench/tilelang_ports/fp8_msl_kernels.json").read_text()
    )
    assert receipt["kernel"] == "fp8_reference_helpers"
    assert receipt["metal_status"]["available"] is False
    assert "direct-MSL Path B is retired" in receipt["metal_status"]["reason"]


# ---------------------------------------------------------------------------
# Encode / decode round-trip
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_round_trip_encode_then_decode_close_to_original() -> None:
    """Encode a representable set of values, decode, expect <= 1 ULP fp16 noise."""

    # Pick values that lie on the e4m3fn grid so the round-trip is exact:
    # multiples of {0.5, 1.0, 2.0, ...} in the normal range.
    vals_np = np.array(
        [0.5, 1.0, 1.5, 2.0, 4.0, -0.5, -1.0, -2.0, 7.5, -7.5, 16.0, 256.0],
        dtype=np.float16,
    )
    x = mx.array(vals_np)
    fp8 = half_to_fp8(x)
    rec = fp8_to_half(fp8)
    mx.eval(fp8, rec)

    rec_np = np.asarray(rec).astype(np.float32)
    assert np.allclose(rec_np, vals_np.astype(np.float32), rtol=0.0, atol=0.0)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_round_trip_off_grid_values_within_3bit_mantissa_noise() -> None:
    """Off-grid fp16 values should encode/decode within FP8's 3-bit mantissa noise."""

    rng = np.random.default_rng(31)
    vals_np = (rng.standard_normal(64).astype(np.float32) * 0.5).astype(np.float16)
    x = mx.array(vals_np)
    fp8 = half_to_fp8(x)
    rec = fp8_to_half(fp8)
    mx.eval(fp8, rec)

    rec_f32 = np.asarray(rec).astype(np.float32)
    src_f32 = vals_np.astype(np.float32)
    err = np.abs(rec_f32 - src_f32)
    rel = err / (np.abs(src_f32) + 1e-9)
    # 3-bit mantissa: 1/16 = 0.0625 absolute relative noise. Allow 2x for
    # the half-precision intermediate.
    assert rel.max() < 0.13, f"max rel err {rel.max():.4e} exceeded 13%"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_round_trip_matches_mx_from_fp8_modulo_signed_zero() -> None:
    """The vendored decode and ``mx.from_fp8`` should match on positive values."""

    vals_np = np.array(
        [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 448.0],
        dtype=np.float16,
    )
    x = mx.array(vals_np)
    fp8 = half_to_fp8(x)
    rec_ours = fp8_to_half(fp8)
    rec_mlx = mx.from_fp8(fp8, dtype=mx.float16)
    mx.eval(rec_ours, rec_mlx)

    np.testing.assert_array_equal(np.asarray(rec_ours), np.asarray(rec_mlx))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_zero_inf_nan_handling() -> None:
    """The retired helper follows MLX's ``from_fp8`` edge-byte contract."""

    # Manually construct uint8 bytes with edge cases.
    bytes_np = np.array(
        [
            0x00,  # +0
            0x80,  # -0
            0x7E,  # +max normal (448.0)
            0xFE,  # -max normal (-448.0)
            0x7F,  # NaN (positive)
            0xFF,  # NaN (negative)
        ],
        dtype=np.uint8,
    )
    fp8 = mx.array(bytes_np)
    decoded = fp8_to_half(fp8)
    expected = mx.from_fp8(fp8, dtype=mx.float16)
    mx.eval(decoded, expected)
    decoded_np = np.asarray(decoded).astype(np.float32)

    # Indices 0..3 are finite.
    assert decoded_np[0] == 0.0
    assert decoded_np[1] == 0.0
    assert decoded_np[2] == 448.0
    assert decoded_np[3] == -448.0
    np.testing.assert_array_equal(np.asarray(decoded), np.asarray(expected))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_encode_dtype_validation() -> None:
    """``half_to_fp8`` requires fp16 input; fp32 must raise ``TypeError``."""

    x_f32 = mx.array([0.5, 1.0], dtype=mx.float32)
    with pytest.raises(TypeError, match="float16"):
        half_to_fp8(x_f32)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_decode_dtype_validation() -> None:
    """``fp8_to_half`` requires uint8 input; fp16 must raise ``TypeError``."""

    fp16 = mx.array([0.5, 1.0], dtype=mx.float16)
    with pytest.raises(TypeError, match="uint8"):
        fp8_to_half(fp16)


# ---------------------------------------------------------------------------
# Scaled matmul parity (no scale factor)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_matmul_parity_per_tensor_scale_unity() -> None:
    """Kernel output must be bit-exact against ``mx.from_fp8 + mx.matmul``."""

    rng = np.random.default_rng(0)
    M, N, K = 16, 24, 32
    A = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float32))
    B = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))

    A_fp8 = mx.to_fp8(A)
    B_fp8 = mx.to_fp8(B)
    mx.eval(A_fp8, B_fp8)

    sa = mx.array([1.0], dtype=mx.float32)
    sb = mx.array([1.0], dtype=mx.float32)
    C = fp8_scaled_matmul_raw(A_fp8, B_fp8, scale_a=sa, scale_b=sb)

    A_rec = mx.from_fp8(A_fp8, dtype=mx.float32)
    B_rec = mx.from_fp8(B_fp8, dtype=mx.float32)
    C_ref = mx.matmul(A_rec, mx.swapaxes(B_rec, 0, 1))

    mx.eval(C, C_ref)
    np.testing.assert_array_equal(np.asarray(C), np.asarray(C_ref))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_matmul_per_tensor_scale_applies() -> None:
    """Per-tensor scales should multiply through into the output."""

    rng = np.random.default_rng(1)
    M, N, K = 8, 12, 16
    A = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float32))
    B = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))

    A_fp8 = mx.to_fp8(A)
    B_fp8 = mx.to_fp8(B)
    mx.eval(A_fp8, B_fp8)

    sa = mx.array([2.0], dtype=mx.float32)
    sb = mx.array([3.0], dtype=mx.float32)
    C = fp8_scaled_matmul_raw(A_fp8, B_fp8, scale_a=sa, scale_b=sb)
    mx.eval(C)

    A_rec = mx.from_fp8(A_fp8, dtype=mx.float32) * 2.0
    B_rec = mx.from_fp8(B_fp8, dtype=mx.float32) * 3.0
    C_ref = mx.matmul(A_rec, mx.swapaxes(B_rec, 0, 1))
    mx.eval(C_ref)
    np.testing.assert_allclose(np.asarray(C), np.asarray(C_ref), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_matmul_per_row_scale_applies() -> None:
    """Per-row (per-channel) scales should multiply each row independently."""

    rng = np.random.default_rng(2)
    M, N, K = 8, 8, 16
    A = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float32))
    B = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))

    A_fp8 = mx.to_fp8(A)
    B_fp8 = mx.to_fp8(B)
    mx.eval(A_fp8, B_fp8)

    sa = mx.array(rng.uniform(0.5, 2.0, size=M).astype(np.float32))
    sb = mx.array(rng.uniform(0.5, 2.0, size=N).astype(np.float32))
    C = fp8_scaled_matmul_raw(A_fp8, B_fp8, scale_a=sa, scale_b=sb)
    mx.eval(C)

    A_rec = mx.from_fp8(A_fp8, dtype=mx.float32) * sa.reshape(M, 1)
    B_rec = mx.from_fp8(B_fp8, dtype=mx.float32) * sb.reshape(N, 1)
    C_ref = mx.matmul(A_rec, mx.swapaxes(B_rec, 0, 1))
    mx.eval(C_ref)
    np.testing.assert_allclose(np.asarray(C), np.asarray(C_ref), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_matmul_shape_validation() -> None:
    """Mismatched K dims must raise ``ValueError``."""

    A_fp8 = mx.zeros((4, 8), dtype=mx.uint8)
    B_fp8 = mx.zeros((4, 16), dtype=mx.uint8)
    sa = mx.array([1.0], dtype=mx.float32)
    sb = mx.array([1.0], dtype=mx.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        fp8_scaled_matmul_raw(A_fp8, B_fp8, scale_a=sa, scale_b=sb)


# ---------------------------------------------------------------------------
# Vecmat parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_vecmat_parity() -> None:
    """Vec-matmul output must match dequantize + ``mx.matmul`` exactly."""

    rng = np.random.default_rng(0)
    N, K = 24, 32
    x = mx.array((rng.standard_normal((K,)) * 0.1).astype(np.float32))
    W = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))

    x_fp8 = mx.to_fp8(x)
    W_fp8 = mx.to_fp8(W)
    mx.eval(x_fp8, W_fp8)

    sx = mx.array([1.0], dtype=mx.float32)
    sw = mx.array([1.0], dtype=mx.float32)
    y = fp8_scaled_vecmat(x_fp8, W_fp8, scale_x=sx, scale_w=sw)

    x_rec = mx.from_fp8(x_fp8, dtype=mx.float32)
    W_rec = mx.from_fp8(W_fp8, dtype=mx.float32)
    y_ref = mx.matmul(W_rec, x_rec)

    mx.eval(y, y_ref)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(y_ref))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_vecmat_per_row_scale() -> None:
    """Per-row scale on W must multiply each output row."""

    rng = np.random.default_rng(7)
    N, K = 16, 32
    x = mx.array((rng.standard_normal((K,)) * 0.1).astype(np.float32))
    W = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))

    x_fp8 = mx.to_fp8(x)
    W_fp8 = mx.to_fp8(W)
    mx.eval(x_fp8, W_fp8)

    sx = mx.array([1.5], dtype=mx.float32)
    sw = mx.array(rng.uniform(0.5, 2.0, size=N).astype(np.float32))
    y = fp8_scaled_vecmat(x_fp8, W_fp8, scale_x=sx, scale_w=sw)
    mx.eval(y)

    x_rec = mx.from_fp8(x_fp8, dtype=mx.float32) * 1.5
    W_rec = mx.from_fp8(W_fp8, dtype=mx.float32) * sw.reshape(N, 1)
    y_ref = mx.matmul(W_rec, x_rec)
    mx.eval(y_ref)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_vecmat_K_must_be_multiple_of_4() -> None:
    x_fp8 = mx.zeros((33,), dtype=mx.uint8)
    W_fp8 = mx.zeros((8, 33), dtype=mx.uint8)
    sx = mx.array([1.0], dtype=mx.float32)
    sw = mx.array([1.0], dtype=mx.float32)
    with pytest.raises(ValueError, match="multiple of 4"):
        fp8_scaled_vecmat(x_fp8, W_fp8, scale_x=sx, scale_w=sw)


# ---------------------------------------------------------------------------
# Autograd VJP
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_matmul_vjp_returns_finite_grads() -> None:
    """The ``mx.custom_function`` VJP must produce finite gradients."""

    rng = np.random.default_rng(11)
    M, N, K = 8, 12, 16
    A = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float32))
    B = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    A_fp8 = mx.to_fp8(A)
    B_fp8 = mx.to_fp8(B)
    sa = mx.array([0.7], dtype=mx.float32)
    sb = mx.array([1.3], dtype=mx.float32)
    mx.eval(A_fp8, B_fp8)

    def loss(scale_a, scale_b):
        out = fp8_scaled_matmul(A_fp8, B_fp8, scale_a, scale_b)
        return mx.sum(out * out)

    grads = mx.grad(loss, argnums=(0, 1))(sa, sb)
    mx.eval(*grads)
    for g in grads:
        g_np = np.asarray(g)
        assert np.all(np.isfinite(g_np)), "FP8 matmul VJP grads must be finite"


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal unavailable")
def test_fp8_scaled_matmul_vjp_matches_dequant_oracle() -> None:
    """VJP grads must match what you'd get from dequant + ``mx.matmul`` directly."""

    rng = np.random.default_rng(13)
    M, N, K = 4, 6, 8
    A = mx.array((rng.standard_normal((M, K)) * 0.1).astype(np.float32))
    B = mx.array((rng.standard_normal((N, K)) * 0.1).astype(np.float32))
    A_fp8 = mx.to_fp8(A)
    B_fp8 = mx.to_fp8(B)
    sa = mx.array([0.5], dtype=mx.float32)
    sb = mx.array([1.5], dtype=mx.float32)
    mx.eval(A_fp8, B_fp8)

    cot = mx.array((rng.standard_normal((M, N)) * 0.1).astype(np.float32))

    # Use mx.vjp for direct comparison against the dequant + matmul oracle.
    def kernel_call(sa_in, sb_in):
        return fp8_scaled_matmul(A_fp8, B_fp8, sa_in, sb_in)

    _, kernel_vjps = mx.vjp(kernel_call, (sa, sb), (cot,))

    def oracle_call(sa_in, sb_in):
        a = mx.from_fp8(A_fp8, dtype=mx.float32) * sa_in[0]
        b = mx.from_fp8(B_fp8, dtype=mx.float32) * sb_in[0]
        return mx.matmul(a, mx.swapaxes(b, 0, 1))

    _, oracle_vjps = mx.vjp(oracle_call, (sa, sb), (cot,))
    mx.eval(*kernel_vjps, *oracle_vjps)

    for kg, og in zip(kernel_vjps, oracle_vjps):
        np.testing.assert_allclose(
            np.asarray(kg), np.asarray(og), rtol=1e-5, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Empty-input fallback
# ---------------------------------------------------------------------------


def test_fp8_to_half_empty_returns_empty() -> None:
    fp8 = mx.zeros((0,), dtype=mx.uint8)
    out = fp8_to_half(fp8)
    mx.eval(out)
    assert out.shape == (0,)
    assert out.dtype == mx.float16


def test_half_to_fp8_empty_returns_empty() -> None:
    x = mx.zeros((0,), dtype=mx.float16)
    out = half_to_fp8(x)
    mx.eval(out)
    assert out.shape == (0,)
    assert out.dtype == mx.uint8
