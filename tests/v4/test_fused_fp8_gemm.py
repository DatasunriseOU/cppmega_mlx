"""Tests for fused FP8 GEMM Metal kernel."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4._tilelang.fused_fp8_gemm import fused_fp8_gemm
from cppmega_v4.nn._external._mlx_lm_fp8_dequant_vendored import dequant_block_fp8


def _make_fp8(m, n, seed=0):
    rng = np.random.default_rng(seed)
    bs = 128
    blocks_m = (m + bs - 1) // bs
    blocks_n = (n + bs - 1) // bs
    scale_inv = mx.array(rng.uniform(0.5, 1.5, (blocks_m, blocks_n)).astype(np.float32))
    bf = mx.array(rng.standard_normal((m, n)).astype(np.float32) * 0.1).astype(mx.bfloat16)
    pad_b = (-m) % bs
    pad_s = (-n) % bs
    padded = mx.pad(bf.astype(mx.float32), ((0, pad_b), (0, pad_s)))
    blocks = padded.reshape(blocks_m, bs, blocks_n, bs)
    scaled = (blocks / scale_inv[:, None, :, None]).reshape(m + pad_b, n + pad_s)[:m, :n]
    fp8 = mx.to_fp8(scaled)
    return fp8, scale_inv


def test_fused_fp8_gemm_shape():
    M, K, B = 128, 128, 4
    w, s = _make_fp8(M, K)
    a = mx.random.normal((B, K))
    out = fused_fp8_gemm(w, s, a)
    assert out.shape == (B, M)


def test_fused_fp8_gemm_matches_dequant_then_matmul():
    """Fused kernel must match the unfused dequant→matmul path within fp8 precision."""
    M, K, B = 128, 128, 8
    w, s = _make_fp8(M, K, seed=7)
    rng = np.random.default_rng(99)
    a = mx.array(rng.standard_normal((B, K)).astype(np.float32))

    out_fused = fused_fp8_gemm(w, s, a)
    # Reference path: dequant then matmul.
    w_bf16 = dequant_block_fp8(w, s)
    out_ref = (a.astype(mx.float32) @ w_bf16.T.astype(mx.float32))

    np.testing.assert_allclose(
        np.array(out_fused.astype(mx.float32)),
        np.array(out_ref),
        atol=2e-2, rtol=5e-2,  # fp8 rounding inside the kernel
    )


def test_fused_fp8_gemm_rejects_dtype_mismatch():
    M, K = 128, 128
    w = mx.zeros((M, K), dtype=mx.float32)
    s = mx.ones((1, 1), dtype=mx.float32)
    a = mx.zeros((1, K))
    with pytest.raises(TypeError, match="uint8"):
        fused_fp8_gemm(w, s, a)


def test_fused_fp8_gemm_rejects_shape_mismatch():
    M, K = 128, 128
    w, s = _make_fp8(M, K)
    a = mx.zeros((1, K + 1))
    with pytest.raises(ValueError, match="a.shape"):
        fused_fp8_gemm(w, s, a)


def test_fused_fp8_gemm_handles_non_block_aligned():
    """Shapes not divisible by 128 must still produce correct (n, M) output."""
    M, K, B = 100, 200, 2
    w, s = _make_fp8(M, K)
    a = mx.random.normal((B, K))
    out = fused_fp8_gemm(w, s, a)
    assert out.shape == (B, M)
    assert np.all(np.isfinite(np.array(out)))
