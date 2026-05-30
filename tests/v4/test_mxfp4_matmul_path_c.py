"""Tier-3 Path C #10 regression: MXFP4 e2m1 dequant + cooperative GEMM.

Covers the real Path C surface in
``cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c``:

  * vectorized MLX dequant matches the per-row reference codec
  * the cooperative-tensor ``matmul2d`` GEMM matches a dequant-then-MLX
    matmul reference within the 4-bit tolerance (~5e-2)
  * RULE #1: a sub-tile shape RAISES (no silent degraded path)
  * RULE #1: a non-OCP block size RAISES

The Metal GEMM tests skip cleanly when MLX Metal is unavailable.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from cppmega_mlx.quant.mxfp4_metal import (
    MXFP4_BLOCK_SIZE,
    dequantize_mxfp4_blockwise,
    quantize_mxfp4_blockwise,
)
from cppmega_mlx.nn._tilelang import _msl_transform
from cppmega_mlx.nn._tilelang.mxfp4_matmul_path_c import (
    MXFP4MatmulPathCError,
    mxfp4_dequant_blockwise_2d,
    mxfp4_matmul_path_c,
)


_METAL = _msl_transform.can_run_metal()


def _quant_matrix(x_np: np.ndarray):
    """Pack an ``(rows, cols)`` fp32 matrix row-major (blocks tile cols=K)."""
    rows, cols = x_np.shape
    assert cols % MXFP4_BLOCK_SIZE == 0
    qrows, srows = [], []
    for r in range(rows):
        qd, sc = quantize_mxfp4_blockwise(mx.array(x_np[r]))
        qrows.append(np.asarray(qd))
        srows.append(np.asarray(sc))
    return mx.array(np.stack(qrows)), mx.array(np.stack(srows))


def test_vectorized_dequant_matches_reference_codec():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((16, 64)).astype(np.float32)
    Aq, As = _quant_matrix(A)
    got = np.asarray(
        mxfp4_dequant_blockwise_2d(Aq, As, rows=16, cols=64, out_dtype=mx.float32)
    ).astype(np.float32)
    ref = np.stack([
        np.asarray(dequantize_mxfp4_blockwise(
            mx.array(np.asarray(Aq)[r]), mx.array(np.asarray(As)[r]), numel=64))
        for r in range(16)
    ]).astype(np.float32)
    assert float(np.max(np.abs(got - ref))) < 1e-3


def test_dequant_bad_block_size_raises():
    rng = np.random.default_rng(1)
    Aq, As = _quant_matrix(rng.standard_normal((16, 32)).astype(np.float32))
    with pytest.raises(MXFP4MatmulPathCError, match="block_size"):
        mxfp4_dequant_blockwise_2d(Aq, As, rows=16, cols=32, block_size=8)


@pytest.mark.skipif(not _METAL, reason="MLX Metal backend unavailable")
@pytest.mark.parametrize("shape", [(32, 64, 64), (128, 128, 128)])
def test_matmul2d_gemm_parity(shape):
    M, N, K = shape
    rng = np.random.default_rng(7)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((N, K)).astype(np.float32)
    Aq, As = _quant_matrix(A)
    Bq, Bs = _quant_matrix(B)
    A_dq = mxfp4_dequant_blockwise_2d(Aq, As, rows=M, cols=K, out_dtype=mx.float32)
    B_dq = mxfp4_dequant_blockwise_2d(Bq, Bs, rows=N, cols=K, out_dtype=mx.float32)
    C_ref = np.asarray(mx.matmul(A_dq, B_dq.T)).astype(np.float32)

    out = mx.zeros((M, N), dtype=mx.float32)
    C = np.asarray(
        mxfp4_matmul_path_c(Aq, As, Bq, Bs, M=M, N=N, K=K, out=out)
    ).astype(np.float32)
    rel = float(np.max(np.abs(C - C_ref)) / (np.max(np.abs(C_ref)) + 1e-6))
    assert rel < 5e-2, f"{shape}: rel_maxdiff {rel}"


@pytest.mark.skipif(not _METAL, reason="MLX Metal backend unavailable")
def test_subtile_shape_raises_no_silent_fallback():
    """RULE #1: a shape with no legal cooperative tile RAISES."""
    rng = np.random.default_rng(3)
    Aq, As = _quant_matrix(rng.standard_normal((16, 32)).astype(np.float32))
    Bq, Bs = _quant_matrix(rng.standard_normal((16, 32)).astype(np.float32))
    out = mx.zeros((16, 16), dtype=mx.float32)
    with pytest.raises(MXFP4MatmulPathCError, match="no legal"):
        mxfp4_matmul_path_c(Aq, As, Bq, Bs, M=16, N=16, K=32, out=out)
