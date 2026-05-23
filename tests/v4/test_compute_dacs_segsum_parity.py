"""V7-D31 (za1.11): compute_dacs_segsum bit-exact vs numpy ref."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.kernels.compute_dacs_segsum import (
    compute_dacs_segsum, compute_dacs_segsum_numpy_ref,
)


@pytest.mark.parametrize("chunk_size", [4, 8, 16, 32])
def test_compute_dacs_segsum_bit_exact_vs_numpy(chunk_size: int):
    np.random.seed(chunk_size)
    dt_np = np.random.randn(2, 64, 8).astype(np.float32)
    A_np = np.random.randn(2, 64, 8).astype(np.float32)
    dt = mx.array(dt_np)
    A = mx.array(A_np)

    got = compute_dacs_segsum(dt, A, chunk_size=chunk_size)
    mx.eval(got)
    got_np = np.array(got, copy=False)

    ref = compute_dacs_segsum_numpy_ref(dt_np, A_np, chunk_size=chunk_size)
    # fp32 cumsum order is identical between mlx and numpy.
    np.testing.assert_allclose(got_np, ref, atol=1e-5, rtol=1e-5)


def test_compute_dacs_segsum_shape_mismatch_raises():
    dt = mx.zeros((1, 4, 2))
    A = mx.zeros((1, 4, 3))
    with pytest.raises(ValueError):
        compute_dacs_segsum(dt, A)
