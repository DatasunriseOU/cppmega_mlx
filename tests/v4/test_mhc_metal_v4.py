"""Tests for vendored mlx-lm PR #1189 mHC (HyperConnection + fused Sinkhorn).

Two surfaces:
  - HyperConnection / HyperHead nn.Modules (hyper_connection.py)
  - hc_split_sinkhorn pure function with optional fused Metal kernel
    (sinkhorn.py)

Parity test: Metal-kernel path must match the pure-MLX reference within
fp32 rounding for the same inputs.
"""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn._external._mlx_lm_hyper_connection_vendored import (
    HyperConnection,
    HyperHead,
)
from cppmega_v4.nn._external._mlx_lm_sinkhorn_vendored import (
    _make_sinkhorn_kernel,
    hc_split_sinkhorn,
)


def _rand_inputs(n=8, hc=4, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    mix_hc = (2 + hc) * hc
    mixes = mx.array(rng.standard_normal((n, mix_hc)).astype(np.float32))
    hc_scale = mx.array(rng.uniform(0.5, 1.5, (3,)).astype(np.float32))
    hc_base = mx.array(rng.standard_normal((mix_hc,)).astype(np.float32) * 0.1)
    return mixes, hc_scale, hc_base


def test_hc_split_sinkhorn_shapes():
    n, hc = 16, 4
    mixes, hc_scale, hc_base = _rand_inputs(n=n, hc=hc)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc)
    assert pre.shape == (n, hc)
    assert post.shape == (n, hc)
    assert comb.shape == (n, hc, hc)


def test_hc_split_sinkhorn_comb_is_doubly_stochastic():
    n, hc = 32, 4
    mixes, hc_scale, hc_base = _rand_inputs(n=n, hc=hc, seed=42)
    _, _, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc,
                                    sinkhorn_iters=20, eps=1e-6)
    rows = np.array(comb.sum(axis=2))
    cols = np.array(comb.sum(axis=1))
    # Doubly-stochastic ⇒ row sums and col sums ≈ 1.
    np.testing.assert_allclose(rows, np.ones_like(rows), atol=5e-3)
    np.testing.assert_allclose(cols, np.ones_like(cols), atol=5e-3)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_hc_split_sinkhorn_metal_matches_mlx():
    """Force the kernel path vs the pure-MLX fallback and compare."""
    n, hc = 16, 4
    mixes, hc_scale, hc_base = _rand_inputs(n=n, hc=hc, seed=7)
    # Kernel path (hc<=8 enables it on Metal)
    pre_k, post_k, comb_k = hc_split_sinkhorn(mixes, hc_scale, hc_base,
                                              hc_mult=hc, sinkhorn_iters=20)
    # Pure-MLX path: temporarily monkey-patch metal availability check by
    # passing hc_mult>8 forces the else branch — but we want the same hc.
    # Instead, recompute the else-branch math inline for comparison.
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]
    n_ = mixes.shape[0]
    eps = 1e-6
    pre_log = mixes[:, :hc] * s0 + hc_base[:hc]
    post_log = mixes[:, hc:2*hc] * s1 + hc_base[hc:2*hc]
    comb_log = (
        mixes[:, 2*hc:].reshape(n_, hc, hc) * s2
        + hc_base[2*hc:].reshape(hc, hc)
    )
    pre_ref = mx.sigmoid(pre_log) + eps
    post_ref = 2 * mx.sigmoid(post_log)
    comb_ref = mx.softmax(comb_log, axis=-1, precise=True) + eps
    col_sum = comb_ref.sum(axis=1, keepdims=True) + eps
    comb_ref = comb_ref / col_sum
    for _ in range(19):
        row_sum = comb_ref.sum(axis=2, keepdims=True) + eps
        comb_ref = comb_ref / row_sum
        col_sum = comb_ref.sum(axis=1, keepdims=True) + eps
        comb_ref = comb_ref / col_sum

    np.testing.assert_allclose(np.array(pre_k), np.array(pre_ref), atol=1e-6)
    np.testing.assert_allclose(np.array(post_k), np.array(post_ref), atol=1e-6)
    np.testing.assert_allclose(np.array(comb_k), np.array(comb_ref), atol=1e-3)


def test_hyper_connection_module_shapes():
    B, S, hc, D = 1, 4, 4, 16
    mod = HyperConnection(dim=D, hc_mult=hc, norm_eps=1e-5,
                          sinkhorn_iters=10, hc_eps=1e-6)
    x = mx.random.normal((B, S, hc, D))
    y, post, comb = mod.hc_pre(x)
    assert y.shape == (B, S, D)
    assert post.shape == (B, S, hc)
    assert comb.shape == (B, S, hc, hc)
    f_out = mx.random.normal((B, S, D))
    out = mod.hc_post(f_out, x, post, comb)
    assert out.shape == (B, S, hc, D)


def test_hyper_head_shape():
    B, S, hc, D = 1, 3, 4, 16
    head = HyperHead(dim=D, hc_mult=hc, norm_eps=1e-5, hc_eps=1e-6)
    x = mx.random.normal((B, S, hc, D))
    y = head(x)
    assert y.shape == (B, S, D)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal not available")
def test_sinkhorn_kernel_cached():
    """Repeated build for the same (hc, iters) must return the cached kernel."""
    k1 = _make_sinkhorn_kernel(4, 20, 1e-6)
    k2 = _make_sinkhorn_kernel(4, 20, 1e-6)
    assert k1 is k2
