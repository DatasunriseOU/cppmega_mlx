"""Tests for cppmega_v4.nn._external.tilekernels_mhc — port of DeepSeek mHC torch reference.

Includes parity tests against the original PyTorch reference at
``~/sources/TileKernels/tile_kernels/torch/mhc.py`` when torch is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn._external.tilekernels_mhc import (
    expand_to_mhc_ref,
    mhc_head_compute_mix_ref,
    mhc_post_ref,
    mhc_pre_apply_mix_ref,
    mhc_pre_norm_fn_ref,
    mhc_pre_split_mixes_ref,
    sinkhorn_normalize_ref,
)


# ----- shape contracts -----


def test_expand_to_mhc_ref_shape():
    hidden = mx.random.normal((2, 3, 8))
    out = expand_to_mhc_ref(hidden, mhc_mult=4)
    assert out.shape == (2, 3, 4, 8)


def test_sinkhorn_normalize_converges_to_doubly_stochastic():
    """After enough iterations row sums and col sums should each be ~1."""
    x = mx.random.normal((1, 1, 4, 4))
    out = sinkhorn_normalize_ref(x, repeat=20)
    row_sums = np.array(mx.sum(out, axis=-1))
    col_sums = np.array(mx.sum(out, axis=-2))
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-3)
    np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=1e-3)


def test_head_compute_mix_returns_sigmoid_plus_eps():
    input_mix = mx.zeros((2, 3, 4))
    scale = mx.array([1.0])
    base = mx.array([0.0])
    out = mhc_head_compute_mix_ref(input_mix, scale, base, mhc_pre_eps=0.01)
    # sigmoid(0) = 0.5; +eps = 0.51 everywhere.
    np.testing.assert_allclose(np.array(out), np.full((2, 3, 4), 0.51), atol=1e-6)


def test_pre_split_mixes_shapes():
    mhc_mult = 3
    last = 2 * mhc_mult + mhc_mult * mhc_mult  # = 15
    input_mixes = mx.random.normal((2, 4, last))
    scale = mx.array([0.5, 0.5, 0.5])
    base = mx.zeros((last,))
    pre, post, comb = mhc_pre_split_mixes_ref(
        input_mixes, scale, base, mhc_mult, mhc_post_mult_value=2.0, mhc_pre_eps=0.01
    )
    assert pre.shape == (2, 4, mhc_mult, 1)
    assert post.shape == (2, 4, mhc_mult, 1)
    assert comb.shape == (2, 4, mhc_mult, mhc_mult)


def test_pre_apply_mix_collapses_mhc_axis():
    x = mx.random.normal((2, 4, 3, 8))
    mix = mx.ones((2, 4, 3, 1))
    out = mhc_pre_apply_mix_ref(x, mix)
    assert out.shape == (2, 4, 8)


def test_post_returns_bfloat16():
    x = mx.random.normal((2, 4, 8)).astype(mx.bfloat16)
    residual = mx.random.normal((2, 4, 3, 8)).astype(mx.bfloat16)
    post_layer_mix = mx.ones((2, 4, 3, 1))
    comb_res_mix = mx.zeros((2, 4, 3, 3))
    out = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    assert out.dtype == mx.bfloat16
    assert out.shape == (2, 4, 3, 8)


def test_pre_norm_fn_shape():
    residual = mx.random.normal((2, 4, 3, 8)).astype(mx.float32)
    mhc_fn = mx.random.normal((3, 1, 8)).astype(mx.float32)
    out = mhc_pre_norm_fn_ref(residual, mhc_fn, mhc_norm_weight=None, mhc_norm_eps=1e-6)
    # residual.flatten(2,3) -> (2, 4, 24); mixes reshaped to (2, 4, -1).
    assert out.shape[0] == 2 and out.shape[1] == 4


# ----- parity vs PyTorch TileKernels reference -----


@pytest.fixture(scope="module")
def tk_torch():
    torch = pytest.importorskip("torch")
    repo = Path("/Users/dave/sources/TileKernels")
    if not repo.exists():
        pytest.skip("TileKernels repo not present at expected path")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from tile_kernels.torch.mhc import (
            expand_to_mhc_ref as tk_expand,
            mhc_head_compute_mix_ref as tk_head_mix,
            mhc_post_ref as tk_post,
            mhc_pre_apply_mix_ref as tk_pre_apply,
            mhc_pre_split_mixes_ref as tk_pre_split,
            sinkhorn_normalize_ref as tk_sinkhorn,
        )
    except Exception as exc:
        pytest.skip(f"could not import TileKernels torch.mhc: {exc}")
    return {
        "torch": torch,
        "expand": tk_expand,
        "sinkhorn": tk_sinkhorn,
        "head_mix": tk_head_mix,
        "pre_split": tk_pre_split,
        "pre_apply": tk_pre_apply,
        "post": tk_post,
    }


def test_parity_expand_to_mhc(tk_torch):
    rng = np.random.default_rng(1)
    h_np = rng.standard_normal((2, 3, 8)).astype(np.float32)
    t_out = tk_torch["expand"](tk_torch["torch"].from_numpy(h_np), 4)
    m_out = expand_to_mhc_ref(mx.array(h_np), 4)
    np.testing.assert_allclose(np.array(m_out), t_out.numpy(), atol=1e-6)


def test_parity_sinkhorn_normalize(tk_torch):
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    t_out = tk_torch["sinkhorn"](tk_torch["torch"].from_numpy(x_np), repeat=10)
    m_out = sinkhorn_normalize_ref(mx.array(x_np), repeat=10)
    np.testing.assert_allclose(np.array(m_out), t_out.numpy(), atol=1e-5, rtol=1e-4)


def test_parity_head_compute_mix(tk_torch):
    torch = tk_torch["torch"]
    rng = np.random.default_rng(3)
    inp = rng.standard_normal((2, 3, 4)).astype(np.float32)
    scale = np.array([0.7], dtype=np.float32)
    base = np.array([0.1], dtype=np.float32)
    t_out = tk_torch["head_mix"](
        torch.from_numpy(inp), torch.from_numpy(scale), torch.from_numpy(base), 0.01
    )
    m_out = mhc_head_compute_mix_ref(
        mx.array(inp), mx.array(scale), mx.array(base), 0.01
    )
    np.testing.assert_allclose(np.array(m_out), t_out.numpy(), atol=1e-6)


def test_parity_pre_split_mixes(tk_torch):
    torch = tk_torch["torch"]
    mhc_mult = 3
    last = 2 * mhc_mult + mhc_mult * mhc_mult
    rng = np.random.default_rng(4)
    inp = rng.standard_normal((2, 4, last)).astype(np.float32)
    scale = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    base = np.zeros((last,), dtype=np.float32)
    t_pre, t_post, t_comb = tk_torch["pre_split"](
        torch.from_numpy(inp), torch.from_numpy(scale), torch.from_numpy(base),
        mhc_mult, 2.0, 0.01,
    )
    m_pre, m_post, m_comb = mhc_pre_split_mixes_ref(
        mx.array(inp), mx.array(scale), mx.array(base), mhc_mult, 2.0, 0.01,
    )
    np.testing.assert_allclose(np.array(m_pre), t_pre.numpy(), atol=1e-6)
    np.testing.assert_allclose(np.array(m_post), t_post.numpy(), atol=1e-6)
    np.testing.assert_allclose(np.array(m_comb), t_comb.numpy(), atol=1e-6)


def test_parity_pre_apply_mix(tk_torch):
    torch = tk_torch["torch"]
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal((2, 4, 3, 8)).astype(np.float32)
    mix_np = rng.standard_normal((2, 4, 3, 1)).astype(np.float32)
    t_out = tk_torch["pre_apply"](torch.from_numpy(x_np), torch.from_numpy(mix_np))
    m_out = mhc_pre_apply_mix_ref(mx.array(x_np), mx.array(mix_np))
    # Both should be bfloat16 — cast back to fp32 for comparison.
    np.testing.assert_allclose(
        np.array(m_out.astype(mx.float32)), t_out.float().numpy(), atol=5e-3
    )
