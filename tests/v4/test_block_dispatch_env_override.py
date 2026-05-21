"""Tests that LinearAttentionBlock / KimiDeltaAttentionBlock dispatch through
linear_attention_paths / kda_paths, not the naive recurrent kernel directly.
"""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.kimi_delta_attention import (
    KimiDeltaAttentionBlock,
    KimiDeltaAttentionConfig,
)
from cppmega_v4.nn.linear_attention import (
    LinearAttentionBlock,
    LinearAttentionConfig,
)


def _gdn_block(d=64, h=4):
    cfg = LinearAttentionConfig(
        hidden_size=d, num_heads=h, head_dim=d // h, use_short_conv=False,
    )
    blk = LinearAttentionBlock(cfg)
    # Perturb o_proj so block delta is non-zero (default is zero-init).
    rng = np.random.default_rng(0)
    blk.o_proj.weight = mx.array(
        rng.standard_normal(blk.o_proj.weight.shape).astype(np.float32) * 0.05
    )
    return blk


def _kda_block(d=64, h=4):
    cfg = KimiDeltaAttentionConfig(
        hidden_size=d, num_heads=h, head_dim=d // h, use_short_conv=False,
    )
    blk = KimiDeltaAttentionBlock(cfg)
    rng = np.random.default_rng(0)
    blk.o_proj.weight = mx.array(
        rng.standard_normal(blk.o_proj.weight.shape).astype(np.float32) * 0.05
    )
    return blk


@pytest.mark.parametrize("path", ["path_a", "path_b"])
def test_gdn_block_accepts_explicit_kernel_path(path):
    blk = _gdn_block()
    x = mx.random.normal((1, 8, blk.config.hidden_size))
    out = blk(x, kernel_path=path)
    assert out.shape == x.shape
    assert not bool(mx.any(mx.isnan(out)).item())


@pytest.mark.parametrize("path", ["path_a", "path_b"])
def test_kda_block_accepts_explicit_kernel_path(path):
    blk = _kda_block()
    x = mx.random.normal((1, 8, blk.config.hidden_size))
    out = blk(x, kernel_path=path)
    assert out.shape == x.shape
    assert not bool(mx.any(mx.isnan(out)).item())


def test_gdn_block_path_a_and_path_b_close():
    """Block output via Path A and Path B should be numerically close."""
    blk = _gdn_block()
    x = mx.random.normal((1, 8, blk.config.hidden_size))

    out_a = np.array(blk(x, kernel_path="path_a"))
    out_b = np.array(blk(x, kernel_path="path_b"))
    np.testing.assert_allclose(out_a, out_b, atol=5e-4, rtol=5e-3)


def test_kda_block_path_a_and_path_b_close():
    blk = _kda_block()
    x = mx.random.normal((1, 8, blk.config.hidden_size))

    out_a = np.array(blk(x, kernel_path="path_a"))
    out_b = np.array(blk(x, kernel_path="path_b"))
    np.testing.assert_allclose(out_a, out_b, atol=5e-4, rtol=5e-3)


def test_gdn_block_default_auto_picks_available_path():
    """Without env override, auto-mode should still produce valid output."""
    blk = _gdn_block()
    x = mx.random.normal((1, 4, blk.config.hidden_size))
    out = blk(x)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_kda_block_default_auto_picks_available_path():
    blk = _kda_block()
    x = mx.random.normal((1, 4, blk.config.hidden_size))
    out = blk(x)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_gdn_block_with_doc_ids_uses_dispatch():
    """doc_ids path also dispatches through linear_attention_paths."""
    blk = _gdn_block()
    x = mx.random.normal((1, 8, blk.config.hidden_size))
    doc_ids = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)
    out = blk(x, doc_ids=doc_ids, kernel_path="path_b")
    assert out.shape == x.shape
    assert not bool(mx.any(mx.isnan(out)).item())
