"""ROI 7 — Lightning Indexer scaffold tests."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.lightning_indexer import (
    LightningIndexer,
    LightningIndexerConfig,
    _apply_non_interleaved_rope,
)


def _freqs_cis(seq_len: int, half_dim: int) -> tuple[mx.array, mx.array]:
    pos = mx.arange(seq_len, dtype=mx.float32)
    freq = 1.0 / (10000.0 ** (mx.arange(half_dim, dtype=mx.float32) / half_dim))
    angles = pos[:, None] * freq[None, :]
    return mx.cos(angles), mx.sin(angles)


def test_config_validation():
    with pytest.raises(ValueError):
        LightningIndexerConfig(hidden_size=16, n_heads=2, head_dim=0)
    with pytest.raises(ValueError):
        LightningIndexerConfig(hidden_size=16, n_heads=2, head_dim=16, rope_head_dim=24)
    with pytest.raises(ValueError):
        LightningIndexerConfig(hidden_size=16, n_heads=0, head_dim=16)
    with pytest.raises(ValueError):
        LightningIndexerConfig(hidden_size=16, n_heads=2, head_dim=16, index_topk=0)


def test_non_interleaved_rope_zero_angle_is_identity():
    """At cos=1, sin=0, RoPE leaves input untouched."""
    d_half = 4
    cos = mx.ones((d_half,))
    sin = mx.zeros((d_half,))
    x = mx.random.normal((2, 2 * d_half))
    out = _apply_non_interleaved_rope(x, cos, sin)
    np.testing.assert_allclose(np.array(out), np.array(x), atol=1e-6)


def test_non_interleaved_rope_90deg_rotates_halves():
    """At cos=0, sin=1, x1->−x2, x2->x1 (planar 90° rotation across halves)."""
    d_half = 3
    cos = mx.zeros((d_half,))
    sin = mx.ones((d_half,))
    x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    out = _apply_non_interleaved_rope(x, cos, sin)
    # First half rotated: [1,2,3]*0 - [4,5,6]*1 = [-4,-5,-6]
    # Second half rotated: [4,5,6]*0 + [1,2,3]*1 = [1,2,3]
    np.testing.assert_allclose(
        np.array(out), np.array([[-4.0, -5.0, -6.0, 1.0, 2.0, 3.0]]), atol=1e-6
    )


def test_non_interleaved_rope_rejects_odd_dim():
    with pytest.raises(ValueError, match="rope dim must be even"):
        _apply_non_interleaved_rope(mx.zeros((1, 3)), mx.zeros((1,)), mx.zeros((1,)))


def test_indexer_forward_shape():
    cfg = LightningIndexerConfig(
        hidden_size=32, n_heads=2, head_dim=8, rope_head_dim=4,
        q_lora_rank=16, index_topk=3,
    )
    ix = LightningIndexer(cfg)
    b, t = 1, 5
    x = mx.random.normal((b, t, cfg.hidden_size))
    qr = mx.random.normal((b, t, cfg.q_lora_rank))
    cos, sin = _freqs_cis(t, cfg.rope_head_dim // 2)
    topk_idx = ix(x, qr, (cos, sin))
    assert topk_idx.shape == (b, t, cfg.index_topk)
    assert topk_idx.dtype == mx.int32


def test_indexer_topk_clipped_to_seq():
    """If index_topk > T_kv, output is clipped to T_kv."""
    cfg = LightningIndexerConfig(
        hidden_size=16, n_heads=2, head_dim=8, rope_head_dim=4,
        q_lora_rank=8, index_topk=100,
    )
    ix = LightningIndexer(cfg)
    b, t = 1, 3
    x = mx.random.normal((b, t, cfg.hidden_size))
    qr = mx.random.normal((b, t, cfg.q_lora_rank))
    cos, sin = _freqs_cis(t, cfg.rope_head_dim // 2)
    topk_idx = ix(x, qr, (cos, sin))
    assert topk_idx.shape == (b, t, t)  # clipped to seq length


def test_indexer_indices_in_range():
    cfg = LightningIndexerConfig(
        hidden_size=16, n_heads=2, head_dim=8, rope_head_dim=4,
        q_lora_rank=8, index_topk=2,
    )
    ix = LightningIndexer(cfg)
    b, t = 2, 4
    x = mx.random.normal((b, t, cfg.hidden_size))
    qr = mx.random.normal((b, t, cfg.q_lora_rank))
    cos, sin = _freqs_cis(t, cfg.rope_head_dim // 2)
    topk_idx = np.array(ix(x, qr, (cos, sin)))
    assert (topk_idx >= 0).all() and (topk_idx < t).all()
