"""ROI 8/9 — NSA / CSA+HCA scaffold tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.sparse_attention_v4 import (
    CsaHcaHybridAttention,
    CsaHcaHybridConfig,
    NativeSparseAttention,
    NativeSparseAttentionConfig,
)


# ----- NSA -----


def test_nsa_config_validation():
    with pytest.raises(ValueError):
        NativeSparseAttentionConfig(hidden_size=0, num_heads=2, head_dim=8)
    with pytest.raises(ValueError):
        NativeSparseAttentionConfig(hidden_size=15, num_heads=2, head_dim=8)
    with pytest.raises(ValueError):
        NativeSparseAttentionConfig(
            hidden_size=16, num_heads=2, head_dim=8, compress_block_size=0
        )


def test_nsa_forward_shape():
    cfg = NativeSparseAttentionConfig(hidden_size=16, num_heads=2, head_dim=8)
    nsa = NativeSparseAttention(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = nsa(x)
    assert out.shape == x.shape


def test_nsa_is_identity_at_init():
    cfg = NativeSparseAttentionConfig(hidden_size=16, num_heads=2, head_dim=8)
    nsa = NativeSparseAttention(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = nsa(x)
    np.testing.assert_allclose(np.array(out), np.zeros_like(np.array(x)), atol=1e-5)


def test_nsa_rejects_bad_input():
    cfg = NativeSparseAttentionConfig(hidden_size=16, num_heads=2, head_dim=8)
    nsa = NativeSparseAttention(cfg)
    with pytest.raises(ValueError, match="x must be"):
        nsa(mx.zeros((1, 4, 17)))


# ----- CSA + HCA -----


def test_csahca_config_validation():
    with pytest.raises(ValueError):
        CsaHcaHybridConfig(hidden_size=16, num_heads=0, head_dim=8)
    with pytest.raises(ValueError):
        CsaHcaHybridConfig(hidden_size=15, num_heads=2, head_dim=8)
    with pytest.raises(ValueError):
        CsaHcaHybridConfig(hidden_size=16, num_heads=2, head_dim=8, m_token_compression=0)


def test_csahca_forward_shape():
    cfg = CsaHcaHybridConfig(hidden_size=16, num_heads=2, head_dim=8)
    blk = CsaHcaHybridAttention(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = blk(x)
    assert out.shape == x.shape


def test_csahca_is_identity_at_init():
    cfg = CsaHcaHybridConfig(hidden_size=16, num_heads=2, head_dim=8)
    blk = CsaHcaHybridAttention(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size))
    out = blk(x)
    np.testing.assert_allclose(np.array(out), np.zeros_like(np.array(x)), atol=1e-5)


def test_csahca_gradient_flows_through_projections():
    cfg = CsaHcaHybridConfig(hidden_size=16, num_heads=2, head_dim=8)
    blk = CsaHcaHybridAttention(cfg)
    blk.o_proj.weight = mx.random.normal(blk.o_proj.weight.shape) * 0.1
    x = mx.random.normal((1, 4, cfg.hidden_size))

    def loss_fn(params):
        blk.update(params)
        return mx.mean(mx.square(blk(x)))

    grads = mx.grad(loss_fn)(blk.trainable_parameters())
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        g = grads[name]["weight"]
        assert float(mx.max(mx.abs(g)).item()) > 0.0, f"{name} got zero grad"
