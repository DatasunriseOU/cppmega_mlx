"""Tests for ROI 9 — real CSA + HCA hybrid V4 attention."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.csa_hca_v4 import (
    CSAHCAConfig,
    CSAHCAHybridV4,
    _compress_kv,
    _compressed_attention,
)


def _cfg(**overrides):
    base = dict(
        hidden_size=64, num_heads=4, head_dim=16,
        m_csa=2, m_hca=4,
    )
    base.update(overrides)
    return CSAHCAConfig(**base)


# ----- Config validation -----


def test_csahca_config_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="must equal"):
        CSAHCAConfig(hidden_size=64, num_heads=3, head_dim=16)


def test_csahca_config_rejects_m_hca_smaller_than_m_csa():
    with pytest.raises(ValueError, match=">= m_csa"):
        CSAHCAConfig(hidden_size=64, num_heads=4, head_dim=16,
                     m_csa=4, m_hca=2)


# ----- Compression primitive -----


def test_compress_kv_mean_pool_shape():
    B, H, S, D = 1, 2, 12, 8
    k = mx.random.normal((B, H, S, D))
    v = mx.random.normal((B, H, S, D))
    k_c, v_c, n_super = _compress_kv(k, v, m=4)
    assert n_super == 3
    assert k_c.shape == (B, H, 3, D)
    assert v_c.shape == (B, H, 3, D)


def test_compress_kv_pads_when_not_divisible():
    B, H, S, D = 1, 1, 10, 4
    k = mx.random.normal((B, H, S, D))
    v = mx.random.normal((B, H, S, D))
    k_c, v_c, n_super = _compress_kv(k, v, m=4)
    # S=10 → pad to 12 → 3 super-tokens.
    assert n_super == 3
    # First super-token = mean of first 4 tokens of original k.
    expected_first = k[:, :, :4, :].mean(axis=2)
    np.testing.assert_allclose(np.array(k_c[:, :, 0, :]),
                                np.array(expected_first), atol=1e-5)


def test_compressed_attention_causal_in_supertokens():
    B, H, S, D = 1, 1, 8, 4
    rng = np.random.default_rng(7)
    q = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    k_c, v_c, _ = _compress_kv(k, v, m=4)  # 2 super-tokens
    out = _compressed_attention(
        q, k_c, v_c, m=4, original_seq=S, scale=D ** -0.5,
    )
    # Query at position 0 → only super-token 0 visible → out[0] == v_c[0].
    np.testing.assert_allclose(
        np.array(out[:, :, 0, :]), np.array(v_c[:, :, 0, :]), atol=1e-5,
    )


def test_compressed_attention_respects_select_indices():
    B, H, S, D = 1, 1, 8, 4
    rng = np.random.default_rng(11)
    q = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    k_c, v_c, _ = _compress_kv(k, v, m=4)  # 2 super-tokens
    # Force every query to only see super-token 0 via select indices.
    sel = mx.zeros((B, H, S, 1), dtype=mx.int32)
    out = _compressed_attention(
        q, k_c, v_c, m=4, original_seq=S, scale=D ** -0.5,
        select_indices=sel,
    )
    # All queries that can causally see ST 0 must produce v_c[0].
    # ST 0 spans tokens [0, 4); queries at positions [0, 3] all only see ST 0.
    np.testing.assert_allclose(
        np.array(out[:, :, :4, :]),
        np.tile(np.array(v_c[:, :, 0:1, :]), (1, 1, 4, 1)),
        atol=1e-5,
    )


# ----- Module end-to-end -----


def test_csahca_module_forward_shape():
    cfg = _cfg()
    mod = CSAHCAHybridV4(cfg)
    x = mx.random.normal((1, 16, cfg.hidden_size))
    y = mod(x)
    assert y.shape == x.shape
    assert not bool(mx.any(mx.isnan(y)).item())


def test_csahca_module_accepts_select_indices():
    cfg = _cfg()
    mod = CSAHCAHybridV4(cfg)
    B, S = 1, 16
    x = mx.random.normal((B, S, cfg.hidden_size))
    # CSA uses m_csa=2 → 8 supers; HCA uses m_hca=4 → 4 supers
    csa_sel = mx.zeros((B, cfg.num_heads, S, 2), dtype=mx.int32)
    hca_sel = mx.zeros((B, cfg.num_heads, S, 1), dtype=mx.int32)
    y = mod(x, csa_select_indices=csa_sel, hca_select_indices=hca_sel)
    assert y.shape == x.shape


def test_csahca_module_rejects_wrong_hidden():
    cfg = _cfg()
    mod = CSAHCAHybridV4(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size + 1))
    with pytest.raises(ValueError, match="must be"):
        mod(x)


def test_csahca_short_sequence_handles_gracefully():
    cfg = _cfg(m_csa=4, m_hca=8)
    mod = CSAHCAHybridV4(cfg)
    x = mx.random.normal((1, 3, cfg.hidden_size))  # S=3 < both m values
    y = mod(x)
    assert y.shape == x.shape
    assert not bool(mx.any(mx.isnan(y)).item())
