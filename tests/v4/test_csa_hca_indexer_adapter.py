"""Tests for the Lightning Indexer → CSA+HCA select-indices adapter."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.csa_hca_indexer_adapter import (
    _tokens_to_super_indices,
    apply_indexer_to_csa_hca,
)
from cppmega_v4.nn.csa_hca_v4 import CSAHCAConfig, CSAHCAHybridV4
from cppmega_v4.nn.lightning_indexer import (
    LightningIndexer,
    LightningIndexerConfig,
)
from cppmega_v4.nn.lightning_indexer_fp8 import (
    LightningIndexerFP8,
    LightningIndexerFP8Config,
)


# ----- Token → super-token projection -----


def test_token_to_super_floor_div():
    """super_idx[b, s, k] = token[b, s, k] // m, dedup then broadcast."""
    # B=1, S=2, top_k=4; m=4; k_super=2; H=1
    tok = mx.array([[[0, 3, 8, 9], [4, 5, 6, 7]]], dtype=mx.int32)  # [1,2,4]
    out = _tokens_to_super_indices(tok, m=4, k_super=2, num_heads=1)
    # tok//4 = [[[0,0,2,2],[1,1,1,1]]], dedup → [[[0,2],[1,1]]]
    expected = np.array([[[[0, 2], [1, 1]]]], dtype=np.int32)  # [1,1,2,2]
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(np.array(out), expected)


def test_token_to_super_dedup_and_pad():
    """All top-k tokens map to the same super → pad with the dedup result."""
    tok = mx.array([[[0, 1, 2, 3]]], dtype=mx.int32)   # all in super 0
    out = _tokens_to_super_indices(tok, m=8, k_super=3, num_heads=1)
    # tok//8 = [[[0,0,0,0]]] → dedup [0] → pad to k_super=3 → [0,0,0]
    np.testing.assert_array_equal(
        np.array(out), np.array([[[[0, 0, 0]]]], dtype=np.int32),
    )


def test_token_to_super_broadcasts_across_heads():
    tok = mx.array([[[0, 8, 16]]], dtype=mx.int32)
    out = _tokens_to_super_indices(tok, m=8, k_super=3, num_heads=4)
    assert out.shape == (1, 4, 1, 3)
    # All H heads see the same indices.
    for h in range(4):
        np.testing.assert_array_equal(
            np.array(out[0, h, 0]), np.array(out[0, 0, 0]),
        )


def test_token_to_super_validation():
    tok = mx.array([[[0]]], dtype=mx.int32)
    with pytest.raises(ValueError, match="m must be positive"):
        _tokens_to_super_indices(tok, m=0, k_super=1, num_heads=1)
    with pytest.raises(ValueError, match="k_super must be positive"):
        _tokens_to_super_indices(tok, m=2, k_super=0, num_heads=1)
    with pytest.raises(ValueError, match="num_heads must be positive"):
        _tokens_to_super_indices(tok, m=2, k_super=1, num_heads=0)


# ----- End-to-end: indexer (fp32) + CSA+HCA -----


def _freqs(seq, rope_half, seed=0):
    rng = np.random.default_rng(seed)
    cos = mx.array(rng.uniform(-1, 1, (seq, rope_half)).astype(np.float32))
    sin = mx.array(rng.uniform(-1, 1, (seq, rope_half)).astype(np.float32))
    return cos, sin


def test_end_to_end_with_lightning_indexer_fp32():
    B, S, H, D = 1, 16, 4, 16
    hidden = H * D
    indexer = LightningIndexer(LightningIndexerConfig(
        hidden_size=hidden, n_heads=2, head_dim=32, rope_head_dim=16,
        q_lora_rank=hidden, index_topk=8,
    ))
    csa_hca = CSAHCAHybridV4(CSAHCAConfig(
        hidden_size=hidden, num_heads=H, head_dim=D, m_csa=2, m_hca=4,
    ))

    x = mx.random.normal((B, S, hidden))
    qr = mx.random.normal((B, S, hidden))
    cos, sin = _freqs(S, 8)

    out = apply_indexer_to_csa_hca(indexer, csa_hca, x, qr, (cos, sin))
    assert out.shape == (B, S, hidden)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_end_to_end_with_lightning_indexer_fp8():
    B, S, H, D = 1, 16, 4, 16
    hidden = H * D
    indexer = LightningIndexerFP8(LightningIndexerFP8Config(
        hidden_size=hidden, n_heads=2, head_dim=32, rope_head_dim=16,
        q_lora_rank=hidden, index_topk=8, fp8_blocks=True,
    ))
    csa_hca = CSAHCAHybridV4(CSAHCAConfig(
        hidden_size=hidden, num_heads=H, head_dim=D, m_csa=2, m_hca=4,
    ))

    x = mx.random.normal((B, S, hidden))
    qr = mx.random.normal((B, S, hidden))
    cos, sin = _freqs(S, 8)

    out = apply_indexer_to_csa_hca(indexer, csa_hca, x, qr, (cos, sin))
    assert out.shape == (B, S, hidden)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_select_indices_actually_restricts_attention():
    """Force k_super=1: each query sees exactly one super-token. The CSA+HCA
    output then differs from the unrestricted (all-super) call."""
    B, S, H, D = 1, 16, 2, 8
    hidden = H * D
    indexer = LightningIndexer(LightningIndexerConfig(
        hidden_size=hidden, n_heads=2, head_dim=16, rope_head_dim=8,
        q_lora_rank=hidden, index_topk=4,
    ))
    csa_hca = CSAHCAHybridV4(CSAHCAConfig(
        hidden_size=hidden, num_heads=H, head_dim=D, m_csa=2, m_hca=4,
    ))

    x = mx.random.normal((B, S, hidden))
    qr = mx.random.normal((B, S, hidden))
    cos, sin = _freqs(S, 4)

    out_with = np.array(apply_indexer_to_csa_hca(
        indexer, csa_hca, x, qr, (cos, sin),
        k_super_csa=1, k_super_hca=1,
    ))
    out_without = np.array(csa_hca(x))   # no select_indices → all-super
    # Restricting attention to 1 super-token per query should change the output.
    assert not np.allclose(out_with, out_without, atol=1e-6)
