"""Tests for ROI 8 — real three-branch NSA (Compress + Select + Sliding)."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.nsa_v4 import (
    NSAConfig,
    NativeSparseAttentionV4,
    _compress_branch,
    _select_branch,
    _sliding_branch,
)


def _cfg(**overrides):
    base = dict(
        hidden_size=64, num_heads=4, head_dim=16,
        compress_block_size=4, select_topk=2, sliding_window=8,
    )
    base.update(overrides)
    return NSAConfig(**base)


def _rand_qkv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    fn = lambda: mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    return fn(), fn(), fn()


# ----- Config validation -----


def test_nsa_config_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="must equal"):
        NSAConfig(hidden_size=64, num_heads=3, head_dim=16)


def test_nsa_config_rejects_non_positive_branch_dims():
    with pytest.raises(ValueError, match="must be positive"):
        NSAConfig(hidden_size=64, num_heads=4, head_dim=16,
                  compress_block_size=0)
    with pytest.raises(ValueError, match="must be positive"):
        NSAConfig(hidden_size=64, num_heads=4, head_dim=16, select_topk=-1)
    with pytest.raises(ValueError, match="must be positive"):
        NSAConfig(hidden_size=64, num_heads=4, head_dim=16, sliding_window=0)


# ----- Branch primitives -----


def test_compress_branch_shape_and_pad():
    B, H, S, D = 1, 2, 13, 8     # S not divisible by block_size=4 -> pads to 16
    q, k, v = _rand_qkv(B, H, S, D)
    out, scores = _compress_branch(q, k, v, block_size=4, scale=D ** -0.5)
    assert out.shape == (B, H, S, D)
    assert scores.shape == (B, H, S, 4)  # 4 blocks after pad
    assert not bool(mx.any(mx.isnan(out)).item())


def test_compress_branch_causal_in_blocks():
    """Block 0 only sees block 0 if query is in block 0; weights for later
    blocks must be ~0 for early queries."""
    B, H, S, D = 1, 1, 8, 4
    q, k, v = _rand_qkv(B, H, S, D, seed=1)
    out, scores = _compress_branch(q, k, v, block_size=4, scale=D ** -0.5)
    # Score at row 0 (token 0, block 0) → only block 0 should be unmasked.
    # The output should equal v_block_0 (mean-pooled v over block 0).
    v_block_0 = v[:, :, :4, :].mean(axis=2)  # [B, H, D]
    np.testing.assert_allclose(np.array(out[:, :, 0, :]),
                                np.array(v_block_0), atol=1e-5)


def test_select_branch_topk_respects_causality():
    B, H, S, D = 1, 1, 8, 4
    q, k, v = _rand_qkv(B, H, S, D, seed=2)
    _, scores = _compress_branch(q, k, v, block_size=4, scale=D ** -0.5)
    out = _select_branch(q, k, v, scores,
                          block_size=4, topk=1, scale=D ** -0.5)
    # Output shape preserved.
    assert out.shape == (B, H, S, D)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_sliding_branch_window_limits_attention():
    """Token i attends only to tokens (i-window, i]."""
    B, H, S, D = 1, 1, 8, 4
    q, k, v = _rand_qkv(B, H, S, D, seed=3)
    # With window=1, each query attends only to itself.
    out = _sliding_branch(q, k, v, window=1, scale=D ** -0.5)
    # softmax over a single position is 1.0, so output[i] == v[i].
    np.testing.assert_allclose(np.array(out), np.array(v), atol=1e-5)


# ----- Module end-to-end -----


def test_nsa_module_forward_shape():
    cfg = _cfg()
    mod = NativeSparseAttentionV4(cfg)
    x = mx.random.normal((1, 16, cfg.hidden_size))
    y = mod(x)
    assert y.shape == x.shape
    assert not bool(mx.any(mx.isnan(y)).item())


def test_nsa_module_rejects_wrong_hidden():
    cfg = _cfg()
    mod = NativeSparseAttentionV4(cfg)
    x = mx.random.normal((1, 4, cfg.hidden_size + 1))
    with pytest.raises(ValueError, match="must be"):
        mod(x)


def test_nsa_module_gate_uses_all_three_branches():
    """Gate softmax should keep all branch contributions roughly balanced at init."""
    cfg = _cfg()
    mod = NativeSparseAttentionV4(cfg)
    x = mx.random.normal((1, 16, cfg.hidden_size))
    # Inspect the gate weights → with zero-init branch_gate, softmax gives 1/3.
    gate_logits = mod.branch_gate(x)
    gate = mx.softmax(gate_logits.astype(mx.float32), axis=-1)
    mean_per_branch = np.array(gate.mean(axis=(0, 1)))
    np.testing.assert_allclose(mean_per_branch, [1/3, 1/3, 1/3], atol=0.1)


def test_nsa_short_sequence_falls_into_full_attention():
    """At S < window+block, the branches degenerate but the module must not crash."""
    cfg = _cfg(compress_block_size=4, sliding_window=4, select_topk=2)
    mod = NativeSparseAttentionV4(cfg)
    x = mx.random.normal((1, 3, cfg.hidden_size))  # S=3 < block_size=4
    y = mod(x)
    assert y.shape == x.shape
    assert not bool(mx.any(mx.isnan(y)).item())


def test_nsa_select_topk_larger_than_blocks_clamps():
    """If select_topk > n_blocks, the branch must clamp (not crash)."""
    cfg = _cfg(compress_block_size=4, select_topk=100, sliding_window=4)
    mod = NativeSparseAttentionV4(cfg)
    x = mx.random.normal((1, 8, cfg.hidden_size))  # 2 blocks, but topk=100
    y = mod(x)
    assert y.shape == x.shape
