"""Tests for StreamingPoolCache — streaming m-token K/V mean-pool."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.streaming_kv_cache import StreamingPoolCache


def _rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def test_cache_construct_validation():
    with pytest.raises(ValueError, match="m must be positive"):
        StreamingPoolCache(m=0, batch=1, n_heads=1, head_dim=1)
    with pytest.raises(ValueError, match="batch / n_heads / head_dim"):
        StreamingPoolCache(m=2, batch=0, n_heads=1, head_dim=1)


def test_cache_append_validation():
    c = StreamingPoolCache(m=4, batch=1, n_heads=2, head_dim=4)
    bad = mx.zeros((1, 2, 2, 5))  # wrong head_dim
    with pytest.raises(ValueError, match="incompatible"):
        c.append(bad, bad)


def test_cache_matches_oneshot_mean_pool():
    """Streaming append must produce the same super-tokens as a one-shot
    reshape+mean over the full sequence."""
    B, T, H, D = 1, 16, 2, 4
    m = 4
    k = _rand((B, T, H, D), seed=1)
    v = _rand((B, T, H, D), seed=2)

    # Streaming path: append in 3 unequal chunks.
    c = StreamingPoolCache(m=m, batch=B, n_heads=H, head_dim=D)
    c.append(k[:, :5], v[:, :5])
    c.append(k[:, 5:10], v[:, 5:10])
    c.append(k[:, 10:], v[:, 10:])
    k_stream, v_stream = c.snapshot(include_partial=False)

    # One-shot path: reshape & mean.
    n_super = T // m
    k_ref = k[:, : n_super * m].reshape(B, n_super, m, H, D).mean(axis=2)
    v_ref = v[:, : n_super * m].reshape(B, n_super, m, H, D).mean(axis=2)

    np.testing.assert_allclose(np.array(k_stream), np.array(k_ref), atol=1e-5)
    np.testing.assert_allclose(np.array(v_stream), np.array(v_ref), atol=1e-5)


def test_cache_partial_block_finalized():
    """With include_partial=True, an in-progress block is finalized as
    sum / current_count."""
    B, T, H, D = 1, 6, 1, 2  # T=6, m=4 → 1 full + 2-token partial
    m = 4
    k = _rand((B, T, H, D), seed=3)
    v = _rand((B, T, H, D), seed=4)
    c = StreamingPoolCache(m=m, batch=B, n_heads=H, head_dim=D)
    c.append(k, v)
    k_with, _ = c.snapshot(include_partial=True)
    k_no, _ = c.snapshot(include_partial=False)
    assert k_with.shape == (B, 2, H, D)
    assert k_no.shape == (B, 1, H, D)
    # Partial block = mean of last 2 tokens.
    np.testing.assert_allclose(
        np.array(k_with[:, 1]), np.array(k[:, 4:6].mean(axis=1)), atol=1e-5,
    )


def test_cache_counters_tracked():
    B, T, H, D = 1, 10, 2, 4
    m = 4
    c = StreamingPoolCache(m=m, batch=B, n_heads=H, head_dim=D)
    assert c.total_tokens == 0
    assert c.n_super_completed == 0
    c.append(_rand((B, T, H, D)), _rand((B, T, H, D), seed=7))
    assert c.total_tokens == T
    assert c.n_super_completed == 2  # 10 // 4 = 2


def test_cache_reset_clears_state():
    B, H, D = 1, 1, 2
    c = StreamingPoolCache(m=2, batch=B, n_heads=H, head_dim=D)
    c.append(_rand((B, 5, H, D)), _rand((B, 5, H, D)))
    assert c.total_tokens == 5
    c.reset()
    assert c.total_tokens == 0
    assert c.n_super_completed == 0
    k, _ = c.snapshot()
    assert k.shape == (B, 0, H, D)


def test_cache_handles_single_token_appends():
    """Decode-style streaming: feed one token at a time."""
    B, H, D = 1, 1, 2
    m = 3
    rng = np.random.default_rng(42)
    full = mx.array(rng.standard_normal((B, 6, H, D)).astype(np.float32))
    c = StreamingPoolCache(m=m, batch=B, n_heads=H, head_dim=D)
    for t in range(6):
        c.append(full[:, t : t + 1], full[:, t : t + 1])
    k_stream, _ = c.snapshot(include_partial=False)
    assert k_stream.shape == (B, 2, H, D)
    # Compare to one-shot.
    k_ref = full.reshape(B, 2, m, H, D).mean(axis=2)
    np.testing.assert_allclose(np.array(k_stream), np.array(k_ref), atol=1e-5)
