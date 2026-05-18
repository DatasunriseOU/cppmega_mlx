"""Tests for CSAHCAHybridV4.decode_step + StreamingPoolCache integration."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.csa_hca_v4 import CSAHCAConfig, CSAHCAHybridV4
from cppmega_v4.nn.streaming_kv_cache import StreamingPoolCache


def _mk(B=1, H=2, D=8, hidden=None):
    hidden = hidden if hidden is not None else H * D
    cfg = CSAHCAConfig(
        hidden_size=hidden, num_heads=H, head_dim=D,
        m_csa=2, m_hca=4,
    )
    return CSAHCAHybridV4(cfg)


def test_make_streaming_caches_returns_two_caches():
    mod = _mk()
    csa_cache, hca_cache = mod.make_streaming_caches(batch=1)
    assert isinstance(csa_cache, StreamingPoolCache)
    assert isinstance(hca_cache, StreamingPoolCache)
    assert csa_cache.m == mod.config.m_csa
    assert hca_cache.m == mod.config.m_hca


def test_decode_step_validates_input_shape():
    mod = _mk()
    csa, hca = mod.make_streaming_caches(batch=1)
    bad = mx.zeros((1, 1, mod.config.hidden_size + 1))
    with pytest.raises(ValueError, match="must be"):
        mod.decode_step(bad, csa, hca)


def test_decode_step_single_token_output_shape():
    mod = _mk()
    csa, hca = mod.make_streaming_caches(batch=1)
    x_new = mx.random.normal((1, 1, mod.config.hidden_size))
    out = mod.decode_step(x_new, csa, hca)
    assert out.shape == (1, 1, mod.config.hidden_size)
    assert csa.total_tokens == 1
    assert hca.total_tokens == 1


def test_decode_step_accumulates_history():
    mod = _mk()
    csa, hca = mod.make_streaming_caches(batch=1)
    rng = np.random.default_rng(0)
    for t in range(8):
        x = mx.array(rng.standard_normal((1, 1, mod.config.hidden_size)).astype(np.float32))
        out = mod.decode_step(x, csa, hca)
        assert out.shape == (1, 1, mod.config.hidden_size)
    # After 8 tokens: m_csa=2 → 4 super-tokens completed; m_hca=4 → 2.
    assert csa.total_tokens == 8
    assert hca.total_tokens == 8
    assert csa.n_super_completed == 4
    assert hca.n_super_completed == 2


def test_streaming_decode_produces_finite_per_step_outputs():
    """Decode token-by-token: each step must be shape-correct, finite, and
    distinct across timesteps (proves the cache evolves rather than
    repeating the same compressed K/V)."""
    mod = _mk()
    rng = np.random.default_rng(7)
    S = 8
    x_full = mx.array(rng.standard_normal((1, S, mod.config.hidden_size)).astype(np.float32))
    csa, hca = mod.make_streaming_caches(batch=1)
    outs = []
    for t in range(S):
        out_t = np.array(mod.decode_step(x_full[:, t : t + 1], csa, hca))
        assert out_t.shape == (1, 1, mod.config.hidden_size)
        assert np.all(np.isfinite(out_t))
        outs.append(out_t)
    # First step output ≠ second step output (different history → different KV).
    assert not np.allclose(outs[0][0, 0], outs[-1][0, 0], atol=1e-6)


def test_streaming_decode_cache_kv_matches_recomputed_full():
    """The streaming cache's snapshot K/V must equal the full prefill's
    _compress_kv output at the same step count. This is the *correctness*
    contract: cache evolution is identical to recomputing from scratch."""
    from cppmega_v4.nn.csa_hca_v4 import _compress_kv

    mod = _mk()
    rng = np.random.default_rng(13)
    S = 8
    x_full = mx.array(rng.standard_normal((1, S, mod.config.hidden_size)).astype(np.float32))
    # Drive the cache token-by-token.
    csa, hca = mod.make_streaming_caches(batch=1)
    # Pre-compute the full K/V the *prefill* path would see.
    B = 1
    k_full = mod.k_proj(x_full).reshape(B, S, mod.config.num_heads, mod.config.head_dim)
    v_full = mod.v_proj(x_full).reshape(B, S, mod.config.num_heads, mod.config.head_dim)
    # Run prefill compression once.
    k_full_T = mx.transpose(k_full, (0, 2, 1, 3))
    v_full_T = mx.transpose(v_full, (0, 2, 1, 3))
    k_csa_prefill, v_csa_prefill, _ = _compress_kv(k_full_T, v_full_T, mod.config.m_csa)
    # Decode step-by-step.
    for t in range(S):
        mod.decode_step(x_full[:, t : t + 1], csa, hca)
    # Compare the cache's snapshot to the prefill compression.
    k_csa_stream, _ = csa.snapshot(include_partial=False)
    k_csa_stream_T = mx.transpose(k_csa_stream, (0, 2, 1, 3))
    np.testing.assert_allclose(
        np.array(k_csa_stream_T), np.array(k_csa_prefill),
        atol=1e-5,
    )
