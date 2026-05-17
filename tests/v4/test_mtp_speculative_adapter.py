"""Tests for SequentialMTPHead → mlx-lm PR #990 MTPModule adapter."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from cppmega_v4.nn._external._mlx_lm_mtp_module_vendored import MTPModule
from cppmega_v4.nn.mtp_speculative_adapter import SequentialMTPHeadAsMTPModule
from cppmega_v4.nn.mtp_v4 import SequentialMTPDepthBlock, SequentialMTPHead

pytest.importorskip("mlx_lm", reason="mlx_lm not available")
from cppmega_mlx.training.mtp import MTPLossConfig  # noqa: E402


def _make_emb(vocab=32, hidden=16) -> nn.Embedding:
    return nn.Embedding(vocab, hidden)


def test_vendored_mtp_module_factory_shape_contract():
    """Vendored MTPModule must produce (B, N, H) from (B, N, H) + (B, N)."""
    B, N, H = 1, 4, 16
    vocab = 32
    embed = _make_emb(vocab=vocab, hidden=H)

    def decoder_layer():
        # Minimal pass-through layer — just RMSNorm. The MTPModule API only
        # requires layer(x, mask, cache) → x with same shape.
        norm = nn.RMSNorm(H)

        class PassThrough(nn.Module):
            def __call__(self, x, mask=None, cache=None):
                return norm(x)
        return PassThrough()

    mod = MTPModule(
        hidden_size=H, num_layers=1, rms_norm_eps=1e-5,
        decoder_layer_factory=decoder_layer,
    )
    hidden = mx.random.normal((B, N, H))
    next_ids = mx.array(np.random.randint(0, vocab, (B, N)).astype(np.int32))
    out = mod(hidden, next_ids, embed)
    assert out.shape == (B, N, H)


def test_adapter_from_fresh_matches_shape_contract():
    """Adapter built fresh must accept (hidden, next_ids, embed) and return (B,N,H)."""
    B, N, H = 1, 4, 16
    vocab = 32
    embed = _make_emb(vocab=vocab, hidden=H)
    mod = SequentialMTPHeadAsMTPModule.fresh(H)
    hidden = mx.random.normal((B, N, H))
    next_ids = mx.array(np.random.randint(0, vocab, (B, N)).astype(np.int32))
    out = mod(hidden, next_ids, embed)
    assert out.shape == (B, N, H)
    assert not bool(mx.any(mx.isnan(out)).item())


def test_adapter_from_training_head_reuses_weights():
    """Building from an existing SequentialMTPHead must share parameters with depth-0."""
    B, N, H = 1, 4, 16
    vocab = 32
    embed = _make_emb(vocab=vocab, hidden=H)
    head_lm = nn.Linear(H, vocab, bias=False)
    head = SequentialMTPHead(
        embed, head_lm, config=MTPLossConfig(depth=2),
    )
    adapter = SequentialMTPHeadAsMTPModule.from_head(head, depth_index=0)
    # adapter.depth_block should be the same object as head.depth_blocks[0]
    assert adapter.depth_block is head.depth_blocks[0]
    # And calling it should produce a valid fused hidden state.
    hidden = mx.random.normal((B, N, H))
    next_ids = mx.array(np.random.randint(0, vocab, (B, N)).astype(np.int32))
    out = adapter(hidden, next_ids, embed)
    assert out.shape == (B, N, H)


def test_adapter_rejects_zero_depth_head():
    H = 16
    embed = _make_emb(hidden=H)
    head_lm = nn.Linear(H, 32, bias=False)
    head = SequentialMTPHead(embed, head_lm, config=MTPLossConfig(depth=0))
    with pytest.raises(ValueError, match="depth=0"):
        SequentialMTPHeadAsMTPModule.from_head(head)


def test_adapter_rejects_out_of_range_depth_index():
    H = 16
    embed = _make_emb(hidden=H)
    head_lm = nn.Linear(H, 32, bias=False)
    head = SequentialMTPHead(embed, head_lm, config=MTPLossConfig(depth=2))
    with pytest.raises(ValueError, match="out of range"):
        SequentialMTPHeadAsMTPModule.from_head(head, depth_index=2)


def test_adapter_call_is_deterministic():
    """Same inputs → same outputs (no hidden RNG)."""
    B, N, H = 1, 3, 8
    vocab = 16
    embed = _make_emb(vocab=vocab, hidden=H)
    mod = SequentialMTPHeadAsMTPModule.fresh(H)
    hidden = mx.random.normal((B, N, H))
    next_ids = mx.array(np.random.randint(0, vocab, (B, N)).astype(np.int32))
    out_a = mod(hidden, next_ids, embed)
    out_b = mod(hidden, next_ids, embed)
    np.testing.assert_array_equal(np.array(out_a), np.array(out_b))
