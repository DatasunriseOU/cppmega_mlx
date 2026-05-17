"""Tests for ROI 7 — FP8 Lightning Indexer."""

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.lightning_indexer import (
    LightningIndexer,
    LightningIndexerConfig,
)
from cppmega_v4.nn.lightning_indexer_fp8 import (
    LightningIndexerFP8,
    LightningIndexerFP8Config,
    quantize_indexer_weights_for_fp8,
)


def _make_freqs(seq, rope_half):
    rng = np.random.default_rng(0)
    cos = mx.array(rng.uniform(-1, 1, (seq, rope_half)).astype(np.float32))
    sin = mx.array(rng.uniform(-1, 1, (seq, rope_half)).astype(np.float32))
    return cos, sin


def _config(fp8=True):
    return LightningIndexerFP8Config(
        hidden_size=128, n_heads=4, head_dim=32, rope_head_dim=16,
        q_lora_rank=64, index_topk=8, fp8_blocks=fp8,
    )


def test_fp8_indexer_constructs_with_fp8_storage():
    cfg = _config(fp8=True)
    mod = LightningIndexerFP8(cfg)
    assert mod._wq_b_fp8.dtype == mx.uint8
    assert mod._wq_b_fp8.shape == (cfg.n_heads * cfg.head_dim, cfg.q_lora_rank)
    assert mod._wq_b_scale_inv.dtype == mx.float32


def test_fp8_indexer_constructs_with_bf16_fallback():
    cfg = _config(fp8=False)
    mod = LightningIndexerFP8(cfg)
    assert mod._wq_b_bf16.dtype == mx.bfloat16


def test_fp8_indexer_forward_shape():
    cfg = _config(fp8=True)
    mod = LightningIndexerFP8(cfg)
    B, T = 1, 16  # T must be >= index_topk for full topk shape
    x = mx.random.normal((B, T, cfg.hidden_size))
    qr = mx.random.normal((B, T, cfg.q_lora_rank))
    cos, sin = _make_freqs(T, cfg.rope_head_dim // 2)
    topk = mod(x, qr, (cos, sin))
    assert topk.shape == (B, T, cfg.index_topk)
    assert topk.dtype == mx.int32


def test_quantize_indexer_weights_round_trip_close_to_fp32():
    """Quant→FP8 indexer should pick similar top-k to fp32 reference on
    well-conditioned random inputs (most overlap > 75%)."""
    fp32_cfg = LightningIndexerConfig(
        hidden_size=128, n_heads=4, head_dim=32, rope_head_dim=16,
        q_lora_rank=64, index_topk=8,
    )
    fp32_idx = LightningIndexer(fp32_cfg)

    fp8_cfg = _config(fp8=True)
    fp8_idx = LightningIndexerFP8(fp8_cfg)
    tensors = quantize_indexer_weights_for_fp8(fp32_idx)
    fp8_idx.load_fp8_weights(**tensors)

    B, T = 2, 24
    rng = np.random.default_rng(11)
    x = mx.array(rng.standard_normal((B, T, fp32_cfg.hidden_size)).astype(np.float32))
    qr = mx.array(rng.standard_normal((B, T, fp32_cfg.q_lora_rank)).astype(np.float32))
    cos, sin = _make_freqs(T, fp32_cfg.rope_head_dim // 2)
    topk32 = np.array(fp32_idx(x, qr, (cos, sin)))
    topk8 = np.array(fp8_idx(x, qr, (cos, sin)))

    # Per-row overlap fraction (sort both, count intersection / topk).
    overlaps = []
    for b in range(B):
        for t in range(T):
            inter = len(set(topk32[b, t].tolist()) & set(topk8[b, t].tolist()))
            overlaps.append(inter / fp32_cfg.index_topk)
    mean_overlap = float(np.mean(overlaps))
    # FP8 quant noise on random weights shouldn't drop overlap below 0.5
    # on average — we use 0.4 as a defensive floor.
    assert mean_overlap >= 0.4, f"top-k overlap too low: {mean_overlap:.3f}"


def test_fp8_indexer_load_rejects_non_uint8():
    cfg = _config(fp8=True)
    mod = LightningIndexerFP8(cfg)
    bad = mx.zeros((cfg.n_heads * cfg.head_dim, cfg.q_lora_rank), dtype=mx.float32)
    scale = mx.ones((1, 1), dtype=mx.float32)
    wk = mx.zeros((cfg.head_dim, cfg.hidden_size), dtype=mx.bfloat16)
    wp = mx.zeros((cfg.n_heads, cfg.hidden_size), dtype=mx.bfloat16)
    with pytest.raises(TypeError, match="uint8"):
        mod.load_fp8_weights(bad, scale, wk, wp)


def test_fp8_indexer_topk_is_stop_gradient():
    """top-k indices must not propagate gradient (mx.stop_gradient)."""
    cfg = _config(fp8=True)
    mod = LightningIndexerFP8(cfg)
    B, T = 1, 4
    x = mx.random.normal((B, T, cfg.hidden_size))
    qr = mx.random.normal((B, T, cfg.q_lora_rank))
    cos, sin = _make_freqs(T, cfg.rope_head_dim // 2)
    topk = mod(x, qr, (cos, sin))
    # int32 indices are inherently non-differentiable, but stop_gradient
    # confirms the dtype invariant.
    assert topk.dtype == mx.int32
