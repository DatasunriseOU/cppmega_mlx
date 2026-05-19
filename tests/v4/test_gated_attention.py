"""Tests for Gated Attention block (mlx-lm Qwen3NextAttention re-export)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from cppmega_v4.nn.gated_attention import GatedAttentionBlock, GatedAttentionConfig
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


def test_block_registers_in_block_builders():
    assert "gated_attention" in BLOCK_BUILDERS


def test_direct_import_from_mlx_lm():
    """The wrapper must use mlx-lm's actual Qwen3NextAttention, not a fork."""
    from mlx_lm.models.qwen3_next import Qwen3NextAttention as _Upstream
    cfg = GatedAttentionConfig(hidden_size=128, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=32)
    block = GatedAttentionBlock(cfg)
    assert isinstance(block.attn, _Upstream)


def test_forward_shape_preserved():
    cfg = GatedAttentionConfig(
        hidden_size=256, num_attention_heads=8, num_key_value_heads=2, head_dim=32,
    )
    block = GatedAttentionBlock(cfg)
    x = mx.random.normal((2, 16, 256))
    y = block(x)
    mx.eval(y)
    assert y.shape == x.shape


def test_qproj_is_2x_for_gate_split():
    """q_proj outputs 2x size so half can be split off as the sigmoid gate."""
    cfg = GatedAttentionConfig(
        hidden_size=128, num_attention_heads=4, num_key_value_heads=1, head_dim=32,
    )
    block = GatedAttentionBlock(cfg)
    # weight shape is [out, in] in mlx
    assert block.attn.q_proj.weight.shape == (4 * 32 * 2, 128)


def test_asymmetric_gqa():
    """Gated Attention uses extreme GQA (e.g. 6:1 in Qwen 3.6)."""
    cfg = GatedAttentionConfig(
        hidden_size=512, num_attention_heads=24, num_key_value_heads=4, head_dim=128,
    )
    block = GatedAttentionBlock(cfg)
    assert block.attn.q_proj.weight.shape == (24 * 128 * 2, 512)
    assert block.attn.k_proj.weight.shape == (4 * 128, 512)
    assert block.attn.v_proj.weight.shape == (4 * 128, 512)


def test_q_and_k_norm_present():
    cfg = GatedAttentionConfig(hidden_size=128, num_attention_heads=4,
                                num_key_value_heads=1, head_dim=32)
    block = GatedAttentionBlock(cfg)
    assert hasattr(block.attn, "q_norm")
    assert hasattr(block.attn, "k_norm")


def test_unified_superblock_builder_round_trip():
    """Building via BLOCK_BUILDERS produces a working module."""
    builder = BLOCK_BUILDERS["gated_attention"]
    block = builder(
        hidden_size=128,
        params={
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
        },
    )
    x = mx.random.normal((1, 8, 128))
    y = block(x)
    mx.eval(y)
    assert y.shape == x.shape
    assert not bool(mx.any(mx.isnan(y)).item())


def test_qwen36_attention_slot_shape():
    """Qwen 3.6-27B Gated Attention slot: 24 Q / 4 KV / head_dim=128 / hidden=5120.

    Use shrunk dims so the test is fast but ratios match the production
    config (8:1 GQA, 2x q_proj for the gate split).
    """
    cfg = GatedAttentionConfig(
        hidden_size=512,           # /10 of 5120
        num_attention_heads=24,
        num_key_value_heads=4,     # 6:1 GQA ratio
        head_dim=64,                # /2 of 128
        partial_rotary_factor=0.25,
    )
    block = GatedAttentionBlock(cfg)
    x = mx.random.normal((1, 32, 512))
    y = block(x)
    mx.eval(y)
    assert y.shape == x.shape
