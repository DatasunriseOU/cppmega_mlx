"""Smoke + integration tests for mlx-lm PR bricks (re-exports).

Covers:
  - direct import from our patched mlx-lm (cppmega-integration branch)
  - each wrapper instantiates and runs end-to-end
  - registration in BLOCK_BUILDERS so UnifiedSuperblock can compose them
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.nn.mlx_lm_bricks import (
    BailingLinearAttnBlock, BailingLinearConfig,
    BailingMLABlock, BailingMLAConfig,
    BailingMoEBlock, BailingMoEConfig,
    DSv4AttentionBlock, DSv4AttentionConfig,
    Mistral4MLABlock, Mistral4MLAConfig,
)
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


_H = 256
_X = mx.random.normal((1, 8, _H))


def test_all_5_bricks_registered_in_block_builders():
    expected = {
        "mistral4_mla", "dsv4_attention",
        "bailing_linear", "bailing_mla", "bailing_moe",
    }
    assert expected.issubset(set(BLOCK_BUILDERS.keys()))


# ----- Mistral Small 4 MLA (PR #1037) -----


def test_mistral4_mla_imports_from_mlx_lm():
    from mlx_lm.models.mistral4 import MLAAttention as _Upstream
    cfg = Mistral4MLAConfig(hidden_size=_H, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=64, q_lora_rank=128, kv_lora_rank=64,
                             qk_rope_head_dim=32, qk_nope_head_dim=64)
    block = Mistral4MLABlock(cfg)
    assert isinstance(block.inner, _Upstream)


def test_mistral4_mla_forward_shape():
    cfg = Mistral4MLAConfig(hidden_size=_H, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=64, q_lora_rank=128, kv_lora_rank=64,
                             qk_rope_head_dim=32, qk_nope_head_dim=64)
    block = Mistral4MLABlock(cfg)
    y = block(_X)
    mx.eval(y)
    assert y.shape == _X.shape


# ----- DeepSeek-V4 attention (PR #1201) -----


def test_dsv4_attention_imports_from_mlx_lm():
    from mlx_lm.models.deepseek_v4 import Attention as _Upstream
    cfg = DSv4AttentionConfig(
        hidden_size=_H, num_attention_heads=4, num_key_value_heads=4,
        head_dim=64, q_lora_rank=128, o_lora_rank=64,
        qk_rope_head_dim=32, o_groups=1, index_n_heads=2, index_head_dim=64, index_topk=4,
    )
    block = DSv4AttentionBlock(cfg)
    assert isinstance(block.inner, _Upstream)


def test_dsv4_attention_forward_shape():
    cfg = DSv4AttentionConfig(
        hidden_size=_H, num_attention_heads=4, num_key_value_heads=4,
        head_dim=64, q_lora_rank=128, o_lora_rank=64,
        qk_rope_head_dim=32, o_groups=1, index_n_heads=2, index_head_dim=64, index_topk=4,
    )
    block = DSv4AttentionBlock(cfg)
    y = block(_X)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    assert y.shape == _X.shape


# ----- Bailing / Ling-2.6 (PR #1227) -----


def test_bailing_linear_imports_from_mlx_lm():
    from mlx_lm.models.bailing_hybrid import LinearAttention as _Upstream
    cfg = BailingLinearConfig(hidden_size=_H, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=64)
    block = BailingLinearAttnBlock(cfg)
    assert isinstance(block.inner, _Upstream)


def test_bailing_linear_forward_shape():
    cfg = BailingLinearConfig(hidden_size=_H, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=64)
    block = BailingLinearAttnBlock(cfg)
    y = block(_X)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    assert y.shape == _X.shape


def test_bailing_mla_forward_shape():
    cfg = BailingMLAConfig(hidden_size=_H, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=64, kv_lora_rank=64,
                            qk_rope_head_dim=32, qk_nope_head_dim=64, v_head_dim=64)
    block = BailingMLABlock(cfg)
    y = block(_X)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    assert y.shape == _X.shape


def test_bailing_moe_forward_shape():
    cfg = BailingMoEConfig(hidden_size=_H, num_experts=4, num_experts_per_tok=2,
                            num_shared_experts=1, moe_intermediate_size=128,
                            n_group=1, topk_group=1)
    block = BailingMoEBlock(cfg)
    y = block(_X)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    assert y.shape == _X.shape


# ----- UnifiedSuperblock integration -----


@pytest.mark.parametrize("kind,extra", [
    ("mistral4_mla", dict(num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                          q_lora_rank=128, kv_lora_rank=64,
                          qk_rope_head_dim=32, qk_nope_head_dim=64)),
    ("dsv4_attention", dict(num_attention_heads=4, num_key_value_heads=4, head_dim=64,
                             q_lora_rank=128, o_lora_rank=64, qk_rope_head_dim=32,
                             o_groups=1, index_n_heads=2, index_head_dim=64, index_topk=4)),
    ("bailing_linear", dict(num_attention_heads=4, num_key_value_heads=2, head_dim=64)),
    ("bailing_mla", dict(num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                          kv_lora_rank=64, qk_rope_head_dim=32, qk_nope_head_dim=64,
                          v_head_dim=64)),
    ("bailing_moe", dict(num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                          moe_intermediate_size=128, n_group=1, topk_group=1)),
])
def test_unified_superblock_builder_round_trip(kind, extra):
    block = BLOCK_BUILDERS[kind](hidden_size=_H, params=extra)
    y = block(_X)
    if isinstance(y, tuple):
        y = y[0]
    mx.eval(y)
    assert y.shape == _X.shape, f"{kind}: {y.shape} != {_X.shape}"
    assert not bool(mx.any(mx.isnan(y)).item()), f"{kind}: NaN in output"
