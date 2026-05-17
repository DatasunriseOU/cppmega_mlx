"""Tests for composability extensions: N (engram) / C (concept) blocks,
integrated MTP head, YAML round-trip, and full/gqa attention modes."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from cppmega_mlx.models.hybrid_lm import (
    HybridTinyBlock,
    HybridTinyConfig,
    HybridTinyLM,
)
from cppmega_mlx.nn.attention import AttentionConfig, CausalSelfAttention
from cppmega_mlx.nn.concept import ConceptBlock, ConceptBlockConfig
from cppmega_mlx.nn.engram import EngramBranch
from cppmega_mlx.recipes.pattern import expand_nam_pattern, parse_nam_pattern
from cppmega_mlx.training.mtp import MinimalMTPHead


# ---------------------------------------------------------------------------
# Pattern symbol coverage
# ---------------------------------------------------------------------------


def test_pattern_accepts_engram_and_concept_symbols():
    parsed = parse_nam_pattern("ANEC")
    assert parsed == ("A", "N", "E", "C")

    expanded = expand_nam_pattern("ANEC", 4)
    assert expanded.symbols == ("A", "N", "E", "C")
    assert expanded.engram_layer_numbers == (2,)
    assert expanded.concept_layer_numbers == (4,)
    assert expanded.role_counts["engram"] == 1
    assert expanded.role_counts["concept"] == 1


def test_pattern_still_rejects_upstream_only_symbols():
    for bad in ("AGMR", "ADMR", "A|MR"):
        with pytest.raises(ValueError, match="supported symbols are A, E, M, R, N, C"):
            parse_nam_pattern(bad)


# ---------------------------------------------------------------------------
# Engram block in stack
# ---------------------------------------------------------------------------


def test_hybrid_block_constructs_engram_for_n_symbol():
    cfg = HybridTinyConfig(
        vocab_size=8,
        hidden_size=8,
        pattern="N",
        depth=1,
        num_attention_heads=2,
        max_seq_length=4,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        engram_ngram_orders=(2, 3),
        engram_conv_kernel=0,
    )
    layer = cfg.expanded_pattern().layers[0]
    block = HybridTinyBlock(layer, cfg)

    assert block.backend == "engram"
    assert isinstance(block.block, EngramBranch)
    assert block.engram_block is block.block
    assert block.attention_block is None

    x = mx.random.normal((1, 4, 8))
    delta = block.route_delta(x, mask=None)
    assert delta.shape == x.shape


def test_hybrid_block_constructs_concept_for_c_symbol():
    cfg = HybridTinyConfig(
        vocab_size=8,
        hidden_size=8,
        pattern="C",
        depth=1,
        num_attention_heads=2,
        max_seq_length=4,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        concept_num_concepts=8,
        concept_num_heads=2,
    )
    layer = cfg.expanded_pattern().layers[0]
    block = HybridTinyBlock(layer, cfg)

    assert block.backend == "concept"
    assert isinstance(block.block, ConceptBlock)
    assert block.concept_block is block.block


def test_concept_block_is_identity_at_init():
    config = ConceptBlockConfig(hidden_size=8, num_concepts=4, num_heads=2)
    block = ConceptBlock(config)
    # out_proj is zero-init → block returns the all-zero delta.
    x = mx.random.normal((2, 5, 8))
    delta = block(x)
    assert delta.shape == x.shape
    assert float(mx.max(mx.abs(delta)).item()) == 0.0


def test_hybrid_lm_with_engram_and_concept_forward_runs():
    cfg = HybridTinyConfig(
        vocab_size=16,
        hidden_size=8,
        pattern="ANCE",
        depth=4,
        num_attention_heads=2,
        max_seq_length=4,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
        concept_num_concepts=4,
        concept_num_heads=2,
    )
    model = HybridTinyLM(cfg)
    expanded = cfg.expanded_pattern()
    assert tuple(layer.backend for layer in model.layers) == (
        "attention",
        "engram",
        "concept",
        "moe",
    )
    assert expanded.engram_layer_numbers == (2,)
    assert expanded.concept_layer_numbers == (3,)

    input_ids = mx.array([[0, 1, 2, 3]])
    logits = model(input_ids)
    assert logits.shape == (1, 4, 16)


# ---------------------------------------------------------------------------
# MTP integration
# ---------------------------------------------------------------------------


def _tiny_mtp_cfg(**overrides) -> HybridTinyConfig:
    base = dict(
        vocab_size=16,
        hidden_size=8,
        pattern="A",
        depth=1,
        num_attention_heads=2,
        max_seq_length=4,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
    )
    base.update(overrides)
    return HybridTinyConfig(**base)


def test_mtp_head_is_attached_when_enabled():
    cfg = _tiny_mtp_cfg(mtp_enabled=True, mtp_depth=3, mtp_loss_weight=0.25)
    model = HybridTinyLM(cfg)
    assert model.mtp_head is not None
    assert isinstance(model.mtp_head, MinimalMTPHead)
    assert model.mtp_head.config.depth == 3
    assert math.isclose(model.mtp_head.config.loss_weight, 0.25)
    # Head must share the token embedding and lm_head module instances so
    # gradients flow back through the main model parameters once.
    assert model.mtp_head.token_embedding is model.token_embedding
    assert model.mtp_head.lm_head is model.lm_head


def test_mtp_head_is_absent_when_disabled():
    cfg = _tiny_mtp_cfg(mtp_enabled=False)
    model = HybridTinyLM(cfg)
    assert model.mtp_head is None


def test_mtp_head_forward_produces_one_logits_tensor_per_depth():
    cfg = _tiny_mtp_cfg(mtp_enabled=True, mtp_depth=2)
    model = HybridTinyLM(cfg)
    assert model.mtp_head is not None

    input_ids = mx.array([[0, 1, 2, 3]])
    hidden = model.decoder_hidden_states(input_ids)
    logits_by_depth = model.mtp_head(hidden, input_ids)
    assert len(logits_by_depth) == 2
    for logits in logits_by_depth:
        assert logits.shape == (1, 4, cfg.vocab_size)


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


def test_yaml_round_trip_preserves_config_fields():
    cfg = HybridTinyConfig(
        vocab_size=32,
        hidden_size=16,
        pattern="ANEC",
        depth=4,
        num_attention_heads=4,
        max_seq_length=8,
        attention_mode="full",
        moe_num_experts=4,
        moe_top_k=2,
        moe_expert_hidden_size=8,
        moe_shared_expert_hidden_size=8,
        mamba_head_dim=4,
        m2rnn_k_head_dim=4,
        m2rnn_v_head_dim=4,
        engram_ngram_orders=(2, 3, 4),
        engram_gated=True,
        concept_num_concepts=16,
        concept_num_heads=2,
        mtp_enabled=True,
        mtp_depth=3,
        mhc_enabled=False,
    )
    text = cfg.to_yaml()
    restored = HybridTinyConfig.from_yaml(text)
    assert restored == cfg
    # Sanity-check that the YAML is human-readable and contains key fields.
    assert "attention_mode: full" in text
    assert "pattern: ANEC" in text


def test_from_dict_rejects_unknown_field():
    cfg = HybridTinyConfig(
        vocab_size=16,
        hidden_size=8,
        pattern="A",
        depth=1,
        num_attention_heads=2,
        max_seq_length=4,
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
    )
    data = cfg.to_dict()
    data["bogus_field"] = 42
    with pytest.raises(TypeError):
        HybridTinyConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Attention modes: full / gqa
# ---------------------------------------------------------------------------


def test_attention_config_accepts_full_mode():
    cfg = AttentionConfig(d_model=8, num_q_heads=2, mode="full")
    layer = CausalSelfAttention(cfg)
    x = mx.random.normal((1, 4, 8))
    out = layer(x)
    assert out.shape == x.shape


def test_attention_config_full_rejects_mismatched_kv_heads():
    with pytest.raises(ValueError, match="num_kv_heads to equal num_q_heads"):
        AttentionConfig(d_model=8, num_q_heads=4, num_kv_heads=2, mode="full")


def test_attention_config_gqa_requires_smaller_kv_heads():
    # gqa with num_kv_heads < num_q_heads is the legitimate case.
    cfg = AttentionConfig(d_model=8, num_q_heads=4, num_kv_heads=2, mode="gqa")
    assert cfg.is_gqa
    layer = CausalSelfAttention(cfg)
    x = mx.random.normal((1, 4, 8))
    out = layer(x)
    assert out.shape == x.shape


def test_attention_config_gqa_rejects_missing_kv_heads():
    with pytest.raises(ValueError, match="strictly less than num_q_heads"):
        AttentionConfig(d_model=8, num_q_heads=4, mode="gqa")
    with pytest.raises(ValueError, match="strictly less than num_q_heads"):
        AttentionConfig(d_model=8, num_q_heads=4, num_kv_heads=4, mode="gqa")


def test_hybrid_tiny_config_propagates_attention_mode_to_layers():
    cfg = HybridTinyConfig(
        vocab_size=16,
        hidden_size=8,
        pattern="A",
        depth=1,
        num_attention_heads=4,
        num_attention_kv_heads=2,
        max_seq_length=4,
        attention_mode="gqa",
        moe_num_experts=2,
        moe_top_k=1,
        moe_expert_hidden_size=4,
        moe_shared_expert_hidden_size=None,
        m2rnn_k_head_dim=2,
        m2rnn_v_head_dim=2,
    )
    model = HybridTinyLM(cfg)
    attn = model.layers[0].block
    assert isinstance(attn, CausalSelfAttention)
    assert attn.config.mode == "gqa"
    assert attn.config.is_gqa


def test_hybrid_tiny_config_rejects_invalid_attention_mode():
    with pytest.raises(ValueError, match="attention_mode must be one of"):
        HybridTinyConfig(
            vocab_size=16,
            hidden_size=8,
            pattern="A",
            depth=1,
            num_attention_heads=2,
            max_seq_length=4,
            attention_mode="bogus",  # type: ignore[arg-type]
        )
