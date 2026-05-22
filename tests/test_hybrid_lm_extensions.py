"""Tests for composability extensions: N (engram) / C (concept) blocks,
integrated MTP head, YAML round-trip, and full/gqa attention modes."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from cppmega_mlx.models.hybrid_lm import (
    HybridTinyBlock,
    HybridTinyConfig,
    HybridTinyLM,
    PathCActivationBufferCapture,
)
from cppmega_mlx.nn.attention import AttentionConfig, CausalSelfAttention
from cppmega_mlx.nn.concept import ConceptBlock, ConceptBlockConfig
from cppmega_mlx.nn.engram import EngramBranch
from cppmega_mlx.recipes.model_factory import build_local_gb10_quarter_tiny_smoke_model
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


def test_path_c_activation_probe_is_opt_in_and_stores_references():
    cfg = _tiny_mtp_cfg(pattern="MR", depth=2, max_seq_length=4)
    model = HybridTinyLM(cfg)
    capture = PathCActivationBufferCapture(aliases={"layer_0_m_hidden": "hidden"})
    input_ids = mx.array([[0, 1, 2, 3]])

    model.decoder_hidden_states(input_ids)

    assert capture.events == []
    assert all(
        not hasattr(layer, "_path_c_activation_probe_callback")
        for layer in model.layers
    )

    assert model.attach_path_c_activation_probe(capture) == 2
    model.decoder_hidden_states(input_ids)

    expected = {
        "layer_0_m_hidden",
        "layer_0_m_residual_norm_hidden",
        "layer_0_m_mamba3_h0",
        "layer_0_m_state_in",
        "layer_0_m_state",
        "layer_0_m_delta",
        "layer_0_m_hidden_after",
        "layer_1_r_hidden",
        "layer_1_r_residual_norm_hidden",
        "layer_1_r_m2rnn_h0",
        "layer_1_r_m2rnn_conv_state",
        "layer_1_r_delta",
        "layer_1_r_hidden_after",
    }
    assert expected.issubset(capture.buffers)
    for event in capture.events:
        assert "tensor" not in event
        assert event["tensor_dtype"] is not None
        assert isinstance(event["tensor_shape"], tuple)
        for name in event["logical_names"]:
            assert isinstance(capture.buffers[name], mx.array)

    model.detach_path_c_activation_probe()
    assert all(
        not hasattr(layer, "_path_c_activation_probe_callback")
        for layer in model.layers
    )


def test_path_c_activation_probe_captures_vjp_cotangents_without_copy():
    cfg = _tiny_mtp_cfg(pattern="MR", depth=2, max_seq_length=4)
    model = HybridTinyLM(cfg)
    capture = PathCActivationBufferCapture(aliases={"layer_0_m_hidden": "hidden"})
    input_ids = mx.array([[0, 1, 2, 3]], dtype=mx.int32)

    model.attach_path_c_activation_probe(capture)

    def loss_fn(model_arg: HybridTinyLM, tokens: mx.array):
        hidden = model_arg.decoder_hidden_states(tokens)
        return mx.sum(hidden), mx.array(hidden.size, dtype=mx.int32)

    (loss, ntokens), grads = nn.value_and_grad(model, loss_fn)(model, input_ids)
    mx.eval(loss, ntokens, grads)

    expected = {
        "layer_0_m_hidden_grad",
        "layer_0_m_residual_norm_hidden_grad",
        "layer_0_m_delta_grad",
        "layer_0_m_hidden_after_grad",
        "layer_1_r_hidden_grad",
        "layer_1_r_residual_norm_hidden_grad",
        "layer_1_r_delta_grad",
        "layer_1_r_hidden_after_grad",
    }
    assert expected.issubset(capture.buffers)
    for name in expected:
        grad = capture.buffers[name]
        assert grad.shape == (1, 4, cfg.hidden_size)
        assert grad.dtype == mx.float32

    gradient_events = [
        event
        for event in capture.events
        if event.get("phase") == "value_and_grad"
    ]
    assert {event["logical_names"][0] for event in gradient_events}.issuperset(
        expected
    )
    for event in gradient_events:
        assert "tensor" not in event
        assert str(event["tensor_dtype"]).endswith("float32")
        for name in event["logical_names"]:
            assert tuple(int(dim) for dim in capture.buffers[name].shape) == event[
                "tensor_shape"
            ]
    assert capture.buffers["hidden_grad"] is capture.buffers[
        "layer_0_m_hidden_grad"
    ]


def test_path_c_activation_capture_uses_profile_brick_aliases_without_copy():
    model = build_local_gb10_quarter_tiny_smoke_model()
    block = model.layers[10]
    hidden = mx.random.normal((1, 4, model.config.hidden_size))
    capture = PathCActivationBufferCapture(
        aliases={"local_gb10_quarter_brick_10_M_hidden": "hidden"},
        owner_name="local_gb10_quarter.forward_activation_capture",
    )

    assert capture.owner_name == "local_gb10_quarter.forward_activation_capture"
    assert block.path_c_profile_brick_name == "local_gb10_quarter_brick_10_M"
    model.attach_path_c_activation_probe(capture)
    block(hidden, mask=None)

    assert capture.buffers["local_gb10_quarter_brick_10_M_hidden"] is hidden
    assert capture.buffers["hidden"] is hidden


def test_path_c_activation_probe_captures_sparse_mla_prepared_buffers():
    model = build_local_gb10_quarter_tiny_smoke_model()
    block = model.layers[12]
    capture = PathCActivationBufferCapture(
        owner_name="local_gb10_quarter.path_c_forward_activation_capture",
    )
    model.attach_path_c_activation_probe(capture)

    assert block.attention_block is not None
    prepared = block.attention_block.prepare_sparse_mla_fp8(
        mx.zeros((1, 4, model.config.hidden_size), dtype=mx.float32),
        mask="causal",
    )
    mx.eval(prepared.q_fp8, prepared.q_scale, prepared.kv_fp8, prepared.kv_scale)

    assert (
        capture.buffers["local_gb10_quarter_brick_12_A_qkv_projection_q_fp8"]
        is prepared.q_fp8
    )
    assert (
        capture.buffers["local_gb10_quarter_brick_12_A_qkv_projection_q_scale"]
        is prepared.q_scale
    )
    assert (
        capture.buffers["local_gb10_quarter_brick_12_A_qkv_projection_kv_fp8"]
        is prepared.kv_fp8
    )
    assert (
        capture.buffers["local_gb10_quarter_brick_12_A_qkv_projection_kv_scale"]
        is prepared.kv_scale
    )
    assert (
        capture.buffers["local_gb10_quarter_brick_12_A_qkv_projection_indices"]
        is prepared.indices
    )

    model.detach_path_c_activation_probe()
    prepared_after_detach = block.attention_block.prepare_sparse_mla_fp8(
        mx.zeros((1, 4, model.config.hidden_size), dtype=mx.float32),
        mask="causal",
    )
    mx.eval(prepared_after_detach.q_fp8)
    assert capture.buffers[
        "local_gb10_quarter_brick_12_A_qkv_projection_q_fp8"
    ] is prepared.q_fp8


def test_path_c_activation_probe_maps_sparse_mla_apply_runtime_buffers():
    model = build_local_gb10_quarter_tiny_smoke_model()
    block = model.layers[12]
    capture = PathCActivationBufferCapture(
        owner_name="local_gb10_quarter.path_c_forward_activation_capture",
    )
    model.attach_path_c_activation_probe(capture)

    assert block.attention_block is not None
    probe = block.attention_block._path_c_sparse_mla_apply_probe()
    assert callable(probe)
    buffers = {
        "sparse_mla_sm_scale": mx.array([0.5], dtype=mx.float32),
        "sparse_mla_sinks": mx.zeros((4,), dtype=mx.float32),
        "sparse_mla_has_sinks": mx.array([0], dtype=mx.int32),
        "lse": mx.zeros((16,), dtype=mx.float32),
        "out": mx.zeros((1, 4, model.config.hidden_size), dtype=mx.float32),
    }

    for name, tensor in buffers.items():
        probe({"name": name, "tensor": tensor})

    for name, tensor in buffers.items():
        assert (
            capture.buffers[
                f"local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_{name}"
            ]
            is tensor
        )


def test_path_c_parameter_gradient_aliases_use_direct_and_profile_names():
    model = build_local_gb10_quarter_tiny_smoke_model()

    aliases = model.path_c_parameter_gradient_aliases()

    assert aliases["layers.10.block.in_proj.weight_grad"] == (
        "layer_10_m_mamba3_in_proj_weight_grad",
        "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight_grad",
    )
    assert aliases["layers.11.norm.weight_grad"] == (
        "layer_10_m_residual_norm_weight_grad",
        "local_gb10_quarter_brick_10_M_residual_norm_weight_grad",
    )
    assert aliases["layers.12.norm.weight_grad"] == (
        "layer_11_r_residual_norm_weight_grad",
        "local_gb10_quarter_brick_11_R_residual_norm_weight_grad",
    )
    assert aliases["layers.11.block.state_weight_grad"] == (
        "layer_11_r_m2rnn_state_weight_grad",
        "local_gb10_quarter_brick_11_R_m2rnn_state_weight_grad",
    )
    assert aliases["layers.12.block.sparse_kv_proj.weight_grad"] == (
        "layer_12_a_qkv_projection_attention_sparse_kv_proj_weight_grad",
        "local_gb10_quarter_brick_12_A_qkv_projection_attention_sparse_kv_proj_weight_grad",
    )
    assert aliases["norm.weight_grad"] == ("final_norm_weight_grad",)
    assert aliases["lm_head.weight_grad"] == ("lm_head_weight_grad",)
    assert "layers.12.block.q_proj.bias_grad" not in aliases
    assert "layers.12.block.sparse_kv_proj.bias_grad" not in aliases


def test_path_c_direct_logical_owner_uses_model_parameter_references_only():
    model = build_local_gb10_quarter_tiny_smoke_model()
    params = dict(tree_flatten(model.trainable_parameters()))

    owner = model.make_path_c_direct_fusion_chain_logical_buffer_owner()

    assert owner.owner_name == "local_gb10_quarter.path_c_model_parameter_buffers"
    assert owner.hidden_packing_performed is False
    assert owner.no_hidden_allocation_policy is True
    assert owner.buffers[
        "local_gb10_quarter_brick_10_M_mamba3_in_proj_weight"
    ] is params["layers.10.block.in_proj.weight"]
    assert owner.buffers[
        "local_gb10_quarter_brick_10_M_residual_norm_weight"
    ] is params["layers.11.norm.weight"]
    assert owner.buffers[
        "local_gb10_quarter_brick_11_R_residual_norm_weight"
    ] is params["layers.12.norm.weight"]
    assert owner.buffers[
        "local_gb10_quarter_brick_12_A_sparse_mla_fp8_apply_attention_out_proj_weight"
    ] is params["layers.12.block.out_proj.weight"]
    assert all("attention_out_proj_bias" not in name for name in owner.buffers)
    assert all(
        "sparse_mla_fp8_apply_sparse_mla_sm_scale" not in name
        for name in owner.buffers
    )


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
