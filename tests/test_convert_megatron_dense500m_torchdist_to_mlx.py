from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_megatron_dense500m_torchdist_to_mlx.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_megatron_dense500m", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeDType:
    pass


class _FakeProperties:
    dtype = _FakeDType()


class _FakeTensorMeta:
    properties = _FakeProperties()

    def __init__(self, size: tuple[int, ...]):
        self.size = size


class _FakeMetadata:
    def __init__(self, sizes: dict[str, tuple[int, ...]]):
        self.state_dict_metadata = {
            key: _FakeTensorMeta(size) for key, size in sizes.items()
        }


class _FakeTarget:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def _toy_contract(mod, cfg, *, source_keys: set[str] | None = None):
    plan = mod.build_key_plan(cfg, source_keys=source_keys)
    source_sizes: dict[str, tuple[int, ...]] = {}
    targets: dict[str, _FakeTarget] = {
        "position_embedding.weight": _FakeTarget((cfg.max_seq_length, cfg.hidden_size)),
        "platform_embedding.embedding.weight": _FakeTarget((3, cfg.hidden_size)),
        "layers.0.attention.rope_inv_freq": _FakeTarget((1,)),
    }
    for item in plan:
        key = item.source_key
        if key.endswith("linear_qkv.weight"):
            shape = (cfg.qkv_dim, cfg.hidden_size)
        elif key.endswith("linear_fc1.weight"):
            shape = (2 * cfg.ffn_hidden_size, cfg.hidden_size)
        elif key.endswith("linear_fc2.weight"):
            shape = (cfg.hidden_size, cfg.ffn_hidden_size)
        elif key == "embedding.word_embeddings.weight":
            shape = (cfg.vocab_size, cfg.hidden_size)
        elif key.endswith("linear_proj.weight"):
            shape = (cfg.hidden_size, cfg.q_proj_dim)
        elif key.endswith("cppmega_ngram_hash.out_proj.weight"):
            shape = (
                cfg.hidden_size,
                cfg.ngram_hash_heads * cfg.ngram_hash_embed_dim,
            )
        elif key.endswith("cppmega_ngram_hash.unified_table.weight"):
            shape = (7_998_862, cfg.ngram_hash_embed_dim)
        elif key.endswith("cppmega_ngram_hash.hash_mults"):
            shape = (cfg.ngram_hash_heads * 2, 3)
        elif key.endswith("cppmega_ngram_hash.order_mask"):
            shape = (3, cfg.ngram_hash_heads * 2)
        elif any(
            key.endswith(suffix)
            for suffix in (
                "table_offsets",
                "table_sizes_t",
                "hash_bias",
                "order_for_table",
            )
        ):
            shape = (cfg.ngram_hash_heads * 2,)
        elif key.endswith("cppmega_structure.stacked_emb.weight"):
            shape = (25, cfg.structure_bottleneck_dim)
        elif key.endswith("cppmega_structure.up_proj.weight"):
            shape = (cfg.hidden_size, cfg.structure_bottleneck_dim)
        elif key.endswith("cppmega_structure.component_scales"):
            shape = (2,)
        elif key.endswith("cppmega_domain.stacked_emb.weight"):
            shape = (
                cfg.domain_num_domains
                + cfg.domain_num_roles
                + cfg.domain_num_confidences,
                cfg.domain_bottleneck_dim,
            )
        elif key.endswith("cppmega_domain.up_proj.weight"):
            shape = (cfg.hidden_size, cfg.domain_bottleneck_dim)
        elif key.endswith("cppmega_domain.component_scales"):
            shape = (3,)
        else:
            shape = (cfg.hidden_size,)
        source_sizes[key] = shape
        target_shape = shape
        if item.source_slice is not None:
            lo, hi = item.source_slice
            target_shape = (hi - lo, *shape[1:])
        if item.source_rows is not None:
            target_shape = (len(item.source_rows), *shape[1:])
        targets[item.target_key] = _FakeTarget(target_shape)
    return source_sizes, targets


def test_key_plan_folds_megatron_af_layers_into_dense_blocks():
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        hidden_size=8,
        depth=2,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )

    plan = mod.build_key_plan(cfg)

    assert mod.KeyPlan(
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
        "layers.0.attn_norm.weight",
    ) in plan
    assert mod.KeyPlan(
        "decoder.layers.1.mlp.linear_fc1.layer_norm_weight",
        "layers.0.ffn_norm.weight",
    ) in plan
    assert mod.KeyPlan(
        "decoder.layers.2.self_attention.linear_qkv.layer_norm_weight",
        "layers.1.attn_norm.weight",
    ) in plan
    assert mod.KeyPlan(
        "decoder.layers.3.mlp.linear_fc1.layer_norm_weight",
        "layers.1.ffn_norm.weight",
    ) in plan


def test_key_plan_splits_gqa_qkv_and_swiglu_fc1_rows():
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )

    by_target = {item.target_key: item for item in mod.build_key_plan(cfg)}

    assert by_target["layers.0.attention.q_proj.weight"].source_rows == (
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
    )
    assert by_target["layers.0.attention.k_proj.weight"].source_rows == (
        4,
        5,
        12,
        13,
    )
    assert by_target["layers.0.attention.v_proj.weight"].source_rows == (
        6,
        7,
        14,
        15,
    )
    assert by_target["layers.0.ffn.gate_proj.weight"].source_slice == (0, 12)
    assert by_target["layers.0.ffn.up_proj.weight"].source_slice == (12, 24)


def test_validate_plan_rejects_unmapped_target_tensor():
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )
    source_sizes: dict[str, tuple[int, ...]] = {}
    target_arrays: dict[str, _FakeTarget] = {
        "position_embedding.weight": _FakeTarget((4, 8)),
        "platform_embedding.embedding.weight": _FakeTarget((3, 8)),
        "layers.0.attention.rope_inv_freq": _FakeTarget((1,)),
        "unexpected.weight": _FakeTarget((1,)),
    }
    for item in mod.build_key_plan(cfg):
        if item.source_key.endswith("linear_qkv.weight"):
            source_sizes[item.source_key] = (16, 8)
        elif item.source_key.endswith("linear_fc1.weight"):
            source_sizes[item.source_key] = (24, 8)
        elif item.source_key.endswith("linear_fc2.weight"):
            source_sizes[item.source_key] = (8, 12)
        else:
            shape = (8,)
            if item.source_key == "embedding.word_embeddings.weight":
                shape = (65_536, 8)
            elif item.source_key.endswith("linear_proj.weight"):
                shape = (8, 8)
            elif item.source_key.endswith("out_proj.weight"):
                shape = (8, 256)
            elif item.source_key.endswith("up_proj.weight"):
                shape = (8, 64)
            elif item.source_key.endswith("stacked_emb.weight"):
                shape = (25, 64)
            elif item.source_key.endswith("component_scales"):
                shape = (2,)
            elif item.source_key.endswith("unified_table.weight"):
                shape = (7_998_862, 16)
            elif item.source_key.endswith("hash_mults"):
                shape = (16, 3)
            elif item.source_key.endswith("order_mask"):
                shape = (3, 16)
            elif item.source_key.endswith("table_offsets") or item.source_key.endswith("table_sizes_t") or item.source_key.endswith("hash_bias") or item.source_key.endswith("order_for_table"):
                shape = (16,)
            source_sizes[item.source_key] = shape
        target_shape = source_sizes[item.source_key]
        if item.source_slice is not None:
            lo, hi = item.source_slice
            target_shape = (hi - lo, *target_shape[1:])
        if item.source_rows is not None:
            target_shape = (len(item.source_rows), *target_shape[1:])
        target_arrays[item.target_key] = _FakeTarget(target_shape)

    with pytest.raises(KeyError, match="unexpected.weight"):
        mod.validate_plan_against_metadata(cfg, _FakeMetadata(source_sizes), target_arrays)


def test_validate_plan_accepts_full_toy_mapping_with_neutral_tensors():
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )
    source_sizes: dict[str, tuple[int, ...]] = {}
    target_arrays: dict[str, _FakeTarget] = {
        "position_embedding.weight": _FakeTarget((4, 8)),
        "platform_embedding.embedding.weight": _FakeTarget((3, 8)),
        "layers.0.attention.rope_inv_freq": _FakeTarget((1,)),
    }
    for item in mod.build_key_plan(cfg):
        if item.source_key.endswith("linear_qkv.weight"):
            source_sizes[item.source_key] = (16, 8)
        elif item.source_key.endswith("linear_fc1.weight"):
            source_sizes[item.source_key] = (24, 8)
        elif item.source_key.endswith("linear_fc2.weight"):
            source_sizes[item.source_key] = (8, 12)
        elif item.source_key.endswith("linear_proj.weight"):
            source_sizes[item.source_key] = (8, 8)
        elif item.source_key == "embedding.word_embeddings.weight":
            source_sizes[item.source_key] = (65_536, 8)
        elif item.source_key.endswith("out_proj.weight"):
            source_sizes[item.source_key] = (8, 256)
        elif item.source_key.endswith("up_proj.weight"):
            source_sizes[item.source_key] = (8, 64)
        elif item.source_key.endswith("stacked_emb.weight"):
            source_sizes[item.source_key] = (25, 64)
        elif item.source_key.endswith("component_scales"):
            source_sizes[item.source_key] = (2,)
        elif item.source_key.endswith("unified_table.weight"):
            source_sizes[item.source_key] = (7_998_862, 16)
        elif item.source_key.endswith("hash_mults"):
            source_sizes[item.source_key] = (16, 3)
        elif item.source_key.endswith("order_mask"):
            source_sizes[item.source_key] = (3, 16)
        elif item.source_key.endswith("table_offsets") or item.source_key.endswith("table_sizes_t") or item.source_key.endswith("hash_bias") or item.source_key.endswith("order_for_table"):
            source_sizes[item.source_key] = (16,)
        else:
            source_sizes[item.source_key] = (8,)
        target_shape = source_sizes[item.source_key]
        if item.source_slice is not None:
            lo, hi = item.source_slice
            target_shape = (hi - lo, *target_shape[1:])
        if item.source_rows is not None:
            target_shape = (len(item.source_rows), *target_shape[1:])
        target_arrays[item.target_key] = _FakeTarget(target_shape)

    receipt = mod.validate_plan_against_metadata(cfg, _FakeMetadata(source_sizes), target_arrays)

    assert receipt["mapped_tensors"] == len(mod.build_key_plan(cfg))
    assert "position_embedding.weight" in receipt["neutral_tensors"]


def test_optional_domain_learned_tensors_are_mapped_when_all_are_present():
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
        domain_bottleneck_dim=4,
    )
    base_sources = {item.source_key for item in mod.build_key_plan(cfg)}
    source_keys = base_sources | set(mod.DOMAIN_SOURCE_TO_TARGET)

    plan = mod.build_key_plan(cfg, source_keys=source_keys)
    by_source = {item.source_key: item.target_key for item in plan}

    assert by_source["embedding.cppmega_domain.component_scales"] == (
        "domain_embedding.component_scales"
    )
    assert by_source["embedding.cppmega_domain.stacked_emb.weight"] == (
        "domain_embedding.stacked_emb.weight"
    )
    assert by_source["embedding.cppmega_domain.up_proj.weight"] == (
        "domain_embedding.up_proj.weight"
    )


def test_partial_domain_tensor_set_fails_closed() -> None:
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(depth=1)
    source_keys = {item.source_key for item in mod.build_key_plan(cfg)}
    source_keys.add("embedding.cppmega_domain.component_scales")

    with pytest.raises(KeyError, match="partial domain tensor set"):
        mod.build_key_plan(cfg, source_keys=source_keys)


def test_validate_plan_rejects_every_unmapped_learned_source_tensor() -> None:
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )
    source_sizes, targets = _toy_contract(mod, cfg)
    source_sizes["decoder.unmapped_learned.weight"] = (8, 8)

    with pytest.raises(KeyError, match="unmapped learned source tensors"):
        mod.validate_plan_against_metadata(
            cfg, _FakeMetadata(source_sizes), targets
        )


def test_validate_plan_rejects_unmapped_learned_tensor_outside_model_prefixes() -> None:
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
    )
    source_sizes, targets = _toy_contract(mod, cfg)
    source_sizes["auxiliary_adapter.weight"] = (8, 8)

    with pytest.raises(KeyError, match="auxiliary_adapter.weight"):
        mod.validate_plan_against_metadata(
            cfg, _FakeMetadata(source_sizes), targets
        )


def test_runtime_requirements_do_not_claim_domain_tensors_for_old_checkpoint() -> None:
    mod = _load_module()

    old = mod.conversion_runtime_requirements(domain_tensors_present=False)
    enriched = mod.conversion_runtime_requirements(domain_tensors_present=True)

    assert old["graph_routes"]["required"] is True
    assert old["graph_routes"]["beta"] == 1.0
    assert old["domain_routes"]["learned_tensors_present"] is False
    assert old["domain_routes"]["required"] is False
    assert old["domain_routes"]["residual_scale"] == 0.0
    assert enriched["domain_routes"]["learned_tensors_present"] is True
    assert enriched["domain_routes"]["required"] is True
    assert enriched["domain_routes"]["residual_scale"] == 1.0


def _parity_model() -> DenseCppLM:
    return DenseCppLM(
        DenseCppLMConfig(
            vocab_size=64,
            hidden_size=32,
            depth=1,
            ffn_hidden_size=64,
            max_seq_length=8,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            graph_routes_enabled=True,
            graph_attention_bias_beta=1.0,
            domain_residual_scale=1.0,
            require_domain_routes=True,
            ngram_hash_enabled=False,
            structure_residual_scale=0.0,
            platform_residual_scale=0.0,
        )
    )


def test_sidecar_rich_source_target_logit_parity_seam() -> None:
    mod = _load_module()
    mx.random.seed(73)
    source = _parity_model()
    source.domain_embedding.stacked_emb.weight = mx.random.normal(
        source.domain_embedding.stacked_emb.weight.shape
    )
    source.domain_embedding.up_proj.weight = mx.random.normal(
        source.domain_embedding.up_proj.weight.shape
    )
    mx.random.seed(73)
    target = _parity_model()
    target.domain_embedding.stacked_emb.weight = mx.random.normal(
        target.domain_embedding.stacked_emb.weight.shape
    )
    target.domain_embedding.up_proj.weight = mx.random.normal(
        target.domain_embedding.up_proj.weight.shape
    )
    tokens = mx.array([[2, 3, 5, 7, 11, 13, 17, 19]], dtype=mx.int32)
    graph_bias = mx.zeros((1, 8, 8), dtype=mx.float32)
    graph_bias[:, 7, 1] = 1.0
    kind_bias = mx.zeros_like(graph_bias)
    kind_bias[:, 7, 1] = 1.0
    domain_ids = mx.full(tokens.shape, 3, dtype=mx.int32)
    role_ids = mx.full(tokens.shape, 2, dtype=mx.int32)
    confidence_ids = mx.ones(tokens.shape, dtype=mx.int32)

    def source_forward(input_ids, **sidecars):
        graph_attention_bias = sidecars.pop("graph_attention_bias")
        graph_edge_kind_bias = sidecars.pop("graph_edge_kind_bias")
        return source.logits(
            input_ids,
            block_bias=graph_attention_bias,
            edge_kind_bias=graph_edge_kind_bias,
            **sidecars,
        )

    receipt = mod.verify_sidecar_logit_parity(
        source_forward=source_forward,
        target_model=target,
        input_ids=tokens,
        graph_attention_bias=graph_bias,
        graph_edge_kind_bias=kind_bias,
        domain_ids=domain_ids,
        role_ids=role_ids,
        confidence_ids=confidence_ids,
        atol=1e-6,
        rtol=1e-6,
    )

    assert receipt["graph_prior_nonzero"] == 1
    assert receipt["edge_kind_prior_nonzero"] == 1
    assert receipt["domain_tokens_nonzero"] == 8
    assert receipt["max_abs_logit_error"] <= 1e-6


def test_sidecar_parity_seam_detects_dropped_graph_route() -> None:
    mod = _load_module()
    mx.random.seed(91)
    source = _parity_model()
    mx.random.seed(91)
    target = _parity_model()
    tokens = mx.array([[2, 3, 5, 7, 11, 13, 17, 19]], dtype=mx.int32)
    graph_bias = mx.zeros((1, 8, 8), dtype=mx.float32)
    graph_bias[:, 7, 1] = 25.0
    kind_bias = mx.zeros_like(graph_bias)
    domain_ids = mx.full(tokens.shape, 3, dtype=mx.int32)
    role_ids = mx.full(tokens.shape, 2, dtype=mx.int32)
    confidence_ids = mx.ones(tokens.shape, dtype=mx.int32)

    def source_forward(input_ids, **sidecars):
        sidecars.pop("graph_attention_bias")
        sidecars.pop("graph_edge_kind_bias")
        return source.logits(
            input_ids,
            block_bias=mx.zeros_like(graph_bias),
            edge_kind_bias=mx.zeros_like(graph_bias),
            **sidecars,
        )

    with pytest.raises(AssertionError, match="logit parity failed"):
        mod.verify_sidecar_logit_parity(
            source_forward=source_forward,
            target_model=target,
            input_ids=tokens,
            graph_attention_bias=graph_bias,
            graph_edge_kind_bias=kind_bias,
            domain_ids=domain_ids,
            role_ids=role_ids,
            confidence_ids=confidence_ids,
            atol=1e-7,
            rtol=1e-7,
        )
