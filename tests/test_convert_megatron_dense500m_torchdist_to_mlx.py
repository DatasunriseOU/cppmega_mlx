from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
