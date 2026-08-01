from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pickle
import subprocess
import sys
import zipfile
from dataclasses import dataclass
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


class _FakeProperties:
    dtype = "torch.float32"


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


@dataclass(frozen=True)
class _FixtureMetadataIndex:
    fqn: str
    offset: tuple[int, ...]
    index: int


@dataclass
class _FixtureStorageInfo:
    relative_path: str
    offset: int
    length: int


@dataclass
class _FixtureChunk:
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]


@dataclass
class _FixtureProperties:
    dtype: str


@dataclass
class _FixtureTensorMetadata:
    size: tuple[int, ...]
    properties: _FixtureProperties
    chunks: list[_FixtureChunk]


@dataclass
class _FixtureDcpMetadata:
    state_dict_metadata: dict[str, _FixtureTensorMetadata]
    storage_data: dict[_FixtureMetadataIndex, _FixtureStorageInfo]


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


def _fixture_source_state(mod, cfg) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5005)
    state: dict[str, np.ndarray] = {}

    def weight(shape: tuple[int, ...], *, scale: float = 0.08) -> np.ndarray:
        return rng.normal(0.0, scale, shape).astype(np.float32)

    state["embedding.word_embeddings.weight"] = weight(
        (cfg.vocab_size, cfg.hidden_size)
    )
    tables = 2 * cfg.ngram_hash_heads
    table_sizes = np.arange(
        cfg.ngram_hash_table_size,
        cfg.ngram_hash_table_size + tables,
        dtype=np.int64,
    )
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(table_sizes[:-1])))
    ngram = "embedding.cppmega_ngram_hash."
    state[ngram + "table_offsets"] = offsets
    state[ngram + "table_sizes_t"] = table_sizes
    state[ngram + "hash_mults"] = np.array([[3, 5, 7], [11, 13, 17]], dtype=np.int64)
    state[ngram + "hash_bias"] = np.array([19, 23], dtype=np.int64)
    state[ngram + "order_for_table"] = np.array([2, 3], dtype=np.int64)
    state[ngram + "order_mask"] = np.array([[1, 1], [1, 1], [0, 1]], dtype=np.int64)
    state[ngram + "unified_table.weight"] = weight(
        (int(table_sizes.sum()), cfg.ngram_hash_embed_dim)
    )
    state[ngram + "out_proj.weight"] = weight(
        (cfg.hidden_size, tables * cfg.ngram_hash_embed_dim)
    )
    structure = "embedding.cppmega_structure."
    state[structure + "component_scales"] = np.array([0.5, 0.5], dtype=np.float32)
    state[structure + "stacked_emb.weight"] = weight(
        (
            cfg.structure_num_categories + cfg.structure_max_dep_level,
            cfg.structure_bottleneck_dim,
        )
    )
    state[structure + "up_proj.weight"] = weight(
        (cfg.hidden_size, cfg.structure_bottleneck_dim)
    )
    for block in range(cfg.depth):
        attn = f"decoder.layers.{2 * block}.self_attention."
        state[attn + "linear_qkv.layer_norm_weight"] = np.ones(
            (cfg.hidden_size,), dtype=np.float32
        )
        state[attn + "linear_qkv.weight"] = weight((cfg.qkv_dim, cfg.hidden_size))
        state[attn + "linear_proj.weight"] = weight((cfg.hidden_size, cfg.q_proj_dim))
        mlp = f"decoder.layers.{2 * block + 1}.mlp."
        state[mlp + "linear_fc1.layer_norm_weight"] = np.ones(
            (cfg.hidden_size,), dtype=np.float32
        )
        state[mlp + "linear_fc1.weight"] = weight(
            (2 * cfg.ffn_hidden_size, cfg.hidden_size)
        )
        state[mlp + "linear_fc2.weight"] = weight(
            (cfg.hidden_size, cfg.ffn_hidden_size)
        )
    state["decoder.final_norm.weight"] = np.ones((cfg.hidden_size,), dtype=np.float32)
    assert {item.source_key for item in mod.build_key_plan(cfg)} == set(state)
    return state


def _torch_tensor_payload(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", b"fixture")
        archive.writestr("archive/data/0", array.tobytes(order="C"))
        archive.writestr("archive/version", b"3\n")
    return stream.getvalue()


def _write_torch_dist_fixture(mod, cfg, iter_dir: Path) -> dict[str, np.ndarray]:
    iter_dir.mkdir(parents=True)
    source = _fixture_source_state(mod, cfg)
    source["optimizer.state.exp_avg.decoder.final_norm.weight"] = np.zeros(
        (cfg.hidden_size,), dtype=np.float32
    )
    metadata_entries: dict[str, _FixtureTensorMetadata] = {}
    storage: dict[_FixtureMetadataIndex, _FixtureStorageInfo] = {}
    shard = bytearray()
    for index, (key, array) in enumerate(sorted(source.items())):
        payload = _torch_tensor_payload(array)
        offset = len(shard)
        shard.extend(payload)
        dtype = "torch.int64" if array.dtype == np.int64 else "torch.float32"
        metadata_entries[key] = _FixtureTensorMetadata(
            size=tuple(array.shape),
            properties=_FixtureProperties(dtype=dtype),
            chunks=[
                _FixtureChunk(
                    offsets=(0,) * array.ndim,
                    sizes=tuple(array.shape),
                )
            ],
        )
        storage[_FixtureMetadataIndex(key, (0,) * array.ndim, index)] = (
            _FixtureStorageInfo("__0_0.distcp", offset, len(payload))
        )
    (iter_dir / "__0_0.distcp").write_bytes(shard)
    (iter_dir / ".metadata").write_bytes(
        pickle.dumps(_FixtureDcpMetadata(metadata_entries, storage), protocol=4)
    )
    args = argparse.Namespace(
        num_layers=cfg.depth * 2,
        hidden_size=cfg.hidden_size,
        ffn_hidden_size=cfg.ffn_hidden_size,
        num_attention_heads=cfg.num_query_heads,
        num_query_groups=cfg.num_kv_heads,
        kv_channels=cfg.head_dim,
        padded_vocab_size=cfg.vocab_size,
        group_query_attention=True,
        hybrid_layer_pattern="*-" * cfg.depth,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        swiglu=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        position_embedding_type="rope",
        rotary_base=10_000,
        rotary_percent=1.0,
        rotary_interleaved=False,
        untie_embeddings_and_output_weights=False,
        multi_latent_attention=False,
        num_experts=None,
        mtp_num_layers=None,
        seq_length=cfg.max_seq_length,
        max_position_embeddings=cfg.max_seq_length,
    )
    with zipfile.ZipFile(iter_dir / "common.pt", "w") as archive:
        archive.writestr(
            "common/data.pkl",
            pickle.dumps({"args": args, "iteration": 7}, protocol=2),
        )
        archive.writestr("common/version", b"3\n")
    (iter_dir / "metadata.json").write_text(
        json.dumps(
            {
                "sharded_backend": "torch_dist",
                "sharded_backend_version": 1,
                "common_backend": "torch",
                "common_backend_version": 1,
            }
        ),
        encoding="utf-8",
    )
    source.pop("optimizer.state.exp_avg.decoder.final_norm.weight")
    return source


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

    assert (
        mod.KeyPlan(
            "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            "layers.0.attn_norm.weight",
        )
        in plan
    )
    assert (
        mod.KeyPlan(
            "decoder.layers.1.mlp.linear_fc1.layer_norm_weight",
            "layers.0.ffn_norm.weight",
        )
        in plan
    )
    assert (
        mod.KeyPlan(
            "decoder.layers.2.self_attention.linear_qkv.layer_norm_weight",
            "layers.1.attn_norm.weight",
        )
        in plan
    )
    assert (
        mod.KeyPlan(
            "decoder.layers.3.mlp.linear_fc1.layer_norm_weight",
            "layers.1.ffn_norm.weight",
        )
        in plan
    )


def test_converter_rejects_unsupported_dsa_checkpoint_mapping() -> None:
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(attention_mode="dsa")

    with pytest.raises(NotImplementedError, match="DSA indexer projections"):
        mod.build_key_plan(cfg)


def test_converter_module_imports_in_fresh_subprocess() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.convert_megatron_dense500m_torchdist_to_mlx",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
            elif (
                item.source_key.endswith("table_offsets")
                or item.source_key.endswith("table_sizes_t")
                or item.source_key.endswith("hash_bias")
                or item.source_key.endswith("order_for_table")
            ):
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
        mod.validate_plan_against_metadata(
            cfg, _FakeMetadata(source_sizes), target_arrays
        )


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
        elif (
            item.source_key.endswith("table_offsets")
            or item.source_key.endswith("table_sizes_t")
            or item.source_key.endswith("hash_bias")
            or item.source_key.endswith("order_for_table")
        ):
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

    receipt = mod.validate_plan_against_metadata(
        cfg, _FakeMetadata(source_sizes), target_arrays
    )

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
        mod.validate_plan_against_metadata(cfg, _FakeMetadata(source_sizes), targets)


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
        mod.validate_plan_against_metadata(cfg, _FakeMetadata(source_sizes), targets)


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


def test_converter_writes_evaluator_metadata_as_model_json(tmp_path: Path) -> None:
    mod = _load_module()

    weights_path, manifest_path = mod.conversion_output_paths(
        tmp_path / "dense500m.safetensors"
    )

    assert weights_path == (tmp_path / "dense500m.safetensors").resolve()
    assert manifest_path == (tmp_path / "model.json").resolve()


def test_converter_rejects_output_without_safetensors_suffix(tmp_path: Path) -> None:
    mod = _load_module()

    with pytest.raises(ValueError, match="must end in .safetensors"):
        mod.conversion_output_paths(tmp_path / "dense500m.bin")


def test_convert_torch_dist_fixture_runs_raw_dcp_to_mlx_logit_parity(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        vocab_size=32,
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
        max_seq_length=6,
        ngram_hash_heads=1,
        ngram_hash_table_size=7,
        ngram_hash_embed_dim=3,
        structure_num_categories=5,
        structure_max_dep_level=6,
        structure_bottleneck_dim=4,
    )
    checkpoint = tmp_path / "iter_0000001"
    _write_torch_dist_fixture(mod, cfg, checkpoint)
    output = tmp_path / "converted" / "weights.safetensors"

    manifest = mod.convert_checkpoint(checkpoint, output, cfg=cfg, bf16=False)

    assert output.is_file()
    model_json = output.parent / "model.json"
    assert model_json.is_file()
    payload = json.loads(model_json.read_text(encoding="utf-8"))
    assert payload == manifest
    assert payload["schema"] == "cppmega_megatron_dense500m_to_mlx_v4"
    assert payload["source_contract"]["route_pattern"] == "*-"
    assert payload["source_contract"]["tied_embeddings"] is True
    assert payload["validation"]["optimizer_exclusion"]["tensor_count"] == 1
    assert payload["validation"]["unsupported_tensors"] == []
    assert {item["path"] for item in payload["source_artifacts"]} == {
        ".metadata",
        "__0_0.distcp",
        "common.pt",
        "metadata.json",
    }
    assert all(len(item["sha256"]) == 64 for item in payload["source_artifacts"])
    assert payload["logit_parity"]["source_reference"] == (
        "raw_dcp_numpy_megatron_af_gqa_v1"
    )
    assert payload["logit_parity"]["reloaded_safetensors"] == str(output.resolve())
    assert payload["logit_parity"]["max_abs_logit_error"] <= 3e-4


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


class _ConvertedGQAFixture:
    _KEYS = {
        "token.weight",
        "q.weight",
        "k.weight",
        "v.weight",
        "out.weight",
        "lm_head.weight",
        "domain.scale",
        "role.scale",
        "confidence.scale",
    }

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.weights: dict[str, mx.array] = {}

    def load_weights(self, path: str, *, strict: bool = True):
        weights = dict(mx.load(path))
        if strict and set(weights) != self._KEYS:
            raise ValueError(
                f"fixture weight keys {sorted(weights)} do not match {sorted(self._KEYS)}"
            )
        self.weights = weights
        return self

    def parameters(self) -> dict[str, mx.array]:
        return self.weights

    def logits(
        self,
        input_ids: mx.array,
        *,
        block_bias: mx.array,
        edge_kind_bias: mx.array,
        domain_ids: mx.array,
        role_ids: mx.array,
        confidence_ids: mx.array,
        document_ids: mx.array,
    ) -> mx.array:
        cfg = self.cfg
        batch, seq = input_ids.shape
        hidden = (
            self.weights["token.weight"][input_ids]
            + domain_ids[..., None] * self.weights["domain.scale"]
            + role_ids[..., None] * self.weights["role.scale"]
            + confidence_ids[..., None] * self.weights["confidence.scale"]
        )
        q = (hidden @ self.weights["q.weight"].T).reshape(
            batch, seq, cfg.num_query_heads, cfg.head_dim
        )
        k = (hidden @ self.weights["k.weight"].T).reshape(
            batch, seq, cfg.num_kv_heads, cfg.head_dim
        )
        v = (hidden @ self.weights["v.weight"].T).reshape(
            batch, seq, cfg.num_kv_heads, cfg.head_dim
        )
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        heads_per_group = cfg.num_query_heads // cfg.num_kv_heads
        k = mx.broadcast_to(
            k[:, :, None, :, :],
            (batch, cfg.num_kv_heads, heads_per_group, seq, cfg.head_dim),
        ).reshape(batch, cfg.num_query_heads, seq, cfg.head_dim)
        v = mx.broadcast_to(
            v[:, :, None, :, :],
            (batch, cfg.num_kv_heads, heads_per_group, seq, cfg.head_dim),
        ).reshape(batch, cfg.num_query_heads, seq, cfg.head_dim)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / np.sqrt(cfg.head_dim)
        causal = mx.arange(seq)[:, None] >= mx.arange(seq)[None, :]
        same_document = document_ids[:, :, None] == document_ids[:, None, :]
        mask = causal[None, None, :, :] & same_document[:, None, :, :]
        scores = scores + (block_bias + edge_kind_bias)[:, None, :, :]
        scores = mx.where(mask, scores, mx.full(scores.shape, -mx.inf))
        attention = mx.softmax(scores, axis=-1) @ v
        attention = mx.transpose(attention, (0, 2, 1, 3)).reshape(
            batch, seq, cfg.hidden_size
        )
        hidden = hidden + attention @ self.weights["out.weight"].T
        return hidden @ self.weights["lm_head.weight"].T


def test_emitted_safetensors_match_independent_grouped_gqa_source(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    cfg = mod.Dense500MConversionConfig(
        vocab_size=32,
        hidden_size=8,
        depth=1,
        ffn_hidden_size=16,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
        max_seq_length=6,
    )
    rng = np.random.default_rng(811)
    source_qkv = rng.normal(0.0, 0.2, (cfg.qkv_dim, cfg.hidden_size)).astype(np.float32)
    weights_np = {
        "token.weight": rng.normal(0.0, 0.2, (cfg.vocab_size, cfg.hidden_size)).astype(
            np.float32
        ),
        "q.weight": source_qkv[list(mod.qkv_source_rows(cfg, "q"))],
        "k.weight": source_qkv[list(mod.qkv_source_rows(cfg, "k"))],
        "v.weight": source_qkv[list(mod.qkv_source_rows(cfg, "v"))],
        "out.weight": rng.normal(0.0, 0.2, (cfg.hidden_size, cfg.q_proj_dim)).astype(
            np.float32
        ),
        "lm_head.weight": rng.normal(
            0.0, 0.2, (cfg.vocab_size, cfg.hidden_size)
        ).astype(np.float32),
        "domain.scale": rng.normal(0.0, 0.1, (cfg.hidden_size,)).astype(np.float32),
        "role.scale": rng.normal(0.0, 0.1, (cfg.hidden_size,)).astype(np.float32),
        "confidence.scale": rng.normal(0.0, 0.1, (cfg.hidden_size,)).astype(np.float32),
    }
    emitted = tmp_path / "converted.safetensors"
    mx.save_safetensors(
        str(emitted),
        {key: mx.array(value) for key, value in weights_np.items()},
        metadata={"format": "mlx"},
    )

    tokens = mx.array([[2, 3, 5, 7, 11, 13]], dtype=mx.int32)
    document_ids = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)
    graph_bias = mx.zeros((1, 6, 6), dtype=mx.float32)
    graph_bias[:, 2, 0] = 1.25
    graph_bias[:, 5, 3] = -0.75
    kind_bias = mx.zeros_like(graph_bias)
    kind_bias[:, 2, 1] = 0.5
    kind_bias[:, 5, 4] = 0.25
    domain_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    role_ids = mx.array([[2, 3, 4, 5, 6, 7]], dtype=mx.int32)
    confidence_ids = mx.array([[1, 2, 3, 1, 2, 3]], dtype=mx.int32)

    def source_forward(input_ids, **sidecars):
        token_ids = np.asarray(input_ids, dtype=np.int64)
        docs = np.asarray(sidecars["document_ids"], dtype=np.int64)
        batch, seq = token_ids.shape
        hidden = (
            weights_np["token.weight"][token_ids]
            + np.asarray(sidecars["domain_ids"])[..., None] * weights_np["domain.scale"]
            + np.asarray(sidecars["role_ids"])[..., None] * weights_np["role.scale"]
            + np.asarray(sidecars["confidence_ids"])[..., None]
            * weights_np["confidence.scale"]
        )
        fused = hidden @ source_qkv.T
        heads_per_group = cfg.num_query_heads // cfg.num_kv_heads
        q_rows_per_group = heads_per_group * cfg.head_dim
        rows_per_group = q_rows_per_group + 2 * cfg.head_dim
        grouped = fused.reshape(batch, seq, cfg.num_kv_heads, rows_per_group)
        q = grouped[..., :q_rows_per_group].reshape(
            batch, seq, cfg.num_query_heads, cfg.head_dim
        )
        k = grouped[..., q_rows_per_group : q_rows_per_group + cfg.head_dim]
        v = grouped[..., q_rows_per_group + cfg.head_dim :]
        q = q.transpose(0, 2, 1, 3)
        k = np.repeat(k.transpose(0, 2, 1, 3), heads_per_group, axis=1)
        v = np.repeat(v.transpose(0, 2, 1, 3), heads_per_group, axis=1)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(cfg.head_dim)
        scores += (
            np.asarray(sidecars["graph_attention_bias"])
            + np.asarray(sidecars["graph_edge_kind_bias"])
        )[:, None, :, :]
        causal = np.arange(seq)[:, None] >= np.arange(seq)[None, :]
        same_document = docs[:, :, None] == docs[:, None, :]
        scores = np.where(causal[None, None] & same_document[:, None], scores, -np.inf)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attention = (
            np.matmul(probabilities, v)
            .transpose(0, 2, 1, 3)
            .reshape(batch, seq, cfg.hidden_size)
        )
        hidden += attention @ weights_np["out.weight"].T
        return hidden @ weights_np["lm_head.weight"].T

    receipt = mod.verify_emitted_safetensors_logit_parity(
        source_forward=source_forward,
        target_model_factory=lambda: _ConvertedGQAFixture(cfg),
        weights_path=emitted,
        input_ids=tokens,
        graph_attention_bias=graph_bias,
        graph_edge_kind_bias=kind_bias,
        domain_ids=domain_ids,
        role_ids=role_ids,
        confidence_ids=confidence_ids,
        document_ids=document_ids,
        atol=2e-5,
        rtol=2e-5,
    )

    assert receipt["relation_prior_nonzero"] == 2
    assert receipt["edge_kind_prior_nonzero"] == 2
    assert receipt["domain_tokens_nonzero"] == 6
    assert receipt["role_tokens_nonzero"] == 6
    assert receipt["confidence_tokens_nonzero"] == 6
    assert receipt["document_boundaries"] == 1
    assert receipt["reloaded_safetensors"] == str(emitted.resolve())
    assert receipt["max_abs_logit_error"] <= 2e-5


def _toy_conversion_cfg(mod):
    return mod.Dense500MConversionConfig(
        vocab_size=32,
        hidden_size=8,
        depth=1,
        ffn_hidden_size=12,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=2,
        max_seq_length=6,
        ngram_hash_heads=1,
        ngram_hash_table_size=7,
        ngram_hash_embed_dim=3,
        structure_num_categories=5,
        structure_max_dep_level=6,
        structure_bottleneck_dim=4,
    )


def test_publish_receipt_records_weights_sha256_and_completion(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    cfg = _toy_conversion_cfg(mod)
    checkpoint = tmp_path / "iter_0000001"
    _write_torch_dist_fixture(mod, cfg, checkpoint)
    output = tmp_path / "converted" / "weights.safetensors"

    manifest = mod.convert_checkpoint(checkpoint, output, cfg=cfg, bf16=False)

    payload = json.loads((output.parent / "model.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert payload["output_artifacts"] == [
        {
            "path": output.name,
            "bytes": output.stat().st_size,
            "sha256": digest,
        }
    ]
    assert payload["publish"] == {
        "complete": True,
        "completion_marker": "model.json",
        "weights_sha256": digest,
    }
    assert manifest["publish"]["weights_sha256"] == digest
    status = mod.published_checkpoint_status(output)
    assert status == {"complete": True, "weights_sha256": digest}


def test_interrupted_publish_between_writes_lacks_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    cfg = _toy_conversion_cfg(mod)
    checkpoint = tmp_path / "iter_0000001"
    _write_torch_dist_fixture(mod, cfg, checkpoint)
    output = tmp_path / "converted" / "weights.safetensors"

    real_replace = mod.os.replace

    def crashing_replace(src, dst):
        if Path(dst).name == "model.json":
            raise RuntimeError("simulated crash before manifest publish")
        return real_replace(src, dst)

    monkeypatch.setattr(mod.os, "replace", crashing_replace)

    with pytest.raises(RuntimeError, match="simulated crash"):
        mod.convert_checkpoint(checkpoint, output, cfg=cfg, bf16=False)

    assert output.is_file()
    assert not (output.parent / "model.json").exists()
    status = mod.published_checkpoint_status(output)
    assert status["complete"] is False
    assert "completion marker" in status["reason"]


def test_published_checkpoint_status_detects_weights_sha_mismatch(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    cfg = _toy_conversion_cfg(mod)
    checkpoint = tmp_path / "iter_0000001"
    _write_torch_dist_fixture(mod, cfg, checkpoint)
    output = tmp_path / "converted" / "weights.safetensors"

    mod.convert_checkpoint(checkpoint, output, cfg=cfg, bf16=False)
    with output.open("r+b") as fh:
        fh.seek(0)
        fh.write(b"\x00")

    status = mod.published_checkpoint_status(output)
    assert status["complete"] is False
    assert status["reason"] == "weights sha256 mismatch"


def test_sidecar_parity_seam_detects_dropped_graph_route() -> None:
    mod = _load_module()
    mx.random.seed(91)
    source = _parity_model()
    mx.random.seed(91)
    target = _parity_model()
    tokens = mx.array([[2, 3, 5, 7, 11, 13, 17, 19]], dtype=mx.int32)
    graph_bias = mx.zeros((1, 8, 8), dtype=mx.float32)
    graph_bias[:, 7, 4] = 25.0
    kind_bias = mx.zeros_like(graph_bias)
    kind_bias[:, 6, 4] = 1.0
    domain_ids = mx.full(tokens.shape, 3, dtype=mx.int32)
    role_ids = mx.full(tokens.shape, 2, dtype=mx.int32)
    confidence_ids = mx.ones(tokens.shape, dtype=mx.int32)
    document_ids = mx.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=mx.int32)

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
            document_ids=document_ids,
            atol=1e-7,
            rtol=1e-7,
        )
