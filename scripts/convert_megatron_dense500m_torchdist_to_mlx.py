#!/usr/bin/env python3
"""Convert cppmega dense500m Megatron torch_dist checkpoints to MLX safetensors.

The H200 ``h200_cpp_world_mini`` lane trains the same dense C++ model shape as
``DenseCppLM`` but stores it through Megatron's distributed-checkpoint format:
48 alternating layer positions (attention, MLP) for 24 transformer blocks.

This converter is intentionally narrow and fail-closed. It does not try to
interpret arbitrary Megatron checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dense500MConversionConfig:
    vocab_size: int = 65_536
    hidden_size: int = 1280
    depth: int = 24
    ffn_hidden_size: int = 3456
    num_query_heads: int = 20
    num_kv_heads: int = 4
    head_dim: int = 64
    max_seq_length: int = 1024
    ngram_hash_heads: int = 8
    ngram_hash_table_size: int = 500_000
    ngram_hash_embed_dim: int = 16
    structure_components: str = "core"
    structure_num_categories: int = 9
    structure_max_dep_level: int = 16
    structure_bottleneck_dim: int = 64
    domain_num_domains: int = 64
    domain_num_roles: int = 128
    domain_num_confidences: int = 8
    domain_bottleneck_dim: int = 32
    attention_mode: Literal["gqa", "dsa"] = "gqa"

    @property
    def q_proj_dim(self) -> int:
        return self.num_query_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_proj_dim + 2 * self.kv_proj_dim


@dataclass(frozen=True)
class KeyPlan:
    source_key: str
    target_key: str
    source_slice: tuple[int, int] | None = None
    source_rows: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.source_slice is not None and self.source_rows is not None:
            raise ValueError("KeyPlan cannot use both source_slice and source_rows")


DOMAIN_SOURCE_TO_TARGET: dict[str, str] = {
    "embedding.cppmega_domain.component_scales": "domain_embedding.component_scales",
    "embedding.cppmega_domain.stacked_emb.weight": "domain_embedding.stacked_emb.weight",
    "embedding.cppmega_domain.up_proj.weight": "domain_embedding.up_proj.weight",
}

NGRAM_SOURCE_TO_TARGET: dict[str, str] = {
    "embedding.cppmega_ngram_hash.table_offsets": "ngram_hash_embedding.table_offsets",
    "embedding.cppmega_ngram_hash.table_sizes_t": "ngram_hash_embedding.table_sizes_t",
    "embedding.cppmega_ngram_hash.hash_mults": "ngram_hash_embedding.hash_mults",
    "embedding.cppmega_ngram_hash.hash_bias": "ngram_hash_embedding.hash_bias",
    "embedding.cppmega_ngram_hash.order_for_table": "ngram_hash_embedding.order_for_table",
    "embedding.cppmega_ngram_hash.order_mask": "ngram_hash_embedding.order_mask",
    "embedding.cppmega_ngram_hash.unified_table.weight": "ngram_hash_embedding.unified_table.weight",
    "embedding.cppmega_ngram_hash.out_proj.weight": "ngram_hash_embedding.out_proj.weight",
}

STRUCTURE_SOURCE_TO_TARGET: dict[str, str] = {
    "embedding.cppmega_structure.component_scales": "structure_embedding.component_scales",
    "embedding.cppmega_structure.stacked_emb.weight": "structure_embedding.stacked_emb.weight",
    "embedding.cppmega_structure.up_proj.weight": "structure_embedding.up_proj.weight",
}

# Tensor-valued model entries omitted from conversion must be listed here with
# a reason. The allowlist is intentionally empty for this narrow checkpoint
# family: every model tensor is either mapped or conversion fails.
UNMAPPED_SOURCE_TENSOR_ALLOWLIST: dict[str, str] = {}


def conversion_output_paths(output: Path) -> tuple[Path, Path]:
    """Return the evaluator-compatible weights and metadata paths."""

    weights_path = output.expanduser().resolve()
    if weights_path.suffix != ".safetensors":
        raise ValueError(
            f"conversion output must end in .safetensors, got {weights_path}"
        )
    return weights_path, weights_path.parent / "model.json"


def _require_supported_attention_mode(cfg: Dense500MConversionConfig) -> None:
    if cfg.attention_mode != "gqa":
        raise NotImplementedError(
            "Dense500M checkpoint conversion supports attention_mode='gqa' only; "
            "DSA indexer projections and graph-bias parameters do not have an "
            "implemented Megatron-to-MLX mapping"
        )


def qkv_source_rows(
    cfg: Dense500MConversionConfig,
    part: Literal["q", "k", "v"],
) -> tuple[int, ...]:
    """Rows for Megatron's grouped GQA QKV storage.

    Megatron packs self-attention QKV rows as query-groups:

        q...q, k, v | q...q, k, v | ...

    not as one contiguous ``all_q | all_k | all_v`` tensor. The MLX model keeps
    separate q/k/v projections, so conversion must gather the rows for each
    part across every query group.
    """

    if cfg.num_query_heads % cfg.num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads {cfg.num_query_heads} must be divisible by "
            f"num_kv_heads {cfg.num_kv_heads}"
        )
    query_heads_per_group = cfg.num_query_heads // cfg.num_kv_heads
    q_rows_per_group = query_heads_per_group * cfg.head_dim
    kv_rows_per_group = cfg.head_dim
    rows_per_group = q_rows_per_group + 2 * kv_rows_per_group
    rows: list[int] = []
    for group in range(cfg.num_kv_heads):
        base = group * rows_per_group
        if part == "q":
            start, end = base, base + q_rows_per_group
        elif part == "k":
            start, end = (
                base + q_rows_per_group,
                base + q_rows_per_group + kv_rows_per_group,
            )
        elif part == "v":
            start = base + q_rows_per_group + kv_rows_per_group
            end = start + kv_rows_per_group
        else:
            raise ValueError(f"unsupported qkv part {part!r}")
        rows.extend(range(start, end))
    return tuple(rows)


def _repo_imports() -> None:
    root_s = str(REPO_ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


class _StubPickleObject:
    def __init__(self, *args: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _TorchSize(tuple):
    def __new__(cls, value: Any = ()) -> "_TorchSize":
        return tuple.__new__(cls, value)


def _get_layout_stub(value: Any) -> Any:
    return value


def _mem_format_encoding_stub(value: Any) -> Any:
    return value


class _DcpMetadataUnpickler(pickle.Unpickler):
    """Read DCP metadata without importing the full torch/DTensor stack."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name == "Size":
            return _TorchSize
        if module == "torch" and name in {
            "bfloat16",
            "float16",
            "float32",
            "float",
            "int32",
            "int64",
            "strided",
        }:
            return f"{module}.{name}"
        if module == "torch.serialization" and name == "_get_layout":
            return _get_layout_stub
        if (
            module == "torch.distributed.checkpoint.metadata"
            and name == "_MEM_FORMAT_ENCODING"
        ):
            return _mem_format_encoding_stub
        return type(name, (_StubPickleObject,), {"__module__": module})


class _CommonStateUnpickler(_DcpMetadataUnpickler):
    """Read Megatron ``common.pt`` without importing Torch or Megatron."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "argparse" and name == "Namespace":
            return argparse.Namespace
        return super().find_class(module, name)


def read_megatron_common_state(iter_dir: Path) -> dict[str, Any]:
    common_path = iter_dir / "common.pt"
    if not common_path.is_file():
        raise FileNotFoundError(common_path)
    with zipfile.ZipFile(common_path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError(
                f"{common_path}: expected one Torch data.pkl, found {names}"
            )
        state = _CommonStateUnpickler(io.BytesIO(archive.read(names[0]))).load()
    if not isinstance(state, dict) or not isinstance(
        state.get("args"), argparse.Namespace
    ):
        raise TypeError(f"{common_path}: expected Megatron args Namespace")
    return state


def _require_arg(args: argparse.Namespace, name: str, expected: Any) -> Any:
    actual = getattr(args, name, None)
    if actual != expected:
        raise ValueError(
            f"unsupported Megatron config {name}={actual!r}; expected {expected!r}"
        )
    return actual


def validate_source_checkpoint_contract(
    iter_dir: Path,
    cfg: Dense500MConversionConfig,
    metadata: Any,
) -> dict[str, Any]:
    """Fail closed unless the source is the supported dense AF/GQA checkpoint."""

    backend_path = iter_dir / "metadata.json"
    if not backend_path.is_file():
        raise FileNotFoundError(backend_path)
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    expected_backend = {
        "sharded_backend": "torch_dist",
        "sharded_backend_version": 1,
        "common_backend": "torch",
        "common_backend_version": 1,
    }
    if backend != expected_backend:
        raise ValueError(
            f"unsupported checkpoint backend metadata {backend!r}; "
            f"expected {expected_backend!r}"
        )

    common = read_megatron_common_state(iter_dir)
    args = common["args"]
    expected_pattern = "*-" * cfg.depth
    for name, expected in (
        ("num_layers", cfg.depth * 2),
        ("hidden_size", cfg.hidden_size),
        ("ffn_hidden_size", cfg.ffn_hidden_size),
        ("num_attention_heads", cfg.num_query_heads),
        ("num_query_groups", cfg.num_kv_heads),
        ("kv_channels", cfg.head_dim),
        ("padded_vocab_size", cfg.vocab_size),
        ("group_query_attention", True),
        ("hybrid_layer_pattern", expected_pattern),
        ("normalization", "RMSNorm"),
        ("layernorm_epsilon", 1e-5),
        ("swiglu", True),
        ("add_bias_linear", False),
        ("add_qkv_bias", False),
        ("position_embedding_type", "rope"),
        ("rotary_base", 10_000),
        ("rotary_percent", 1.0),
        ("rotary_interleaved", False),
        ("untie_embeddings_and_output_weights", False),
        ("multi_latent_attention", False),
        ("num_experts", None),
        ("mtp_num_layers", None),
    ):
        _require_arg(args, name, expected)

    specs = _metadata_tensor_specs(metadata)
    model_keys = sorted(
        key
        for key in specs
        if key.startswith(("embedding.", "decoder.", "output_layer."))
    )
    unsupported_families = {
        "mla": [key for key in model_keys if "mla" in key.lower()],
        "moe": [
            key
            for key in model_keys
            if any(token in key.lower() for token in ("expert", "router", "moe"))
        ],
        "untied_output": [key for key in model_keys if key.startswith("output_layer.")],
    }
    blocking = {family: keys for family, keys in unsupported_families.items() if keys}
    if blocking:
        raise ValueError(
            "unsupported source tensor families:\n"
            + json.dumps(blocking, indent=2, sort_keys=True)
        )
    return {
        "backend": expected_backend,
        "checkpoint_iteration": int(common.get("iteration", -1)),
        "source_sequence_length": int(getattr(args, "seq_length")),
        "source_max_position_embeddings": int(getattr(args, "max_position_embeddings")),
        "route_pattern": expected_pattern,
        "attention": "grouped_query_attention",
        "q_heads": cfg.num_query_heads,
        "kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "tied_embeddings": True,
        "mla": False,
        "moe": False,
        "optimizer": "excluded",
    }


def _sha256_receipt(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def source_artifact_hashes(iter_dir: Path) -> list[dict[str, Any]]:
    return [
        _sha256_receipt(path)
        for path in sorted(item for item in iter_dir.iterdir() if item.is_file())
    ]


def _fsync_file(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resolve_checkpoint_iter(path: Path) -> Path:
    """Return the Megatron iteration directory for a stage or iter path."""

    path = path.expanduser().resolve()
    if (
        path.is_dir()
        and (path / ".metadata").is_file()
        and (path / "__0_0.distcp").is_file()
    ):
        return path
    latest = path / "latest_checkpointed_iteration.txt"
    if path.is_dir() and latest.is_file():
        raw = latest.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError(f"{latest}: empty latest checkpoint marker")
        iteration = int(raw)
        iter_dir = path / f"iter_{iteration:07d}"
        if not iter_dir.is_dir():
            raise FileNotFoundError(iter_dir)
        return iter_dir
    raise ValueError(f"{path}: expected Megatron stage dir or iter_XXXXXXX dir")


def build_key_plan(
    cfg: Dense500MConversionConfig,
    *,
    source_keys: set[str] | None = None,
) -> list[KeyPlan]:
    _require_supported_attention_mode(cfg)
    plan = [
        KeyPlan("embedding.word_embeddings.weight", "token_embedding.weight"),
        KeyPlan("embedding.word_embeddings.weight", "lm_head.weight"),
        *(
            KeyPlan(source, target)
            for source, target in NGRAM_SOURCE_TO_TARGET.items()
        ),
        *(
            KeyPlan(source, target)
            for source, target in STRUCTURE_SOURCE_TO_TARGET.items()
        ),
        KeyPlan("decoder.final_norm.weight", "norm.weight"),
    ]
    for block in range(cfg.depth):
        attn_layer = block * 2
        mlp_layer = block * 2 + 1
        plan.extend(
            [
                KeyPlan(
                    f"decoder.layers.{attn_layer}.self_attention.linear_qkv.layer_norm_weight",
                    f"layers.{block}.attn_norm.weight",
                ),
                KeyPlan(
                    f"decoder.layers.{attn_layer}.self_attention.linear_qkv.weight",
                    f"layers.{block}.attention.q_proj.weight",
                    source_rows=qkv_source_rows(cfg, "q"),
                ),
                KeyPlan(
                    f"decoder.layers.{attn_layer}.self_attention.linear_qkv.weight",
                    f"layers.{block}.attention.k_proj.weight",
                    source_rows=qkv_source_rows(cfg, "k"),
                ),
                KeyPlan(
                    f"decoder.layers.{attn_layer}.self_attention.linear_qkv.weight",
                    f"layers.{block}.attention.v_proj.weight",
                    source_rows=qkv_source_rows(cfg, "v"),
                ),
                KeyPlan(
                    f"decoder.layers.{attn_layer}.self_attention.linear_proj.weight",
                    f"layers.{block}.attention.out_proj.weight",
                ),
                KeyPlan(
                    f"decoder.layers.{mlp_layer}.mlp.linear_fc1.layer_norm_weight",
                    f"layers.{block}.ffn_norm.weight",
                ),
                KeyPlan(
                    f"decoder.layers.{mlp_layer}.mlp.linear_fc1.weight",
                    f"layers.{block}.ffn.gate_proj.weight",
                    (0, cfg.ffn_hidden_size),
                ),
                KeyPlan(
                    f"decoder.layers.{mlp_layer}.mlp.linear_fc1.weight",
                    f"layers.{block}.ffn.up_proj.weight",
                    (cfg.ffn_hidden_size, 2 * cfg.ffn_hidden_size),
                ),
                KeyPlan(
                    f"decoder.layers.{mlp_layer}.mlp.linear_fc2.weight",
                    f"layers.{block}.ffn.down_proj.weight",
                ),
            ]
        )
    if source_keys is not None:
        present_domain = set(DOMAIN_SOURCE_TO_TARGET) & set(source_keys)
        if present_domain and present_domain != set(DOMAIN_SOURCE_TO_TARGET):
            missing = sorted(set(DOMAIN_SOURCE_TO_TARGET) - present_domain)
            raise KeyError(
                f"partial domain tensor set in Megatron checkpoint; missing {missing}"
            )
        if present_domain:
            plan.extend(
                KeyPlan(source, target)
                for source, target in DOMAIN_SOURCE_TO_TARGET.items()
            )
    return plan


def _import_runtime():
    _repo_imports()
    # Local Mac runs often have a broad, dirty site-packages tree. PyTorch 2.13
    # autoloads third-party device backends by scanning entry points at import
    # time; in this workspace that can hang on unrelated broken dist-info. The
    # converter only needs CPU tensor loading from DCP, so disable the autoload.
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "converter requires local MLX runtime; run with cppmega.mlx Python paths"
        ) from exc
    from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig

    return mx, tree_flatten, DenseCppLM, DenseCppLMConfig


def _make_target_model(
    cfg: Dense500MConversionConfig,
    *,
    bf16: bool,
    domain_tensors_present: bool = False,
):
    _require_supported_attention_mode(cfg)
    mx, _tree_flatten, DenseCppLM, DenseCppLMConfig = _import_runtime()
    dtype = mx.bfloat16 if bf16 else None
    model_cfg = DenseCppLMConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        ffn_hidden_size=cfg.ffn_hidden_size,
        max_seq_length=cfg.max_seq_length,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        attention_mode=cfg.attention_mode,
        structure_components=cfg.structure_components,
        structure_num_categories=cfg.structure_num_categories,
        structure_max_dep_level=cfg.structure_max_dep_level,
        structure_max_ast_depth=20,
        structure_max_sibling_index=10,
        structure_num_node_types=64,
        structure_bottleneck_dim=cfg.structure_bottleneck_dim,
        ngram_hash_heads=cfg.ngram_hash_heads,
        ngram_hash_table_size=cfg.ngram_hash_table_size,
        ngram_hash_embed_dim=cfg.ngram_hash_embed_dim,
        require_graph_routes=True,
        graph_routes_enabled=True,
        domain_residual_scale=1.0 if domain_tensors_present else 0.0,
        require_domain_routes=domain_tensors_present,
        domain_num_domains=cfg.domain_num_domains,
        domain_num_roles=cfg.domain_num_roles,
        domain_num_confidences=cfg.domain_num_confidences,
        domain_bottleneck_dim=cfg.domain_bottleneck_dim,
    )
    return model_cfg, DenseCppLM(model_cfg, dtype=dtype)


def _target_arrays(
    cfg: Dense500MConversionConfig,
    *,
    bf16: bool,
    domain_tensors_present: bool = False,
) -> dict[str, Any]:
    mx, tree_flatten, _DenseCppLM, _DenseCppLMConfig = _import_runtime()
    _model_cfg, model = _make_target_model(
        cfg,
        bf16=bf16,
        domain_tensors_present=domain_tensors_present,
    )
    arrays = dict(tree_flatten(model.parameters()))
    # Megatron used RoPE-only positions and has no platform embedding in this
    # checkpoint. Keep the target tensors explicit and neutral.
    if "position_embedding.weight" in arrays:
        arrays["position_embedding.weight"] = mx.zeros_like(
            arrays["position_embedding.weight"]
        )
    if "platform_embedding.embedding.weight" in arrays:
        arrays["platform_embedding.embedding.weight"] = mx.zeros_like(
            arrays["platform_embedding.embedding.weight"]
        )
    return arrays


def _metadata_tensor_specs(metadata: Any) -> dict[str, tuple[tuple[int, ...], Any]]:
    specs: dict[str, tuple[tuple[int, ...], Any]] = {}
    for key, value in metadata.state_dict_metadata.items():
        if hasattr(value, "size") and hasattr(value, "properties"):
            specs[str(key)] = (
                tuple(int(x) for x in value.size),
                _metadata_dtype(value),
            )
    return specs


def _metadata_dtype(value: Any) -> str:
    properties = getattr(value, "properties", None)
    if properties is None:
        raise TypeError(f"metadata entry {value!r} has no tensor properties")
    state = getattr(properties, "state", None)
    if state is None and hasattr(properties, "dtype"):
        return str(properties.dtype)
    if isinstance(state, tuple) and state:
        dtype = state[0]
        if isinstance(dtype, type):
            return dtype.__name__
        return str(dtype)
    raise TypeError(f"unsupported tensor properties state {state!r}")


def _is_learned_source_tensor(key: str) -> bool:
    if key.startswith("optimizer."):
        return False
    if key.startswith(("embedding.", "decoder.", "output_layer.")):
        return True
    return key.endswith(
        (
            ".weight",
            ".bias",
            ".component_scales",
            ".layer_norm_weight",
            ".A_log",
            ".D",
        )
    )


def validate_plan_against_metadata(
    cfg: Dense500MConversionConfig,
    metadata: Any,
    target_arrays: dict[str, Any],
) -> dict[str, Any]:
    specs = _metadata_tensor_specs(metadata)
    plan = build_key_plan(cfg, source_keys=set(specs))
    required_sources = sorted({item.source_key for item in plan})
    missing_sources = [key for key in required_sources if key not in specs]
    if missing_sources:
        raise KeyError(
            f"Megatron checkpoint missing required source tensors: {missing_sources}"
        )
    mapped_sources = set(required_sources)
    allowed_sources = set(UNMAPPED_SOURCE_TENSOR_ALLOWLIST)
    unmapped_learned_sources = sorted(
        key
        for key in set(specs) - mapped_sources - allowed_sources
        if _is_learned_source_tensor(key)
    )
    if unmapped_learned_sources:
        raise KeyError(f"unmapped learned source tensors: {unmapped_learned_sources}")

    plan_targets = {item.target_key for item in plan}
    neutral_targets = {
        "position_embedding.weight",
        "platform_embedding.embedding.weight",
    } | {key for key in target_arrays if key.endswith(".rope_inv_freq")}
    missing_targets = [key for key in plan_targets if key not in target_arrays]
    if missing_targets:
        raise KeyError(f"MLX model missing target tensors: {missing_targets}")
    unmapped = sorted(set(target_arrays) - plan_targets - neutral_targets)
    if unmapped:
        raise KeyError(
            f"MLX target tensors are neither mapped nor neutral-initialized: {unmapped}"
        )

    errors: list[str] = []
    tensor_map: list[dict[str, Any]] = []
    for item in plan:
        src_shape, src_dtype = specs[item.source_key]
        if src_dtype not in {
            "torch.bfloat16",
            "bfloat16",
            "torch.float32",
            "torch.float",
            "float32",
            "float",
            "torch.int64",
            "int64",
            "LongStorage",
            "torch.int32",
            "int32",
            "IntStorage",
        }:
            errors.append(
                f"{item.source_key}->{item.target_key}: unsupported dtype {src_dtype}"
            )
        target_shape = tuple(int(x) for x in target_arrays[item.target_key].shape)
        expected_shape = src_shape
        if item.source_slice is not None:
            lo, hi = item.source_slice
            expected_shape = (hi - lo, *src_shape[1:])
        if item.source_rows is not None:
            expected_shape = (len(item.source_rows), *src_shape[1:])
        if target_shape != expected_shape:
            errors.append(
                f"{item.source_key}->{item.target_key}: source {expected_shape} "
                f"target {target_shape} dtype={src_dtype}"
            )
        transform = "identity"
        if item.source_slice is not None:
            transform = f"row_slice[{item.source_slice[0]}:{item.source_slice[1]}]"
        elif item.source_rows is not None:
            transform = f"grouped_gqa_row_gather[{len(item.source_rows)}]"
        if item.target_key == "lm_head.weight":
            transform = "tied_embedding_alias"
        tensor_map.append(
            {
                "source": item.source_key,
                "target": item.target_key,
                "source_shape": list(src_shape),
                "target_shape": list(target_shape),
                "source_dtype": src_dtype,
                "target_dtype": str(
                    getattr(target_arrays[item.target_key], "dtype", "unknown")
                ),
                "transform": transform,
            }
        )
    if errors:
        raise ValueError("tensor mapping blockers:\n" + "\n".join(errors))
    optimizer_keys = sorted(key for key in specs if key.startswith("optimizer."))
    return {
        "mapped_tensors": len(plan),
        "unique_sources": len(required_sources),
        "neutral_tensors": sorted(neutral_targets & set(target_arrays)),
        "target_tensors": len(target_arrays),
        "domain_tensors_present": set(DOMAIN_SOURCE_TO_TARGET) <= set(specs),
        "allowed_unmapped_source_tensors": sorted(set(specs) & allowed_sources),
        "optimizer_exclusion": {
            "namespace": "optimizer.",
            "tensor_count": len(optimizer_keys),
        },
        "unsupported_tensors": [],
        "tensor_map": tensor_map,
    }


def read_dcp_metadata(iter_dir: Path) -> Any:
    metadata_path = iter_dir / ".metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("rb") as fh:
        return _DcpMetadataUnpickler(fh).load()


def load_required_source_tensors(
    iter_dir: Path,
    cfg: Dense500MConversionConfig,
    metadata: Any,
) -> dict[str, np.ndarray]:
    specs = _metadata_tensor_specs(metadata)
    source_keys = sorted(
        {item.source_key for item in build_key_plan(cfg, source_keys=set(specs))}
    )
    state: dict[str, np.ndarray] = {}
    for key in source_keys:
        shape, dtype = specs[key]
        state[key] = _read_dcp_tensor(iter_dir, metadata, key, shape, dtype)
    return state


def _read_dcp_tensor(
    iter_dir: Path,
    metadata: Any,
    key: str,
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    matches = [
        (index, info)
        for index, info in metadata.storage_data.items()
        if getattr(index, "fqn", None) == key
    ]
    if not matches:
        raise ValueError(f"{key}: missing DCP storage chunks")
    chunk_sizes = {
        tuple(int(x) for x in chunk.offsets): tuple(int(x) for x in chunk.sizes)
        for chunk in metadata.state_dict_metadata[key].chunks
    }
    if len(matches) != len(chunk_sizes):
        raise ValueError(
            f"{key}: storage chunks {len(matches)} do not match metadata chunks "
            f"{len(chunk_sizes)}"
        )
    full = np.empty(shape, dtype=_numpy_storage_dtype(dtype))
    for index, info in sorted(
        matches, key=lambda item: int(getattr(item[0], "index", 0) or 0)
    ):
        offsets = tuple(int(x) for x in getattr(index, "offset", ()))
        if offsets not in chunk_sizes:
            raise ValueError(
                f"{key}: storage offset {offsets} missing from chunk metadata"
            )
        chunk_shape = chunk_sizes[offsets]
        chunk = _read_dcp_tensor_chunk(iter_dir, info, key, chunk_shape, dtype)
        target = tuple(
            slice(start, start + size) for start, size in zip(offsets, chunk_shape)
        )
        full[target] = chunk
    return full


def _read_dcp_tensor_chunk(
    iter_dir: Path,
    info: Any,
    key: str,
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    data_file = iter_dir / info.relative_path
    with data_file.open("rb") as fh:
        fh.seek(int(info.offset))
        payload = fh.read(int(info.length))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw = archive.read("archive/data/0")

    expected_items = int(np.prod(shape, dtype=np.int64))
    if dtype in {"torch.bfloat16", "bfloat16"}:
        expected_bytes = expected_items * 2
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{key}: expected {expected_bytes} bf16 bytes, got {len(raw)}"
            )
        return np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    if dtype in {"torch.float32", "torch.float", "float32", "float"}:
        expected_bytes = expected_items * 4
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{key}: expected {expected_bytes} fp32 bytes, got {len(raw)}"
            )
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    if dtype in {"torch.int64", "int64", "LongStorage"}:
        expected_bytes = expected_items * 8
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{key}: expected {expected_bytes} int64 bytes, got {len(raw)}"
            )
        return np.frombuffer(raw, dtype="<i8").reshape(shape)
    if dtype in {"torch.int32", "int32", "IntStorage"}:
        expected_bytes = expected_items * 4
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{key}: expected {expected_bytes} int32 bytes, got {len(raw)}"
            )
        return np.frombuffer(raw, dtype="<i4").reshape(shape)
    raise TypeError(f"{key}: unsupported source tensor dtype {dtype!r}")


def _numpy_storage_dtype(dtype: str) -> np.dtype:
    if dtype in {"torch.bfloat16", "bfloat16"}:
        return np.dtype(np.uint16)
    if dtype in {"torch.float32", "torch.float", "float32", "float"}:
        return np.dtype(np.float32)
    if dtype in {"torch.int64", "int64", "LongStorage"}:
        return np.dtype(np.int64)
    if dtype in {"torch.int32", "int32", "IntStorage"}:
        return np.dtype(np.int32)
    raise TypeError(f"unsupported source tensor dtype {dtype!r}")


def _numpy_to_mlx_array(tensor: np.ndarray, *, bf16: bool) -> Any:
    mx, _tree_flatten, _DenseCppLM, _DenseCppLMConfig = _import_runtime()
    if tensor.dtype == np.uint16:
        fp32 = (tensor.astype(np.uint32) << 16).view(np.float32)
        arr = mx.array(fp32)
        return arr.astype(mx.bfloat16 if bf16 else mx.float32)
    if tensor.dtype == np.float32:
        arr = mx.array(tensor)
        return arr.astype(mx.bfloat16 if bf16 else mx.float32)
    if tensor.dtype in (np.int64, np.int32):
        return mx.array(tensor, dtype=mx.int64)
    raise TypeError(f"unsupported source tensor dtype {tensor.dtype}")


def apply_mapping(
    target_arrays: dict[str, Any],
    source_state: dict[str, np.ndarray],
    cfg: Dense500MConversionConfig,
    *,
    bf16: bool,
) -> dict[str, Any]:
    for item in build_key_plan(cfg, source_keys=set(source_state)):
        tensor = source_state[item.source_key]
        if item.source_slice is not None:
            lo, hi = item.source_slice
            tensor = np.ascontiguousarray(tensor[lo:hi, ...])
        if item.source_rows is not None:
            tensor = np.ascontiguousarray(tensor[list(item.source_rows), ...])
        target_arrays[item.target_key] = _numpy_to_mlx_array(tensor, bf16=bf16)
    return target_arrays


def conversion_runtime_requirements(
    *,
    domain_tensors_present: bool,
    source_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Runtime sidecar contract recorded beside converted weights."""

    if source_keys is None:
        ngram_present = True
        structure_present = True
    else:
        ngram_present = set(NGRAM_SOURCE_TO_TARGET) <= set(source_keys)
        structure_present = set(STRUCTURE_SOURCE_TO_TARGET) <= set(source_keys)

    return {
        "recipe": "stage1_graph_domain_v1",
        "graph_routes": {
            "required": True,
            "beta": 1.0,
            "sidecar_schema": "cppmega_graph_routes_v2",
            "requires_edge_kinds": True,
            "compiled_arrays": [
                "graph_attention_bias",
                "graph_edge_kind_bias",
            ],
        },
        "domain_routes": {
            "learned_tensors_present": bool(domain_tensors_present),
            "required": bool(domain_tensors_present),
            "residual_scale": 1.0 if domain_tensors_present else 0.0,
            "required_sidecars": [
                "domain_ids",
                "role_ids",
                "confidence_ids",
            ]
            if domain_tensors_present
            else [],
        },
        "side_channels": {
            "ngram_hash": {
                "learned_tensors_present": bool(ngram_present),
                "required": True,
                "source_schema": "cppmega_ngram_hash_v1",
                "target_tensors": sorted(NGRAM_SOURCE_TO_TARGET.values()),
            },
            "structure": {
                "learned_tensors_present": bool(structure_present),
                "required": True,
                "source_schema": "cppmega_structure_v1",
                "target_tensors": sorted(STRUCTURE_SOURCE_TO_TARGET.values()),
            },
        },
        "packed_documents": {
            "document_ids_required_when_packed": True,
            "cross_document_graph_routes_allowed": False,
        },
    }


def verify_sidecar_logit_parity(
    *,
    source_forward: Any,
    target_model: Any,
    input_ids: Any,
    graph_attention_bias: Any,
    graph_edge_kind_bias: Any,
    domain_ids: Any,
    role_ids: Any,
    confidence_ids: Any,
    document_ids: Any,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Compare source/target logits with the production sidecar channels live."""

    mx, _tree_flatten, _DenseCppLM, _DenseCppLMConfig = _import_runtime()
    batch_shape = tuple(int(value) for value in input_ids.shape)
    if len(batch_shape) != 2:
        raise ValueError(f"input_ids must be shaped (B,S), got {batch_shape}")
    bias_shape = (batch_shape[0], batch_shape[1], batch_shape[1])
    for name, value in (
        ("graph_attention_bias", graph_attention_bias),
        ("graph_edge_kind_bias", graph_edge_kind_bias),
    ):
        if tuple(value.shape) != bias_shape or value.dtype != mx.float32:
            raise ValueError(
                f"{name} must be float32 with shape {bias_shape}, got "
                f"{value.dtype} {tuple(value.shape)}"
            )
    sidecars = {
        "graph_attention_bias": graph_attention_bias,
        "graph_edge_kind_bias": graph_edge_kind_bias,
        "domain_ids": domain_ids,
        "role_ids": role_ids,
        "confidence_ids": confidence_ids,
        "document_ids": document_ids,
    }
    for name in ("domain_ids", "role_ids", "confidence_ids", "document_ids"):
        if tuple(sidecars[name].shape) != batch_shape:
            raise ValueError(
                f"{name} must match input_ids shape {batch_shape}, got "
                f"{tuple(sidecars[name].shape)}"
            )

    relation_np = np.asarray(graph_attention_bias)
    edge_kind_np = np.asarray(graph_edge_kind_bias)
    document_np = np.asarray(document_ids)
    if not np.issubdtype(document_np.dtype, np.integer):
        raise ValueError(
            f"document_ids must use an integer dtype, got {document_np.dtype}"
        )
    document_boundaries = int(
        np.count_nonzero(document_np[:, 1:] != document_np[:, :-1])
    )
    if document_boundaries == 0:
        raise ValueError(
            "fixed-input logit parity requires at least one document boundary"
        )
    for name, values in (
        ("graph_attention_bias", relation_np),
        ("graph_edge_kind_bias", edge_kind_np),
    ):
        if not np.count_nonzero(values):
            raise ValueError(f"fixed-input logit parity requires nonzero {name}")
        cross_document = document_np[:, :, None] != document_np[:, None, :]
        if np.any(values[cross_document] != 0):
            raise ValueError(
                f"{name} contains a route crossing a fixed-input document boundary"
            )
    channel_nonzero = {
        name: int(np.count_nonzero(np.asarray(sidecars[name])))
        for name in ("domain_ids", "role_ids", "confidence_ids")
    }
    missing_channels = [name for name, count in channel_nonzero.items() if count == 0]
    if missing_channels:
        raise ValueError(
            "fixed-input logit parity requires nonzero domain channels; got zero "
            f"values for {missing_channels}"
        )

    source_logits = source_forward(input_ids, **dict(sidecars))
    if isinstance(source_logits, tuple):
        source_logits = source_logits[0]
    target_logits = target_model.logits(
        input_ids,
        block_bias=graph_attention_bias,
        edge_kind_bias=graph_edge_kind_bias,
        domain_ids=domain_ids,
        role_ids=role_ids,
        confidence_ids=confidence_ids,
        document_ids=document_ids,
    )
    if isinstance(source_logits, mx.array):
        mx.eval(source_logits, target_logits)
    else:
        mx.eval(target_logits)
    source_np = np.asarray(source_logits, dtype=np.float32)
    target_np = np.asarray(target_logits, dtype=np.float32)
    if source_np.shape != target_np.shape:
        raise AssertionError(
            f"logit parity failed: source shape {source_np.shape} != "
            f"target shape {target_np.shape}"
        )
    error = np.abs(source_np - target_np)
    tolerance = float(atol) + float(rtol) * np.abs(source_np)
    if np.any(error > tolerance):
        raise AssertionError(
            "logit parity failed: max_abs_error="
            f"{float(error.max(initial=0.0)):.9g} atol={atol} rtol={rtol}"
        )
    combined_graph = relation_np + edge_kind_np
    return {
        "max_abs_logit_error": float(error.max(initial=0.0)),
        "graph_prior_nonzero": int(np.count_nonzero(combined_graph)),
        "relation_prior_nonzero": int(np.count_nonzero(relation_np)),
        "edge_kind_prior_nonzero": int(np.count_nonzero(edge_kind_np)),
        "domain_tokens_nonzero": channel_nonzero["domain_ids"],
        "role_tokens_nonzero": channel_nonzero["role_ids"],
        "confidence_tokens_nonzero": channel_nonzero["confidence_ids"],
        "document_boundaries": document_boundaries,
    }


def verify_emitted_safetensors_logit_parity(
    *,
    source_forward: Any,
    target_model_factory: Any,
    weights_path: Path,
    input_ids: Any,
    graph_attention_bias: Any,
    graph_edge_kind_bias: Any,
    domain_ids: Any,
    role_ids: Any,
    confidence_ids: Any,
    document_ids: Any,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Reload emitted safetensors before comparing fixed-input logits."""

    weights_path = weights_path.expanduser().resolve()
    if weights_path.suffix != ".safetensors":
        raise ValueError(f"parity weights must be .safetensors, got {weights_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    target_model = target_model_factory()
    target_model.load_weights(str(weights_path), strict=True)
    mx, _tree_flatten, _DenseCppLM, _DenseCppLMConfig = _import_runtime()
    mx.eval(target_model.parameters())
    receipt = verify_sidecar_logit_parity(
        source_forward=source_forward,
        target_model=target_model,
        input_ids=input_ids,
        graph_attention_bias=graph_attention_bias,
        graph_edge_kind_bias=graph_edge_kind_bias,
        domain_ids=domain_ids,
        role_ids=role_ids,
        confidence_ids=confidence_ids,
        document_ids=document_ids,
        atol=atol,
        rtol=rtol,
    )
    receipt["reloaded_safetensors"] = str(weights_path)
    return receipt


def _source_float32(tensor: np.ndarray) -> np.ndarray:
    if tensor.dtype == np.uint16:
        return (tensor.astype(np.uint32) << 16).view(np.float32)
    if tensor.dtype == np.float32:
        return tensor
    raise TypeError(f"expected floating source tensor, got {tensor.dtype}")


def _numpy_rms_norm(
    hidden: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    variance = np.mean(np.square(hidden), axis=-1, keepdims=True)
    return hidden * np.reciprocal(np.sqrt(variance + np.float32(1e-5))) * weight


def _numpy_rope(tensor: np.ndarray, *, theta: float = 10_000.0) -> np.ndarray:
    head_dim = tensor.shape[-1]
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(tensor.shape[-2], dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    cos = np.cos(freqs, dtype=np.float32)[None, None, :, :]
    sin = np.sin(freqs, dtype=np.float32)[None, None, :, :]
    first, second = tensor[..., :half], tensor[..., half:]
    return np.concatenate(
        (first * cos - second * sin, second * cos + first * sin),
        axis=-1,
    )


def _numpy_ngram_embedding(
    source: dict[str, np.ndarray],
    input_ids: np.ndarray,
) -> np.ndarray:
    prefix = "embedding.cppmega_ngram_hash."
    offsets = source[prefix + "table_offsets"].astype(np.int64, copy=False)
    table_sizes = source[prefix + "table_sizes_t"].astype(np.int64, copy=False)
    mults = source[prefix + "hash_mults"].astype(np.int64, copy=False)
    bias = source[prefix + "hash_bias"].astype(np.int64, copy=False)
    order_mask = source[prefix + "order_mask"].astype(np.int64, copy=False)
    batch, seq = input_ids.shape
    shifted = np.zeros((mults.shape[1], batch, seq), dtype=np.int64)
    shifted[0] = input_ids
    for position in range(1, shifted.shape[0]):
        shifted[position, :, position:] = input_ids[:, :-position]
    with np.errstate(over="ignore"):
        product = (
            mults.T[:, :, None, None]
            * shifted[:, None, :, :]
            * order_mask[:, :, None, None]
        )
    hashed = product[0]
    for position in range(1, product.shape[0]):
        hashed = np.bitwise_xor(hashed, product[position])
    hashed = np.bitwise_xor(hashed, bias[:, None, None])
    hashed = np.remainder(hashed, table_sizes[:, None, None])
    hashed = hashed + offsets[:, None, None]
    indices = hashed.transpose(1, 0, 2)
    table = _source_float32(source[prefix + "unified_table.weight"])
    embedded = table[indices].transpose(0, 2, 1, 3)
    embedded = embedded.reshape(batch, seq, -1)
    projection = _source_float32(source[prefix + "out_proj.weight"])
    return embedded @ projection.T


def _numpy_structure_embedding(
    source: dict[str, np.ndarray],
    cfg: Dense500MConversionConfig,
    structure_ids: np.ndarray,
    dep_levels: np.ndarray,
) -> np.ndarray:
    prefix = "embedding.cppmega_structure."
    table = _source_float32(source[prefix + "stacked_emb.weight"])
    projection = _source_float32(source[prefix + "up_proj.weight"])
    scales = _source_float32(source[prefix + "component_scales"])
    if cfg.structure_components != "core" or scales.shape != (2,):
        raise ValueError(
            "NumPy source parity supports exactly the core structure components"
        )
    ids = np.stack(
        (structure_ids, dep_levels + cfg.structure_num_categories),
        axis=-1,
    )
    weighted = np.sum(table[ids] * scales[None, None, :, None], axis=2)
    return weighted @ projection.T


def _numpy_domain_embedding(
    source: dict[str, np.ndarray],
    cfg: Dense500MConversionConfig,
    domain_ids: np.ndarray,
    role_ids: np.ndarray,
    confidence_ids: np.ndarray,
) -> np.ndarray:
    prefix = "embedding.cppmega_domain."
    table = _source_float32(source[prefix + "stacked_emb.weight"])
    projection = _source_float32(source[prefix + "up_proj.weight"])
    scales = _source_float32(source[prefix + "component_scales"])
    ids = np.stack(
        (
            domain_ids,
            role_ids + cfg.domain_num_domains,
            confidence_ids + cfg.domain_num_domains + cfg.domain_num_roles,
        ),
        axis=-1,
    )
    weighted = np.sum(table[ids] * scales[None, None, :, None], axis=2)
    return weighted @ projection.T


def megatron_dcp_reference_logits(
    source: dict[str, np.ndarray],
    cfg: Dense500MConversionConfig,
    *,
    input_ids: np.ndarray,
    graph_attention_bias: np.ndarray,
    graph_edge_kind_bias: np.ndarray,
    structure_ids: np.ndarray,
    dep_levels: np.ndarray,
    domain_ids: np.ndarray,
    role_ids: np.ndarray,
    confidence_ids: np.ndarray,
) -> np.ndarray:
    """Independent Megatron AF/GQA inference over raw DCP tensor names."""

    token_weight = _source_float32(source["embedding.word_embeddings.weight"])
    hidden = token_weight[input_ids]
    hidden = hidden + _numpy_ngram_embedding(source, input_ids)
    hidden = hidden + _numpy_structure_embedding(
        source,
        cfg,
        structure_ids,
        dep_levels,
    )
    if set(DOMAIN_SOURCE_TO_TARGET) <= set(source):
        hidden = hidden + _numpy_domain_embedding(
            source,
            cfg,
            domain_ids,
            role_ids,
            confidence_ids,
        )

    batch, seq = input_ids.shape
    heads_per_group = cfg.num_query_heads // cfg.num_kv_heads
    q_rows_per_group = heads_per_group * cfg.head_dim
    rows_per_group = q_rows_per_group + 2 * cfg.head_dim
    graph_bias = graph_attention_bias + graph_edge_kind_bias
    causal = np.arange(seq)[:, None] >= np.arange(seq)[None, :]
    for block in range(cfg.depth):
        attn_layer = block * 2
        mlp_layer = attn_layer + 1
        attn_prefix = f"decoder.layers.{attn_layer}.self_attention."
        norm = _numpy_rms_norm(
            hidden,
            _source_float32(source[attn_prefix + "linear_qkv.layer_norm_weight"]),
        )
        fused = norm @ _source_float32(source[attn_prefix + "linear_qkv.weight"]).T
        grouped = fused.reshape(batch, seq, cfg.num_kv_heads, rows_per_group)
        query = grouped[..., :q_rows_per_group].reshape(
            batch, seq, cfg.num_query_heads, cfg.head_dim
        )
        key = grouped[..., q_rows_per_group : q_rows_per_group + cfg.head_dim]
        value = grouped[..., q_rows_per_group + cfg.head_dim :]
        query = _numpy_rope(query.transpose(0, 2, 1, 3))
        key = _numpy_rope(key.transpose(0, 2, 1, 3))
        value = value.transpose(0, 2, 1, 3)
        key = np.repeat(key, heads_per_group, axis=1)
        value = np.repeat(value, heads_per_group, axis=1)
        scores = np.matmul(query, key.transpose(0, 1, 3, 2))
        scores *= np.float32(cfg.head_dim**-0.5)
        scores = scores + graph_bias[:, None, :, :]
        scores = np.where(causal[None, None, :, :], scores, -np.inf)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores, dtype=np.float32)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        attention = np.matmul(probabilities, value)
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, seq, cfg.q_proj_dim)
        hidden = (
            hidden
            + attention @ _source_float32(source[attn_prefix + "linear_proj.weight"]).T
        )

        mlp_prefix = f"decoder.layers.{mlp_layer}.mlp."
        norm = _numpy_rms_norm(
            hidden,
            _source_float32(source[mlp_prefix + "linear_fc1.layer_norm_weight"]),
        )
        gate_up = norm @ _source_float32(source[mlp_prefix + "linear_fc1.weight"]).T
        gate, up = np.split(gate_up, 2, axis=-1)
        silu = gate / (1.0 + np.exp(-gate, dtype=np.float32))
        hidden = (
            hidden
            + (silu * up) @ _source_float32(source[mlp_prefix + "linear_fc2.weight"]).T
        )

    hidden = _numpy_rms_norm(
        hidden,
        _source_float32(source["decoder.final_norm.weight"]),
    )
    return hidden @ token_weight.T


def _fixed_dcp_parity_inputs(cfg: Dense500MConversionConfig) -> dict[str, Any]:
    mx, _tree_flatten, _DenseCppLM, _DenseCppLMConfig = _import_runtime()
    seq = min(8, cfg.max_seq_length)
    if seq < 4:
        raise ValueError(
            f"conversion logit parity requires max_seq_length >= 4, got {seq}"
        )
    tokens = (np.arange(seq, dtype=np.int32) % max(cfg.vocab_size - 1, 1)) + 1
    graph = np.zeros((1, seq, seq), dtype=np.float32)
    graph[0, seq - 1, 0] = 1.25
    edge_kind = np.zeros_like(graph)
    edge_kind[0, seq - 2, 1] = 0.5
    return {
        "input_ids": mx.array(tokens[None, :], dtype=mx.int32),
        "graph_attention_bias": mx.array(graph),
        "graph_edge_kind_bias": mx.array(edge_kind),
        "structure_ids": mx.array(
            (np.arange(seq, dtype=np.int32) % cfg.structure_num_categories)[None, :]
        ),
        "dep_levels": mx.array(
            (np.arange(seq, dtype=np.int32) % cfg.structure_max_dep_level)[None, :]
        ),
        "domain_ids": mx.ones((1, seq), dtype=mx.int32),
        "role_ids": mx.full((1, seq), 2, dtype=mx.int32),
        "confidence_ids": mx.ones((1, seq), dtype=mx.int32),
        # Megatron's n-gram module is not document-boundary aware. A single
        # document isolates checkpoint mapping/forward parity without claiming
        # parity for MLX's stricter packed-document behavior.
        "document_ids": mx.zeros((1, seq), dtype=mx.int32),
    }


def verify_emitted_dcp_logit_parity(
    *,
    source: dict[str, np.ndarray],
    cfg: Dense500MConversionConfig,
    target_model_factory: Any,
    weights_path: Path,
    parity_inputs: dict[str, Any],
    atol: float = 4e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Compare raw DCP NumPy execution to reloaded MLX safetensors."""

    mx, _tree_flatten, _DenseCppLM, _DenseCppLMConfig = _import_runtime()
    target_model = target_model_factory()
    target_model.load_weights(str(weights_path), strict=True)
    target_model.set_dtype(mx.float32)
    mx.eval(target_model.parameters())
    numpy_inputs = {
        key: np.asarray(value)
        for key, value in parity_inputs.items()
        if key != "document_ids"
    }
    source_logits = megatron_dcp_reference_logits(source, cfg, **numpy_inputs)
    target_logits = target_model.logits(
        parity_inputs["input_ids"],
        block_bias=parity_inputs["graph_attention_bias"],
        edge_kind_bias=parity_inputs["graph_edge_kind_bias"],
        structure_ids=parity_inputs["structure_ids"],
        dep_levels=parity_inputs["dep_levels"],
        domain_ids=parity_inputs["domain_ids"],
        role_ids=parity_inputs["role_ids"],
        confidence_ids=parity_inputs["confidence_ids"],
        document_ids=parity_inputs["document_ids"],
    )
    mx.eval(target_logits)
    target = np.asarray(target_logits, dtype=np.float32)
    error = np.abs(source_logits - target)
    tolerance = atol + rtol * np.abs(source_logits)
    if np.any(error > tolerance):
        index = np.unravel_index(int(np.argmax(error)), error.shape)
        raise AssertionError(
            "raw DCP -> MLX logit parity failed: "
            f"max_abs_error={float(error[index]):.9g} at={index} "
            f"source={float(source_logits[index]):.9g} "
            f"target={float(target[index]):.9g} atol={atol} rtol={rtol}"
        )
    return {
        "source_reference": "raw_dcp_numpy_megatron_af_gqa_v1",
        "compute_dtype": "float32",
        "reloaded_safetensors": str(weights_path.resolve()),
        "max_abs_logit_error": float(error.max(initial=0.0)),
        "mean_abs_logit_error": float(np.mean(error)),
        "p99_abs_logit_error": float(np.quantile(error, 0.99)),
        "rms_logit_error": float(np.sqrt(np.mean(np.square(error)))),
        "atol": atol,
        "rtol": rtol,
        "input_shape": list(parity_inputs["input_ids"].shape),
        "graph_route_nonzero": int(
            np.count_nonzero(numpy_inputs["graph_attention_bias"])
        ),
        "edge_kind_nonzero": int(
            np.count_nonzero(numpy_inputs["graph_edge_kind_bias"])
        ),
        "structure_tokens": int(np.size(numpy_inputs["structure_ids"])),
        "packed_document_parity": "not_claimed_source_ngram_has_no_boundary_input",
    }


def convert_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    cfg: Dense500MConversionConfig,
    bf16: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    _require_supported_attention_mode(cfg)
    output, manifest_path = conversion_output_paths(output)
    mx, _tree_flatten, _DenseCppLM, _DenseCppLMConfig = _import_runtime()
    iter_dir = resolve_checkpoint_iter(checkpoint)
    metadata = read_dcp_metadata(iter_dir)
    source_specs = _metadata_tensor_specs(metadata)
    source_contract = validate_source_checkpoint_contract(iter_dir, cfg, metadata)
    domain_tensors_present = set(DOMAIN_SOURCE_TO_TARGET) <= set(source_specs)
    target = _target_arrays(
        cfg,
        bf16=bf16,
        domain_tensors_present=domain_tensors_present,
    )
    validation = validate_plan_against_metadata(cfg, metadata, target)
    manifest = {
        "schema": "cppmega_megatron_dense500m_to_mlx_v4",
        "source_checkpoint": str(iter_dir),
        "source_artifacts": source_artifact_hashes(iter_dir),
        "source_contract": source_contract,
        "output": str(output),
        "config": asdict(cfg),
        "dtype": "bfloat16" if bf16 else "float32",
        "validation": validation,
        "runtime_requirements": conversion_runtime_requirements(
            domain_tensors_present=domain_tensors_present,
            source_keys=set(source_specs),
        ),
        "notes": [
            "Megatron AF alternating layer positions are folded into DenseCppLM blocks",
            "position_embedding.weight is zero because source checkpoint used RoPE-only positions",
            "platform_embedding.embedding.weight is zero because source checkpoint has doc-level platform sidecars, not a learned platform table",
            "optimizer.* DCP tensors are explicitly excluded from inference conversion",
            "fixed-input logits are compared from raw DCP tensors through an independent NumPy Megatron AF/GQA forward against a fresh MLX model reloaded from emitted safetensors",
            (
                "domain learned tensors were present and mapped; production domain residual is required"
                if domain_tensors_present
                else "source checkpoint had no domain learned tensors; manifest disables the domain residual"
            ),
        ],
    }
    if dry_run:
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    source = load_required_source_tensors(iter_dir, cfg, metadata)
    target = apply_mapping(target, source, cfg, bf16=bf16)
    parity_inputs = _fixed_dcp_parity_inputs(cfg)

    def target_model_factory() -> Any:
        return _make_target_model(
            cfg,
            bf16=bf16,
            domain_tensors_present=domain_tensors_present,
        )[1]

    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}-conversion-",
        dir=output.parent,
    ) as tmp:
        stage_dir = Path(tmp)
        staged_weights = stage_dir / output.name
        staged_manifest = stage_dir / "model.json"
        mx.save_safetensors(str(staged_weights), target, metadata={"format": "mlx"})
        parity = verify_emitted_dcp_logit_parity(
            source=source,
            cfg=cfg,
            target_model_factory=target_model_factory,
            weights_path=staged_weights,
            parity_inputs=parity_inputs,
        )
        parity["reloaded_safetensors"] = str(output)
        parity["domain_tensors_present"] = domain_tensors_present
        manifest["logit_parity"] = parity
        manifest["output_artifacts"] = [_sha256_receipt(staged_weights)]
        manifest["publish"] = {
            "complete": True,
            "completion_marker": manifest_path.name,
            "weights_sha256": manifest["output_artifacts"][0]["sha256"],
        }
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Publish order is the crash contract: the weights are renamed into
        # place and fsynced first; model.json records the weights SHA-256 and
        # is renamed last, so its presence marks the pair as complete.
        _fsync_file(staged_weights)
        os.replace(staged_weights, output)
        _fsync_file(staged_manifest)
        os.replace(staged_manifest, manifest_path)
        _fsync_dir(output.parent)
    return manifest


def published_checkpoint_status(output: Path) -> dict[str, Any]:
    """Report whether the published weights + model.json pair is complete.

    model.json is published last and records the safetensors SHA-256, so a
    crash before the manifest rename leaves weights without the completion
    marker and is reported incomplete here.
    """

    weights_path, manifest_path = conversion_output_paths(output)
    if not manifest_path.is_file():
        return {
            "complete": False,
            "reason": f"missing completion marker {manifest_path.name}",
        }
    if not weights_path.is_file():
        return {
            "complete": False,
            "reason": f"missing weights {weights_path.name}",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publish = manifest.get("publish")
    expected = publish.get("weights_sha256") if isinstance(publish, dict) else None
    if not expected:
        return {
            "complete": False,
            "reason": "manifest lacks publish.weights_sha256",
        }
    actual = _sha256_receipt(weights_path)["sha256"]
    if actual != expected:
        return {
            "complete": False,
            "reason": "weights sha256 mismatch",
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    return {"complete": True, "weights_sha256": actual}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument(
        "--attention-mode",
        choices=("gqa", "dsa"),
        default="gqa",
        help="DSA is accepted for explicit fail-closed rejection only",
    )
    ap.add_argument("--fp32", action="store_true", help="write fp32 instead of bf16")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Dense500MConversionConfig(
        max_seq_length=args.seq_len,
        attention_mode=args.attention_mode,
    )
    manifest = convert_checkpoint(
        args.checkpoint,
        args.output,
        cfg=cfg,
        bf16=not args.fp32,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
