from __future__ import annotations

from dataclasses import replace
import json
import os
import subprocess
import sys
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from numpy.typing import DTypeLike

from cppmega_mlx.data.batch import LMTokenBatch
from cppmega_mlx.data.megatron_indexed import (
    MegatronIndexedDataset,
    MegatronIndexedMultiShardDataset,
    megatron_indexed_graph_sidecar_schema,
    megatron_indexed_side_channel_schema,
    open_megatron_indexed_dataset,
)
from cppmega_mlx.data.token_dataset import open_token_dataset
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.compiled import CompiledPretrainingStep
from cppmega_mlx.training.compiled import normalize_compiled_batch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_HYBRID_TINY = ROOT / "scripts" / "train_hybrid_tiny.py"
STRUCTURE_MODEL_KWARG_KEYS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)
CONTRACT_SUBPROCESS_TIMEOUT_SECONDS = 60
BACKEND_SMOKE_TIMEOUT_SECONDS = 180

_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {
    np.dtype(np.uint16): 8,
    np.dtype(np.int32): 4,
    np.dtype(np.int64): 5,
}


def _write_mmididx(
    prefix: Path,
    docs: list[np.ndarray],
    *,
    dtype: DTypeLike,
    sequence_modes: bool = False,
) -> None:
    dtype = np.dtype(dtype)
    flat = np.concatenate([doc.astype(dtype, copy=False) for doc in docs])
    flat.tofile(prefix.with_suffix(".bin"))

    lengths = np.asarray([len(doc) for doc in docs], dtype=np.int32)
    pointers = np.empty(len(docs), dtype=np.int64)
    offset = 0
    for i, length in enumerate(lengths):
        pointers[i] = offset
        offset += int(length) * dtype.itemsize
    documents = np.arange(len(docs) + 1, dtype=np.int64)

    with prefix.with_suffix(".idx").open("wb") as fh:
        fh.write(_HEADER)
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<B", _DTYPE_CODES[dtype]))
        fh.write(struct.pack("<Q", len(docs)))
        fh.write(struct.pack("<Q", len(documents)))
        lengths.tofile(fh)
        pointers.tofile(fh)
        documents.tofile(fh)
        if sequence_modes:
            np.zeros(len(docs), dtype=np.int8).tofile(fh)


def _write_graph_sidecars(
    prefix: Path,
    *,
    call_edges: list[np.ndarray],
    type_edges: list[np.ndarray],
    chunk_starts: list[np.ndarray],
    chunk_ends: list[np.ndarray],
    chunk_kinds: list[np.ndarray],
    chunk_dep_levels: list[np.ndarray],
) -> dict[str, object]:
    def write_edge(name: str, rows: list[np.ndarray]) -> dict[str, object]:
        offsets = [0]
        flattened = []
        total = 0
        for row in rows:
            arr = np.asarray(row, dtype=np.int32).reshape(-1, 2)
            flattened.append(arr)
            total += int(arr.shape[0])
            offsets.append(total)
        data = np.concatenate(flattened, axis=0) if flattened else np.zeros((0, 2), dtype=np.int32)
        offsets_path = prefix.with_name(f"{prefix.name}_{name}_offsets.bin")
        data_path = prefix.with_name(f"{prefix.name}_{name}_data.bin")
        np.asarray(offsets, dtype=np.int64).tofile(offsets_path)
        data.astype(np.int32, copy=False).tofile(data_path)
        return {
            "kind": "edge_pairs",
            "offsets_path": offsets_path.name,
            "data_path": data_path.name,
            "offset_dtype": "int64",
            "dtype": "int32",
            "item_count": total,
            "shape_tail": [2],
        }

    def write_vector(name: str, rows: list[np.ndarray], dtype: DTypeLike) -> dict[str, object]:
        dtype = np.dtype(dtype)
        offsets = [0]
        flattened = []
        total = 0
        for row in rows:
            arr = np.asarray(row, dtype=dtype).reshape(-1)
            flattened.append(arr)
            total += int(arr.shape[0])
            offsets.append(total)
        data = np.concatenate(flattened, axis=0) if flattened else np.zeros((0,), dtype=dtype)
        offsets_path = prefix.with_name(f"{prefix.name}_{name}_offsets.bin")
        data_path = prefix.with_name(f"{prefix.name}_{name}_data.bin")
        np.asarray(offsets, dtype=np.int64).tofile(offsets_path)
        data.tofile(data_path)
        return {
            "kind": "ragged_1d",
            "offsets_path": offsets_path.name,
            "data_path": data_path.name,
            "offset_dtype": "int64",
            "dtype": dtype.name,
            "item_count": total,
        }

    return {
        "graph_sidecar_schema": "cppmega_graph_routes_v1",
        "graph_sidecar_paths": {
            "token_call_edges": write_edge("token_call_edges", call_edges),
            "token_type_edges": write_edge("token_type_edges", type_edges),
            "token_chunk_starts": write_vector("token_chunk_starts", chunk_starts, np.uint32),
            "token_chunk_ends": write_vector("token_chunk_ends", chunk_ends, np.uint32),
            "token_chunk_kinds": write_vector("token_chunk_kinds", chunk_kinds, np.uint16),
            "token_chunk_dep_levels": write_vector("token_chunk_dep_levels", chunk_dep_levels, np.uint16),
        },
    }


def _batch_tokens(dataset: MegatronIndexedDataset) -> np.ndarray:
    return np.array(next(dataset.iter_batches()).tokens)


def _write_indexed_train_smoke_fixture(prefix: Path) -> None:
    docs = [
        (np.arange(8, dtype=np.int32) % 32),
        ((np.arange(8, dtype=np.int32) + 8) % 32),
    ]
    _write_mmididx(prefix, docs, dtype=np.int32)
    flat = np.concatenate(docs)
    structure_ids = (flat % 7).astype(np.int16)
    dep_levels = (flat % 3).astype(np.uint8)
    ast_depth_ids = (flat % 5).astype(np.uint8)
    sibling_index_ids = (flat % 4).astype(np.uint16)
    node_type_ids = (flat % 11).astype(np.int32)
    attention_mask = np.ones(flat.shape, dtype=np.float32)
    structure_ids.tofile(prefix.with_name("structure_ids.bin"))
    dep_levels.tofile(prefix.with_name("dep_levels.bin"))
    ast_depth_ids.tofile(prefix.with_name("ast_depth_ids.bin"))
    sibling_index_ids.tofile(prefix.with_name("sibling_index_ids.bin"))
    node_type_ids.tofile(prefix.with_name("node_type_ids.bin"))
    attention_mask.tofile(prefix.with_name("attention_mask.bin"))
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "vocab_size": 32,
                "tokenizer_contract": "local_profile",
                "source_format": "megatron-indexed-test",
                "side_channel_paths": {
                    "structure_ids": {
                        "path": "structure_ids.bin",
                        "dtype": "int16",
                    },
                    "dep_levels": {
                        "path": "dep_levels.bin",
                        "dtype": "uint8",
                    },
                    "ast_depth_ids": {
                        "path": "ast_depth_ids.bin",
                        "dtype": "uint8",
                    },
                    "sibling_index_ids": {
                        "path": "sibling_index_ids.bin",
                        "dtype": "uint16",
                    },
                    "node_type_ids": {
                        "path": "node_type_ids.bin",
                        "dtype": "int32",
                    },
                    "attention_mask": "attention_mask.bin",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_structured_multishard_fixture(
    root: Path,
    *,
    shard_docs: list[list[np.ndarray]],
    with_structure: bool = True,
    with_document_ids: bool = True,
) -> list[Path]:
    prefixes: list[Path] = []
    next_doc_id = 0
    for shard_index, docs in enumerate(shard_docs):
        prefix = root / f"shard_{shard_index:05d}"
        prefixes.append(prefix)
        _write_mmididx(prefix, docs, dtype=np.int32)
        flat = np.concatenate(docs)
        side_channel_paths: dict[str, object] = {}
        if with_structure:
            structure_path = prefix.with_name(f"{prefix.name}_structure_ids.bin")
            dep_path = prefix.with_name(f"{prefix.name}_dep_levels.bin")
            (flat % 7).astype(np.int16).tofile(structure_path)
            (flat % 3).astype(np.uint8).tofile(dep_path)
            side_channel_paths = {
                "structure_ids": {
                    "path": structure_path.name,
                    "dtype": "int16",
                },
                "dep_levels": {
                    "path": dep_path.name,
                    "dtype": "uint8",
                },
            }
        sidecar: dict[str, object] = {
            "vocab_size": 256,
            "tokenizer_contract": "local_profile",
            "source_format": "megatron-indexed-test",
        }
        if side_channel_paths:
            sidecar["side_channel_paths"] = side_channel_paths
        if with_document_ids:
            doc_path = prefix.with_name(f"{prefix.name}_doc_ids.bin")
            doc_ids = []
            for doc in docs:
                doc_ids.extend([next_doc_id] * len(doc))
                next_doc_id += 1
            np.asarray(doc_ids, dtype=np.int16).tofile(doc_path)
            sidecar["doc_ids"] = {
                "path": doc_path.name,
                "dtype": "int16",
            }
        prefix.with_suffix(".idx.json").write_text(
            json.dumps(sidecar),
            encoding="utf-8",
        )
    return prefixes


def _run_train_hybrid_tiny(
    *args: str,
    timeout_seconds: int,
    kernel_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if kernel_path is not None:
        env["CPPMEGA_KERNEL_PATH"] = kernel_path
    try:
        return subprocess.run(
            [sys.executable, str(TRAIN_HYBRID_TINY), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "train_hybrid_tiny subprocess timed out after "
            f"{timeout_seconds}s; stdout={exc.stdout!r}; stderr={exc.stderr!r}"
        ) from exc


def _load_script_json(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def test_mmididx_int32_reads_fixed_windows_without_crossing_sequences(tmp_path) -> None:
    prefix = tmp_path / "clang_semantic_4k_v10_train"
    _write_mmididx(
        prefix,
        [
            np.arange(10, dtype=np.int32),
            np.arange(100, 112, dtype=np.int32),
        ],
        dtype=np.int32,
    )

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=2)
    batch = _batch_tokens(dataset)

    assert dataset.num_samples == 5
    assert dataset.num_batches == 2
    assert dataset.dropped_samples == 1
    assert dataset.index_metadata.dtype == "int32"
    assert dataset.index_metadata.metadata_path is None
    assert dataset.index_metadata.sequence_count == 2
    assert dataset.index_metadata.document_count == 2
    np.testing.assert_array_equal(
        batch,
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.array(next(dataset.iter_batches()).inputs[0]), [0, 1, 2])
    np.testing.assert_array_equal(np.array(next(dataset.iter_batches()).targets[0]), [1, 2, 3])


def test_accepts_prefix_bin_idx_and_optional_sequence_modes(tmp_path) -> None:
    prefix = tmp_path / "tokens"
    _write_mmididx(
        prefix,
        [np.arange(8, dtype=np.uint16), np.arange(10, 18, dtype=np.uint16)],
        dtype=np.uint16,
        sequence_modes=True,
    )

    by_prefix = MegatronIndexedDataset(prefix, seq_len=4, batch_size=2)
    by_bin = MegatronIndexedDataset(prefix.with_suffix(".bin"), seq_len=4, batch_size=2)
    by_idx = MegatronIndexedDataset(prefix.with_suffix(".idx"), seq_len=4, batch_size=2)

    np.testing.assert_array_equal(_batch_tokens(by_prefix), _batch_tokens(by_bin))
    np.testing.assert_array_equal(_batch_tokens(by_prefix), _batch_tokens(by_idx))


def test_open_token_dataset_routes_explicit_megatron_format_to_indexed_reader(
    tmp_path,
) -> None:
    prefix = tmp_path / "explicit_megatron"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)

    dataset = open_token_dataset(
        prefix,
        seq_len=4,
        batch_size=2,
        format="megatron",
    )

    assert isinstance(dataset, MegatronIndexedDataset)
    assert dataset.index_metadata.source_format == "megatron"
    np.testing.assert_array_equal(
        np.array(next(dataset.iter_batches()).tokens),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
    )


def test_open_megatron_indexed_dataset_is_standalone_training_ingress(tmp_path) -> None:
    prefix = tmp_path / "standalone_megatron"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)

    dataset = open_megatron_indexed_dataset(prefix, seq_len=4, batch_size=2)

    assert isinstance(dataset, MegatronIndexedDataset)
    assert dataset.index_metadata.source_format == "megatron"
    np.testing.assert_array_equal(
        _batch_tokens(dataset),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
    )


def test_megatron_indexed_fixture_has_deterministic_training_contract(
    tmp_path,
) -> None:
    prefix = tmp_path / "clang_semantic_4k_v10_train"
    _write_indexed_train_smoke_fixture(prefix)

    dataset = open_token_dataset(
        prefix,
        seq_len=4,
        batch_size=1,
        format="megatron",
    )
    batch = next(dataset.iter_batches())

    assert isinstance(dataset, MegatronIndexedDataset)
    assert dataset.metadata.vocab_size == 32
    assert dataset.metadata.tokenizer_contract == "local_profile"
    assert dataset.metadata.source_format == "megatron-indexed-test"
    assert sorted(getattr(dataset, "_side_channels")) == [
        "ast_depth_ids",
        "attention_mask",
        "dep_levels",
        "node_type_ids",
        "sibling_index_ids",
        "structure_ids",
    ]
    np.testing.assert_array_equal(np.array(batch.tokens), [[0, 1, 2, 3]])
    model_kwargs = batch.model_kwargs()
    assert tuple(model_kwargs) == STRUCTURE_MODEL_KWARG_KEYS
    for key in STRUCTURE_MODEL_KWARG_KEYS:
        assert tuple(model_kwargs[key].shape) == (1, 3)
    np.testing.assert_array_equal(np.array(model_kwargs["structure_ids"]), [[0, 1, 2]])
    np.testing.assert_array_equal(np.array(model_kwargs["dep_levels"]), [[0, 1, 2]])
    np.testing.assert_array_equal(np.array(model_kwargs["ast_depth_ids"]), [[0, 1, 2]])
    np.testing.assert_array_equal(np.array(model_kwargs["sibling_index_ids"]), [[0, 1, 2]])
    np.testing.assert_array_equal(np.array(model_kwargs["node_type_ids"]), [[0, 1, 2]])


def test_train_hybrid_tiny_reference_backend_smoke_reports_full_receipt(
    tmp_path,
) -> None:
    prefix = tmp_path / "clang_semantic_4k_v10_train"
    _write_indexed_train_smoke_fixture(prefix)

    result = _run_train_hybrid_tiny(
        str(prefix),
        "--json",
        "--data-format",
        "megatron",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--mamba-expand",
        "1",
        "--mamba-head-dim",
        "4",
        "--mamba-state-dim",
        "4",
        "--mamba-groups",
        "1",
        "--mamba-chunk-size",
        "4",
        timeout_seconds=BACKEND_SMOKE_TIMEOUT_SECONDS,
        kernel_path="reference",
    )
    payload = _load_script_json(result)

    assert payload["status"] == "ok"
    assert payload["synthetic_npz"] is False
    assert payload["dataset"]["path"] == str(prefix)
    assert payload["dataset"]["metadata"]["source_format"] == "megatron-indexed-test"
    assert payload["dataset"]["metadata"]["tokenizer_contract"] == "local_profile"
    receipt = payload["dataset"]["dataset_receipt"]
    assert receipt["source_format"] == "megatron-indexed-test"
    assert receipt["source_path"] == str(prefix)
    assert receipt["source_dataset_name"] == prefix.name
    assert receipt["index_metadata"] == {
        "bin_path": str(prefix.with_suffix(".bin")),
        "idx_path": str(prefix.with_suffix(".idx")),
        "metadata_path": str(prefix.with_suffix(".idx.json")),
        "dtype": "int32",
        "sequence_count": 2,
        "document_count": 2,
        "token_count": 16,
        "source_format": "megatron-indexed-test",
    }
    assert receipt["megatron_indexed_receipt"] == {
        "distributed_megatron_parity_claim": False,
        "gb10_training_correctness_claim": False,
        "ingress": "MegatronIndexedDataset",
        "local_only": True,
        "m4_vs_gb10_throughput_parity_claim": False,
        "megatron_runtime_imported": False,
        "path_accepts_suffixless_prefix": True,
        "receipt_scope": "local_mlx_training_ingress",
        "sidecar_schema": "explicit_token_aligned_binary_side_channel_paths",
    }
    assert payload["dataset"]["side_channels"] == [
        "ast_depth_ids",
        "attention_mask",
        "dep_levels",
        "node_type_ids",
        "sibling_index_ids",
        "structure_ids",
    ]
    assert payload["dataset"]["side_channel_contract"]["structure_side_channels"][
        "threaded_to_model"
    ] == [
        "ast_depth_ids",
        "dep_levels",
        "node_type_ids",
        "sibling_index_ids",
        "structure_ids",
    ]
    assert payload["dataset"]["side_channel_contract"]["structure_side_channels"][
        "model_kwarg_names"
    ] == list(STRUCTURE_MODEL_KWARG_KEYS)
    assert payload["route_symbols"] == "M"
    assert payload["route_roles"] == ["mamba3"]
    assert payload["tokens_per_step"] == 3
    assert payload["trained_tokens"] == 3
    assert payload["step_metrics"][0]["ntokens"] == 3
    assert payload["final_loss"] > 0


def test_train_script_megatron_format_validation_accepts_suffixless_prefix(
    tmp_path,
) -> None:
    prefix = tmp_path / "suffixless_megatron"
    _write_indexed_train_smoke_fixture(prefix)

    result = _run_train_hybrid_tiny(
        str(prefix),
        "--dry-run-json",
        "--data-format",
        "megatron",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "A",
        "--depth",
        "1",
        timeout_seconds=CONTRACT_SUBPROCESS_TIMEOUT_SECONDS,
    )
    payload = _load_script_json(result)

    assert payload["status"] == "dry_run"
    assert payload["dataset"]["path"] == str(prefix)
    assert payload["dataset"]["metadata"]["source_format"] == "megatron-indexed-test"
    receipt = payload["dataset"]["dataset_receipt"]
    assert receipt["source_path"] == str(prefix)
    assert receipt["index_metadata"]["metadata_path"] == str(
        prefix.with_suffix(".idx.json")
    )
    assert receipt["megatron_indexed_receipt"]["local_only"] is True
    assert receipt["megatron_indexed_receipt"]["megatron_runtime_imported"] is False
    assert (
        receipt["megatron_indexed_receipt"]["distributed_megatron_parity_claim"]
        is False
    )
    assert receipt["megatron_indexed_receipt"]["gb10_training_correctness_claim"] is False
    assert (
        receipt["megatron_indexed_receipt"][
            "m4_vs_gb10_throughput_parity_claim"
        ]
        is False
    )


def test_train_script_reports_missing_megatron_prefix_cleanly(tmp_path) -> None:
    missing_prefix = tmp_path / "missing_megatron"

    result = _run_train_hybrid_tiny(
        str(missing_prefix),
        "--json",
        "--data-format",
        "megatron",
        timeout_seconds=CONTRACT_SUBPROCESS_TIMEOUT_SECONDS,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert "token shard path does not exist" in payload["error"]


def test_open_token_dataset_infers_megatron_from_bin_and_idx_suffixes(tmp_path) -> None:
    prefix = tmp_path / "suffix_megatron"
    _write_mmididx(prefix, [np.arange(16, dtype=np.uint16)], dtype=np.uint16)

    by_bin = open_token_dataset(prefix.with_suffix(".bin"), seq_len=4, batch_size=2)
    by_idx = open_token_dataset(prefix.with_suffix(".idx"), seq_len=4, batch_size=2)

    assert isinstance(by_bin, MegatronIndexedDataset)
    assert isinstance(by_idx, MegatronIndexedDataset)
    np.testing.assert_array_equal(_batch_tokens(by_bin), _batch_tokens(by_idx))


def test_open_token_dataset_infers_megatron_from_suffixless_prefix(tmp_path) -> None:
    prefix = tmp_path / "clang_semantic_4k_v10_train"
    _write_mmididx(prefix, [np.arange(16, dtype=np.int32)], dtype=np.int32)
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "vocab_size": 131072,
                "tokenizer_contract": "custom",
                "source_format": "megatron-indexed-sidecar",
            }
        ),
        encoding="utf-8",
    )

    dataset = open_token_dataset(prefix, seq_len=4, batch_size=2)

    assert isinstance(dataset, MegatronIndexedDataset)
    assert dataset.path == prefix
    assert dataset.index_metadata.source_format == "megatron-indexed-sidecar"
    assert dataset.metadata.vocab_size == 131072
    assert dataset.metadata.tokenizer_contract == "custom"
    assert dataset.metadata.source_format == "megatron-indexed-sidecar"
    np.testing.assert_array_equal(
        _batch_tokens(dataset),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
    )


def test_open_token_dataset_accepts_multishard_directory_and_spans_shards(
    tmp_path,
) -> None:
    _write_structured_multishard_fixture(
        tmp_path,
        shard_docs=[
            [np.arange(8, dtype=np.int32)],
            [np.arange(100, 108, dtype=np.int32)],
        ],
    )

    dataset = open_token_dataset(tmp_path, seq_len=4, batch_size=3)
    batch = next(dataset.iter_batches())

    assert isinstance(dataset, MegatronIndexedMultiShardDataset)
    assert dataset.path == tmp_path
    assert dataset.metadata.source_format == "megatron-multishard"
    assert dataset.index_metadata.source_format == "megatron-multishard"
    assert dataset.index_metadata.shard_count == 2
    assert len(dataset.index_metadata.shards) == 2
    assert dataset.num_samples == 4
    assert dataset.num_batches == 1
    assert dataset.dropped_samples == 1
    assert dataset.token_id_range() == (0, 107)
    np.testing.assert_array_equal(
        np.array(batch.tokens),
        np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [100, 101, 102, 103],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        np.array(batch.document_ids),
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        np.array(batch.structure_ids),
        np.array(batch.tokens) % 7,
    )
    np.testing.assert_array_equal(
        np.array(batch.dep_levels),
        np.array(batch.tokens) % 3,
    )
    assert tuple(batch.model_kwargs()) == ("structure_ids", "dep_levels")
    np.testing.assert_array_equal(
        np.array(batch.model_kwargs()["structure_ids"]),
        np.array(batch.tokens[:, :-1]) % 7,
    )


def test_megatron_multishard_shuffle_resume_and_cursor_are_global(
    tmp_path,
) -> None:
    _write_structured_multishard_fixture(
        tmp_path,
        shard_docs=[
            [np.arange(12, dtype=np.int32)],
            [np.arange(100, 112, dtype=np.int32)],
        ],
        with_structure=False,
        with_document_ids=False,
    )

    left = open_megatron_indexed_dataset(
        tmp_path,
        seq_len=4,
        batch_size=2,
        shuffle=True,
        seed=123,
    )
    right = open_megatron_indexed_dataset(
        tmp_path,
        seq_len=4,
        batch_size=2,
        shuffle=True,
        seed=123,
    )
    other_seed = open_megatron_indexed_dataset(
        tmp_path,
        seq_len=4,
        batch_size=2,
        shuffle=True,
        seed=124,
    )

    assert isinstance(left, MegatronIndexedMultiShardDataset)
    left_batches = list(left.iter_batches())
    right_batches = list(right.iter_batches())
    assert [np.array(batch.tokens).tolist() for batch in left_batches] == [
        np.array(batch.tokens).tolist() for batch in right_batches
    ]
    assert left.sample_order(epoch=0).tolist() != other_seed.sample_order(
        epoch=0
    ).tolist()
    resumed = next(left.iter_batches(resume_batch=1))
    np.testing.assert_array_equal(
        np.array(resumed.tokens),
        np.array(left_batches[1].tokens),
    )
    cursor = left.cursor_after(3)
    assert cursor.epoch == 1
    assert cursor.batch_offset == 0
    assert cursor.global_batch_offset == 3


def test_megatron_multishard_schema_mismatch_fails_closed(tmp_path) -> None:
    _write_structured_multishard_fixture(
        tmp_path,
        shard_docs=[
            [np.arange(8, dtype=np.int32)],
            [np.arange(100, 108, dtype=np.int32)],
        ],
    )
    second_sidecar = tmp_path / "shard_00001.idx.json"
    payload = json.loads(second_sidecar.read_text(encoding="utf-8"))
    payload.pop("side_channel_paths")
    second_sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="multi-shard schema mismatch"):
        open_megatron_indexed_dataset(tmp_path, seq_len=4, batch_size=2)


def test_train_script_accepts_multishard_megatron_directory(tmp_path) -> None:
    _write_structured_multishard_fixture(
        tmp_path,
        shard_docs=[
            [np.arange(8, dtype=np.int32)],
            [np.arange(100, 108, dtype=np.int32)],
        ],
    )

    result = _run_train_hybrid_tiny(
        str(tmp_path),
        "--dry-run-json",
        "--data-format",
        "megatron",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--steps",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--pattern",
        "A",
        "--depth",
        "1",
        timeout_seconds=CONTRACT_SUBPROCESS_TIMEOUT_SECONDS,
    )
    payload = _load_script_json(result)

    assert payload["status"] == "dry_run"
    assert payload["dataset"]["path"] == str(tmp_path)
    assert payload["dataset"]["metadata"]["source_format"] == "megatron-multishard"
    receipt = payload["dataset"]["dataset_receipt"]
    assert receipt["source_format"] == "megatron-multishard"
    assert receipt["index_metadata"]["source_format"] == "megatron-multishard"
    assert receipt["index_metadata"]["shard_count"] == 2
    assert receipt["megatron_indexed_receipt"]["ingress"] == (
        "MegatronIndexedMultiShardDataset"
    )
    assert receipt["megatron_indexed_receipt"]["megatron_runtime_imported"] is False


def test_open_megatron_indexed_dataset_rejects_empty_multishard_directory(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="must contain .idx shards"):
        open_megatron_indexed_dataset(tmp_path, seq_len=4, batch_size=2)


def test_mmididx_int64_converts_safe_token_ids_to_int32(tmp_path) -> None:
    prefix = tmp_path / "int64_tokens"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int64)], dtype=np.int64)

    dataset = MegatronIndexedDataset(prefix, seq_len=6, batch_size=2)
    batch = _batch_tokens(dataset)

    assert batch.dtype == np.int32
    np.testing.assert_array_equal(
        batch,
        np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.int32),
    )


def test_raw_bin_uses_json_sidecar_dtype_token_count_and_metadata(tmp_path) -> None:
    prefix = tmp_path / "raw"
    np.arange(20, dtype=np.uint32).tofile(prefix.with_suffix(".bin"))
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "dtype": "uint32",
                "token_count": 10,
                "vocab_size": 131072,
                "tokenizer_contract": "custom",
                "source_format": "megatron-raw-sidecar",
            }
        ),
        encoding="utf-8",
    )

    dataset = MegatronIndexedDataset(prefix.with_suffix(".bin"), seq_len=5, batch_size=2)

    assert dataset.num_samples == 2
    assert dataset.metadata.vocab_size == 131072
    assert dataset.metadata.tokenizer_contract == "custom"
    assert dataset.metadata.source_format == "megatron-raw-sidecar"
    np.testing.assert_array_equal(
        _batch_tokens(dataset),
        np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32),
    )


def test_mmididx_uses_json_sidecar_vocab_tokenizer_and_source_metadata(
    tmp_path,
) -> None:
    prefix = tmp_path / "indexed_metadata"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "dtype": "int32",
                "vocab_size": 131072,
                "tokenizer_contract": "custom",
                "source_format": "megatron-indexed-sidecar",
            }
        ),
        encoding="utf-8",
    )

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=2)

    assert dataset.index_metadata.dtype == "int32"
    assert dataset.index_metadata.source_format == "megatron-indexed-sidecar"
    assert dataset.metadata.vocab_size == 131072
    assert dataset.metadata.tokenizer_contract == "custom"
    assert dataset.metadata.source_format == "megatron-indexed-sidecar"


def test_raw_bin_allows_explicit_dtype_without_sidecar(tmp_path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(12, dtype=np.uint16).tofile(path)

    dataset = MegatronIndexedDataset(path, seq_len=4, batch_size=3, dtype="uint16")

    np.testing.assert_array_equal(
        _batch_tokens(dataset),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int32),
    )


def test_shuffle_resume_and_cursor_match_fixed_dataset_semantics(tmp_path) -> None:
    path = tmp_path / "shuffle.bin"
    np.arange(60, dtype=np.uint16).tofile(path)

    left = MegatronIndexedDataset(path, seq_len=5, batch_size=2, dtype="uint16", shuffle=True, seed=123)
    right = MegatronIndexedDataset(path, seq_len=5, batch_size=2, dtype="uint16", shuffle=True, seed=123)
    other_seed = MegatronIndexedDataset(path, seq_len=5, batch_size=2, dtype="uint16", shuffle=True, seed=124)

    left_batches = list(left.iter_batches())
    right_batches = list(right.iter_batches())
    resumed = next(left.iter_batches(resume_batch=2))
    cursor = left.cursor_after(2)

    assert [np.array(b.tokens).tolist() for b in left_batches] == [
        np.array(b.tokens).tolist() for b in right_batches
    ]
    assert [np.array(b.tokens).tolist() for b in left_batches] != [
        np.array(b.tokens).tolist() for b in other_seed.iter_batches()
    ]
    np.testing.assert_array_equal(np.array(resumed.tokens), np.array(left_batches[2].tokens))
    assert cursor.epoch == 0
    assert cursor.batch_offset == 2
    assert cursor.global_batch_offset == 2


def test_cursor_after_includes_dataset_resume_batch_across_epoch_rollover(
    tmp_path,
) -> None:
    prefix = tmp_path / "resume_cursor"
    _write_mmididx(prefix, [np.arange(40, dtype=np.int32)], dtype=np.int32)

    dataset = MegatronIndexedDataset(
        prefix,
        seq_len=5,
        batch_size=2,
        shuffle=True,
        seed=7,
        loop=True,
        resume_batch=3,
    )
    stream = dataset.iter_batches()
    next(stream)
    next(stream)
    expected_next = next(stream)

    cursor = dataset.cursor_after(2)
    restored = MegatronIndexedDataset(
        prefix,
        seq_len=5,
        batch_size=2,
        shuffle=True,
        seed=7,
        loop=True,
    )
    actual_next = next(
        restored.iter_batches(
            resume_batch=cursor.batch_offset,
            epoch=cursor.epoch,
        )
    )

    assert cursor.epoch == 1
    assert cursor.batch_offset == 1
    assert cursor.global_batch_offset == 5
    np.testing.assert_array_equal(
        np.array(actual_next.tokens),
        np.array(expected_next.tokens),
    )


def test_mmididx_side_channel_paths_are_sliced_with_token_windows(tmp_path) -> None:
    prefix = tmp_path / "structured_side_channels"
    docs = [
        np.arange(8, dtype=np.int32),
        np.arange(100, 108, dtype=np.int32),
    ]
    _write_mmididx(prefix, docs, dtype=np.int32)
    flat = np.concatenate(docs)
    structure_ids = (flat % 7).astype(np.int16)
    dep_levels = (flat % 3).astype(np.uint8)
    attention_mask = np.ones(flat.shape, dtype=np.float32)
    structure_ids.tofile(tmp_path / "structure_ids.bin")
    dep_levels.tofile(tmp_path / "dep_levels.bin")
    attention_mask.tofile(tmp_path / "attention_mask.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "side_channel_paths": {
                    "structure_ids": {
                        "path": "structure_ids.bin",
                        "dtype": "int16",
                    },
                    "dep_levels": {
                        "path": "dep_levels.bin",
                        "dtype": "uint8",
                    },
                    "attention_mask": "attention_mask.bin",
                },
            }
        ),
        encoding="utf-8",
    )

    batch = next(MegatronIndexedDataset(prefix, seq_len=4, batch_size=2).iter_batches())

    assert batch.structure_ids is not None
    assert batch.dep_levels is not None
    assert batch.attention_mask is not None
    assert np.array(batch.structure_ids).dtype == np.int32
    assert np.array(batch.dep_levels).dtype == np.int32
    assert np.array(batch.attention_mask).dtype == np.float32
    np.testing.assert_array_equal(np.array(batch.tokens), [[0, 1, 2, 3], [4, 5, 6, 7]])
    np.testing.assert_array_equal(np.array(batch.structure_ids), np.array(batch.tokens) % 7)
    np.testing.assert_array_equal(np.array(batch.dep_levels), np.array(batch.tokens) % 3)
    np.testing.assert_array_equal(
        np.array(batch.model_kwargs()["structure_ids"]),
        np.array(batch.tokens[:, :-1]) % 7,
    )
    np.testing.assert_array_equal(np.array(batch.target_mask), np.ones((2, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "sidecar",
    [
        {
            "doc_ids": {
                "path": "doc_ids.bin",
                "dtype": "int16",
            },
        },
        {
            "side_channel_paths": {
                "packing_document_ids": {
                    "path": "doc_ids.bin",
                    "dtype": "int16",
                },
            },
        },
    ],
)
def test_mmididx_document_ids_are_sliced_to_lm_batch(tmp_path, sidecar) -> None:
    prefix = tmp_path / "document_id_sidecar"
    docs = [
        np.arange(8, dtype=np.int32),
        np.arange(100, 108, dtype=np.int32),
    ]
    _write_mmididx(prefix, docs, dtype=np.int32)
    doc_ids = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 7, 7, 7, 7, 8, 8, 8, 8],
        dtype=np.int16,
    )
    doc_ids.tofile(tmp_path / "doc_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )

    batch = next(MegatronIndexedDataset(prefix, seq_len=4, batch_size=2).iter_batches())

    assert batch.document_ids is not None
    np.testing.assert_array_equal(
        np.array(batch.document_ids),
        np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.input_document_ids),
        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32),
    )
    assert "document_ids" not in batch.model_kwargs()


def test_mmididx_top_level_side_channel_entry_is_supported(tmp_path) -> None:
    prefix = tmp_path / "top_level_structure"
    docs = [np.arange(12, dtype=np.int32)]
    _write_mmididx(prefix, docs, dtype=np.int32)
    flat = np.concatenate(docs)
    node_type_ids = (flat % 11).astype(np.uint16)
    node_type_ids.tofile(tmp_path / "node_type_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "node_type_ids": {
                    "path": "node_type_ids.bin",
                    "dtype": "uint16",
                },
            }
        ),
        encoding="utf-8",
    )

    batch = next(MegatronIndexedDataset(prefix, seq_len=6, batch_size=2).iter_batches())

    assert batch.node_type_ids is not None
    np.testing.assert_array_equal(np.array(batch.node_type_ids), np.array(batch.tokens) % 11)


def test_mmididx_stage3_metadata_only_sidecar_does_not_infer_side_channels(
    tmp_path,
) -> None:
    prefix = tmp_path / "stage3_token_only"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "source_format": "cppmega-stage3-token-only",
                "token_column": "token_ids",
                "parquet_columns": [
                    "token_ids",
                    "token_structure_ids",
                    "token_dep_levels",
                    "token_ast_depth",
                    "token_sibling_index",
                    "token_ast_node_type",
                ],
                "original_schema": {
                    "token_structure_ids": "large_list<uint8>",
                    "token_dep_levels": "large_list<uint16>",
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)
    batch = next(dataset.iter_batches())

    assert dataset.index_metadata.source_format == "cppmega-stage3-token-only"
    assert getattr(dataset, "_side_channels") == {}
    assert batch.model_kwargs() == {}
    assert batch.structure_ids is None
    assert batch.dep_levels is None


def test_mmididx_token_side_channel_aliases_are_normalized(tmp_path) -> None:
    prefix = tmp_path / "alias_structure"
    docs = [np.arange(8, dtype=np.int32)]
    _write_mmididx(prefix, docs, dtype=np.int32)
    flat = np.concatenate(docs)
    structure_ids = (flat % 7).astype(np.int16)
    dep_levels = (flat % 3).astype(np.uint8)
    ast_depth_ids = (flat % 5).astype(np.uint8)
    sibling_index_ids = (flat % 4).astype(np.uint16)
    node_type_ids = (flat % 11).astype(np.int32)
    structure_ids.tofile(tmp_path / "token_structure_ids.bin")
    dep_levels.tofile(tmp_path / "token_dep_levels.bin")
    ast_depth_ids.tofile(tmp_path / "token_ast_depth.bin")
    sibling_index_ids.tofile(tmp_path / "token_sibling_index.bin")
    node_type_ids.tofile(tmp_path / "token_ast_node_type.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "side_channel_paths": {
                    "token_structure_ids": {
                        "path": "token_structure_ids.bin",
                        "dtype": "int16",
                    },
                    "token_dep_levels": {
                        "path": "token_dep_levels.bin",
                        "dtype": "uint8",
                    },
                    "token_ast_depth": {
                        "path": "token_ast_depth.bin",
                        "dtype": "uint8",
                    },
                    "token_sibling_index": {
                        "path": "token_sibling_index.bin",
                        "dtype": "uint16",
                    },
                    "token_ast_node_type": {
                        "path": "token_ast_node_type.bin",
                        "dtype": "int32",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    batch = next(MegatronIndexedDataset(prefix, seq_len=4, batch_size=2).iter_batches())

    np.testing.assert_array_equal(np.array(batch.structure_ids), np.array(batch.tokens) % 7)
    np.testing.assert_array_equal(np.array(batch.dep_levels), np.array(batch.tokens) % 3)
    np.testing.assert_array_equal(np.array(batch.ast_depth_ids), np.array(batch.tokens) % 5)
    np.testing.assert_array_equal(np.array(batch.sibling_index_ids), np.array(batch.tokens) % 4)
    np.testing.assert_array_equal(np.array(batch.node_type_ids), np.array(batch.tokens) % 11)
    assert tuple(batch.model_kwargs()) == STRUCTURE_MODEL_KWARG_KEYS


def test_mmididx_full_cppmega_token_sidecars_are_grouped(tmp_path) -> None:
    prefix = tmp_path / "full_cppmega_sidecars"
    docs = [np.arange(8, dtype=np.int32)]
    _write_mmididx(prefix, docs, dtype=np.int32)
    flat = np.concatenate(docs)
    symbol_values = np.arange(8, dtype=np.uint64) + np.uint64((1 << 63) + 100)
    call_values = np.arange(8, dtype=np.uint64) + np.uint64((1 << 63) + 200)
    type_values = np.arange(8, dtype=np.uint64) + np.uint64((1 << 63) + 300)
    sidecar_specs = {
        "token_structure_ids": ((flat % 7).astype(np.uint8), "uint8"),
        "token_dep_levels": ((flat % 3).astype(np.uint16), "uint16"),
        "token_ast_depth": ((flat % 5).astype(np.uint16), "uint16"),
        "token_sibling_index": ((flat % 4).astype(np.uint16), "uint16"),
        "token_ast_node_type": ((flat % 11).astype(np.uint16), "uint16"),
        "token_platform_ids": ((flat % 13).astype(np.uint16), "uint16"),
        "token_symbol_ids": (symbol_values, "uint64"),
        "token_call_targets": (call_values, "uint64"),
        "token_type_refs": (type_values, "uint64"),
        "token_def_use": ((flat % 2).astype(np.uint8), "uint8"),
        "token_change_mask_pre": ((flat == 2).astype(np.uint8), "uint8"),
        "token_change_mask_post": ((flat == 3).astype(np.uint8), "uint8"),
    }
    side_channel_paths = {}
    for name, (values, dtype) in sidecar_specs.items():
        path = tmp_path / f"{name}.bin"
        values.tofile(path)
        side_channel_paths[name] = {"path": path.name, "dtype": dtype}
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "symbol_identity_schema_version": 3,
                "side_channel_paths": side_channel_paths,
            }
        ),
        encoding="utf-8",
    )

    batch = next(MegatronIndexedDataset(prefix, seq_len=4, batch_size=2).iter_batches())

    np.testing.assert_array_equal(np.array(batch.structure_ids), np.array(batch.tokens) % 7)
    np.testing.assert_array_equal(np.array(batch.platform_ids), np.array(batch.tokens) % 13)
    semantic = batch.side_channel_map()["semantic_graph"]
    temporal = batch.side_channel_map()["temporal_diff"]
    np.testing.assert_array_equal(
        np.array(semantic["symbol_ids"]),
        symbol_values.reshape(2, 4),
    )
    np.testing.assert_array_equal(
        np.array(semantic["call_targets"]),
        call_values.reshape(2, 4),
    )
    np.testing.assert_array_equal(
        np.array(semantic["type_refs"]),
        type_values.reshape(2, 4),
    )
    np.testing.assert_array_equal(
        np.array(semantic["def_use"]),
        np.array(batch.tokens) % 2,
    )
    assert int(np.array(temporal["change_mask_pre"]).sum()) == 1
    assert int(np.array(temporal["change_mask_post"]).sum()) == 1


def test_mmididx_graph_sidecars_are_sequence_aligned_routes(tmp_path) -> None:
    prefix = tmp_path / "graph_routes"
    _write_mmididx(
        prefix,
        [
            np.arange(4, dtype=np.int32),
            np.arange(10, 14, dtype=np.int32),
        ],
        dtype=np.int32,
    )
    sidecar = _write_graph_sidecars(
        prefix,
        call_edges=[
            np.array([[1, 0], [2, 1]], dtype=np.int32),
            np.array([[0, 1]], dtype=np.int32),
        ],
        type_edges=[
            np.array([[0, 2]], dtype=np.int32),
            np.array([], dtype=np.int32).reshape(0, 2),
        ],
        chunk_starts=[
            np.array([0, 1, 3], dtype=np.uint32),
            np.array([0, 2], dtype=np.uint32),
        ],
        chunk_ends=[
            np.array([1, 3, 4], dtype=np.uint32),
            np.array([2, 4], dtype=np.uint32),
        ],
        chunk_kinds=[
            np.array([5, 6, 7], dtype=np.uint16),
            np.array([8, 9], dtype=np.uint16),
        ],
        chunk_dep_levels=[
            np.array([0, 1, 2], dtype=np.uint16),
            np.array([0, 1], dtype=np.uint16),
        ],
    )
    prefix.with_suffix(".idx.json").write_text(json.dumps(sidecar), encoding="utf-8")

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)
    packet0 = dataset.graph_route_packet_for_sample(0)
    packet1 = dataset.graph_route_packet_for_sample(1)
    batch0 = next(dataset.iter_batches())

    assert packet0.num_chunks == 3
    assert packet0.graph.edge("call").to_pairs() == [(1, 0), (2, 1)]
    assert packet0.graph.edge("type").to_pairs() == [(0, 2)]
    np.testing.assert_array_equal(packet0.chunk_starts, np.array([0, 1, 3], dtype=np.int32))
    np.testing.assert_array_equal(packet0.chunk_ends, np.array([1, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(packet0.chunk_kinds, np.array([5, 6, 7], dtype=np.int32))
    np.testing.assert_array_equal(packet0.chunk_dep_levels, np.array([0, 1, 2], dtype=np.int32))
    assert packet1.graph.edge("call").to_pairs() == [(0, 1)]
    assert packet1.graph.edge("type").num_edges == 0
    assert batch0.graph_batch is not None
    assert batch0.graph_batch.graphs[0].edge("call").to_pairs() == [(1, 0), (2, 1)]
    np.testing.assert_array_equal(
        np.asarray(batch0.graph_batch.chunk_starts[0]),
        np.array([0, 1, 3], dtype=np.int32),
    )


def test_compact_fixed_rows_restore_padding_and_document_graphs(tmp_path) -> None:
    prefix = tmp_path / "compact_fixed_rows"
    docs = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([10, 11], dtype=np.int32),
    ]
    _write_mmididx(prefix, docs, dtype=np.int32)
    sidecar = _write_graph_sidecars(
        prefix,
        call_edges=[
            np.array([[0, 1]], dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
        ],
        type_edges=[
            np.zeros((0, 2), dtype=np.int32),
            np.array([[0, 0]], dtype=np.int32),
        ],
        chunk_starts=[np.array([0, 1]), np.array([0])],
        chunk_ends=[np.array([1, 3]), np.array([2])],
        chunk_kinds=[np.array([4, 5]), np.array([6])],
        chunk_dep_levels=[np.array([0, 1]), np.array([0])],
    )
    loss_mask = np.array([1, 1, 0, 1, 0], dtype=np.uint8)
    document_ids = np.array([1, 1, 1, 2, 2], dtype=np.uint16)
    hunk_ids = np.array([0, 0, -1, 2, -1], dtype=np.int32)
    loss_mask.tofile(tmp_path / "loss_mask.bin")
    document_ids.tofile(tmp_path / "doc_ids.bin")
    hunk_ids.tofile(tmp_path / "hunk_ids.bin")
    sidecar.update(
        {
            "token_count": 5,
            "source_capacity_token_count": 8,
            "document_count": 2,
            "side_channel_paths": {
                "loss_mask": {"path": "loss_mask.bin", "dtype": "uint8"},
                "doc_ids": {"path": "doc_ids.bin", "dtype": "uint16"},
                "hunk_id_per_token": {"path": "hunk_ids.bin", "dtype": "int32"},
            },
        }
    )
    prefix.with_suffix(".idx.json").write_text(json.dumps(sidecar), encoding="utf-8")

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=2)
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 2
    np.testing.assert_array_equal(
        np.array(batch.tokens),
        np.array([[1, 2, 3, 0], [10, 11, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.loss_mask),
        np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(batch.document_ids),
        np.array([[1, 1, 1, 0], [2, 2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(batch.side_channel_map()["temporal_diff"]["hunk_ids"]),
        np.array([[0, 0, -1, -1], [2, -1, -1, -1]], dtype=np.int32),
    )
    assert dataset.graph_route_packet_for_sample(0).graph.edge("call").to_pairs() == [
        (0, 1)
    ]
    assert dataset.graph_route_packet_for_sample(1).graph.edge("type").to_pairs() == [
        (0, 0)
    ]


def test_compact_fixed_rows_reject_inconsistent_capacity_metadata(tmp_path) -> None:
    prefix = tmp_path / "bad_compact_fixed_rows"
    _write_mmididx(prefix, [np.arange(3, dtype=np.int32)], dtype=np.int32)
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "token_count": 3,
                "source_capacity_token_count": 5,
                "document_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"sequence_count \* seq_len"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mmididx_v2_domain_routes_and_token_sidecars_round_trip(tmp_path) -> None:
    prefix = tmp_path / "domain_routes_v2"
    docs = [np.arange(4, dtype=np.int32), np.arange(10, 14, dtype=np.int32)]
    _write_mmididx(prefix, docs, dtype=np.int32)
    sidecar = _write_graph_sidecars(
        prefix,
        call_edges=[np.array([[0, 1]], dtype=np.int32), np.zeros((0, 2), dtype=np.int32)],
        type_edges=[np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)],
        chunk_starts=[np.array([0, 2]), np.array([0, 2])],
        chunk_ends=[np.array([2, 4]), np.array([2, 4])],
        chunk_kinds=[np.array([1, 2]), np.array([1, 2])],
        chunk_dep_levels=[np.array([0, 1]), np.array([0, 1])],
    )
    sidecar["graph_sidecar_schema"] = "cppmega_graph_routes_v2"
    graph_paths = sidecar["graph_sidecar_paths"]
    for key, entry in graph_paths.items():
        entry["coordinate_space"] = (
            "chunk_index"
            if key in {"token_call_edges", "token_type_edges", "token_chunk_kinds", "token_chunk_dep_levels"}
            else "token_index"
        )

    offsets_path = tmp_path / "domain_offsets.bin"
    data_path = tmp_path / "domain_data.bin"
    np.array([0, 1, 2], dtype=np.int64).tofile(offsets_path)
    np.array([[1, 0, 60], [1, 2, 21]], dtype=np.int32).tofile(data_path)
    graph_paths["token_domain_edges"] = {
        "kind": "edge_triples",
        "offsets_path": offsets_path.name,
        "data_path": data_path.name,
        "offset_dtype": "int64",
        "dtype": "int32",
        "item_count": 2,
        "shape_tail": [3],
        "coordinate_space": "token_index",
    }
    for name in (
        "token_build_edges",
        "token_shell_edges",
        "token_diagnostic_edges",
        "token_cross_domain_edges",
    ):
        empty_offsets = tmp_path / f"{name}_offsets.bin"
        empty_data = tmp_path / f"{name}_data.bin"
        np.zeros(3, dtype=np.int64).tofile(empty_offsets)
        np.zeros((0, 3), dtype=np.int32).tofile(empty_data)
        graph_paths[name] = {
            "kind": "edge_triples",
            "offsets_path": empty_offsets.name,
            "data_path": empty_data.name,
            "offset_dtype": "int64",
            "dtype": "int32",
            "item_count": 0,
            "shape_tail": [3],
            "coordinate_space": "token_index",
        }

    side_channel_paths = {}
    token_sidecars = {
        "loss_mask": np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=np.uint8),
        "doc_ids": np.array([1] * 8, dtype=np.int32),
        "token_domain_ids": np.array([20, 20, 20, 20, 21, 21, 21, 21], dtype=np.uint16),
        "token_role_ids": np.arange(8, dtype=np.uint16),
        "token_entity_ids": np.arange(8, dtype=np.uint32),
        "token_scope_ids": np.arange(8, dtype=np.uint32),
        "token_source_doc_ids": np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.uint32),
        "token_confidence_ids": np.ones(8, dtype=np.uint8),
        "hunk_id_per_token": np.array([-1, 0, 0, -1, -1, 1, 1, -1], dtype=np.int32),
        "edit_op_per_token": np.arange(8, dtype=np.uint8),
    }
    for name, values in token_sidecars.items():
        path = tmp_path / f"{name}.bin"
        values.tofile(path)
        side_channel_paths[name] = {"path": path.name, "dtype": values.dtype.name}
    sidecar["side_channel_paths"] = side_channel_paths
    prefix.with_suffix(".idx.json").write_text(json.dumps(sidecar), encoding="utf-8")

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)
    batch = next(dataset.iter_batches())
    packet = dataset.graph_route_packet_for_sample(0)

    np.testing.assert_array_equal(np.array(batch.loss_mask), [np.array([1, 1, 1, 0])])
    domain = batch.side_channel_map()["domain_routes"]
    np.testing.assert_array_equal(np.array(domain["domain_ids"]), [[20, 20, 20, 20]])
    temporal = batch.side_channel_map()["temporal_diff"]
    np.testing.assert_array_equal(np.array(temporal["hunk_ids"]), [[-1, 0, 0, -1]])
    assert packet.graph.edge("call").to_pairs() == [(0, 1)]
    assert packet.graph.edge("domain").to_pairs() == [(1, 0)]
    np.testing.assert_array_equal(packet.edge_kinds["domain"], np.array([60], dtype=np.int32))
    assert batch.graph_batch is not None
    np.testing.assert_array_equal(
        np.asarray(batch.graph_batch.edge_kinds[0]["domain"]),
        np.array([60], dtype=np.int32),
    )
    normalized = normalize_compiled_batch(batch, graph_routes_enabled=True)
    assert float(normalized["graph_attention_bias"][0, 1, 0].item()) == 1.0
    assert float(normalized["graph_edge_kind_bias"][0, 1, 0].item()) == 1.0

    build_graph = replace(
        batch.graph_batch,
        edge_kinds=({"domain": mx.array([20], dtype=mx.int32)},),
    )
    build_batch = replace(batch, graph_batch=build_graph)
    cfg = DenseCppLMConfig(
        vocab_size=256,
        hidden_size=32,
        depth=1,
        ffn_hidden_size=64,
        max_seq_length=4,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        graph_routes_enabled=True,
        graph_attention_bias_beta=1.0,
        ngram_hash_enabled=False,
        structure_residual_scale=0.0,
        platform_residual_scale=0.0,
    )

    def run(current: LMTokenBatch):
        mx.random.seed(211)
        model = DenseCppLM(cfg)
        optimizer = optim.SGD(learning_rate=1e-2)
        mx.eval(model.parameters(), optimizer.state)
        before = {
            name: np.asarray(value).copy()
            for name, value in tree_flatten(model.parameters())
            if isinstance(value, mx.array)
        }
        fixed = normalize_compiled_batch(current, graph_routes_enabled=True)
        logits = model.logits(
            current.inputs,
            block_bias=fixed["graph_attention_bias"],
            edge_kind_bias=fixed["graph_edge_kind_bias"],
        )
        metrics = CompiledPretrainingStep(model, optimizer, compile=True)(current)
        mx.eval(logits, model.parameters())
        updates = {
            name: np.asarray(value) - before[name]
            for name, value in tree_flatten(model.parameters())
            if isinstance(value, mx.array)
        }
        return np.asarray(logits), metrics.loss, updates

    diagnostic_logits, diagnostic_loss, diagnostic_updates = run(batch)
    build_logits, build_loss, build_updates = run(build_batch)
    assert not np.array_equal(diagnostic_logits, build_logits)
    assert abs(diagnostic_loss - build_loss) > 1e-6
    assert any(
        not np.array_equal(diagnostic_updates[name], build_updates[name])
        for name in diagnostic_updates
    )


def test_mmididx_graph_sidecars_fail_closed_for_window_slicing(tmp_path) -> None:
    prefix = tmp_path / "graph_routes_long_sequence"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    sidecar = _write_graph_sidecars(
        prefix,
        call_edges=[np.array([[1, 0]], dtype=np.int32)],
        type_edges=[np.array([], dtype=np.int32).reshape(0, 2)],
        chunk_starts=[np.array([0, 4], dtype=np.uint32)],
        chunk_ends=[np.array([4, 8], dtype=np.uint32)],
        chunk_kinds=[np.array([1, 1], dtype=np.uint16)],
        chunk_dep_levels=[np.array([0, 1], dtype=np.uint16)],
    )
    prefix.with_suffix(".idx.json").write_text(json.dumps(sidecar), encoding="utf-8")

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)

    with pytest.raises(NotImplementedError, match="sequence length to equal seq_len"):
        dataset.graph_route_packet_for_sample(0)


def test_mmididx_alias_and_canonical_side_channel_collision_fails_closed(tmp_path) -> None:
    prefix = tmp_path / "duplicate_alias"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    np.arange(8, dtype=np.int32).tofile(tmp_path / "structure_ids.bin")
    np.arange(8, dtype=np.int32).tofile(tmp_path / "token_structure_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "side_channel_paths": {
                    "structure_ids": "structure_ids.bin",
                    "token_structure_ids": "token_structure_ids.bin",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="structure_ids side-channel declared more than once"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mmididx_top_level_alias_and_canonical_collision_fails_closed(tmp_path) -> None:
    prefix = tmp_path / "top_level_duplicate_alias"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    np.arange(8, dtype=np.int32).tofile(tmp_path / "structure_ids.bin")
    np.arange(8, dtype=np.int32).tofile(tmp_path / "token_structure_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "structure_ids": "structure_ids.bin",
                "token_structure_ids": "token_structure_ids.bin",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="structure_ids side-channel declared more than once"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


@pytest.mark.parametrize(
    "side_channel_paths, top_level",
    [
        (
            {"structure_ids": "structure_ids.bin"},
            {"token_structure_ids": "token_structure_ids.bin"},
        ),
        (
            {"token_structure_ids": "token_structure_ids.bin"},
            {"structure_ids": "structure_ids.bin"},
        ),
    ],
)
def test_mmididx_cross_location_alias_and_canonical_collision_fails_closed(
    tmp_path,
    side_channel_paths,
    top_level,
) -> None:
    prefix = tmp_path / "cross_location_duplicate_alias"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    np.arange(8, dtype=np.int32).tofile(tmp_path / "structure_ids.bin")
    np.arange(8, dtype=np.int32).tofile(tmp_path / "token_structure_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "side_channel_paths": side_channel_paths,
                **top_level,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="structure_ids side-channel declared more than once"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_megatron_indexed_side_channel_schema_documents_aliases_and_dtypes() -> None:
    schema = megatron_indexed_side_channel_schema()
    graph_schema = megatron_indexed_graph_sidecar_schema()

    assert schema["attention_mask"]["default_dtype"] == "float32"
    assert schema["attention_mask"]["target_dtype"] == "float32"
    assert schema["attention_mask"]["model_kwarg"] is False
    structure_aliases = schema["structure_ids"]["aliases"]
    ast_depth_aliases = schema["ast_depth_ids"]["aliases"]
    assert isinstance(structure_aliases, list)
    assert isinstance(ast_depth_aliases, list)
    assert "token_structure_ids" in structure_aliases
    assert "token_ast_depth" in ast_depth_aliases
    assert "token_symbol_ids" in schema["symbol_ids"]["aliases"]
    assert schema["symbol_ids"]["family"] == "semantic_graph"
    assert schema["change_mask_pre"]["family"] == "temporal_diff"
    assert schema["structure_ids"]["default_dtype"] == "int32"
    assert schema["structure_ids"]["target_dtype"] == "int32"
    assert schema["structure_ids"]["model_kwarg"] is True
    structure_dtypes = schema["structure_ids"]["allowed_dtypes"]
    assert isinstance(structure_dtypes, list)
    assert "float32" not in structure_dtypes
    assert graph_schema["token_call_edges"]["kind"] == "edge_pairs"
    assert graph_schema["token_call_edges"]["shape_tail"] == [2]
    assert graph_schema["token_chunk_starts"]["dtype"] == "uint32"
    assert graph_schema["token_chunk_dep_levels"]["kind"] == "ragged_1d"


def test_mmididx_ambiguous_side_channel_list_fails_closed(tmp_path) -> None:
    prefix = tmp_path / "ambiguous_side_channels"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)
    prefix.with_suffix(".idx.json").write_text(
        json.dumps({"side_channels": ["attention_mask"]}),
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="ambiguous keys: side_channels"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mmididx_side_channel_length_mismatch_fails_closed(tmp_path) -> None:
    prefix = tmp_path / "short_side_channel"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)
    np.arange(8, dtype=np.int32).tofile(tmp_path / "structure_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps({"side_channel_paths": {"structure_ids": "structure_ids.bin"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match token shard count"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mmididx_document_ids_length_mismatch_fails_closed(tmp_path) -> None:
    prefix = tmp_path / "short_document_ids"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)
    np.arange(8, dtype=np.int32).tofile(tmp_path / "doc_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps({"doc_ids": "doc_ids.bin"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="document_ids token count .* token shard count"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mmididx_duplicate_document_id_sidecars_fail_closed(tmp_path) -> None:
    prefix = tmp_path / "duplicate_document_ids"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    np.arange(8, dtype=np.int32).tofile(tmp_path / "doc_ids.bin")
    np.arange(8, dtype=np.int32).tofile(tmp_path / "document_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "side_channel_paths": {"doc_ids": "doc_ids.bin"},
                "document_ids": "document_ids.bin",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="document_ids sidecar declared more than once"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mmididx_rejects_side_channel_values_outside_int32_range(tmp_path) -> None:
    prefix = tmp_path / "oversized_side_channel"
    _write_mmididx(prefix, [np.arange(4, dtype=np.int32)], dtype=np.int32)
    np.array(
        [0, np.iinfo(np.int32).max + 1, 2, 3],
        dtype=np.int64,
    ).tofile(tmp_path / "structure_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(
            {
                "side_channel_paths": {
                    "structure_ids": {
                        "path": "structure_ids.bin",
                        "dtype": "int64",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)

    with pytest.raises(ValueError, match="structure_ids side-channel IDs exceed int32 range"):
        next(dataset.iter_batches())


def test_mmididx_negative_document_ids_are_rejected(tmp_path) -> None:
    prefix = tmp_path / "negative_document_ids"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    np.array([0, 0, -1, 1, 1, 1, 2, 2], dtype=np.int32).tofile(
        tmp_path / "doc_ids.bin"
    )
    prefix.with_suffix(".idx.json").write_text(
        json.dumps({"packing_document_ids": "doc_ids.bin"}),
        encoding="utf-8",
    )

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)
    with pytest.raises(ValueError, match="document_ids must be non-negative"):
        next(dataset.iter_batches())


@pytest.mark.parametrize(
    "sidecar, error_type, match",
    [
        (
            {"side_channel_paths": {"structure_ids": {"dtype": "int32"}}},
            ValueError,
            "include a path",
        ),
        (
            {"side_channel_paths": {"attention_mask": {"path": "mask.bin", "dtype": "int16"}}},
            ValueError,
            "attention_mask side-channel dtype must be float32",
        ),
        (
            {"side_channel_paths": {"relation_ids": "relation_ids.bin"}},
            NotImplementedError,
            "unsupported side-channel key",
        ),
        (
            {"ngram_hash_ids": {"path": "ngram_hash_ids.bin", "dtype": "int32"}},
            NotImplementedError,
            "ngram sidecars are not supported",
        ),
    ],
)
def test_mmididx_bad_side_channel_metadata_fails_closed(
    tmp_path,
    sidecar,
    error_type,
    match,
) -> None:
    prefix = tmp_path / "bad_side_channel_metadata"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)
    np.arange(12, dtype=np.int16).tofile(tmp_path / "mask.bin")
    np.arange(12, dtype=np.int32).tofile(tmp_path / "relation_ids.bin")
    np.arange(12, dtype=np.int32).tofile(tmp_path / "ngram_hash_ids.bin")
    prefix.with_suffix(".idx.json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )

    with pytest.raises(error_type, match=match):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mmididx_negative_structure_side_channel_values_are_rejected(tmp_path) -> None:
    prefix = tmp_path / "negative_structure"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    np.array([0, 1, -1, 3, 4, 5, 6, 7], dtype=np.int32).tofile(
        tmp_path / "structure_ids.bin"
    )
    prefix.with_suffix(".idx.json").write_text(
        json.dumps({"side_channel_paths": {"structure_ids": "structure_ids.bin"}}),
        encoding="utf-8",
    )

    dataset = MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)
    with pytest.raises(ValueError, match="structure_ids side-channel IDs must be non-negative"):
        next(dataset.iter_batches())


def test_raw_bin_without_dtype_fails_closed(tmp_path) -> None:
    path = tmp_path / "ambiguous.bin"
    np.arange(8, dtype=np.int32).tofile(path)

    with pytest.raises(ValueError, match="explicit dtype"):
        MegatronIndexedDataset(path, seq_len=4, batch_size=1)


def test_unsupported_idx_header_fails_closed(tmp_path) -> None:
    prefix = tmp_path / "bad"
    np.arange(8, dtype=np.int32).tofile(prefix.with_suffix(".bin"))
    prefix.with_suffix(".idx").write_bytes(b"NOTMMIDXX")

    with pytest.raises(NotImplementedError, match="unsupported Megatron .idx header"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_short_bin_referenced_by_idx_is_rejected(tmp_path) -> None:
    prefix = tmp_path / "short"
    _write_mmididx(prefix, [np.arange(8, dtype=np.int32)], dtype=np.int32)
    prefix.with_suffix(".bin").write_bytes(np.arange(4, dtype=np.int32).tobytes())

    with pytest.raises(ValueError, match="references bytes past"):
        MegatronIndexedDataset(prefix, seq_len=4, batch_size=1)


def test_mismatched_dtype_code_and_pointer_layout_fails_closed(tmp_path) -> None:
    prefix = tmp_path / "bad_dtype_code"
    np.arange(12, dtype=np.uint16).tofile(prefix.with_suffix(".bin"))

    lengths = np.array([6, 6], dtype=np.int32)
    pointers = np.array([0, 12], dtype=np.int64)
    documents = np.array([0, 2], dtype=np.int64)
    with prefix.with_suffix(".idx").open("wb") as fh:
        fh.write(_HEADER)
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<B", 1))  # Upstream Megatron code 1 is uint8.
        fh.write(struct.pack("<Q", 2))
        fh.write(struct.pack("<Q", 2))
        lengths.tofile(fh)
        pointers.tofile(fh)
        documents.tofile(fh)

    with pytest.raises(ValueError, match="pointers do not match token dtype"):
        MegatronIndexedDataset(prefix, seq_len=3, batch_size=1)


def test_negative_or_out_of_range_token_ids_are_rejected(tmp_path) -> None:
    negative = tmp_path / "negative"
    _write_mmididx(negative, [np.array([0, -1, 2, 3], dtype=np.int32)], dtype=np.int32)
    with pytest.raises(ValueError, match="non-negative"):
        next(MegatronIndexedDataset(negative, seq_len=4, batch_size=1).iter_batches())

    too_large = tmp_path / "too_large"
    np.array([0, np.iinfo(np.int32).max + 1, 2, 3], dtype=np.int64).tofile(
        too_large.with_suffix(".bin")
    )
    with pytest.raises(ValueError, match="int32 range"):
        next(
            MegatronIndexedDataset(
                too_large.with_suffix(".bin"),
                seq_len=4,
                batch_size=1,
                dtype="int64",
            ).iter_batches()
        )


def test_reader_module_does_not_import_megatron_torch_or_cuda() -> None:
    source = Path("cppmega_mlx/data/megatron_indexed.py").read_text(encoding="utf-8")

    assert "import megatron" not in source
    assert "import torch" not in source
    assert "cuda" not in source.lower()
