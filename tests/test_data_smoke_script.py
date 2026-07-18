from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_megatron_indexed import _write_structured_multishard_fixture


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data_smoke.py"


def write_npz(path: Path, *, include_structure: bool) -> None:
    tokens = (np.arange(16, dtype=np.int32) % 32).reshape(4, 4)
    arrays: dict[str, Any] = {
        "attention_mask": np.ones_like(tokens, dtype=np.float32),
        "tokens": tokens,
        "tokenizer_contract": np.array("local_profile"),
        "vocab_size": np.array(32, dtype=np.int64),
    }
    if include_structure:
        arrays["structure_ids"] = (tokens % 7).astype(np.int32)
        arrays["dep_levels"] = (tokens % 3).astype(np.int32)
    np.savez(path, **arrays)


def write_parquet(path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "token_ids": pa.array(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                type=pa.large_list(pa.uint32()),
            ),
            "structure_ids": pa.array(
                [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                ],
                type=pa.large_list(pa.int8()),
            ),
        }
    )
    pq.write_table(table, path)


def write_parquet_with_family_side_channels(path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(
        {
            "token_ids": pa.array(
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ],
                type=pa.large_list(pa.uint32()),
            ),
            "token_symbol_ids": pa.array(
                [
                    [10, 11, 12, 13],
                    [14, 15, 16, 17],
                    [18, 19, 20, 21],
                    [22, 23, 24, 25],
                ],
                type=pa.large_list(pa.int32()),
            ),
            "edit_op_per_token": pa.array(
                [
                    [0, 0, 2, 2],
                    [0, 3, 0, 3],
                    [1, 1, 0, 0],
                    [4, 0, 4, 0],
                ],
                type=pa.large_list(pa.int32()),
            ),
        }
    )
    pq.write_table(table, path)


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def load_json(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def test_npz_smoke_reports_local_ingress_contract(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, include_structure=True)

    result = run_script(
        str(npz_path),
        "--dataset-format",
        "npz",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--batches",
        "2",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["status"] == "ok"
    assert payload["dataset_format"] == "npz"
    assert payload["batch_shape"] == [2, 4]
    assert payload["batch_size"] == 2
    assert payload["seq_len"] == 4
    assert payload["batches_read"] == 2
    assert payload["dataset"]["num_samples"] == 4
    assert payload["dataset"]["num_batches"] == 2
    assert payload["dataset"]["metadata"]["vocab_size"] == 32
    assert payload["side_channels"] == [
        "attention_mask",
        "dep_levels",
        "structure_ids",
    ]
    assert payload["structure_side_channels"] == ["structure_ids", "dep_levels"]
    assert payload["structure_side_channels_present"] is True
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
    assert payload["m4_vs_gb10_parity_claim"] is False
    assert payload["distributed_megatron_parity_claim"] is False
    assert payload["trainable_metal_kernel_adoption_claim"] is False
    assert payload["forward_wired"] is False
    assert payload["forward"] == {"enabled": False}
    assert payload["training_wired"] is False


def test_npz_smoke_can_run_deterministic_packing(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, include_structure=True)

    result = run_script(
        str(npz_path),
        "--dataset-format",
        "npz",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--pack-documents",
        "--eos-token-id",
        "31",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["packing"] == {
        "boundary_mask_shape": [2, 4, 4],
        "doc_ids_shape": [2, 4],
        "document_source": "first_batch_rows_without_final_token",
        "enabled": True,
        "packed_shape": [2, 4],
        "token_mask_true": 8,
    }


def test_require_structure_side_channels_fails_closed(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, include_structure=False)

    result = run_script(
        str(npz_path),
        "--dataset-format",
        "npz",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--require-structure-side-channels",
    )

    assert result.returncode == 2
    payload = load_json(result)
    assert payload["status"] == "error"
    assert payload["dataset_format"] == "npz"
    assert "structure side channels" in payload["error"]
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False


def test_parquet_smoke_reports_local_ingress_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tokens.parquet"
    write_parquet(dataset_path)

    result = run_script(
        str(dataset_path),
        "--token-key",
        "token_ids",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["status"] == "ok"
    assert payload["dataset_format"] == "parquet"
    assert payload["token_key"] == "token_ids"
    assert payload["batch_shape"] == [2, 4]
    assert payload["dataset"]["metadata"]["source_format"] == "parquet"
    assert payload["dataset"]["parquet_receipt"]["source_format"] == "parquet"
    assert payload["dataset"]["parquet_receipt"]["token_source"] == {
        "mode": "token_column",
        "column": "token_ids",
        "type": "large_list<element: uint32>",
    }
    assert payload["dataset"]["parquet_receipt"]["side_channel_sources"] == {}
    assert payload["dataset"]["parquet_receipt"]["skipped_side_channel_columns"] == [
        {
            "field": "structure_ids",
            "column": "structure_ids",
            "type": "large_list<element: int8>",
            "reason": "not_token_aligned",
        }
    ]
    assert payload["side_channels"] == []
    assert payload["family_side_channels"] == []
    assert payload["family_side_channel_presence"] == {}
    assert payload["structure_side_channels"] == []
    assert payload["structure_side_channels_present"] is False
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
    assert payload["m4_vs_gb10_parity_claim"] is False
    assert payload["distributed_megatron_parity_claim"] is False
    assert payload["training_wired"] is False


def test_parquet_smoke_reports_generic_family_side_channels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "family_side_channels.parquet"
    write_parquet_with_family_side_channels(dataset_path)

    result = run_script(
        str(dataset_path),
        "--token-key",
        "token_ids",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--forward-smoke",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["side_channels"] == []
    assert payload["family_side_channels"] == ["semantic_graph", "temporal_diff"]
    assert payload["family_side_channel_presence"] == {
        "semantic_graph": {"token_symbol_ids": True},
        "temporal_diff": {"edit_op_per_token": True},
    }
    assert payload["dataset"]["parquet_receipt"]["family_side_channel_sources"] == {
        "semantic_graph": {
            "token_symbol_ids": {
                "column": "token_symbol_ids",
                "type": "large_list<element: int32>",
            }
        },
        "temporal_diff": {
            "edit_op_per_token": {
                "column": "edit_op_per_token",
                "type": "large_list<element: int32>",
            }
        },
    }
    assert payload["forward"]["side_channel_model_kwargs"] == []


def test_megatron_multishard_smoke_reports_side_channels(tmp_path: Path) -> None:
    _write_structured_multishard_fixture(
        tmp_path,
        shard_docs=[
            [np.arange(8, dtype=np.int32)],
            [np.arange(100, 108, dtype=np.int32)],
        ],
    )

    result = run_script(
        str(tmp_path),
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--require-structure-side-channels",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["status"] == "ok"
    assert payload["dataset_format"] == "megatron"
    assert payload["batch_shape"] == [2, 4]
    assert payload["dataset"]["metadata"]["source_format"] == "megatron-multishard"
    assert payload["dataset"]["index_metadata"]["source_format"] == "megatron-multishard"
    assert payload["dataset"]["index_metadata"]["shard_count"] == 2
    assert payload["dataset"]["token_id_range"] == [0, 107]
    assert payload["side_channels"] == [
        "attention_mask",
        "dep_levels",
        "structure_ids",
    ]
    assert payload["structure_side_channels"] == ["structure_ids", "dep_levels"]
    assert payload["structure_side_channels_present"] is True
    assert payload["distributed_megatron_parity_claim"] is False
    assert payload["forward_wired"] is False


def test_npz_smoke_can_run_forward_only_model_path(tmp_path: Path) -> None:
    npz_path = tmp_path / "tokens.npz"
    write_npz(npz_path, include_structure=True)

    result = run_script(
        str(npz_path),
        "--dataset-format",
        "npz",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--forward-smoke",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["status"] == "ok"
    assert payload["forward_wired"] is True
    assert payload["training_wired"] is False
    assert payload["forward"]["enabled"] is True
    assert payload["forward"]["forward_only"] is True
    assert payload["forward"]["finite_loss"] is True
    assert payload["forward"]["logits_shape"] == [2, 3, 32]
    assert payload["forward"]["target_shape"] == [2, 3]
    assert payload["forward"]["ntokens"] == 6.0
    assert payload["forward"]["document_ids_used"] is False
    assert payload["forward"]["side_channel_model_kwargs"] == [
        "dep_levels",
        "structure_ids",
    ]
    assert payload["forward"]["training_wired"] is False
    assert payload["gb10_parity_claim"] is False


def test_megatron_multishard_smoke_can_run_forward_with_document_ids(
    tmp_path: Path,
) -> None:
    _write_structured_multishard_fixture(
        tmp_path,
        shard_docs=[
            [np.arange(8, dtype=np.int32)],
            [np.arange(100, 108, dtype=np.int32)],
        ],
    )

    result = run_script(
        str(tmp_path),
        "--batch-size",
        "2",
        "--seq-len",
        "4",
        "--require-structure-side-channels",
        "--forward-smoke",
    )

    assert result.returncode == 0, result.stderr
    payload = load_json(result)
    assert payload["status"] == "ok"
    assert payload["dataset_format"] == "megatron"
    assert payload["forward_wired"] is True
    assert payload["training_wired"] is False
    assert payload["forward"]["enabled"] is True
    assert payload["forward"]["finite_loss"] is True
    assert payload["forward"]["logits_shape"] == [2, 3, 256]
    assert payload["forward"]["target_shape"] == [2, 3]
    assert payload["forward"]["ntokens"] == 6.0
    assert payload["forward"]["document_ids_used"] is True
    assert payload["forward"]["side_channel_model_kwargs"] == [
        "dep_levels",
        "structure_ids",
    ]
    assert payload["forward"]["training_wired"] is False
    assert payload["distributed_megatron_parity_claim"] is False


def test_unsupported_dataset_format_fails_closed_with_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tokens.jsonl"

    result = run_script(
        str(dataset_path),
        "--dataset-format",
        "jsonl",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
    )

    assert result.returncode == 2
    payload = load_json(result)
    assert payload["status"] == "error"
    assert payload["dataset_format"] == "jsonl"
    assert "unsupported dataset format" in payload["error"]
    assert "npz, parquet, megatron" in payload["error"]
    assert payload["local_only"] is True
