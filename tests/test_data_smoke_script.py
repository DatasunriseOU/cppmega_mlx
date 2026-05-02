from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


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


def test_unsupported_dataset_format_fails_closed_with_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tokens.parquet"

    result = run_script(
        str(dataset_path),
        "--dataset-format",
        "parquet",
        "--batch-size",
        "2",
        "--seq-len",
        "4",
    )

    assert result.returncode == 2
    payload = load_json(result)
    assert payload["status"] == "error"
    assert payload["dataset_format"] == "parquet"
    assert "unsupported dataset format" in payload["error"]
    assert "npz, megatron" in payload["error"]
    assert payload["local_only"] is True
