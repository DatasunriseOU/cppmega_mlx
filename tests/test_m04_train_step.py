from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m04_train_step.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
GB10_SAMPLE = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)
REAL_PARQUET_COLUMNS = (
    "token_ids",
    "structure_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
)


def run_script(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON if PYTHON.exists() else sys.executable), str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def load_json_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def copy_real_parquet_head(source_path: Path, sample_path: Path, *, row_count: int = 4) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    parquet_file = pq.ParquetFile(source_path)
    batch = next(
        parquet_file.iter_batches(
            batch_size=row_count,
            columns=list(REAL_PARQUET_COLUMNS),
        )
    )
    pq.write_table(pa.Table.from_batches([batch]), sample_path)


def tiny_args(output: Path, *, steps: int = 1) -> list[str]:
    return [
        "--synthetic",
        "--steps",
        str(steps),
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--vocab-size",
        "32",
        "--hidden-size",
        "8",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--output",
        str(output),
        "--json",
    ]


def assert_m04_receipt_contract(payload: dict[str, Any]) -> None:
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_mlx_m04_train_step"
    assert payload["issue"]["id"] == "cppmega-mlx-t8f.4"
    assert payload["local_only"] is True
    assert payload["gb10_training_correctness_claim"] is False
    assert payload["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["full_m0_4_acceptance_claim"] is False
    assert payload["workload"]["dtype"] == "bfloat16"
    assert payload["model"]["source"] == "cppmega_mlx.models.hybrid_lm"
    assert payload["baseline_row"] == {
        "batch_size": payload["workload"]["batch_size"],
        "commit": payload["software"]["git_commit"] or "unknown",
        "dtype": "bfloat16",
        "gb10_parity_claim": False,
        "hardware": payload["baseline_row"]["hardware"],
        "local_only": True,
        "mode": payload["workload"]["mode"],
        "model": "HybridTinyLM",
        "route": payload["model"]["route_symbols"],
        "seq_len": payload["workload"]["seq_len"],
        "tokens_per_second": payload["timing"]["tokens_per_second"] or 0.0,
    }


def test_synthetic_one_step_writes_finite_receipt(tmp_path: Path) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output))
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "ok"
    assert payload["workload"]["synthetic"] is True
    assert payload["workload"]["data_format"] == "npz"
    assert payload["training"]["steps_completed"] == 1
    assert payload["training"]["optimizer_updated"] is True
    assert payload["training"]["all_finite"] is True
    assert payload["training"]["loss_decrease_satisfied"] is True
    assert payload["training"]["final_loss"] > 0
    assert payload["training"]["step_metrics"][0]["updated"] is True
    assert payload["memory"]["peak_memory_bytes"] is None or (
        payload["memory"]["peak_memory_bytes"] >= 0
    )


def test_require_loss_decrease_fails_single_step_but_writes_receipt(tmp_path: Path) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output), "--require-loss-decrease")

    assert result.returncode == 2
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "failed"
    assert payload["training"]["steps_completed"] == 1
    assert payload["training"]["loss_decrease_required"] is True
    assert payload["training"]["loss_decrease_satisfied"] is False


def test_missing_dataset_dry_run_reports_blocked_receipt(tmp_path: Path) -> None:
    output = tmp_path / "m04_train_step.json"
    missing = tmp_path / "missing.parquet"

    result = run_script(
        "--data-path",
        str(missing),
        "--dry-run-json",
        "--output",
        str(output),
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["type"] == "missing_dataset"
    assert payload["training"]["steps_completed"] == 0
    assert payload["workload"]["data_path"] == str(missing)


def test_real_parquet_dry_run_uses_gb10_sample_receipt(tmp_path: Path) -> None:
    if not GB10_SAMPLE.exists():
        pytest.skip(f"GB10 parquet sample is not present: {GB10_SAMPLE}")

    sample_path = tmp_path / "clang_semantic_4k_v10_head.parquet"
    copy_real_parquet_head(GB10_SAMPLE, sample_path)
    output = tmp_path / "m04_train_step.json"
    result = run_script(
        "--data-path",
        str(sample_path),
        "--dry-run-json",
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--seq-len",
        "64",
        "--hidden-size",
        "8",
        "--pattern",
        "M",
        "--depth",
        "1",
        "--output",
        str(output),
    )
    payload = load_json_result(result)

    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "dry_run"
    assert payload["workload"]["synthetic"] is False
    assert payload["workload"]["data_format"] == "parquet"
    assert payload["workload"]["data_path"] == str(sample_path)
    assert payload["dataset"]["metadata"]["source_format"] == "parquet"
    assert payload["dataset"]["dataset_receipt"]["source_dataset_name"] == (
        "clang_semantic_4k_v10"
    )
    assert payload["training"]["steps_completed"] == 0
