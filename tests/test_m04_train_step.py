from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

import scripts.m04_train_step as m04_train_step
from cppmega_mlx.training.optimizers import (
    ADAMW_BASE_CLASS,
    ADAMW_FP32_MOMENTS_CLASS,
    AdamWFP32Moments,
    collect_adamw_moment_dtypes,
    dtype_name,
    make_adamw,
)
from scripts.m04_train_step import (
    OBSERVED_OPTIMIZER_IDENTITY,
    GRAD_CHECKPOINT_EXPECTATION,
    REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
    REQUIRED_DTYPE,
    REQUIRED_MODEL_GEOMETRY,
    REQUIRED_MODEL_SOURCE,
    acceptance_gate_payload,
    applied_memory_limit_api_path_from_payload,
    local_gb10_quarter_preflight_payload,
    target_dataset_path,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m04_train_step.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
BASELINE_RECEIPT = ROOT / "bench" / "baselines" / "m04_train_step.json"
GB10_SAMPLE = (
    ROOT
    / "data"
    / "parquet_samples"
    / "gb10"
    / "clang_semantic_4k_v10"
    / "val_00000.parquet"
)
TARGET_PARQUET = "data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet"
REAL_PARQUET_COLUMNS = (
    "token_ids",
    "structure_ids",
    "token_structure_ids",
    "token_dep_levels",
    "token_ast_depth",
    "token_sibling_index",
    "token_ast_node_type",
)


def canonical_allocation_probe(**overrides: Any) -> dict[str, Any]:
    probe = {
        "status": "ok",
        "allocation_ready": True,
        "source": REQUIRED_MODEL_SOURCE,
        "allocation_mode": "full_profile_allocation_probe",
        "required_geometry": REQUIRED_MODEL_GEOMETRY,
        "profile_geometry": REQUIRED_MODEL_GEOMETRY,
        "geometry_matches_required": True,
        "profile_name": "local_gb10_quarter",
        "model_class": "HybridTinyLM",
        "eval_scope": "parameters_only_no_forward_no_training",
        "forward_executed": False,
        "training_executed": False,
        "memory_before": {"active_memory_bytes": 0},
        "memory_after": {"active_memory_bytes": 1024},
    }
    probe.update(overrides)
    return probe


def test_m04_import_preserves_recipes_package_exports() -> None:
    import cppmega_mlx.recipes as recipes

    assert recipes is sys.modules["cppmega_mlx.recipes"]
    assert isinstance(recipes.__all__, list)
    assert "local_gb10_quarter" in recipes.__all__
    assert hasattr(recipes, "local_gb10_quarter")


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
    gate = payload["acceptance_gate"]
    assert gate["full_target_dataset"] == TARGET_PARQUET
    assert gate["full_target_dataset_100_step_required"] is True
    assert gate["local_gb10_quarter_required"] is True
    assert gate["required_model_profile"] == "local_gb10_quarter"
    assert gate["required_dtype"] == REQUIRED_DTYPE
    assert gate["observed_dtype"] == "bfloat16"
    assert gate["dtype_ok"] is True
    assert gate["required_optimizer_name"] == "AdamW"
    assert gate["grad_checkpoint_required"] is True
    assert gate["full_local_gb10_quarter_gate_required"] is True
    for key in (
        "real_parquet_source_identity",
        "target_parquet_path_ok",
        "dataset_name_ok",
        "dataset_format_ok",
        "dtype_ok",
        "local_gb10_quarter_preflight",
        "local_gb10_quarter_preflight_ok",
        "model_identity_ok",
        "model_identity",
        "optimizer_identity_ok",
        "optimizer_identity",
        "required_adamw_master_moment_dtype",
        "observed_adamw_master_moment_dtypes",
        "fp32_adamw_master_moments_ok",
        "adamw_ok",
        "grad_checkpoint_expectation_ok",
        "grad_checkpoint_identity",
        "step_count_ok",
        "loss_decrease_ok",
        "loss_fields_ok",
        "all_finite_ok",
        "optimizer_update_ok",
        "m4_runtime_metadata",
        "m4_runtime_metadata_ok",
        "full_local_gb10_quarter_gate_completed",
        "full_local_gb10_quarter_gate_blockers",
    ):
        assert key in gate
    assert gate["real_parquet_source_identity"]["required_path"] == TARGET_PARQUET
    assert payload["local_gb10_quarter_preflight"] == gate["local_gb10_quarter_preflight"]
    preflight = payload["local_gb10_quarter_preflight"]
    assert preflight["profile_name"] == "local_gb10_quarter"
    assert preflight["source"] == REQUIRED_MODEL_SOURCE
    assert preflight["required_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert preflight["profile_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert preflight["geometry_matches_required"] is True
    assert preflight["tokenizer_contract"]["resolved"] is True
    assert preflight["tokenizer_contract"]["expected_vocab_size"] == 65_536
    assert preflight["tokenizer_contract"]["blocker_id"] == "cppmega-mlx-t8f.1"
    assert preflight["tokenizer_contract"]["milestone"] == "M0.1"
    assert "<FIM_INSTRUCTION>" in preflight["tokenizer_contract"]["required_special_tokens"]
    assert "CODE_START" in preflight["tokenizer_contract"]["reason"]
    assert "M0.1 is closed" in preflight["tokenizer_contract"]["reason"]
    if payload["workload"]["probe_local_gb10_quarter_allocation"]:
        assert preflight["allocation_attempted"] is True
        assert preflight["allocation_ready"] is True
        assert preflight["allocation_mode"] == "full_profile_allocation_probe"
        assert preflight["allocation_probe"]["status"] == "ok"
        assert preflight["allocation_probe"]["source"] == REQUIRED_MODEL_SOURCE
        assert preflight["allocation_probe"]["allocation_mode"] == (
            "full_profile_allocation_probe"
        )
        assert preflight["allocation_probe"]["required_geometry"] == (
            REQUIRED_MODEL_GEOMETRY
        )
        assert preflight["allocation_probe"]["profile_geometry"] == (
            REQUIRED_MODEL_GEOMETRY
        )
        assert preflight["allocation_probe"]["geometry_matches_required"] is True
        assert preflight["allocation_probe"]["model_class"] == "HybridTinyLM"
        assert preflight["allocation_probe"]["forward_executed"] is False
        assert preflight["allocation_probe"]["training_executed"] is False
        assert preflight["ok"] is True
        assert preflight["blockers"] == []
        assert gate["local_gb10_quarter_preflight_ok"] is True
    else:
        assert preflight["allocation_attempted"] is False
        assert preflight["allocation_ready"] is False
        assert preflight["allocation_mode"] == "allocation_free_preflight"
        assert preflight["ok"] is False
        assert {"allocation_attempted", "allocation_ready"}.issubset(
            set(preflight["blockers"])
        )
        assert "tokenizer_contract_resolved" not in preflight["blockers"]
        assert gate["local_gb10_quarter_preflight_ok"] is False
    if gate["full_target_dataset_100_step_completed"]:
        assert gate["uses_full_target_dataset"] is True
        assert gate["real_parquet_source_identity"]["ok"] is True
        assert gate["full_target_dataset_blocker"] is None
    else:
        assert gate["full_target_dataset_blocker"]
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert gate["full_local_gb10_quarter_gate_blockers"]
    model_identity = gate["model_identity"]
    assert model_identity["required_name"] == "local_gb10_quarter"
    assert model_identity["required_source"] == REQUIRED_MODEL_SOURCE
    assert model_identity["required_profile"] == "local_gb10_quarter"
    assert model_identity["required_geometry"] == REQUIRED_MODEL_GEOMETRY
    assert model_identity["ok"] is gate["model_identity_ok"]
    optimizer_identity = gate["optimizer_identity"]
    assert gate["required_adamw_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert optimizer_identity["required_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert optimizer_identity["master_moment_evidence"]["required_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert optimizer_identity["master_moment_dtype_ok"] is (
        gate["fp32_adamw_master_moments_ok"]
    )
    assert payload["workload"]["dtype"] == "bfloat16"
    model_payload = payload["model"]
    if model_payload.get("metadata_only"):
        assert payload["workload"]["mode"] == "metadata_only_no_forward_no_training"
        assert model_payload["source"] is None
        assert model_payload["name"] is None
        assert model_payload["required_source"] == REQUIRED_MODEL_SOURCE
        assert model_payload["required_profile"] == "local_gb10_quarter"
        assert model_payload["profile_matches_required"] is False
        assert model_payload["local_gb10_quarter_preflight"] == preflight
        assert model_payload["forward_executed"] is False
        assert model_payload["training_executed"] is False
    else:
        assert model_payload["source"] == "cppmega_mlx.models.hybrid_lm"
        assert model_payload["name"] == "HybridTinyLM"
        assert model_payload["required_profile"] == "local_gb10_quarter"
        assert model_payload["profile_matches_required"] is False
        assert model_payload["local_gb10_quarter_preflight"] == preflight
    assert payload["training"]["optimizer"]["name"] == "AdamW"
    assert payload["training"]["optimizer"]["class"] == (
        ADAMW_FP32_MOMENTS_CLASS
    )
    assert payload["training"]["optimizer"]["base_class"] == ADAMW_BASE_CLASS
    assert payload["training"]["optimizer"]["adamw"] is True
    assert payload["training"]["optimizer"]["required_master_moment_dtype"] == (
        REQUIRED_ADAMW_MASTER_MOMENT_DTYPE
    )
    assert payload["training"]["optimizer"]["master_moment_evidence"] == (
        optimizer_identity["master_moment_evidence"]
    )
    grad_checkpoint_expected = bool(payload["workload"].get("grad_checkpoint", False))
    assert payload["training"]["grad_checkpoint"]["required"] is True
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is (
        grad_checkpoint_expected
    )
    assert payload["training"]["grad_checkpoint"]["expectation_satisfied"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_observed_enabled"] is grad_checkpoint_expected
    assert gate["grad_checkpoint_expectation_ok"] is grad_checkpoint_expected
    assert gate["grad_checkpoint_identity"]["observed_enabled"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_identity"]["expectation_satisfied"] is (
        grad_checkpoint_expected
    )
    assert gate["grad_checkpoint_identity"]["ok"] is grad_checkpoint_expected
    expected_model = (
        "metadata_only_no_observed_model"
        if model_payload.get("metadata_only")
        else "HybridTinyLM"
    )
    expected_route = (
        "metadata_only_no_forward_no_training"
        if model_payload.get("metadata_only")
        else model_payload["route_symbols"]
    )
    assert payload["baseline_row"] == {
        "batch_size": payload["workload"]["batch_size"],
        "commit": payload["software"]["git_commit"] or "unknown",
        "dtype": "bfloat16",
        "gb10_parity_claim": False,
        "hardware": payload["baseline_row"]["hardware"],
        "local_only": True,
        "mode": payload["workload"]["mode"],
        "model": expected_model,
        "route": expected_route,
        "seq_len": payload["workload"]["seq_len"],
        "tokens_per_second": payload["timing"]["tokens_per_second"] or 0.0,
    }


def test_checked_in_receipt_records_parquet_smoke_without_m0_4_claim() -> None:
    payload = json.loads(BASELINE_RECEIPT.read_text())

    assert_m04_receipt_contract(payload)
    assert payload["status"] == "ok"
    assert payload["full_m0_4_acceptance_claim"] is False
    assert payload["workload"]["synthetic"] is False
    assert payload["workload"]["data_format"] == "parquet"
    assert payload["workload"]["mode"] == "eager"
    assert payload["workload"]["steps_requested"] == 100
    assert payload["workload"]["batch_size"] == 1
    assert payload["workload"]["seq_len"] == 64
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is True
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is True
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is True
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is True
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert payload["acceptance_gate"]["optimizer_identity_ok"] is True
    assert payload["acceptance_gate"]["adamw_ok"] is True
    assert payload["acceptance_gate"]["fp32_adamw_master_moments_ok"] is True
    assert payload["acceptance_gate"]["observed_adamw_master_moment_dtypes"]
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is False
    assert payload["acceptance_gate"]["m4_runtime_metadata_ok"] is True
    assert set(payload["acceptance_gate"]["full_local_gb10_quarter_gate_blockers"]) == {
        "grad_checkpoint_expectation_ok",
        "model_identity_ok",
    }
    assert payload["acceptance_gate"]["full_target_dataset_blocker"] is None
    assert payload["training"]["steps_completed"] == 100
    assert payload["training"]["all_finite"] is True
    assert payload["training"]["optimizer_updated"] is True
    assert payload["training"]["loss_decreased"] is True
    assert payload["training"]["loss_decrease_satisfied"] is True
    assert payload["training"]["final_loss"] < payload["training"]["initial_loss"]
    assert payload["training"]["optimizer"]["name"] == "AdamW"
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is False
    assert payload["model"]["name"] == "HybridTinyLM"
    assert payload["model"]["profile_matches_required"] is False
    assert payload["baseline_row"]["model"] == "HybridTinyLM"
    assert {item["id"] for item in payload["acceptance_blockers"]} == {
        "cppmega-mlx-t8f.4.local_gb10_quarter_gate",
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
    assert payload["workload"]["model_profile"] == "hybrid_tiny"
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is False
    assert payload["training"]["steps_completed"] == 1
    assert payload["training"]["optimizer_updated"] is True
    assert payload["training"]["all_finite"] is True
    assert payload["training"]["loss_decrease_satisfied"] is True
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["full_target_dataset_blocker"]
    assert payload["training"]["final_loss"] > 0
    assert payload["training"]["step_metrics"][0]["updated"] is True
    assert payload["memory"]["peak_memory_bytes"] is None or (
        payload["memory"]["peak_memory_bytes"] >= 0
    )


def test_synthetic_grad_checkpoint_receipt_marks_gate_without_m0_4_claim(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_train_step.json"
    result = run_script(*tiny_args(output), "--grad-checkpoint")
    payload = load_json_result(result)

    assert output.exists()
    assert json.loads(output.read_text()) == payload
    assert_m04_receipt_contract(payload)
    assert payload["status"] == "ok"
    assert payload["workload"]["model_profile"] == "hybrid_tiny"
    assert payload["workload"]["grad_checkpoint"] is True
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is True
    assert payload["training"]["grad_checkpoint"]["expectation_satisfied"] is True
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is True
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "local_gb10_quarter_preflight_ok",
        "model_identity_ok",
    }.issubset(set(payload["acceptance_gate"]["full_local_gb10_quarter_gate_blockers"]))


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
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def assert_local_gb10_metadata_dry_run_contract(payload: dict[str, Any]) -> None:
    status = payload["status"]
    assert status in {"dry_run", "failed"}
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_mlx_m04_train_step"
    assert payload["local_only"] is True
    assert payload["gb10_training_correctness_claim"] is False
    assert payload["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["full_m0_4_acceptance_claim"] is False
    assert "blockers" not in payload
    assert {item["id"] for item in payload["acceptance_blockers"]} == {
        "cppmega-mlx-t8f.4.local_gb10_quarter_gate",
    }
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["mode"] == "metadata_only_no_forward_no_training"
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["optimizer_updated"] is False
    assert payload["training"]["losses"] == []
    assert payload["training"]["optimizer"]["update_observed"] is False
    assert payload["training"]["optimizer"]["master_moment_evidence"]["skipped"] is True
    assert payload["model"]["source"] is None
    assert payload["model"]["name"] is None
    assert payload["model"]["observed_source"] is None
    assert payload["model"]["observed_name"] is None
    assert payload["model"]["required_source"] == REQUIRED_MODEL_SOURCE
    assert payload["model"]["required_name"] == "local_gb10_quarter"
    assert payload["model"]["requested_profile"] == "local_gb10_quarter"
    assert payload["model"]["profile"] is None
    assert payload["model"]["requested_profile_matches_required"] is True
    assert payload["model"]["profile_matches_required"] is False
    assert payload["model"]["metadata_only"] is True
    assert payload["model"]["forward_executed"] is False
    assert payload["model"]["training_executed"] is False
    assert payload["baseline_row"]["model"] == "metadata_only_no_observed_model"
    assert payload["baseline_row"]["tokens_per_second"] == 0.0
    assert payload["baseline_row"]["local_only"] is True
    assert payload["baseline_row"]["gb10_parity_claim"] is False
    gate = payload["acceptance_gate"]
    assert gate["required_model_profile"] == "local_gb10_quarter"
    assert gate["observed_model_name"] is None
    assert gate["observed_model_source"] is None
    assert gate["model_identity_ok"] is False
    assert gate["optimizer_update_ok"] is False
    assert gate["adamw_ok"] is False
    assert gate["full_target_dataset_100_step_completed"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "real_parquet_source_identity_ok",
        "target_parquet_path_ok",
        "dataset_name_ok",
        "dataset_format_ok",
        "model_identity_ok",
        "optimizer_update_ok",
        "loss_decrease_ok",
        "loss_fields_ok",
        "all_finite_ok",
    }.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))


def test_local_gb10_quarter_dry_run_is_metadata_only_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    def fail_allocation_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("full local_gb10_quarter allocation must be opt-in")

    monkeypatch.setattr(m04_train_step, "dry_run_payload", fail_route)
    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_route)
    monkeypatch.setattr(m04_train_step, "local_gb10_quarter", fail_allocation_probe)
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--dry-run-json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert exit_code == 0
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is False
    assert payload["local_gb10_quarter_preflight"]["allocation_attempted"] is False
    assert payload["local_gb10_quarter_preflight"]["allocation_mode"] == (
        "allocation_free_preflight"
    )
    assert payload["acceptance_gate"]["local_gb10_quarter_preflight_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_require_loss_decrease_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    monkeypatch.setattr(m04_train_step, "dry_run_payload", fail_route)
    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_route)
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--dry-run-json",
            "--require-loss-decrease",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert exit_code == 2
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["status"] == "failed"
    assert payload["training"]["loss_decreased"] is False
    assert payload["training"]["loss_decrease_required"] is True
    assert payload["training"]["loss_decrease_satisfied"] is False
    assert payload["acceptance_gate"]["loss_decrease_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_quarter_dry_run_cli_writes_requested_output_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_local_gb10_metadata.json"
    baseline_before = (
        BASELINE_RECEIPT.read_text(encoding="utf-8")
        if BASELINE_RECEIPT.exists()
        else None
    )

    result = run_script(
        "--synthetic",
        "--model-profile",
        "local_gb10_quarter",
        "--dry-run-json",
        "--output",
        str(output),
        "--json",
    )

    payload = load_json_result(result)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert_local_gb10_metadata_dry_run_contract(payload)
    if baseline_before is None:
        assert not BASELINE_RECEIPT.exists()
    else:
        assert BASELINE_RECEIPT.read_text(encoding="utf-8") == baseline_before


def test_local_gb10_quarter_dry_run_records_non_default_optimizer_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_local_gb10_lion.json"
    result = run_script(
        "--synthetic",
        "--model-profile",
        "local_gb10_quarter",
        "--dry-run-json",
        "--optimizer",
        "lion",
        "--output",
        str(output),
        "--json",
    )

    payload = load_json_result(result)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["optimizer"] == {
        "requested": "lion",
        "key": "lion",
        "quant_scheme": "dynamic_int8_v1",
        "source": "cli",
    }
    optimizer = payload["training"]["optimizer"]
    assert optimizer["name"] == "Lion"
    assert optimizer["key"] == "lion"
    assert optimizer["class"] == "cppmega_mlx.training.optimizers.LionFP32Moments"
    assert optimizer["adamw"] is False
    assert optimizer["master_moment_evidence"]["skipped"] is True
    assert payload["acceptance_gate"]["observed_optimizer_name"] == "Lion"
    assert payload["acceptance_gate"]["optimizer_identity_ok"] is False
    assert payload["acceptance_gate"]["adamw_ok"] is False


def test_non_default_optimizer_is_blocked_outside_local_gb10_route(
    tmp_path: Path,
) -> None:
    output = tmp_path / "m04_hybrid_lion.json"
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--optimizer",
            "lion",
            "--output",
            str(output),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["type"] == "unsupported_optimizer_route"
    assert payload["workload"]["optimizer"]["key"] == "lion"
    assert payload["training"]["optimizer"]["name"] == "Lion"


def test_local_gb10_quarter_training_routes_to_monkeypatchable_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    route_calls: list[tuple[m04_train_step.TrainHybridTinyConfig, Path]] = []

    def fail_train(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("HybridTinyLM training route must not be called")

    def fake_local_gb10_route(
        _args: Any,
        *,
        config: m04_train_step.TrainHybridTinyConfig,
        data_path: Path,
    ) -> tuple[dict[str, Any], int]:
        route_calls.append((config, data_path))
        return (
            m04_train_step.blocked_receipt(
                _args,
                "unit-test local_gb10_quarter route called",
                "unit_test_route_called",
            ),
            2,
        )

    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_train)
    monkeypatch.setattr(
        m04_train_step,
        "run_local_gb10_quarter_training",
        fake_local_gb10_route,
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert len(route_calls) == 1
    config, data_path = route_calls[0]
    assert config.model_profile == "local_gb10_quarter"
    assert config.grad_checkpoint is False
    assert config.data_format == "npz"
    assert data_path.suffix == ".npz"
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["full_m0_4_acceptance_claim"] is False
    assert payload["blockers"][0]["type"] == "unit_test_route_called"
    assert payload["blockers"][0]["reason"] == "unit-test local_gb10_quarter route called"
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["training"]["steps_completed"] == 0
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["model_identity_ok"] is False


def test_local_gb10_quarter_grad_checkpoint_routes_to_monkeypatchable_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    route_calls: list[tuple[m04_train_step.TrainHybridTinyConfig, Path]] = []

    def fail_train(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("HybridTinyLM training route must not be called")

    def fake_local_gb10_route(
        _args: Any,
        *,
        config: m04_train_step.TrainHybridTinyConfig,
        data_path: Path,
    ) -> tuple[dict[str, Any], int]:
        route_calls.append((config, data_path))
        return (
            m04_train_step.blocked_receipt(
                _args,
                "unit-test local_gb10_quarter grad-checkpoint route called",
                "unit_test_route_called",
            ),
            2,
        )

    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_train)
    monkeypatch.setattr(
        m04_train_step,
        "run_local_gb10_quarter_training",
        fake_local_gb10_route,
    )
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--grad-checkpoint",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert len(route_calls) == 1
    config, data_path = route_calls[0]
    assert config.model_profile == "local_gb10_quarter"
    assert config.grad_checkpoint is True
    assert config.data_format == "npz"
    assert data_path.suffix == ".npz"
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["blockers"][0]["type"] == "unit_test_route_called"
    assert payload["blockers"][0]["reason"] == (
        "unit-test local_gb10_quarter grad-checkpoint route called"
    )
    assert payload["workload"]["model_profile"] == "local_gb10_quarter"
    assert payload["workload"]["grad_checkpoint"] is True
    assert payload["training"]["steps_completed"] == 0
    assert payload["training"]["grad_checkpoint"]["observed_enabled"] is True
    assert payload["acceptance_gate"]["grad_checkpoint_expectation_ok"] is True
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert (
        payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    )


def test_local_gb10_quarter_dry_run_with_allocation_probe_is_preflight_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probe_called = False

    def fake_probe() -> dict[str, Any]:
        nonlocal probe_called
        probe_called = True
        return canonical_allocation_probe()

    def fail_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("training and dry-run routes must not be called")

    monkeypatch.setattr(m04_train_step, "probe_local_gb10_quarter_allocation", fake_probe)
    monkeypatch.setattr(m04_train_step, "dry_run_payload", fail_route)
    monkeypatch.setattr(m04_train_step, "train_hybrid_tiny", fail_route)
    args = m04_train_step.build_parser().parse_args(
        [
            "--synthetic",
            "--model-profile",
            "local_gb10_quarter",
            "--probe-local-gb10-quarter-allocation",
            "--dry-run-json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    payload, exit_code = m04_train_step.run_receipt(args)

    assert probe_called is True
    assert exit_code == 0
    assert_local_gb10_metadata_dry_run_contract(payload)
    assert payload["workload"]["grad_checkpoint"] is False
    assert payload["workload"]["probe_local_gb10_quarter_allocation"] is True
    preflight = payload["local_gb10_quarter_preflight"]
    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is True
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["forward_executed"] is False
    assert preflight["allocation_probe"]["training_executed"] is False
    assert preflight["ok"] is True
    assert payload["acceptance_gate"]["local_gb10_quarter_preflight_ok"] is True
    assert payload["acceptance_gate"]["model_identity_ok"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False


def test_local_gb10_allocation_probe_success_is_preflight_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe() -> dict[str, Any]:
        return canonical_allocation_probe()

    monkeypatch.setattr(m04_train_step, "probe_local_gb10_quarter_allocation", fake_probe)
    args = m04_train_step.build_parser().parse_args(
        [
            "--probe-local-gb10-quarter-allocation",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    preflight = m04_train_step.local_gb10_quarter_preflight_from_args(args)
    gate = local_gb10_gate(
        model_name="HybridTinyLM",
        grad_checkpoint=grad_checkpoint_identity(enabled=False),
        local_gb10_quarter_preflight=preflight,
    )

    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is True
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["forward_executed"] is False
    assert preflight["allocation_probe"]["training_executed"] is False
    assert preflight["ok"] is True
    assert preflight["blockers"] == []
    assert gate["local_gb10_quarter_preflight_ok"] is True
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert {
        "model_identity_ok",
        "grad_checkpoint_expectation_ok",
    }.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))


def test_local_gb10_allocation_probe_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_probe() -> dict[str, Any]:
        return canonical_allocation_probe(
            status="blocked",
            allocation_ready=False,
            memory_after={"active_memory_bytes": 0},
            error_type="RuntimeError",
            error="synthetic allocation failure",
        )

    monkeypatch.setattr(m04_train_step, "probe_local_gb10_quarter_allocation", fake_probe)
    args = m04_train_step.build_parser().parse_args(
        [
            "--probe-local-gb10-quarter-allocation",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    preflight = m04_train_step.local_gb10_quarter_preflight_from_args(args)
    gate = local_gb10_gate(local_gb10_quarter_preflight=preflight)

    assert preflight["allocation_attempted"] is True
    assert preflight["allocation_ready"] is False
    assert preflight["allocation_mode"] == "full_profile_allocation_probe"
    assert preflight["allocation_probe"]["error_type"] == "RuntimeError"
    assert preflight["allocation_probe"]["error"] == "synthetic allocation failure"
    assert preflight["ok"] is False
    assert preflight["blockers"] == ["allocation_ready"]
    assert gate["local_gb10_quarter_preflight_ok"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert "local_gb10_quarter_preflight_ok" in (
        gate["full_local_gb10_quarter_gate_blockers"]
    )


def test_applied_memory_limit_api_path_preserves_actual_fallback_path() -> None:
    payload = {
        "applied": True,
        "metal_limit_api_path": "mx.set_memory_limit",
    }

    assert applied_memory_limit_api_path_from_payload(payload) == "mx.set_memory_limit"


class _Bf16Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.ones((2, 2), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sum(x @ self.weight)


def _run_bf16_probe_update(
    optimizer: optim.Optimizer,
) -> tuple[_Bf16Probe, dict[str, str]]:
    model = _Bf16Probe()

    def loss_fn(probe: _Bf16Probe, x: mx.array) -> mx.array:
        return probe(x)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    _, grads = loss_and_grad(model, mx.ones((2, 2), dtype=mx.bfloat16))
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return model, collect_adamw_moment_dtypes(optimizer.state)


@pytest.mark.parametrize(
    ("optimizer_key", "expected_key", "expected_name", "quantized"),
    [
        ("adamw", "adamw", "AdamW", False),
        ("muon_adamw", "muon_adamw", "MuonAdamW", False),
        ("nam56r", "muon_adamw", "MuonAdamW", False),
        ("lion", "lion", "Lion", False),
        ("adam8bit", "adam8bit", "Adam8bit", True),
        ("lion8bit", "lion8bit", "Lion8bit", True),
        ("int8", "int8", "MuonAdamWInt8", True),
    ],
)
def test_local_gb10_optimizer_selector_initializes_supported_variants(
    optimizer_key: str,
    expected_key: str,
    expected_name: str,
    quantized: bool,
) -> None:
    args = m04_train_step.build_parser().parse_args(
        [
            "--model-profile",
            "local_gb10_quarter",
            "--optimizer",
            optimizer_key,
        ]
    )
    config = m04_train_step.config_from_args(args, data_path=GB10_SAMPLE)
    model = _Bf16Probe()

    optimizer = m04_train_step.make_local_gb10_optimizer(
        args,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    identity = m04_train_step.optimizer_identity_for_selected_optimizer(
        args,
        config,
        optimizer,
        model,
        optimizer_updated=True,
    )

    assert identity["key"] == expected_key
    assert identity["name"] == expected_name
    assert identity["quantized_state"] is quantized
    assert identity["learning_rate"] == config.learning_rate
    assert identity["weight_decay"] == config.weight_decay
    assert identity["variant"]["requested"] == optimizer_key
    expected_quant_scheme = None if optimizer_key == "adamw" else "dynamic_int8_v1"
    assert identity["variant"]["quant_scheme"] == expected_quant_scheme
    if optimizer_key == "adamw":
        assert identity["adamw"] is True
        assert identity["name_matches_required"] is True
        assert identity["master_moment_dtype_ok"] is True
    else:
        assert identity["adamw"] is False
        assert identity["name_matches_required"] is False
        assert identity["master_moment_dtype_ok"] is False
        assert identity["state_evidence"]["state_dtype_breakdown_bytes"]


def test_stock_mlx_adamw_uses_bf16_moments_for_bf16_params() -> None:
    model, moment_dtypes = _run_bf16_probe_update(
        optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    )

    assert dtype_name(model.weight) == "bfloat16"
    assert moment_dtypes == {
        "weight/m": "bfloat16",
        "weight/v": "bfloat16",
    }


def test_repo_local_adamw_keeps_bf16_params_with_fp32_moments() -> None:
    optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.0)
    model, moment_dtypes = _run_bf16_probe_update(optimizer)

    assert isinstance(optimizer, AdamWFP32Moments)
    assert dtype_name(model.weight) == "bfloat16"
    assert moment_dtypes == {
        "weight/m": "float32",
        "weight/v": "float32",
    }


def test_repo_local_adamw_weight_decay_preserves_fp32_moments() -> None:
    optimizer = make_adamw(learning_rate=1e-3, weight_decay=0.1)
    model, moment_dtypes = _run_bf16_probe_update(optimizer)

    assert isinstance(optimizer, AdamWFP32Moments)
    assert dtype_name(model.weight) == "bfloat16"
    assert bool(mx.all(mx.isfinite(model.weight)).item())
    assert moment_dtypes == {
        "weight/m": "float32",
        "weight/v": "float32",
    }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"applied": False, "metal_limit_api_path": "mx.set_memory_limit"},
        {"applied": True},
        {"applied": True, "metal_limit_api_path": ""},
    ],
)
def test_applied_memory_limit_api_path_requires_applied_recorded_path(
    payload: Any,
) -> None:
    assert applied_memory_limit_api_path_from_payload(payload) is None


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
    assert payload["acceptance_gate"]["uses_full_target_dataset"] is False
    assert payload["acceptance_gate"]["real_parquet_source_identity"]["ok"] is False
    assert payload["acceptance_gate"]["full_target_dataset_100_step_completed"] is False
    assert payload["acceptance_gate"]["full_local_gb10_quarter_gate_completed"] is False
    assert payload["acceptance_gate"]["full_target_dataset_blocker"]
    assert payload["dataset"]["metadata"]["source_format"] == "parquet"
    assert payload["dataset"]["dataset_receipt"]["source_dataset_name"] == (
        "clang_semantic_4k_v10"
    )
    assert payload["training"]["steps_completed"] == 0


def target_dataset_receipt(
    *,
    source_path: str = TARGET_PARQUET,
    source_format: str = "parquet",
    source_dataset_name: str = "clang_semantic_4k_v10",
) -> dict[str, Any]:
    return {
        "path": source_path,
        "dataset_receipt": {
            "source_path": source_path,
            "source_format": source_format,
            "source_dataset_name": source_dataset_name,
        },
        "metadata": {"source_format": source_format},
    }


def adamw_moment_evidence(*, moment_dtype: str = "float32") -> dict[str, Any]:
    return {
        "required_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "observed_parameter_dtype": REQUIRED_DTYPE,
        "observed_moment_dtypes": {
            "weight/m": moment_dtype,
            "weight/v": moment_dtype,
        },
        "optimizer_class": ADAMW_FP32_MOMENTS_CLASS,
        "optimizer_base_class": ADAMW_BASE_CLASS,
        "state_keys": ["learning_rate", "step", "weight"],
        "ok": moment_dtype == REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
    }


def adamw_identity(
    *,
    name: str = "AdamW",
    updated: bool = True,
    moment_dtype: str = "float32",
) -> dict[str, Any]:
    master_moment_evidence = adamw_moment_evidence(moment_dtype=moment_dtype)
    return {
        **OBSERVED_OPTIMIZER_IDENTITY,
        "name": name,
        "required_name": "AdamW",
        "name_matches_required": name == "AdamW",
        "adamw": name == "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "update_observed": updated,
        "required_master_moment_dtype": REQUIRED_ADAMW_MASTER_MOMENT_DTYPE,
        "master_moment_evidence": master_moment_evidence,
        "master_moment_dtype_ok": master_moment_evidence["ok"],
    }


def grad_checkpoint_identity(*, enabled: bool = True) -> dict[str, Any]:
    return {
        "required": True,
        "observed_enabled": enabled,
        "source": GRAD_CHECKPOINT_EXPECTATION["source"],
        "expectation_satisfied": enabled,
    }


def m4_device_metadata(*, device_name: str = "Apple M4 Max") -> dict[str, Any]:
    return {
        "machine": "arm64",
        "metal_available": True,
        "platform": "macOS-26.4.1-arm64-arm-64bit-Mach-O",
        "mlx_device_info": {
            "device_name": device_name,
            "memory_size": 137438953472,
        },
    }


def local_gb10_model_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "profile": "local_gb10_quarter",
        **REQUIRED_MODEL_GEOMETRY,
    }
    config["mtp"] = dict(REQUIRED_MODEL_GEOMETRY["mtp"])
    config.update(overrides)
    return config


def resolved_local_gb10_preflight(**overrides: Any) -> dict[str, Any]:
    allocation_probe = canonical_allocation_probe()
    preflight = local_gb10_quarter_preflight_payload(
        allocation_attempted=True,
        allocation_ready=True,
        allocation_mode="full_profile_allocation_probe",
        allocation_probe=allocation_probe,
    )
    preflight["tokenizer_contract"] = {
        **preflight["tokenizer_contract"],
        "resolved": True,
    }
    preflight["ok"] = True
    preflight["blockers"] = []
    preflight.update(overrides)
    return preflight


def local_gb10_gate(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data_path": target_dataset_path(),
        "data_format": "parquet",
        "dtype": REQUIRED_DTYPE,
        "dataset": target_dataset_receipt(),
        "steps_requested": 100,
        "steps_completed": 100,
        "loss_decreased": True,
        "all_finite": True,
        "optimizer_updated": True,
        "model_name": "local_gb10_quarter",
        "model_source": REQUIRED_MODEL_SOURCE,
        "model_config": local_gb10_model_config(),
        "optimizer": adamw_identity(),
        "grad_checkpoint": grad_checkpoint_identity(),
        "device": m4_device_metadata(),
        "local_gb10_quarter_preflight": resolved_local_gb10_preflight(),
    }
    payload.update(overrides)
    return acceptance_gate_payload(**payload)


def test_acceptance_gate_accepts_complete_local_gb10_quarter_evidence() -> None:
    gate = local_gb10_gate()

    assert gate["real_parquet_source_identity"]["ok"] is True
    assert gate["full_target_dataset_100_step_completed"] is True
    assert gate["dtype_ok"] is True
    assert gate["local_gb10_quarter_preflight_ok"] is True
    assert gate["local_gb10_quarter_preflight"]["source"] == REQUIRED_MODEL_SOURCE
    assert gate["local_gb10_quarter_preflight"]["required_geometry"] == (
        REQUIRED_MODEL_GEOMETRY
    )
    assert gate["local_gb10_quarter_preflight"]["profile_geometry"] == (
        REQUIRED_MODEL_GEOMETRY
    )
    assert gate["model_identity_ok"] is True
    assert gate["optimizer_identity_ok"] is True
    assert gate["required_adamw_master_moment_dtype"] == "float32"
    assert gate["observed_adamw_master_moment_dtypes"] == {
        "weight/m": "float32",
        "weight/v": "float32",
    }
    assert gate["fp32_adamw_master_moments_ok"] is True
    assert gate["adamw_ok"] is True
    assert gate["grad_checkpoint_expectation_ok"] is True
    assert gate["m4_runtime_metadata_ok"] is True
    assert gate["full_local_gb10_quarter_gate_completed"] is True
    assert gate["full_local_gb10_quarter_gate_blockers"] == []


@pytest.mark.parametrize(
    ("overrides", "failed_checks"),
    [
        (
            {
                "dataset": target_dataset_receipt(
                    source_path="/tmp/fake/clang_semantic_4k_v10/val_00000.parquet"
                )
            },
            {"real_parquet_source_identity_ok", "target_parquet_path_ok"},
        ),
        (
            {"dataset": target_dataset_receipt(source_dataset_name="not_clang")},
            {"real_parquet_source_identity_ok", "dataset_name_ok"},
        ),
        (
            {
                "data_format": "npz",
                "dataset": target_dataset_receipt(source_format="npz"),
            },
            {"real_parquet_source_identity_ok", "dataset_format_ok"},
        ),
        (
            {"dtype": "float32"},
            {"dtype_ok"},
        ),
        (
            {
                "model_name": "HybridTinyLM",
                "model_config": local_gb10_model_config(),
            },
            {"model_identity_ok"},
        ),
        (
            {
                "model_name": "local_gb10_quarter",
                "model_config": local_gb10_model_config(profile="HybridTinyLM"),
            },
            {"model_identity_ok"},
        ),
        (
            {"model_source": "fake.local_gb10_quarter"},
            {"model_identity_ok"},
        ),
        (
            {"model_config": local_gb10_model_config(hidden_size=16)},
            {"model_identity_ok"},
        ),
        (
            {
                "model_config": local_gb10_model_config(
                    mtp={"depth": 1, "beta": 0.6, "loss_weight": 0.3}
                )
            },
            {"model_identity_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    source="fake.local_gb10_quarter"
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    profile_geometry={
                        **REQUIRED_MODEL_GEOMETRY,
                        "hidden_size": 16,
                    }
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {
                "local_gb10_quarter_preflight": resolved_local_gb10_preflight(
                    tokenizer_contract={
                        **resolved_local_gb10_preflight()["tokenizer_contract"],
                        "resolved": False,
                    }
                )
            },
            {"local_gb10_quarter_preflight_ok"},
        ),
        (
            {"optimizer": adamw_identity(name="SGD")},
            {"optimizer_identity_ok", "adamw_ok"},
        ),
        (
            {"optimizer": {**adamw_identity(), "class": "fake.AdamW"}},
            {"optimizer_identity_ok", "adamw_ok"},
        ),
        (
            {"optimizer": adamw_identity(moment_dtype="bfloat16")},
            {"fp32_adamw_master_moments_ok"},
        ),
        (
            {"grad_checkpoint": grad_checkpoint_identity(enabled=False)},
            {"grad_checkpoint_expectation_ok"},
        ),
        (
            {
                "grad_checkpoint": {
                    **grad_checkpoint_identity(),
                    "source": "unit-test-local-gb10-quarter",
                }
            },
            {"grad_checkpoint_expectation_ok"},
        ),
        (
            {"device": m4_device_metadata(device_name="Apple M3 Max")},
            {"m4_runtime_metadata_ok"},
        ),
        (
            {"steps_completed": 99},
            {"step_count_ok"},
        ),
        (
            {"steps_requested": 101, "steps_completed": 100},
            {"step_count_ok"},
        ),
        (
            {"loss_decreased": False},
            {"loss_decrease_ok", "loss_fields_ok"},
        ),
        (
            {"all_finite": False},
            {"all_finite_ok", "loss_fields_ok"},
        ),
    ],
)
def test_acceptance_gate_fail_closes_on_fake_or_incomplete_evidence(
    overrides: dict[str, Any],
    failed_checks: set[str],
) -> None:
    gate = local_gb10_gate(**overrides)

    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert failed_checks.issubset(set(gate["full_local_gb10_quarter_gate_blockers"]))
    if any(
        check in failed_checks
        for check in (
            "real_parquet_source_identity_ok",
            "target_parquet_path_ok",
            "dataset_name_ok",
            "dataset_format_ok",
            "dtype_ok",
            "step_count_ok",
            "loss_fields_ok",
            "optimizer_update_ok",
        )
    ):
        assert gate["full_target_dataset_100_step_completed"] is False


@pytest.mark.parametrize(
    "preflight_overrides",
    [
        {"allocation_probe": None},
        {"allocation_probe": canonical_allocation_probe(status="blocked")},
        {"allocation_probe": canonical_allocation_probe(allocation_ready=False)},
        {"allocation_probe": canonical_allocation_probe(source="fake.local_gb10_quarter")},
        {"allocation_probe": canonical_allocation_probe(source=None)},
        {"allocation_probe": canonical_allocation_probe(allocation_mode="caller_supplied_allocation_evidence")},
        {"allocation_probe": canonical_allocation_probe(allocation_mode=None)},
        {"allocation_probe": canonical_allocation_probe(profile_name="HybridTinyLM")},
        {"allocation_probe": canonical_allocation_probe(model_class="FakeTinyLM")},
        {"allocation_probe": canonical_allocation_probe(model_class=None)},
        {"allocation_probe": canonical_allocation_probe(eval_scope="forward_smoke")},
        {"allocation_probe": canonical_allocation_probe(forward_executed=True)},
        {"allocation_probe": canonical_allocation_probe(training_executed=True)},
        {"allocation_probe": canonical_allocation_probe(geometry_matches_required=False)},
        {
            "allocation_probe": canonical_allocation_probe(
                required_geometry={**REQUIRED_MODEL_GEOMETRY, "hidden_size": 16}
            )
        },
        {
            "allocation_probe": canonical_allocation_probe(
                profile_geometry={**REQUIRED_MODEL_GEOMETRY, "hidden_size": 16}
            )
        },
        {
            "allocation_mode": "caller_supplied_allocation_evidence",
            "allocation_probe": canonical_allocation_probe(),
        },
    ],
)
def test_acceptance_gate_requires_canonical_allocation_probe(
    preflight_overrides: dict[str, Any],
) -> None:
    gate = local_gb10_gate(
        local_gb10_quarter_preflight=resolved_local_gb10_preflight(
            **preflight_overrides
        )
    )

    assert gate["local_gb10_quarter_preflight_ok"] is False
    assert gate["full_local_gb10_quarter_gate_completed"] is False
    assert "local_gb10_quarter_preflight_ok" in (
        gate["full_local_gb10_quarter_gate_blockers"]
    )
