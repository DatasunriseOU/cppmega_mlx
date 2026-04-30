from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_tiny.py"


def run_bench(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_help_lists_required_flags() -> None:
    result = run_bench("--help")

    assert result.returncode == 0
    assert "--batch-size" in result.stdout
    assert "--dry-run-json" in result.stdout
    assert "--dtype" in result.stdout
    assert "--hardware-label" in result.stdout
    assert "--compare-line" in result.stdout
    assert "--auto-wired-limit" in result.stdout
    assert "--wired-limit-bytes" in result.stdout


def test_dry_run_json_reports_shape_and_device() -> None:
    result = run_bench(
        "--dry-run-json",
        "--batch-size",
        "1",
        "--seq-len",
        "8",
        "--vocab-size",
        "64",
        "--d-model",
        "16",
        "--n-heads",
        "2",
        "--n-layers",
        "1",
        "--mlp-dim",
        "32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["tokens_per_step"] == 8
    assert payload["hardware_label"]
    assert payload["dtype"] == "bfloat16"
    assert payload["batch_size"] == 1
    assert payload["seq_len"] == 8
    assert payload["warmup_steps"] == 0
    assert payload["measured_steps"] == 1
    assert payload["compile"] is True
    assert payload["include_structure"] is False
    assert payload["tokens_per_second"] is None
    assert payload["peak_memory_bytes"] is None
    assert payload["model_source"] in {
        "cppmega_mlx.models.tiny_lm",
        "self_contained_fallback",
    }
    assert payload["config"]["dtype"] == "bfloat16"
    assert "default_device" in payload["device"]
    assert "metal" in payload["device"]
    assert payload["profile"]["enabled"] is False
    assert payload["profile"]["helpers"]["memory_snapshot"].endswith("MemorySnapshot")
    run_metadata = payload["run_metadata"]
    assert run_metadata["schema_version"] == 1
    assert run_metadata["workload"]["data_contract"] == "synthetic_tokens"
    assert run_metadata["matched_run"]["key"]["data_contract"] == "synthetic_tokens"
    assert run_metadata["framework"]["mlx"] == payload["device"]["mlx"]
    assert payload["matched_run"] == run_metadata["matched_run"]
    assert payload["matched_run"]["claim_policy"] == "No GB10 parity claim from a single-host row."
    assert payload["matched_run"]["key"]["dtype"] == "bfloat16"
    memory = payload["memory"]
    assert memory["active_bytes"] is None
    assert memory["peak_bytes"] is None
    assert memory["cache_bytes"] is None
    assert memory["after_measured_steps"]["available"] is False
    assert memory["after_warmup"]["measured"] is False
    assert memory["after_measured_steps"]["measured"] is False
    assert memory["wired_limit"]["mode"] == "off"
    assert memory["wired_limit"]["applied"] is False
    assert "max_recommended_working_set_size_bytes" in memory["wired_limit"]


def test_invalid_shape_returns_error_json() -> None:
    result = run_bench(
        "--dry-run-json",
        "--d-model",
        "15",
        "--n-heads",
        "2",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "d_model must be divisible by n_heads" in payload["error"]


def test_minimal_real_benchmark_reports_core_metrics() -> None:
    result = run_bench(
        "--json",
        "--no-compile",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--vocab-size",
        "32",
        "--d-model",
        "8",
        "--n-heads",
        "1",
        "--n-layers",
        "1",
        "--mlp-dim",
        "16",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["config"]["compile"] is False
    assert payload["hardware_label"]
    assert payload["dtype"] == "float32"
    assert payload["batch_size"] == 1
    assert payload["seq_len"] == 4
    assert payload["warmup_steps"] == 0
    assert payload["measured_steps"] == 1
    assert payload["compile"] is False
    assert payload["include_structure"] is False
    assert payload["tokens_per_step"] == 4
    assert payload["first_call_time_s"] > 0
    assert payload["compile_time_s"] == 0.0
    assert payload["mean_step_time_s"] > 0
    assert payload["wall_time_s"] == payload["mean_step_time_s"]
    assert payload["mean_wall_time_s"] == payload["mean_step_time_s"]
    assert payload["total_wall_time_s"] == sum(payload["step_times_s"])
    assert payload["tokens_per_second"] > 0
    assert payload["peak_memory_bytes"] >= 0
    assert payload["parameter_count"] > 0
    assert payload["model_source"] in {
        "cppmega_mlx.models.tiny_lm",
        "self_contained_fallback",
    }
    profile = payload["profile"]
    assert profile["enabled"] is True
    assert profile["helpers"]["profile_step"].endswith("profile_step")
    assert set(profile["scopes"]) >= {"first_call", "measured_steps"}
    assert profile["scopes"]["first_call"]["tokens"] == 4
    assert profile["scopes"]["measured_steps"]["tokens"] == 4
    assert profile["scopes"]["measured_steps"]["evaluated"] is True
    assert profile["scopes"]["measured_steps"]["synchronized"] is True
    assert profile["scopes"]["measured_steps"]["peak_memory_bytes"] == payload["peak_memory_bytes"]
    assert profile["scopes"]["measured_steps"]["wall_time_s"] == profile["scopes"]["measured_steps"]["seconds"]
    assert profile["scopes"]["measured_steps"]["elapsed_wall_time_s"] == profile["scopes"]["measured_steps"]["seconds"]
    run_metadata = payload["run_metadata"]
    assert run_metadata["workload"]["model_source"] == payload["model_source"]
    assert run_metadata["workload"]["data_contract"] == "synthetic_tokens"
    assert run_metadata["matched_run"]["key"]["data_contract"] == "synthetic_tokens"
    assert run_metadata["framework"]["mlx"] == payload["device"]["mlx"]
    assert payload["matched_run"]["key"]["model_source"] == payload["model_source"]
    assert "No GB10 parity claim" in payload["matched_run"]["claim_policy"]
    memory = payload["memory"]
    assert memory["active_bytes"] >= 0
    assert memory["peak_bytes"] == payload["peak_memory_bytes"]
    assert memory["cache_bytes"] >= 0
    assert memory["active_gib"] >= 0
    assert memory["peak_gib"] == payload["peak_memory_gib"]
    assert memory["cache_gib"] >= 0
    assert memory["after_warmup"]["measured"] is True
    assert memory["after_measured_steps"]["measured"] is True
    assert memory["wired_limit"]["mode"] == "off"
    assert memory["wired_limit"]["applied"] is False


def test_compiled_project_tiny_path_if_available() -> None:
    result = run_bench(
        "--json",
        "--batch-size",
        "1",
        "--seq-len",
        "8",
        "--vocab-size",
        "64",
        "--d-model",
        "16",
        "--n-heads",
        "2",
        "--n-layers",
        "1",
        "--mlp-dim",
        "32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["config"]["compile"] is True
    assert payload["compile_time_s"] > 0
    assert payload["tokens_per_second"] > 0


def test_compare_line_reports_stable_fields() -> None:
    result = run_bench(
        "--compare-line",
        "--no-compile",
        "--hardware-label",
        "test-m4",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--vocab-size",
        "32",
        "--d-model",
        "8",
        "--n-heads",
        "1",
        "--n-layers",
        "1",
        "--mlp-dim",
        "16",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    line = lines[0]
    parts = line.split()
    assert [part.split("=", maxsplit=1)[0] for part in parts] == [
        "hardware_label",
        "dtype",
        "batch_size",
        "seq_len",
        "warmup_steps",
        "measured_steps",
        "compile",
        "include_structure",
        "tokens_per_second",
        "peak_memory_bytes",
    ]
    assert "hardware_label=test-m4" in parts
    assert "dtype=float32" in parts
    assert "batch_size=1" in parts
    assert "seq_len=4" in parts
    assert "warmup_steps=0" in parts
    assert "measured_steps=1" in parts
    assert "compile=False" in parts
    assert "include_structure=False" in parts
    assert any(part.startswith("tokens_per_second=") for part in parts)
    assert any(part.startswith("peak_memory_bytes=") for part in parts)


def test_compare_line_contract_matches_archived_baseline_field_order(tmp_path: Path) -> None:
    matrix_script = ROOT / "scripts" / "bench_matrix.py"
    archive_path = tmp_path / "archive.json"
    matrix = subprocess.run(
        [
            sys.executable,
            str(matrix_script),
            "--dry-run-json",
            "--archive-baseline",
            str(archive_path),
            "--hardware-label",
            "test-m4",
            "--batch-sizes",
            "1",
            "--seq-lens",
            "4",
            "--profiles",
            "smoke",
            "--routes",
            "plain",
            "--compile-modes",
            "eager",
            "--dtype",
            "float32",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert matrix.returncode == 0, matrix.stderr
    archived_fields = json.loads(archive_path.read_text(encoding="utf-8"))["records"][0][
        "compare_line_contract"
    ]["fields"]

    line = run_bench(
        "--compare-line",
        "--no-compile",
        "--hardware-label",
        "test-m4",
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--vocab-size",
        "32",
        "--d-model",
        "8",
        "--n-heads",
        "1",
        "--n-layers",
        "1",
        "--mlp-dim",
        "16",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )
    assert line.returncode == 0, line.stderr
    assert [part.split("=", maxsplit=1)[0] for part in line.stdout.split()] == archived_fields


def test_dry_run_reports_explicit_wired_limit_without_applying() -> None:
    result = run_bench(
        "--dry-run-json",
        "--wired-limit-bytes",
        "0",
        "--batch-size",
        "1",
        "--seq-len",
        "8",
        "--vocab-size",
        "64",
        "--d-model",
        "16",
        "--n-heads",
        "2",
        "--n-layers",
        "1",
        "--mlp-dim",
        "32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    wired = payload["memory"]["wired_limit"]
    assert wired["mode"] == "explicit"
    assert wired["requested_bytes"] == 0
    assert wired["applied_bytes"] is None
    assert wired["previous_bytes"] is None
    assert wired["applied"] is False
