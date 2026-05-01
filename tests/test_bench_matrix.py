from __future__ import annotations

import json
import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_matrix.py"


def run_matrix(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def test_bench_matrix_module_is_importable() -> None:
    assert importlib.import_module("scripts.bench_matrix") is not None


def test_help_includes_matched_run_guard() -> None:
    result = run_matrix("--help")

    assert result.returncode == 0
    assert "--seq-lens" in result.stdout
    assert "--batch-sizes" in result.stdout
    assert "--profiles" in result.stdout
    assert "--routes" in result.stdout
    assert "Matched-run guard" in result.stdout


def test_dry_run_json_expands_matrix_schema() -> None:
    result = run_matrix(
        "--dry-run-json",
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1,2",
        "--seq-lens",
        "4",
        "--profiles",
        "smoke",
        "--routes",
        "plain,structure",
        "--compile-modes",
        "eager",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["schema_version"] == 1
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_only"
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
    assert payload["hardware_label"] == "test-m4"
    assert payload["case_count"] == 4
    assert "matched rows" in payload["matched_run_guard"]
    assert "No GB10" not in payload["parity_claim_policy"]
    assert "claims only after both hardware labels" in payload["parity_claim_policy"]
    assert "Single-host matrix receipt only" in payload["local_only_policy"]
    assert "software.mlx_version" in payload["required_receipt_fields"]
    assert "timing.wall_time_s" in payload["required_receipt_fields"]
    assert "timing.tokens_per_second_or_step_time" in payload["required_receipt_fields"]
    cases = payload["cases"]
    assert {case["route"] for case in cases} == {"plain", "structure"}
    assert {case["profile"] for case in cases} == {"smoke"}
    assert {case["batch_size"] for case in cases} == {1, 2}
    for case in cases:
        assert case["status"] == "dry_run"
        assert case["receipt_schema_version"] == 1
        assert case["receipt_scope"] == "local_only"
        assert case["local_only"] is True
        assert case["gb10_parity_claim"] is False
        assert case["hardware_label"] == "test-m4"
        assert case["mlx_version"] == case["device"]["mlx"]
        assert case["tokens_per_second"] is None
        assert case["peak_memory_bytes"] is None
        assert "matched_run_guard" in case
        assert "matched rows" in case["matched_run_guard"]
        assert case["matched_run_key"]["profile"] == case["profile"]
        assert case["matched_run_key"]["route"] == case["route"]
        assert case["matched_run_key"]["data_contract"] == "synthetic_tokens"
        assert case["workload_key"] == case["matched_run_key"]
        assert case["comparison_key"]["workload"] == case["workload_key"]
        assert case["comparison_key"]["software"] == case["software_key"]
        assert case["comparison_key"]["software"]["framework"] == "mlx"
        assert case["comparison_key"]["software"]["backend"] == case["backend"]
        assert case["software_key"]["python_version"] == case["device"]["python"]
        assert case["software_key"]["platform"] == case["device"]["platform"]
        assert case["software_key"]["machine"] == case["device"]["machine"]
        assert case["software_key"]["mlx_version"] == case["device"]["mlx"]
        assert case["software_key"]["mlx_lm_version"] == case["device"]["mlx_lm"]
        assert case["software_key"]["mlx_metal_version"] == case["device"]["mlx_metal"]
        assert case["software_key"]["default_device"] == case["device"]["default_device"]
        assert case["software_key"]["device_name"] == case["device"]["mlx_device_info"]["device_name"]
        assert case["software_key"]["metal"] == case["device"]["metal"]
        assert case["matched_run"]["key"] == case["matched_run_key"]
        assert case["matched_run"]["receipt_scope"] == "local_only"
        assert case["matched_run"]["local_only"] is True
        assert case["matched_run"]["gb10_parity_claim"] is False
        assert "No GB10 parity claim" in case["matched_run"]["claim_policy"]
        receipt = case["bench_receipt"]
        assert receipt["schema_version"] == 1
        assert receipt["receipt_scope"] == "local_only"
        assert receipt["local_only"] is True
        assert receipt["gb10_parity_claim"] is False
        assert receipt["hardware_label"] == "test-m4"
        assert receipt["route"] == case["route"]
        assert receipt["seq_len"] == case["seq_len"]
        assert receipt["batch_size"] == case["batch_size"]
        assert receipt["warmup_steps"] == case["warmup_steps"]
        assert receipt["measured_steps"] == case["measured_steps"]
        assert receipt["compile"] == case["compile"]
        assert receipt["workload"] == case["workload_key"]
        assert receipt["workload"]["data_contract"] == "synthetic_tokens"
        assert receipt["software"] == case["software_key"]
        assert receipt["comparison_key"] == case["comparison_key"]
        assert receipt["device"]["device_name"] == case["software_key"]["device_name"]
        assert receipt["timing"]["warmup_steps"] == 0
        assert receipt["timing"]["measured_steps"] == 1
        assert receipt["timing"]["compile"] is False
        assert receipt["timing"]["tokens_per_second"] is None
        assert receipt["timing"]["mean_step_time_s"] is None
        assert receipt["timing"]["wall_time_s"] is None
        assert receipt["timing"]["mean_wall_time_s"] is None
        assert receipt["timing"]["total_wall_time_s"] is None
        assert receipt["timing"]["tokens_per_second_or_step_time"] is False
        assert receipt["timing"]["synchronized_timing"] is None
        assert "claims only after both hardware labels" in receipt["parity_claim_policy"]
        assert "Single-host matrix receipt only" in receipt["local_only_policy"]
        assert case["run_metadata"]["framework"]["mlx"] == case["device"]["mlx"]
        assert case["run_metadata"]["workload"]["data_contract"] == "synthetic_tokens"
        assert case["profile_hooks"]["enabled"] is False
        assert case["config"]["vocab_size"] == 32
        assert case["config"]["model_route"] == "tiny"


def test_archive_baseline_writes_append_only_local_m4_schema(tmp_path: Path) -> None:
    archive_path = tmp_path / "m4_baselines.json"
    result = run_matrix(
        "--dry-run-json",
        "--archive-baseline",
        str(archive_path),
        "--baseline-note",
        "lane 4 schema smoke",
        "--hardware-label",
        "M4 Max",
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
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["baseline_archive"]["path"] == str(archive_path)
    assert payload["baseline_archive"]["schema_version"] == 1
    assert payload["baseline_archive"]["kind"] == "cppmega.mlx.local_m4_benchmark_baselines"
    assert payload["baseline_archive"]["record_count"] == 1
    assert payload["baseline_archive"]["receipt_scope"] == "local_only"
    assert payload["baseline_archive"]["local_only"] is True
    assert payload["baseline_archive"]["gb10_parity_claim"] is False
    assert "does not contain a GB10 parity claim" in payload["baseline_archive"]["parity_claim_policy"]
    assert "Single-host matrix receipt only" in payload["baseline_archive"]["local_only_policy"]

    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    assert archive["schema_version"] == 1
    assert archive["kind"] == "cppmega.mlx.local_m4_benchmark_baselines"
    assert archive["receipt_scope"] == "local_only"
    assert archive["local_only"] is True
    assert archive["gb10_parity_claim"] is False
    assert archive["guards"]["local_only"] is True
    assert archive["guards"]["gb10_parity_claim"] is False
    assert "local M4 baseline" in archive["guards"]["parity_claim_policy"]
    assert "Single-host matrix receipt only" in archive["guards"]["local_only_policy"]
    assert len(archive["records"]) == 1
    record = archive["records"][0]
    assert record["schema_version"] == 1
    assert record["kind"] == "cppmega.mlx.local_m4_benchmark_baseline_record"
    assert record["receipt_scope"] == "local_only"
    assert record["local_only"] is True
    assert record["gb10_parity_claim"] is False
    assert record["hardware_label"] == "M4 Max"
    assert record["note"] == "lane 4 schema smoke"
    assert record["case_count"] == 1
    assert record["source"] == {
        "script": "scripts/bench_matrix.py",
        "receipt_schema_version": 1,
        "matrix_schema_version": 1,
    }
    assert record["compare_line_contract"]["fields"] == [
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
    assert "do not reorder" in record["compare_line_contract"]["stability"]
    assert record["guards"]["local_only"] is True
    assert record["guards"]["gb10_parity_claim"] is False
    assert "does not contain a GB10 parity claim" in record["guards"]["parity_claim_policy"]
    assert "Single-host matrix receipt only" in record["guards"]["local_only_policy"]
    rows = record["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "cppmega.mlx.local_m4_benchmark_baseline_record"
    assert row["receipt_scope"] == "local_only"
    assert row["local_only"] is True
    assert row["gb10_parity_claim"] is False
    assert row["status"] == "dry_run"
    assert row["case_id"] == "smoke-route_plain-b1-s4-float32-eager"
    assert row["profile"] == "smoke"
    assert row["route"] == "plain"
    assert row["comparison_key"]["workload"] == row["workload_key"]
    assert row["comparison_key"]["software"] == row["software_key"]
    assert row["bench_receipt"]["comparison_key"] == row["comparison_key"]
    assert row["bench_receipt"]["receipt_scope"] == "local_only"
    assert row["bench_receipt"]["local_only"] is True
    assert row["bench_receipt"]["gb10_parity_claim"] is False
    assert row["bench_receipt"]["hardware_label"] == "M4 Max"
    assert row["guards"]["local_only"] is True
    assert row["guards"]["gb10_parity_claim"] is False
    assert row["metrics"] == {
        "tokens_per_second": None,
        "mean_step_time_s": None,
        "median_step_time_s": None,
        "wall_time_s": None,
        "total_wall_time_s": None,
        "peak_memory_bytes": None,
    }

    second = run_matrix(
        "--dry-run-json",
        "--archive-baseline",
        str(archive_path),
        "--hardware-label",
        "M4 Max",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "5",
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
    )

    assert second.returncode == 0, second.stderr
    appended = json.loads(archive_path.read_text(encoding="utf-8"))
    assert len(appended["records"]) == 2
    assert appended["records"][0] == record
    assert appended["records"][1]["rows"][0]["case_id"] == "smoke-route_plain-b1-s5-float32-eager"


def test_archive_baseline_refuses_existing_parity_claim(tmp_path: Path) -> None:
    archive_path = tmp_path / "m4_baselines.json"
    result = run_matrix(
        "--dry-run-json",
        "--archive-baseline",
        str(archive_path),
        "--hardware-label",
        "M4 Max",
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
    )
    assert result.returncode == 0, result.stderr
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["records"][0]["rows"][0]["bench_receipt"]["gb10_parity_claim"] = True
    archive_path.write_text(json.dumps(archive), encoding="utf-8")

    blocked = run_matrix(
        "--dry-run-json",
        "--archive-baseline",
        str(archive_path),
        "--hardware-label",
        "M4 Max",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "5",
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
    )

    assert blocked.returncode == 2
    payload = json.loads(blocked.stdout)
    assert payload["status"] == "error"
    assert "gb10_parity_claim must be false" in payload["error"]
    assert len(json.loads(archive_path.read_text(encoding="utf-8"))["records"]) == 1


def test_archive_baseline_refuses_existing_archive_without_local_only_guard(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "m4_baselines.json"
    archive_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "cppmega.mlx.local_m4_benchmark_baselines",
                "created_at_utc": "2026-04-30T00:00:00Z",
                "updated_at_utc": "2026-04-30T00:00:00Z",
                "records": [],
                "guards": {
                    "local_only": True,
                    "gb10_parity_claim": False,
                    "matched_run_guard": "legacy",
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_matrix(
        "--dry-run-json",
        "--archive-baseline",
        str(archive_path),
        "--hardware-label",
        "M4 Max",
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
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "baseline archive receipt_scope must be 'local_only'" in payload["error"]


def test_minimal_real_matrix_reports_comparable_metrics() -> None:
    result = run_matrix(
        "--json",
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
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["receipt_scope"] == "local_only"
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
    assert payload["case_count"] == 1
    case = payload["cases"][0]
    assert case["status"] == "ok"
    assert case["receipt_schema_version"] == 1
    assert case["receipt_scope"] == "local_only"
    assert case["local_only"] is True
    assert case["gb10_parity_claim"] is False
    assert case["case_id"] == "smoke-route_plain-b1-s4-float32-eager"
    assert case["hardware_label"] == "test-m4"
    assert case["mlx_version"]
    assert case["dtype"] == "float32"
    assert case["batch_size"] == 1
    assert case["seq_len"] == 4
    assert case["profile"] == "smoke"
    assert case["route"] == "plain"
    assert case["compile"] is False
    assert case["tokens_per_second"] > 0
    assert case["mean_step_time_s"] > 0
    assert case["wall_time_s"] == case["mean_step_time_s"]
    assert case["mean_wall_time_s"] == case["mean_step_time_s"]
    assert case["total_wall_time_s"] == sum(case["step_times_s"])
    assert case["median_step_time_s"] > 0
    assert case["peak_memory_bytes"] >= 0
    assert case["memory"]["peak_bytes"] == case["peak_memory_bytes"]
    assert case["profile_hooks"]["enabled"] is True
    assert "measured_steps" in case["profile_hooks"]["scopes"]
    assert case["matched_run_key"]["profile"] == "smoke"
    assert case["matched_run_key"]["route"] == "plain"
    assert case["matched_run_key"]["dtype"] == "float32"
    assert case["workload_key"] == case["matched_run_key"]
    assert case["comparison_key"]["workload"] == case["matched_run_key"]
    assert case["comparison_key"]["software"]["framework"] == "mlx"
    assert case["comparison_key"]["software"]["backend"] == case["backend"]
    assert case["comparison_key"]["software"]["mlx_version"] == case["mlx_version"]
    assert case["comparison_key"]["software"]["mlx_lm_version"] == case["mlx_lm_version"]
    assert case["comparison_key"]["software"]["mlx_metal_version"] == case["device"]["mlx_metal"]
    receipt = case["bench_receipt"]
    assert receipt["receipt_scope"] == "local_only"
    assert receipt["local_only"] is True
    assert receipt["gb10_parity_claim"] is False
    assert receipt["hardware_label"] == "test-m4"
    assert receipt["profile"] == "smoke"
    assert receipt["route"] == "plain"
    assert receipt["seq_len"] == 4
    assert receipt["batch_size"] == 1
    assert receipt["warmup_steps"] == 0
    assert receipt["measured_steps"] == 1
    assert receipt["compile"] is False
    assert receipt["tokens_per_second"] == case["tokens_per_second"]
    assert receipt["mean_step_time_s"] == case["mean_step_time_s"]
    assert receipt["wall_time_s"] == case["wall_time_s"]
    assert receipt["mean_wall_time_s"] == case["mean_wall_time_s"]
    assert receipt["total_wall_time_s"] == case["total_wall_time_s"]
    assert receipt["median_step_time_s"] == case["median_step_time_s"]
    assert receipt["device"]["default_device"] == case["software_key"]["default_device"]
    assert receipt["software"] == case["software_key"]
    assert receipt["workload"] == case["workload_key"]
    assert receipt["timing"]["warmup_steps"] == 0
    assert receipt["timing"]["measured_steps"] == 1
    assert receipt["timing"]["compile"] is False
    assert receipt["timing"]["first_call_time_s"] == case["first_call_time_s"]
    assert receipt["timing"]["compile_time_s"] == case["compile_time_s"]
    assert receipt["timing"]["mean_step_time_s"] == case["mean_step_time_s"]
    assert receipt["timing"]["wall_time_s"] == case["wall_time_s"]
    assert receipt["timing"]["mean_wall_time_s"] == case["mean_wall_time_s"]
    assert receipt["timing"]["total_wall_time_s"] == case["total_wall_time_s"]
    assert receipt["timing"]["median_step_time_s"] == case["median_step_time_s"]
    assert receipt["timing"]["tokens_per_second"] == case["tokens_per_second"]
    assert receipt["timing"]["tokens_per_second_or_step_time"] is True
    assert receipt["timing"]["synchronized_timing"] is True
    assert len(receipt["timing"]["step_times_s"]) == 1
    assert "both rows" in case["matched_run"]["guard"]


def test_dry_run_json_expands_hybrid_model_routes() -> None:
    result = run_matrix(
        "--dry-run-json",
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "4",
        "--profiles",
        "smoke",
        "--routes",
        "hybrid-a,hybrid-e,hybrid-m,hybrid-r",
        "--compile-modes",
        "eager",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    cases = payload["cases"]
    assert {case["route"] for case in cases} == {
        "hybrid-a",
        "hybrid-e",
        "hybrid-m",
        "hybrid-r",
    }
    for case in cases:
        assert case["config"]["model_route"] == case["route"]
        assert case["config"]["include_structure"] is True
        assert case["matched_run_key"]["model_route"] == case["route"]
        assert case["matched_run_key"]["profile"] == "smoke"
        assert case["matched_run_key"]["route"] == case["route"]
        assert case["model_source"] == "cppmega_mlx.models.hybrid_lm"
        assert case["route_plan"]["route_symbols"]
        assert case["backend_plan"]["backend_summary"]


def test_dry_run_json_expands_named_mamba_m2rnn_alias_rows() -> None:
    result = run_matrix(
        "--dry-run-json",
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "4",
        "--profiles",
        "hybrid-smoke",
        "--routes",
        "mamba3,m2rnn,hybrid-aemr",
        "--compile-modes",
        "eager",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "dry_run"
    assert payload["case_count"] == 3
    cases = {case["route"]: case for case in payload["cases"]}
    assert set(cases) == {"mamba3", "m2rnn", "hybrid-aemr"}

    expected = {
        "mamba3": ("hybrid-m", "M", {"mamba3": 1}),
        "m2rnn": ("hybrid-r", "R", {"m2rnn": 1}),
        "hybrid-aemr": (
            "hybrid",
            "AEMR",
            {"attention": 1, "moe": 1, "m2rnn": 1, "mamba3": 1},
        ),
    }
    for route, (model_route, symbols, backend_summary) in expected.items():
        case = cases[route]
        assert case["requested_route"] == route
        assert case["route_alias"] == route
        assert case["resolved_model_route"] == model_route
        assert case["is_route_alias"] is True
        assert case["config"]["model_route"] == model_route
        assert case["config"]["include_structure"] is True
        assert case["matched_run_key"]["route"] == route
        assert case["matched_run_key"]["model_route"] == model_route
        assert case["workload_key"]["route"] == route
        assert case["comparison_key"]["workload"]["route"] == route
        assert case["route_plan"]["model_route"] == model_route
        assert case["route_plan"]["route_symbols"] == symbols
        assert case["backend_plan"]["backend_summary"] == backend_summary
        receipt = case["bench_receipt"]
        assert receipt["route"] == route
        assert receipt["model_route"] == model_route
        assert receipt["requested_route"] == route
        assert receipt["route_alias"] == route
        assert receipt["resolved_model_route"] == model_route
        assert receipt["is_route_alias"] is True
        assert receipt["workload"] == case["workload_key"]
        assert receipt["comparison_key"] == case["comparison_key"]


def test_minimal_real_hybrid_route_reports_backend_metadata() -> None:
    result = run_matrix(
        "--json",
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "4",
        "--profiles",
        "smoke",
        "--routes",
        "hybrid-e",
        "--compile-modes",
        "eager",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    case = payload["cases"][0]
    assert case["status"] == "ok"
    assert case["route"] == "hybrid-e"
    assert case["requested_route"] == "hybrid-e"
    assert case["is_route_alias"] is False
    assert case["config"]["model_route"] == "hybrid-e"
    assert case["model_source"] == "cppmega_mlx.models.hybrid_lm"
    assert case["route_plan"]["route_symbols"] == "E"
    assert case["route_plan"]["route_roles"] == ["moe"]
    assert case["backend_plan"]["backend_summary"] == {"moe": 1}
    assert case["matched_run_key"]["model_route"] == "hybrid-e"
    assert case["matched_run_key"]["data_contract"] == "synthetic_tokens"
    assert case["software_key"]["framework"] == "mlx"
    assert case["software_key"]["framework_backend"] == "metal"
    assert case["comparison_key"]["workload"] == case["workload_key"]
    assert case["comparison_key"]["software"] == case["software_key"]
    context = case["profile_hooks"]["scopes"]["measured_steps"]["extra"]["context"]
    assert context["route"] == "hybrid-e"
    assert context["backend"] == "mlx"
    assert context["device"]
    assert context["backend_plan"]["backend_summary"] == {"moe": 1}
    receipt = case["bench_receipt"]
    assert receipt["hardware_label"] == "test-m4"
    assert receipt["route"] == "hybrid-e"
    assert receipt["model_route"] == "hybrid-e"
    assert receipt["requested_route"] == "hybrid-e"
    assert receipt["is_route_alias"] is False
    assert receipt["workload"]["data_contract"] == "synthetic_tokens"
    assert receipt["workload"]["backend_plan"]["backend_summary"] == {"moe": 1}
    assert receipt["software"] == case["software_key"]
    assert receipt["comparison_key"] == case["comparison_key"]
    assert receipt["timing"]["synchronized_timing"] is True
    assert receipt["timing"]["timing_method"].startswith("wall-clock timing around MLX")


def test_minimal_real_named_custom_routes_report_timing_metadata() -> None:
    result = run_matrix(
        "--json",
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "4",
        "--profiles",
        "hybrid-smoke",
        "--routes",
        "mamba3,m2rnn,hybrid-aemr",
        "--compile-modes",
        "eager",
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
    assert payload["case_count"] == 3
    cases = {case["route"]: case for case in payload["cases"]}
    expected = {
        "mamba3": ("hybrid-m", "M", {"mamba3": 1}),
        "m2rnn": ("hybrid-r", "R", {"m2rnn": 1}),
        "hybrid-aemr": (
            "hybrid",
            "AEMR",
            {"attention": 1, "moe": 1, "m2rnn": 1, "mamba3": 1},
        ),
    }
    for route, (model_route, symbols, backend_summary) in expected.items():
        case = cases[route]
        assert case["status"] == "ok"
        assert case["tokens_per_second"] > 0
        assert case["mean_step_time_s"] > 0
        assert case["wall_time_s"] == case["mean_step_time_s"]
        assert case["mean_wall_time_s"] == case["mean_step_time_s"]
        assert case["total_wall_time_s"] == sum(case["step_times_s"])
        assert case["requested_route"] == route
        assert case["resolved_model_route"] == model_route
        assert case["route_plan"]["route_symbols"] == symbols
        assert case["backend_plan"]["backend_summary"] == backend_summary
        assert case["software_key"]["mlx_version"] == case["device"]["mlx"]
        assert case["software_key"]["mlx_lm_version"] == case["device"]["mlx_lm"]
        assert case["software_key"]["mlx_metal_version"] == case["device"]["mlx_metal"]
        assert case["software_key"]["metal"] == case["device"]["metal"]
        assert case["comparison_key"]["workload"]["route"] == route
        assert case["comparison_key"]["software"] == case["software_key"]
        assert case["workload_key"]["data_contract"] == "synthetic_tokens"
        measured = case["profile_hooks"]["scopes"]["measured_steps"]
        assert measured["tokens"] == case["tokens_per_step"]
        assert measured["wall_time_s"] == measured["seconds"]
        assert measured["elapsed_wall_time_s"] == measured["seconds"]
        assert measured["synchronized"] is True
        assert measured["evaluated"] is True
        receipt = case["bench_receipt"]
        assert receipt["route"] == route
        assert receipt["model_route"] == model_route
        assert receipt["workload"]["data_contract"] == "synthetic_tokens"
        assert receipt["software"] == case["software_key"]
        assert receipt["wall_time_s"] == case["wall_time_s"]
        assert receipt["mean_wall_time_s"] == case["mean_wall_time_s"]
        assert receipt["total_wall_time_s"] == case["total_wall_time_s"]
        assert receipt["timing"]["wall_time_s"] == case["wall_time_s"]
        assert receipt["timing"]["mean_wall_time_s"] == case["mean_wall_time_s"]
        assert receipt["timing"]["total_wall_time_s"] == case["total_wall_time_s"]
        assert receipt["timing"]["tokens_per_second_or_step_time"] is True
        assert receipt["timing"]["synchronized_timing"] is True


def test_human_output_includes_route_backend_and_device_metadata() -> None:
    result = run_matrix(
        "--dry-run-json",
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "4",
        "--profiles",
        "smoke",
        "--routes",
        "hybrid-m,hybrid-r",
        "--compile-modes",
        "eager",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["cases"][0]["backend_summary"] == {"mamba3": 1}
    assert payload["cases"][1]["backend_summary"] == {"m2rnn": 1}

    human = run_matrix(
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "4",
        "--profiles",
        "smoke",
        "--routes",
        "hybrid-m,hybrid-r",
        "--compile-modes",
        "eager",
        "--dtype",
        "float32",
        "--steps",
        "1",
        "--warmup-steps",
        "0",
    )

    assert human.returncode == 0, human.stderr
    assert "tokens_per_second mean_step_time_s median_step_time_s" in human.stdout
    assert "model_route route_symbols backend backend_summary device_name" in human.stdout
    assert "local_only_policy: Single-host matrix receipt only" in human.stdout
    assert "gb10_parity_claim: False" in human.stdout
    assert "hybrid-m M mlx mamba3:1" in human.stdout
    assert "hybrid-r R mlx m2rnn:1" in human.stdout
    assert '"Apple M4 Max"' in human.stdout or '"Device(gpu, 0)"' in human.stdout


def test_jsonl_outputs_one_case_per_line() -> None:
    result = run_matrix(
        "--jsonl",
        "--hardware-label",
        "test-m4",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "4,5",
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
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert [row["seq_len"] for row in rows] == [4, 5]
    assert all(row["tokens_per_second"] > 0 for row in rows)


def test_invalid_profile_returns_error_json() -> None:
    result = run_matrix("--dry-run-json", "--profiles", "missing")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "unknown profile" in payload["error"]
