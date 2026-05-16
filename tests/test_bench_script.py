from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


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
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_only"
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
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
    assert payload["matched_run"]["receipt_scope"] == "local_only"
    assert payload["matched_run"]["local_only"] is True
    assert payload["matched_run"]["gb10_parity_claim"] is False
    assert payload["matched_run"]["key"]["dtype"] == "bfloat16"
    assert payload["matched_run_key"] == payload["workload_key"]
    assert payload["workload_key"] == payload["matched_run"]["key"]
    assert payload["software_key"]["framework"] == "mlx"
    assert payload["software_key"]["backend"] == payload["backend"]
    assert payload["software_key"]["execution_backend"] == payload["backend"]
    assert payload["software_key"]["framework_backend"] == "metal"
    assert payload["software_key"]["python_version"] == payload["device"]["python"]
    assert payload["software_key"]["platform"] == payload["device"]["platform"]
    assert payload["software_key"]["machine"] == payload["device"]["machine"]
    assert payload["software_key"]["mlx_version"] == payload["device"]["mlx"]
    assert payload["software_key"]["mlx_lm_version"] == payload["device"]["mlx_lm"]
    assert payload["software_key"]["mlx_metal_version"] == payload["device"]["mlx_metal"]
    assert payload["software_key"]["default_device"] == payload["device"]["default_device"]
    assert payload["software_key"]["device_name"] == payload["device"]["mlx_device_info"]["device_name"]
    assert payload["software_key"]["metal"] == payload["device"]["metal"]
    assert payload["comparison_key"]["workload"] == payload["workload_key"]
    assert payload["comparison_key"]["software"] == payload["software_key"]
    receipt = payload["bench_receipt"]
    assert receipt["schema_version"] == 1
    assert receipt["receipt_scope"] == "local_only"
    assert receipt["local_only"] is True
    assert receipt["gb10_parity_claim"] is False
    assert receipt["hardware_label"] == payload["hardware_label"]
    assert receipt["seq_len"] == payload["seq_len"]
    assert receipt["batch_size"] == payload["batch_size"]
    assert receipt["dtype"] == payload["dtype"]
    assert receipt["warmup_steps"] == payload["warmup_steps"]
    assert receipt["measured_steps"] == payload["measured_steps"]
    assert receipt["compile"] == payload["compile"]
    assert receipt["include_structure"] == payload["include_structure"]
    assert receipt["software"] == payload["software_key"]
    assert receipt["workload"] == payload["workload_key"]
    assert receipt["comparison_key"] == payload["comparison_key"]
    assert receipt["timing"]["tokens_per_step"] == payload["tokens_per_step"]
    assert receipt["timing"]["warmup_steps"] == 0
    assert receipt["timing"]["measured_steps"] == 1
    assert receipt["timing"]["compile"] is True
    assert receipt["timing"]["tokens_per_second"] is None
    assert receipt["timing"]["mean_step_time_s"] is None
    assert receipt["timing"]["wall_time_s"] is None
    assert receipt["timing"]["tokens_per_second_or_step_time"] is False
    assert receipt["timing"]["synchronized_timing"] is None
    assert "Single-host tiny benchmark receipt only" in receipt["local_only_policy"]
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
    assert payload["receipt_schema_version"] == 1
    assert payload["receipt_scope"] == "local_only"
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
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
    assert payload["matched_run"]["receipt_scope"] == "local_only"
    assert payload["matched_run"]["local_only"] is True
    assert payload["matched_run"]["gb10_parity_claim"] is False
    assert payload["matched_run_key"] == payload["workload_key"]
    assert payload["workload_key"] == payload["matched_run"]["key"]
    assert payload["software_key"]["framework"] == "mlx"
    assert payload["software_key"]["backend"] == payload["backend"]
    assert payload["software_key"]["execution_backend"] == payload["backend"]
    assert payload["software_key"]["framework_backend"] == "metal"
    assert payload["software_key"]["mlx_version"] == payload["device"]["mlx"]
    assert payload["software_key"]["mlx_lm_version"] == payload["device"]["mlx_lm"]
    assert payload["software_key"]["mlx_metal_version"] == payload["device"]["mlx_metal"]
    assert payload["software_key"]["default_device"] == payload["device"]["default_device"]
    assert payload["software_key"]["device_name"] == payload["device"]["mlx_device_info"]["device_name"]
    assert payload["comparison_key"]["workload"] == payload["workload_key"]
    assert payload["comparison_key"]["software"] == payload["software_key"]
    receipt = payload["bench_receipt"]
    assert receipt["receipt_scope"] == "local_only"
    assert receipt["local_only"] is True
    assert receipt["gb10_parity_claim"] is False
    assert receipt["hardware_label"] == payload["hardware_label"]
    assert receipt["seq_len"] == payload["seq_len"]
    assert receipt["batch_size"] == payload["batch_size"]
    assert receipt["dtype"] == payload["dtype"]
    assert receipt["warmup_steps"] == payload["warmup_steps"]
    assert receipt["measured_steps"] == payload["measured_steps"]
    assert receipt["compile"] == payload["compile"]
    assert receipt["include_structure"] == payload["include_structure"]
    assert receipt["tokens_per_second"] == payload["tokens_per_second"]
    assert receipt["mean_step_time_s"] == payload["mean_step_time_s"]
    assert receipt["wall_time_s"] == payload["wall_time_s"]
    assert receipt["mean_wall_time_s"] == payload["mean_wall_time_s"]
    assert receipt["total_wall_time_s"] == payload["total_wall_time_s"]
    assert receipt["median_step_time_s"] == payload["median_step_time_s"]
    assert receipt["software"] == payload["software_key"]
    assert receipt["workload"] == payload["workload_key"]
    assert receipt["comparison_key"] == payload["comparison_key"]
    assert receipt["timing"]["tokens_per_step"] == payload["tokens_per_step"]
    assert receipt["timing"]["warmup_steps"] == 0
    assert receipt["timing"]["measured_steps"] == 1
    assert receipt["timing"]["compile"] is False
    assert receipt["timing"]["first_call_time_s"] == payload["first_call_time_s"]
    assert receipt["timing"]["compile_time_s"] == payload["compile_time_s"]
    assert receipt["timing"]["mean_step_time_s"] == payload["mean_step_time_s"]
    assert receipt["timing"]["wall_time_s"] == payload["wall_time_s"]
    assert receipt["timing"]["mean_wall_time_s"] == payload["mean_wall_time_s"]
    assert receipt["timing"]["total_wall_time_s"] == payload["total_wall_time_s"]
    assert receipt["timing"]["median_step_time_s"] == payload["median_step_time_s"]
    assert receipt["timing"]["tokens_per_second"] == payload["tokens_per_second"]
    assert receipt["timing"]["tokens_per_second_or_step_time"] is True
    assert receipt["timing"]["synchronized_timing"] is True
    assert len(receipt["timing"]["step_times_s"]) == 1
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


def test_hybrid_r_path_c_benchmark_startup_uses_explicit_m2rnn_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mlx.core as mx

    from cppmega_mlx.nn._tilelang import m2rnn_path_c
    from scripts import bench_tiny

    seen: list[dict[str, tuple[int, ...]]] = []

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH", "path_c")
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_mapped_packed_path_c_status",
        lambda *_args, **_kwargs: m2rnn_path_c.M2RNNPathCStatus(True, "forced available"),
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_post_residual_gate_path_c_status",
        lambda *_args, **_kwargs: m2rnn_path_c.M2RNNPathCStatus(True, "forced available"),
    )

    def fake_recurrence_path_c(
        conv_input: mx.array,
        W: mx.array,
        xf: mx.array,
        h0: mx.array,
        **_kwargs: object,
    ) -> tuple[mx.array, mx.array]:
        batch, seq, _conv_dim = conv_input.shape
        heads = h0.shape[1]
        v_dim = W.shape[-1]
        seen.append(
            {
                "W": tuple(W.shape),
                "xf": tuple(xf.shape),
                "h0": tuple(h0.shape),
            }
        )
        return mx.zeros((batch, seq, heads * v_dim), dtype=conv_input.dtype), h0

    def fake_post_path_c(
        y: mx.array,
        _conv_input: mx.array,
        _D: mx.array,
        _projected: mx.array,
        **_kwargs: object,
    ) -> mx.array:
        return y

    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_mapped_packed_with_state_path_c",
        fake_recurrence_path_c,
    )
    monkeypatch.setattr(
        m2rnn_path_c,
        "m2rnn_apply_post_residual_gate_path_c",
        fake_post_path_c,
    )

    payload = bench_tiny.run_benchmark(
        bench_tiny.BenchConfig(
            batch_size=1,
            seq_len=4,
            vocab_size=32,
            d_model=8,
            n_heads=1,
            n_layers=1,
            mlp_dim=16,
            dtype="float32",
            warmup_steps=0,
            steps=1,
            compile=False,
            model_route="hybrid-r",
            include_structure=True,
        )
    )

    assert payload["status"] == "ok"
    assert payload["model_route"] == "hybrid-r"
    assert seen
    assert {item["W"] for item in seen} == {(1, 8, 8)}
    assert {item["xf"] for item in seen} == {(1, 3, 1)}
    assert {item["h0"] for item in seen} == {(1, 1, 8, 8)}


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


def test_wired_limit_report_uses_memory_helper_with_fake_mlx(
    monkeypatch,
) -> None:
    import scripts.bench_tiny as bench

    calls: list[tuple[str, int]] = []

    fake_mx = SimpleNamespace(
        set_wired_limit=lambda value: calls.append(("wired", value)) or 123,
        metal=SimpleNamespace(
            set_memory_limit=lambda value: calls.append(("metal", value)) or 456,
        ),
    )
    monkeypatch.setattr(
        bench,
        "device_memory_limits",
        lambda: {
            "memory_size_bytes": 1000,
            "max_recommended_working_set_size_bytes": 700,
        },
    )
    monkeypatch.setattr(bench, "metal_is_available", lambda: True)

    dry = bench.wired_limit_report(
        bench.BenchConfig(wired_limit_bytes=500),
        apply=False,
        mx_module=fake_mx,
    )
    assert dry["memory_limit_plan"] == {
        "metal_limit_bytes": 850,
        "metal_ratio": 0.85,
        "total_bytes": 1000,
        "wired_limit_bytes": 500,
        "wired_ratio": 0.5,
    }
    assert dry["applied"] is False
    assert calls == []

    applied = bench.wired_limit_report(
        bench.BenchConfig(wired_limit_bytes=500),
        apply=True,
        mx_module=fake_mx,
    )
    assert applied["applied"] is True
    assert applied["applied_bytes"] == 500
    assert applied["previous_bytes"] == 123
    assert applied["previous_metal_limit_bytes"] == 456
    assert calls == [("wired", 500), ("metal", 850)]
