from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def run_bench(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.bench_inference_quality", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_inference_quality_smoke_reports_q4_arc_mmlu_humaneval(tmp_path):
    out_path = tmp_path / "quality.json"

    result = run_bench(
        "--json",
        "--out",
        str(out_path),
        "--suites",
        "arc",
        "mmlu",
        "humaneval",
        "--dtype",
        "float32",
        "--seed",
        "175",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == json.loads(out_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["receipt_scope"] == "local_inference_quality_smoke"
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
    assert payload["leaderboard_claim"] is False
    assert payload["dataset_source"] == "built_in_token_id_smoke"
    assert payload["quantization"]["bits"] == 4
    assert payload["quantization"]["group_size"] == 32
    assert payload["quantization"]["quantized_linear_modules"] > 0

    rows = payload["suite_rows"]
    assert {row["suite"] for row in rows} == {"arc", "mmlu", "humaneval"}
    for row in rows:
        assert row["status"] == "ok"
        assert row["num_tasks"] >= 1
        assert 0.0 <= row["score"] <= 1.0
        assert row["metric"] in {"accuracy", "pass_at_1_exact_token_match"}


def test_inference_quality_rejects_malformed_task_rows(tmp_path):
    bad_tasks = tmp_path / "bad_tasks.jsonl"
    bad_tasks.write_text(json.dumps({"suite": "unknown", "prompt_ids": [1]}) + "\n")

    result = run_bench(
        "--json",
        "--tasks-jsonl",
        str(bad_tasks),
        "--out",
        str(tmp_path / "blocked.json"),
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "suite" in payload["error"]


def test_inference_quality_help_lists_task_and_quantization_flags():
    result = run_bench("--help")

    assert result.returncode == 0
    assert "--tasks-jsonl" in result.stdout
    assert "--suites" in result.stdout
    assert "--bits" in result.stdout
    assert "--group-size" in result.stdout

