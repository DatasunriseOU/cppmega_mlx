from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def run_bench(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.bench_inference_long_context", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_inference_long_context_smoke_reports_niah_and_ruler_kv_q4(tmp_path):
    out_path = tmp_path / "long_context.json"

    result = run_bench(
        "--json",
        "--out",
        str(out_path),
        "--suites",
        "niah",
        "ruler",
        "--dtype",
        "float32",
        "--context-tokens",
        "16",
        "--decode-tokens",
        "1",
        "--warmup-steps",
        "0",
        "--measured-steps",
        "1",
        "--seed",
        "176",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == json.loads(out_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["receipt_scope"] == "local_kv_q4_long_context_smoke"
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
    assert payload["leaderboard_claim"] is False
    assert payload["full_model_long_context_claim"] is False
    assert payload["dataset_source"] == "built_in_token_id_smoke"

    kv = payload["kv_cache"]
    assert kv["quantized"] is True
    assert kv["bits"] == 4
    assert kv["group_size"] == 32
    assert kv["quantized_kv_start"] == 0

    rows = payload["rows"]
    assert {row["suite"] for row in rows} == {"niah", "ruler"}
    for row in rows:
        assert row["status"] == "ok"
        assert row["task_class"] in {"needle_in_haystack", "ruler_variable_tracking"}
        assert row["context_length"] == 16
        assert row["generated_tokens"] == 1
        assert row["kv_cache"]["final_position"] == 17
        assert row["kv_cache"]["quantized_layer_count"] > 0
        assert row["timing"]["tokens_per_second"] > 0
        assert isinstance(row["exact_match"], bool)
        assert 0.0 <= row["score"] <= 1.0


def test_inference_long_context_rejects_malformed_task_rows(tmp_path):
    bad_tasks = tmp_path / "bad_tasks.jsonl"
    bad_tasks.write_text(
        json.dumps({"suite": "niah", "task_id": "bad", "context_ids": []}) + "\n"
    )

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
    assert "context_ids" in payload["error"]


def test_inference_long_context_help_lists_kv_and_context_flags():
    result = run_bench("--help")

    assert result.returncode == 0
    assert "--tasks-jsonl" in result.stdout
    assert "--context-tokens" in result.stdout
    assert "--kv-bits" in result.stdout
    assert "--kv-group-size" in result.stdout
    assert "--quantized-kv-start" in result.stdout
