from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def run_bench(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.bench_inference_throughput", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_inference_throughput_smoke_reports_prefill_and_decode_rows(tmp_path):
    out_path = tmp_path / "inference_throughput.json"

    result = run_bench(
        "--json",
        "--out",
        str(out_path),
        "--profiles",
        "qwen3_4b_class",
        "nam56r_class",
        "--dtype",
        "float32",
        "--batch-size",
        "1",
        "--prefill-tokens",
        "4",
        "--decode-tokens",
        "2",
        "--warmup-steps",
        "0",
        "--measured-steps",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == json.loads(out_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["receipt_scope"] == "local_inference_throughput_smoke"
    assert payload["local_only"] is True
    assert payload["gb10_parity_claim"] is False
    assert payload["full_model_throughput_claim"] is False
    assert payload["scale"] == "smoke"
    assert payload["batch_size"] == 1
    assert payload["prefill_tokens"] == 4
    assert payload["decode_tokens"] == 2

    rows = payload["rows"]
    assert {row["profile"] for row in rows} == {"qwen3_4b_class", "nam56r_class"}
    for row in rows:
        assert row["status"] == "ok"
        assert row["profile_class"] in {"qwen3-4b-class", "nam56r-class"}
        assert row["actual_config"]["scale"] == "smoke"
        assert row["actual_config"]["hidden_size"] <= 32
        assert row["prefill"]["tokens_per_second"] > 0
        assert row["prefill"]["tokens"] == 4
        assert row["decode"]["tokens_per_second"] > 0
        assert row["decode"]["tokens"] == 2
        assert row["decode"]["mode"] in {"contiguous_kv_cache", "eager_full_prefix"}
        assert row["memory_safety"]["estimated_model_bytes"] < 10 * 1024**3


def test_inference_throughput_full_scale_requires_explicit_large_allowance(tmp_path):
    result = run_bench(
        "--json",
        "--out",
        str(tmp_path / "blocked.json"),
        "--scale",
        "full",
        "--profiles",
        "nam56r_class",
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert "--allow-large" in payload["error"]


def test_inference_throughput_help_lists_profile_and_shape_flags():
    result = run_bench("--help")

    assert result.returncode == 0
    assert "--profiles" in result.stdout
    assert "--prefill-tokens" in result.stdout
    assert "--decode-tokens" in result.stdout
    assert "--allow-large" in result.stdout

