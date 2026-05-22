"""V7-F07: per-preset inference latency bench smoke."""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_v7_f07_inference_latency_bench_smoke(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_inference_latency",
         "--presets", "llama3_8b", "--hidden", "128",
         "--max-new-tokens", "4", "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=120)
    assert rc.returncode == 0, rc.stderr
    csvs = list(tmp_path.glob("cppmega_inference_latency_*.csv"))
    htmls = list(tmp_path.glob("cppmega_inference_latency_*.html"))
    assert csvs and htmls
    with csvs[0].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    row = rows[0]
    for k in ("preset", "B", "S", "hidden", "max_new_tokens",
              "ms_per_token", "tokens_per_s", "wallclock_s"):
        assert k in row, f"missing {k}"
    assert row["preset"] == "llama3_8b"
    assert float(row["ms_per_token"]) > 0
    assert float(row["tokens_per_s"]) > 0
