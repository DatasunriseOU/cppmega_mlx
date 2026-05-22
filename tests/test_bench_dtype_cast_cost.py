"""V7-D06: cast-cost bench smoke + shape gate."""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_v7_d06_bench_dtype_cast_cost_smoke(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_dtype_cast_cost",
         "--n-iter", "3", "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=120)
    assert rc.returncode == 0, rc.stderr

    csv_path = tmp_path / "bench_dtype_cast_cost.csv"
    json_path = tmp_path / "bench_dtype_cast_cost.json"
    html_path = tmp_path / "bench_dtype_cast_cost.html"
    assert csv_path.exists() and json_path.exists() and html_path.exists()

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    dtypes = {r["dtype"] for r in rows}
    for d in ("fp32", "bf16", "fp16"):
        assert d in dtypes, f"missing dtype row {d}"
    for r in rows:
        # AC: cast_overhead_ms column present.
        assert "cast_overhead_ms" in r
        assert float(r["cast_overhead_ms"]) >= 0
        assert float(r["fwd_ms"]) > 0
        assert float(r["fwdbwd_ms"]) > 0

    parsed = json.loads(json_path.read_text())
    assert "rows" in parsed and len(parsed["rows"]) >= 3
