"""V7-I05: production-preset baseline bench smoke + JSON shape gate."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_v7_i05_bench_production_presets_smoke(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_production_presets",
         "--presets", "llama3_8b",
         "--hidden", "256", "--num-steps", "2",
         "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=120)
    assert rc.returncode == 0, rc.stderr

    stable = tmp_path / "production_preset_baseline_latest.json"
    assert stable.exists()
    data = json.loads(stable.read_text())
    assert "date_utc" in data
    assert "rows" in data and len(data["rows"]) >= 1
    row = data["rows"][0]
    for k in ("preset", "hidden", "n_layers", "num_steps", "status",
              "ms_per_step_warm", "peak_memory_mb", "total_elapsed_s"):
        assert k in row, f"missing {k} in baseline row"
    assert row["preset"] == "llama3_8b"
    assert row["status"] == "ok"
    assert float(row["ms_per_step_warm"]) > 0
