"""V7-I05: production-preset baseline bench smoke + JSON shape gate +
real wall-clock receipt at llama3_8b H=4096 (item 50)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _device_hbm_bytes() -> int | None:
    try:
        import mlx.core as mx  # noqa
        if hasattr(mx, "metal"):
            info = mx.metal.device_info()
            return int(info.get("memory_size", 0))
    except Exception:
        return None
    return None


def test_v7_i05_bench_production_presets_smoke(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_production_presets",
         "--presets", "llama3_8b",
         "--hidden", "256", "--num-steps", "3",
         "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=180)
    assert rc.returncode == 0, rc.stderr

    stable = tmp_path / "production_preset_baseline_latest.json"
    assert stable.exists()
    data = json.loads(stable.read_text())
    assert "date_utc" in data
    assert "rows" in data and len(data["rows"]) >= 1
    row = data["rows"][0]
    for k in ("preset", "hidden", "n_layers", "num_steps", "S",
              "compile_mode", "status", "first_step_ms",
              "ms_per_step_warm", "ms_per_step_warm_min",
              "per_step_ms", "peak_memory_mb", "total_elapsed_s",
              "compile_engaged", "compile_status"):
        assert k in row, f"missing {k} in baseline row"
    assert row["preset"] == "llama3_8b"
    assert row["status"] == "ok"
    assert float(row["ms_per_step_warm"]) > 0
    assert float(row["first_step_ms"]) > 0
    assert len(row["per_step_ms"]) == row["num_steps"]


def test_v7_i05_item50_llama3_8b_h4096_wall_clock_receipt(tmp_path):
    """Item 50: prove the production preset llama3_8b @ H=4096 can
    actually run a wall-clock baseline. Skipped on hosts without
    >=16 GB HBM (the H=4096 baseline peak is ~12 GB)."""
    hbm = _device_hbm_bytes()
    if hbm is None:
        pytest.skip("no Metal device info")
    if hbm < 16 * 1024 ** 3:
        pytest.skip(f"device HBM {hbm / 1024 ** 3:.1f} GB < 16 GB cap")
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_production_presets",
         "--presets", "llama3_8b",
         "--hidden", "4096", "--num-steps", "4",
         "--skip-loss-guard", "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=600)
    assert rc.returncode == 0, rc.stderr
    stable = tmp_path / "production_preset_baseline_latest.json"
    data = json.loads(stable.read_text())
    row = data["rows"][0]
    assert row["preset"] == "llama3_8b"
    assert row["hidden"] == 4096
    assert row["status"] == "ok", f"H=4096 bench failed: {row}"
    assert float(row["ms_per_step_warm"]) > 0
    # Peak must fit under 70% of HBM — sanity that the baseline
    # row is a realistic-not-OOM measurement.
    peak_bytes = float(row["peak_memory_mb"]) * 1024 ** 2
    assert peak_bytes / hbm < 0.70, (
        f"H=4096 peak {peak_bytes / 1024 ** 3:.2f} GB "
        f"= {peak_bytes / hbm * 100:.1f}% of HBM (>70% cap)")
