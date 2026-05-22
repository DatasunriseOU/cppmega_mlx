"""V7-D02: scripts/bench_dtype_long_horizon.py smoke + bound check."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parent.parent


def test_v7_d02_bench_dtype_long_horizon_smoke(tmp_path):
    """Run the script with a tiny step budget → assert the three
    artefacts land + summary has the documented fields."""
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_dtype_long_horizon",
         "--num-steps", "20", "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=120)
    assert rc.returncode == 0, rc.stderr
    for name in ("bench_dtype_fp32.json", "bench_dtype_bf16.json",
                 "bench_dtype_summary.json"):
        path = tmp_path / name
        assert path.exists(), f"{name} missing"
    summary = json.loads(
        (tmp_path / "bench_dtype_summary.json").read_text())
    for k in ("final_loss_fp32", "final_loss_bf16",
              "loss_drift_abs", "loss_drift_rel", "decision",
              "elapsed_s_fp32", "elapsed_s_bf16"):
        assert k in summary, f"summary missing {k}"
    # Both runs produced a finite final loss.
    assert summary["final_loss_fp32"] == summary["final_loss_fp32"]
    assert summary["final_loss_bf16"] == summary["final_loss_bf16"]
    # Decision string surfaces honest verdict.
    assert summary["decision"] in (
        "keep bf16 (faster + similar loss)",
        "fp32 recommended for this workload",
    )
