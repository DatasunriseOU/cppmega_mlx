"""V7-I01: whole_model compile bench smoke."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_v7_i01_bench_whole_model_compile_smoke(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_whole_model_compile",
         "--preset", "llama3_8b", "--hidden", "128",
         "--num-steps", "3", "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=120)
    assert rc.returncode == 0, rc.stderr
    files = list(tmp_path.glob("whole_model_compile_bench_*.json"))
    assert files, "no bench json written"
    data = json.loads(files[0].read_text())
    assert data["status"] == "ok"
    for k in ("preset", "hidden", "compile_mode", "num_steps",
              "first_step_ms", "warm_step_ms_mean", "warm_step_ms_min",
              "peak_memory_mb"):
        assert k in data, f"missing {k}"
    assert data["compile_mode"] == "whole_model"
