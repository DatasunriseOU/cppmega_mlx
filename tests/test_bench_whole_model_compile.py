"""V7-I01 / item 52: whole_model compile bench smoke + real-compile gate.

Asserts that the bench script actually engages mx.compile (not just
echoes the metadata) and that the first step is measurably slower
than the warm steady-state (compile produced a real speed-up, not a
no-op pass-through).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def _run_bench(tmp_path: pathlib.Path, compile_mode: str,
               num_steps: int = 6) -> dict:
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_whole_model_compile",
         "--preset", "llama3_8b", "--hidden", "128",
         "--num-steps", str(num_steps),
         "--compile-mode", compile_mode,
         "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=180)
    assert rc.returncode == 0, rc.stderr
    files = sorted(tmp_path.glob("whole_model_compile_bench_*.json"))
    assert files, "no bench json written"
    return json.loads(files[-1].read_text())


def test_v7_i01_bench_whole_model_compile_smoke(tmp_path):
    data = _run_bench(tmp_path, compile_mode="whole_model", num_steps=3)
    assert data["status"] == "ok"
    for k in ("preset", "hidden", "compile_mode", "num_steps",
              "first_step_ms", "warm_step_ms_mean", "warm_step_ms_min",
              "peak_memory_mb", "compile_engaged", "compile_status",
              "first_vs_warm_ratio"):
        assert k in data, f"missing {k}"
    assert data["compile_mode"] == "whole_model"


def test_v7_i01_whole_model_actually_engages_mx_compile(tmp_path):
    """Item 52: prove mx.compile actually fired in the whole-model path.

    The runner must set compile_engaged=True and report a first-step
    wall-clock that is measurably slower than the warm-step mean —
    the compile + lazy graph build only happens once. We use 6 steps
    so the warm mean is averaged over 5 steady-state steps.
    """
    data = _run_bench(tmp_path, compile_mode="whole_model", num_steps=6)
    assert data["compile_engaged"] is True, (
        f"compile_engaged must be True; got {data}")
    assert data["compile_status"] == "engaged"
    assert data["compile_error"] is None
    # Compile produces a measurable first-vs-warm gap. We bound this
    # loosely at >=2.0x — on Apple-silicon Metal the real ratio is
    # typically 10–200x depending on device + model size, but bench
    # noise on tiny H=128 can drop it close to the floor.
    ratio = float(data["first_vs_warm_ratio"])
    assert ratio >= 2.0, (
        f"compile must produce >=2x first-vs-warm ratio; got "
        f"{ratio}× from per_step_ms={data['per_step_ms']}")


def test_v7_i01_off_path_reports_no_compile(tmp_path):
    """Sanity inverse: compile-off must report compile_engaged=False."""
    data = _run_bench(tmp_path, compile_mode="off", num_steps=3)
    assert data["compile_engaged"] is False
    assert data["compile_status"] == "off"
