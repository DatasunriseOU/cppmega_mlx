"""V7-I02 / item 49: measured MemoryBar parity bench gate.

Runs the V7-I02 memory-parity bench in a fresh subprocess (clean
Metal allocator) at multiple H scales and asserts the ratio between
the analytical estimate and the actual Metal peak is strictly less
than 2.0x at every H — proving the original 500x analytical/measured
gap has collapsed below the V7-I02 target.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_v7_i02_memory_parity_under_2x_across_realistic_H(tmp_path):
    """Measured parity bound: max ratio < 2.0x for H in {128, 512}.

    The bench runs in a fresh Python process so the Metal allocator
    starts cold; in-process pytest re-uses the allocator across
    tests, which inflates the apparent peak. Run with two H values
    so the bound is exercised at both small and mid model sizes.
    """
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_memory_parity",
         "--H", "128", "--H", "512", "--num-steps", "2",
         "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=180)
    assert rc.returncode == 0, rc.stderr
    stable = tmp_path / "memory_parity_latest.json"
    assert stable.exists()
    data = json.loads(stable.read_text())
    rows = data["rows"]
    assert len(rows) >= 2
    for row in rows:
        ratio = row["ratio_max_min"]
        assert ratio is not None, (
            f"no measured peak at H={row['H']} — Metal backend "
            f"missing on this host?")
        assert ratio < 2.0, (
            f"V7-I02 measured parity > 2x at H={row['H']}: "
            f"estimate={row['estimate_bytes']}B "
            f"actual={row['actual_bytes']}B ratio={ratio}×")
    max_ratio = data["max_ratio"]
    assert max_ratio < 2.0, f"max ratio {max_ratio}× exceeded 2x bound"


def test_v7_i02_memory_parity_writes_stable_receipt(tmp_path):
    """Sanity: the bench writes both a timestamped file and the
    stable `memory_parity_latest.json` so CI can diff successive
    runs against the same path."""
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_memory_parity",
         "--H", "128", "--num-steps", "2",
         "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=120)
    assert rc.returncode == 0, rc.stderr
    timestamped = list(tmp_path.glob("memory_parity_2*.json"))
    assert timestamped, "missing timestamped receipt"
    stable = tmp_path / "memory_parity_latest.json"
    assert stable.exists(), "missing stable latest receipt"
    data = json.loads(stable.read_text())
    for k in ("date_utc", "rows", "max_ratio", "S"):
        assert k in data, f"missing top-level key {k}"
    row0 = data["rows"][0]
    for k in ("H", "S", "num_steps", "estimate_bytes",
              "actual_bytes", "ratio_max_min"):
        assert k in row0, f"missing row key {k}"
