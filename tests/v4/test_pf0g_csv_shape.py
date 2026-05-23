"""V7-B15 (bd cppmega-mlx-pf0g): --simulated CSV columns + warm_ms>0."""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys


REQUIRED_COLUMNS = (
    "buf_mb", "bytes", "world_size", "rank", "backend", "real",
    "all_reduce_ms_per_iter", "all_gather_ms_per_iter",
    "reduce_scatter_ms_per_iter", "warm_ms",
)


def test_simulated_flag_emits_canonical_csv(tmp_path):
    """Run the bench with --simulated; verify the artifact lives at
    reports/allreduce_simulated_*.csv and has the required columns +
    warm_ms > 0."""
    out_dir = tmp_path / "reports"
    cmd = [
        sys.executable, "-m", "scripts.bench_allreduce_distributed",
        "--simulated",
        "--buf-mb", "0.25", "1",
        "--n-iter", "3",
        "--out-dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"bench failed: stderr={proc.stderr}")

    csvs = glob.glob(str(out_dir / "allreduce_simulated_*.csv"))
    assert csvs, f"no allreduce_simulated_*.csv in {out_dir}"

    with open(csvs[0]) as f:
        reader = csv.DictReader(f)
        cols = tuple(reader.fieldnames or ())
        rows = list(reader)

    for required in REQUIRED_COLUMNS:
        assert required in cols, (
            f"required column {required!r} missing: have {cols}")

    assert len(rows) == 2, f"expected 2 rows (buf-mb), got {len(rows)}"
    for r in rows:
        warm_ms = float(r["warm_ms"])
        assert warm_ms > 0, f"warm_ms must be > 0; got {warm_ms}"
        # Simulated runs report real='False' (or '0' depending on row
        # rendering); accept either.
        assert r["real"].lower() in {"false", "0.0", "0"}
        assert int(float(r["world_size"])) == 1
