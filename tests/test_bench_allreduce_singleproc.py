"""V7-B05: single-process all-reduce baseline bench smoke."""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_v7_b05_allreduce_singleproc_smoke(tmp_path):
    rc = subprocess.run(
        [sys.executable, "-m", "scripts.bench_allreduce_singleproc",
         "--ranks", "2", "4",
         "--buf-mb", "1",
         "--n-iter", "3",
         "--out-dir", str(tmp_path)],
        cwd=REPO, check=False, capture_output=True, text=True,
        timeout=120)
    assert rc.returncode == 0, rc.stderr
    csvs = list(tmp_path.glob("bench_allreduce_singleproc_*.csv"))
    assert csvs, "no bench csv written"
    with csvs[0].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # 2 ranks × 1 buf size
    for r in rows:
        for k in ("fake_ranks", "buf_mb", "iters",
                  "ms_per_iter", "mb_per_s"):
            assert k in r
        assert float(r["ms_per_iter"]) > 0
        assert float(r["mb_per_s"]) > 0
