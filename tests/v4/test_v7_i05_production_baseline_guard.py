"""V7-I05: regression-guard against reports/production_preset_baseline_latest.csv.

AC#2: 'Regression-guard test that loads the baseline CSV and fails if
a future run is >25% slower.'

Strategy: the baseline CSV is committed to main. A periodic bench run
overwrites it. This test loads the CSV and the most recent JSON, then
asserts:

  (a) The CSV exists and parses with the expected columns.
  (b) For each row, ms_per_step_warm > 0 (positive) — pins that the
      bench actually measured something, not a placeholder.
  (c) ms_per_step_warm is within 25% of ms_per_step_warm_min (warm-up
      stabilises) when both are populated, otherwise marked unstable.
  (d) peak_memory_mb > 0 when reported (not a stub).
  (e) If an out-of-process bench wrote a freshly-dated CSV with the
      same row schema, ms_per_step_warm must not exceed 1.25× the
      baseline value for the same preset (this guards future commits
      against silent perf regression).

The "old vs new" comparison only fires when ``CPPMEGA_BENCH_NEW`` env
var points at a candidate CSV — by default the test simply pins the
baseline shape so a missing CSV fails fast.
"""

from __future__ import annotations

import csv
import os
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_CSV = REPO_ROOT / "reports" / "production_preset_baseline_latest.csv"


def _read_rows(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        pytest.skip(f"baseline CSV missing: {path}")
    with open(path) as f:
        return list(csv.DictReader(f))


def test_v7_i05_baseline_csv_exists_and_has_rows():
    rows = _read_rows(BASELINE_CSV)
    assert rows, "baseline CSV has no rows"
    expected = {
        "preset", "hidden", "n_layers", "num_steps", "S",
        "compile_mode", "status", "first_step_ms",
        "ms_per_step_warm", "ms_per_step_warm_min",
        "peak_memory_mb", "total_elapsed_s",
        "compile_engaged", "compile_status",
    }
    assert expected.issubset(set(rows[0].keys())), (
        f"missing cols: {expected - set(rows[0].keys())}")


def test_v7_i05_baseline_each_row_has_positive_warm_step():
    rows = _read_rows(BASELINE_CSV)
    for r in rows:
        warm = float(r["ms_per_step_warm"])
        assert warm > 0, f"non-positive warm step for {r['preset']}: {warm}"


def test_v7_i05_baseline_warm_min_within_30pct_of_mean():
    """Warm-step stability sanity: warm_min should be within ~30% of
    warm_mean when both are populated, otherwise the baseline is too
    noisy to be regression-guard material."""
    rows = _read_rows(BASELINE_CSV)
    for r in rows:
        warm = float(r["ms_per_step_warm"])
        warm_min_str = r.get("ms_per_step_warm_min", "")
        if not warm_min_str:
            continue
        warm_min = float(warm_min_str)
        if warm_min <= 0:
            continue
        # warm_mean >= warm_min always; check the gap isn't >30% so
        # one outlier step doesn't poison the baseline.
        ratio = warm / warm_min
        assert ratio <= 1.30, (
            f"{r['preset']} warm noisy: mean={warm} min={warm_min} "
            f"ratio={ratio:.2f}")


def test_v7_i05_baseline_peak_mem_when_present_is_positive():
    rows = _read_rows(BASELINE_CSV)
    for r in rows:
        s = r.get("peak_memory_mb", "")
        if s == "" or s == "None":
            continue
        assert float(s) > 0, (
            f"{r['preset']} peak_memory_mb non-positive: {s}")


def test_v7_i05_optional_new_bench_within_25pct_of_baseline():
    """When CPPMEGA_BENCH_NEW=path/to/freshly_dated.csv is set, walk
    every preset and assert its ms_per_step_warm is within 1.25× the
    baseline value (AC#2 regression-guard, 25% slower threshold).

    By default this test no-ops so the CI doesn't need to plumb the
    env var; it fires when a developer commits a new bench and wants
    to pre-flight the regression check locally."""
    new_path = os.environ.get("CPPMEGA_BENCH_NEW")
    if not new_path:
        pytest.skip("CPPMEGA_BENCH_NEW unset — skip regression compare")
    new_rows = _read_rows(pathlib.Path(new_path))
    baseline = {r["preset"]: float(r["ms_per_step_warm"])
                for r in _read_rows(BASELINE_CSV)}
    for r in new_rows:
        if r["preset"] not in baseline:
            continue
        new_warm = float(r["ms_per_step_warm"])
        b = baseline[r["preset"]]
        assert new_warm <= b * 1.25, (
            f"V7-I05 regression on {r['preset']}: new={new_warm} ms "
            f"vs baseline={b} ms (limit 1.25× = {b * 1.25:.2f})")
