"""End-to-end, FULLY ISOLATED test for the conveyor's graceful SIGTERM
checkpoint + zero-rework resume (extract-sentinel HIT + per-range manifest skip
+ temp retention).

It launches ``tests/conveyor_ckpt_harness.py`` as a REAL subprocess (so the
SIGINT/SIGTERM handler and process-level resume are exercised for real), drives
the REAL ``streaming_conveyor`` orchestration with only the heavy leaf stages
faked, and asserts across a kill+restart:

  (a) the first run, on SIGTERM, exits 130 after a GRACEFUL checkpoint -- the
      manifest is flushed, the extract sentinel + jsonl + work temp are RETAINED,
      and the checkpoint log lines are printed;
  (b) on restart with the SAME args, ``extract_git_history`` is NOT re-run for the
      already-extracted repo (the EXTRACT-CKPT HIT short-circuit fires and the
      durable extract-event log still shows exactly one extract for that repo,
      with the jsonl mtime unchanged);
  (c) already-done ranges are NOT re-processed (manifest SKIP; the durable
      range-event log has no duplicate (repo,start) across both runs);
  (d) the final output is complete + correct -- every code-half and every range
      for every repo is marked done, with exactly one persistent marker per
      range and no failures.

Everything lives under ``$CKPT_TEST_ROOT`` (default: a pytest tmp dir). The live
conveyor's outputs/conveyor, outputs/reindexed*, dedup_seen.sqlite, pr_ingest and
its work temp are never referenced.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

MLX_ROOT = Path(__file__).resolve().parents[1]
HARNESS = MLX_ROOT / "tests" / "conveyor_ckpt_harness.py"
VENV_PY = MLX_ROOT / ".venv" / "bin" / "python"

REPOS = ["repoA", "repoB"]
NRECORDS = 10            # -> 10 ranges per repo at --range-size 1
RANGE_SLEEP = 0.8        # s per fake range: wide window to SIGTERM mid-repoA
FIRST_REPO = "repoA"


def _python() -> str:
    return str(VENV_PY) if VENV_PY.exists() else sys.executable


def _read_manifest(path: Path) -> dict:
    """Read the atomically-replaced manifest; tolerate the brief replace window."""
    for _ in range(50):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
    return {"done": {}, "failed": {}}


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]


def _ckpt_root() -> Path:
    env = os.environ.get("CKPT_TEST_ROOT")
    if env:
        return Path(env)
    return None  # caller substitutes tmp_path


def _build(ckpt_root: Path):
    out = ckpt_root / "outputs"
    conveyor = out / "conveyor"
    paths = {
        "ckpt_root": ckpt_root,
        "manifest": conveyor / "_done.json",
        "extract_cache": conveyor / "extract_cache",
        "work_dir": ckpt_root / "work",
        "work_parent": conveyor / "tmp",
        "run_lock_dir": conveyor / "locks",
        "progress": conveyor / "progress.jsonl",
        "markers": out / "reindexed_commits" / "markers",
        "extract_events": ckpt_root / "events" / "extract_events.jsonl",
        "range_events": ckpt_root / "events" / "range_events.jsonl",
    }
    env = dict(os.environ)
    env.update({
        "CKPT_ROOT": str(ckpt_root),
        "CKPT_REPOS": ",".join(REPOS),
        "CKPT_NRECORDS": str(NRECORDS),
        "CKPT_RANGE_SLEEP": str(RANGE_SLEEP),
        "CKPT_EXTRACT_EVENTS": str(paths["extract_events"]),
        "CKPT_RANGE_EVENTS": str(paths["range_events"]),
    })
    argv = [
        "--streams", "both",
        "--workers", "2",
        "--repo-workers", "1",
        "--range-size", "1",
        "--max-repos", str(len(REPOS)),
        "--work-dir", str(paths["work_dir"]),
        "--work-parent-dir", str(paths["work_parent"]),
        "--run-lock-dir", str(paths["run_lock_dir"]),
        "--progress-jsonl", str(paths["progress"]),
        "--dedup-db", "",
        "--pr-store", "",
        "--repo-list", "",
        "--dedup-checkpoint-tokens", "0",
        "--memory-limit-gb", "99",
        "--memory-budget-gb", "0",
        "--retain-partial-work",
    ]
    return paths, env, argv


def _launch(env, argv, logpath: Path) -> subprocess.Popen:
    logpath.parent.mkdir(parents=True, exist_ok=True)
    log = logpath.open("wb")
    proc = subprocess.Popen(
        [_python(), str(HARNESS), *argv],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    proc._logfh = log  # type: ignore[attr-defined]
    return proc


def _range_keys_done(manifest: dict, repo: str) -> set[str]:
    return {k for k in manifest.get("done", {}) if k.startswith(f"{repo}::r")}


def test_sigterm_checkpoint_then_zero_rework_resume(tmp_path, capsys):
    ckpt_root = _ckpt_root() or (tmp_path / "ckpt")
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    ckpt_root.mkdir(parents=True)
    paths, env, argv = _build(ckpt_root)
    proof: dict[str, object] = {}

    # ---------------- RUN 1: start, SIGTERM mid-repoA ----------------------- #
    p1 = _launch(env, argv, ckpt_root / "logs" / "run1.log")
    sent = False
    deadline = time.time() + 60
    try:
        while time.time() < deadline:
            if p1.poll() is not None:
                break  # exited before we could signal -> handled by assert below
            man = _read_manifest(paths["manifest"])
            done_r = _range_keys_done(man, FIRST_REPO)
            sentinel = paths["extract_cache"] / FIRST_REPO / f"{FIRST_REPO}_commits.jsonl.done"
            # SIGTERM exactly when the extract is done AND >=1 but NOT all ranges
            # are committed -> guarantees a mid-repo interruption.
            if sentinel.exists() and 1 <= len(done_r) < NRECORDS:
                os.kill(p1.pid, signal.SIGTERM)
                sent = True
                break
            time.sleep(0.05)
        assert sent, "never reached the partial-progress window to SIGTERM"
        rc1 = p1.wait(timeout=60)
    finally:
        if p1.poll() is None:
            p1.kill()
        p1._logfh.close()  # type: ignore[attr-defined]
    run1_log = (ckpt_root / "logs" / "run1.log").read_text(encoding="utf-8", errors="replace")

    # ---- (a) graceful checkpoint asserts ----
    assert rc1 == 130, f"run1 should exit 130 on signal, got {rc1}\n{run1_log[-2000:]}"
    assert "CHECKPOINTING" in run1_log, "no CHECKPOINTING log line on signal"
    assert "CHECKPOINTED on signal" in run1_log, "no graceful-exit checkpoint line"

    man1 = _read_manifest(paths["manifest"])
    done1 = _range_keys_done(man1, FIRST_REPO)
    assert f"{FIRST_REPO}::code" in man1["done"], "code half not flushed to manifest"
    assert 1 <= len(done1) < NRECORDS, f"expected partial repoA ranges, got {sorted(done1)}"
    assert not man1.get("failed"), f"unexpected failures: {man1.get('failed')}"
    # repoB must NOT have been staged after the stop (sequential driver).
    assert not _range_keys_done(man1, "repoB"), "repoB staged after SIGTERM stop"

    cache_jsonl = paths["extract_cache"] / FIRST_REPO / f"{FIRST_REPO}_commits.jsonl"
    sentinel = Path(str(cache_jsonl) + ".done")
    assert cache_jsonl.exists(), "extract jsonl NOT retained after interrupt"
    assert sentinel.exists(), "extract sentinel NOT retained after interrupt"
    assert (paths["work_dir"] / FIRST_REPO / "_src").exists(), "work _src NOT retained"

    ex1 = _read_events(paths["extract_events"])
    a_extracts_1 = [e for e in ex1 if e["repo"] == FIRST_REPO]
    assert len(a_extracts_1) == 1, f"repoA extracted {len(a_extracts_1)}x in run1 (expected 1)"
    a_mtime_run1 = a_extracts_1[0]["mtime_ns"]

    graceful_checkpoint_ok = True
    proof["run1_exit"] = rc1
    proof["run1_repoA_code_done"] = f"{FIRST_REPO}::code" in man1["done"]
    proof["run1_repoA_ranges_done"] = sorted(int(k.split("::r")[1]) for k in done1)
    proof["run1_extract_calls_repoA"] = len(a_extracts_1)
    proof["run1_sentinel_retained"] = sentinel.exists()
    proof["run1_jsonl_mtime_ns"] = a_mtime_run1
    proof["run1_checkpoint_loglines"] = [
        ln for ln in run1_log.splitlines()
        if "CHECKPOINT" in ln.upper()
    ][:4]

    # ---------------- RUN 2: restart with the SAME args -------------------- #
    p2 = _launch(env, argv, ckpt_root / "logs" / "run2.log")
    try:
        rc2 = p2.wait(timeout=120)
    finally:
        if p2.poll() is None:
            p2.kill()
        p2._logfh.close()  # type: ignore[attr-defined]
    run2_log = (ckpt_root / "logs" / "run2.log").read_text(encoding="utf-8", errors="replace")
    assert rc2 == 0, f"run2 should exit 0 (clean), got {rc2}\n{run2_log[-2000:]}"

    # ---- (b) extract NOT re-run for the already-extracted repo ----
    assert f"EXTRACT-CKPT HIT {FIRST_REPO}" in run2_log, \
        "resume did not hit the extract sentinel short-circuit for repoA"
    assert f"EXTRACT-CKPT FRESH {FIRST_REPO}" not in run2_log, \
        "repoA was re-extracted on resume (FRESH)"
    ex2 = _read_events(paths["extract_events"])
    a_extracts_2 = [e for e in ex2 if e["repo"] == FIRST_REPO]
    assert len(a_extracts_2) == 1, \
        f"repoA extract ran again on resume: {len(a_extracts_2)} total calls"
    assert a_extracts_2[0]["mtime_ns"] == a_mtime_run1, \
        "repoA jsonl was rewritten on resume (mtime changed)"
    extract_not_rerun_on_resume = True

    # ---- (c) already-done ranges NOT re-processed ----
    # The run-1 done ranges must be SKIP-logged in run2 and must NOT reappear in
    # the durable range-event log; no (repo,start) processed twice across runs.
    for k in sorted(done1):
        assert f"SKIP (done) {k}" in run2_log, f"resume did not skip done range {k}"
    rng_events = _read_events(paths["range_events"])
    seen_pairs = [(e["repo"], e["start"]) for e in rng_events]
    assert len(seen_pairs) == len(set(seen_pairs)), \
        f"a range was processed twice across runs: {seen_pairs}"
    run1_done_starts = {int(k.split("::r")[1]) for k in done1}
    run2_range_events = [
        e for e in rng_events if e["pid"] == p2.pid and e["repo"] == FIRST_REPO
    ]
    redone = {e["start"] for e in run2_range_events} & run1_done_starts
    assert not redone, f"run2 re-processed already-done repoA ranges: {sorted(redone)}"
    ranges_not_redone = True

    # ---- (d) final output complete + correct ----
    man2 = _read_manifest(paths["manifest"])
    expected = set()
    for repo in REPOS:
        expected.add(f"{repo}::code")
        for s in range(NRECORDS):
            expected.add(f"{repo}::r{s}")
    missing = expected - set(man2["done"])
    assert not missing, f"final manifest missing units: {sorted(missing)}"
    assert not man2.get("failed"), f"final manifest has failures: {man2.get('failed')}"
    marker_files = sorted(p.name for p in paths["markers"].glob("*.parquet"))
    expected_markers = sorted(
        f"{repo}_r{s}.parquet" for repo in REPOS for s in range(NRECORDS)
    )
    assert marker_files == expected_markers, \
        f"range markers mismatch\n got={marker_files}\n want={expected_markers}"
    # exactly one marker per range -> no duplicate range outputs
    assert len(marker_files) == len(set(marker_files)) == len(REPOS) * NRECORDS
    # fully-done repos: temp + extract cache reclaimed.
    assert not (paths["work_dir"] / FIRST_REPO).exists(), \
        "fully-done repo work dir not reclaimed"
    assert not (paths["extract_cache"] / FIRST_REPO).exists(), \
        "fully-done repo extract cache not reclaimed"
    final_correct = True

    total_extracts = {r: len([e for e in ex2 if e["repo"] == r]) for r in REPOS}
    proof.update({
        "run2_exit": rc2,
        "run2_extract_HIT_repoA": f"EXTRACT-CKPT HIT {FIRST_REPO}" in run2_log,
        "run2_extract_FRESH_repoA_absent": f"EXTRACT-CKPT FRESH {FIRST_REPO}" not in run2_log,
        "extract_calls_total_per_repo": total_extracts,
        "repoA_jsonl_mtime_ns_unchanged": a_extracts_2[0]["mtime_ns"] == a_mtime_run1,
        "run1_done_range_skip_lines": [
            ln for ln in run2_log.splitlines() if "SKIP (done) repoA::r" in ln
        ],
        "total_range_executions": len(seen_pairs),
        "duplicate_range_executions": len(seen_pairs) - len(set(seen_pairs)),
        "final_done_units": len(man2["done"]),
        "final_failed_units": len(man2.get("failed", {})),
        "marker_count": len(marker_files),
        "graceful_checkpoint_ok": graceful_checkpoint_ok,
        "extract_not_rerun_on_resume": extract_not_rerun_on_resume,
        "ranges_not_redone": ranges_not_redone,
        "final_correct": final_correct,
    })
    print("CKPT_PROOF " + json.dumps(proof, sort_keys=True, default=str))

    assert graceful_checkpoint_ok and extract_not_rerun_on_resume \
        and ranges_not_redone and final_correct


if __name__ == "__main__":  # allow standalone run for manual proof capture
    sys.exit(pytest.main([__file__, "-s", "-v"]))
