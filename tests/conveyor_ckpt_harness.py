#!/usr/bin/env python3
"""Isolated subprocess harness for the conveyor SIGTERM-checkpoint / resume test.

This is NOT a pytest module (no ``test_`` prefix, so pytest does not collect it).
It is launched as a real subprocess by
``tests/test_conveyor_signal_checkpoint_resume.py`` so that SIGINT/SIGTERM is
delivered to a real process and the REAL conveyor orchestration runs:

    streaming_conveyor.main()  ->  process_one_repo  ->  run_commits_half
    ->  ensure_commit_records  ->  ConcurrentManifest  ->  _on_signal handler
    ->  repo_fully_done / temp-retention finally.

The conveyor's output paths are HARD-CODED off ``streaming_reindex.MLX_ROOT``
(the live corpus repo), with no CLI/env override. To run in FULL ISOLATION
without touching the live conveyor (pid 48133) we override those module path
constants to point under ``$CKPT_ROOT`` BEFORE calling ``main()``. Only the
EXPENSIVE LEAF STAGES are replaced with deterministic fakes -- the git-history
extract subprocess, the clang code stage, the per-range commit stage and the
.git-preserving tar source. The resume LOGIC under test (sentinel hit, manifest
skip, signal-drain, temp retention) is exercised verbatim. The fakes record
every call to a durable side log OUTSIDE the work/cache dirs so the test can
prove, across a kill+restart, that the extract ran exactly once per repo and no
range was processed twice.

Config via environment (all required except where noted):
  CKPT_ROOT          isolated base dir (everything lives under here)
  CKPT_REPOS         comma list of fake repo names (order = stream order)
  CKPT_NRECORDS      int: fake commit records emitted per repo (== #ranges at
                     --range-size 1)
  CKPT_RANGE_SLEEP   float seconds each fake range "takes" (lets the test
                     SIGTERM mid-repo)
  CKPT_EXTRACT_EVENTS  path: append-only jsonl, one line per REAL extract call
  CKPT_RANGE_EVENTS    path: append-only jsonl, one line per REAL range call

All conveyor CLI flags are passed through ``sys.argv[1:]`` by the test.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

HARNESS_FILE = Path(__file__).resolve()
MLX_ROOT = HARNESS_FILE.parents[1]
SCRIPTS = MLX_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import streaming_reindex as sr  # noqa: E402
import streaming_reindex_commits as src  # noqa: E402
import streaming_conveyor as conv  # noqa: E402


def _env(name: str) -> str:
    val = os.environ.get(name)
    if val is None or val == "":
        raise SystemExit(f"harness: required env {name} is unset")
    return val


CKPT_ROOT = Path(_env("CKPT_ROOT"))
REPOS = [r for r in _env("CKPT_REPOS").split(",") if r]
NRECORDS = int(_env("CKPT_NRECORDS"))
RANGE_SLEEP = float(_env("CKPT_RANGE_SLEEP"))
EXTRACT_EVENTS = Path(_env("CKPT_EXTRACT_EVENTS"))
RANGE_EVENTS = Path(_env("CKPT_RANGE_EVENTS"))

_APPEND_LOCK = threading.Lock()


def _append_event(path: Path, payload: dict) -> None:
    """Atomic-enough append of one small jsonl line (thread + process safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


# --------------------------------------------------------------------------- #
# Isolate every hard-coded output path onto $CKPT_ROOT so the live conveyor's
# outputs/conveyor, outputs/reindexed*, dedup_seen.sqlite, etc. are NEVER read
# or written by this run.
# --------------------------------------------------------------------------- #
OUT = CKPT_ROOT / "outputs"
conv.CONVEYOR_ROOT = OUT / "conveyor"
conv.CONVEYOR_MANIFEST = conv.CONVEYOR_ROOT / "_done.json"
conv.EXTRACT_CACHE_ROOT = conv.CONVEYOR_ROOT / "extract_cache"
conv.DEFAULT_RUN_LOCK_DIR = conv.CONVEYOR_ROOT / "locks"
conv.DEFAULT_PROGRESS_JSONL = conv.CONVEYOR_ROOT / "progress.jsonl"
conv.DEFAULT_WORK_PARENT = conv.CONVEYOR_ROOT / "tmp"
conv.DEFAULT_DEDUP_DB = OUT / "dedup_seen.sqlite"
conv.DEFAULT_PR_STORE = OUT / "pr_ingest" / "prs.sqlite"
conv.DEFAULT_REPO_LIST = OUT / "pr_ingest" / "repo_list.json"
sr.OUTPUT_ROOT = OUT / "reindexed"
src.COMMIT_OUTPUT_ROOT = OUT / "reindexed_commits"
MARKERS = src.COMMIT_OUTPUT_ROOT / "markers"

# The up-front required-path existence check (RULE #1 fail-loud) must pass
# without invoking any real binary; point every required path at this harness
# file (which exists). The stages that would USE them are faked below.
for _mod, _name in (
    (conv, "VENV_PYTHON"),
    (conv, "EXTRACT_GIT"),
    (sr, "TOKENIZER_PATH"),
    (sr, "MATERIALIZER"),
    (sr, "PACKER"),
    (sr, "INDEX_PROJECT"),
    (sr, "PROCESS_COMMITS"),
):
    setattr(_mod, _name, HARNESS_FILE)


# --------------------------------------------------------------------------- #
# Fake leaf stages -- match the exact signatures the real orchestration calls.
# --------------------------------------------------------------------------- #
def fake_stream_repo_subtrees_with_git(work_root, should_process, *, on_no_git=None):
    """Yield (repo, repo_dir==work_root/<repo>/_src) for each configured repo."""
    for repo in REPOS:
        if not should_process(repo):
            continue
        repo_dir = Path(work_root) / repo / "_src"
        (repo_dir / ".git").mkdir(parents=True, exist_ok=True)
        yield repo, repo_dir


def fake_get_commit_list(repo_dir):
    # Non-empty so ensure_commit_records does not raise the empty-git-log error.
    return list(range(NRECORDS))


def fake_stage_extract_commits(repo, repo_dir, cache_dir):
    """The EXPENSIVE extract. Writes <repo>_commits.jsonl with NRECORDS lines and
    records ONE durable event so the test can prove it ran exactly once / repo."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl = cache_dir / f"{repo}_commits.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for i in range(NRECORDS):
            fh.write(json.dumps({"repo": repo, "i": i}) + "\n")
    st = jsonl.stat()
    _append_event(EXTRACT_EVENTS, {
        "repo": repo,
        "jsonl": str(jsonl),
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "pid": os.getpid(),
        "ts": time.time(),
    })
    return jsonl


def _lengths_info(lengths, valid):
    return {
        str(L): {"rows": 1, "valid_tokens": int(valid), "pad_tokens": 0,
                 "capacity_tokens": int(valid)}
        for L in lengths
    }


def fake_run_code_half(repo, repo_dir, lengths_code, work_root, dedup_db,
                       dedup_near, global_symbol_index=None, memory_limit_gb=10.0,
                       parse_workers=2, index_timeout_s=None,
                       index_stall_timeout_s=None, recompressor=None):
    return {"source": f"{repo}::code", "lengths": _lengths_info(lengths_code, 10)}


def fake_process_range(repo, repo_dir, records_jsonl, start, end, lengths_sorted,
                       repo_work, dedup_db, dedup_near, pr_store, repo_list,
                       memory_limit_gb=10.0, analysis_cache_entries=128):
    """Fake per-range commit stage. Sleeps (so the test can SIGTERM mid-repo),
    records ONE durable event per ACTUAL execution, and writes a persistent
    marker 'parquet' so final completeness can be checked."""
    time.sleep(RANGE_SLEEP)
    _append_event(RANGE_EVENTS, {
        "repo": repo, "start": int(start), "end": int(end),
        "pid": os.getpid(), "ts": time.time(),
    })
    MARKERS.mkdir(parents=True, exist_ok=True)
    (MARKERS / f"{repo}_r{int(start)}.parquet").write_text("ok", encoding="utf-8")
    return {
        "source": f"{repo}::r{start}",
        "repo": repo,
        "range": [int(start), int(end)],
        "lengths": _lengths_info(lengths_sorted, 5),
    }


conv.stream_repo_subtrees_with_git = fake_stream_repo_subtrees_with_git
conv.get_commit_list = fake_get_commit_list
conv.stage_extract_commits = fake_stage_extract_commits
conv.run_code_half = fake_run_code_half
conv.process_range = fake_process_range


if __name__ == "__main__":
    raise SystemExit(conv.main(sys.argv[1:]))
