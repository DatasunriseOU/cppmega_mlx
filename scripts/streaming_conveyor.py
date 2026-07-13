#!/usr/bin/env python3
"""Unified per-repo CONVEYOR: ONE .git-preserving tarball pass, BOTH streams per repo.

This is the single driver that packages the manually-proven orchestration
(apex / abseil-cpp / amrex were run stage-by-stage by hand) into ONE resumable
conveyor. It does NOT reimplement any stage logic -- it imports and orchestrates
the EXACT existing stages from ``streaming_reindex`` (code) and
``streaming_reindex_commits`` (commits + PR-discussion injection).

Per repo, in ONE pass over the ``.git``-preserving tar stream:

  extract cpp_all/<repo>/ (incl .git) -> work/<repo>/_src   (ONE extraction)
    -> CODE pipeline (streaming_reindex.process_one_repo):
         index_project.py --enriched [C/C++ + build files]
           -> clang_enriched_to_parquet (tokenize @65536, materialize)
           -> route_by_fit (--target-lengths-code, default 1024,2048,4096)
           -> pack_enriched_rows per bucket
           -> outputs/reindexed/<L>/<repo>.parquet   (recompressed zstd-max)
       The SHARED --dedup-db is threaded in (function-level tokenized-hash exact
       + MinHash-LSH near).
    -> COMMITS pipeline (streaming_reindex_commits stages):
         extract_git_history.py once -> <repo>_commits.jsonl (records)
           -> DELETE .git immediately (records captured; bounded disk)
           -> split records into ranges of --range-size, fan to the SAME pool:
                process_commits.py --format both --pr-store --repo-list --dedup-db
                  (PR @discussion injected via PRDiscussionLookup, commit 36c717a)
                -> materialize -> route_by_fit (--target-lengths-commits,
                   default 1024,2048,4096,8192,16384) -> pack
                -> outputs/reindexed_commits/<L>/<repo>_r<start>.parquet (zstd-max)
       The SAME SHARED --dedup-db is threaded in (commit-DOC tokenized-hash).
    -> DELETE the whole repo work dir (bounded disk: ~1 repo on disk at a time).

Resumable: ONE manifest at outputs/conveyor/_done.json. The CODE half is keyed
``<repo>::code``; each COMMIT range is keyed ``<repo>::r<start>`` (exactly the
commit driver's range key). Resume skips completed code-halves and completed
ranges exactly; a repo whose code is done but ranges remain is resumed mid-way.

Fail-loud per repo (RULE #1): every stage runs ONE clear subprocess path and
RAISES RepoFailure on any non-zero exit / missing / empty output. A failure is
recorded in the manifest (``failed``) and the conveyor moves to the next unit --
no degraded / best-effort / silent-fallback path anywhere.

SHARED dedup: ONE --dedup-db (default outputs/dedup_seen.sqlite) is passed to
BOTH the code stage AND every commit range stage, so the user's clever dedup
(function-level for code, doc-level for commits; exact + near) is global across
repos AND across the two streams.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Reuse the proven machinery. streaming_reindex_commits already re-exports the
# shared primitives from streaming_reindex (MLX_ROOT, Manifest, RepoFailure, ...)
# and adds the .git-preserving stream + range pipeline. Import BOTH, do not
# duplicate a single line of stage logic.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import streaming_reindex as sr
import streaming_reindex_commits as src

from streaming_reindex_commits import (  # noqa: F401
    MLX_ROOT,
    VENV_PYTHON,
    RepoFailure,
    Manifest,
    EXTRACT_GIT,
    DEFAULT_RANGE_SIZE,
    stream_repo_subtrees_with_git,
    get_commit_list,
    stage_extract_commits,
    process_range,
    range_key,
)

# Unified conveyor manifest lives in its OWN root so it never collides with the
# per-stream manifests (the code/commit outputs themselves still land in the
# existing outputs/reindexed and outputs/reindexed_commits trees, so already-run
# parquet is reused as-is).
CONVEYOR_ROOT = MLX_ROOT / "outputs" / "conveyor"
CONVEYOR_MANIFEST = CONVEYOR_ROOT / "_done.json"

DEFAULT_DEDUP_DB = MLX_ROOT / "outputs" / "dedup_seen.sqlite"
DEFAULT_PR_STORE = MLX_ROOT / "outputs" / "pr_ingest" / "prs.sqlite"
DEFAULT_REPO_LIST = MLX_ROOT / "outputs" / "pr_ingest" / "repo_list.json"

DEFAULT_TARGET_LENGTHS_CODE = (1024, 2048, 4096)
DEFAULT_TARGET_LENGTHS_COMMITS = (1024, 2048, 4096, 8192, 16384)
DEFAULT_PROGRESS_JSONL = CONVEYOR_ROOT / "progress.jsonl"
DEFAULT_DEDUP_CHECKPOINT_TOKENS = 25_000_000
DEFAULT_RUN_LOCK_DIR = CONVEYOR_ROOT / "locks"
DEFAULT_WORK_PARENT = CONVEYOR_ROOT / "tmp"
DEFAULT_RESERVATION_FILE = CONVEYOR_ROOT / "_reservations.json"
DEFAULT_RANGE_SUBMIT_WINDOW_MULTIPLIER = 2
DEFAULT_MEMORY_BUDGET_FRACTION = 0.55
DEFAULT_MEMORY_BUDGET_FALLBACK_GB = 48.0
DEFAULT_MIN_RETRY_RANGE_SIZE = 25
DEFAULT_RANGE_TARGET_BYTES = 32 * 1024 * 1024
DEFAULT_DEDUP_PROMOTE_BATCH_SIZE = 8
DEFAULT_MIN_FREE_DISK_GB = 50.0

# Stable, repo-keyed cache for the EXPENSIVE git-history extraction output. This
# deliberately lives OUTSIDE the randomized per-run work_root so the commit
# records (e.g. php-src's ~6h / 10GB jsonl) survive a kill/restart with the SAME
# args -- the new run gets a fresh random work_root, so a per-run jsonl would be
# orphaned and re-extracted from scratch. The cache holds <repo>_commits.jsonl
# plus a <repo>_commits.jsonl.done sentinel (line/size/mtime), and is only
# deleted once EVERY unit of the repo (code + all ranges) is marked done.
EXTRACT_CACHE_ROOT = CONVEYOR_ROOT / "extract_cache"

_PRINT_LOCK = threading.Lock()


def configure_output_roots(
    *,
    code_output_root: str | os.PathLike[str] | None = None,
    commit_output_root: str | os.PathLike[str] | None = None,
    conveyor_root: str | os.PathLike[str] | None = None,
) -> None:
    """Rebase all runtime output roots used by the conveyor.

    The conveyor orchestrates code and commit drivers that historically kept
    module-level output constants. This explicit production seam lets a full
    regeneration run write to a fresh tree instead of mixing new packed shards
    with legacy ``outputs/reindexed*`` files.
    """

    global CONVEYOR_ROOT
    global CONVEYOR_MANIFEST
    global DEFAULT_PROGRESS_JSONL
    global DEFAULT_RUN_LOCK_DIR
    global DEFAULT_WORK_PARENT
    global DEFAULT_RESERVATION_FILE
    global EXTRACT_CACHE_ROOT

    if code_output_root is not None:
        sr.OUTPUT_ROOT = Path(code_output_root)
        sr.MANIFEST_PATH = sr.OUTPUT_ROOT / "_done.json"
    if commit_output_root is not None:
        src.COMMIT_OUTPUT_ROOT = Path(commit_output_root)
        src.COMMIT_MANIFEST = src.COMMIT_OUTPUT_ROOT / "_done.json"
    if conveyor_root is not None:
        CONVEYOR_ROOT = Path(conveyor_root)
        CONVEYOR_MANIFEST = CONVEYOR_ROOT / "_done.json"
        DEFAULT_PROGRESS_JSONL = CONVEYOR_ROOT / "progress.jsonl"
        DEFAULT_RUN_LOCK_DIR = CONVEYOR_ROOT / "locks"
        DEFAULT_WORK_PARENT = CONVEYOR_ROOT / "tmp"
        DEFAULT_RESERVATION_FILE = CONVEYOR_ROOT / "_reservations.json"
        EXTRACT_CACHE_ROOT = CONVEYOR_ROOT / "extract_cache"


def configure_runtime_paths_from_args(args: argparse.Namespace) -> None:
    old_defaults = {
        "progress_jsonl": DEFAULT_PROGRESS_JSONL,
        "run_lock_dir": DEFAULT_RUN_LOCK_DIR,
        "work_parent_dir": DEFAULT_WORK_PARENT,
        "reservation_file": DEFAULT_RESERVATION_FILE,
    }
    configure_output_roots(
        code_output_root=args.code_output_root,
        commit_output_root=args.commit_output_root,
        conveyor_root=args.conveyor_root,
    )

    if Path(args.progress_jsonl) == old_defaults["progress_jsonl"]:
        args.progress_jsonl = str(DEFAULT_PROGRESS_JSONL)
    if Path(args.run_lock_dir) == old_defaults["run_lock_dir"]:
        args.run_lock_dir = str(DEFAULT_RUN_LOCK_DIR)
    if Path(args.work_parent_dir) == old_defaults["work_parent_dir"]:
        args.work_parent_dir = str(DEFAULT_WORK_PARENT)
    if Path(args.reservation_file) == old_defaults["reservation_file"]:
        args.reservation_file = str(DEFAULT_RESERVATION_FILE)


class RepoNoCommitRecords(RuntimeError):
    """Commit stream completed discovery but has no trainable commit records."""

    def __init__(self, repo: str, *, reason: str, detail: str):
        super().__init__(f"[{repo}] no commit records: {reason}: {detail}")
        self.repo = repo
        self.reason = reason
        self.detail = detail


# Cooperative shutdown flag set by the SIGINT/SIGTERM handlers in main(). When
# set, the conveyor stops SUBMITTING new repos/ranges, lets in-flight subprocess
# tasks drain, and cancels queued-but-unstarted range futures (their units stay
# un-marked so resume re-runs them). A SECOND signal forces an immediate exit.
STOP_EVENT = threading.Event()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, file=sys.stderr, flush=True)


class BackgroundRecompressor:
    """Track deferred parquet recompress jobs and surface failures at shutdown."""

    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._lock = threading.Lock()
        self._futures: list[tuple[Path, Future]] = []
        self._handled: set[int] = set()

    def submit(self, path: Path) -> Future:
        fut = self._pool.submit(src.recompress_zstd_max, path)
        with self._lock:
            self._futures.append((path, fut))
        return fut

    def wait(self, jobs: Sequence[tuple[Path, Future]]) -> None:
        failures: list[str] = []
        try:
            for path, fut in jobs:
                try:
                    fut.result()
                except Exception as exc:
                    failures.append(f"{path}: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._handled.update(id(fut) for _path, fut in jobs)
        if failures:
            raise RuntimeError(
                "background code parquet recompress failed:\n"
                + "\n".join(failures[:20])
            )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        failures: list[str] = []
        with self._lock:
            futures = [
                (path, fut)
                for path, fut in self._futures
                if id(fut) not in self._handled
            ]
        for path, fut in futures:
            try:
                fut.result()
            except Exception as exc:
                failures.append(f"{path}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(
                "background code parquet recompress failed:\n"
                + "\n".join(failures[:20])
            )


def disk_free_gb(path: Path) -> float:
    """Return free space on the filesystem containing ``path`` in GiB."""
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    return float(usage.free) / (1024**3)


def ensure_min_free_disk(path: Path, min_free_gb: float, *, context: str) -> None:
    """Fail loud before staging more raw source when the work disk is too full."""
    threshold = float(min_free_gb or 0.0)
    if threshold <= 0.0:
        return
    free = disk_free_gb(path)
    if free < threshold:
        raise SystemExit(
            f"unsafe conveyor disk state before {context}: "
            f"{free:.2f} GiB free under {path}, "
            f"minimum required is {threshold:.2f} GiB. "
            "Clean outputs/conveyor/tmp or lower --min-free-disk-gb explicitly."
        )


def remove_tree(path: Path, *, reason: str) -> None:
    """Best-effort tree removal with a single useful log line."""
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
    _log(f"CLEANUP {reason}: removed {path}")


def physical_memory_gb() -> float | None:
    """Best-effort physical RAM in GiB using only stdlib/system tools."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and pages > 0:
            return float(page_size * pages) / (1024**3)
    except (AttributeError, OSError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(int(proc.stdout.strip())) / (1024**3)
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None
    return None


def default_memory_budget_gb() -> float:
    total = physical_memory_gb()
    if total is None:
        return DEFAULT_MEMORY_BUDGET_FALLBACK_GB
    return max(8.0, total * DEFAULT_MEMORY_BUDGET_FRACTION)


def heavy_subprocess_slots(*, streams: str, workers: int, repo_workers: int) -> int:
    """Conservative count of concurrently alive heavy stage subprocesses.

    Commit range workers spawn process_commits/materialize/pack. Code repo workers
    spawn index_project/materialize/pack. The exact stage mix changes over time,
    but this bound catches configurations like 20 repo workers + 16 range workers
    before they can launch hundreds of GiB worth of native clang processes.
    """
    slots = 0
    if streams in {"both", "commits"}:
        slots += max(1, int(workers))
    if streams in {"both", "code"}:
        slots += max(1, int(repo_workers))
    return max(1, slots)


def validate_memory_plan(
    *,
    streams: str,
    workers: int,
    repo_workers: int,
    memory_limit_gb: float,
    code_memory_limit_gb: float | None = None,
    commit_memory_limit_gb: float | None = None,
    memory_budget_gb: float,
    allow_oversubscription: bool,
) -> dict[str, float | int | bool]:
    """Fail before extraction when requested parallelism cannot fit RAM."""
    stage_limit = max(0.0, float(memory_limit_gb))
    code_limit = (
        stage_limit if code_memory_limit_gb is None
        else max(0.0, float(code_memory_limit_gb))
    )
    commit_limit = (
        stage_limit if commit_memory_limit_gb is None
        else max(0.0, float(commit_memory_limit_gb))
    )
    commit_slots = max(1, int(workers)) if streams in {"both", "commits"} else 0
    code_slots = max(1, int(repo_workers)) if streams in {"both", "code"} else 0
    slots = max(1, commit_slots + code_slots)
    budget = max(0.0, float(memory_budget_gb))
    code_reserved = code_slots * code_limit
    commit_reserved = commit_slots * commit_limit
    reserved = code_reserved + commit_reserved
    plan = {
        "heavy_slots": slots,
        "code_heavy_slots": code_slots,
        "commit_heavy_slots": commit_slots,
        "memory_limit_gb": stage_limit,
        "code_memory_limit_gb": code_limit,
        "commit_memory_limit_gb": commit_limit,
        "code_reserved_gb": code_reserved,
        "commit_reserved_gb": commit_reserved,
        "memory_budget_gb": budget,
        "reserved_gb": reserved,
        "allow_oversubscription": bool(allow_oversubscription),
    }
    if budget > 0 and reserved > budget and not allow_oversubscription:
        raise SystemExit(
            "unsafe conveyor memory plan: "
            f"heavy_slots={slots} * --memory-limit-gb={stage_limit:.2f} "
            f"= {reserved:.2f} GiB exceeds --memory-budget-gb={budget:.2f}. "
            "Lower --workers/--repo-workers/--memory-limit-gb, increase the "
            "budget, or pass --allow-memory-oversubscription explicitly."
        )
    return plan


# --------------------------------------------------------------------------- #
# Extract checkpoint: never re-run the (~6h) extract_git_history for a repo whose
# commit extraction already completed. The records jsonl is written to a STABLE
# repo-keyed cache (EXTRACT_CACHE_ROOT/<repo>/) guarded by a .done sentinel.
# --------------------------------------------------------------------------- #
def extract_cache_dir(repo: str) -> Path:
    """Stable, repo-keyed dir holding <repo>_commits.jsonl[+.done]."""
    return EXTRACT_CACHE_ROOT / repo


def _extract_sentinel_path(jsonl: Path) -> Path:
    return Path(str(jsonl) + ".done")


def _count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def _write_extract_sentinel(jsonl: Path, n_records: int) -> Path:
    """Atomically stamp the completion sentinel next to ``jsonl``.

    Records line_count plus the jsonl's size+mtime so a later resume can validate
    the cache with a cheap stat() instead of re-reading the (10GB) file.
    """
    st = jsonl.stat()
    sentinel = _extract_sentinel_path(jsonl)
    payload = {
        "jsonl": str(jsonl),
        "line_count": int(n_records),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = sentinel.with_name(f"{sentinel.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(sentinel)  # atomic on POSIX
    return sentinel


def _read_valid_sentinel(jsonl: Path) -> int | None:
    """Return the recorded record count iff a sentinel matches ``jsonl`` exactly.

    A match requires the jsonl to exist non-empty AND its current size+mtime to
    equal the stamped values. Any mismatch (truncated by a kill mid-extract,
    corrupt sentinel, missing file) returns None -> the caller re-extracts the
    full records (the one clear path; NOT a degraded/partial output).
    """
    sentinel = _extract_sentinel_path(jsonl)
    if not jsonl.exists() or not sentinel.exists():
        return None
    st = jsonl.stat()
    if st.st_size == 0:
        return None
    try:
        meta = json.loads(sentinel.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if int(meta.get("size_bytes", -1)) != int(st.st_size):
        return None
    if int(meta.get("mtime_ns", -1)) != int(st.st_mtime_ns):
        return None
    lc = meta.get("line_count")
    return int(lc) if lc is not None else None


def _discover_existing_jsonl(repo: str, work_root: Path, work_parent: Path) -> Path | None:
    """Locate a pre-existing <repo>_commits.jsonl from this or a prior run.

    Deterministic search order: stable cache, the current work_root, then any
    prior randomized conveyor work dir under work_parent / DEFAULT_WORK_PARENT
    (back-compat: php-src was extracted by older code into a now-orphaned random
    work_root). Returns the first existing non-empty path, else None. Safe under
    the per-stream RunLock, which guarantees no OTHER live conveyor owns these
    dirs while we adopt from them.
    """
    candidates: list[Path] = [
        extract_cache_dir(repo) / f"{repo}_commits.jsonl",
        work_root / repo / f"{repo}_commits.jsonl",
    ]
    parents = {p for p in (work_parent, DEFAULT_WORK_PARENT) if p is not None}
    for parent in parents:
        if parent.exists():
            candidates.extend(
                sorted(parent.glob(f"streaming_conveyor_*/{repo}/{repo}_commits.jsonl"))
            )
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None


def ensure_commit_records(
    repo: str,
    repo_dir: Path,
    work_root: Path,
    work_parent: Path,
    manifest: Manifest,
    resume: bool,
) -> tuple[Path, int, str]:
    """Return (records_jsonl, n_records), running extract_git_history ONLY if needed.

    Resolution order (all writing/reading the STABLE EXTRACT_CACHE_ROOT/<repo>):
      (a) HIT: a valid .done sentinel matches the cached jsonl -> reuse instantly.
      (b) ADOPT/BACK-COMPAT: a jsonl is discoverable from this/a prior run (or the
          manifest already has a <repo>::r<...> range unit, proving the prior
          extract finished) -> adopt it into the cache, count lines, stamp the
          sentinel retroactively, reuse. This is what preserves php-src's ~6h on
          the upcoming restart.
      (c) FRESH: nothing reusable -> run extract_git_history into the cache and
          stamp the sentinel on success.
    RAISES (RULE #1) on empty git log / empty extract; never returns a partial set.
    """
    cache_dir = extract_cache_dir(repo)
    cache_jsonl = cache_dir / f"{repo}_commits.jsonl"
    repo_has_range_done = any(k.startswith(f"{repo}::r") for k in manifest.done)

    if resume:
        # (a) Cheap stat-validated cache hit.
        lc = _read_valid_sentinel(cache_jsonl)
        if lc is not None:
            _log(f"EXTRACT-CKPT HIT {repo}: reuse {cache_jsonl} ({lc} records); "
                 f"skip ~extract_git_history")
            return cache_jsonl, lc, "hit"

        # (b) Adopt an existing jsonl (stable cache miss but a prior extract exists).
        existing = _discover_existing_jsonl(repo, work_root, work_parent)
        if existing is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if existing.resolve() != cache_jsonl.resolve():
                cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(existing), str(cache_jsonl))  # rename within fs / copy across
            n = _count_jsonl_lines(cache_jsonl)
            if n == 0:
                raise RepoFailure(repo, "extract_git_history",
                                  f"adopted commit jsonl is empty: {cache_jsonl}")
            _write_extract_sentinel(cache_jsonl, n)
            tag = "BACK-COMPAT" if repo_has_range_done else "ADOPT"
            _log(f"EXTRACT-CKPT {tag} {repo}: adopted {existing} -> {cache_jsonl} "
                 f"({n} records); stamped sentinel; skip ~extract_git_history")
            return cache_jsonl, n, tag.lower().replace("-", "_")

        cache_status = "fresh"
        if repo_has_range_done:
            # Manifest proves the extract finished, but no jsonl survived anywhere.
            # extract_git_history is deterministic (git log order), so a fresh
            # re-extract reproduces the SAME record order and already-done ranges
            # still resume-skip. Loud, not silent.
            _log(f"EXTRACT-CKPT MISS {repo}: manifest has done ranges but NO jsonl "
                 f"found on disk; re-extracting (deterministic order preserves "
                 f"done-range alignment)")
            cache_status = "miss_reextract"
    else:
        cache_status = "fresh"

    # (c) Fresh extract into the stable cache. FAIL LOUD on empty git log / output.
    commit_list = get_commit_list(repo_dir)
    if not commit_list:
        raise RepoNoCommitRecords(
            repo,
            reason="no_matching_commits",
            detail="no --no-merges --diff-filter=M commits",
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    records_jsonl = stage_extract_commits(repo, repo_dir, cache_dir)
    n = _count_jsonl_lines(records_jsonl)
    if n == 0:
        raise RepoNoCommitRecords(
            repo,
            reason="no_cpp_commit_records",
            detail=f"zero records after extract for {repo}",
        )
    _write_extract_sentinel(records_jsonl, n)
    _log(f"EXTRACT-CKPT FRESH {repo}: extracted {n} records -> {records_jsonl}; "
         f"stamped sentinel")
    return records_jsonl, n, cache_status


def repo_fully_done(
    repo: str, manifest: Manifest, streams: str, all_ranges_done: bool
) -> bool:
    """True iff EVERY unit this run is responsible for is marked done in the manifest.

    Used to gate temp + extract-cache deletion: a mid-repo kill, a failed unit,
    or a signal-cancelled range leaves this False -> temp is RETAINED for resume.
    """
    code_ok = (streams not in {"both", "code"}) or manifest.is_done(code_key(repo))
    commits_ok = (streams not in {"both", "commits"}) or all_ranges_done
    return code_ok and commits_ok


def commit_plan_key(repo: str) -> str:
    return f"{repo}::commit_plan"


def manifest_complete_commit_ranges(
    repo: str, manifest: Manifest, range_size: int
) -> tuple[tuple[int, int], ...] | None:
    """Return done ranges only with an authoritative extracted record count.

    A range shorter than ``range_size`` is not EOF evidence because adaptive
    planning also cuts ranges at a byte target. Completion is proven only by a
    persisted commit-plan record and exact coverage of ``[0, n_records)``.
    """
    if range_size <= 0:
        raise ValueError(f"range_size must be positive, got {range_size}")
    # A repo-level failure records an earlier extraction attempt, not a unit of
    # range coverage. Once the authoritative plan and every range prove exact
    # coverage it is stale. A failed range remains authoritative and must keep
    # completion fail-closed until that same range key is marked done.
    if any(k.startswith(f"{repo}::r") for k in manifest.failed):
        return None

    plan = manifest.done.get(commit_plan_key(repo))
    if not isinstance(plan, dict) or plan.get("source") != "commit_plan":
        return None
    try:
        n_records = int(plan["n_records"])
    except (KeyError, TypeError, ValueError):
        return None
    if n_records <= 0:
        return None

    ranges: list[tuple[int, int]] = []
    prefix = f"{repo}::r"
    for key, info in manifest.done.items():
        if not key.startswith(prefix):
            continue
        raw = info.get("range") if isinstance(info, dict) else None
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            return None
        try:
            start = int(raw[0])
            end = int(raw[1])
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start:
            return None
        # The key and payload must agree; otherwise the manifest is not a
        # reliable source of coverage truth.
        try:
            key_start = int(key.rsplit("::r", 1)[1])
        except (IndexError, ValueError):
            return None
        if key_start != start:
            return None
        ranges.append((start, end))

    if not ranges:
        return None
    ranges.sort()
    expected_start = 0
    for start, end in ranges:
        if start != expected_start:
            return None
        if end - start > range_size:
            return None
        expected_start = end
    if expected_start != n_records:
        return None
    return tuple(ranges)


def mark_commit_stream_complete(
    repo: str,
    manifest: Manifest,
    manifest_lock: threading.Lock | None,
    complete_ranges: Sequence[tuple[int, int]],
) -> dict:
    """Persist aggregate completion proven by the plan and exact range coverage.

    ``Manifest.mark_done`` also removes an earlier aggregate failure. Keeping
    this as a derived summary avoids treating ``repo::commits`` as a second,
    independent source of completion truth.
    """
    if not complete_ranges:
        raise ValueError(f"cannot mark {repo}::commits complete without ranges")
    plan = manifest.done.get(commit_plan_key(repo))
    if not isinstance(plan, dict) or plan.get("source") != "commit_plan":
        raise RuntimeError(
            f"cannot mark {repo}::commits complete without authoritative commit_plan"
        )
    try:
        n_records = int(plan["n_records"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot mark {repo}::commits complete: invalid commit_plan n_records"
        ) from exc
    if complete_ranges[0][0] != 0 or complete_ranges[-1][1] != n_records:
        raise RuntimeError(
            f"cannot mark {repo}::commits complete: ranges do not cover "
            f"[0, {n_records})"
        )
    info = {
        "source": "commits",
        "repo": repo,
        "complete": True,
        "completion_proof": "commit_plan_exact_range_coverage",
        "n_records": n_records,
        "range_count": len(complete_ranges),
    }
    if manifest_lock is None:
        manifest.mark_done(f"{repo}::commits", info)
    else:
        with manifest_lock:
            manifest.mark_done(f"{repo}::commits", info)
    return info


def manifest_done_commit_intervals(repo: str, manifest: Manifest) -> tuple[tuple[int, int], ...]:
    """Return validated done commit intervals for ``repo``.

    Adaptive range splitting means a manifest key such as ``repo::r23000`` may
    cover only ``[23000, 23250)`` instead of the original 500-record range. All
    resume and cleanup decisions must therefore use the payload interval, not
    only the key.
    """
    intervals: list[tuple[int, int]] = []
    prefix = f"{repo}::r"
    for key, info in manifest.done.items():
        if not key.startswith(prefix):
            continue
        raw = info.get("range") if isinstance(info, dict) else None
        if not isinstance(raw, list | tuple) or len(raw) != 2:
            continue
        try:
            start = int(raw[0])
            end = int(raw[1])
            key_start = int(key.rsplit("::r", 1)[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start < 0 or end <= start or key_start != start:
            continue
        intervals.append((start, end))
    return tuple(sorted(intervals))


def missing_commit_subranges(
    start: int, end: int, done_intervals: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Return uncovered subranges inside ``[start, end)``.

    Used on resume so an adaptively split completed half does not cause the
    uncompleted half to be skipped.
    """
    if end <= start:
        raise ValueError(f"invalid range [{start}:{end}]")
    missing: list[tuple[int, int]] = []
    cursor = start
    for done_start, done_end in done_intervals:
        if done_end <= cursor:
            continue
        if done_start >= end:
            break
        if done_start > cursor:
            missing.append((cursor, min(done_start, end)))
        cursor = max(cursor, min(done_end, end))
        if cursor >= end:
            break
    if cursor < end:
        missing.append((cursor, end))
    return tuple(missing)


def plan_commit_ranges(
    records_jsonl: Path,
    n_records: int,
    *,
    max_records: int,
    target_bytes: int = DEFAULT_RANGE_TARGET_BYTES,
) -> tuple[tuple[int, int], ...]:
    """Plan commit ranges bounded by both record count and raw JSONL bytes.

    Fixed-size 500-record ranges are badly imbalanced on repos with large diffs:
    one range can produce 2-3x the JSONL/materialization work of another. This
    keeps the old ``max_records`` ceiling while cutting earlier when the raw
    extracted commit-record bytes cross ``target_bytes``. Manifest keys remain
    ``repo::r<start>`` and payloads carry the exact ``[start, end)`` interval, so
    resume compatibility is preserved.
    """
    if n_records < 0:
        raise ValueError(f"n_records must be non-negative, got {n_records}")
    if max_records <= 0:
        raise ValueError(f"max_records must be positive, got {max_records}")
    if n_records == 0:
        return ()
    if target_bytes <= 0:
        return tuple(
            (s, min(s + max_records, n_records))
            for s in range(0, n_records, max_records)
        )

    ranges: list[tuple[int, int]] = []
    start = 0
    current_bytes = 0
    current_records = 0
    with records_jsonl.open("rb") as fh:
        for idx, line in enumerate(fh):
            if idx >= n_records:
                break
            line_bytes = len(line)
            if current_records and (
                current_records >= max_records
                or current_bytes + line_bytes > target_bytes
            ):
                ranges.append((start, idx))
                start = idx
                current_bytes = 0
                current_records = 0
            current_bytes += line_bytes
            current_records += 1

    if current_records:
        ranges.append((start, start + current_records))

    if not ranges or ranges[-1][1] != n_records:
        # This means the caller supplied a bad n_records for the file currently on
        # disk. Fail here instead of silently skipping or duplicating records.
        raise RuntimeError(
            f"range planning covered {ranges[-1][1] if ranges else 0} records "
            f"but n_records={n_records} for {records_jsonl}"
        )
    return tuple(ranges)


def manifest_covers_commit_span(repo: str, manifest: Manifest, n_records: int) -> bool:
    """True iff manifest done intervals cover ``[0, n_records)`` exactly enough."""
    if n_records <= 0:
        return False
    cursor = 0
    for start, end in manifest_done_commit_intervals(repo, manifest):
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= n_records:
            return True
    return False


class ConcurrentManifest(Manifest):
    """Cross-process-safe conveyor resume manifest (fixes the H4 clobber).

    The base ``streaming_reindex.Manifest.save()`` rewrites the whole file from
    its in-memory ``done``/``failed`` dicts with NO disk re-read and NO OS-level
    lock (``manifest_lock`` is only a ``threading.Lock`` -- useless across
    processes). Two conveyor processes that legitimately run at the same time --
    a ``--streams code`` run and a ``--streams commits`` run, each holding only
    its OWN per-stream :class:`RunLock` -- therefore CLOBBER the single shared
    ``outputs/conveyor/_done.json``: whichever writes last overwrites the file
    with its own snapshot and drops the other stream's keys (lost update),
    silently destroying that stream's resume + token accounting even though the
    two key spaces are disjoint (``<repo>::code`` vs ``<repo>::r<start>``).

    This subclass makes every manifest mutation atomic ACROSS processes: under an
    exclusive ``flock`` on a sibling ``.lock`` file it RE-READS the on-disk
    manifest, MERGES only the one changed key into it, then atomically replaces
    the file. The in-memory dicts are refreshed to the merged on-disk state so
    in-process resume checks (``is_done``) also observe the other process's
    committed keys. ONE clear write path; any error RAISES (RULE #1). A blind
    full-file ``save()`` is disabled so the clobber cannot be reintroduced.
    """

    def __init__(self, path: Path, done: dict | None = None, failed: dict | None = None):
        # Bypass the dataclass __init__ so we can attach lock state.
        self.path = path
        self.done = dict(done or {})
        self.failed = dict(failed or {})
        self._lock_path = Path(str(path) + ".lock")
        self._thread_lock = threading.Lock()

    @classmethod
    def load(cls, path: Path) -> "ConcurrentManifest":
        base = Manifest.load(path)
        return cls(path=path, done=base.done, failed=base.failed)

    def _read_disk(self) -> tuple[dict, dict]:
        if self.path.exists():
            blob = json.loads(self.path.read_text())
            return blob.get("done", {}), blob.get("failed", {})
        return {}, {}

    def _atomic_replace(self, done: dict, failed: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps({"done": done, "failed": failed}, indent=2, sort_keys=True)
        )
        tmp.replace(self.path)  # atomic on POSIX

    def _merge_under_lock(self, apply_change) -> None:
        """Apply ``apply_change(done, failed)`` to the freshest on-disk state
        under an exclusive cross-process flock, then atomically persist and
        refresh the in-memory dicts to the merged result."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            fh = self._lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                done, failed = self._read_disk()
                apply_change(done, failed)
                self._atomic_replace(done, failed)
                self.done = done
                self.failed = failed
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()

    def mark_done(self, key: str, info: dict) -> None:
        def apply(done: dict, failed: dict) -> None:
            done[key] = info
            failed.pop(key, None)

        self._merge_under_lock(apply)

    def mark_failed(self, key: str, stage: str, detail: str) -> None:
        rec = {
            "stage": stage,
            "detail": detail[:2000],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        def apply(done: dict, failed: dict) -> None:
            failed[key] = rec

        self._merge_under_lock(apply)

    def save(self) -> None:  # pragma: no cover - guard against accidental reuse
        raise RuntimeError(
            "ConcurrentManifest persists via mark_done/mark_failed (atomic "
            "cross-process merge under flock); a blind save() would re-introduce "
            "the lost-update clobber this class exists to prevent."
        )


class ProgressWriter:
    """Append-only machine-readable progress events for live throughput tracking."""

    def __init__(self, path: Path | None):
        self.path = path
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.extract_cache_seen = 0
        self.extract_cache_hits = 0
        self.extract_cache_reused = 0
        self.extract_cache_status_counts: dict[str, int] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def extract_cache_metrics(self) -> dict:
        with self.lock:
            return {
                "seen": self.extract_cache_seen,
                "hits": self.extract_cache_hits,
                "hit_rate": (
                    self.extract_cache_hits / self.extract_cache_seen
                    if self.extract_cache_seen else 0.0
                ),
                "reused": self.extract_cache_reused,
                "reuse_rate": (
                    self.extract_cache_reused / self.extract_cache_seen
                    if self.extract_cache_seen else 0.0
                ),
                "status_counts": dict(self.extract_cache_status_counts),
            }

    def emit(self, event: str, **payload) -> None:
        if self.path is None:
            return
        now = time.time()
        with self.lock:
            if event == "extract_cache":
                status = str(payload.get("status") or "unknown")
                self.extract_cache_seen += 1
                self.extract_cache_status_counts[status] = (
                    self.extract_cache_status_counts.get(status, 0) + 1
                )
                if status == "hit":
                    self.extract_cache_hits += 1
                if status in {"hit", "adopt", "back_compat", "orphan_adopt"}:
                    self.extract_cache_reused += 1
                payload = {
                    **payload,
                    "extract_cache_seen": self.extract_cache_seen,
                    "extract_cache_hits": self.extract_cache_hits,
                    "extract_cache_hit_rate": (
                        self.extract_cache_hits / self.extract_cache_seen
                    ),
                    "extract_cache_reused": self.extract_cache_reused,
                    "extract_cache_reuse_rate": (
                        self.extract_cache_reused / self.extract_cache_seen
                    ),
                    "extract_cache_status_counts": dict(
                        self.extract_cache_status_counts
                    ),
                }
            row = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": round(now - self.started_at, 3),
                "event": event,
                **payload,
            }
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")


class RunLock:
    """Fail-loud per-stream process lock.

    The conveyor writes per-stream outputs and uses stream-local temp filenames.
    Two live conveyors for the same stream can therefore race on repo outputs.
    This lock prevents that class of bug before any repo is extracted.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.seek(0)
            holder = self._fh.read().strip()
            self._fh.close()
            self._fh = None
            raise RuntimeError(
                f"conveyor stream lock is already held: {self.path}\n"
                f"holder:\n{holder}"
            ) from exc
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "argv": sys.argv,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class WorkReservation:
    """Handle for one active conveyor unit reservation."""

    def __init__(
        self,
        ledger: "UnitReservationLedger",
        key: str,
        token: str | None,
        holder: dict | None,
    ):
        self.ledger = ledger
        self.key = key
        self.token = token
        self.holder = holder
        self.acquired = token is not None
        self._released = False

    def release(self) -> None:
        if not self.acquired or self._released:
            return
        assert self.token is not None
        self.ledger.release(self.key, self.token)
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class UnitReservationLedger:
    """Small cross-process active-unit ledger for high-parallel conveyor runs.

    The dedup DB no longer accepts staging writes from subprocesses. This ledger
    solves a different problem: do not let two parent conveyor workers process
    the same output unit at once when repo/range parallelism is high or when
    separate stream-specific conveyors share the manifest. It is a tiny JSON
    file under an OS flock; completed history still lives in the manifest.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock_path = Path(str(self.path) + ".lock")
        self._thread_lock = threading.Lock()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read(self) -> dict:
        if not self.path.exists():
            return {"active": {}}
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"reservation ledger is corrupt: {self.path}") from exc
        active = blob.get("active", {})
        if not isinstance(active, dict):
            raise RuntimeError(f"reservation ledger has invalid active map: {self.path}")
        return {"active": active}

    def _atomic_replace(self, blob: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def _with_lock(self, fn):
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            fh = self._lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                blob = self._read()
                result = fn(blob)
                self._atomic_replace(blob)
                return result
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()

    def cleanup_stale(self) -> int:
        """Drop reservations whose owning parent process no longer exists."""

        def apply(blob: dict) -> int:
            active = blob["active"]
            stale = [
                key for key, rec in active.items()
                if not self._pid_alive(int(rec.get("pid", -1)))
            ]
            for key in stale:
                active.pop(key, None)
            return len(stale)

        return int(self._with_lock(apply))

    def acquire(self, key: str, *, stream: str, repo: str) -> WorkReservation:
        if not key:
            raise ValueError("reservation key must be non-empty")
        token = f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}"

        def apply(blob: dict) -> WorkReservation:
            active = blob["active"]
            stale = [
                item_key for item_key, rec in active.items()
                if not self._pid_alive(int(rec.get("pid", -1)))
            ]
            for item_key in stale:
                active.pop(item_key, None)
            holder = active.get(key)
            if holder is not None:
                return WorkReservation(self, key, None, dict(holder))
            active[key] = {
                "key": key,
                "stream": stream,
                "repo": repo,
                "pid": os.getpid(),
                "thread": threading.get_ident(),
                "token": token,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "argv": sys.argv,
            }
            return WorkReservation(self, key, token, None)

        return self._with_lock(apply)

    def release(self, key: str, token: str) -> None:
        def apply(blob: dict) -> None:
            active = blob["active"]
            rec = active.get(key)
            if rec is not None and rec.get("token") == token:
                active.pop(key, None)

        self._with_lock(apply)


class DedupCheckpointController:
    """Token-milestone SQLite WAL checkpoints for the shared dedup store."""

    def __init__(
        self,
        *,
        dedup_db: Path | None,
        interval_tokens: int,
        mode: str,
        busy_timeout_ms: int,
        progress: ProgressWriter,
    ):
        self.dedup_db = dedup_db
        self.interval_tokens = int(interval_tokens)
        self.mode = mode.upper()
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.progress = progress
        self.lock = threading.Lock()
        self.next_threshold = (
            self.interval_tokens if self.dedup_db and self.interval_tokens > 0 else None
        )

    @staticmethod
    def _wal_path(db_path: Path) -> Path:
        return Path(str(db_path) + "-wal")

    def maybe_checkpoint(self, cumulative_valid_tokens: int) -> None:
        if self.next_threshold is None:
            return
        with self.lock:
            if self.next_threshold is None or cumulative_valid_tokens < self.next_threshold:
                return
            threshold = self.next_threshold
            while self.next_threshold <= cumulative_valid_tokens:
                self.next_threshold += self.interval_tokens
        self._checkpoint(threshold, cumulative_valid_tokens)

    def _checkpoint(self, threshold: int, cumulative_valid_tokens: int) -> None:
        assert self.dedup_db is not None
        wal = self._wal_path(self.dedup_db)
        wal_before = wal.stat().st_size if wal.exists() else 0
        started = time.time()
        try:
            conn = sqlite3.connect(
                str(self.dedup_db),
                timeout=max(self.busy_timeout_ms / 1000.0, 0.001),
            )
            try:
                conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
                row = conn.execute(f"PRAGMA wal_checkpoint({self.mode})").fetchone()
            finally:
                conn.close()
            wal_after = wal.stat().st_size if wal.exists() else 0
            busy, log_pages, checkpointed_pages = [int(x) for x in row]
            self.progress.emit(
                "dedup_checkpoint",
                dedup_db=str(self.dedup_db),
                mode=self.mode,
                threshold_tokens=threshold,
                cumulative_valid_tokens=cumulative_valid_tokens,
                busy=busy,
                log_pages=log_pages,
                checkpointed_pages=checkpointed_pages,
                wal_bytes_before=wal_before,
                wal_bytes_after=wal_after,
                checkpoint_elapsed_s=round(time.time() - started, 3),
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint failure is telemetry, not data loss
            wal_after = wal.stat().st_size if wal.exists() else 0
            self.progress.emit(
                "dedup_checkpoint_failed",
                dedup_db=str(self.dedup_db),
                mode=self.mode,
                threshold_tokens=threshold,
                cumulative_valid_tokens=cumulative_valid_tokens,
                wal_bytes_before=wal_before,
                wal_bytes_after=wal_after,
                detail=str(exc)[:2000],
                checkpoint_elapsed_s=round(time.time() - started, 3),
            )


def _length_totals(info: dict) -> dict[str, int]:
    totals = {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0}
    for bucket, st in info.get("lengths", {}).items():
        for key in totals:
            # Fail-loud (RULE #1): a malformed stat dict must crash here rather
            # than silently default to 0 -- cumulative['valid'] drives the
            # --token-budget stop gate and the dedup WAL checkpoint thresholds,
            # so an undercount overshoots the budget / skips checkpoints. Mirror
            # the fail-loud st[...] reads in the run report aggregation below.
            if key not in st:
                raise KeyError(
                    f"_length_totals: malformed stat dict for length bucket "
                    f"{bucket!r} (source={info.get('source')!r}): missing field "
                    f"{key!r}; present fields = {sorted(st)}"
                )
            totals[key] += int(st[key])
    return totals


def _primary_bucket_progress(info: dict, lengths: Sequence[int]) -> dict[str, int]:
    """Return the smallest configured bucket's real-token count for live logs."""
    if not lengths:
        return {"primary_bucket_length": 0, "primary_bucket_valid_tokens": 0}
    primary = min(int(x) for x in lengths)
    stats = (info.get("lengths") or {}).get(str(primary), {})
    return {
        "primary_bucket_length": primary,
        "primary_bucket_valid_tokens": int(stats.get("valid_tokens", 0)),
    }


def code_key(repo: str) -> str:
    """Checkpoint key for one repo's CODE half."""
    return f"{repo}::code"


def no_git_key(repo: str) -> str:
    """Checkpoint key proving the .git-preserving stream saw no git metadata."""
    return f"{repo}::no_git"


def manifest_repo_known_no_git(repo: str, manifest: Manifest) -> bool:
    info = manifest.done.get(no_git_key(repo))
    return isinstance(info, dict) and info.get("no_git") is True


def should_stage_repo_from_manifest(
    repo: str,
    *,
    streams: str,
    resume: bool,
    manifest: Manifest,
    range_size: int,
    only_repos: set[str] | None,
    manifest_lock: threading.Lock | None = None,
) -> bool:
    """Return False when the manifest proves this repo needs no extraction.

    The conservative commit proof comes from ``manifest_complete_commit_ranges``:
    if it cannot prove full coverage without reading the repo's commit JSONL, we
    keep the old safe behavior and stage the repo.
    """
    if only_repos is not None:
        return repo in only_repos
    if not resume:
        return True
    if streams in {"both", "commits"} and manifest_repo_known_no_git(repo, manifest):
        return False

    code_needed = streams in {"both", "code"} and not manifest.is_done(code_key(repo))
    commits_needed = False
    if streams in {"both", "commits"}:
        complete_ranges = manifest_complete_commit_ranges(repo, manifest, range_size)
        commits_needed = complete_ranges is None
        if complete_ranges is not None:
            # This callback can skip extraction entirely, so reconcile the
            # derived aggregate sentinel here instead of waiting for
            # run_commits_half, which will never be called for this repo.
            mark_commit_stream_complete(
                repo,
                manifest,
                manifest_lock,
                complete_ranges,
            )
    return code_needed or commits_needed


def stream_lock_names(streams: str) -> tuple[str, ...]:
    if streams == "both":
        return ("code", "commits")
    return (streams,)


def populate_code_source_cache(
    work_root: Path,
    source_cache_dir: Path,
    should_process,
    progress: ProgressWriter,
    *,
    max_repos: int | None = None,
) -> dict:
    """Populate the code source cache without running index/tokenize stages."""
    started = time.monotonic()

    def emit_repo_ready(repo: str, repo_dir: Path, repo_count: int) -> None:
        progress.emit(
            "source_cache_repo_ready",
            repo=repo,
            repo_dir=str(repo_dir),
            source_cache_dir=str(source_cache_dir),
            repo_count=repo_count,
        )

    report = sr.populate_source_cache(
        work_root,
        should_process,
        source_cache_dir,
        max_repos=max_repos,
        on_repo_ready=emit_repo_ready,
    )
    report = {
        **report,
        "elapsed_s": round(time.monotonic() - started, 6),
    }
    progress.emit("source_cache_populated", **report)
    return report


# --------------------------------------------------------------------------- #
# CODE half: orchestrate streaming_reindex.process_one_repo (no reimplementation).
# Its append_output already lands outputs/reindexed/<L>/<repo>.parquet; we then
# recompress those per-length files with MAX zstd to match the commit stream.
# --------------------------------------------------------------------------- #
def run_code_half(
    repo: str,
    repo_dir: Path,
    lengths_code: Sequence[int],
    work_root: Path,
    dedup_db: Path | None,
    dedup_near: bool,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
    recompressor: BackgroundRecompressor | None = None,
) -> dict:
    """index+route+pack the repo's source via the EXISTING code stage, zstd-max.

    Calls streaming_reindex.process_one_repo verbatim (it threads the SHARED
    dedup_db into index_project). Then recompresses each produced
    outputs/reindexed/<L>/<repo>.parquet with MAX zstd (the code stage writes
    plain parquet; the commit stage already recompresses, so we make code match).
    RAISES RepoFailure on any stage failure (no fallback).
    """
    stage_id = sr.code_stage_id(repo) if dedup_db is not None else None
    stage_db = sr.code_stage_db(work_root / repo, repo) if dedup_db is not None else None
    promoted = False
    try:
        try:
            info = sr.process_one_repo(
                repo, repo_dir, lengths_code, work_root, dedup_db, dedup_near,
                global_symbol_index, memory_limit_gb, parse_workers, index_timeout_s,
                index_stall_timeout_s,
                promote_dedup_on_success=False,
            )
        except RepoFailure as exc:
            skip_reason = code_skip_reason(exc)
            if skip_reason is not None:
                return {
                    "source": "code",
                    "repo": repo,
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "lengths": {},
                    "stage_timings_s": {},
                    "detail": exc.detail,
                }
            raise
        if info.get("skipped"):
            return info
        timings = dict(info.get("stage_timings_s", {}))
        # Recompression is part of publication, not deferred cleanup. Multiple
        # bucket files can still run concurrently, but this unit cannot promote
        # dedup or become manifest-done until every recompress future succeeds.
        started = time.monotonic()
        jobs: list[tuple[Path, Future]] = []
        try:
            for L in info.get("lengths", {}):
                dest = sr.OUTPUT_ROOT / str(L) / f"{repo}.parquet"
                if dest.exists():
                    if recompressor is None:
                        src.recompress_zstd_max(dest)
                    else:
                        jobs.append((dest, recompressor.submit(dest)))
            if recompressor is not None:
                recompressor.wait(jobs)
        except Exception as exc:
            remove_code_outputs(repo, info.get("lengths", {}).keys())
            raise RepoFailure(repo, "recompress", f"{type(exc).__name__}: {exc}") from exc
        timings["recompress_s"] = round(time.monotonic() - started, 6)

        try:
            timings.update(sr.promote_dedup_stage(dedup_db, stage_id, stage_db))
        except Exception as exc:
            remove_code_outputs(repo, info.get("lengths", {}).keys())
            raise RepoFailure(
                repo,
                "dedup_promote",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        promoted = True
        info["stage_timings_s"] = timings
        return info
    finally:
        if not promoted:
            sr.discard_dedup_stage(dedup_db, stage_id, stage_db)


CodeRunner = Callable[
    [
        str,
        Path,
        Sequence[int],
        Path,
        Path | None,
        bool,
        Path | None,
        float,
        int,
        int | None,
        int | None,
        BackgroundRecompressor | None,
    ],
    dict,
]


def remove_code_outputs(repo: str, lengths: Iterable[str | int]) -> None:
    for length in lengths:
        dest = sr.OUTPUT_ROOT / str(length) / f"{repo}.parquet"
        if dest.exists():
            dest.unlink()


def is_no_trainable_source_failure(exc: RepoFailure) -> bool:
    return (
        exc.stage == "index_project"
        and "no training docs (no_trainable_source)" in exc.detail.lower()
    )


def code_skip_reason(exc: RepoFailure) -> str | None:
    if is_no_trainable_source_failure(exc):
        return "no_trainable_source"
    if (
        exc.stage == "index_project"
        and "no training docs (dedup_exhausted)" in exc.detail.lower()
    ):
        return "dedup_exhausted"
    return None


def is_retryable_index_project_failure(exc: RepoFailure) -> bool:
    detail = exc.detail.lower()
    return (
        exc.stage == "index_project"
        and (
            "exceeded memory limit" in detail
            or "exit code 137" in detail
            or "killed: 9" in detail
            or "timed out after" in detail
            or "stalled after" in detail
        )
    )


def is_index_project_memory_failure(exc: RepoFailure) -> bool:
    return is_retryable_index_project_failure(exc)


_PARSE_WORKERS_RE = re.compile(r"\bUsing\s+(\d+)\s+parse workers\b", re.IGNORECASE)


def _failure_parse_workers(detail: str) -> int | None:
    match = _PARSE_WORKERS_RE.search(detail)
    if match is None:
        return None
    return int(match.group(1))


def failed_code_unit_was_index_memory(
    repo: str,
    manifest: Manifest,
    *,
    parse_workers: int | None = None,
) -> bool:
    rec = manifest.failed.get(code_key(repo))
    if not isinstance(rec, dict):
        return False
    detail = str(rec.get("detail", "")).lower()
    is_retryable_failure = (
        rec.get("stage") == "index_project"
        and (
            "exceeded memory limit" in detail
            or "exit code 137" in detail
            or "killed: 9" in detail
            or "timed out after" in detail
            or "stalled after" in detail
        )
    )
    if not is_retryable_failure:
        return False
    if parse_workers is not None:
        prior_parse_workers = _failure_parse_workers(detail)
        if prior_parse_workers is not None and prior_parse_workers > parse_workers:
            return False
    return True


def run_code_half_adaptive(
    repo: str,
    repo_dir: Path,
    lengths_code: Sequence[int],
    work_root: Path,
    dedup_db: Path | None,
    dedup_near: bool,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
    *,
    runner: CodeRunner | None = None,
    recompressor: BackgroundRecompressor | None = None,
) -> dict:
    """Run the code half, retrying index_project peaks/stalls with one parser.

    Large repos can peak while merging multiprocessing parse payloads into the
    repo-wide ProjectIndex; some also stall in parser-heavy paths. Retrying with
    one parser preserves the same enriched output contract and graph routes
    while removing avoidable IPC/parser concurrency pressure.
    """
    active_runner = run_code_half if runner is None else runner
    try:
        return active_runner(
            repo,
            repo_dir,
            lengths_code,
            work_root,
            dedup_db,
            dedup_near,
            global_symbol_index,
            memory_limit_gb,
            parse_workers,
            index_timeout_s,
            index_stall_timeout_s,
            recompressor,
        )
    except RepoFailure as exc:
        if parse_workers <= 1 or not is_retryable_index_project_failure(exc):
            raise
        _log(
            f"RETRY {code_key(repo)}: {exc.stage} retryable failure with "
            f"parse_workers={parse_workers}; retrying parse_workers=1 with "
            "global exact/chunk dedup and near-dedup disabled"
        )
        return active_runner(
            repo,
            repo_dir,
            lengths_code,
            work_root,
            dedup_db,
            False,
            global_symbol_index,
            memory_limit_gb,
            1,
            index_timeout_s,
            index_stall_timeout_s,
            recompressor,
        )


RangeRunner = Callable[
    [
        str,
        Path,
        Path,
        int,
        int,
        Sequence[int],
        Path,
        Path | None,
        bool,
        Path | None,
        Path | None,
        float,
        int,
    ],
    dict,
]

CommitRecordsProvider = Callable[
    [
        str,
        Path,
        Path,
        Path,
        Manifest,
        bool,
    ],
    tuple[Path, int, str],
]


def is_splitworthy_range_failure(exc: RepoFailure) -> bool:
    """Return True for range-local peak failures that may shrink by splitting."""
    detail = exc.detail.lower()
    return (
        exc.stage in {"process_commits", "materialize", "pack"}
        and (
            "exceeded memory limit" in detail
            or "exit code 137" in detail
            or "killed: 9" in detail
        )
    )


def process_range_adaptive(
    repo: str,
    repo_dir: Path,
    records_jsonl: Path,
    start: int,
    end: int,
    lengths_sorted: Sequence[int],
    repo_work: Path,
    dedup_db: Path | None,
    dedup_near: bool,
    pr_store: Path | None,
    repo_list: Path | None,
    memory_limit_gb: float,
    *,
    analysis_cache_entries: int = 128,
    min_range_size: int = DEFAULT_MIN_RETRY_RANGE_SIZE,
    runner: RangeRunner | None = None,
) -> dict[str, list[tuple[int, int, dict]] | list[tuple[int, int, RepoFailure]]]:
    """Run a commit range, recursively splitting only peak-OOM failures.

    Large C++ repos contain pathological commits/ranges whose enriched sidecars
    transiently peak above the per-stage RSS guard. Splitting the affected range
    preserves the exact same downstream processors and manifest accounting while
    keeping retry work transactional.
    """
    if end <= start:
        raise ValueError(f"invalid range [{start}:{end}]")
    if min_range_size <= 0:
        raise ValueError(f"min_range_size must be positive, got {min_range_size}")
    active_runner = process_range if runner is None else runner

    try:
        info = active_runner(
            repo,
            repo_dir,
            records_jsonl,
            start,
            end,
            lengths_sorted,
            repo_work,
            dedup_db,
            dedup_near,
            pr_store,
            repo_list,
            memory_limit_gb,
            analysis_cache_entries,
        )
        return {"done": [(start, end, info)], "failed": []}
    except RepoFailure as exc:
        if (end - start) <= min_range_size or not is_splitworthy_range_failure(exc):
            return {"done": [], "failed": [(start, end, exc)]}
        mid = start + ((end - start) // 2)
        _log(
            f"SPLIT {range_key(repo, start)} [{start}:{end}] after "
            f"{exc.stage} peak failure -> [{start}:{mid}] + [{mid}:{end}]"
        )
        left = process_range_adaptive(
            repo,
            repo_dir,
            records_jsonl,
            start,
            mid,
            lengths_sorted,
            repo_work,
            dedup_db,
            dedup_near,
            pr_store,
            repo_list,
            memory_limit_gb,
            analysis_cache_entries=analysis_cache_entries,
            min_range_size=min_range_size,
            runner=active_runner,
        )
        right = process_range_adaptive(
            repo,
            repo_dir,
            records_jsonl,
            mid,
            end,
            lengths_sorted,
            repo_work,
            dedup_db,
            dedup_near,
            pr_store,
            repo_list,
            memory_limit_gb,
            analysis_cache_entries=analysis_cache_entries,
            min_range_size=min_range_size,
            runner=active_runner,
        )
        return {
            "done": [*left["done"], *right["done"]],
            "failed": [*left["failed"], *right["failed"]],
        }


def run_bounded_future_queue(
    items,
    *,
    max_pending: int,
    submit: Callable,
    handle_done: Callable,
    stop_event: threading.Event | None = None,
    cancel_pending_on_stop: bool = False,
) -> int:
    """Submit futures lazily and keep at most ``max_pending`` outstanding.

    ``submit(item)`` returns ``(future, state)`` or ``None`` to skip an item.
    ``handle_done(item, future, state)`` is called for every submitted future,
    including futures cancelled by ``cancel_pending_on_stop``.
    """
    if max_pending < 1:
        raise ValueError("max_pending must be >= 1")

    iterator = iter(items)
    futures = {}
    submitted = 0
    exhausted = False
    cancelled_pending = False

    def fill_window() -> None:
        nonlocal exhausted, submitted
        while len(futures) < max_pending and not exhausted:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                return
            submitted_future = submit(item)
            if submitted_future is None:
                continue
            future, state = submitted_future
            futures[future] = (item, state)
            submitted += 1

    fill_window()
    while futures:
        if (
            stop_event is not None
            and stop_event.is_set()
            and cancel_pending_on_stop
            and not cancelled_pending
        ):
            for future in list(futures):
                if not future.done():
                    future.cancel()
            cancelled_pending = True
        done, _pending = wait(futures.keys(), return_when=FIRST_COMPLETED)
        for future in done:
            item, state = futures.pop(future)
            handle_done(item, future, state)
        fill_window()

    return submitted


# --------------------------------------------------------------------------- #
# COMMITS half: orchestrate the commit range pipeline (no reimplementation).
# extract_git_history once -> delete .git -> fan ranges to the pool via the
# EXISTING process_range (which runs process_commits --pr-store/--repo-list with
# the SHARED dedup_db, then materialize/route/pack/zstd-max).
# --------------------------------------------------------------------------- #
def run_commits_half(
    repo: str,
    repo_dir: Path,
    repo_work: Path,
    work_root: Path,
    work_parent: Path,
    lengths_commits: Sequence[int],
    range_size: int,
    pool: ThreadPoolExecutor,
    manifest: Manifest,
    manifest_lock: threading.Lock,
    resume: bool,
    cumulative: dict,
    dedup_db: Path | None,
    dedup_near: bool,
    pr_store: Path | None,
    repo_list: Path | None,
    memory_limit_gb: float = 10.0,
    progress: ProgressWriter | None = None,
    checkpoint: DedupCheckpointController | None = None,
    reservations: UnitReservationLedger | None = None,
    range_submit_window: int | None = None,
    analysis_cache_entries: int = 128,
    range_target_bytes: int = DEFAULT_RANGE_TARGET_BYTES,
    dedup_promote_batch_size: int = DEFAULT_DEDUP_PROMOTE_BATCH_SIZE,
    range_runner_override: RangeRunner | None = None,
    commit_records_override: CommitRecordsProvider | None = None,
) -> tuple[int, int, bool]:
    """Extract commit records once (checkpointed), fan ranges to the pool.

    Returns ``(done, failed, all_ranges_done)``. The records jsonl comes from
    :func:`ensure_commit_records`, which REUSES a stable cached extract whenever
    possible (never re-running the ~6h extract_git_history on resume). .git is
    deleted immediately after the records are available, keeping disk bounded.
    Each range is checkpointed exactly as ``<repo>::r<start>`` so resume skips
    finished ranges. On STOP_EVENT we stop submitting new ranges and cancel any
    queued-but-unstarted ones (their units stay un-marked -> resume re-runs them),
    while in-flight ranges drain. RAISES RepoFailure only for the up-front
    git-log / extract steps; per-range failures are recorded and skipped.
    ``all_ranges_done`` is True iff EVERY range key is marked done afterwards
    (drives temp/extract-cache retention).
    """
    lengths_sorted = tuple(sorted(int(x) for x in lengths_commits))
    smallest = lengths_sorted[0]

    if resume:
        complete_ranges = manifest_complete_commit_ranges(repo, manifest, range_size)
        if complete_ranges is not None:
            first = complete_ranges[0][0]
            last = complete_ranges[-1][1]
            mark_commit_stream_complete(
                repo,
                manifest,
                manifest_lock,
                complete_ranges,
            )
            _log(
                f"SKIP (done) {repo}::commits: manifest covers "
                f"{len(complete_ranges)} range(s) [{first}:{last}); "
                "skip ~extract_git_history"
            )
            return 0, 0, True

    # Reuse a cached/adopted extract when available; never re-run the ~6h extract
    # for a repo whose commit extraction already completed (sentinel or manifest).
    records_provider = (
        ensure_commit_records if commit_records_override is None else commit_records_override
    )
    try:
        records_jsonl, n_records, extract_cache_status = records_provider(
            repo, repo_dir, work_root, work_parent, manifest, resume,
        )
    except RepoNoCommitRecords as exc:
        raise RepoFailure(
            repo,
            "extract_git_history",
            f"no commit records ({exc.reason}): {exc.detail}",
        ) from exc
    if progress is not None:
        progress.emit(
            "extract_cache",
            stream="commits",
            repo=repo,
            status=extract_cache_status,
            n_records=n_records,
            records_jsonl=str(records_jsonl),
        )

    # This is the only safe EOF proof. Old manifests without it are deliberately
    # re-staged and counted from the extraction cache before ranges are skipped.
    records_stat = records_jsonl.stat()
    with manifest_lock:
        manifest.mark_done(
            commit_plan_key(repo),
            {
                "source": "commit_plan",
                "repo": repo,
                "n_records": int(n_records),
                "records_size_bytes": int(records_stat.st_size),
            },
        )

    # .git no longer needed -> free disk now (records already captured/cached).
    git_dir = repo_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)

    # Range over ACTUAL emitted record count (extract keeps only C/C++-touching
    # commits, so the records JSONL is a subset; slicing by line index aligns).
    if n_records == 0:
        raise RepoFailure(repo, "extract_git_history",
                          f"zero records after extract for {repo}")
    ranges = plan_commit_ranges(
        records_jsonl,
        n_records,
        max_records=range_size,
        target_bytes=range_target_bytes,
    )
    if progress is not None:
        range_lengths = [end - start for start, end in ranges]
        progress.emit(
            "commit_range_plan",
            stream="commits",
            repo=repo,
            n_records=n_records,
            range_count=len(ranges),
            range_size=range_size,
            range_target_bytes=range_target_bytes,
            min_range_records=min(range_lengths) if range_lengths else 0,
            max_range_records=max(range_lengths) if range_lengths else 0,
        )
    if ranges:
        range_lengths = [end - start for start, end in ranges]
        _log(
            f"Commit range plan for {repo}: {len(ranges)} ranges, "
            f"records={n_records}, max_records={range_size}, "
            f"target_bytes={range_target_bytes}, "
            f"range_records=[{min(range_lengths)}..{max(range_lengths)}]"
        )

    done_intervals = manifest_done_commit_intervals(repo, manifest) if resume else ()
    pending_ranges: list[tuple[int, int]] = []
    for (start, end) in ranges:
        if STOP_EVENT.is_set():
            _log(f"STOP: not submitting further ranges for {repo} (from r{start})")
            break
        pending_subranges = (
            missing_commit_subranges(start, end, done_intervals)
            if resume
            else ((start, end),)
        )
        if not pending_subranges:
            _log(f"SKIP (done) {range_key(repo, start)}")
            continue
        for sub_start, sub_end in pending_subranges:
            pending_ranges.append((sub_start, sub_end))

    submit_window = max(
        1,
        int(range_submit_window or len(pending_ranges) or 1),
    )
    if pending_ranges and submit_window < len(pending_ranges):
        _log(
            f"Commit range queue for {repo}: {len(pending_ranges)} pending, "
            f"submit_window={submit_window}"
        )

    done = 0
    failed = 0
    promote_batch_size = max(1, int(dedup_promote_batch_size or 1))
    defer_range_promotes = dedup_db is not None and promote_batch_size > 1
    deferred_stage_dir = repo_work / "_deferred_promote"
    deferred_promotions: list[
        tuple[list[tuple[int, int, dict]], WorkReservation | None]
    ] = []

    def unlink_sqlite_family(db_path: Path) -> None:
        for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def range_runner(
        range_repo,
        range_repo_dir,
        range_records_jsonl,
        range_start,
        range_end,
        range_lengths_sorted,
        range_repo_work,
        range_dedup_db,
        range_dedup_near,
        range_pr_store,
        range_repo_list,
        range_memory_limit_gb,
        range_analysis_cache_entries,
    ):
        args = (
            range_repo,
            range_repo_dir,
            range_records_jsonl,
            range_start,
            range_end,
            range_lengths_sorted,
            range_repo_work,
            range_dedup_db,
            range_dedup_near,
            range_pr_store,
            range_repo_list,
            range_memory_limit_gb,
            range_analysis_cache_entries,
        )
        active_process_range = (
            process_range if range_runner_override is None else range_runner_override
        )
        if defer_range_promotes:
            return active_process_range(
                *args,
                defer_promote=True,
                deferred_stage_dir=deferred_stage_dir,
            )
        return active_process_range(*args)

    def mark_failed_range(
        failed_start: int,
        failed_end: int,
        exc: RepoFailure,
    ) -> None:
        nonlocal failed
        frkey = range_key(repo, failed_start)
        _log(f"FAIL {frkey}: {exc}")
        failed += 1
        with manifest_lock:
            manifest.mark_failed(frkey, exc.stage, exc.detail)
        if progress is not None:
            progress.emit(
                "unit_failed",
                stream="commits",
                repo=repo,
                unit=frkey,
                range=[failed_start, failed_end],
                stage=exc.stage,
                detail=exc.detail[:2000],
            )

    def mark_done_range(done_start: int, done_end: int, rinfo: dict) -> None:
        nonlocal done
        drkey = range_key(repo, done_start)
        rinfo["extract_cache_status"] = extract_cache_status
        with manifest_lock:
            manifest.mark_done(drkey, rinfo)
        done += 1
        totals = _length_totals(rinfo)
        added = rinfo["lengths"].get(str(smallest), {}).get("valid_tokens", 0)
        with manifest_lock:
            cumulative["valid"] += totals["valid_tokens"]
            cumulative_valid = cumulative["valid"]
        if progress is not None:
            progress.emit(
                "unit_done",
                stream="commits",
                repo=repo,
                unit=drkey,
                range=[done_start, done_end],
                rows=totals["rows"],
                valid_tokens=totals["valid_tokens"],
                capacity_tokens=totals["capacity_tokens"],
                **_primary_bucket_progress(rinfo, lengths_sorted),
                lengths=rinfo.get("lengths", {}),
                stage_timings_s=rinfo.get("stage_timings_s", {}),
                extract_cache_status=extract_cache_status,
                cumulative_valid_tokens=cumulative_valid,
            )
        if checkpoint is not None:
            checkpoint.maybe_checkpoint(cumulative_valid)
        _log(
            f"DONE {drkey}: ranges [{done_start}:{done_end}] "
            f"buckets={sorted(rinfo['lengths'].keys())} "
            f"(+{added} @ {smallest}, cum_all={cumulative['valid']})"
        )

    def flush_deferred_promotions() -> None:
        if not deferred_promotions:
            return
        batch = list(deferred_promotions)
        deferred_promotions.clear()
        stages: list[tuple[str, Path | None]] = []
        stage_paths: list[Path] = []
        for done_items, _reservation in batch:
            for _done_start, _done_end, rinfo in done_items:
                stage = rinfo.get("dedup_stage")
                if not stage:
                    continue
                stage_id = stage["stage_id"]
                stage_db = Path(stage["stage_db"])
                stages.append((stage_id, stage_db))
                stage_paths.append(stage_db)
        metrics = sr.promote_dedup_stages(dedup_db, stages)
        stage_count = max(1, int(metrics.get("promote_batch_size") or len(stages) or 1))
        per_wait = float(metrics.get("promote_wait_s", 0.0)) / stage_count
        per_duration = float(metrics.get("promote_duration_s", 0.0)) / stage_count
        try:
            for done_items, reservation in batch:
                try:
                    for done_start, done_end, rinfo in done_items:
                        stage = rinfo.pop("dedup_stage", None)
                        if stage:
                            timings = dict(rinfo.get("stage_timings_s", {}))
                            timings["promote_wait_s"] = round(per_wait, 6)
                            timings["promote_duration_s"] = round(per_duration, 6)
                            timings["promote_batch_size"] = stage_count
                            timings["promote_batch_wait_s"] = metrics.get(
                                "promote_wait_s", 0.0
                            )
                            timings["promote_batch_duration_s"] = metrics.get(
                                "promote_duration_s", 0.0
                            )
                            timings["promote_deferred"] = 0.0
                            rinfo["stage_timings_s"] = timings
                            rinfo["dedup_stage_promoted"] = {
                                "stage_id": stage["stage_id"],
                            }
                        mark_done_range(done_start, done_end, rinfo)
                finally:
                    if reservation is not None:
                        reservation.release()
        finally:
            for stage_db in stage_paths:
                unlink_sqlite_family(stage_db)

    def submit_commit_range(item: tuple[int, int]):
        sub_start, sub_end = item
        rkey = range_key(repo, sub_start)
        if STOP_EVENT.is_set():
            _log(f"STOP: not submitting further ranges for {repo} (from r{sub_start})")
            return None
        reservation: WorkReservation | None = None
        if reservations is not None:
            reservation = reservations.acquire(rkey, stream="commits", repo=repo)
            if not reservation.acquired:
                _log(
                    f"SKIP (reserved) {rkey}: holder="
                    f"{reservation.holder}"
                )
                return None
        try:
            if progress is not None:
                progress.emit(
                    "unit_started",
                    stream="commits",
                    repo=repo,
                    unit=rkey,
                    range=[sub_start, sub_end],
                    extract_cache_status=extract_cache_status,
                )
            fut = pool.submit(
                process_range_adaptive, repo, repo_dir, records_jsonl,
                sub_start, sub_end, lengths_sorted, repo_work,
                dedup_db, dedup_near, pr_store, repo_list,
                memory_limit_gb,
                analysis_cache_entries=analysis_cache_entries,
                runner=range_runner,
            )
        except Exception:
            if reservation is not None:
                reservation.release()
            raise
        return fut, reservation

    def handle_commit_range(
        item: tuple[int, int],
        fut,
        reservation: WorkReservation | None,
    ) -> None:
        nonlocal done, failed
        start, end = item
        rkey = range_key(repo, start)
        try:
            adaptive_result = fut.result()
        except CancelledError:
            # Never started -> leave un-marked so resume re-runs this range.
            return
        except RepoFailure as exc:
            mark_failed_range(start, end, exc)
            return
        except Exception as exc:  # surface unexpected failures loud
            mark_failed_range(
                start,
                end,
                RepoFailure(
                    repo,
                    "unexpected",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return
        else:
            for failed_start, failed_end, exc in adaptive_result["failed"]:
                mark_failed_range(failed_start, failed_end, exc)

            done_items = list(adaptive_result["done"])
            if done_items and defer_range_promotes and any(
                bool(rinfo.get("dedup_stage"))
                for _done_start, _done_end, rinfo in done_items
            ):
                deferred_promotions.append((done_items, reservation))
                reservation = None
                if len(deferred_promotions) >= promote_batch_size:
                    flush_deferred_promotions()
            else:
                for done_start, done_end, rinfo in done_items:
                    mark_done_range(done_start, done_end, rinfo)
        finally:
            if reservation is not None:
                reservation.release()

    run_bounded_future_queue(
        pending_ranges,
        max_pending=submit_window,
        submit=submit_commit_range,
        handle_done=handle_commit_range,
        stop_event=STOP_EVENT,
        cancel_pending_on_stop=True,
    )
    flush_deferred_promotions()

    # True iff EVERY range for this repo is now marked done in the manifest
    # (covers resume-skipped + newly-done; excludes cancelled/failed). Drives
    # temp + extract-cache retention in process_one_repo.
    complete_ranges = manifest_complete_commit_ranges(repo, manifest, range_size)
    all_ranges_done = complete_ranges is not None
    if complete_ranges is not None:
        mark_commit_stream_complete(
            repo,
            manifest,
            manifest_lock,
            complete_ranges,
        )
    return done, failed, all_ranges_done


# --------------------------------------------------------------------------- #
# Per-repo conveyor unit: ONE extraction, BOTH halves, then delete the repo.
# --------------------------------------------------------------------------- #
def process_one_repo(
    repo: str,
    repo_dir: Path,
    lengths_code: Sequence[int],
    lengths_commits: Sequence[int],
    range_size: int,
    range_target_bytes: int,
    work_root: Path,
    work_parent: Path,
    pool: ThreadPoolExecutor,
    manifest: Manifest,
    manifest_lock: threading.Lock,
    resume: bool,
    cumulative: dict,
    keep_temp: bool,
    dedup_db: Path | None,
    dedup_near: bool,
    pr_store: Path | None,
    repo_list: Path | None,
    streams: str = "both",
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    progress: ProgressWriter | None = None,
    checkpoint: DedupCheckpointController | None = None,
    *,
    code_memory_limit_gb: float | None = None,
    commit_memory_limit_gb: float | None = None,
    code_index_timeout_s: int | None = None,
    code_index_stall_timeout_s: int | None = None,
    code_recompressor: BackgroundRecompressor | None = None,
    reservations: UnitReservationLedger | None = None,
    range_submit_window: int = 1,
    analysis_cache_entries: int = 128,
    dedup_promote_batch_size: int = DEFAULT_DEDUP_PROMOTE_BATCH_SIZE,
    retain_partial_work: bool = False,
) -> dict:
    """Run BOTH halves for one already-extracted repo subtree, then delete it.

    The repo was extracted ONCE (incl .git) by the .git-preserving stream into
    ``repo_dir`` (== work_root/<repo>/_src). The CODE half runs first (it does
    NOT touch .git); then the COMMITS half consumes + deletes .git. The repo work
    dir (and the stable extract cache) are removed at the end. Fully-done repos
    are always cleaned. Interrupted / failed / partial repos are also cleaned by
    default so the conveyor cannot fill the disk with raw source clones; use
    --retain-partial-work for the older zero-rework resume mode. RULE #1: a
    failure in one half is recorded; the other half still runs.
    """
    repo_work = work_root / repo
    repo_work.mkdir(parents=True, exist_ok=True)
    result = {"repo": repo, "code": None, "commits_done": 0, "commits_failed": 0}
    all_ranges_done = streams not in {"both", "commits"}  # True when commits disabled
    code_limit = memory_limit_gb if code_memory_limit_gb is None else code_memory_limit_gb
    commit_limit = (
        memory_limit_gb if commit_memory_limit_gb is None else commit_memory_limit_gb
    )
    try:
        # ---- CODE half (skip if already done) ----
        if streams in {"both", "code"}:
            ck = code_key(repo)
            if not sr.is_code_worktree_repo(repo):
                info = {
                    "source": "code",
                    "repo": repo,
                    "skipped": True,
                    "reason": "bare git repository, not a source worktree",
                    "lengths": {},
                    "stage_timings_s": {},
                }
                with manifest_lock:
                    manifest.mark_done(ck, info)
                if progress is not None:
                    progress.emit(
                        "unit_skipped",
                        stream="code",
                        repo=repo,
                        unit=ck,
                        reason=info["reason"],
                    )
                result["code"] = "skipped_non_worktree"
            elif resume and manifest.is_done(ck):
                _log(f"SKIP (done) {ck}")
                result["code"] = "skipped"
            else:
                code_reservation: WorkReservation | None = None
                run_code_unit = True
                try:
                    if reservations is not None:
                        code_reservation = reservations.acquire(
                            ck,
                            stream="code",
                            repo=repo,
                        )
                        if not code_reservation.acquired:
                            _log(
                                f"SKIP (reserved) {ck}: holder="
                                f"{code_reservation.holder}"
                            )
                            result["code"] = "reserved"
                            code_reservation = None
                            run_code_unit = False
                    if run_code_unit:
                        code_parse_workers = parse_workers
                        code_dedup_db = dedup_db
                        code_dedup_near = dedup_near
                        if (
                            parse_workers > 1
                            and resume
                            and failed_code_unit_was_index_memory(
                                repo,
                                manifest,
                                parse_workers=parse_workers,
                            )
                        ):
                            code_parse_workers = 1
                            code_dedup_db = dedup_db
                            code_dedup_near = False
                            _log(
                                f"RETRY {ck}: prior manifest failure was "
                                "retryable index_project failure; starting parse_workers=1 "
                                "with global exact/chunk dedup and near-dedup disabled"
                            )
                        if progress is not None:
                            progress.emit(
                                "unit_started",
                                stream="code",
                                repo=repo,
                                unit=ck,
                                parse_workers=code_parse_workers,
                                memory_limit_gb=code_limit,
                                index_timeout_s=code_index_timeout_s,
                                index_stall_timeout_s=code_index_stall_timeout_s,
                            )
                        cinfo = run_code_half_adaptive(
                            repo, repo_dir, lengths_code, work_root,
                            code_dedup_db, code_dedup_near,
                            global_symbol_index, code_limit, code_parse_workers,
                            code_index_timeout_s,
                            code_index_stall_timeout_s,
                            recompressor=code_recompressor,
                        )
                        if cinfo.get("skipped"):
                            with manifest_lock:
                                manifest.mark_done(ck, cinfo)
                            if progress is not None:
                                progress.emit(
                                    "unit_skipped",
                                    stream="code",
                                    repo=repo,
                                    unit=ck,
                                    reason=cinfo.get("skip_reason", "skipped"),
                                    detail=str(cinfo.get("detail", ""))[:2000],
                                )
                            result["code"] = cinfo
                            _log(
                                f"SKIP {ck}: {cinfo.get('skip_reason', 'skipped')}"
                            )
                        else:
                            with manifest_lock:
                                manifest.mark_done(ck, cinfo)
                            totals = _length_totals(cinfo)
                            with manifest_lock:
                                cumulative["valid"] += totals["valid_tokens"]
                                cumulative_valid = cumulative["valid"]
                            if progress is not None:
                                progress.emit(
                                    "unit_done",
                                    stream="code",
                                    repo=repo,
                                    unit=ck,
                                    rows=totals["rows"],
                                    valid_tokens=totals["valid_tokens"],
                                    capacity_tokens=totals["capacity_tokens"],
                                    **_primary_bucket_progress(cinfo, lengths_code),
                                    lengths=cinfo.get("lengths", {}),
                                    stage_timings_s=cinfo.get("stage_timings_s", {}),
                                    cumulative_valid_tokens=cumulative_valid,
                                )
                            if checkpoint is not None:
                                checkpoint.maybe_checkpoint(cumulative_valid)
                            result["code"] = cinfo
                            _log(
                                f"DONE {ck}: buckets={sorted(cinfo['lengths'].keys())} "
                                f"(cum_all={cumulative['valid']})"
                            )
                except RepoFailure as exc:
                    _log(f"FAIL {ck}: {exc}")
                    with manifest_lock:
                        manifest.mark_failed(ck, exc.stage, exc.detail)
                    if progress is not None:
                        progress.emit(
                            "unit_failed",
                            stream="code",
                            repo=repo,
                            unit=ck,
                            stage=exc.stage,
                            detail=exc.detail[:2000],
                    )
                    result["code"] = "failed"
                finally:
                    if code_reservation is not None:
                        code_reservation.release()
        else:
            result["code"] = "disabled"

        # ---- COMMITS half (per-range resume + checkpoint inside) ----
        if streams in {"both", "commits"}:
            try:
                done, failed, all_ranges_done = run_commits_half(
                    repo, repo_dir, repo_work, work_root, work_parent,
                    lengths_commits, range_size,
                    pool, manifest, manifest_lock, resume, cumulative,
                    dedup_db, dedup_near, pr_store, repo_list, commit_limit,
                    progress, checkpoint, reservations, range_submit_window,
                    analysis_cache_entries,
                    range_target_bytes=range_target_bytes,
                    dedup_promote_batch_size=dedup_promote_batch_size,
                )
                result["commits_done"] = done
                result["commits_failed"] = failed
            except RepoFailure as exc:
                _log(f"FAIL {repo}::commits: {exc}")
                all_ranges_done = False
                with manifest_lock:
                    manifest.mark_failed(f"{repo}::commits", exc.stage, exc.detail)
        return result
    finally:
        # Production default is disk-bounded: keep completed parquet + manifest,
        # but delete raw source / JSONL / stage scratch even when a repo is only
        # partially done. Resume re-extracts unfinished units instead of pinning
        # hundreds of GiB in outputs/conveyor/tmp. The old zero-rework behavior is
        # still explicit via --retain-partial-work or --keep-temp.
        fully_done = repo_fully_done(repo, manifest, streams, all_ranges_done)
        if keep_temp:
            pass
        elif fully_done:
            remove_tree(repo_work, reason=f"{repo} fully done work")
            remove_tree(extract_cache_dir(repo), reason=f"{repo} fully done extract cache")
        elif retain_partial_work:
            _log(f"RETAIN temp for {repo}: not all units marked done "
                 f"(interrupted/failed/partial); kept {repo_work} + extract cache "
                 f"{extract_cache_dir(repo)} for zero-rework resume")
        elif STOP_EVENT.is_set():
            remove_tree(repo_work, reason=f"{repo} interrupted partial work")
            _log(f"RETAIN extract cache for interrupted {repo}: "
                 f"{extract_cache_dir(repo)} kept for checkpoint resume")
        else:
            remove_tree(repo_work, reason=f"{repo} partial work")
            remove_tree(extract_cache_dir(repo), reason=f"{repo} partial extract cache")
            _log(f"CLEANUP partial for {repo}: unfinished units remain unmarked; "
                 "resume will re-extract/re-run only unfinished work from manifest")


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target-lengths-code", default="1024,2048,4096",
                   help="Route-by-fit ladder for CODE (default 1024,2048,4096).")
    p.add_argument("--target-lengths-commits", default="1024,2048,4096,8192,16384",
                   help="Route-by-fit ladder for COMMITS "
                        "(default 1024,2048,4096,8192,16384).")
    p.add_argument("--range-size", type=int, default=DEFAULT_RANGE_SIZE,
                   help="Commits per checkpointed range (default 500).")
    p.add_argument("--range-target-bytes", type=int,
                   default=DEFAULT_RANGE_TARGET_BYTES,
                   help="Soft cap for raw extracted commit-record JSONL bytes per "
                        "range. The conveyor still respects --range-size as a hard "
                        "record-count cap. Use 0 to restore fixed --range-size "
                        f"ranges. Default {DEFAULT_RANGE_TARGET_BYTES}.")
    p.add_argument("--range-submit-window", type=int, default=0,
                   help="Maximum outstanding commit range futures per repo. "
                        "Default 0 means 2 * --workers. Bounds reservation "
                        "claims and ThreadPoolExecutor queue size for huge repos.")
    p.add_argument("--dedup-promote-batch-size", type=int,
                   default=DEFAULT_DEDUP_PROMOTE_BATCH_SIZE,
                   help="Commit ranges defer local dedup stage promotion and the "
                        "parent promotes this many completed stages while holding "
                        "the global SQLite writer lock once. Use 1 to restore "
                        "per-range promotion. "
                        f"Default {DEFAULT_DEDUP_PROMOTE_BATCH_SIZE}.")
    p.add_argument("--analysis-cache-entries", type=int, default=128,
                   help="Bounded per-process LRU entries passed to "
                        "process_commits.py for repeated old/new file analyses. "
                        "Default 128; use 0 to disable.")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="ThreadPoolExecutor size for commit ranges "
                        "(default = os.cpu_count()).")
    p.add_argument("--repo-workers", type=int, default=1,
                   help="Number of repos to process concurrently. Default 1 "
                        "preserves the legacy one-repo-at-a-time conveyor.")
    p.add_argument("--max-active-repos", type=int, default=None,
                   help="Bound staged, not-yet-finished repos on disk. Default "
                        "= --repo-workers. Use 20-30 for a wide streaming window "
                        "when disk has room.")
    p.add_argument("--streams", choices=("both", "code", "commits"), default="both",
                   help="Which streams to emit. 'code' uses the source-only tar "
                        "stream and does not extract .git / run PR/commit stages.")
    p.add_argument("--max-repos", type=int, default=None,
                   help="Process at most N repos this run (after resume filtering).")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Stop after cumulative valid tokens (code + all commit "
                        "buckets) reaches this.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--retain-partial-work", action="store_true",
                   help="Retain raw source clones, range scratch and extracted "
                        "commit JSONL for interrupted/failed/partial repos. "
                        "Default is OFF: completed parquet + manifest are kept, "
                        "but intermediates are deleted so disk stays bounded.")
    p.add_argument("--min-free-disk-gb", type=float,
                   default=DEFAULT_MIN_FREE_DISK_GB,
                   help="Fail loud before staging a new repo when the filesystem "
                        "under --work-parent-dir has less free space than this. "
                        "Use 0 to disable. "
                        f"Default {DEFAULT_MIN_FREE_DISK_GB:.0f} GiB.")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--work-parent-dir", default=str(DEFAULT_WORK_PARENT),
                   help="Parent directory for conveyor temporary work dirs when "
                        "--work-dir is not set. Default lives under outputs/ on "
                        "the external corpus disk, not the macOS /var/folders "
                        f"temp volume: {DEFAULT_WORK_PARENT}.")
    p.add_argument("--code-output-root", default=str(sr.OUTPUT_ROOT),
                   help="Output root for packed CODE parquet buckets. "
                        f"Default {sr.OUTPUT_ROOT}.")
    p.add_argument("--commit-output-root", default=str(src.COMMIT_OUTPUT_ROOT),
                   help="Output root for packed COMMIT parquet buckets. "
                        f"Default {src.COMMIT_OUTPUT_ROOT}.")
    p.add_argument("--conveyor-root", default=str(CONVEYOR_ROOT),
                   help="Root for conveyor manifest, locks, extract cache, "
                        f"progress and tmp defaults. Default {CONVEYOR_ROOT}.")
    p.add_argument("--only-repo", action="append", default=[],
                   help="Restrict the conveyor to these repo names (repeatable). "
                        "Other repos are DRAINED from the .git-preserving stream "
                        "without touching disk. Useful for targeted (re)runs.")
    p.add_argument("--dedup-db", default=str(DEFAULT_DEDUP_DB),
                   help="SHARED global dedup SQLite store threaded into BOTH the "
                        "code stage (function-level tokenized-hash) AND every "
                        "commit range stage (commit-doc tokenized-hash). "
                        f"Default {DEFAULT_DEDUP_DB}.")
    p.add_argument("--no-near-dedup", action="store_true",
                   help="Disable MinHash-LSH near dedup (exact-only) for both.")
    p.add_argument("--pr-store", default=str(DEFAULT_PR_STORE),
                   help="Tier-2 PR-discussion SQLite store passed to "
                        "process_commits; on a hit the rendered PR discussion is "
                        "injected as record['pr_discussion'] (HEAD of the commit "
                        f"doc). Default {DEFAULT_PR_STORE}.")
    p.add_argument("--repo-list", default=str(DEFAULT_REPO_LIST),
                   help="repo_list.json (bare-name -> owner/repo) for resolving "
                        f"the PR-store key. Default {DEFAULT_REPO_LIST}.")
    p.add_argument("--global-symbol-index", default=None,
                   help="Path to the GLOBAL cross-repo base-lib symbol SQLite store "
                        "(built by scripts/crossrepo/build_global_symbol_index.py). "
                        "When set, the CODE half threads it into index_project so "
                        "unresolved base-lib callees are pulled as bounded depth-1 "
                        "deps tagged crosslib:<repo>. DEFAULT off (unchanged).")
    p.add_argument("--memory-limit-gb", type=float, default=10.0,
                   help="Per-stage fail-loud RSS limit passed to index/materialize/"
                        "commit processors (default 10.0).")
    p.add_argument("--code-memory-limit-gb", type=float, default=None,
                   help="CODE-stage RSS limit passed to index_project/materialize/"
                        "pack. Default: --memory-limit-gb.")
    p.add_argument("--commit-memory-limit-gb", type=float, default=None,
                   help="COMMITS-stage RSS limit passed to process_commits/"
                        "materialize/pack. Default: --memory-limit-gb. Use this "
                        "to raise --workers without over-reserving for code.")
    p.add_argument("--parse-workers", type=int, default=2,
                   help="Parse workers passed to index_project for the CODE half "
                        "(default 2; avoid multiplying --repo-workers by the old "
                        "index_project default of 8 clang workers).")
    p.add_argument("--code-index-timeout-s", type=int, default=0,
                   help="Optional fail-loud timeout for each CODE index_project "
                        "stage. 0 disables the timeout (default).")
    p.add_argument("--code-index-stall-timeout-s", type=int, default=0,
                   help="Optional fail-loud CODE index_project stall watchdog: "
                        "kill when the index log has no size/mtime progress for "
                        "this many seconds. 0 disables the watchdog (default).")
    p.add_argument("--background-code-recompress", action="store_true",
                   help="Defer code parquet zstd-max recompress to a background "
                        "pool so repo processing can continue after valid parquet "
                        "has been written. The pool is drained before exit.")
    p.add_argument("--code-recompress-workers", type=int, default=2,
                   help="Background code parquet recompress workers when "
                        "--background-code-recompress is set (default 2).")
    p.add_argument("--source-cache-dir", default=None,
                   help="Optional repo-level source cache for --streams code. "
                        "Complete cached repos are processed before opening the "
                        "source tarball; uncached repos are materialized into the "
                        "cache with a completion sentinel.")
    p.add_argument("--source-cache-only", action="store_true",
                   help="For --streams code, only process complete repos already "
                        "in --source-cache-dir; do not open/decompress the source "
                        "tarball.")
    p.add_argument("--source-cache-populate-only", action="store_true",
                   help="For --streams code, populate --source-cache-dir from "
                        "the source tarball and exit without indexing/tokenizing. "
                        "Follow with --source-cache-only for hot code runs.")
    p.add_argument("--source-dir-root", action="append", default=[],
                   help="For --streams code, process already-extracted repo dirs "
                        "directly without opening the source tarball. May be passed "
                        "multiple times; children named <repo>/_src or <repo> are "
                        "treated as repo dirs.")
    p.add_argument("--memory-budget-gb", type=float, default=None,
                   help="Global conveyor memory budget for heavy subprocesses. "
                        "Default is 55%% of physical RAM (or 48 GiB if RAM size "
                        "cannot be detected). Use 0 to disable the preflight.")
    p.add_argument("--allow-memory-oversubscription", action="store_true",
                   help="Allow heavy_slots * --memory-limit-gb to exceed "
                        "--memory-budget-gb. This is intentionally explicit.")
    p.add_argument("--progress-jsonl", default=str(DEFAULT_PROGRESS_JSONL),
                   help="Append unit-level progress events here for live "
                        "throughput monitoring. Use empty string to disable.")
    p.add_argument("--dedup-checkpoint-tokens", type=int,
                   default=DEFAULT_DEDUP_CHECKPOINT_TOKENS,
                   help="Run a SQLite WAL checkpoint on --dedup-db after each "
                        "N valid tokens produced by this conveyor run. Use 0 "
                        "to disable. Default 25,000,000.")
    p.add_argument("--dedup-checkpoint-mode",
                   choices=("PASSIVE", "FULL", "RESTART", "TRUNCATE"),
                   default="TRUNCATE",
                   help="SQLite wal_checkpoint mode for token milestones. "
                        "Default TRUNCATE caps WAL growth when no writer holds it.")
    p.add_argument("--dedup-checkpoint-busy-timeout-ms", type=int, default=5000,
                   help="Busy timeout for token-milestone dedup WAL checkpoints.")
    p.add_argument("--run-lock-dir", default=str(DEFAULT_RUN_LOCK_DIR),
                   help="Directory for fail-loud per-stream conveyor locks. "
                        f"Default {DEFAULT_RUN_LOCK_DIR}.")
    p.add_argument("--no-run-lock", action="store_true",
                   help="Disable per-stream conveyor process locks. Intended "
                        "only for controlled tests; normal runs should keep the "
                        "lock so duplicate stream writers fail before extraction.")
    p.add_argument("--reservation-file", default=str(DEFAULT_RESERVATION_FILE),
                   help="Cross-process active-unit reservation JSON file. Prevents "
                        "duplicate repo/range workers from processing the same "
                        f"output unit. Default {DEFAULT_RESERVATION_FILE}.")
    p.add_argument("--no-reservation-ledger", action="store_true",
                   help="Disable active-unit reservations. Intended only for "
                        "controlled tests.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    configure_runtime_paths_from_args(args)
    lengths_code = tuple(sorted(
        int(x) for x in args.target_lengths_code.split(",") if x.strip()))
    lengths_commits = tuple(sorted(
        int(x) for x in args.target_lengths_commits.split(",") if x.strip()))
    if not lengths_code:
        raise SystemExit("--target-lengths-code produced no lengths")
    if not lengths_commits:
        raise SystemExit("--target-lengths-commits produced no lengths")
    workers = max(1, int(args.workers or 1))
    repo_workers = max(1, int(args.repo_workers or 1))
    max_active_repos = int(args.max_active_repos or repo_workers)
    max_active_repos = max(1, max_active_repos)
    if max_active_repos < repo_workers:
        raise SystemExit("--max-active-repos must be >= --repo-workers")
    parse_workers = max(1, int(args.parse_workers or 1))
    args.parse_workers = parse_workers
    code_index_stall_timeout_s = int(args.code_index_stall_timeout_s or 0)
    source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
    source_dir_roots = [Path(p) for p in args.source_dir_root]
    if args.source_cache_only and source_cache_dir is None:
        raise SystemExit("--source-cache-only requires --source-cache-dir")
    if args.source_cache_populate_only and source_cache_dir is None:
        raise SystemExit("--source-cache-populate-only requires --source-cache-dir")
    if args.source_cache_populate_only and args.source_cache_only:
        raise SystemExit("--source-cache-populate-only cannot be combined with --source-cache-only")
    if args.source_cache_populate_only and args.streams != "code":
        raise SystemExit("--source-cache-populate-only is only valid with --streams code")
    if args.source_cache_only and args.streams != "code":
        raise SystemExit("--source-cache-only is only valid with --streams code")
    if source_cache_dir is not None and args.streams != "code":
        raise SystemExit("--source-cache-dir is currently only supported with --streams code")
    if source_dir_roots and args.streams != "code":
        raise SystemExit("--source-dir-root is currently only supported with --streams code")
    if source_dir_roots and (source_cache_dir is not None or args.source_cache_only):
        raise SystemExit("--source-dir-root cannot be combined with source cache flags")
    range_submit_window = (
        max(1, workers * DEFAULT_RANGE_SUBMIT_WINDOW_MULTIPLIER)
        if int(args.range_submit_window or 0) <= 0
        else max(1, int(args.range_submit_window))
    )
    dedup_promote_batch_size = max(1, int(args.dedup_promote_batch_size or 1))
    code_memory_limit_gb = (
        float(args.memory_limit_gb)
        if args.code_memory_limit_gb is None
        else float(args.code_memory_limit_gb)
    )
    commit_memory_limit_gb = (
        float(args.memory_limit_gb)
        if args.commit_memory_limit_gb is None
        else float(args.commit_memory_limit_gb)
    )
    memory_budget_gb = (
        default_memory_budget_gb()
        if args.memory_budget_gb is None
        else float(args.memory_budget_gb)
    )
    memory_plan = validate_memory_plan(
        streams=args.streams,
        workers=workers,
        repo_workers=repo_workers,
        memory_limit_gb=args.memory_limit_gb,
        code_memory_limit_gb=code_memory_limit_gb,
        commit_memory_limit_gb=commit_memory_limit_gb,
        memory_budget_gb=memory_budget_gb,
        allow_oversubscription=args.allow_memory_oversubscription,
    )

    # SIGNAL HANDLER (requirement 1): cooperative, fail-loud shutdown. The first
    # SIGINT/SIGTERM sets STOP_EVENT -> the submission loops stop staging new
    # repos and run_commits_half stops/cancels new ranges, while in-flight
    # subprocess tasks drain and only COMPLETED units are marked done (the
    # manifest is already persisted atomically per mark, so nothing partial can be
    # reported as done). In-progress repo temp + extract caches are RETAINED. A
    # SECOND signal forces an immediate os._exit (skips all cleanup, so temp is
    # preserved). Installed in the main thread before any streaming begins.
    def _on_signal(signum, _frame):
        name = signal.Signals(signum).name
        if STOP_EVENT.is_set():
            _log(f"Signal {name} again: FORCING immediate exit (130). Completed "
                 "units already persisted; in-progress temp may be retained.")
            os._exit(130)
        STOP_EVENT.set()
        _log(f"Signal {name} received: CHECKPOINTING -- no new repos/ranges "
             "submitted; draining in-flight tasks; manifest is atomic. Send the "
             "signal again to force-exit.")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    run_locks: list[RunLock] = []
    if not args.no_run_lock:
        lock_dir = Path(args.run_lock_dir)
        try:
            for name in stream_lock_names(args.streams):
                lock = RunLock(lock_dir / f"{name}.lock")
                lock.acquire()
                run_locks.append(lock)
            _log(
                "Run-lock: acquired "
                + ", ".join(str(lock.path) for lock in run_locks)
            )
        except Exception:
            for lock in run_locks:
                lock.close()
            raise

    # FAIL LOUD up front (RULE #1): every required stage binary must exist.
    required_paths = [VENV_PYTHON, sr.TOKENIZER_PATH, sr.MATERIALIZER, sr.PACKER]
    if args.streams in {"both", "code"}:
        required_paths.append(sr.INDEX_PROJECT)
    if args.streams in {"both", "commits"}:
        required_paths.extend([sr.PROCESS_COMMITS, EXTRACT_GIT])
    for path in required_paths:
        if not Path(path).exists():
            raise SystemExit(f"required path missing: {path}")

    # Pre-create output trees for BOTH streams.
    CONVEYOR_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    sr.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for L in lengths_code:
        (sr.OUTPUT_ROOT / str(L)).mkdir(parents=True, exist_ok=True)
    src.COMMIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for L in lengths_commits:
        (src.COMMIT_OUTPUT_ROOT / str(L)).mkdir(parents=True, exist_ok=True)

    # SHARED dedup db for BOTH streams. FAIL LOUD: open once now so a bad path /
    # missing datasketch crashes before any work (RULE #1).
    dedup_db = Path(args.dedup_db) if args.dedup_db else None
    dedup_near = not args.no_near_dedup
    if dedup_db is not None:
        sys.path.insert(0, str(MLX_ROOT / "tools" / "clang_indexer"))
        from dedup_store import DedupStore  # noqa: E402
        dedup_db.parent.mkdir(parents=True, exist_ok=True)
        # Path/schema validation only. Do not rebuild the persisted MinHash/LSH
        # index in the driver parent; real worker stages open near-dedup only
        # when requested. On full corpora the LSH reload dominates startup.
        DedupStore(str(dedup_db), near=False, commit_every=1000).close()
        _log(f"Dedup: SHARED global store at {dedup_db} threaded into BOTH "
             f"code + commit stages (exact{'+near' if dedup_near else ''}, "
             f"tokenized hash)")

    # PR-discussion live lookup (fail-loud on bad paths up front).
    pr_store = Path(args.pr_store) if args.pr_store else None
    repo_list = Path(args.repo_list) if args.repo_list else None
    if pr_store is not None and not pr_store.exists():
        raise SystemExit(f"--pr-store does not exist: {pr_store}")
    if repo_list is not None and not repo_list.exists():
        raise SystemExit(f"--repo-list does not exist: {repo_list}")
    if pr_store is not None and args.streams in {"both", "commits"}:
        _log(f"PR-store: inject record['pr_discussion'] from {pr_store} "
             f"(repo_list={repo_list})")

    # Optional cross-repo base-lib symbol index. FAIL LOUD if given but missing.
    global_symbol_index = (
        Path(args.global_symbol_index) if args.global_symbol_index else None
    )
    if global_symbol_index is not None:
        if not global_symbol_index.exists():
            raise SystemExit(f"--global-symbol-index not found: {global_symbol_index}")
        if args.streams in {"both", "code"}:
            _log(f"Cross-lib: GLOBAL base-lib symbol index at {global_symbol_index} "
                 f"threaded into CODE half (bounded depth-1 pulls, crosslib:<repo>).")

    # Cross-process-safe manifest: a concurrent --streams code run and
    # --streams commits run (each holding only its own per-stream RunLock) share
    # CONVEYOR_MANIFEST; ConcurrentManifest merges + flocks every write so they
    # cannot clobber each other's resume/accounting keys (H4 fix).
    manifest = ConcurrentManifest.load(CONVEYOR_MANIFEST)
    manifest_lock = threading.Lock()
    resume = not args.no_resume
    progress = ProgressWriter(Path(args.progress_jsonl) if args.progress_jsonl else None)
    code_recompressor = (
        BackgroundRecompressor(args.code_recompress_workers)
        if args.background_code_recompress and args.streams in {"both", "code"}
        else None
    )
    reservations = None
    if not args.no_reservation_ledger:
        reservations = UnitReservationLedger(Path(args.reservation_file))
        stale = reservations.cleanup_stale()
        if stale:
            _log(f"Reservations: cleaned {stale} stale active unit claim(s)")
    checkpoint = DedupCheckpointController(
        dedup_db=dedup_db,
        interval_tokens=args.dedup_checkpoint_tokens,
        mode=args.dedup_checkpoint_mode,
        busy_timeout_ms=args.dedup_checkpoint_busy_timeout_ms,
        progress=progress,
    )
    progress.emit(
        "run_started",
        streams=args.streams,
        workers=workers,
        repo_workers=repo_workers,
        max_active_repos=max_active_repos,
        range_size=args.range_size,
        range_target_bytes=args.range_target_bytes,
        range_submit_window=range_submit_window,
        dedup_promote_batch_size=dedup_promote_batch_size,
        target_lengths_code=list(lengths_code),
        target_lengths_commits=list(lengths_commits),
        manifest=str(CONVEYOR_MANIFEST),
        retain_partial_work=args.retain_partial_work,
        keep_temp=args.keep_temp,
        min_free_disk_gb=args.min_free_disk_gb,
        dedup_checkpoint_tokens=args.dedup_checkpoint_tokens,
        dedup_checkpoint_mode=args.dedup_checkpoint_mode,
        parse_workers=parse_workers,
        code_index_timeout_s=args.code_index_timeout_s,
        code_index_stall_timeout_s=code_index_stall_timeout_s,
        code_memory_limit_gb=code_memory_limit_gb,
        commit_memory_limit_gb=commit_memory_limit_gb,
        background_code_recompress=code_recompressor is not None,
        code_recompress_workers=args.code_recompress_workers if code_recompressor else 0,
        source_cache_dir=str(source_cache_dir) if source_cache_dir is not None else None,
        source_cache_only=args.source_cache_only,
        source_cache_populate_only=args.source_cache_populate_only,
        source_dir_roots=[str(p) for p in source_dir_roots],
        reservation_file=str(args.reservation_file)
        if reservations is not None else None,
        memory_plan=memory_plan,
    )
    _log(
        "Memory plan: "
        f"heavy_slots={memory_plan['heavy_slots']} "
        f"code_slots={memory_plan['code_heavy_slots']} "
        f"commit_slots={memory_plan['commit_heavy_slots']} "
        f"code_limit={memory_plan['code_memory_limit_gb']:.2f}GiB "
        f"commit_limit={memory_plan['commit_memory_limit_gb']:.2f}GiB "
        f"reserved={memory_plan['reserved_gb']:.2f}GiB "
        f"budget={memory_plan['memory_budget_gb']:.2f}GiB "
        f"parse_workers={parse_workers} "
        f"range_submit_window={range_submit_window} "
        f"dedup_promote_batch_size={dedup_promote_batch_size}"
    )

    # work_parent is always the configured parent: it both hosts the randomized
    # per-run work_root (when --work-dir is unset) AND is scanned by the extract
    # checkpoint to ADOPT a prior run's orphaned <repo>_commits.jsonl on resume.
    work_parent = Path(args.work_parent_dir)
    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        own_work_root = False
    else:
        work_parent.mkdir(parents=True, exist_ok=True)
        ensure_min_free_disk(
            work_parent,
            args.min_free_disk_gb,
            context="creating conveyor work root",
        )
        os.environ.setdefault("TMPDIR", str(work_parent))
        os.environ.setdefault("TMP", str(work_parent))
        os.environ.setdefault("TEMP", str(work_parent))
        work_root = Path(tempfile.mkdtemp(prefix="streaming_conveyor_", dir=str(work_parent)))
        own_work_root = True
    ensure_min_free_disk(
        work_parent,
        args.min_free_disk_gb,
        context="starting conveyor",
    )
    progress.emit("work_root_ready", work_root=str(work_root), own_work_root=own_work_root)

    cumulative = {"valid": 0}
    processed_repos = 0
    code_done = 0
    ranges_done = 0
    ranges_failed = 0
    submitted_repo_names: set[str] = set()

    only_repos = set(args.only_repo) if args.only_repo else None

    def should_process(repo: str) -> bool:
        if args.streams == "code" and not sr.is_code_worktree_repo(repo):
            return False
        # Restrict to --only-repo when given (others drained without extraction).
        # Otherwise skip extraction when the manifest itself proves every stream
        # for this repo is complete; partial/ambiguous repos still stage and use
        # the existing per-half/per-range resume below.
        should_stage = should_stage_repo_from_manifest(
            repo,
            streams=args.streams,
            resume=resume,
            manifest=manifest,
            range_size=args.range_size,
            only_repos=only_repos,
            manifest_lock=manifest_lock,
        )
        if should_stage:
            ensure_min_free_disk(
                work_parent,
                args.min_free_disk_gb,
                context=f"staging repo {repo}",
            )
        return should_stage

    if args.source_cache_populate_only:
        try:
            report = populate_code_source_cache(
                work_root,
                source_cache_dir,
                should_process,
                progress,
                max_repos=args.max_repos,
            )
            summary = {
                "repos_this_run": 0,
                "code_halves_this_run": 0,
                "commit_ranges_this_run": 0,
                "commit_ranges_failed_this_run": 0,
                "workers": workers,
                "repo_workers": repo_workers,
                "max_active_repos": max_active_repos,
                "parse_workers": parse_workers,
                "memory_plan": memory_plan,
                "streams": args.streams,
                "source_cache_populate_only": True,
                "source_cache_report": report,
                "manifest": str(CONVEYOR_MANIFEST),
                "interrupted": STOP_EVENT.is_set(),
            }
            progress.emit("run_finished", **summary)
            print(json.dumps(summary, indent=2))
            return 130 if STOP_EVENT.is_set() else 0
        finally:
            if code_recompressor is not None:
                code_recompressor.shutdown()
            if own_work_root and not args.keep_temp:
                remove_tree(work_root, reason="source-cache populate work_root")
            for lock in reversed(run_locks):
                lock.close()

    range_pool = ThreadPoolExecutor(max_workers=workers)
    repo_pool = ThreadPoolExecutor(max_workers=repo_workers) if repo_workers > 1 else None

    def _handle_repo_result(fut, repo: str) -> tuple[int, int, int, int]:
        """Return increments: processed_repos, code_done, ranges_done, ranges_failed."""
        try:
            res = fut.result()
        except RepoFailure as exc:
            _log(f"FAIL {repo}::repo: {exc}")
            with manifest_lock:
                manifest.mark_failed(f"{repo}::repo", exc.stage, exc.detail)
            return 1, 0, 0, 1
        except Exception as exc:  # surface unexpected worker failures loudly
            _log(f"FAIL {repo}::repo: unexpected {type(exc).__name__}: {exc}")
            with manifest_lock:
                manifest.mark_failed(f"{repo}::repo", "unexpected", str(exc))
            return 1, 0, 0, 1
        return (
            1,
            1 if isinstance(res.get("code"), dict) else 0,
            int(res.get("commits_done", 0)),
            int(res.get("commits_failed", 0)),
        )

    def claim_repo_once(repo: str) -> None:
        if repo in submitted_repo_names:
            raise RepoFailure(
                repo,
                "duplicate_repo",
                f"repo {repo!r} was yielded/submitted twice in one conveyor run",
            )
        submitted_repo_names.add(repo)

    def mark_no_git_repo(repo: str) -> None:
        info = {
            "source": "commits",
            "no_git": True,
            "reason": "missing .git metadata",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with manifest_lock:
            manifest.mark_done(no_git_key(repo), info)
        progress.emit("repo_no_git", repo=repo, unit=no_git_key(repo), **info)

    try:
        gen = (
            sr.stream_repo_dirs(source_dir_roots, should_process)
            if source_dir_roots
            else sr.stream_repo_subtrees(
                work_root,
                should_process,
                source_cache_dir=source_cache_dir,
                source_cache_only=args.source_cache_only,
            )
            if args.streams == "code"
            else stream_repo_subtrees_with_git(
                work_root,
                should_process,
                on_no_git=mark_no_git_repo,
            )
        )

        if repo_pool is None:
            for repo, repo_dir in gen:
                # Signal-driven stop: do not stage a NEW repo. The repo currently
                # in process_one_repo (if any) already drained its own ranges.
                if STOP_EVENT.is_set():
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                claim_repo_once(repo)
                res = process_one_repo(
                    repo, repo_dir, lengths_code, lengths_commits, args.range_size,
                    args.range_target_bytes,
                    work_root, work_parent, range_pool, manifest, manifest_lock,
                    resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    args.parse_workers,
                    progress, checkpoint,
                    code_memory_limit_gb=code_memory_limit_gb,
                    commit_memory_limit_gb=commit_memory_limit_gb,
                    code_index_timeout_s=args.code_index_timeout_s,
                    code_index_stall_timeout_s=code_index_stall_timeout_s,
                    code_recompressor=code_recompressor,
                    reservations=reservations,
                    range_submit_window=range_submit_window,
                    analysis_cache_entries=args.analysis_cache_entries,
                    dedup_promote_batch_size=dedup_promote_batch_size,
                    retain_partial_work=args.retain_partial_work,
                )
                processed_repos += 1
                if isinstance(res.get("code"), dict):
                    code_done += 1
                ranges_done += res.get("commits_done", 0)
                ranges_failed += res.get("commits_failed", 0)

                stop = STOP_EVENT.is_set()
                if args.max_repos is not None and processed_repos >= args.max_repos:
                    stop = True
                if args.token_budget is not None and cumulative["valid"] >= args.token_budget:
                    _log(f"Token budget {args.token_budget} reached.")
                    stop = True
                if stop:
                    if hasattr(gen, "close"):
                        gen.close()
                    break
        else:
            inflight: dict = {}
            submitted_repos = 0
            stop_submitting = False

            def drain_one_or_more(block: bool = True) -> None:
                nonlocal processed_repos, code_done, ranges_done, ranges_failed
                if not inflight:
                    return
                timeout = None if block else 0
                done, _pending = wait(
                    inflight.keys(),
                    timeout=timeout,
                    return_when=FIRST_COMPLETED,
                )
                for fut in done:
                    repo = inflight.pop(fut)
                    pr, cd, rd, rf = _handle_repo_result(fut, repo)
                    processed_repos += pr
                    code_done += cd
                    ranges_done += rd
                    ranges_failed += rf

            for repo, repo_dir in gen:
                # Signal-driven stop: stop submitting NEW repos; already-inflight
                # repos drain (and self-cancel their queued ranges) in the loop
                # below. Their un-finished units stay un-marked -> resume re-runs.
                if STOP_EVENT.is_set():
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                claim_repo_once(repo)
                while len(inflight) >= max_active_repos:
                    drain_one_or_more(block=True)
                    # Worker threads mutate cumulative['valid'] under manifest_lock
                    # (see process_one_repo / run_commits_half); snapshot under the
                    # same lock before the budget comparison so this drain-path read
                    # is not a concurrent unlocked read while writers are in flight.
                    with manifest_lock:
                        cumulative_valid_snapshot = cumulative["valid"]
                    if args.token_budget is not None and cumulative_valid_snapshot >= args.token_budget:
                        _log(f"Token budget {args.token_budget} reached.")
                        stop_submitting = True
                        break
                    if STOP_EVENT.is_set():
                        stop_submitting = True
                        break
                if stop_submitting or STOP_EVENT.is_set():
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                if args.max_repos is not None and submitted_repos >= args.max_repos:
                    if hasattr(gen, "close"):
                        gen.close()
                    break
                fut = repo_pool.submit(
                    process_one_repo,
                    repo, repo_dir, lengths_code, lengths_commits, args.range_size,
                    args.range_target_bytes,
                    work_root, work_parent, range_pool, manifest, manifest_lock,
                    resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    args.parse_workers,
                    progress, checkpoint,
                    code_memory_limit_gb=code_memory_limit_gb,
                    commit_memory_limit_gb=commit_memory_limit_gb,
                    code_index_timeout_s=args.code_index_timeout_s,
                    code_index_stall_timeout_s=code_index_stall_timeout_s,
                    code_recompressor=code_recompressor,
                    reservations=reservations,
                    range_submit_window=range_submit_window,
                    analysis_cache_entries=args.analysis_cache_entries,
                    dedup_promote_batch_size=dedup_promote_batch_size,
                    retain_partial_work=args.retain_partial_work,
                )
                inflight[fut] = repo
                submitted_repos += 1
                drain_one_or_more(block=False)

            while inflight:
                drain_one_or_more(block=True)
    finally:
        recompress_error: Exception | None = None
        if repo_pool is not None:
            repo_pool.shutdown(wait=True)
        range_pool.shutdown(wait=True)
        if code_recompressor is not None:
            try:
                code_recompressor.shutdown()
            except Exception as exc:
                recompress_error = exc
        interrupted = STOP_EVENT.is_set()
        # Reclaim the randomized work_root whenever production cleanup is active.
        # --retain-partial-work keeps the older zero-rework signal/debug mode.
        if own_work_root and not args.keep_temp and not interrupted:
            remove_tree(work_root, reason="clean run work_root")
        elif own_work_root and not args.keep_temp and interrupted and not args.retain_partial_work:
            remove_tree(work_root, reason="interrupted run work_root")
        for lock in reversed(run_locks):
            lock.close()
        if recompress_error is not None:
            raise recompress_error

    # ----- cumulative per-length report from the manifest (both streams) -----
    def _empty_totals(lengths):
        return {str(L): {"rows": 0, "valid_tokens": 0, "pad_tokens": 0,
                         "capacity_tokens": 0} for L in lengths}

    code_totals = _empty_totals(lengths_code)
    commit_totals = _empty_totals(lengths_commits)
    for key, info in manifest.done.items():
        is_code = key.endswith("::code")
        totals = code_totals if is_code else commit_totals
        for tl_s, st in info.get("lengths", {}).items():
            if tl_s not in totals:
                continue
            agg = totals[tl_s]
            agg["rows"] += st["rows"]
            agg["valid_tokens"] += st["valid_tokens"]
            agg["pad_tokens"] += st["pad_tokens"]
            agg["capacity_tokens"] += st["capacity_tokens"]

    summary = {
        "repos_this_run": processed_repos,
        "code_halves_this_run": code_done,
        "commit_ranges_this_run": ranges_done,
        "commit_ranges_failed_this_run": ranges_failed,
        "workers": workers,
        "repo_workers": repo_workers,
        "max_active_repos": max_active_repos,
        "parse_workers": parse_workers,
        "memory_plan": memory_plan,
        "streams": args.streams,
        "range_size": args.range_size,
        "target_lengths_code": list(lengths_code),
        "target_lengths_commits": list(lengths_commits),
        "cumulative_valid_tokens_this_run": cumulative["valid"],
        "total_done_units": len(manifest.done),
        "total_failed_units": len(manifest.failed),
        "code_per_length_totals": code_totals,
        "commit_per_length_totals": commit_totals,
        "dedup_db": str(dedup_db) if dedup_db else None,
        "pr_store": str(pr_store) if pr_store else None,
        "manifest": str(CONVEYOR_MANIFEST),
        "extract_cache_metrics": progress.extract_cache_metrics(),
        "interrupted": interrupted,
    }
    progress.emit("run_finished", **summary)
    print(json.dumps(summary, indent=2))
    if interrupted:
        _log("CHECKPOINTED on signal: every completed unit is recorded in the "
             f"manifest ({CONVEYOR_MANIFEST}); in-progress repo temp + the "
             "persistent extract cache were retained. RESUME WITH THE SAME ARGS "
             "to continue exactly where this left off (zero work lost).")
        return 130
    return 0 if not manifest.failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
