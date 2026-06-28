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
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Sequence

# Reuse the proven machinery. streaming_reindex_commits already re-exports the
# shared primitives from streaming_reindex (MLX_ROOT, Manifest, RepoFailure, ...)
# and adds the .git-preserving stream + range pipeline. Import BOTH, do not
# duplicate a single line of stage logic.
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

# Stable, repo-keyed cache for the EXPENSIVE git-history extraction output. This
# deliberately lives OUTSIDE the randomized per-run work_root so the commit
# records (e.g. php-src's ~6h / 10GB jsonl) survive a kill/restart with the SAME
# args -- the new run gets a fresh random work_root, so a per-run jsonl would be
# orphaned and re-extracted from scratch. The cache holds <repo>_commits.jsonl
# plus a <repo>_commits.jsonl.done sentinel (line/size/mtime), and is only
# deleted once EVERY unit of the repo (code + all ranges) is marked done.
EXTRACT_CACHE_ROOT = CONVEYOR_ROOT / "extract_cache"

_PRINT_LOCK = threading.Lock()

# Cooperative shutdown flag set by the SIGINT/SIGTERM handlers in main(). When
# set, the conveyor stops SUBMITTING new repos/ranges, lets in-flight subprocess
# tasks drain, and cancels queued-but-unstarted range futures (their units stay
# un-marked so resume re-runs them). A SECOND signal forces an immediate exit.
STOP_EVENT = threading.Event()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, file=sys.stderr, flush=True)


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
) -> tuple[Path, int]:
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
            return cache_jsonl, lc

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
            return cache_jsonl, n

        if repo_has_range_done:
            # Manifest proves the extract finished, but no jsonl survived anywhere.
            # extract_git_history is deterministic (git log order), so a fresh
            # re-extract reproduces the SAME record order and already-done ranges
            # still resume-skip. Loud, not silent.
            _log(f"EXTRACT-CKPT MISS {repo}: manifest has done ranges but NO jsonl "
                 f"found on disk; re-extracting (deterministic order preserves "
                 f"done-range alignment)")

    # (c) Fresh extract into the stable cache. FAIL LOUD on empty git log / output.
    commit_list = get_commit_list(repo_dir)
    if not commit_list:
        raise RepoFailure(repo, "git_log", "no --no-merges --diff-filter=M commits")
    cache_dir.mkdir(parents=True, exist_ok=True)
    records_jsonl = stage_extract_commits(repo, repo_dir, cache_dir)
    n = _count_jsonl_lines(records_jsonl)
    if n == 0:
        raise RepoFailure(repo, "extract_git_history",
                          f"zero records after extract for {repo}")
    _write_extract_sentinel(records_jsonl, n)
    _log(f"EXTRACT-CKPT FRESH {repo}: extracted {n} records -> {records_jsonl}; "
         f"stamped sentinel")
    return records_jsonl, n


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
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload) -> None:
        if self.path is None:
            return
        now = time.time()
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(now - self.started_at, 3),
            "event": event,
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self.lock:
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


def code_key(repo: str) -> str:
    """Checkpoint key for one repo's CODE half."""
    return f"{repo}::code"


def stream_lock_names(streams: str) -> tuple[str, ...]:
    if streams == "both":
        return ("code", "commits")
    return (streams,)


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
) -> dict:
    """index+route+pack the repo's source via the EXISTING code stage, zstd-max.

    Calls streaming_reindex.process_one_repo verbatim (it threads the SHARED
    dedup_db into index_project). Then recompresses each produced
    outputs/reindexed/<L>/<repo>.parquet with MAX zstd (the code stage writes
    plain parquet; the commit stage already recompresses, so we make code match).
    RAISES RepoFailure on any stage failure (no fallback).
    """
    info = sr.process_one_repo(
        repo, repo_dir, lengths_code, work_root, dedup_db, dedup_near,
        global_symbol_index, memory_limit_gb,
    )
    # zstd-max the per-length code parquet files this repo just wrote.
    for L in info.get("lengths", {}):
        dest = sr.OUTPUT_ROOT / str(L) / f"{repo}.parquet"
        if dest.exists():
            src.recompress_zstd_max(dest)
    return info


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

    # Reuse a cached/adopted extract when available; never re-run the ~6h extract
    # for a repo whose commit extraction already completed (sentinel or manifest).
    records_jsonl, n_records = ensure_commit_records(
        repo, repo_dir, work_root, work_parent, manifest, resume,
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
    ranges = [(s, min(s + range_size, n_records))
              for s in range(0, n_records, range_size)]

    futures = {}
    for (start, end) in ranges:
        if STOP_EVENT.is_set():
            _log(f"STOP: not submitting further ranges for {repo} (from r{start})")
            break
        rkey = range_key(repo, start)
        if resume and manifest.is_done(rkey):
            _log(f"SKIP (done) {rkey}")
            continue
        fut = pool.submit(
            process_range, repo, repo_dir, records_jsonl,
            start, end, lengths_sorted, repo_work,
            dedup_db, dedup_near, pr_store, repo_list,
            memory_limit_gb,
        )
        futures[fut] = (start, end)

    done = 0
    failed = 0
    cancelled_pending = False
    for fut in as_completed(futures):
        if STOP_EVENT.is_set() and not cancelled_pending:
            # Cancel queued-but-unstarted ranges; they stay un-marked so resume
            # re-runs them. Running ranges cannot be cancelled and drain below.
            n_cancelled = sum(1 for f in futures if not f.done() and f.cancel())
            cancelled_pending = True
            if n_cancelled:
                _log(f"STOP: cancelled {n_cancelled} queued range(s) for {repo}; "
                     f"draining in-flight")
        start, end = futures[fut]
        rkey = range_key(repo, start)
        try:
            rinfo = fut.result()
        except CancelledError:
            # Never started -> leave un-marked so resume re-runs this range.
            continue
        except RepoFailure as exc:
            _log(f"FAIL {rkey}: {exc}")
            failed += 1
            with manifest_lock:
                manifest.mark_failed(rkey, exc.stage, exc.detail)
            if progress is not None:
                progress.emit(
                    "unit_failed",
                    stream="commits",
                    repo=repo,
                    unit=rkey,
                    range=[start, end],
                    stage=exc.stage,
                    detail=exc.detail[:2000],
                )
            continue
        except Exception as exc:  # surface unexpected failures loud
            _log(f"FAIL {rkey}: unexpected {type(exc).__name__}: {exc}")
            failed += 1
            with manifest_lock:
                manifest.mark_failed(rkey, "unexpected", str(exc))
            if progress is not None:
                progress.emit(
                    "unit_failed",
                    stream="commits",
                    repo=repo,
                    unit=rkey,
                    range=[start, end],
                    stage="unexpected",
                    detail=str(exc)[:2000],
                )
            continue
        else:
            with manifest_lock:
                manifest.mark_done(rkey, rinfo)
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
                    unit=rkey,
                    range=[start, end],
                    rows=totals["rows"],
                    valid_tokens=totals["valid_tokens"],
                    capacity_tokens=totals["capacity_tokens"],
                    lengths=rinfo.get("lengths", {}),
                    cumulative_valid_tokens=cumulative_valid,
                )
            if checkpoint is not None:
                checkpoint.maybe_checkpoint(cumulative_valid)
            _log(f"DONE {rkey}: ranges [{start}:{end}] "
                 f"buckets={sorted(rinfo['lengths'].keys())} "
                 f"(+{added} @ {smallest}, cum_all={cumulative['valid']})")

    # True iff EVERY range for this repo is now marked done in the manifest
    # (covers resume-skipped + newly-done; excludes cancelled/failed). Drives
    # temp + extract-cache retention in process_one_repo.
    all_ranges_done = all(manifest.is_done(range_key(repo, s)) for (s, _e) in ranges)
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
    progress: ProgressWriter | None = None,
    checkpoint: DedupCheckpointController | None = None,
) -> dict:
    """Run BOTH halves for one already-extracted repo subtree, then delete it.

    The repo was extracted ONCE (incl .git) by the .git-preserving stream into
    ``repo_dir`` (== work_root/<repo>/_src). The CODE half runs first (it does
    NOT touch .git); then the COMMITS half consumes + deletes .git. The repo work
    dir (and the stable extract cache) are removed at the end ONLY when EVERY unit
    of the repo is marked done; an interrupted / failed / partial repo RETAINS its
    temp so resume loses zero work. RULE #1: a failure in one half is recorded;
    the other half still runs.
    """
    repo_work = work_root / repo
    repo_work.mkdir(parents=True, exist_ok=True)
    result = {"repo": repo, "code": None, "commits_done": 0, "commits_failed": 0}
    all_ranges_done = streams not in {"both", "commits"}  # True when commits disabled
    try:
        # ---- CODE half (skip if already done) ----
        if streams in {"both", "code"}:
            ck = code_key(repo)
            if resume and manifest.is_done(ck):
                _log(f"SKIP (done) {ck}")
                result["code"] = "skipped"
            else:
                try:
                    cinfo = run_code_half(
                        repo, repo_dir, lengths_code, work_root, dedup_db, dedup_near,
                        global_symbol_index, memory_limit_gb,
                    )
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
                            lengths=cinfo.get("lengths", {}),
                            cumulative_valid_tokens=cumulative_valid,
                        )
                    if checkpoint is not None:
                        checkpoint.maybe_checkpoint(cumulative_valid)
                    result["code"] = cinfo
                    _log(f"DONE {ck}: buckets={sorted(cinfo['lengths'].keys())} "
                         f"(cum_all={cumulative['valid']})")
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
        else:
            result["code"] = "disabled"

        # ---- COMMITS half (per-range resume + checkpoint inside) ----
        if streams in {"both", "commits"}:
            try:
                done, failed, all_ranges_done = run_commits_half(
                    repo, repo_dir, repo_work, work_root, work_parent,
                    lengths_commits, range_size,
                    pool, manifest, manifest_lock, resume, cumulative,
                    dedup_db, dedup_near, pr_store, repo_list, memory_limit_gb,
                    progress, checkpoint,
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
        # TEMP RETENTION (requirement 3): delete the repo work dir (incl _src) AND
        # the stable extract cache ONLY when EVERY unit of this repo is marked done
        # (code + every range). A mid-repo kill, a failed unit, or signal-cancelled
        # ranges leave this False -> the temp + cached commit records are RETAINED
        # so resume re-uses them and loses zero work. Only on full completion do we
        # reclaim disk (bounded ~1 repo of source on disk for fully-done repos).
        fully_done = repo_fully_done(repo, manifest, streams, all_ranges_done)
        if keep_temp:
            pass
        elif fully_done:
            shutil.rmtree(repo_work, ignore_errors=True)
            shutil.rmtree(extract_cache_dir(repo), ignore_errors=True)
        else:
            _log(f"RETAIN temp for {repo}: not all units marked done "
                 f"(interrupted/failed/partial); kept {repo_work} + extract cache "
                 f"{extract_cache_dir(repo)} for resume")


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
    p.add_argument("--work-dir", default=None)
    p.add_argument("--work-parent-dir", default=str(DEFAULT_WORK_PARENT),
                   help="Parent directory for conveyor temporary work dirs when "
                        "--work-dir is not set. Default lives under outputs/ on "
                        "the external corpus disk, not the macOS /var/folders "
                        f"temp volume: {DEFAULT_WORK_PARENT}.")
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
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
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
                 "units already persisted; in-progress temp retained for resume.")
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
        target_lengths_code=list(lengths_code),
        target_lengths_commits=list(lengths_commits),
        manifest=str(CONVEYOR_MANIFEST),
        dedup_checkpoint_tokens=args.dedup_checkpoint_tokens,
        dedup_checkpoint_mode=args.dedup_checkpoint_mode,
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
        os.environ.setdefault("TMPDIR", str(work_parent))
        os.environ.setdefault("TMP", str(work_parent))
        os.environ.setdefault("TEMP", str(work_parent))
        work_root = Path(tempfile.mkdtemp(prefix="streaming_conveyor_", dir=str(work_parent)))
        own_work_root = True
    progress.emit("work_root_ready", work_root=str(work_root), own_work_root=own_work_root)

    cumulative = {"valid": 0}
    processed_repos = 0
    code_done = 0
    ranges_done = 0
    ranges_failed = 0
    submitted_repo_names: set[str] = set()

    only_repos = set(args.only_repo) if args.only_repo else None

    def should_process(repo: str) -> bool:
        # Restrict to --only-repo when given (others drained without extraction);
        # otherwise stage every repo. Per-half + per-range resume is downstream.
        if only_repos is not None:
            return repo in only_repos
        if args.streams == "code" and resume and manifest.is_done(code_key(repo)):
            return False
        return True

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

    try:
        gen = (
            sr.stream_repo_subtrees(work_root, should_process)
            if args.streams == "code"
            else stream_repo_subtrees_with_git(work_root, should_process)
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
                    work_root, work_parent, range_pool, manifest, manifest_lock,
                    resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    progress, checkpoint,
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
                    work_root, work_parent, range_pool, manifest, manifest_lock,
                    resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    progress, checkpoint,
                )
                inflight[fut] = repo
                submitted_repos += 1
                drain_one_or_more(block=False)

            while inflight:
                drain_one_or_more(block=True)
    finally:
        if repo_pool is not None:
            repo_pool.shutdown(wait=True)
        range_pool.shutdown(wait=True)
        interrupted = STOP_EVENT.is_set()
        # On a clean full run we reclaim the randomized work_root. On a SIGNAL
        # stop we RETAIN it (in-progress _src kept for resume; the expensive
        # commit records already live in the persistent extract cache regardless).
        if own_work_root and not args.keep_temp and not interrupted:
            shutil.rmtree(work_root, ignore_errors=True)
        for lock in reversed(run_locks):
            lock.close()

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
    return 0 if (not manifest.failed or processed_repos > 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
