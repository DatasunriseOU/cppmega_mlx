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
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
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

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, file=sys.stderr, flush=True)


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


def _length_totals(info: dict) -> dict[str, int]:
    totals = {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0}
    for st in info.get("lengths", {}).values():
        for key in totals:
            totals[key] += int(st.get(key, 0))
    return totals


def code_key(repo: str) -> str:
    """Checkpoint key for one repo's CODE half."""
    return f"{repo}::code"


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
) -> tuple[int, int]:
    """Extract commit records once, fan ranges to the pool. Returns (done, failed).

    .git is deleted immediately after extract_git_history (records captured),
    keeping disk bounded. Each range is checkpointed exactly as ``<repo>::r<start>``
    so resume skips finished ranges. RAISES RepoFailure only for the up-front
    git-log / extract steps; per-range failures are recorded and skipped.
    """
    lengths_sorted = tuple(sorted(int(x) for x in lengths_commits))
    smallest = lengths_sorted[0]

    commit_list = get_commit_list(repo_dir)
    if not commit_list:
        raise RepoFailure(repo, "git_log", "no --no-merges --diff-filter=M commits")
    records_jsonl = stage_extract_commits(repo, repo_dir, repo_work)

    # .git no longer needed -> free disk now (records already captured).
    git_dir = repo_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)

    # Range over ACTUAL emitted record count (extract keeps only C/C++-touching
    # commits, so the records JSONL is a subset; slicing by line index aligns).
    with records_jsonl.open("r", encoding="utf-8") as fh:
        n_records = sum(1 for _ in fh)
    if n_records == 0:
        raise RepoFailure(repo, "extract_git_history",
                          f"zero records after extract for {repo}")
    ranges = [(s, min(s + range_size, n_records))
              for s in range(0, n_records, range_size)]

    futures = {}
    for (start, end) in ranges:
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
    for fut in as_completed(futures):
        start, end = futures[fut]
        rkey = range_key(repo, start)
        try:
            rinfo = fut.result()
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
            _log(f"DONE {rkey}: ranges [{start}:{end}] "
                 f"buckets={sorted(rinfo['lengths'].keys())} "
                 f"(+{added} @ {smallest}, cum_all={cumulative['valid']})")
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
    return done, failed


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
) -> dict:
    """Run BOTH halves for one already-extracted repo subtree, then delete it.

    The repo was extracted ONCE (incl .git) by the .git-preserving stream into
    ``repo_dir`` (== work_root/<repo>/_src). The CODE half runs first (it does
    NOT touch .git); then the COMMITS half consumes + deletes .git. The whole
    repo work dir is removed at the end (bounded disk). RULE #1: a failure in
    one half is recorded; the other half still runs.
    """
    repo_work = work_root / repo
    repo_work.mkdir(parents=True, exist_ok=True)
    result = {"repo": repo, "code": None, "commits_done": 0, "commits_failed": 0}
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
                done, failed = run_commits_half(
                    repo, repo_dir, repo_work, lengths_commits, range_size,
                    pool, manifest, manifest_lock, resume, cumulative,
                    dedup_db, dedup_near, pr_store, repo_list, memory_limit_gb,
                    progress,
                )
                result["commits_done"] = done
                result["commits_failed"] = failed
            except RepoFailure as exc:
                _log(f"FAIL {repo}::commits: {exc}")
                with manifest_lock:
                    manifest.mark_failed(f"{repo}::commits", exc.stage, exc.detail)
        return result
    finally:
        # CODE half already cleaned work_root/<repo>/<repo>-internal via
        # process_one_repo; here we remove the whole repo dir incl _src so only
        # ~1 repo of source ever exists on disk.
        if not keep_temp:
            shutil.rmtree(repo_work, ignore_errors=True)


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
        DedupStore(str(dedup_db), near=dedup_near, commit_every=1000).close()
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

    manifest = Manifest.load(CONVEYOR_MANIFEST)
    manifest_lock = threading.Lock()
    resume = not args.no_resume
    progress = ProgressWriter(Path(args.progress_jsonl) if args.progress_jsonl else None)
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
    )

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        own_work_root = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="streaming_conveyor_"))
        own_work_root = True

    cumulative = {"valid": 0}
    processed_repos = 0
    code_done = 0
    ranges_done = 0
    ranges_failed = 0

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

    try:
        gen = (
            sr.stream_repo_subtrees(work_root, should_process)
            if args.streams == "code"
            else stream_repo_subtrees_with_git(work_root, should_process)
        )

        if repo_pool is None:
            for repo, repo_dir in gen:
                res = process_one_repo(
                    repo, repo_dir, lengths_code, lengths_commits, args.range_size,
                    work_root, range_pool, manifest, manifest_lock, resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    progress,
                )
                processed_repos += 1
                if isinstance(res.get("code"), dict):
                    code_done += 1
                ranges_done += res.get("commits_done", 0)
                ranges_failed += res.get("commits_failed", 0)

                stop = False
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
                while len(inflight) >= max_active_repos:
                    drain_one_or_more(block=True)
                    if args.token_budget is not None and cumulative["valid"] >= args.token_budget:
                        _log(f"Token budget {args.token_budget} reached.")
                        stop_submitting = True
                        break
                if stop_submitting:
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
                    work_root, range_pool, manifest, manifest_lock, resume, cumulative,
                    args.keep_temp, dedup_db, dedup_near, pr_store, repo_list,
                    args.streams, global_symbol_index, args.memory_limit_gb,
                    progress,
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
        if own_work_root and not args.keep_temp:
            shutil.rmtree(work_root, ignore_errors=True)

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
    }
    progress.emit("run_finished", **summary)
    print(json.dumps(summary, indent=2))
    return 0 if (not manifest.failed or processed_repos > 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
