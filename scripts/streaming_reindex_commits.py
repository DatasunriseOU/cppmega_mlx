#!/usr/bin/env python3
"""Parallel, range-checkpointed, route-by-fit COMMIT re-indexer for cpp_all.

This is the PARALLEL rewrite of the commit driver. It keeps the proven
``zstd -dc | tarfile r|`` ``.git``-preserving single-pass stream as the
SEQUENTIAL producer (one repo on disk at a time, bounded disk), but fans each
repo's commits out to a ``ThreadPoolExecutor`` as per-RANGE pipeline tasks that
run concurrently. Each range task spawns the real ``process_commits`` /
``materialize`` / ``pack_enriched_rows`` subprocesses, so the GIL is irrelevant
(threads only orchestrate; the cores are used by the subprocesses).

Design (RULE #1: fail-loud, no silent fallback; checkpoints EXACT):

  Sequential producer (this file's ``.git``-preserving tar stream, reused from
  the prior commit driver) stages ONE repo at a time:
      extract cpp_all/<repo>/ (incl .git) -> work/<repo>/_src
        -> extract_git_history.py once  (git log --no-merges --diff-filter=M)
           -> <repo>_commits.jsonl   (one JSON record per eligible commit, in
              the SAME order as the commit list)
        -> DELETE .git immediately (commit records already captured)
        -> split the commit list into ranges of R commits (--range-size)
        -> for each range fan a task to the pool:
             slice the records JSONL by line range -> <repo>_r<start>.jsonl
               -> process_commits.py (--format both; edit-signal columns)
               -> clang_enriched_to_parquet.py (tokenize @65536, materialize)
               -> ROUTE-BY-FIT: split tokenized docs by token count into the
                  smallest length bucket that fits the whole commit doc
                  (commits are ATOMIC blocks; never split a doc), ladder
                  1024/2048/4096/8192/16384; N>16384 -> own over-long row in
                  the 16384 bucket.
               -> pack_enriched_rows.py per non-empty bucket at ITS length
               -> recompress packed parquet with MAX zstd (level 22)
               -> outputs/reindexed_commits/{L}/<repo>_r<start>.parquet
               -> mark (repo, range_start_idx) done; rm range temp
      -> move to the next repo only after this repo's ranges all complete
         (bounded disk).

Resume skips COMPLETED RANGES exactly (manifest keyed by
``repo::r<start>`` in outputs/reindexed_commits/_done.json), not whole repos.

Disk-frugal: ``.git`` deleted right after extraction; ``_src`` kept only while
its ranges run (process_commits needs it for include resolution); per-range
intermediates deleted after each range; output parquet is MAX-zstd. Only ~1
repo's source + the currently-running ranges' temp exist at once.

Output root is SEPARATE from the code stream: outputs/reindexed_commits/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

# Reuse the proven machinery from the code driver.
import streaming_reindex as sr
from streaming_reindex import (  # noqa: F401
    MLX_ROOT,
    VENV_PYTHON,
    TARBALL,
    TAR_MEMBER_ROOT,
    RepoFailure,
    Manifest,
    run_checked,
    stage_index_commits,
    stage_materialize,
    stage_pack,
    _parquet_stats,
    _subprocess_env,
)

EXTRACT_GIT = MLX_ROOT / "scripts" / "nanochat_data" / "extract_git_history.py"
COMMIT_OUTPUT_ROOT = MLX_ROOT / "outputs" / "reindexed_commits"
COMMIT_MANIFEST = COMMIT_OUTPUT_ROOT / "_done.json"

# Route-by-fit length ladder (smallest length each whole commit doc fits in).
DEFAULT_TARGET_LENGTHS = (1024, 2048, 4096, 8192, 16384)
DEFAULT_RANGE_SIZE = 500
# Max zstd for output parquet (pyarrow accepts up to 22 for zstd).
ZSTD_LEVEL = 22

# Exclude build/VCS junk but KEEP .git during extraction (commits need history;
# we delete .git ourselves right after extract_git_history runs).
COMMIT_EXCLUDE_PARTS = frozenset(
    {".svn", "node_modules", "build", "_build", "cmake-build-debug"}
)

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, file=sys.stderr, flush=True)


def _is_excluded_commit(within: str) -> bool:
    return any(part in COMMIT_EXCLUDE_PARTS for part in within.split("/"))


def range_key(repo: str, start_idx: int) -> str:
    """Exact checkpoint key for one (repo, range_start_idx)."""
    return f"{repo}::r{start_idx}"


# --------------------------------------------------------------------------- #
# Sequential .git-preserving producer (one repo at a time, bounded disk).
# --------------------------------------------------------------------------- #
def stream_repo_subtrees_with_git(work_root: Path, should_process):
    """Yield (repo, repo_dir) for every cpp_all/<repo>/ subtree, ONE pass, .git kept.

    Only ONE repo is on disk at a time. Resume-skipped repos are drained from the
    stream without touching disk. RULE #1: a non-contiguous repo subtree RAISES.
    """
    import tarfile

    if not TARBALL.exists():
        raise FileNotFoundError(f"source tarball missing: {TARBALL}")
    zstd = subprocess.Popen(
        ["zstd", "-dc", "--long=31", str(TARBALL)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert zstd.stdout is not None
    tar = tarfile.open(fileobj=zstd.stdout, mode="r|")
    prefix = TAR_MEMBER_ROOT + "/"
    finalized: set[str] = set()
    cur_repo: str | None = None
    cur_dir: Path | None = None
    active = False
    try:
        for member in tar:
            name = member.name
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            slash = rest.find("/")
            if slash <= 0:
                continue
            repo = rest[:slash]
            within = rest[slash + 1:]
            if repo != cur_repo:
                if cur_repo is not None and active:
                    yield cur_repo, cur_dir
                    finalized.add(cur_repo)
                if repo in finalized:
                    raise RepoFailure(
                        repo, "stream",
                        "non-contiguous repo subtree in tarball "
                        "(would re-extract an already-closed repo)",
                    )
                cur_repo = repo
                active = should_process(repo)
                cur_dir = work_root / repo / "_src"
                if active:
                    cur_dir.mkdir(parents=True, exist_ok=True)
            if not active or not within:
                continue
            if _is_excluded_commit(within):
                continue
            target = cur_dir / within
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.issym() or member.islnk():
                try:
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    if member.issym():
                        os.symlink(member.linkname, target)
                    else:
                        link_src = cur_dir / member.linkname
                        if link_src.exists():
                            os.link(link_src, target)
                        else:
                            os.symlink(member.linkname, target)
                except OSError:
                    pass
                continue
            if not member.isfile():
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)
        if cur_repo is not None and active:
            yield cur_repo, cur_dir
            finalized.add(cur_repo)
    finally:
        try:
            tar.close()
        except Exception:
            pass
        if zstd.poll() is None:
            zstd.send_signal(signal.SIGTERM)
        try:
            zstd.wait(timeout=10)
        except subprocess.TimeoutExpired:
            zstd.kill()


# --------------------------------------------------------------------------- #
# Per-repo commit-list + records extraction (sequential, once per repo).
# --------------------------------------------------------------------------- #
def get_commit_list(repo_dir: Path) -> list[str]:
    """git log --no-merges --diff-filter=M --format=%H -> commit hashes (log order)."""
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "--format=%H",
         "--no-merges", "--diff-filter=M"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RepoFailure(repo_dir.name, "git_log",
                          f"git log failed: {proc.stderr.strip()[:500]}")
    out = proc.stdout.strip()
    return out.split("\n") if out else []


def stage_extract_commits(repo: str, repo_dir: Path, work: Path) -> Path:
    """extract_git_history.py --repo <_src> -> <repo>_commits.jsonl (all commits)."""
    commits_jsonl = work / f"{repo}_commits.jsonl"
    run_checked(
        repo,
        "extract_git_history",
        [
            VENV_PYTHON, EXTRACT_GIT,
            "--repo", repo_dir,
            "--output", commits_jsonl,
            "--max_commits", "0",
        ],
        log_path=work / f"{repo}.extract.log",
    )
    if not commits_jsonl.exists() or commits_jsonl.stat().st_size == 0:
        raise RepoFailure(repo, "extract_git_history", f"empty commit jsonl: {commits_jsonl}")
    return commits_jsonl


def slice_records_jsonl(records_jsonl: Path, start: int, end: int, dest: Path) -> int:
    """Write records[start:end] (0-based line range) to dest. Returns line count.

    extract_git_history emits one JSON record per eligible commit in the SAME
    order as the commit list, so slicing by line index == slicing by commit
    range. RULE #1: an empty slice RAISES (the range exists in the commit list,
    so it must have records).
    """
    n = 0
    with records_jsonl.open("r", encoding="utf-8") as src, \
            dest.open("w", encoding="utf-8") as out:
        for idx, line in enumerate(src):
            if idx < start:
                continue
            if idx >= end:
                break
            out.write(line)
            n += 1
    if n == 0:
        raise RepoFailure(
            records_jsonl.parent.name, "slice_records",
            f"empty record slice [{start}:{end}] of {records_jsonl}",
        )
    return n


# --------------------------------------------------------------------------- #
# Route-by-fit: split tokenized docs into per-length buckets (atomic docs).
# --------------------------------------------------------------------------- #
TOKEN_IDS_COLUMN = "token_ids"


def bucket_for(token_count: int, lengths_sorted: Sequence[int]) -> int:
    """Smallest length L >= token_count; N>max -> the max length (own over-long row)."""
    for L in lengths_sorted:
        if token_count <= L:
            return L
    return lengths_sorted[-1]


def route_by_fit(tok_parquet: Path, lengths_sorted: Sequence[int], out_dir: Path) -> dict[int, Path]:
    """Split tok parquet rows into per-length parquet files by whole-doc token count.

    Each commit doc (one row) is routed INTACT to the smallest length bucket that
    fits its full ``token_ids`` length; docs longer than the largest length get
    their own over-long row in the largest bucket. Returns {length: parquet_path}
    for non-empty buckets only.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(tok_parquet))
    if TOKEN_IDS_COLUMN not in set(pf.schema_arrow.names):
        raise RepoFailure(tok_parquet.parent.name, "route_by_fit",
                          f"{tok_parquet} missing column {TOKEN_IDS_COLUMN}")
    schema = pf.schema_arrow
    writers: dict[int, pq.ParquetWriter] = {}
    paths: dict[int, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for batch in pf.iter_batches(batch_size=512):
            tbl = pa.Table.from_batches([batch], schema=schema)
            tok_col = tbl.column(TOKEN_IDS_COLUMN).to_pylist()
            # group row indices by destination length
            by_len: dict[int, list[int]] = {}
            for ri, ids in enumerate(tok_col):
                n = len(ids) if ids is not None else 0
                L = bucket_for(n, lengths_sorted)
                by_len.setdefault(L, []).append(ri)
            for L, idxs in by_len.items():
                sub = tbl.take(pa.array(idxs))
                if L not in writers:
                    p = out_dir / f"route_{L}.parquet"
                    paths[L] = p
                    writers[L] = pq.ParquetWriter(str(p), schema)
                writers[L].write_table(sub)
    finally:
        for w in writers.values():
            w.close()
    return paths


def recompress_zstd_max(path: Path) -> None:
    """Rewrite a parquet in place with MAX zstd compression (level 22)."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(str(path))
    tmp = path.with_suffix(".zstd.tmp.parquet")
    pq.write_table(tbl, str(tmp), compression="zstd", compression_level=ZSTD_LEVEL)
    tmp.replace(path)


def append_range_output(repo: str, start_idx: int, packed: Path, target_length: int) -> dict:
    """Place packed parquet at outputs/reindexed_commits/<L>/<repo>_r<start>.parquet (zstd-max)."""
    out_dir = COMMIT_OUTPUT_ROOT / str(target_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{repo}_r{start_idx}.parquet"
    shutil.copyfile(packed, dest)
    recompress_zstd_max(dest)
    return _parquet_stats(dest, target_length)


# --------------------------------------------------------------------------- #
# Per-range pipeline task (runs in a worker thread; spawns subprocesses).
# --------------------------------------------------------------------------- #
def process_range(
    repo: str,
    repo_dir: Path,
    records_jsonl: Path,
    start_idx: int,
    end_idx: int,
    lengths_sorted: Sequence[int],
    repo_work: Path,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    pr_store: Path | None = None,
    repo_list: Path | None = None,
) -> dict:
    """Full per-range pipeline. RAISES RepoFailure on any failure (no fallback)."""
    rkey = range_key(repo, start_idx)
    rwork = repo_work / f"r{start_idx}"
    rwork.mkdir(parents=True, exist_ok=True)
    try:
        slice_jsonl = rwork / f"{repo}_r{start_idx}.jsonl"
        n_records = slice_records_jsonl(records_jsonl, start_idx, end_idx, slice_jsonl)

        # process_commits needs the source tree for include resolution.
        enriched = stage_index_commits(rkey, [slice_jsonl], rwork, repo_dir, None,
                                       dedup_db, dedup_near,
                                       pr_store=pr_store, repo_list=repo_list)
        tok = stage_materialize(rkey, enriched, rwork)

        route_dir = rwork / "routed"
        routed = route_by_fit(tok, lengths_sorted, route_dir)
        if not routed:
            raise RepoFailure(repo, "route_by_fit",
                              f"no docs routed for range [{start_idx}:{end_idx}]")

        per_length: dict[str, dict] = {}
        for L, route_parquet in sorted(routed.items()):
            packed = stage_pack(rkey, route_parquet, L, rwork)
            per_length[str(L)] = append_range_output(repo, start_idx, packed, L)
        return {
            "source": "commits",
            "repo": repo,
            "range": [start_idx, end_idx],
            "n_records": n_records,
            "lengths": per_length,
        }
    finally:
        shutil.rmtree(rwork, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def process_one_repo(
    repo: str,
    repo_dir: Path,
    lengths_sorted: Sequence[int],
    range_size: int,
    work_root: Path,
    pool: ThreadPoolExecutor,
    manifest: Manifest,
    manifest_lock: threading.Lock,
    resume: bool,
    token_budget: int | None,
    cumulative: dict,
    keep_temp: bool,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    pr_store: Path | None = None,
    repo_list: Path | None = None,
) -> int:
    """Extract one repo, fan its ranges to the pool, wait for completion.

    Returns the number of ranges completed this run for this repo. ``.git`` is
    deleted immediately after extraction; ``_src`` is kept until all ranges
    finish (process_commits needs it), then the whole repo work dir is removed.
    """
    repo_work = work_root / repo
    repo_work.mkdir(parents=True, exist_ok=True)
    smallest = lengths_sorted[0]
    try:
        commit_list = get_commit_list(repo_dir)
        if not commit_list:
            raise RepoFailure(repo, "git_log", "no --no-merges --diff-filter=M commits")
        records_jsonl = stage_extract_commits(repo, repo_dir, repo_work)

        # .git is no longer needed (records captured); delete to free disk now.
        git_dir = repo_dir / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)

        # Range over the ACTUAL emitted record count, not len(commit_list):
        # extract_git_history keeps only commits that modify C/C++ files, so the
        # records JSONL is a (possibly shorter) subset of the diff-filter=M list.
        # Slicing by record line index is what aligns the per-range JSONL slices.
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
            )
            futures[fut] = (start, end)

        completed = 0
        for fut in as_completed(futures):
            start, end = futures[fut]
            rkey = range_key(repo, start)
            try:
                info = fut.result()
                with manifest_lock:
                    manifest.mark_done(rkey, info)
                completed += 1
                added = info["lengths"].get(str(smallest), {}).get("valid_tokens", 0)
                with manifest_lock:
                    cumulative["valid"] += sum(
                        st["valid_tokens"] for st in info["lengths"].values()
                    )
                _log(f"DONE {rkey}: ranges [{start}:{end}] "
                     f"buckets={sorted(info['lengths'].keys())} "
                     f"(+{added} @ {smallest}, cum_all={cumulative['valid']})")
            except RepoFailure as exc:
                _log(f"FAIL {rkey}: {exc}")
                with manifest_lock:
                    manifest.mark_failed(rkey, exc.stage, exc.detail)
            except Exception as exc:  # surface unexpected failures loud
                _log(f"FAIL {rkey}: unexpected {type(exc).__name__}: {exc}")
                with manifest_lock:
                    manifest.mark_failed(rkey, "unexpected", str(exc))
            if token_budget is not None and cumulative["valid"] >= token_budget:
                _log(f"Token budget {token_budget} reached.")
                break
        return completed
    finally:
        if not keep_temp:
            shutil.rmtree(repo_work, ignore_errors=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target-lengths", default="1024,2048,4096,8192,16384",
                   help="Route-by-fit length ladder (default 1024,2048,4096,8192,16384).")
    p.add_argument("--range-size", type=int, default=DEFAULT_RANGE_SIZE,
                   help="Commits per checkpointed range (default 500).")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="ThreadPoolExecutor size (default = os.cpu_count()).")
    p.add_argument("--max-repos", type=int, default=None,
                   help="Process at most N repos this run (after resume filtering).")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Stop after cumulative valid tokens (all lengths) reaches this.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--repo-path", action="append", default=[],
                   help="Process this local git repo directly (skip tarball). "
                        "Repeatable. The repo basename is the manifest/output key.")
    p.add_argument("--dedup-db", default=None,
                   help="Path to the SHARED global dedup SQLite store. Pass the SAME "
                        "path as streaming_reindex.py so commit DOCS dedup (by "
                        "tokenized hash) against the code stream too (drops identical "
                        "commits / cherry-picks). Fail-loud, no fallback.")
    p.add_argument("--no-near-dedup", action="store_true",
                   help="Disable MinHash-LSH near dedup (exact-only).")
    p.add_argument("--pr-store", default=None,
                   help="Path to the Tier-2 PR-discussion SQLite store "
                        "(e.g. outputs/pr_ingest/prs.sqlite). When set, each "
                        "commit record is looked up by (owner_repo, pr_number) "
                        "then (owner_repo, commit_hash) and, on a hit, the "
                        "rendered PR discussion is attached as "
                        "record['pr_discussion'] (HEAD of the commit doc). Miss "
                        "= Tier-1 git-only (no fail).")
    p.add_argument("--repo-list", default=None,
                   help="Path to outputs/pr_ingest/repo_list.json (bare-name -> "
                        "owner/repo map) for resolving the PR-store key.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    target_lengths = sorted(int(x) for x in args.target_lengths.split(",") if x.strip())
    if not target_lengths:
        raise SystemExit("--target-lengths produced no lengths")
    lengths_sorted = tuple(target_lengths)
    smallest = lengths_sorted[0]
    workers = max(1, int(args.workers or 1))

    for path in (VENV_PYTHON, EXTRACT_GIT, sr.PROCESS_COMMITS,
                 sr.MATERIALIZER, sr.PACKER):
        if not Path(path).exists():
            raise SystemExit(f"required path missing: {path}")

    COMMIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tl in lengths_sorted:
        (COMMIT_OUTPUT_ROOT / str(tl)).mkdir(parents=True, exist_ok=True)

    # Shared global dedup db (SAME path as the code stream for cross-stream dedup).
    # FAIL LOUD up front (RULE #1): bad path / missing datasketch crashes now.
    dedup_db = Path(args.dedup_db) if args.dedup_db else None
    dedup_near = not args.no_near_dedup
    if dedup_db is not None:
        sys.path.insert(0, str(MLX_ROOT / "tools" / "clang_indexer"))
        from dedup_store import DedupStore  # noqa: E402
        dedup_db.parent.mkdir(parents=True, exist_ok=True)
        DedupStore(str(dedup_db), near=dedup_near, commit_every=1000).close()
        _log(f"Dedup: SHARED global commit-doc store at {dedup_db} "
             f"(exact{'+near' if dedup_near else ''}, tokenized hash)")

    # Tier-2 PR-discussion live lookup (fail-loud on a bad path up front).
    pr_store = Path(args.pr_store) if args.pr_store else None
    repo_list = Path(args.repo_list) if args.repo_list else None
    if pr_store is not None and not pr_store.exists():
        raise SystemExit(f"--pr-store does not exist: {pr_store}")
    if repo_list is not None and not repo_list.exists():
        raise SystemExit(f"--repo-list does not exist: {repo_list}")
    if pr_store is not None:
        _log(f"PR-store: live lookup into record['pr_discussion'] from {pr_store} "
             f"(repo_list={repo_list})")

    manifest = Manifest.load(COMMIT_MANIFEST)
    manifest_lock = threading.Lock()
    resume = not args.no_resume

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        own_work_root = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="streaming_reindex_commits_"))
        own_work_root = True

    cumulative = {"valid": 0}
    processed_repos = 0
    ranges_done = 0

    def should_process(repo: str) -> bool:
        # Always stage the repo; per-range resume happens inside process_one_repo.
        return True

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        if args.repo_path:
            repo_iter = []
            for rp in args.repo_path:
                rp_path = Path(rp).resolve()
                if not (rp_path / ".git").exists():
                    raise SystemExit(f"--repo-path is not a git repo (no .git): {rp_path}")
                # Stage into work_root/<name>/_src so .git deletion is isolated to the copy.
                name = rp_path.name
                staged = work_root / name / "_src"
                staged.mkdir(parents=True, exist_ok=True)
                _log(f"STAGE (local copy) {name} <- {rp_path}")
                # Copy including .git (commits need it); excludes match the stream.
                for child in rp_path.iterdir():
                    if child.name in COMMIT_EXCLUDE_PARTS:
                        continue
                    dest = staged / child.name
                    if child.is_dir():
                        shutil.copytree(child, dest, symlinks=True,
                                        ignore=shutil.ignore_patterns(*COMMIT_EXCLUDE_PARTS))
                    else:
                        shutil.copy2(child, dest, follow_symlinks=False)
                repo_iter.append((name, staged))
            gen = iter(repo_iter)
        else:
            gen = stream_repo_subtrees_with_git(work_root, should_process)

        for repo, repo_dir in gen:
            try:
                done = process_one_repo(
                    repo, repo_dir, lengths_sorted, args.range_size,
                    work_root, pool, manifest, manifest_lock, resume,
                    args.token_budget, cumulative, args.keep_temp,
                    dedup_db, dedup_near, pr_store, repo_list,
                )
                ranges_done += done
                processed_repos += 1
            except RepoFailure as exc:
                _log(f"FAIL repo {repo}: {exc}")
                with manifest_lock:
                    manifest.mark_failed(f"{repo}::repo", exc.stage, exc.detail)
            stop = False
            if args.max_repos is not None and processed_repos >= args.max_repos:
                stop = True
            if args.token_budget is not None and cumulative["valid"] >= args.token_budget:
                stop = True
            if stop:
                if hasattr(gen, "close"):
                    gen.close()
                break
    finally:
        pool.shutdown(wait=True)
        if own_work_root and not args.keep_temp:
            shutil.rmtree(work_root, ignore_errors=True)

    # ----- cumulative per-length report from the manifest -----
    totals = {str(tl): {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0}
              for tl in lengths_sorted}
    for info in manifest.done.values():
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
        "ranges_this_run": ranges_done,
        "workers": workers,
        "range_size": args.range_size,
        "target_lengths": list(lengths_sorted),
        "total_done_ranges": len([k for k in manifest.done]),
        "total_failed": len(manifest.failed),
        "cumulative_valid_tokens_this_run": cumulative["valid"],
        "per_length_totals": totals,
        "manifest": str(COMMIT_MANIFEST),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not manifest.failed or ranges_done > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
