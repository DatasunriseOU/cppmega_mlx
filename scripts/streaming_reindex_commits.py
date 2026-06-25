#!/usr/bin/env python3
"""Resumable per-repo streaming COMMIT re-indexer for the cpp_all corpus.

Sibling of ``streaming_reindex.py`` (the CODE driver). Same TRUE single-pass
``zstd -dc | tarfile r|`` stream, one repo at a time, ``_done.json`` resume,
fail-loud per repo (RULE #1). The ONLY differences from the code driver:

  * The repo subtree is extracted WITH its ``.git`` directory (the code driver
    strips it) -- commit history is required. Symlinks/hardlinks inside .git are
    recreated so the object store stays usable.
  * Per repo the pipeline is:
        extract cpp_all/<repo> (incl .git) -> work/<repo>/_src
          -> extract_git_history.py  (git log --no-merges --diff-filter=M
             -> <repo>_commits.jsonl  with old_content/new_content/diff/...)
          -> process_commits.py (--format both; emits change_mask_pre/post,
             hunk_id_per_char, edit_op_per_char + the call/type graph)
          -> clang_enriched_to_parquet.py (tokenize @65536, materialize; the
             materializer DERIVES changed_chunk_ids/spans + projects per-char
             edit columns to per-token)
          -> pack_enriched_rows.py once per target length (dependency-topological
             grouping + per-length edge remap, whole-functions-only)
          -> outputs/reindexed_commits/{1024,2048,4096}/<repo>.parquet
          -> mark <repo> done; rm temp.

Output root is SEPARATE from the code stream: outputs/reindexed_commits/.

All 6 plan commit-edit columns are exact: change_mask_pre/post, edit_op_per_token,
changed_chunk_ids/spans (derived by the materializer), and hunk_id_per_token
(real 0-based per-hunk index from process_commits; -1 for unchanged/context).
No degraded columns; clang handles the full content (docstring-from-message +
pre-image chain + post-image chain + diff).
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
from pathlib import Path
from typing import Sequence

# Reuse the proven machinery from the code driver.
import streaming_reindex as sr
from streaming_reindex import (  # noqa: F401
    MLX_ROOT,
    VENV_PYTHON,
    TARBALL,
    TAR_MEMBER_ROOT,
    TOKENIZE_BUDGET,
    RepoFailure,
    Manifest,
    run_checked,
    stage_index_commits,
    stage_materialize,
    stage_pack,
    _parquet_stats,
)

EXTRACT_GIT = MLX_ROOT / "scripts" / "nanochat_data" / "extract_git_history.py"
COMMIT_OUTPUT_ROOT = MLX_ROOT / "outputs" / "reindexed_commits"
COMMIT_MANIFEST = COMMIT_OUTPUT_ROOT / "_done.json"

# Exclude build/VCS junk but KEEP .git (commits need history).
COMMIT_EXCLUDE_PARTS = frozenset(
    {".svn", "node_modules", "build", "_build", "cmake-build-debug"}
)


def _is_excluded_commit(within: str) -> bool:
    return any(part in COMMIT_EXCLUDE_PARTS for part in within.split("/"))


def stream_repo_subtrees_with_git(work_root: Path, should_process):
    """Yield (repo, repo_dir) for every cpp_all/<repo>/ subtree, ONE pass, .git kept.

    Mirrors streaming_reindex.stream_repo_subtrees but (a) does NOT exclude
    ``.git`` and (b) recreates symlinks/hardlinks so the git object store stays
    usable. Only ONE repo is on disk at a time. Resume-skipped repos are drained
    from the stream without touching disk. RULE #1: a non-contiguous repo subtree
    RAISES (a re-extract of an already-closed repo).
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
                # Recreate links so the .git object store / refs resolve.
                try:
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    if member.issym():
                        os.symlink(member.linkname, target)
                    else:  # hardlink target is another extracted member
                        link_src = cur_dir / member.linkname
                        if link_src.exists():
                            os.link(link_src, target)
                        else:
                            os.symlink(member.linkname, target)
                except OSError:
                    # A link we cannot recreate is not fatal for git in general;
                    # the end-to-end empty-JSONL check below fails loud if .git
                    # turns out unusable for this repo.
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


def stage_extract_commits(repo: str, repo_dir: Path, work: Path) -> Path:
    """extract_git_history.py --repo <_src> -> <repo>_commits.jsonl.

    Empty output (no eligible commits) RAISES RepoFailure -> the driver records
    it and moves on (fail-closed, not a silent skip).
    """
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


def append_commit_output(repo: str, packed: Path, target_length: int) -> dict:
    out_dir = COMMIT_OUTPUT_ROOT / str(target_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{repo}.parquet"
    shutil.copyfile(packed, dest)
    return _parquet_stats(dest, target_length)


def process_one_repo_commits(
    repo: str, repo_dir: Path, target_lengths: Sequence[int], work_root: Path
) -> dict:
    work = work_root / repo
    work.mkdir(parents=True, exist_ok=True)
    commits_jsonl = stage_extract_commits(repo, repo_dir, work)
    # repo_root = the extracted source tree so process_commits resolves includes.
    enriched = stage_index_commits(repo, [commits_jsonl], work, repo_dir, None)
    tok = stage_materialize(repo, enriched, work)
    per_length = {}
    for tl in target_lengths:
        packed = stage_pack(repo, tok, tl, work)
        per_length[str(tl)] = append_commit_output(repo, packed, tl)
    return {"source": "commits", "lengths": per_length}


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--target-lengths", default="1024,2048,4096")
    p.add_argument("--max-repos", type=int, default=None)
    p.add_argument("--token-budget", type=int, default=None,
                   help="Stop after cumulative valid tokens (smallest length) reaches this.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--work-dir", default=None)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    target_lengths = [int(x) for x in args.target_lengths.split(",") if x.strip()]
    if not target_lengths:
        raise SystemExit("--target-lengths produced no lengths")
    smallest = min(target_lengths)

    for path in (VENV_PYTHON, EXTRACT_GIT, sr.PROCESS_COMMITS, sr.MATERIALIZER, sr.PACKER):
        if not Path(path).exists():
            raise SystemExit(f"required path missing: {path}")

    COMMIT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tl in target_lengths:
        (COMMIT_OUTPUT_ROOT / str(tl)).mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(COMMIT_MANIFEST)
    resume = not args.no_resume

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        own_work_root = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="streaming_reindex_commits_"))
        own_work_root = True

    processed = 0
    cumulative_valid = 0
    run_report: dict[str, dict] = {}

    def should_process(repo: str) -> bool:
        return not (resume and manifest.is_done(repo))

    gen = stream_repo_subtrees_with_git(work_root, should_process)
    try:
        for repo, repo_dir in gen:
            try:
                info = process_one_repo_commits(repo, repo_dir, target_lengths, work_root)
                manifest.mark_done(repo, info)
                run_report[repo] = info
                processed += 1
                added = info["lengths"][str(smallest)]["valid_tokens"]
                cumulative_valid += added
                print(f"DONE {repo}: +{added} commit-tok @ {smallest} "
                      f"(cum {cumulative_valid}, repos {processed})",
                      file=sys.stderr, flush=True)
            except RepoFailure as exc:
                print(f"FAIL {repo}: {exc}", file=sys.stderr)
                manifest.mark_failed(repo, exc.stage, exc.detail)
            finally:
                if not args.keep_temp:
                    shutil.rmtree(work_root / repo, ignore_errors=True)
            stop = False
            if args.max_repos is not None and processed >= args.max_repos:
                stop = True
            if args.token_budget is not None and cumulative_valid >= args.token_budget:
                print(f"Token budget {args.token_budget} reached.", file=sys.stderr)
                stop = True
            if stop:
                break
    finally:
        gen.close()

    if own_work_root and not args.keep_temp:
        shutil.rmtree(work_root, ignore_errors=True)

    totals = {str(tl): {"rows": 0, "valid_tokens": 0, "pad_tokens": 0, "capacity_tokens": 0}
              for tl in target_lengths}
    for info in run_report.values():
        for tl_s, st in info["lengths"].items():
            agg = totals[tl_s]
            agg["rows"] += st["rows"]
            agg["valid_tokens"] += st["valid_tokens"]
            agg["pad_tokens"] += st["pad_tokens"]
            agg["capacity_tokens"] += st["capacity_tokens"]
    summary = {
        "processed_this_run": processed,
        "total_done": len(manifest.done),
        "total_failed": len(manifest.failed),
        "per_length_totals": totals,
        "manifest": str(COMMIT_MANIFEST),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not manifest.failed or processed > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
