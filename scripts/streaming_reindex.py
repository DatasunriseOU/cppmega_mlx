#!/usr/bin/env python3
"""Resumable per-repo streaming re-indexer for the cpp_all corpus.

Pipeline (one repo at a time, fail-loud per repo, continue on failure):

    extract cpp_all/<repo> subtree from the zstd tarball  ->  temp dir
      -> index_project.py  (--enriched; channel A build_info from the repo's
         own compile_commands.json when present, empty otherwise -- NO boilerplate)
      -> clang_enriched_to_parquet.py (tokenize @ 65536, --materialize-tokenized-enriched,
         --overflow-policy drop)
      -> pack_enriched_rows.py once per --target-length (best_fit, no-crop)
      -> append packed parquet to outputs/reindexed/{1024,2048,4096}/<repo>.parquet
      -> mark <repo> done in outputs/reindexed/_done.json
      -> rm temp

Commit sources (--commit-source ...) are re-indexed the SAME way via
process_commits.py instead of index_project.py; everything downstream is
identical. Commits average ~1986 tokens so they mostly land in the 2048/4096
buckets -- that is expected, not an error.

NO automated/silent fallbacks (RULE #1): every step runs through ONE clear
subprocess path; any non-zero exit, missing output, or empty result RAISES a
RepoFailure with WHERE+WHAT, which is recorded and the driver moves to the next
repo. There is no degraded / best-effort / zero-output path.

Source: /Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst
  members: cpp_all/<repo>/...
  decompress: zstd -dc --long=31
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
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Fixed environment contract (verified by the task brief).
# --------------------------------------------------------------------------- #
MLX_ROOT = Path("/Volumes/external/sources/cppmega.mlx")
VENV_PYTHON = MLX_ROOT / ".venv" / "bin" / "python"
TOKENIZER_PATH = MLX_ROOT / "cppmega_mlx" / "tokenizer" / "tokenizer.json"

INDEX_PROJECT = MLX_ROOT / "tools" / "clang_indexer" / "index_project.py"
PROCESS_COMMITS = MLX_ROOT / "tools" / "clang_indexer" / "process_commits.py"
MATERIALIZER = MLX_ROOT / "scripts" / "nanochat_data" / "clang_enriched_to_parquet.py"
PACKER = MLX_ROOT / "scripts" / "nanochat_data" / "pack_enriched_rows.py"

TARBALL = Path("/Users/dave/sources/parquet/data-cpp_all/data-cpp_all.tar.zst")
TAR_MEMBER_ROOT = "cpp_all"

OUTPUT_ROOT = MLX_ROOT / "outputs" / "reindexed"
MANIFEST_PATH = OUTPUT_ROOT / "_done.json"

# Tokenize at the model's full context so packing decides the final lengths.
TOKENIZE_BUDGET = 65536
DEFAULT_TARGET_LENGTHS = (1024, 2048, 4096)

# Directories never worth indexing (VCS / build artifacts). index_project has its
# own excludes; we additionally avoid extracting .git to keep staging small for
# pure-source repos. For commit re-indexing the .git history is needed, so we do
# NOT strip it there.
SOURCE_EXTRA_EXCLUDE = ".git,.svn,node_modules,build,_build,cmake-build-debug"
EXCLUDE_PARTS = frozenset(SOURCE_EXTRA_EXCLUDE.split(","))


def _route_by_fit_impl():
    """Import bucket_for/route_by_fit from the commits driver (the ONE shared
    implementation), deferred to avoid a module-load cycle (streaming_reindex_commits
    imports this module at top). Both streams thus route identically."""
    sys.path.insert(0, str(MLX_ROOT / "scripts"))
    from streaming_reindex_commits import bucket_for, route_by_fit  # noqa: F401
    return bucket_for, route_by_fit


class RepoFailure(RuntimeError):
    """A single repo failed at a specific stage. Recorded; driver continues."""

    def __init__(self, repo: str, stage: str, detail: str):
        super().__init__(f"[{repo}] stage={stage}: {detail}")
        self.repo = repo
        self.stage = stage
        self.detail = detail


def code_stage_id(repo: str) -> str:
    return f"code:{repo}"


def commit_stage_id(key: str) -> str:
    return f"commit:{key}"


def _safe_stage_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def code_stage_db(work: Path, repo: str) -> Path:
    return work / f"{_safe_stage_name(repo)}.dedup_stage.sqlite"


def commit_stage_db(work: Path, key: str) -> Path:
    return work / f"{_safe_stage_name(key)}.dedup_stage.sqlite"


def is_code_worktree_repo(repo: str) -> bool:
    """Return False for archive members that are git object stores, not sources."""
    return not repo.endswith(".bare")


def _dedup_store_cls():
    sys.path.insert(0, str(MLX_ROOT / "tools" / "clang_indexer"))
    from dedup_store import DedupStore
    return DedupStore


def dedup_promote_lock_path(dedup_db: Path) -> Path:
    return Path(str(dedup_db) + ".promote.lock")


@contextmanager
def dedup_promote_lock(dedup_db: Path):
    """Serialize parent-stage promotion before touching the global SQLite DB."""
    import fcntl

    lock_path = dedup_promote_lock_path(dedup_db)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_s = float(os.environ.get("CPPMEGA_DEDUP_PROMOTE_LOCK_TIMEOUT_SECONDS", "600"))
    deadline = time.monotonic() + timeout_s
    sleep_s = 0.05
    started = time.monotonic()
    with lock_path.open("a+b") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "DedupStore: global promote lock remained held after "
                        f"{timeout_s:.0f}s at {str(lock_path)!r}"
                    ) from exc
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 1.5, 2.0)
        wait_s = time.monotonic() - started
        try:
            yield wait_s
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def promote_dedup_stage(
    dedup_db: Path | None,
    stage_id: str | None,
    stage_db: Path | None = None,
) -> dict[str, float]:
    if dedup_db is None or stage_id is None:
        return {"promote_wait_s": 0.0, "promote_duration_s": 0.0}
    with dedup_promote_lock(dedup_db) as wait_s:
        started = time.monotonic()
        DedupStore = _dedup_store_cls()
        if stage_db is not None:
            DedupStore.promote_stage_from_db(str(dedup_db), str(stage_db), stage_id)
        else:
            DedupStore.promote_stage_in_db(str(dedup_db), stage_id)
        duration_s = time.monotonic() - started
    return {
        "promote_wait_s": round(wait_s, 6),
        "promote_duration_s": round(duration_s, 6),
    }


def promote_dedup_stages(
    dedup_db: Path | None,
    stages: Sequence[tuple[str, Path | None]],
) -> dict[str, float | int]:
    """Promote multiple dedup stages while holding the global writer lock once."""
    real_stages = [(sid, sdb) for sid, sdb in stages if dedup_db is not None and sid]
    if dedup_db is None or not real_stages:
        return {
            "promote_wait_s": 0.0,
            "promote_duration_s": 0.0,
            "promote_batch_size": 0,
        }
    with dedup_promote_lock(dedup_db) as wait_s:
        started = time.monotonic()
        DedupStore = _dedup_store_cls()
        for stage_id, stage_db in real_stages:
            if stage_db is not None:
                DedupStore.promote_stage_from_db(str(dedup_db), str(stage_db), stage_id)
            else:
                DedupStore.promote_stage_in_db(str(dedup_db), stage_id)
        duration_s = time.monotonic() - started
    return {
        "promote_wait_s": round(wait_s, 6),
        "promote_duration_s": round(duration_s, 6),
        "promote_batch_size": len(real_stages),
    }


def discard_dedup_stage(
    dedup_db: Path | None,
    stage_id: str | None,
    stage_db: Path | None = None,
) -> None:
    if dedup_db is None or stage_id is None:
        return
    DedupStore = _dedup_store_cls()
    DedupStore.discard_stage(
        str(dedup_db),
        stage_id,
        stage_db_path=str(stage_db) if stage_db is not None else None,
    )


# --------------------------------------------------------------------------- #
# Subprocess helper -- fail loud, no swallowed errors.
# --------------------------------------------------------------------------- #
def _subprocess_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    parts = [str(MLX_ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_checked(
    repo: str,
    stage: str,
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    log_path: Path | None = None,
    timeout: int | None = None,
    stall_timeout: int | None = None,
) -> None:
    """Run a command; RAISE RepoFailure on any non-zero exit. No fallback."""
    printable = " ".join(str(c) for c in cmd)
    print(f"  [{repo}] {stage}: {printable}", file=sys.stderr, flush=True)
    log_fh = open(log_path, "wb") if log_path else None
    stdout_data: bytes | None = None
    proc: subprocess.Popen[bytes] | None = None

    def terminate_process(reason: str) -> None:
        if proc is None:
            return
        try:
            if timeout or stall_timeout:
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if timeout or stall_timeout:
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.wait()
        print(f"  [{repo}] {stage}: killed after {reason}", file=sys.stderr, flush=True)

    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            env=_subprocess_env(),
            stdout=log_fh if log_fh else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=bool(timeout or stall_timeout),
        )
        if stall_timeout and log_path is not None:
            deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
            last_activity = time.monotonic()
            last_signature: tuple[int, int] | None = None
            while proc.poll() is None:
                now = time.monotonic()
                if deadline is not None and now > deadline:
                    terminate_process(f"timeout {timeout}s")
                    raise RepoFailure(
                        repo,
                        stage,
                        f"timed out after {timeout}s",
                    )
                if log_path.exists():
                    stat = log_path.stat()
                    signature = (stat.st_size, stat.st_mtime_ns)
                    if signature != last_signature:
                        last_signature = signature
                        last_activity = now
                if now - last_activity > stall_timeout:
                    terminate_process(f"no log progress for {stall_timeout}s")
                    raise RepoFailure(
                        repo,
                        stage,
                        f"stalled after {stall_timeout}s without log progress",
                    )
                time.sleep(1.0)
            stdout_data, _ = proc.communicate(timeout=1)
        else:
            stdout_data, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            terminate_process(f"timeout {timeout}s")
        raise RepoFailure(repo, stage, f"timed out after {timeout}s: {exc}") from exc
    finally:
        if log_fh:
            log_fh.close()
    if proc is None:
        raise RepoFailure(repo, stage, "subprocess did not start")
    if proc.returncode != 0:
        tail = ""
        if log_path and log_path.exists():
            data = log_path.read_bytes()[-4000:]
            tail = data.decode("utf-8", errors="replace")
        elif stdout_data:
            tail = stdout_data.decode("utf-8", errors="replace")[-4000:]
        raise RepoFailure(
            repo, stage, f"exit code {proc.returncode}\n--- last output ---\n{tail}"
        )


# --------------------------------------------------------------------------- #
# Manifest (resume) helpers.
# --------------------------------------------------------------------------- #
@dataclass
class Manifest:
    path: Path
    done: dict = field(default_factory=dict)
    failed: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.exists():
            blob = json.loads(path.read_text())
            return cls(path=path, done=blob.get("done", {}), failed=blob.get("failed", {}))
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"done": self.done, "failed": self.failed},
                indent=2,
                sort_keys=True,
            )
        )
        tmp.replace(self.path)

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str, info: dict) -> None:
        self.done[key] = info
        self.failed.pop(key, None)
        self.save()

    def mark_failed(self, key: str, stage: str, detail: str) -> None:
        self.failed[key] = {
            "stage": stage,
            "detail": detail[:2000],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save()


# --------------------------------------------------------------------------- #
# Tarball streaming (TRUE single-pass, one repo at a time -- no enumeration,
# no /tmp bulk extract).
# --------------------------------------------------------------------------- #
def _is_excluded(within: str) -> bool:
    """True if any path component of the in-repo path is an excluded dir."""
    return any(part in EXCLUDE_PARTS for part in within.split("/"))


SOURCE_CACHE_SENTINEL = ".cppmega_source_cache_complete.json"


def _source_cache_repo_dir(source_cache_dir: Path, repo: str) -> Path:
    return source_cache_dir / repo


def _source_cache_staging_dir(source_cache_dir: Path, repo: str) -> Path:
    staging = source_cache_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging / f"{repo}.{os.getpid()}.{time.time_ns()}"


def _source_cache_complete(repo_dir: Path) -> bool:
    return (repo_dir / SOURCE_CACHE_SENTINEL).exists()


def _mark_source_cache_complete(repo_dir: Path, repo: str) -> None:
    sentinel = repo_dir / SOURCE_CACHE_SENTINEL
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "repo": repo,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": str(TARBALL),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(sentinel)


def _iter_complete_source_cache(
    source_cache_dir: Path,
    should_process: Callable[[str], bool],
) -> set[str]:
    yielded: set[str] = set()
    if not source_cache_dir.exists():
        return yielded
    for repo_dir in sorted(p for p in source_cache_dir.iterdir() if p.is_dir()):
        if repo_dir.name == ".staging":
            continue
        repo = repo_dir.name
        if not _source_cache_complete(repo_dir) or not should_process(repo):
            continue
        yielded.add(repo)
    return yielded


def stream_repo_subtrees(
    work_root: Path,
    should_process,
    *,
    source_cache_dir: Path | None = None,
    source_cache_only: bool = False,
):
    """Yield (repo, repo_dir) for every cpp_all/<repo>/ subtree, in ONE pass.

    A single `zstd -dc --long=31 TARBALL` is piped into a Python
    `tarfile.open(fileobj=..., mode="r|")` stream and read sequentially. Members
    arrive in archive order and each repo's subtree is contiguous (the archive
    was built by a directory walk), so a change in the `<repo>` path component
    marks the end of the previous repo -- which is then yielded, fully extracted
    under work_root/<repo>/_src (the cpp_all/<repo>/ prefix stripped). Repos for
    which should_process(repo) is False are DRAINED from the stream without
    touching disk (resume skip). Only ONE repo is ever on disk at a time.

    RULE #1: a non-contiguous repo subtree (a repo name reappearing after it was
    closed) RAISES -- it would mean a partial re-extract, never silently
    tolerated. There is no separate header/enumeration pass and no /tmp bulk.
    """
    import tarfile

    cached_yielded: set[str] = set()
    if source_cache_dir is not None:
        source_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_yielded = _iter_complete_source_cache(source_cache_dir, should_process)
        for repo in sorted(cached_yielded):
            yield repo, _source_cache_repo_dir(source_cache_dir, repo)
        if source_cache_only:
            return

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
    cur_final_dir: Path | None = None
    active = False
    try:
        for member in tar:
            name = member.name
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            slash = rest.find("/")
            if slash <= 0:
                continue  # a bare entry directly under cpp_all/ -- not a repo
            repo = rest[:slash]
            within = rest[slash + 1:]
            if repo != cur_repo:
                if cur_repo is not None and active:
                    assert cur_dir is not None
                    yield_dir = cur_dir
                    if cur_final_dir is not None:
                        _mark_source_cache_complete(cur_dir, cur_repo)
                        if cur_final_dir.exists() and _source_cache_complete(cur_final_dir):
                            shutil.rmtree(cur_dir, ignore_errors=True)
                            yield_dir = cur_final_dir
                        else:
                            if cur_final_dir.exists():
                                shutil.rmtree(cur_final_dir)
                            cur_dir.rename(cur_final_dir)
                            yield_dir = cur_final_dir
                    yield cur_repo, yield_dir
                    finalized.add(cur_repo)
                if repo in finalized:
                    raise RepoFailure(
                        repo, "stream",
                        "non-contiguous repo subtree in tarball "
                        "(would re-extract an already-closed repo)",
                    )
                cur_repo = repo
                active = repo not in cached_yielded and should_process(repo)
                cur_final_dir = None
                if active and source_cache_dir is not None:
                    cur_final_dir = _source_cache_repo_dir(source_cache_dir, repo)
                    if _source_cache_complete(cur_final_dir):
                        active = False
                        cached_yielded.add(repo)
                        cur_dir = None
                    else:
                        cur_dir = _source_cache_staging_dir(source_cache_dir, repo)
                else:
                    cur_dir = work_root / repo / "_src"
                if active:
                    assert cur_dir is not None
                    cur_dir.mkdir(parents=True, exist_ok=True)
            if not active or not within:
                continue
            if not member.isfile() or _is_excluded(within):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target = cur_dir / within
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)
        if cur_repo is not None and active:
            assert cur_dir is not None
            yield_dir = cur_dir
            if cur_final_dir is not None:
                _mark_source_cache_complete(cur_dir, cur_repo)
                if cur_final_dir.exists() and _source_cache_complete(cur_final_dir):
                    shutil.rmtree(cur_dir, ignore_errors=True)
                    yield_dir = cur_final_dir
                else:
                    if cur_final_dir.exists():
                        shutil.rmtree(cur_final_dir)
                    cur_dir.rename(cur_final_dir)
                    yield_dir = cur_final_dir
            yield cur_repo, yield_dir
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


def stream_repo_dirs(source_roots: Sequence[Path], should_process):
    """Yield already-extracted repo directories without opening the tarball."""
    seen: set[str] = set()
    for root in source_roots:
        if not root.exists():
            raise FileNotFoundError(f"source dir root missing: {root}")
        for entry in sorted(p for p in root.iterdir() if p.is_dir()):
            repo = entry.name
            if repo in seen or not is_code_worktree_repo(repo) or not should_process(repo):
                continue
            repo_dir = entry / "_src" if (entry / "_src").is_dir() else entry
            seen.add(repo)
            yield repo, repo_dir


# --------------------------------------------------------------------------- #
# Per-repo pipeline stages.
# --------------------------------------------------------------------------- #
def stage_index_source(
    repo: str,
    repo_dir: Path,
    work: Path,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    dedup_stage_id: str | None = None,
    dedup_stage_db: Path | None = None,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
) -> Path:
    """index_project.py --enriched -> <repo>.enriched.jsonl.

    Passes --tokenizer-path so index_project runs FUNCTION-LEVEL tokenized-hash
    dedup before grouping; --dedup-db makes that dedup GLOBAL + resumable +
    cross-stream (shared with commits) when given.

    --global-symbol-index, when given, enables bounded cross-repo base-lib symbol
    linking inside index_project (depth-1 pulls tagged crosslib:<repo>). None ->
    behavior unchanged.
    """
    enriched = work / f"{repo}.enriched.jsonl"
    cmd = [
        VENV_PYTHON, INDEX_PROJECT,
        "--project-dir", repo_dir,
        "--output", enriched,
        "--enriched",
        "--max-tokens", str(TOKENIZE_BUDGET),
        "--exclude-dirs", SOURCE_EXTRA_EXCLUDE,
        "--tokenizer-path", TOKENIZER_PATH,
        "--memory-limit-gb", str(memory_limit_gb),
        "--parse-workers", str(max(1, int(parse_workers))),
    ]
    if dedup_db is not None:
        cmd += ["--dedup-db", str(dedup_db)]
        if dedup_stage_id is not None:
            cmd += ["--dedup-stage-id", dedup_stage_id]
            if dedup_stage_db is not None:
                cmd += ["--dedup-stage-db", str(dedup_stage_db)]
    if not dedup_near:
        cmd += ["--no-near-dedup"]
    if global_symbol_index is not None:
        cmd += ["--global-symbol-index", str(global_symbol_index)]
    run_checked(
        repo,
        "index_project",
        cmd,
        log_path=work / f"{repo}.index.log",
        timeout=index_timeout_s if index_timeout_s and index_timeout_s > 0 else None,
        stall_timeout=(
            index_stall_timeout_s
            if index_stall_timeout_s and index_stall_timeout_s > 0
            else None
        ),
    )
    if not enriched.exists() or enriched.stat().st_size == 0:
        raise RepoFailure(repo, "index_project", f"empty enriched jsonl: {enriched}")
    return enriched


def stage_index_commits(repo: str, commit_inputs: Sequence[Path], work: Path,
                        repo_root: Path | None, repo_dir: Path | None,
                        dedup_db: Path | None = None,
                        dedup_near: bool = True,
                        dedup_stage_id: str | None = None,
                        dedup_stage_db: Path | None = None,
                        pr_store: Path | None = None,
                        repo_list: Path | None = None,
                        memory_limit_gb: float = 10.0,
                        analysis_cache_entries: int = 128,
                        allow_empty: bool = False) -> Path | None:
    """process_commits.py -> <repo>.enriched.jsonl (commit edit-signal docs).

    A commit is an ATOMIC change-unit: process_commits dedups whole commit DOCS
    by the tokenized hash of the doc (drops identical commits, e.g. cherry-picks)
    while keeping route-by-fit. --dedup-db makes that dedup share the SAME global
    store as the code stream.
    """
    enriched = work / f"{repo}.enriched.jsonl"
    cmd = [
        VENV_PYTHON, PROCESS_COMMITS,
        "--inputs", *[str(p) for p in commit_inputs],
        "--output", enriched,
        "--max-tokens", str(TOKENIZE_BUDGET),
        "--tokenizer-path", TOKENIZER_PATH,
        "--format", "both",
        "--memory-limit-gb", str(memory_limit_gb),
        "--analysis-cache-entries", str(max(0, int(analysis_cache_entries))),
    ]
    if dedup_db is not None:
        cmd += ["--dedup-db", str(dedup_db)]
        if dedup_stage_id is not None:
            cmd += ["--dedup-stage-id", dedup_stage_id]
            if dedup_stage_db is not None:
                cmd += ["--dedup-stage-db", str(dedup_stage_db)]
    if not dedup_near:
        cmd += ["--no-near-dedup"]
    if repo_root is not None:
        cmd += ["--repo-root", str(repo_root)]
    if repo_dir is not None:
        cmd += ["--repo-dir", str(repo_dir)]
    if pr_store is not None:
        cmd += ["--pr-store", str(pr_store)]
    if repo_list is not None:
        cmd += ["--repo-list", str(repo_list)]
    run_checked(repo, "process_commits", cmd, log_path=work / f"{repo}.commits.log")
    if not enriched.exists():
        raise RepoFailure(repo, "process_commits", f"empty enriched jsonl: {enriched}")
    if enriched.stat().st_size == 0:
        if allow_empty:
            return None
        raise RepoFailure(repo, "process_commits", f"empty enriched jsonl: {enriched}")
    return enriched


def stage_materialize(
    repo: str,
    enriched: Path,
    work: Path,
    memory_limit_gb: float = 10.0,
) -> Path:
    """clang_enriched_to_parquet.py -> tokenized enriched parquet (single file)."""
    tok = work / f"{repo}.tok.parquet"
    run_checked(
        repo,
        "materialize",
        [
            VENV_PYTHON, MATERIALIZER,
            "--input-file", enriched,
            "--output-file", tok,
            "--tokenizer-path", TOKENIZER_PATH,
            "--materialize-tokenized-enriched",
            "--overflow-policy", "drop",
            "--size", _budget_size_label(TOKENIZE_BUDGET),
            "--memory-limit-gb", str(memory_limit_gb),
        ],
        log_path=work / f"{repo}.materialize.log",
    )
    if not tok.exists() or tok.stat().st_size == 0:
        raise RepoFailure(repo, "materialize", f"empty tokenized parquet: {tok}")
    return tok


def _budget_size_label(budget: int) -> str:
    if budget % 1024 != 0:
        raise ValueError(f"budget {budget} is not a multiple of 1024")
    return f"{budget // 1024}k"


def stage_pack(repo: str, tok: Path, target_length: int, work: Path) -> Path:
    """pack_enriched_rows.py -> packed parquet for one target length."""
    packed = work / f"{repo}.packed.{target_length}.parquet"
    run_checked(
        repo,
        f"pack_{target_length}",
        [
            VENV_PYTHON, PACKER,
            "--input", tok,
            "--output", packed,
            "--target-length", str(target_length),
            "--strategy", "best_fit",
        ],
        log_path=work / f"{repo}.pack.{target_length}.log",
    )
    if not packed.exists() or packed.stat().st_size == 0:
        raise RepoFailure(repo, f"pack_{target_length}", f"empty packed parquet: {packed}")
    return packed


def append_output(repo: str, packed: Path, target_length: int) -> dict:
    """Place packed parquet at outputs/reindexed/<L>/<repo>.parquet.

    Per-repo files are the append unit (one file per repo per length); this keeps
    resume trivial and avoids rewriting a growing combined file. Returns row/token
    stats read back from the written file.
    """
    out_dir = OUTPUT_ROOT / str(target_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{repo}.parquet"
    shutil.copyfile(packed, dest)
    return _parquet_stats(dest, target_length)


def _parquet_stats(path: Path, target_length: int) -> dict:
    """Read rows/tokens/padding from a packed parquet via the venv python."""
    code = (
        "import json,sys; import pyarrow.parquet as pq;"
        "t=pq.read_table(sys.argv[1]);"
        "n=t.num_rows;"
        "vtc=t.column('valid_token_count').to_pylist() if 'valid_token_count' in t.column_names else [];"
        "tl=int(sys.argv[2]);"
        "tot=n*tl;"
        "valid=sum(vtc);"
        "pad=tot-valid;"
        "print(json.dumps({'rows':n,'capacity_tokens':tot,'valid_tokens':valid,"
        "'pad_tokens':pad,'pad_frac':(pad/tot if tot else 0.0)}))"
    )
    proc = subprocess.run(
        [str(VENV_PYTHON), "-c", code, str(path), str(target_length)],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"stats read failed for {path}: {proc.stderr}")
    return json.loads(proc.stdout.strip())


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def process_one_repo(
    repo: str,
    repo_dir: Path,
    target_lengths: Sequence[int],
    work_root: Path,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    global_symbol_index: Path | None = None,
    memory_limit_gb: float = 10.0,
    parse_workers: int = 2,
    index_timeout_s: int | None = None,
    index_stall_timeout_s: int | None = None,
    *,
    promote_dedup_on_success: bool = True,
) -> dict:
    """Index a repo, then ROUTE each code doc to exactly ONE length bucket.

    Corrected design (route-by-fit, carried over from the commits driver):
    materialize the repo's tokenized docs ONCE, then split rows by whole-doc
    token count into the smallest fitting length bucket. Docs longer than the
    largest length are explicitly dropped from fixed-shape parquet output and
    reported by the router instead of being written as invalid oversized rows.
    Each routed bucket is packed independently and landed at
    outputs/reindexed/<L>/<repo>.parquet. A given doc therefore appears in
    exactly one bucket -- never replicated across 1024/2048/4096.
    """
    work = work_root / repo
    work.mkdir(parents=True, exist_ok=True)
    lengths_sorted = sorted(int(x) for x in target_lengths)
    stage_id = code_stage_id(repo) if dedup_db is not None else None
    stage_db = code_stage_db(work, repo) if dedup_db is not None else None
    promoted = False
    success_without_promote = False
    try:
        timings: dict[str, float] = {}
        started = time.monotonic()
        enriched = stage_index_source(
            repo, repo_dir, work, dedup_db, dedup_near,
            stage_id, stage_db, global_symbol_index, memory_limit_gb,
            parse_workers, index_timeout_s, index_stall_timeout_s,
        )
        timings["index_project_s"] = round(time.monotonic() - started, 6)
        started = time.monotonic()
        tok = stage_materialize(repo, enriched, work, memory_limit_gb)
        timings["materialize_s"] = round(time.monotonic() - started, 6)

        started = time.monotonic()
        _bucket_for, route_by_fit = _route_by_fit_impl()
        route_dir = work / "routed"
        routed = route_by_fit(tok, lengths_sorted, route_dir)
        if not routed:
            raise RepoFailure(repo, "route_by_fit", f"no docs routed for {repo}")
        timings["route_by_fit_s"] = round(time.monotonic() - started, 6)

        per_length: dict[str, dict] = {}
        started = time.monotonic()
        for L, route_parquet in sorted(routed.items()):
            packed = stage_pack(repo, route_parquet, L, work)
            per_length[str(L)] = append_output(repo, packed, L)
        timings["pack_s"] = round(time.monotonic() - started, 6)
        if promote_dedup_on_success:
            timings.update(promote_dedup_stage(dedup_db, stage_id, stage_db))
            promoted = True
        else:
            success_without_promote = True
        return {"source": "code", "lengths": per_length, "stage_timings_s": timings}
    finally:
        if not promoted and not success_without_promote:
            discard_dedup_stage(dedup_db, stage_id, stage_db)


def process_one_commit_source(
    key: str,
    commit_inputs: Sequence[Path],
    target_lengths: Sequence[int],
    work_root: Path,
    repo_root: Path | None,
    repo_dir: Path | None,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    memory_limit_gb: float = 10.0,
    analysis_cache_entries: int = 128,
) -> dict:
    work = work_root / key
    work.mkdir(parents=True, exist_ok=True)
    lengths_sorted = sorted(int(x) for x in target_lengths)
    stage_id = commit_stage_id(key) if dedup_db is not None else None
    stage_db = commit_stage_db(work, key) if dedup_db is not None else None
    promoted = False
    try:
        timings: dict[str, float] = {}
        started = time.monotonic()
        enriched = stage_index_commits(
            key, commit_inputs, work, repo_root, repo_dir,
            dedup_db, dedup_near, stage_id,
            stage_db,
            memory_limit_gb=memory_limit_gb,
            analysis_cache_entries=analysis_cache_entries,
        )
        timings["process_commits_s"] = round(time.monotonic() - started, 6)
        if enriched is None:
            return {"source": "commits", "lengths": {}, "stage_timings_s": timings}
        started = time.monotonic()
        tok = stage_materialize(key, enriched, work, memory_limit_gb)
        timings["materialize_s"] = round(time.monotonic() - started, 6)

        started = time.monotonic()
        _bucket_for, route_by_fit = _route_by_fit_impl()
        route_dir = work / "routed"
        routed = route_by_fit(tok, lengths_sorted, route_dir)
        if not routed:
            raise RepoFailure(key, "route_by_fit", f"no docs routed for {key}")
        timings["route_by_fit_s"] = round(time.monotonic() - started, 6)

        per_length: dict[str, dict] = {}
        started = time.monotonic()
        for L, route_parquet in sorted(routed.items()):
            packed = stage_pack(key, route_parquet, L, work)
            per_length[str(L)] = append_output(key, packed, L)
        timings["pack_s"] = round(time.monotonic() - started, 6)
        timings.update(promote_dedup_stage(dedup_db, stage_id, stage_db))
        promoted = True
        return {"source": "commits", "lengths": per_length, "stage_timings_s": timings}
    finally:
        if not promoted:
            discard_dedup_stage(dedup_db, stage_id, stage_db)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-lengths", default="1024,2048,4096",
                   help="Comma-separated packed lengths (default: 1024,2048,4096).")
    p.add_argument("--max-repos", type=int, default=None,
                   help="Process at most N repos this run (after resume filtering).")
    p.add_argument("--token-budget", type=int, default=None,
                   help="Stop after the cumulative valid-token count (summed over "
                        "the smallest target length) reaches this many tokens.")
    p.add_argument("--resume", action="store_true",
                   help="Skip repos already marked done in _done.json (default on; "
                        "flag kept for explicitness).")
    p.add_argument("--no-resume", action="store_true",
                   help="Reprocess even repos already marked done.")
    p.add_argument("--commit-source", action="append", default=[],
                   help="key=path[,path2,...] commit JSONL source re-indexed via "
                        "process_commits.py. Repeatable.")
    p.add_argument("--commit-repo-root", default=None,
                   help="Single repo root passed to process_commits --repo-root.")
    p.add_argument("--commit-repo-dir", default=None,
                   help="Parent dir of repos passed to process_commits --repo-dir.")
    p.add_argument("--keep-temp", action="store_true",
                   help="Do not delete per-repo temp dirs (debugging).")
    p.add_argument("--work-dir", default=None,
                   help="Staging root (default: a fresh mkdtemp under the system "
                        "temp dir).")
    p.add_argument("--dedup-db", default=None,
                   help="Path to the SHARED global dedup SQLite store. Function-level "
                        "tokenized-hash dedup (code) and commit-doc tokenized-hash "
                        "dedup (commits) both write to this ONE db, so dedup is "
                        "global + resumable across repos AND across the code+commit "
                        "streams. Pass the SAME path to streaming_reindex_commits.py.")
    p.add_argument("--no-near-dedup", action="store_true",
                   help="Disable MinHash-LSH near dedup (exact-only).")
    p.add_argument("--global-symbol-index", default=None,
                   help="Path to the GLOBAL cross-repo base-lib symbol SQLite store "
                        "(built by scripts/crossrepo/build_global_symbol_index.py). "
                        "When set, the CODE stage threads it into index_project so "
                        "unresolved base-lib callees are pulled in as bounded "
                        "depth-1 deps tagged crosslib:<repo>. DEFAULT off.")
    p.add_argument("--memory-limit-gb", type=float, default=10.0,
                   help="Per-stage fail-loud RSS limit passed to index/materialize/"
                        "commit processors (default 10.0).")
    p.add_argument("--analysis-cache-entries", type=int, default=128,
                   help="Bounded per-process LRU entries passed to "
                        "process_commits.py for commit-source runs. Default 128; "
                        "use 0 to disable.")
    p.add_argument("--parse-workers", type=int, default=2,
                   help="Parse workers passed to index_project for the code stage "
                        "(default 2; keep this low when multiple repos run).")
    p.add_argument("--code-index-timeout-s", type=int, default=0,
                   help="Optional fail-loud timeout for each index_project code "
                        "stage. 0 disables the timeout (default).")
    p.add_argument("--code-index-stall-timeout-s", type=int, default=0,
                   help="Optional fail-loud stall watchdog for index_project: "
                        "kill when the log file has no size/mtime progress for "
                        "this many seconds. 0 disables the watchdog (default).")
    p.add_argument("--source-cache-dir", default=None,
                   help="Optional repo-level source cache for code-only runs. "
                        "Complete cached repos are reused before opening the "
                        "source tarball; uncached repos are materialized into "
                        "the cache and marked with a completion sentinel.")
    p.add_argument("--source-cache-only", action="store_true",
                   help="Only process repos already complete in --source-cache-dir; "
                        "do not open/decompress the source tarball.")
    p.add_argument("--source-dir-root", action="append", default=[],
                   help="Already-extracted repo root to process directly, without "
                        "opening the source tarball. May be passed multiple times; "
                        "children named <repo>/_src or <repo> are treated as repo dirs.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    target_lengths = [int(x) for x in args.target_lengths.split(",") if x.strip()]
    if not target_lengths:
        raise SystemExit("--target-lengths produced no lengths")
    smallest = min(target_lengths)

    for path in (VENV_PYTHON, TOKENIZER_PATH, INDEX_PROJECT, PROCESS_COMMITS,
                 MATERIALIZER, PACKER):
        if not Path(path).exists():
            raise SystemExit(f"required path missing: {path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for tl in target_lengths:
        (OUTPUT_ROOT / str(tl)).mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(MANIFEST_PATH)
    resume = not args.no_resume

    # Shared global dedup db (cross-repo + cross-stream). FAIL LOUD up front:
    # open it once here so a bad path / missing datasketch crashes before any
    # work (RULE #1). Each per-repo subprocess reopens against the same WAL db.
    dedup_db = Path(args.dedup_db) if args.dedup_db else None
    dedup_near = not args.no_near_dedup
    if dedup_db is not None:
        sys.path.insert(0, str(MLX_ROOT / "tools" / "clang_indexer"))
        from dedup_store import DedupStore  # noqa: E402
        dedup_db.parent.mkdir(parents=True, exist_ok=True)
        # Path/schema validation only; avoid rebuilding persisted MinHash/LSH in
        # the driver parent before any repo work starts.
        DedupStore(str(dedup_db), near=False, commit_every=1000).close()
        print(f"Dedup: SHARED global store at {dedup_db} "
              f"(exact{'+near' if dedup_near else ''}, tokenized hash)",
              file=sys.stderr)

    # Optional cross-repo base-lib symbol index. FAIL LOUD if given but missing.
    global_symbol_index = Path(args.global_symbol_index) if args.global_symbol_index else None
    if global_symbol_index is not None:
        if not global_symbol_index.exists():
            raise SystemExit(f"--global-symbol-index not found: {global_symbol_index}")
        print(f"Cross-lib: GLOBAL base-lib symbol index at {global_symbol_index} "
              f"threaded into CODE stage (bounded depth-1 pulls).", file=sys.stderr)

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        own_work_root = False
    else:
        work_root = Path(tempfile.mkdtemp(prefix="streaming_reindex_"))
        own_work_root = True

    processed = 0
    cumulative_valid = 0
    run_report: dict[str, dict] = {}

    # ----- commit sources first (independent of tarball extraction) -----
    commit_sources: list[tuple[str, list[Path]]] = []
    for spec in args.commit_source:
        if "=" not in spec:
            raise SystemExit(f"--commit-source must be key=path,...; got: {spec}")
        key, paths = spec.split("=", 1)
        files = [Path(p) for p in paths.split(",") if p.strip()]
        for f in files:
            if not f.exists():
                raise SystemExit(f"commit source file missing: {f}")
        commit_sources.append((key.strip(), files))

    for key, files in commit_sources:
        manifest_key = f"commits:{key}"
        if resume and manifest.is_done(manifest_key):
            print(f"SKIP (done) {manifest_key}", file=sys.stderr)
            continue
        if args.max_repos is not None and processed >= args.max_repos:
            break
        try:
            info = process_one_commit_source(
                key, files, target_lengths, work_root,
                Path(args.commit_repo_root) if args.commit_repo_root else None,
                Path(args.commit_repo_dir) if args.commit_repo_dir else None,
                dedup_db, dedup_near, args.memory_limit_gb,
                args.analysis_cache_entries,
            )
            manifest.mark_done(manifest_key, info)
            run_report[manifest_key] = info
            processed += 1
            # KeyError-safe: with route-by-fit a source may have NO doc in the
            # smallest bucket. Mirror streaming_reindex_commits.py:466.
            cumulative_valid += info["lengths"].get(str(smallest), {}).get("valid_tokens", 0)
            if not args.keep_temp:
                shutil.rmtree(work_root / key, ignore_errors=True)
        except RepoFailure as exc:
            print(f"FAIL {manifest_key}: {exc}", file=sys.stderr)
            manifest.mark_failed(manifest_key, exc.stage, exc.detail)
        if args.token_budget is not None and cumulative_valid >= args.token_budget:
            print(f"Token budget {args.token_budget} reached.", file=sys.stderr)

    # ----- code repos from the tarball: TRUE single streaming pass -----
    budget_reached = (
        args.token_budget is not None and cumulative_valid >= args.token_budget
    )
    cap_reached = args.max_repos is not None and processed >= args.max_repos

    if not budget_reached and not cap_reached:
        def should_process(repo: str) -> bool:
            return is_code_worktree_repo(repo) and not (resume and manifest.is_done(repo))

        source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
        source_dir_roots = [Path(p) for p in args.source_dir_root]
        if args.source_cache_only and source_cache_dir is None:
            raise SystemExit("--source-cache-only requires --source-cache-dir")
        if source_dir_roots and (source_cache_dir is not None or args.source_cache_only):
            raise SystemExit("--source-dir-root cannot be combined with source cache flags")
        gen = (
            stream_repo_dirs(source_dir_roots, should_process)
            if source_dir_roots
            else stream_repo_subtrees(
                work_root,
                should_process,
                source_cache_dir=source_cache_dir,
                source_cache_only=args.source_cache_only,
            )
        )
        try:
            for repo, repo_dir in gen:
                try:
                    info = process_one_repo(repo, repo_dir, target_lengths, work_root,
                                            dedup_db, dedup_near,
                                            global_symbol_index,
                                            args.memory_limit_gb,
                                            args.parse_workers,
                                            args.code_index_timeout_s,
                                            args.code_index_stall_timeout_s)
                    manifest.mark_done(repo, info)
                    run_report[repo] = info
                    processed += 1
                    # KeyError-safe smallest-bucket stats (route-by-fit may skip it).
                    added = info["lengths"].get(str(smallest), {}).get("valid_tokens", 0)
                    cumulative_valid += added
                    print(
                        f"DONE {repo}: +{added} tok @ {smallest} "
                        f"(cum {cumulative_valid}, repos {processed})",
                        file=sys.stderr, flush=True,
                    )
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

    # ----- final report -----
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
        "manifest": str(MANIFEST_PATH),
    }
    print(json.dumps(summary, indent=2))
    return 0 if not manifest.failed or processed > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
