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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

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
) -> None:
    """Run a command; RAISE RepoFailure on any non-zero exit. No fallback."""
    printable = " ".join(str(c) for c in cmd)
    print(f"  [{repo}] {stage}: {printable}", file=sys.stderr, flush=True)
    log_fh = open(log_path, "wb") if log_path else None
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            env=_subprocess_env(),
            stdout=log_fh if log_fh else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RepoFailure(repo, stage, f"timed out after {timeout}s: {exc}") from exc
    finally:
        if log_fh:
            log_fh.close()
    if proc.returncode != 0:
        tail = ""
        if log_path and log_path.exists():
            data = log_path.read_bytes()[-4000:]
            tail = data.decode("utf-8", errors="replace")
        elif proc.stdout:
            tail = proc.stdout.decode("utf-8", errors="replace")[-4000:]
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


def stream_repo_subtrees(work_root: Path, should_process):
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
                continue  # a bare entry directly under cpp_all/ -- not a repo
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
# Per-repo pipeline stages.
# --------------------------------------------------------------------------- #
def stage_index_source(
    repo: str,
    repo_dir: Path,
    work: Path,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
    global_symbol_index: Path | None = None,
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
    ]
    if dedup_db is not None:
        cmd += ["--dedup-db", str(dedup_db)]
    if not dedup_near:
        cmd += ["--no-near-dedup"]
    if global_symbol_index is not None:
        cmd += ["--global-symbol-index", str(global_symbol_index)]
    run_checked(
        repo,
        "index_project",
        cmd,
        log_path=work / f"{repo}.index.log",
    )
    if not enriched.exists() or enriched.stat().st_size == 0:
        raise RepoFailure(repo, "index_project", f"empty enriched jsonl: {enriched}")
    return enriched


def stage_index_commits(repo: str, commit_inputs: Sequence[Path], work: Path,
                        repo_root: Path | None, repo_dir: Path | None,
                        dedup_db: Path | None = None,
                        dedup_near: bool = True,
                        pr_store: Path | None = None,
                        repo_list: Path | None = None) -> Path:
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
    ]
    if dedup_db is not None:
        cmd += ["--dedup-db", str(dedup_db)]
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
    if not enriched.exists() or enriched.stat().st_size == 0:
        raise RepoFailure(repo, "process_commits", f"empty enriched jsonl: {enriched}")
    return enriched


def stage_materialize(repo: str, enriched: Path, work: Path) -> Path:
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
) -> dict:
    """Index a repo, then ROUTE each code doc to exactly ONE length bucket.

    Corrected design (route-by-fit, carried over from the commits driver):
    materialize the repo's tokenized docs ONCE, then split rows by whole-doc
    token count into the smallest fitting length bucket (docs longer than the
    largest length get their own over-long row in the largest bucket). Each
    routed bucket is packed independently and landed at
    outputs/reindexed/<L>/<repo>.parquet. A given doc therefore appears in
    exactly one bucket -- never replicated across 1024/2048/4096.
    """
    work = work_root / repo
    work.mkdir(parents=True, exist_ok=True)
    lengths_sorted = sorted(int(x) for x in target_lengths)
    enriched = stage_index_source(repo, repo_dir, work, dedup_db, dedup_near,
                                  global_symbol_index)
    tok = stage_materialize(repo, enriched, work)

    _bucket_for, route_by_fit = _route_by_fit_impl()
    route_dir = work / "routed"
    routed = route_by_fit(tok, lengths_sorted, route_dir)
    if not routed:
        raise RepoFailure(repo, "route_by_fit", f"no docs routed for {repo}")

    per_length: dict[str, dict] = {}
    for L, route_parquet in sorted(routed.items()):
        packed = stage_pack(repo, route_parquet, L, work)
        per_length[str(L)] = append_output(repo, packed, L)
    return {"source": "code", "lengths": per_length}


def process_one_commit_source(
    key: str,
    commit_inputs: Sequence[Path],
    target_lengths: Sequence[int],
    work_root: Path,
    repo_root: Path | None,
    repo_dir: Path | None,
    dedup_db: Path | None = None,
    dedup_near: bool = True,
) -> dict:
    work = work_root / key
    work.mkdir(parents=True, exist_ok=True)
    lengths_sorted = sorted(int(x) for x in target_lengths)
    enriched = stage_index_commits(key, commit_inputs, work, repo_root, repo_dir,
                                   dedup_db, dedup_near)
    tok = stage_materialize(key, enriched, work)

    _bucket_for, route_by_fit = _route_by_fit_impl()
    route_dir = work / "routed"
    routed = route_by_fit(tok, lengths_sorted, route_dir)
    if not routed:
        raise RepoFailure(key, "route_by_fit", f"no docs routed for {key}")

    per_length: dict[str, dict] = {}
    for L, route_parquet in sorted(routed.items()):
        packed = stage_pack(key, route_parquet, L, work)
        per_length[str(L)] = append_output(key, packed, L)
    return {"source": "commits", "lengths": per_length}


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
        DedupStore(str(dedup_db), near=dedup_near, commit_every=1000).close()
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
                dedup_db, dedup_near,
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
            return not (resume and manifest.is_done(repo))

        gen = stream_repo_subtrees(work_root, should_process)
        try:
            for repo, repo_dir in gen:
                try:
                    info = process_one_repo(repo, repo_dir, target_lengths, work_root,
                                            dedup_db, dedup_near, global_symbol_index)
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
