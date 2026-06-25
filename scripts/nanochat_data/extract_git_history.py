"""Extract git commit history as raw JSONL for clang commit enrichment.

Extracts per-file commit diffs from git repos and outputs JSONL with:
  {old_content, new_content, diff, subject, body, filepath, repo}

The clang commit processor consumes these records, parses old/new source with
libclang, and produces enriched training documents.

Usage:
    # Extract raw commit data
    python3 scripts/data/extract_git_history.py \
        --repo ~/data/cpp_raw/opencv \
        --output /mnt/nvme/nanochat_data/opencv_commits.jsonl \
        --max_commits 50000

    # Process with clang wrapper
    python3 tools/clang_indexer/process_commits.py \
        --inputs opencv_commits.jsonl \
        --output opencv_training.jsonl --max-tokens 4096 --format both
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, TypedDict

# "Merge pull request #N from owner/source-branch" — GitHub merge-commit subject.
_MERGE_PR_RE = re.compile(
    r'Merge pull request #(\d+) from (\S+)',
)
# Trailing "(#N)" PR marker in a squash/merge subject.
_SUBJECT_PR_RE = re.compile(r'\(#(\d+)\)\s*$')
# Body trailers referencing a PR/issue.
_BODY_PR_RE = re.compile(
    r'(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)[ :]+#(\d+)',
    re.IGNORECASE,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.nanochat_data.memory_guard import check_memory_limit, start_memory_guard

class _ExtractionStats(TypedDict):
    repo: str
    commits_checked: int
    records_written: int


# C/C++ file extensions
CPP_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".h++",
    ".inl",
    ".inc",
    ".ipp",
    ".tcc",
    ".tpp",
}

# Files/paths to skip
SKIP_PATTERNS = {
    "test/",
    "tests/",
    "testing/",
    "unittest/",
    "benchmarks/",
    "third_party/",
    "3rdparty/",
    "vendor/",
    "external/",
    "deps/",
    "generated/",
    "auto_generated/",
    "cmake-build",
    ".pb.h",
    ".pb.cc",
    "_generated.h",
    ".gen.cc",
    ".gen.h",
}

MAX_DIFF_CHARS = 50000
MAX_FILES_PER_COMMIT = 5
MIN_DIFF_CHARS = 50


def stable_repo_id(repo_name: str) -> str:
    return hashlib.sha1(repo_name.encode("utf-8")).hexdigest()[:16]


def stable_filepath_id(repo_name: str, filepath: str) -> str:
    key = f"{repo_name}\0{filepath}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def is_cpp_file(path: str) -> bool:
    return Path(path).suffix.lower() in CPP_EXTENSIONS


def should_skip_path(path: str) -> bool:
    path_lower = path.lower()
    return any(p in path_lower for p in SKIP_PATTERNS)


def run_git(repo_path: str, args: list[str], timeout: int = 60) -> Optional[str]:
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


# GitHub remote URL -> owner/repo. Matches https://github.com/owner/repo(.git)
# and git@github.com:owner/repo(.git) forms.
_GH_REMOTE_RE = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


def resolve_repo_url(repo_path: str) -> Optional[str]:
    """Return the clone's raw ``remote.origin.url`` (or None when unset)."""
    url = run_git(repo_path, ["config", "--get", "remote.origin.url"])
    if not url:
        return None
    url = url.strip()
    return url or None


def resolve_owner_repo(repo_path: str) -> Optional[str]:
    """Resolve canonical ``owner/repo`` from the clone's git remote.origin.url.

    The authoritative owner/repo is NOT in git history — it lives only in the
    clone's remote. Returns ``owner/repo`` or None when the remote is missing /
    not a GitHub URL (caller decides whether to fall back to the bare name).
    """
    url = resolve_repo_url(repo_path)
    if not url:
        return None
    m = _GH_REMOTE_RE.search(url)
    if not m:
        return None
    return f"{m.group('owner')}/{m.group('repo')}"


def get_commit_list(repo_path: str, max_commits: int = 0) -> list[str]:
    args = ["log", "--format=%H", "--no-merges", "--diff-filter=M"]
    if max_commits > 0:
        args.extend(["-n", str(max_commits)])
    output = run_git(repo_path, args, timeout=120)
    if not output:
        return []
    return output.strip().split("\n")


def get_commit_info(repo_path: str, commit_hash: str) -> Optional[dict]:
    # NUL-delimit fields so multi-line trailers cannot collide with the
    # newline-delimited header fields. Order: hash, subject, body, parents,
    # author-date, commit-date, trailers(unfolded).
    fmt = (
        "%H%x00%s%x00%b%x00%P%x00%aI%x00%cI%x00"
        "%(trailers:only=true,unfold=true)"
    )
    output = run_git(repo_path, ["show", "-s", f"--format={fmt}", commit_hash])
    if not output:
        return None
    fields = output.split("\x00")
    if len(fields) < 7:
        return None
    commit_hash_out = fields[0]
    subject = fields[1]
    body = fields[2].strip()
    parent_hashes = [item for item in fields[3].strip().split() if item]
    author_timestamp = fields[4].strip() or None
    commit_timestamp = fields[5].strip() or None
    trailers = fields[6].strip()
    return {
        "hash": commit_hash_out,
        "subject": subject,
        "body": body,
        "trailers": trailers,
        "parent_hashes": parent_hashes,
        "parent_count": len(parent_hashes),
        "is_merge_commit": len(parent_hashes) > 1,
        "author_timestamp": author_timestamp,
        "commit_timestamp": commit_timestamp,
        "timestamp": commit_timestamp or author_timestamp,
    }


def parse_pr_number_from_text(subject: str, body: str) -> Optional[int]:
    """Parse a PR/issue number from the subject ``(#N)`` trailer or body trailers."""
    m = _SUBJECT_PR_RE.search(subject or "")
    if m:
        return int(m.group(1))
    m = _BODY_PR_RE.search(body or "")
    if m:
        return int(m.group(1))
    return None


def build_merge_pr_map(repo_path: str) -> dict[int, dict[str, str]]:
    """Mine merge commits for {pr_number -> {pr_title, source_branch}}.

    A SEPARATE ``git log --merges`` pass over 'Merge pull request #N from ...'
    subjects. Merge commits are NOT added as training records; this map lets a
    non-merge record carry the real PR title/source branch when its parsed
    pr_number matches.
    """
    out: dict[int, dict[str, str]] = {}
    fmt = "%x00%s%x00%b%x00"
    output = run_git(
        repo_path,
        ["log", "--merges", f"--format={fmt}"],
        timeout=180,
    )
    if not output:
        return out
    # Records are NUL-delimited triples (leading empty, subject, body) per commit.
    fields = output.split("\x00")
    # Walk in (subject, body) pairs: fields[1::3]=subject, fields[2::3]=body.
    for idx in range(1, len(fields) - 1, 3):
        subject = fields[idx]
        body = fields[idx + 1] if idx + 1 < len(fields) else ""
        m = _MERGE_PR_RE.search(subject)
        if not m:
            continue
        pr_number = int(m.group(1))
        source_branch = m.group(2)
        # The first non-empty body line is the squashed PR title in GitHub merges.
        pr_title = ""
        for line in body.splitlines():
            if line.strip():
                pr_title = line.strip()
                break
        out[pr_number] = {"pr_title": pr_title, "source_branch": source_branch}
    return out


def get_commit_note(repo_path: str, commit_hash: str) -> Optional[str]:
    """Best-effort Gerrit/code-review note text for a commit via ``git notes``.

    Reads from all note refs (``--notes=*``). Returns None when no note exists.
    Raises (via run_git returning None on git failure) are surfaced as absence;
    a present-but-empty note yields an empty string.
    """
    output = run_git(
        repo_path,
        ["log", "-1", "--notes=*", "--format=%N", commit_hash],
        timeout=30,
    )
    if output is None:
        return None
    text = output.strip()
    return text or None


def repo_has_notes(repo_path: str) -> bool:
    """True when the repo has any refs/notes/* (so note extraction is meaningful)."""
    output = run_git(repo_path, ["for-each-ref", "--format=%(refname)", "refs/notes/"])
    return bool(output and output.strip())


def compute_file_local_commit_indices(
    repo_path: str,
    commit_hashes: list[str],
) -> dict[tuple[str, str], int]:
    counters: dict[str, int] = {}
    indices: dict[tuple[str, str], int] = {}
    for commit_hash in reversed(commit_hashes):
        file_diffs = get_commit_diffs(repo_path, commit_hash)
        if not file_diffs:
            continue
        for item in file_diffs:
            filepath = item["filepath"]
            next_index = counters.get(filepath, 0)
            indices[(commit_hash, filepath)] = next_index
            counters[filepath] = next_index + 1
    return indices


def get_commit_diffs(repo_path: str, commit_hash: str) -> Optional[list[dict]]:
    """Get per-file diffs for C/C++ files in a commit."""
    name_status = run_git(
        repo_path, ["diff-tree", "--no-commit-id", "-r", "--name-status", commit_hash]
    )
    if not name_status:
        return None

    files = []
    for line in name_status.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]
        filepath = parts[-1]
        if not is_cpp_file(filepath):
            continue
        if should_skip_path(filepath):
            continue
        if status not in ("M",):
            continue
        files.append(filepath)

    if not files or len(files) > MAX_FILES_PER_COMMIT:
        return None

    results = []
    for filepath in files:
        old_content = run_git(
            repo_path, ["show", f"{commit_hash}^:{filepath}"], timeout=30
        )
        new_content = run_git(
            repo_path, ["show", f"{commit_hash}:{filepath}"], timeout=30
        )
        diff = run_git(
            repo_path,
            ["diff", f"{commit_hash}^", commit_hash, "--", filepath],
            timeout=30,
        )

        if old_content is None or new_content is None or diff is None:
            continue
        diff_len = len(diff)
        if diff_len < MIN_DIFF_CHARS or diff_len > MAX_DIFF_CHARS:
            continue
        if len(old_content) > 200000 or len(new_content) > 200000:
            continue

        results.append(
            {
                "filepath": filepath,
                "old_content": old_content,
                "new_content": new_content,
                "diff": diff,
            }
        )

    return results if results else None


def process_repo(
    repo_path: str,
    output_file,
    max_commits: int = 0,
    repo_name: str = "",
    memory_limit_gb: float = 10.0,
    *,
    notes: str = "auto",
) -> _ExtractionStats:
    if not repo_name:
        repo_name = Path(repo_path).name

    # Canonical owner/repo from the clone's git remote (authoritative; not in
    # git history). When resolvable it overrides the bare directory name so each
    # record carries the same key the Tier-2 PR store is keyed by. repo_url keeps
    # the raw remote so downstream consumers can reconstruct the canonical URL.
    repo_url = resolve_repo_url(repo_path)
    owner_repo = resolve_owner_repo(repo_path)
    if owner_repo:
        repo_name = owner_repo

    stats: _ExtractionStats = {
        "repo": repo_name,
        "commits_checked": 0,
        "records_written": 0,
    }

    # Merge-commit mining: per-repo {pr_number -> {pr_title, source_branch}}.
    merge_pr_map = build_merge_pr_map(repo_path)
    if merge_pr_map:
        print(f"  [{repo_name}] Mined {len(merge_pr_map):,} PR merge commits")

    # Gerrit/code-review note extraction. notes="off" disables; "on" requires
    # refs/notes to exist (fail loud if requested-but-missing); "auto" enables
    # only when the repo actually has notes.
    if notes == "off":
        use_notes = False
    elif notes == "on":
        if not repo_has_notes(repo_path):
            raise RuntimeError(
                f"[{repo_name}] --notes on requested but repo has no refs/notes/*"
            )
        use_notes = True
    elif notes == "auto":
        use_notes = repo_has_notes(repo_path)
    else:
        raise ValueError(f"Invalid --notes value: {notes!r}")
    if use_notes:
        print(f"  [{repo_name}] Gerrit/code-review notes enabled")

    depth_output = run_git(repo_path, ["rev-list", "--count", "HEAD"])
    if depth_output:
        commit_count = int(depth_output.strip())
        if commit_count <= 1:
            print(f"  [{repo_name}] Shallow clone (1 commit), skipping")
            return stats
        print(f"  [{repo_name}] {commit_count:,} commits available")
    else:
        print(f"  [{repo_name}] Cannot count commits, skipping")
        return stats

    commits = get_commit_list(repo_path, max_commits)
    if not commits:
        print(f"  [{repo_name}] No commits found")
        return stats

    file_local_commit_indices = compute_file_local_commit_indices(repo_path, commits)

    for i, commit_hash in enumerate(commits):
        if i > 0 and i % 1000 == 0:
            check_memory_limit(memory_limit_gb, label="extract_git_history")
            print(
                f"  [{repo_name}] Processed {i:,}/{len(commits):,} commits, "
                f"{stats['records_written']:,} records"
            )

        stats["commits_checked"] += 1

        commit_info = get_commit_info(repo_path, commit_hash)
        if not commit_info:
            continue

        subject = commit_info["subject"].lower()
        if any(
            skip in subject
            for skip in [
                "merge branch",
                "merge pull request",
                "update submodule",
                "bump version",
                "auto-generated",
                "clang-format",
                "fix whitespace",
                "fix typo in comment",
            ]
        ):
            continue

        file_diffs = get_commit_diffs(repo_path, commit_hash)
        if not file_diffs:
            continue

        # Resolve PR provenance: parse pr_number from subject/body, then attach
        # the real PR title / source branch from the merge map when known.
        pr_number = parse_pr_number_from_text(
            commit_info["subject"], commit_info["body"]
        )
        pr_title = ""
        source_branch = ""
        if pr_number is not None and pr_number in merge_pr_map:
            pr_title = merge_pr_map[pr_number]["pr_title"]
            source_branch = merge_pr_map[pr_number]["source_branch"]

        note_text = ""
        if use_notes:
            fetched_note = get_commit_note(repo_path, commit_hash)
            if fetched_note:
                note_text = fetched_note

        for fd in file_diffs:
            record = {
                "old_content": fd["old_content"],
                "new_content": fd["new_content"],
                "diff": fd["diff"],
                "subject": commit_info["subject"],
                "body": commit_info["body"],
                "filepath": fd["filepath"],
                "repo": repo_name,
                "repo_url": repo_url or "",
                "repo_path": os.path.abspath(repo_path),
                "commit_hash": commit_info["hash"],
                "timestamp": commit_info["timestamp"],
                "pr_number": pr_number,
                "pr_title": pr_title,
                "source_branch": source_branch,
                "trailers": commit_info.get("trailers", ""),
                "note_text": note_text,
                "parent_hashes": list(commit_info["parent_hashes"]),
                "parent_count": int(commit_info["parent_count"]),
                "is_merge_commit": bool(commit_info["is_merge_commit"]),
                "author_timestamp": commit_info["author_timestamp"],
                "commit_timestamp": commit_info["commit_timestamp"],
                "repo_stable_id": stable_repo_id(repo_name),
                "filepath_stable_id": stable_filepath_id(repo_name, fd["filepath"]),
                "file_local_commit_index": int(
                    file_local_commit_indices[(commit_hash, fd["filepath"])]
                ),
                "has_ambiguous_reconstruction": False,
                "has_rename_ambiguity": False,
            }
            output_file.write(json.dumps(record) + "\n")
            stats["records_written"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw git commit data as JSONL for clang commit enrichment"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="Path to a single git repository")
    group.add_argument("--repo_dir", help="Directory containing multiple repos")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--max_commits", type=int, default=0, help="Max commits per repo (0 = all)"
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=10.0,
        help="Abort if this Python wrapper exceeds this max RSS in GiB (default: 10).",
    )
    parser.add_argument(
        "--notes",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Gerrit/code-review note extraction: 'auto' (only when refs/notes "
            "exist), 'on' (require notes, fail loud if absent), 'off' (disable)."
        ),
    )
    args = parser.parse_args()
    start_memory_guard(args.memory_limit_gb, label="extract_git_history")

    repos = []
    if args.repo:
        repos.append(args.repo)
    else:
        for entry in sorted(os.listdir(args.repo_dir)):
            path = os.path.join(args.repo_dir, entry)
            if os.path.isdir(os.path.join(path, ".git")):
                repos.append(path)

    print(f"Found {len(repos)} repositories")
    print(f"Max commits per repo: {args.max_commits or 'all'}")
    print(f"Memory limit: {args.memory_limit_gb} GiB")
    print(f"Output: {args.output}")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_records = 0
    start_time = time.time()

    with open(args.output, "w") as f:
        for i, repo_path in enumerate(repos):
            repo_name = Path(repo_path).name
            print(f"[{i + 1}/{len(repos)}] {repo_name}...")

            try:
                stats = process_repo(
                    repo_path,
                    f,
                    args.max_commits,
                    repo_name,
                    args.memory_limit_gb,
                    notes=args.notes,
                )
                total_records += stats["records_written"]
                print(f"  [{repo_name}] {stats['records_written']:,} records")
            except Exception as e:
                print(f"  [{repo_name}] ERROR: {e}")

    elapsed = time.time() - start_time
    output_size = os.path.getsize(args.output)
    print("\n=== SUMMARY ===")
    print(f"Repos: {len(repos)}")
    print(f"Total records: {total_records:,}")
    print(f"Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"Output: {args.output} ({output_size / (1024**3):.2f} GB)")


if __name__ == "__main__":
    main()
