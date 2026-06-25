#!/usr/bin/env python3
"""Tier-2 PR ingest, component (1): build a repo_list of owner/repo for the corpus.

The git extractor (scripts/data/extract_git_history.py) records only the *bare*
repo name (the clone directory basename, e.g. ``opencv``). To join against GH
Archive / the GitHub GraphQL API we need the authoritative ``owner/repo`` slug.
The single source of truth for that is the clone's git remote:

    git -C <clone> config --get remote.origin.url

This script walks one or more directories of cloned repos (or an explicit list
of clone paths), resolves ``owner/repo`` from ``remote.origin.url`` for each,
and emits a JSON repo_list mapping the *bare* name (the parquet ``repo`` field)
to the canonical ``owner/repo`` plus the resolved clone path.

RULE #1 (fail loud): a clone whose owner/repo cannot be resolved is NOT silently
dropped. Every unresolved clone is collected and, unless ``--allow-unresolved``
is given, the script raises and lists them. There is no silent fallback.

Usage:
    # From one or more directories that each contain cloned repos:
    python3 scripts/pr_ingest/build_repo_list.py \
        --repo_dir ~/data/cpp_raw \
        --out /mnt/nvme/cppmega_data/pr_ingest/repo_list.json

    # From explicit clone paths:
    python3 scripts/pr_ingest/build_repo_list.py \
        --repo /path/to/opencv --repo /path/to/llvm-project \
        --out repo_list.json

    # Offline re-run: seed from / merge into a previously-stored map so clones
    # that are no longer on disk keep their resolved owner/repo:
    python3 scripts/pr_ingest/build_repo_list.py \
        --repo_dir ~/data/cpp_raw --stored_map prior_repo_list.json \
        --out repo_list.json

Output schema (repo_list.json):
    {
      "repos": [
        {"bare_name": "opencv", "owner_repo": "opencv/opencv",
         "owner": "opencv", "repo": "opencv",
         "remote_url": "git@github.com:opencv/opencv.git",
         "clone_path": "/abs/path/opencv", "source": "git_remote"}
      ],
      "by_bare_name": {"opencv": "opencv/opencv"},
      "repo_names": ["opencv/opencv"],     # ready for the BigQuery IN(...) list
      "unresolved": []
    }
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# git@github.com:owner/repo.git | https://github.com/owner/repo(.git) |
# ssh://git@github.com/owner/repo.git | git://github.com/owner/repo.git
_REMOTE_RE = re.compile(
    r"""(?:github\.com[:/])      # host + ':' (scp-style) or '/' (url-style)
        (?P<owner>[^/]+)/        # owner
        (?P<repo>[^/]+?)         # repo (non-greedy so we can drop trailing .git)
        (?:\.git)?/?$            # optional .git and trailing slash
    """,
    re.VERBOSE,
)


def run_git(repo_path: str, args: list[str], timeout: int = 30) -> Optional[str]:
    """Run ``git -C repo_path <args>``; return stdout or None on failure.

    Mirrors scripts/data/extract_git_history.py:run_git so resolution behaves
    identically to the extractor that produced the bare names.
    """
    cmd = ["git", "-C", repo_path] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_owner_repo(remote_url: str) -> Optional[tuple[str, str]]:
    """Parse ``(owner, repo)`` from a GitHub remote URL, or None if not GitHub."""
    if not remote_url:
        return None
    m = _REMOTE_RE.search(remote_url.strip())
    if not m:
        return None
    owner = m.group("owner").strip()
    repo = m.group("repo").strip()
    if not owner or not repo:
        return None
    return owner, repo


def resolve_clone(clone_path: str) -> dict:
    """Resolve a single clone to a repo_list entry (resolved or unresolved)."""
    bare_name = Path(clone_path).name
    if not os.path.isdir(os.path.join(clone_path, ".git")):
        return {
            "bare_name": bare_name,
            "clone_path": os.path.abspath(clone_path),
            "resolved": False,
            "reason": "no .git directory (not a clone)",
        }
    remote_url = run_git(clone_path, ["config", "--get", "remote.origin.url"])
    if remote_url is None:
        return {
            "bare_name": bare_name,
            "clone_path": os.path.abspath(clone_path),
            "resolved": False,
            "reason": "git config --get remote.origin.url failed (no origin?)",
        }
    remote_url = remote_url.strip()
    parsed = parse_owner_repo(remote_url)
    if parsed is None:
        return {
            "bare_name": bare_name,
            "clone_path": os.path.abspath(clone_path),
            "remote_url": remote_url,
            "resolved": False,
            "reason": f"could not parse owner/repo from remote url: {remote_url!r}",
        }
    owner, repo = parsed
    return {
        "bare_name": bare_name,
        "owner_repo": f"{owner}/{repo}",
        "owner": owner,
        "repo": repo,
        "remote_url": remote_url,
        "clone_path": os.path.abspath(clone_path),
        "source": "git_remote",
        "resolved": True,
    }


def discover_clones(repo_dirs: list[str], repos: list[str]) -> list[str]:
    """Return the list of clone paths from --repo_dir entries and --repo paths."""
    clone_paths: list[str] = []
    for rd in repo_dirs:
        if not os.path.isdir(rd):
            raise SystemExit(f"[build_repo_list] --repo_dir does not exist: {rd}")
        for entry in sorted(os.listdir(rd)):
            path = os.path.join(rd, entry)
            if os.path.isdir(os.path.join(path, ".git")):
                clone_paths.append(path)
    for r in repos:
        clone_paths.append(r)
    # De-dup preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in clone_paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def load_stored_map(path: Optional[str]) -> dict[str, dict]:
    """Load a prior repo_list.json into {bare_name: entry} for offline re-runs."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(f"[build_repo_list] --stored_map does not exist: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    out: dict[str, dict] = {}
    for entry in data.get("repos", []):
        if entry.get("owner_repo"):
            out[entry["bare_name"]] = entry
    return out


def build(
    repo_dirs: list[str],
    repos: list[str],
    stored_map_path: Optional[str],
    allow_unresolved: bool,
) -> dict:
    stored = load_stored_map(stored_map_path)
    clone_paths = discover_clones(repo_dirs, repos)
    if not clone_paths and not stored:
        raise SystemExit(
            "[build_repo_list] no clones found and no --stored_map given; nothing to do"
        )

    resolved: dict[str, dict] = {}
    unresolved: list[dict] = []

    for clone_path in clone_paths:
        entry = resolve_clone(clone_path)
        if entry.get("resolved"):
            entry.pop("resolved", None)
            resolved[entry["bare_name"]] = entry
        else:
            # Offline fallback that is NOT silent: only an EXPLICIT prior
            # resolution (from --stored_map) can rescue a clone we cannot read
            # now. We record that the value came from the stored map.
            bare = entry["bare_name"]
            if bare in stored:
                rescued = dict(stored[bare])
                rescued["source"] = "stored_map"
                rescued["clone_path"] = entry.get("clone_path", rescued.get("clone_path", ""))
                resolved[bare] = rescued
            else:
                unresolved.append(entry)

    # Merge in stored entries whose clones were not on disk at all this run.
    for bare, entry in stored.items():
        if bare not in resolved:
            merged = dict(entry)
            merged["source"] = "stored_map"
            resolved[bare] = merged

    repos_out = sorted(resolved.values(), key=lambda e: e["owner_repo"])
    by_bare = {e["bare_name"]: e["owner_repo"] for e in repos_out}
    repo_names = sorted({e["owner_repo"] for e in repos_out})

    result = {
        "repos": repos_out,
        "by_bare_name": by_bare,
        "repo_names": repo_names,
        "unresolved": unresolved,
    }

    if unresolved and not allow_unresolved:
        names = ", ".join(sorted(e["bare_name"] for e in unresolved))
        reasons = "\n".join(
            f"  - {e['bare_name']}: {e.get('reason', 'unknown')}" for e in unresolved
        )
        raise SystemExit(
            "[build_repo_list] FAILING LOUD: could not resolve owner/repo for "
            f"{len(unresolved)} clone(s): {names}\n{reasons}\n"
            "Fix the remotes (git remote set-url origin ...), provide a "
            "--stored_map with prior resolutions, or pass --allow-unresolved to "
            "proceed WITHOUT them (they will be excluded from the repo_list)."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build owner/repo repo_list for the corpus from git remotes."
    )
    parser.add_argument(
        "--repo_dir", action="append", default=[],
        help="Directory containing cloned repos (repeatable).",
    )
    parser.add_argument(
        "--repo", action="append", default=[],
        help="Explicit clone path (repeatable).",
    )
    parser.add_argument(
        "--stored_map",
        help="Prior repo_list.json to seed/merge from for offline re-runs.",
    )
    parser.add_argument("--out", required=True, help="Output repo_list.json path.")
    parser.add_argument(
        "--allow-unresolved", action="store_true",
        help="Do not fail on unresolved clones; exclude them (still listed).",
    )
    args = parser.parse_args()

    result = build(
        repo_dirs=args.repo_dir,
        repos=args.repo,
        stored_map_path=args.stored_map,
        allow_unresolved=args.allow_unresolved,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(
        f"[build_repo_list] resolved {len(result['repos'])} repo(s), "
        f"{len(result['unresolved'])} unresolved -> {args.out}",
        file=sys.stderr,
    )
    for e in result["repos"]:
        print(f"  {e['bare_name']:30s} -> {e['owner_repo']}", file=sys.stderr)


if __name__ == "__main__":
    main()
