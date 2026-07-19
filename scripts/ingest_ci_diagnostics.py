#!/usr/bin/env python3
"""Ingest CI build diagnostics from GitHub Actions logs.

Fetches recent completed CI runs for repos in our corpus, downloads job logs
for failed runs, and extracts structured compiler/linker diagnostic records
as JSONL for the training pipeline.

Sidecar Design (downstream conversion)
======================================
The JSONL records produced here are converted to training sidecars as follows:

1. token_diagnostic_edges: [error_token_start, error_token_end, source_file_identity_id]
   - Maps error location (file:line:col) to token positions in the indexed source.
   - Edge kind encodes severity:
       error=1, warning=2, link_error=3, cmake_error=4
   - For each diagnostic, tokenize the source file at commit_sha with our BPE
     tokenizer, map (line, col) -> token offset range.

2. token_build_edges: [command_token_start, command_token_end, target_identity_id]
   - From the build_command field: links compiler invocation to build target.
   - target_identity is derived from the CMake target or binary name.

3. token_cross_domain_edges: [diagnostic_start, diagnostic_end, fix_commit_identity]
   - When a subsequent commit in the same PR fixes the error, we link the
     diagnostic token span to the fix commit identity, enabling the model to
     learn error->fix associations.

Usage:
    # Dry run: list repos with available CI
    python3 scripts/ingest_ci_diagnostics.py --dry-run

    # Fetch diagnostics for specific repos
    python3 scripts/ingest_ci_diagnostics.py --repos SFML/SFML catchorg/Catch2

    # Fetch for first N repos from the list (testing)
    python3 scripts/ingest_ci_diagnostics.py --limit 3

    # Fetch for all repos (rate-limited, ~1s between calls)
    python3 scripts/ingest_ci_diagnostics.py

    # Limit runs per repo
    python3 scripts/ingest_ci_diagnostics.py --max-runs 5 --limit 10
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_LIST_PATH = Path("outputs/pr_ingest/repo_list.json")
OUTPUT_DIR = Path("outputs/ci_diagnostics")
DEFAULT_MAX_RUNS = 25
API_DELAY = 1.0  # seconds between API calls
LOG_DOWNLOAD_DELAY = 1.5  # extra delay for log downloads

# ---------------------------------------------------------------------------
# Diagnostic regex patterns
# ---------------------------------------------------------------------------

# GCC/Clang: path/file.cpp:123:45: error: message
RE_GCC_CLANG = re.compile(
    r"^(?P<file>[^\s:][^\s]*):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<severity>error|warning|fatal error|note):\s*(?P<message>.+)$"
)

# GCC/Clang without column: path/file.cpp:123: error: message
RE_GCC_CLANG_NOCOL = re.compile(
    r"^(?P<file>[^\s:][^\s]*):(?P<line>\d+):\s*"
    r"(?P<severity>error|warning|fatal error|note):\s*(?P<message>.+)$"
)

# MSVC: path/file.cpp(123): error C2065: message
RE_MSVC = re.compile(
    r"^(?P<file>[^\(]+)\((?P<line>\d+)(?:,(?P<col>\d+))?\):\s*"
    r"(?P<severity>error|warning|fatal error)\s+(?P<code>[A-Z]+\d+):\s*(?P<message>.+)$"
)

# Link errors (GCC/Clang): undefined reference to `symbol' or 'symbol'
RE_LINK_UNDEFINED_REF = re.compile(
    r"undefined reference to [`'\"](?P<symbol>[^'\"]+)['\"]?"
)

# Link errors (MSVC): LNK2019: unresolved external symbol
RE_LINK_LNK = re.compile(
    r"(?P<code>LNK\d+):\s*unresolved external symbol\s+\"?(?P<symbol>[^\"]+)\"?"
)

# CMake errors: CMake Error at path/CMakeLists.txt:123 (function):
RE_CMAKE_ERROR = re.compile(
    r"CMake Error at (?P<file>[^:]+):(?P<line>\d+)"
    r"(?:\s*\((?P<context>[^)]*)\))?:\s*(?P<message>.*)"
)

# CMake warnings
RE_CMAKE_WARNING = re.compile(
    r"CMake Warning(?: \(dev\))? at (?P<file>[^:]+):(?P<line>\d+)"
    r"(?:\s*\((?P<context>[^)]*)\))?:\s*(?P<message>.*)"
)

# Ninja: FAILED: target
RE_NINJA_FAILED = re.compile(
    r"^FAILED:\s*(?P<target>.+)$"
)

# Detect compiler from log context
RE_COMPILER_GCC = re.compile(r"\bgcc\b|\bg\+\+\b|\bcc1plus\b", re.IGNORECASE)
RE_COMPILER_CLANG = re.compile(r"\bclang\b|\bclang\+\+\b", re.IGNORECASE)
RE_COMPILER_MSVC = re.compile(r"\bcl\.exe\b|\bMSVC\b|\bVisual Studio\b", re.IGNORECASE)

# Detect platform from runner labels / log content
RE_PLATFORM_UBUNTU = re.compile(r"ubuntu[- ]?(\d+\.\d+)?", re.IGNORECASE)
RE_PLATFORM_MACOS = re.compile(r"macos[- ]?(\d+)?", re.IGNORECASE)
RE_PLATFORM_WINDOWS = re.compile(r"windows[- ]?(\d+)?", re.IGNORECASE)

# Detect build command
RE_BUILD_CMD = re.compile(
    r"(?:cmake\s+--build\s+\S+(?:\s+--target\s+\S+)*(?:\s+--\s+\S+)?"
    r"|(?:^|\s)make\s+(?:-j\d+\s+)?(?:-C\s+\S+\s+)?(?:-f\s+\S+\s+)?[A-Za-z_]\w*"
    r"|(?:^|\s)ninja\s+(?:-C\s+\S+\s+)?(?:-j\d+\s+)?[A-Za-z_]\S*"
    r"|msbuild\s+\S+(?:\s+/[^\s]+)*)"
)

# Detect compiler version strings (require dotted version or known prefix-N pattern)
RE_COMPILER_VERSION = re.compile(
    r"((?:clang|gcc|g\+\+)\s+version\s+\d+\.\d+(?:\.\d+)?"
    r"|(?:clang|gcc)-\d+(?:\.\d+)*"
    r"|(?:Apple\s+)?clang\s+version\s+\d+\.\d+)"
)


# ---------------------------------------------------------------------------
# GitHub API helpers (via gh CLI)
# ---------------------------------------------------------------------------

def gh_api(endpoint: str, accept: Optional[str] = None) -> Optional[dict | list]:
    """Call gh api and return parsed JSON, or None on failure."""
    cmd = ["gh", "api", endpoint]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if any(code in stderr for code in ("404", "Not Found", "410")):
                return None
            if "403" in stderr or "429" in stderr:
                print(f"  [RATE LIMITED] {endpoint}", file=sys.stderr)
                # Back off on rate limit
                time.sleep(5)
                return None
            print(f"  [gh api error] {endpoint}: {stderr[:200]}", file=sys.stderr)
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {endpoint}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        return None


def gh_api_raw(endpoint: str) -> Optional[str]:
    """Call gh api and return raw text output (for log downloads)."""
    cmd = ["gh", "api", endpoint]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if any(code in stderr for code in ("404", "410", "403", "429")):
                return None
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        return None


# ---------------------------------------------------------------------------
# CI log fetching
# ---------------------------------------------------------------------------

def fetch_completed_runs(owner_repo: str, max_runs: int, status_filter: str = "completed") -> list[dict]:
    """Fetch recent completed workflow runs for a repo."""
    endpoint = (
        f"repos/{owner_repo}/actions/runs"
        f"?per_page={min(max_runs, 100)}&status={status_filter}"
    )
    data = gh_api(endpoint)
    if not data or "workflow_runs" not in data:
        return []
    return data["workflow_runs"]


def fetch_jobs_for_run(owner_repo: str, run_id: int) -> list[dict]:
    """Fetch jobs for a specific workflow run."""
    endpoint = f"repos/{owner_repo}/actions/runs/{run_id}/jobs?per_page=30"
    data = gh_api(endpoint)
    if not data or "jobs" not in data:
        return []
    return data["jobs"]


def fetch_job_log(owner_repo: str, job_id: int) -> Optional[str]:
    """Download the raw log for a specific job."""
    endpoint = f"repos/{owner_repo}/actions/jobs/{job_id}/logs"
    return gh_api_raw(endpoint)


def fetch_annotations(owner_repo: str, check_run_id: int) -> list[dict]:
    """Fetch structured annotations for a check run."""
    endpoint = f"repos/{owner_repo}/check-runs/{check_run_id}/annotations"
    data = gh_api(endpoint)
    if not data or not isinstance(data, list):
        return []
    return data


# ---------------------------------------------------------------------------
# Diagnostic parsing
# ---------------------------------------------------------------------------

def detect_compiler(log_text: str) -> str:
    """Best-effort compiler detection from log content."""
    head = log_text[:8000]
    if RE_COMPILER_MSVC.search(head):
        return "msvc"
    if RE_COMPILER_CLANG.search(head):
        return "clang"
    if RE_COMPILER_GCC.search(head):
        return "gcc"
    return "unknown"


def detect_compiler_version(log_text: str) -> Optional[str]:
    """Try to extract a compiler version string."""
    m = RE_COMPILER_VERSION.search(log_text[:8000])
    return m.group(1).strip() if m else None


def detect_platform(job: dict, log_text: str) -> str:
    """Detect platform from job runner label or log content."""
    # Check runner labels first
    labels = job.get("labels", [])
    if isinstance(labels, list):
        for label in labels:
            if "ubuntu" in label.lower():
                return label
            if "macos" in label.lower():
                return label
            if "windows" in label.lower():
                return label
    # Fallback: check runner_name
    runner = job.get("runner_name", "")
    if runner:
        return runner
    # Fallback: detect from log
    head = log_text[:5000]
    m = RE_PLATFORM_UBUNTU.search(head)
    if m:
        return f"ubuntu-{m.group(1)}" if m.group(1) else "ubuntu"
    m = RE_PLATFORM_MACOS.search(head)
    if m:
        return f"macos-{m.group(1)}" if m.group(1) else "macos"
    m = RE_PLATFORM_WINDOWS.search(head)
    if m:
        return f"windows-{m.group(1)}" if m.group(1) else "windows"
    return "unknown"


def detect_build_command(log_text: str) -> Optional[str]:
    """Extract the build command from the log."""
    for line in log_text.splitlines():
        line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        # Strip timestamp prefix
        line_clean = re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*", "", line_clean)
        m = RE_BUILD_CMD.search(line_clean)
        if m:
            return m.group(0).strip()
    return None


def parse_diagnostics(log_text: str) -> list[dict]:
    """Extract structured diagnostics from a CI log."""
    diagnostics = []
    compiler = detect_compiler(log_text)
    seen = set()
    in_ninja_failure = False

    for line in log_text.splitlines():
        # Strip ANSI escape codes and GitHub Actions timestamp prefixes
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        line = re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*", "", line)
        line = re.sub(r"^##\[(?:error|warning|group|endgroup)\]\s*", "", line)
        line = line.strip()
        if not line:
            continue

        # Track Ninja FAILED blocks
        m_ninja = RE_NINJA_FAILED.match(line)
        if m_ninja:
            in_ninja_failure = True
            continue
        # A blank line or new command ends a ninja failure block
        if in_ninja_failure and (line.startswith("[") or line.startswith("$")):
            in_ninja_failure = False

        diag = None

        # GCC/Clang with column
        m = RE_GCC_CLANG.match(line)
        if m:
            sev = m.group("severity")
            if sev == "fatal error":
                sev = "error"
            elif sev == "note":
                # Include notes only inside ninja failure blocks
                if not in_ninja_failure:
                    continue
                sev = "note"
            diag = {
                "file": m.group("file"),
                "line": int(m.group("line")),
                "col": int(m.group("col")),
                "severity": sev,
                "message": m.group("message").strip(),
                "compiler": compiler,
            }
        else:
            # GCC/Clang without column
            m = RE_GCC_CLANG_NOCOL.match(line)
            if m:
                sev = m.group("severity")
                if sev == "fatal error":
                    sev = "error"
                elif sev == "note":
                    if not in_ninja_failure:
                        continue
                    sev = "note"
                diag = {
                    "file": m.group("file"),
                    "line": int(m.group("line")),
                    "col": None,
                    "severity": sev,
                    "message": m.group("message").strip(),
                    "compiler": compiler,
                }
            else:
                # MSVC
                m = RE_MSVC.match(line)
                if m:
                    sev = m.group("severity")
                    if sev == "fatal error":
                        sev = "error"
                    diag = {
                        "file": m.group("file").strip(),
                        "line": int(m.group("line")),
                        "col": int(m.group("col")) if m.group("col") else None,
                        "severity": sev,
                        "message": f"{m.group('code')}: {m.group('message').strip()}",
                        "compiler": "msvc",
                    }
                else:
                    # CMake Error
                    m = RE_CMAKE_ERROR.search(line)
                    if m:
                        diag = {
                            "file": m.group("file"),
                            "line": int(m.group("line")),
                            "col": None,
                            "severity": "error",
                            "message": m.group("message").strip() or f"CMake error in {m.group('context') or 'unknown'}",
                            "compiler": "cmake",
                        }
                    else:
                        # CMake Warning
                        m = RE_CMAKE_WARNING.search(line)
                        if m:
                            diag = {
                                "file": m.group("file"),
                                "line": int(m.group("line")),
                                "col": None,
                                "severity": "warning",
                                "message": m.group("message").strip() or f"CMake warning in {m.group('context') or 'unknown'}",
                                "compiler": "cmake",
                            }
                        else:
                            # Link errors: undefined reference
                            m = RE_LINK_UNDEFINED_REF.search(line)
                            if m:
                                diag = {
                                    "file": None,
                                    "line": None,
                                    "col": None,
                                    "severity": "link_error",
                                    "message": f"undefined reference to '{m.group('symbol')}'",
                                    "compiler": compiler,
                                }
                            else:
                                # Link errors: LNK2019
                                m = RE_LINK_LNK.search(line)
                                if m:
                                    diag = {
                                        "file": None,
                                        "line": None,
                                        "col": None,
                                        "severity": "link_error",
                                        "message": f"{m.group('code')}: unresolved external symbol \"{m.group('symbol')}\"",
                                        "compiler": "msvc",
                                    }

        if diag:
            # Dedup by (file, line, severity, message prefix)
            key = (diag["file"], diag["line"], diag["severity"], diag["message"][:120])
            if key not in seen:
                seen.add(key)
                diagnostics.append(diag)

    return diagnostics


def parse_annotations(annotations: list[dict]) -> list[dict]:
    """Convert GitHub check-run annotations to our diagnostic format."""
    diagnostics = []
    seen = set()
    for ann in annotations:
        sev_raw = ann.get("annotation_level", "notice")
        if sev_raw == "failure":
            severity = "error"
        elif sev_raw == "warning":
            severity = "warning"
        else:
            severity = "note"

        diag = {
            "file": ann.get("path"),
            "line": ann.get("start_line"),
            "col": ann.get("start_column"),
            "severity": severity,
            "message": ann.get("message", "").strip(),
            "compiler": "annotations",
        }
        key = (diag["file"], diag["line"], diag["severity"], diag["message"][:120])
        if key not in seen:
            seen.add(key)
            diagnostics.append(diag)
    return diagnostics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_repo_list(repo_list_path: Path) -> list[dict]:
    """Load repos from repo_list.json, deduplicating by owner_repo."""
    with open(repo_list_path) as f:
        data = json.load(f)
    repos = data.get("repos", [])
    gh_repos = [r for r in repos if r.get("owner_repo")]
    seen = set()
    unique = []
    for r in gh_repos:
        if r["owner_repo"] not in seen:
            seen.add(r["owner_repo"])
            unique.append(r)
    return unique


def process_repo(
    owner_repo: str,
    max_runs: int,
    output_dir: Path,
    dry_run: bool = False,
    api_delay: float = API_DELAY,
    log_delay: float = LOG_DOWNLOAD_DELAY,
) -> dict:
    """Process a single repo: fetch CI runs, parse diagnostics, write JSONL."""
    stats = {
        "repo": owner_repo,
        "runs_checked": 0,
        "failed_runs": 0,
        "jobs_with_logs": 0,
        "total_diagnostics": 0,
        "has_ci": False,
    }

    print(f"\n{'='*60}")
    print(f"Processing: {owner_repo}")
    print(f"{'='*60}")

    # Fetch completed runs (prioritize failures)
    runs = fetch_completed_runs(owner_repo, max_runs)
    time.sleep(api_delay)

    if not runs:
        print(f"  No completed CI runs found.")
        return stats

    stats["has_ci"] = True
    stats["runs_checked"] = len(runs)

    # Sort: failures first, then others (most recent within each group)
    failed_runs = [r for r in runs if r.get("conclusion") == "failure"]
    other_runs = [r for r in runs if r.get("conclusion") != "failure"]
    prioritized_runs = failed_runs + other_runs
    stats["failed_runs"] = len(failed_runs)

    if dry_run:
        print(f"  Found {len(runs)} completed runs ({len(failed_runs)} failed)")
        return stats

    if not failed_runs:
        print(f"  Found {len(runs)} completed runs, none failed. Skipping.")
        return stats

    print(f"  Found {len(runs)} completed runs, {len(failed_runs)} failed.")

    records = []

    for run in prioritized_runs:
        run_id = run["id"]
        workflow_name = run.get("name") or run.get("display_title") or "unknown"
        head_sha = run.get("head_sha", "")
        created_at = run.get("created_at", "")
        conclusion = run.get("conclusion", "")

        # For non-failed runs, only process if we haven't gotten enough from failures
        if conclusion != "failure" and len(records) >= 10:
            break

        print(f"  Run #{run_id} ({workflow_name}) [{conclusion}] sha={head_sha[:12]}...")

        # Fetch jobs
        jobs = fetch_jobs_for_run(owner_repo, run_id)
        time.sleep(api_delay)

        if not jobs:
            continue

        # Process failed jobs first, then others for warnings
        target_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
        if not target_jobs and conclusion != "failure":
            # For successful runs, look at all jobs for warnings
            target_jobs = jobs

        for job in target_jobs:
            job_id = job["id"]
            job_name = job.get("name", "unknown")
            job_conclusion = job.get("conclusion", "")

            print(f"    Job: {job_name} (id={job_id}) [{job_conclusion}]")

            # Try annotations endpoint first (structured data)
            check_run_id = job.get("check_run_id")
            annotation_diags = []
            if check_run_id:
                raw_annotations = fetch_annotations(owner_repo, check_run_id)
                time.sleep(api_delay)
                if raw_annotations:
                    annotation_diags = parse_annotations(raw_annotations)

            # Download raw log for full parsing
            log_text = fetch_job_log(owner_repo, job_id)
            time.sleep(log_delay)

            if not log_text and not annotation_diags:
                print(f"      [no log or annotations available]")
                continue

            stats["jobs_with_logs"] += 1

            # Parse diagnostics from log
            log_diags = parse_diagnostics(log_text) if log_text else []

            # Merge: log diagnostics + annotations (dedup by key)
            all_diags = log_diags[:]
            seen_keys = {(d["file"], d["line"], d["severity"], d["message"][:120]) for d in all_diags}
            for ad in annotation_diags:
                key = (ad["file"], ad["line"], ad["severity"], ad["message"][:120])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_diags.append(ad)

            if all_diags:
                # Detect metadata
                platform = detect_platform(job, log_text or "")
                build_cmd = detect_build_command(log_text) if log_text else None
                compiler_info = detect_compiler_version(log_text) if log_text else None
                if not compiler_info:
                    compiler_info = detect_compiler(log_text) if log_text else None

                record = {
                    "repo": owner_repo,
                    "run_id": run_id,
                    "job_id": job_id,
                    "commit_sha": head_sha,
                    "workflow": workflow_name,
                    "job_name": job_name,
                    "conclusion": job_conclusion,
                    "created_at": created_at,
                    "diagnostics": all_diags,
                    "build_command": build_cmd,
                    "platform": platform,
                    "compiler_info": compiler_info,
                }
                records.append(record)
                stats["total_diagnostics"] += len(all_diags)
                print(f"      Extracted {len(all_diags)} diagnostics "
                      f"(log={len(log_diags)}, annotations={len(annotation_diags)})")
            else:
                print(f"      No parseable diagnostics")

    # Write output
    if records:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = owner_repo.replace("/", "_")
        out_path = output_dir / f"{safe_name}.jsonl"
        with open(out_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n  Wrote {len(records)} records to {out_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fetch CI build diagnostics from GitHub Actions logs."
    )
    parser.add_argument(
        "--repo-list",
        type=Path,
        default=REPO_LIST_PATH,
        help="Path to repo_list.json",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        metavar="OWNER/REPO",
        help="Specific repos to process (owner/repo format). Overrides --repo-list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N repos from the list (for testing)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=DEFAULT_MAX_RUNS,
        help=f"Max CI runs to fetch per repo (default: {DEFAULT_MAX_RUNS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list repos with available CI logs, don't download",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=API_DELAY,
        help=f"Delay between API calls in seconds (default: {API_DELAY})",
    )
    args = parser.parse_args()

    # Determine repo list
    if args.repos:
        repos = [{"owner_repo": r} for r in args.repos]
        print(f"Processing {len(repos)} explicitly specified repos")
    else:
        if not args.repo_list.exists():
            print(f"Error: repo list not found at {args.repo_list}", file=sys.stderr)
            sys.exit(1)
        repos = load_repo_list(args.repo_list)
        print(f"Loaded {len(repos)} unique GitHub repos from {args.repo_list}")

    if args.limit:
        repos = repos[: args.limit]
        print(f"Limited to first {args.limit} repos")

    # Process repos
    all_stats = []
    for repo_info in repos:
        owner_repo = repo_info["owner_repo"]
        stats = process_repo(
            owner_repo,
            max_runs=args.max_runs,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            api_delay=args.delay,
        )
        all_stats.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_repos = len(all_stats)
    repos_with_ci = sum(1 for s in all_stats if s["has_ci"])
    total_diags = sum(s["total_diagnostics"] for s in all_stats)
    total_jobs = sum(s["jobs_with_logs"] for s in all_stats)
    total_records_jobs = sum(1 for s in all_stats if s["total_diagnostics"] > 0)
    print(f"  Repos processed:      {total_repos}")
    print(f"  Repos with CI:        {repos_with_ci}")
    print(f"  Repos with diags:     {total_records_jobs}")
    print(f"  Jobs with logs:       {total_jobs}")
    print(f"  Total diagnostics:    {total_diags}")
    if not args.dry_run and total_diags > 0:
        print(f"  Output directory:     {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
