"""Extract raw git commit data from ALL repos in parallel.

Runs extract_git_history.py as subprocesses across N repos simultaneously.
Each repo gets its own output JSONL file. After all repos finish, concatenates
into a single file.

Usage:
    python3 scripts/data/extract_all_commits.py \
        --repo_dir ~/data/cpp_raw \
        --output_dir ~/data/raw_commits_all \
        --workers 40
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.nanochat_data.memory_guard import start_memory_guard


def extract_repo(args_tuple):
    """Extract commits from a single repo (runs as subprocess)."""
    repo_path, output_file, script_path, memory_limit_gb = args_tuple
    repo_name = Path(repo_path).name

    # Check if output already exists and has content
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        if size > 0:
            # Count lines to verify it's not corrupt
            try:
                with open(output_file) as f:
                    lines = sum(1 for _ in f)
                if lines > 0:
                    return {
                        "repo": repo_name,
                        "records": lines,
                        "status": "skipped (already exists)",
                        "time": 0,
                    }
            except Exception:
                pass  # Re-extract if corrupt

    start = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable,
                script_path,
                "--repo",
                repo_path,
                "--output",
                output_file,
                "--memory-limit-gb",
                str(memory_limit_gb),
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per repo
        )

        elapsed = time.time() - start

        # Count output records
        records = 0
        if os.path.exists(output_file):
            with open(output_file) as f:
                records = sum(1 for _ in f)

        if result.returncode != 0:
            return {
                "repo": repo_name,
                "records": records,
                "status": f"error (rc={result.returncode})",
                "time": elapsed,
                "stderr": result.stderr[-500:] if result.stderr else "",
            }

        return {
            "repo": repo_name,
            "records": records,
            "status": "ok",
            "time": elapsed,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        records = 0
        if os.path.exists(output_file):
            with open(output_file) as f:
                records = sum(1 for _ in f)
        return {
            "repo": repo_name,
            "records": records,
            "status": "timeout (2h)",
            "time": elapsed,
        }
    except Exception as e:
        return {
            "repo": repo_name,
            "records": 0,
            "status": f"exception: {e}",
            "time": time.time() - start,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw git commits from all repos in parallel"
    )
    parser.add_argument(
        "--repo_dir", required=True, help="Directory containing git repos"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory for per-repo JSONL outputs"
    )
    parser.add_argument(
        "--workers", type=int, default=40, help="Parallel extraction workers"
    )
    parser.add_argument(
        "--concat_output",
        default="",
        help="If set, concatenate all per-repo files into this single JSONL",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Only process these repos (by name)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Skip these repos (by name)",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=10.0,
        help="Abort each Python wrapper process above this max RSS in GiB (default: 10).",
    )
    args = parser.parse_args()
    start_memory_guard(args.memory_limit_gb, label="extract_all_commits")

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "extract_git_history.py"
    )
    if not os.path.exists(script_path):
        print(f"ERROR: {script_path} not found")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all repos with .git directory (skip .bare repos)
    repos = []
    exclude_set = set(args.exclude or [])
    only_set = set(args.only) if args.only else None

    for entry in sorted(os.listdir(args.repo_dir)):
        if entry.endswith(".bare"):
            continue
        path = os.path.join(args.repo_dir, entry)
        if not os.path.isdir(path):
            continue
        git_dir = os.path.join(path, ".git")
        if not os.path.exists(git_dir):
            continue
        if entry in exclude_set:
            continue
        if only_set and entry not in only_set:
            continue
        repos.append(path)

    print(f"Found {len(repos)} repositories in {args.repo_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Memory limit per process: {args.memory_limit_gb} GiB")
    print()

    # Build task list
    tasks = []
    for repo_path in repos:
        repo_name = Path(repo_path).name
        output_file = os.path.join(args.output_dir, f"{repo_name}_commits.jsonl")
        tasks.append((repo_path, output_file, script_path, args.memory_limit_gb))

    # Run in parallel
    total_records = 0
    completed = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_repo, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            total_records += result["records"]

            status_icon = "OK" if result["status"] == "ok" else (
                "SKIP" if "skipped" in result["status"] else "ERR"
            )
            if "skipped" in result["status"]:
                skipped += 1
            elif result["status"] != "ok":
                errors += 1

            elapsed_total = time.time() - start_time
            print(
                f"[{completed}/{len(tasks)}] {status_icon} {result['repo']}: "
                f"{result['records']:,} records in {result['time']:.0f}s "
                f"({result['status']}) | total: {total_records:,} records, "
                f"{elapsed_total:.0f}s elapsed"
            )

    # Sort results by record count (descending) for the summary
    results.sort(key=lambda r: r["records"], reverse=True)

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Repos processed: {len(tasks)}")
    print(f"  Successful: {completed - errors - skipped}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")
    print(f"Total records: {total_records:,}")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}m)")

    # Top 20 repos by records
    print("\nTop 20 repos by record count:")
    for r in results[:20]:
        print(f"  {r['repo']:40s} {r['records']:>10,}")

    # Save stats
    stats_file = os.path.join(args.output_dir, "extraction_stats.json")
    with open(stats_file, "w") as f:
        json.dump(
            {
                "total_repos": len(tasks),
                "total_records": total_records,
                "errors": errors,
                "skipped": skipped,
                "elapsed_seconds": elapsed_total,
                "per_repo": results,
            },
            f,
            indent=2,
        )
    print(f"\nStats saved to: {stats_file}")

    # Concatenate if requested
    if args.concat_output:
        print(f"\nConcatenating to {args.concat_output}...")
        total_lines = 0
        with open(args.concat_output, "w") as out:
            for repo_path in repos:
                repo_name = Path(repo_path).name
                per_repo_file = os.path.join(
                    args.output_dir, f"{repo_name}_commits.jsonl"
                )
                if not os.path.exists(per_repo_file):
                    continue
                with open(per_repo_file) as inp:
                    for line in inp:
                        out.write(line)
                        total_lines += 1
        size_gb = os.path.getsize(args.concat_output) / (1024**3)
        print(f"Concatenated: {total_lines:,} records, {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
