#!/usr/bin/env python3
"""Fetch CI logs from GitHub Actions and extract compiler diagnostics."""

import json
import os
import re
import subprocess
import sys
import time

REPOS = [
    "AMReX-Codes/amrex",
    "ARM-software/CMSIS-DSP",
    "ARM-software/CMSIS-NN",
    "ARM-software/CMSIS_5",
    "ARM-software/ComputeLibrary",
    "ARM-software/arm-trusted-firmware",
    "ARM-software/armnn",
    "ARMmbed/mbed-os",
    "AcademySoftwareFoundation/MaterialX",
    "AcademySoftwareFoundation/OpenShadingLanguage",
    "AcademySoftwareFoundation/openexr",
    "AcademySoftwareFoundation/openvdb",
    "ApolloAuto/apollo",
    "ArangoDB/arangodb",
    "ArduPilot/ardupilot",
]

OUTPUT_DIR = "/Volumes/external/sources/cppmega.mlx/outputs/ci_diagnostics"

# Diagnostic patterns
GCC_CLANG_RE = re.compile(
    r'(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning|fatal error):\s*(?P<message>.+)'
)
MSVC_RE = re.compile(
    r'(?P<file>[^\(]+)\((?P<line>\d+)\)\s*:\s*(?P<severity>error|warning)\s+(?P<code>C\d+):\s*(?P<message>.+)'
)
LINK_RE = re.compile(
    r'(undefined reference to .+|LNK2019:.+|LNK2001:.+)'
)
CMAKE_RE = re.compile(
    r'CMake Error at (?P<file>[^:]+):(?P<line>\d+)'
)


def run_gh(args, timeout=60):
    """Run gh command and return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "404" in stderr or "403" in stderr or "410" in stderr:
                return None
            # For log downloads, large output may cause issues
            if result.stdout:
                return result.stdout
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def parse_diagnostics(log_text):
    """Parse compiler diagnostics from log text."""
    diagnostics = []
    seen = set()

    for line in log_text.split('\n'):
        # GCC/Clang pattern
        m = GCC_CLANG_RE.search(line)
        if m:
            key = (m.group('file'), m.group('line'), m.group('col'), m.group('message')[:80])
            if key not in seen:
                seen.add(key)
                compiler = "clang" if "clang" in line.lower() else "gcc"
                diagnostics.append({
                    "file": m.group('file'),
                    "line": int(m.group('line')),
                    "col": int(m.group('col')),
                    "severity": m.group('severity'),
                    "message": m.group('message').strip()[:200],
                    "compiler": compiler,
                })
            continue

        # MSVC pattern
        m = MSVC_RE.search(line)
        if m:
            key = (m.group('file'), m.group('line'), m.group('message')[:80])
            if key not in seen:
                seen.add(key)
                diagnostics.append({
                    "file": m.group('file').strip(),
                    "line": int(m.group('line')),
                    "col": 0,
                    "severity": m.group('severity'),
                    "message": f"{m.group('code')}: {m.group('message').strip()[:200]}",
                    "compiler": "msvc",
                })
            continue

        # Linker errors
        m = LINK_RE.search(line)
        if m:
            msg = m.group(0).strip()[:200]
            key = ("linker", 0, 0, msg[:80])
            if key not in seen:
                seen.add(key)
                diagnostics.append({
                    "file": "",
                    "line": 0,
                    "col": 0,
                    "severity": "error",
                    "message": msg,
                    "compiler": "linker",
                })
            continue

        # CMake errors
        m = CMAKE_RE.search(line)
        if m:
            key = ("cmake", m.group('file'), m.group('line'))
            if key not in seen:
                seen.add(key)
                diagnostics.append({
                    "file": m.group('file'),
                    "line": int(m.group('line')),
                    "col": 0,
                    "severity": "error",
                    "message": line.strip()[:200],
                    "compiler": "cmake",
                })

    # Cap at 100 diagnostics per job to avoid huge files
    return diagnostics[:100]


def detect_platform(job_name, log_text=""):
    """Try to detect platform from job name or log content."""
    combined = (job_name + " " + log_text[:2000]).lower()
    if "ubuntu" in combined:
        for ver in ["24.04", "22.04", "20.04"]:
            if ver in combined:
                return f"ubuntu-{ver}"
        return "ubuntu"
    if "windows" in combined or "msvc" in combined or "win" in combined:
        return "windows"
    if "macos" in combined or "mac" in combined:
        return "macos"
    return "unknown"


def detect_build_command(log_text):
    """Try to detect build command from log."""
    patterns = [
        r'(cmake --build[^\n]{0,100})',
        r'(make -j\d+[^\n]{0,50})',
        r'(ninja[^\n]{0,50})',
        r'(msbuild[^\n]{0,100})',
    ]
    for pat in patterns:
        m = re.search(pat, log_text[:50000])
        if m:
            return m.group(1).strip()[:150]
    return ""


def process_repo(repo):
    """Process a single repo and return list of diagnostic records."""
    print(f"  Processing {repo}...", flush=True)
    records = []

    # Step 1: Fetch last 10 completed runs
    url = f'repos/{repo}/actions/runs?per_page=10&status=completed'
    out = run_gh(["api", url, "--jq",
                  '.workflow_runs[] | {id, name, conclusion, head_sha} | @json'])
    time.sleep(1)

    if not out:
        print(f"    No runs found or API error for {repo}", flush=True)
        return records

    runs = []
    for line in out.strip().split('\n'):
        line = line.strip()
        if line:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not runs:
        return records

    # Step 2: Select failed runs, or up to 3 most recent if none failed
    failed_runs = [r for r in runs if r.get("conclusion") == "failure"]
    if not failed_runs:
        failed_runs = runs[:3]

    # Limit to 3 runs max to control API usage
    failed_runs = failed_runs[:3]

    for run in failed_runs:
        run_id = run["id"]
        run_name = run.get("name", "")
        conclusion = run.get("conclusion", "")
        commit_sha = run.get("head_sha", "")

        # Get jobs
        jobs_url = f'repos/{repo}/actions/runs/{run_id}/jobs?per_page=20'
        jobs_out = run_gh(["api", jobs_url, "--jq",
                          '.jobs[] | {id, name, conclusion} | @json'])
        time.sleep(1)

        if not jobs_out:
            continue

        jobs = []
        for line in jobs_out.strip().split('\n'):
            line = line.strip()
            if line:
                try:
                    jobs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Find failed or build-related jobs
        target_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
        if not target_jobs:
            # Try build/compile jobs
            target_jobs = [j for j in jobs if any(
                kw in j.get("name", "").lower()
                for kw in ["build", "compile", "make", "cmake"]
            )]
        if not target_jobs:
            target_jobs = jobs[:2]

        # Limit to 3 jobs per run
        target_jobs = target_jobs[:3]

        for job in target_jobs:
            job_id = job["id"]
            job_name = job.get("name", "")
            job_conclusion = job.get("conclusion", "")

            # Try annotations first (structured data)
            annotations = []
            ann_out = run_gh(["api", f'repos/{repo}/check-runs/{job_id}/annotations'])
            time.sleep(1)
            if ann_out:
                try:
                    ann_list = json.loads(ann_out)
                    for ann in ann_list:
                        if ann.get("annotation_level") in ("failure", "warning"):
                            annotations.append({
                                "file": ann.get("path", ""),
                                "line": ann.get("start_line", 0),
                                "col": ann.get("start_column", 0),
                                "severity": "error" if ann["annotation_level"] == "failure" else "warning",
                                "message": ann.get("message", "")[:200],
                                "compiler": "github-check",
                            })
                except (json.JSONDecodeError, TypeError):
                    pass

            # Download and parse log
            diagnostics = list(annotations)
            build_cmd = ""
            platform = detect_platform(job_name)

            log_out = run_gh(["api", f'repos/{repo}/actions/jobs/{job_id}/logs'], timeout=90)
            time.sleep(1)

            if log_out:
                # For very large logs, only process lines with error/warning keywords
                if len(log_out) > 10_000_000:
                    filtered_lines = []
                    for l in log_out.split('\n'):
                        ll = l.lower()
                        if any(kw in ll for kw in ['error', 'warning', 'undefined reference', 'lnk', 'cmake error']):
                            filtered_lines.append(l)
                    log_out = '\n'.join(filtered_lines)

                parsed = parse_diagnostics(log_out)
                if parsed:
                    # Deduplicate with annotations
                    existing_keys = {(d["file"], d["line"], d["message"][:50]) for d in diagnostics}
                    for d in parsed:
                        key = (d["file"], d["line"], d["message"][:50])
                        if key not in existing_keys:
                            diagnostics.append(d)
                            existing_keys.add(key)

                if platform == "unknown":
                    platform = detect_platform(job_name, log_out[:5000])
                build_cmd = detect_build_command(log_out)

            if diagnostics or job_conclusion == "failure":
                record = {
                    "repo": repo,
                    "run_id": run_id,
                    "job_name": job_name,
                    "commit_sha": commit_sha,
                    "conclusion": job_conclusion or conclusion,
                    "platform": platform,
                    "diagnostics": diagnostics[:50],
                    "build_command": build_cmd,
                }
                records.append(record)

    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_repos = 0
    total_diagnostics = 0
    failures = []

    for repo in REPOS:
        repo_name = repo.split("/")[1]
        out_file = os.path.join(OUTPUT_DIR, f"{repo_name}.jsonl")

        try:
            records = process_repo(repo)
        except Exception as e:
            failures.append(f"{repo}: {e}")
            records = []

        # Write JSONL
        with open(out_file, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')

        # Verify
        if os.path.exists(out_file):
            size = os.path.getsize(out_file)
            n_diags = sum(len(r.get("diagnostics", [])) for r in records)
            total_diagnostics += n_diags
            total_repos += 1
            print(f"    -> {out_file} ({size} bytes, {len(records)} records, {n_diags} diagnostics)", flush=True)
        else:
            failures.append(f"{repo}: output file not created")

        time.sleep(1)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Repos processed: {total_repos}/{len(REPOS)}")
    print(f"Total diagnostics found: {total_diagnostics}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("Failures: none")


if __name__ == "__main__":
    main()
