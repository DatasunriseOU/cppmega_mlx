#!/usr/bin/env python3
"""Fetch CI logs and extract compiler diagnostics for a batch of repos."""

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
    r'^(?:.*?[/\\])?([A-Za-z0-9_./\\-]+\.(?:cpp|cc|cxx|c|h|hpp|hxx|C|S|s|asm)):(\d+):(\d+):\s*(error|warning|fatal error):\s*(.+)$'
)
MSVC_RE = re.compile(
    r'^(?:.*?[/\\])?([A-Za-z0-9_./\\ -]+\.(?:cpp|cc|cxx|c|h|hpp))\((\d+)\)\s*:\s*(error|warning)\s+(C\d+):\s*(.+)$'
)
LINK_RE = re.compile(
    r'(undefined reference to\s+[`\'](.+?)[`\']|LNK2019|LNK2001)'
)
CMAKE_RE = re.compile(
    r'^CMake Error at\s+(.+?):(\d+)'
)

def run_gh(args, timeout=60):
    """Run gh command and return stdout or None on failure."""
    cmd = ["gh"] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "404" in stderr or "403" in stderr or "410" in stderr:
                return None
            # Rate limit or other transient error
            if "rate limit" in stderr.lower() or "secondary rate" in stderr.lower():
                time.sleep(30)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result.returncode != 0:
                    return None
            else:
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
        # Strip ANSI codes and timestamps
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        line = re.sub(r'^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*', '', line)
        line = line.strip()
        
        # GCC/Clang pattern
        m = GCC_CLANG_RE.match(line)
        if m:
            file_path, lineno, col, severity, message = m.groups()
            compiler = "clang" if "clang" in line.lower() else "gcc"
            key = (file_path, lineno, col, severity, message[:80])
            if key not in seen:
                seen.add(key)
                diagnostics.append({
                    "file": file_path,
                    "line": int(lineno),
                    "col": int(col),
                    "severity": severity.replace("fatal error", "error"),
                    "message": message.strip(),
                    "compiler": compiler,
                })
            continue
        
        # MSVC pattern
        m = MSVC_RE.match(line)
        if m:
            file_path, lineno, severity, code, message = m.groups()
            key = (file_path, lineno, severity, code, message[:80])
            if key not in seen:
                seen.add(key)
                diagnostics.append({
                    "file": file_path,
                    "line": int(lineno),
                    "col": 0,
                    "severity": severity,
                    "message": f"{code}: {message.strip()}",
                    "compiler": "msvc",
                })
            continue
        
        # Linker errors
        m = LINK_RE.search(line)
        if m:
            msg = m.group(0).strip()
            key = ("link", 0, 0, "error", msg[:80])
            if key not in seen:
                seen.add(key)
                diagnostics.append({
                    "file": "",
                    "line": 0,
                    "col": 0,
                    "severity": "error",
                    "message": msg[:200],
                    "compiler": "linker",
                })
            continue
        
        # CMake errors
        m = CMAKE_RE.match(line)
        if m:
            file_path, lineno = m.groups()
            key = ("cmake", file_path, lineno)
            if key not in seen:
                seen.add(key)
                diagnostics.append({
                    "file": file_path,
                    "line": int(lineno),
                    "col": 0,
                    "severity": "error",
                    "message": line.strip()[:200],
                    "compiler": "cmake",
                })
    
    return diagnostics[:100]  # Cap at 100 diagnostics per job


def detect_platform(job_name, log_text):
    """Try to detect the platform/OS from job name or log."""
    name_lower = job_name.lower()
    log_lower = log_text[:5000].lower() if log_text else ""
    
    if "windows" in name_lower or "windows" in log_lower or "msvc" in name_lower:
        return "windows"
    elif "macos" in name_lower or "mac" in name_lower or "darwin" in log_lower:
        return "macos"
    elif "ubuntu-24" in log_lower:
        return "ubuntu-24.04"
    elif "ubuntu-22" in log_lower:
        return "ubuntu-22.04"
    elif "ubuntu" in name_lower or "linux" in name_lower or "ubuntu" in log_lower:
        return "ubuntu"
    return "unknown"


def detect_build_command(log_text):
    """Try to detect the build command from log."""
    if not log_text:
        return ""
    # Look for common build commands in first 10000 chars
    snippet = log_text[:10000]
    patterns = [
        r'(cmake --build\s+[^\n]+)',
        r'(make\s+-j\s*\d+)',
        r'(ninja\s+[^\n]*)',
        r'(msbuild\s+[^\n]+)',
    ]
    for pat in patterns:
        m = re.search(pat, snippet)
        if m:
            return m.group(1).strip()[:200]
    return ""


def process_repo(repo):
    """Process a single repo and return list of diagnostic records."""
    print(f"  Processing {repo}...", flush=True)
    records = []
    
    # Fetch last 10 completed runs
    output = run_gh([
        "api",
        f"repos/{repo}/actions/runs?per_page=10&status=completed",
        "--jq", ".workflow_runs[] | {id, name, conclusion, head_sha} | @json"
    ])
    
    if output is None:
        print(f"    WARNING: Could not fetch runs for {repo}", flush=True)
        return records
    
    runs = []
    for line in output.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            run_data = json.loads(line)
            runs.append(run_data)
        except json.JSONDecodeError:
            continue
    
    if not runs:
        print(f"    No completed runs found for {repo}", flush=True)
        return records
    
    # Select runs to inspect: failed ones, or up to 3 most recent
    failed_runs = [r for r in runs if r.get("conclusion") == "failure"]
    if failed_runs:
        target_runs = failed_runs[:5]  # Cap at 5 failed runs
    else:
        target_runs = runs[:3]
    
    for run_data in target_runs:
        run_id = run_data["id"]
        run_name = run_data.get("name", "")
        conclusion = run_data.get("conclusion", "")
        commit_sha = run_data.get("head_sha", "")
        
        time.sleep(1)  # Rate limiting
        
        # Get jobs for this run
        jobs_output = run_gh([
            "api",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=20",
            "--jq", ".jobs[] | {id, name, conclusion} | @json"
        ])
        
        if jobs_output is None:
            continue
        
        jobs = []
        for line in jobs_output.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        
        # Select failed/build jobs
        failed_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
        if not failed_jobs:
            # If no failures, take first 2 jobs
            failed_jobs = jobs[:2]
        
        for job in failed_jobs[:3]:  # Cap at 3 jobs per run
            job_id = job["id"]
            job_name = job.get("name", "")
            job_conclusion = job.get("conclusion", "")
            
            time.sleep(1)  # Rate limiting
            
            # Try annotations first (structured data)
            annotations_output = run_gh([
                "api",
                f"repos/{repo}/check-runs/{job_id}/annotations",
            ])
            
            diagnostics = []
            if annotations_output:
                try:
                    annotations = json.loads(annotations_output)
                    if isinstance(annotations, list):
                        for ann in annotations[:50]:
                            if ann.get("annotation_level") in ("failure", "warning"):
                                diagnostics.append({
                                    "file": ann.get("path", ""),
                                    "line": ann.get("start_line", 0),
                                    "col": ann.get("start_column", 0),
                                    "severity": "error" if ann.get("annotation_level") == "failure" else "warning",
                                    "message": ann.get("message", "")[:200],
                                    "compiler": "github-checks",
                                })
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # If no annotations, fetch logs
            if not diagnostics:
                log_output = run_gh([
                    "api",
                    f"repos/{repo}/actions/jobs/{job_id}/logs",
                ], timeout=90)
                
                if log_output:
                    diagnostics = parse_diagnostics(log_output)
                    platform = detect_platform(job_name, log_output)
                    build_cmd = detect_build_command(log_output)
                else:
                    platform = detect_platform(job_name, "")
                    build_cmd = ""
            else:
                platform = detect_platform(job_name, "")
                build_cmd = ""
            
            record = {
                "repo": repo,
                "run_id": run_id,
                "job_name": job_name,
                "commit_sha": commit_sha,
                "conclusion": job_conclusion,
                "platform": platform,
                "diagnostics": diagnostics,
                "build_command": build_cmd,
            }
            records.append(record)
    
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_diagnostics = 0
    repos_processed = 0
    failures = []
    
    for repo in REPOS:
        repo_name = repo.split("/")[-1]
        output_file = os.path.join(OUTPUT_DIR, f"{repo_name}.jsonl")
        
        try:
            records = process_repo(repo)
            repos_processed += 1
            
            with open(output_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
                    total_diagnostics += len(record.get("diagnostics", []))
            
            # Verify file
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"    Written: {output_file} ({size} bytes, {len(records)} records)", flush=True)
            else:
                failures.append(f"{repo}: output file not created")
                
        except Exception as e:
            failures.append(f"{repo}: {str(e)}")
            print(f"    ERROR processing {repo}: {e}", flush=True)
        
        time.sleep(1)  # Rate limiting between repos
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Repos processed: {repos_processed}/{len(REPOS)}")
    print(f"Total diagnostics found: {total_diagnostics}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("Failures: none")


if __name__ == "__main__":
    main()
