#!/usr/bin/env python3
"""Fetch CI logs and extract compiler diagnostics for a batch of repos."""
import json
import os
import re
import subprocess
import sys
import time

REPOS = ["CGAL/cgal","ChibiOS/ChibiOS","Cisco-Talos/clamav","ClickHouse/ClickHouse",
"CrowCpp/Crow","DPDK/dpdk","Dao-AILab/flash-attention","DaveGamble/cJSON",
"DiligentGraphics/DiligentEngine","FFTW/fftw3","FreeCAD/FreeCAD","FreeRTOS/FreeRTOS",
"GNOME/libxml2","Geant4/geant4","HDFGroup/hdf5"]

OUT_DIR = "/Volumes/external/sources/cppmega.mlx/outputs/ci_diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# Diagnostic regex patterns
GCC_CLANG_RE = re.compile(r'^(?P<file>[^\s:][^:]*?):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>error|warning|fatal error):\s*(?P<msg>.+)$')
MSVC_RE = re.compile(r'^(?P<file>[^(]+)\((?P<line>\d+)\)\s*:\s*(?P<sev>error|warning)\s+(?P<code>[A-Z]\d+):\s*(?P<msg>.+)$')
LINK_RE = re.compile(r'(undefined reference to [`\']?(?P<sym>[^\']+)`?\'?|(?P<lnk>LNK2019|LNK2001):.*unresolved external symbol\s+(?P<sym2>\S+))')
CMAKE_RE = re.compile(r'CMake Error at (?P<file>[^:]+):(?P<line>\d+)')

def gh(args, timeout=120):
    """Run gh api, return (stdout, returncode)."""
    try:
        r = subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.returncode, r.stderr
    except subprocess.TimeoutExpired:
        return "", 1, "timeout"

def detect_platform(job_name, log_text):
    j = (job_name or "").lower()
    for p in ["ubuntu-24.04","ubuntu-22.04","ubuntu-20.04","ubuntu","windows","macos","macos-14","macos-13"]:
        if p in j:
            return p
    if log_text:
        m = re.search(r'(ubuntu-\d+\.\d+|windows-\d+|macos-\d+)', log_text[:5000])
        if m:
            return m.group(1)
    return "unknown"

def detect_compiler(log_text):
    t = log_text[:20000].lower()
    if "clang" in t:
        return "clang"
    if "gcc" in t or "g++" in t:
        return "gcc"
    if "cl.exe" in t or "msvc" in t:
        return "msvc"
    return "unknown"

def detect_build_command(log_text):
    patterns = [
        r'(cmake --build[^\n]{0,120})',
        r'(make(?: -j\d+)?(?:\s+\w+)?[^\n]{0,80})',
        r'(ninja[^\n]{0,80})',
        r'(msbuild[^\n]{0,120})',
        r'(meson compile[^\n]{0,80})',
    ]
    for p in patterns:
        m = re.search(p, log_text)
        if m:
            return m.group(1).strip()[:150]
    return ""

def parse_diagnostics(log_text, compiler_hint):
    diags = []
    seen = set()
    for line in log_text.splitlines():
        line = line.rstrip()
        # strip GH Actions timestamp prefix like "2024-01-01T00:00:00.0000000Z "
        line = re.sub(r'^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*', '', line)
        # GCC/Clang
        m = GCC_CLANG_RE.match(line)
        if m:
            sev = m.group('sev')
            if sev == "fatal error":
                sev = "error"
            key = (m.group('file'), m.group('line'), sev, m.group('msg')[:80])
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": m.group('file'),
                    "line": int(m.group('line')),
                    "col": int(m.group('col')),
                    "severity": sev,
                    "message": m.group('msg').strip()[:300],
                    "compiler": compiler_hint if compiler_hint != "unknown" else "gcc-clang",
                })
            continue
        # MSVC
        m = MSVC_RE.match(line)
        if m:
            key = (m.group('file'), m.group('line'), m.group('sev'), m.group('msg')[:80])
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": m.group('file'),
                    "line": int(m.group('line')),
                    "col": 0,
                    "severity": m.group('sev'),
                    "message": f"{m.group('code')}: {m.group('msg').strip()}"[:300],
                    "compiler": "msvc",
                })
            continue
        # Linker
        m = LINK_RE.search(line)
        if m:
            sym = m.group('sym') or m.group('sym2') or ""
            key = ("link", 0, "error", sym[:80])
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": "",
                    "line": 0,
                    "col": 0,
                    "severity": "error",
                    "message": line.strip()[:300],
                    "compiler": "linker",
                })
            continue
        # CMake
        m = CMAKE_RE.search(line)
        if m:
            key = ("cmake", m.group('file'), m.group('line'))
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": m.group('file'),
                    "line": int(m.group('line')),
                    "col": 0,
                    "severity": "error",
                    "message": line.strip()[:300],
                    "compiler": "cmake",
                })
    return diags[:200]  # cap

def process_repo(repo):
    safe_name = repo.replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"{safe_name}.jsonl")
    records = []
    notes = []

    # 1. fetch runs
    out, rc, err = gh(["api", f"repos/{repo}/actions/runs?per_page=10&status=completed",
                       "--jq", ".workflow_runs[] | {id, name, conclusion, head_sha}"])
    time.sleep(1)
    if rc != 0:
        notes.append(f"runs_fetch_failed: {err.strip()[:200]}")
        return records, notes

    runs = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not runs:
        notes.append("no_completed_runs")
        return records, notes

    # pick failed runs, else up to 3 most recent
    failed = [r for r in runs if r.get("conclusion") == "failure"]
    targets = failed if failed else runs[:3]
    if not failed:
        notes.append(f"no_failures_using_recent:{len(targets)}")
    # cap targets to 5 to limit API usage
    targets = targets[:5]

    for run in targets:
        run_id = run["id"]
        sha = run.get("head_sha", "")
        # jobs
        jout, jrc, jerr = gh(["api", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=20",
                              "--jq", ".jobs[] | {id, name, conclusion}"])
        time.sleep(1)
        if jrc != 0:
            notes.append(f"jobs_fetch_failed run={run_id}: {jerr.strip()[:150]}")
            continue
        jobs = []
        for line in jout.strip().splitlines():
            if not line.strip():
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # pick failed jobs, else build-like jobs
        failed_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
        if not failed_jobs:
            failed_jobs = [j for j in jobs if any(k in (j.get("name") or "").lower()
                           for k in ["build", "compile", "make", "cmake", "linux", "ubuntu", "gcc", "clang"])][:2]
        if not failed_jobs:
            failed_jobs = jobs[:1]

        for job in failed_jobs[:3]:
            job_id = job["id"]
            job_name = job.get("name", "")
            conclusion = job.get("conclusion", "")
            # fetch log
            lout, lrc, lerr = gh(["api", f"repos/{repo}/actions/jobs/{job_id}/logs"], timeout=180)
            time.sleep(1)
            if lrc != 0:
                emsg = lerr.strip()[:200]
                if "404" in emsg or "403" in emsg or "expired" in emsg.lower() or "Not Found" in emsg:
                    notes.append(f"log_expired job={job_id}")
                else:
                    notes.append(f"log_fetch_failed job={job_id}: {emsg}")
                # try annotations
                aout, arc, _ = gh(["api", f"repos/{repo}/check-runs/{job_id}/annotations"])
                time.sleep(1)
                diags = []
                if arc == 0 and aout.strip():
                    try:
                        anns = json.loads(aout)
                        for a in anns[:50]:
                            diags.append({
                                "file": a.get("path", ""),
                                "line": a.get("start_line", 0),
                                "col": a.get("start_column", 0) or 0,
                                "severity": a.get("annotation_level", "error"),
                                "message": (a.get("message", "") or "")[:300],
                                "compiler": "annotation",
                            })
                    except json.JSONDecodeError:
                        pass
                records.append({
                    "repo": repo, "run_id": run_id, "job_name": job_name,
                    "commit_sha": sha, "conclusion": conclusion,
                    "platform": detect_platform(job_name, ""),
                    "diagnostics": diags, "build_command": "",
                })
                continue

            # log fetched - if huge, it's already in memory; parse it
            compiler = detect_compiler(lout)
            diags = parse_diagnostics(lout, compiler)
            # also try annotations to supplement if no diags
            if not diags:
                aout, arc, _ = gh(["api", f"repos/{repo}/check-runs/{job_id}/annotations"])
                time.sleep(1)
                if arc == 0 and aout.strip():
                    try:
                        anns = json.loads(aout)
                        for a in anns[:50]:
                            diags.append({
                                "file": a.get("path", ""),
                                "line": a.get("start_line", 0),
                                "col": a.get("start_column", 0) or 0,
                                "severity": a.get("annotation_level", "error"),
                                "message": (a.get("message", "") or "")[:300],
                                "compiler": "annotation",
                            })
                    except json.JSONDecodeError:
                        pass

            records.append({
                "repo": repo, "run_id": run_id, "job_name": job_name,
                "commit_sha": sha, "conclusion": conclusion,
                "platform": detect_platform(job_name, lout),
                "diagnostics": diags,
                "build_command": detect_build_command(lout),
            })

    # write jsonl
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return records, notes

def main():
    summary = {"repos_processed": 0, "total_diagnostics": 0, "failures": []}
    for repo in REPOS:
        print(f"=== {repo} ===", flush=True)
        try:
            records, notes = process_repo(repo)
            summary["repos_processed"] += 1
            total = sum(len(r["diagnostics"]) for r in records)
            summary["total_diagnostics"] += total
            print(f"  records={len(records)} diagnostics={total} notes={notes}", flush=True)
            if notes:
                summary["failures"].append({"repo": repo, "notes": notes})
        except Exception as e:
            print(f"  EXCEPTION: {e}", flush=True)
            summary["failures"].append({"repo": repo, "notes": [f"exception: {e}"]})
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
