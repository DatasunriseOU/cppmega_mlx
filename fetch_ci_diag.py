#!/usr/bin/env python3
"""Fetch CI logs from GitHub Actions and extract compiler diagnostics."""
import json, subprocess, sys, re, time, os

REPOS = [
    "CGAL/cgal","ChibiOS/ChibiOS","Cisco-Talos/clamav","ClickHouse/ClickHouse",
    "CrowCpp/Crow","DPDK/dpdk","Dao-AILab/flash-attention","DaveGamble/cJSON",
    "DiligentGraphics/DiligentEngine","FFTW/fftw3","FreeCAD/FreeCAD",
    "FreeRTOS/FreeRTOS","GNOME/libxml2","Geant4/geant4","HDFGroup/hdf5"
]

OUTDIR = "/Volumes/external/sources/cppmega.mlx/outputs/ci_diagnostics"
os.makedirs(OUTDIR, exist_ok=True)

# Diagnostic patterns
PATTERNS = [
    # GCC/Clang: path:line:col: error|warning: message
    (re.compile(r'^(.+?):(\d+):(\d+):\s*(error|warning|fatal error):\s*(.+)$'), 'gcc_clang'),
    # MSVC: path(line): error|warning Cxxxx: message
    (re.compile(r'^(.+?)\((\d+)\)\s*:\s*(error|warning)\s+(C\d+):\s*(.+)$'), 'msvc'),
    # Link errors
    (re.compile(r'undefined reference to [`\'](.+?)[`\']'), 'linker'),
    (re.compile(r'(LNK2019|LNK2001):\s*(.+)'), 'msvc_linker'),
    # CMake errors
    (re.compile(r'CMake Error at (.+?):(\d+)'), 'cmake'),
]

def run_gh(args, timeout=60):
    """Run gh command and return stdout or None on error."""
    try:
        r = subprocess.run(['gh'] + args, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None, r.stderr.strip()
        return r.stdout, None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)

def parse_diagnostics(log_text):
    """Parse log text for compiler diagnostics."""
    diags = []
    seen = set()
    for line in log_text.split('\n'):
        line = line.rstrip()
        if not line:
            continue
        # GCC/Clang pattern
        m = PATTERNS[0][0].match(line)
        if m:
            filepath, lineno, col, severity, msg = m.groups()
            # Skip noise
            if 'cc1plus' in filepath or filepath.startswith('<'):
                continue
            key = (filepath, lineno, col, msg[:80])
            if key not in seen:
                seen.add(key)
                compiler = 'clang' if 'clang' in line.lower() else 'gcc'
                diags.append({
                    "file": filepath, "line": int(lineno), "col": int(col),
                    "severity": severity.replace("fatal error", "error"),
                    "message": msg.strip()[:300], "compiler": compiler
                })
            continue
        # MSVC pattern
        m = PATTERNS[1][0].match(line)
        if m:
            filepath, lineno, severity, code, msg = m.groups()
            key = (filepath, lineno, code, msg[:80])
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": filepath, "line": int(lineno), "col": 0,
                    "severity": severity, "message": f"{code}: {msg.strip()[:280]}",
                    "compiler": "msvc"
                })
            continue
        # Linker: undefined reference
        m = PATTERNS[2][0].search(line)
        if m:
            key = ('linker', m.group(1)[:80])
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": "", "line": 0, "col": 0,
                    "severity": "error",
                    "message": f"undefined reference to `{m.group(1)}`",
                    "compiler": "ld"
                })
            continue
        # MSVC linker
        m = PATTERNS[3][0].search(line)
        if m:
            key = ('msvc_link', m.group(1), m.group(2)[:60])
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": "", "line": 0, "col": 0,
                    "severity": "error",
                    "message": f"{m.group(1)}: {m.group(2).strip()[:280]}",
                    "compiler": "msvc"
                })
            continue
        # CMake error
        m = PATTERNS[4][0].search(line)
        if m:
            key = ('cmake', m.group(1), m.group(2))
            if key not in seen:
                seen.add(key)
                diags.append({
                    "file": m.group(1), "line": int(m.group(2)), "col": 0,
                    "severity": "error",
                    "message": line.strip()[:300],
                    "compiler": "cmake"
                })
            continue
        # Cap diagnostics per log
        if len(diags) >= 200:
            break
    return diags

def detect_platform(job_name, log_text=""):
    """Try to detect platform from job name or log content."""
    combined = (job_name + " " + log_text[:2000]).lower()
    if 'ubuntu-24' in combined: return 'ubuntu-24.04'
    if 'ubuntu-22' in combined: return 'ubuntu-22.04'
    if 'ubuntu-20' in combined: return 'ubuntu-20.04'
    if 'ubuntu' in combined: return 'ubuntu'
    if 'macos-14' in combined: return 'macos-14'
    if 'macos-13' in combined: return 'macos-13'
    if 'macos' in combined: return 'macos'
    if 'windows-2022' in combined: return 'windows-2022'
    if 'windows-2019' in combined: return 'windows-2019'
    if 'windows' in combined: return 'windows'
    return 'unknown'

def detect_build_command(log_text):
    """Try to detect build command from log."""
    for line in log_text.split('\n')[:500]:
        l = line.strip()
        if 'cmake --build' in l: return l[:200]
        if 'make -j' in l or 'make ' in l and 'Makefile' not in l: return l[:200]
        if 'ninja' in l.lower() and ('build' in l.lower() or '-j' in l): return l[:200]
        if 'meson compile' in l: return l[:200]
    return ""

def process_repo(repo):
    """Process a single repo, return (records, error_note)."""
    records = []
    notes = []
    
    # Step 1: Get last 10 completed runs
    out, err = run_gh(['api', f'repos/{repo}/actions/runs?per_page=10&status=completed',
                       '--jq', '.workflow_runs[] | {id, name, conclusion, head_sha}'])
    time.sleep(1)
    if err:
        return [], f"runs fetch error: {err}"
    if not out or not out.strip():
        return [], "no completed runs found"
    
    # Parse runs (jq outputs one JSON per line with --jq)
    runs = []
    for line in out.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except:
            pass
    
    if not runs:
        return [], "could not parse runs"
    
    # Select runs to inspect: failed ones, or up to 3 most recent
    failed_runs = [r for r in runs if r.get('conclusion') == 'failure']
    if failed_runs:
        target_runs = failed_runs[:5]  # limit to 5 failed runs
    else:
        target_runs = runs[:3]
    
    for run in target_runs:
        run_id = run['id']
        run_name = run.get('name', '')
        conclusion = run.get('conclusion', '')
        head_sha = run.get('head_sha', '')
        
        # Step 2: Get jobs
        out, err = run_gh(['api', f'repos/{repo}/actions/runs/{run_id}/jobs?per_page=20',
                           '--jq', '.jobs[] | {id, name, conclusion}'])
        time.sleep(1)
        if err:
            notes.append(f"run {run_id}: jobs error: {err}")
            continue
        
        jobs = []
        for line in (out or '').strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except:
                pass
        
        # Select failed jobs, or build-related jobs
        failed_jobs = [j for j in jobs if j.get('conclusion') == 'failure']
        if not failed_jobs:
            # pick build/compile jobs
            failed_jobs = [j for j in jobs if any(k in j.get('name','').lower() for k in ['build', 'compile', 'make', 'cmake'])]
        if not failed_jobs and jobs:
            failed_jobs = jobs[:2]
        
        for job in failed_jobs[:3]:  # max 3 jobs per run
            job_id = job['id']
            job_name = job.get('name', '')
            
            # Try annotations first
            diags = []
            ann_out, ann_err = run_gh(['api', f'repos/{repo}/check-runs/{job_id}/annotations'])
            time.sleep(1)
            if ann_out and not ann_err:
                try:
                    annotations = json.loads(ann_out)
                    for a in annotations[:50]:
                        if a.get('annotation_level') in ('failure', 'warning', 'error'):
                            diags.append({
                                "file": a.get('path', ''),
                                "line": a.get('start_line', 0),
                                "col": a.get('start_column', 0) or 0,
                                "severity": "error" if a.get('annotation_level') == 'failure' else a.get('annotation_level', 'warning'),
                                "message": a.get('message', '')[:300],
                                "compiler": "annotation"
                            })
                except:
                    pass
            
            # Fetch log if we need more diagnostics
            if len(diags) < 5:
                log_out, log_err = run_gh(['api', f'repos/{repo}/actions/jobs/{job_id}/logs'], timeout=90)
                time.sleep(1)
                if log_out:
                    log_diags = parse_diagnostics(log_out)
                    # Merge, avoiding duplicates
                    existing_keys = {(d['file'], d['line'], d['message'][:50]) for d in diags}
                    for d in log_diags:
                        k = (d['file'], d['line'], d['message'][:50])
                        if k not in existing_keys:
                            diags.append(d)
                            existing_keys.add(k)
                        if len(diags) >= 100:
                            break
                    platform = detect_platform(job_name, log_out[:3000])
                    build_cmd = detect_build_command(log_out)
                elif log_err:
                    notes.append(f"job {job_id}: log error: {log_err[:100]}")
                    platform = detect_platform(job_name)
                    build_cmd = ""
                else:
                    platform = detect_platform(job_name)
                    build_cmd = ""
            else:
                platform = detect_platform(job_name)
                build_cmd = ""
            
            record = {
                "repo": repo,
                "run_id": run_id,
                "job_name": job_name,
                "commit_sha": head_sha,
                "conclusion": conclusion,
                "platform": platform,
                "diagnostics": diags[:100],
                "build_command": build_cmd
            }
            records.append(record)
    
    return records, "; ".join(notes) if notes else None

# Main
total_diags = 0
results_summary = []

for repo in REPOS:
    repo_short = repo.split('/')[-1]
    outfile = os.path.join(OUTDIR, f"{repo_short}.jsonl")
    
    records, err_note = process_repo(repo)
    
    with open(outfile, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    n_diags = sum(len(r['diagnostics']) for r in records)
    total_diags += n_diags
    
    status = "ok"
    if err_note:
        status = f"partial ({err_note[:120]})"
    if not records:
        status = f"no data ({err_note or 'unknown'})" if err_note else "no data"
    
    results_summary.append(f"  {repo}: {len(records)} records, {n_diags} diagnostics - {status}")
    print(f"Processed {repo}: {len(records)} records, {n_diags} diags", file=sys.stderr)

print("\n=== SUMMARY ===")
print(f"Repos processed: {len(REPOS)}")
print(f"Total diagnostics: {total_diags}")
print(f"Output dir: {OUTDIR}")
for line in results_summary:
    print(line)
