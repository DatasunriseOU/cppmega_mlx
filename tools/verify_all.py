#!/usr/bin/env python3
"""Unified validation suite for cppmega.mlx.

Runs:
1. Python Ruff check
2. Python Mypy typecheck
3. Python custom semantic auditor (tools/lint_mlx.py)
4. Python Pytest unit tests
5. React UI linter (npm run lint)
6. React UI typecheck (npm run typecheck)
7. React UI Vitest tests (npm run test)
8. React UI Vite production bundle build (npm run build)
"""

import os
import subprocess
import sys
import time

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_stage(name: str, cmd: list[str], cwd: str | None = None) -> bool:
    print(f"\n{BOLD}{BLUE}========================================")
    print(f" STAGE: {name}")
    print(f" Running: {' '.join(cmd)}")
    print(f"========================================{RESET}")
    
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=cwd)
    dt = time.perf_counter() - t0
    
    if res.returncode == 0:
        print(f"{BOLD}{GREEN}✓ Stage '{name}' PASSED ({dt:.2f}s){RESET}")
        return True
    else:
        print(f"{BOLD}{RED}✗ Stage '{name}' FAILED ({dt:.2f}s) with exit code {res.returncode}{RESET}")
        return False


def main() -> int:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(root_dir, ".venv", "bin", "python")
    venv_ruff = os.path.join(root_dir, ".venv", "bin", "ruff")
    venv_mypy = os.path.join(root_dir, ".venv", "bin", "mypy")
    venv_pytest = os.path.join(root_dir, ".venv", "bin", "pytest")
    vbgui_dir = os.path.join(root_dir, "vbgui")
    
    # 1. Check virtual environment binaries
    if not os.path.exists(venv_python):
        print(f"{RED}Error: Virtual environment python not found at {venv_python}{RESET}")
        return 1
        
    stages = [
        ("Python: Ruff Linter", [venv_ruff, "check", "cppmega_mlx", "cppmega_v4"]),
        ("Python: Mypy Typecheck", [venv_mypy, "cppmega_mlx", "cppmega_v4", "--ignore-missing-imports", "--check-untyped-defs"]),
        ("Python: Codebase Auditor", [venv_python, "tools/lint_mlx.py"]),
        ("Python: Pytest Suite", [venv_pytest, "-m", "not slow"]),
        ("React UI: ESLint", ["npm", "run", "lint"], vbgui_dir),
        ("React UI: TypeScript", ["npm", "run", "typecheck"], vbgui_dir),
        ("React UI: Vitest Suite", ["npm", "run", "test"], vbgui_dir),
        ("React UI: Production Build", ["npm", "run", "build"], vbgui_dir),
    ]
    
    failed = []
    for name, cmd, *cwd_list in stages:
        cwd = cwd_list[0] if cwd_list else None
        # Make command paths absolute if relative in root_dir
        resolved_cmd = []
        for c in cmd:
            if c.startswith("tools/"):
                resolved_cmd.append(os.path.join(root_dir, c))
            else:
                resolved_cmd.append(c)
        
        success = run_stage(name, resolved_cmd, cwd=cwd)
        if not success:
            failed.append(name)
            
    print(f"\n{BOLD}{BLUE}========================================")
    print(f" FINAL SUMMARY")
    print(f"========================================{RESET}")
    
    if not failed:
        print(f"{BOLD}{GREEN}✓ ALL STAGES PASSED SUCCESSFULLY! {RESET}")
        return 0
    else:
        print(f"{BOLD}{RED}✗ SOME STAGES FAILED:")
        for f in failed:
            print(f"  - {f}")
        print(f"{RESET}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
