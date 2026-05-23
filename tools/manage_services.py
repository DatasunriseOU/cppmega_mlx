#!/usr/bin/env python3
"""Unified service manager for cppmega.mlx.

Manages both the FastAPI backend and Vite frontend dev server.

Usage:
  python tools/manage_services.py start     # Launches both servers in background
  python tools/manage_services.py stop      # Shuts down both servers gracefully
  python tools/manage_services.py status    # Checks server health/PID status
  python tools/manage_services.py restart   # Restarts both servers
"""

import argparse
import os
import signal
import subprocess
import sys
import time

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_pid_file_path() -> str:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, ".services.pid")


def save_pids(backend_pid: int, frontend_pid: int) -> None:
    path = get_pid_file_path()
    with open(path, "w") as f:
        f.write(f"backend={backend_pid}\nfrontend={frontend_pid}\n")


def load_pids() -> dict[str, int]:
    path = get_pid_file_path()
    pids = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                for line in f:
                    if "=" in line:
                        name, val = line.strip().split("=")
                        pids[name] = int(val)
        except Exception:
            pass
    return pids


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_services() -> int:
    pids = load_pids()
    if pids:
        any_running = False
        for name, pid in pids.items():
            if is_pid_running(pid):
                print(f"{YELLOW}Warning: {name} is already running with PID {pid}{RESET}")
                any_running = True
        if any_running:
            print(f"{RED}Please stop services first using: python tools/manage_services.py stop{RESET}")
            return 1

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(root_dir, ".venv", "bin", "python")
    vbgui_dir = os.path.join(root_dir, "vbgui")

    print(f"{BOLD}{BLUE}Starting cppmega.mlx services...{RESET}")

    # 1. Start backend server
    print(f"-> Starting API Backend server on http://127.0.0.1:8765...")
    backend_log = open(os.path.join(root_dir, "backend.log"), "w")
    backend_proc = subprocess.Popen(
        [venv_python, "-m", "cppmega_v4.jsonrpc.server"],
        cwd=root_dir,
        stdout=backend_log,
        stderr=backend_log,
        start_new_session=True
    )

    # 2. Start frontend dev server
    print(f"-> Starting React Flow Frontend server on http://localhost:5173/...")
    frontend_log = open(os.path.join(root_dir, "frontend.log"), "w")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=vbgui_dir,
        stdout=frontend_log,
        stderr=frontend_log,
        start_new_session=True
    )

    # Allow startup buffer
    time.sleep(2)

    # Double check if running
    backend_ok = is_pid_running(backend_proc.pid)
    frontend_ok = is_pid_running(frontend_proc.pid)

    if backend_ok and frontend_ok:
        save_pids(backend_proc.pid, frontend_proc.pid)
        print(f"\n{BOLD}{GREEN}✓ Services started successfully!{RESET}")
        print(f"  - API Backend: PID {backend_proc.pid} (logs at backend.log)")
        print(f"  - React GUI  : PID {frontend_proc.pid} (logs at frontend.log)")
        print(f"\nAccess the GUI at: {BOLD}http://localhost:5173/{RESET}\n")
        return 0
    else:
        print(f"\n{BOLD}{RED}✗ Failed to start one or both services.{RESET}")
        if not backend_ok:
            print(f"  - Backend failed to start (check backend.log)")
        if not frontend_ok:
            print(f"  - Frontend failed to start (check frontend.log)")
        # Attempt cleanup of any spawned proc
        try:
            backend_proc.terminate()
        except Exception:
            pass
        try:
            frontend_proc.terminate()
        except Exception:
            pass
        return 1


def stop_services() -> int:
    pids = load_pids()
    if not pids:
        print(f"{YELLOW}No active services found in PID file.{RESET}")
        return 0

    print(f"{BOLD}{BLUE}Stopping cppmega.mlx services...{RESET}")
    for name, pid in pids.items():
        if is_pid_running(pid):
            print(f"-> Terminating {name} (PID {pid})...")
            try:
                # Send SIGTERM first
                os.kill(pid, signal.SIGTERM)
                
                # Wait up to 3 seconds for graceful shutdown
                for _ in range(30):
                    if not is_pid_running(pid):
                        break
                    time.sleep(0.1)
                
                # If still running, force kill
                if is_pid_running(pid):
                    print(f"-> {name} still alive, sending SIGKILL...")
                    os.kill(pid, signal.SIGKILL)
            except Exception as e:
                print(f"{RED}Error terminating {name}: {e}{RESET}")
        else:
            print(f"-> {name} (PID {pid}) was not running.")

    # Remove pid file
    pid_path = get_pid_file_path()
    if os.path.exists(pid_path):
        os.remove(pid_path)

    print(f"{BOLD}{GREEN}✓ All services stopped.{RESET}")
    return 0


def check_status() -> int:
    pids = load_pids()
    if not pids:
        print(f"{BOLD}{YELLOW}Services status: STOPPED{RESET}")
        return 0

    all_ok = True
    print(f"\n{BOLD}{BLUE}========================================")
    print(f" cppmega.mlx Service Status")
    print(f"========================================{RESET}")
    
    for name, pid in pids.items():
        running = is_pid_running(pid)
        status_text = f"{GREEN}RUNNING{RESET}" if running else f"{RED}STOPPED{RESET}"
        print(f"  - {name:<10}: {status_text:<15} (PID {pid})")
        if not running:
            all_ok = False

    print(f"{BLUE}========================================{RESET}")
    if all_ok:
        print(f"Access GUI at: {BOLD}http://localhost:5173/{RESET}\n")
        return 0
    else:
        print(f"{YELLOW}Warning: One or more services are stopped.{RESET}\n")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Service Orchestration for cppmega.mlx")
    parser.add_argument("action", choices=["start", "stop", "status", "restart"],
                        help="Action to perform on services")
    args = parser.parse_args()

    if args.action == "start":
        return start_services()
    elif args.action == "stop":
        return stop_services()
    elif args.action == "status":
        return check_status()
    elif args.action == "restart":
        stop_services()
        time.sleep(1)
        return start_services()


if __name__ == "__main__":
    sys.exit(main())
