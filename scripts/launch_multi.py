"""V7-B-real: thin wrapper around mlx.launch for cppmega multi-process.

Honest-closure: docs/multimac_training.md described a multi-host
runbook but shipped no launcher. This script wraps mlx.launch (when
present) and mpirun (fallback) so a developer can do:

    python -m scripts.launch_multi --np 2 -- \\
        python -m scripts.bench_allreduce_distributed --buf-mb 1 4

without having to remember the exact mlx.launch flags. It picks
mlx.launch when MLX_USE_LAUNCHER=1 or mlx.launch is importable, and
falls back to a system mpirun.

Features added:
- Programmatic macOS Application Firewall authorization via `socketfilterfw`.
- Network interface bindings (--interface en0/en1/lo0) mapped to Open MPI MCA.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys


def _has_mlx_launch() -> bool:
    try:
        import importlib
        importlib.import_module("mlx.launch")
        return True
    except Exception:
        return False


def _get_active_interfaces() -> dict[str, str]:
    """Retrieve all active network interfaces and their associated IPv4 addresses on macOS."""
    ifaces = {}
    if sys.platform != "darwin":
        return ifaces
    try:
        res = subprocess.run(["ifconfig", "-u"], capture_output=True, text=True, check=True)
        blocks = res.stdout.split("\n\n")
        for block in blocks:
            lines = block.split("\n")
            if not lines:
                continue
            first_line = lines[0]
            m = re.match(r"^([a-zA-Z0-9]+):", first_line)
            if not m:
                continue
            ifname = m.group(1)
            
            is_active = False
            ip_addr = None
            for line in lines[1:]:
                if "status: active" in line:
                    is_active = True
                if "inet " in line:
                    inet_m = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", line)
                    if inet_m:
                        ip_addr = inet_m.group(1)
            
            if ifname == "lo0" and not ip_addr:
                ip_addr = "127.0.0.1"
                is_active = True
                
            if is_active and ip_addr:
                ifaces[ifname] = ip_addr
    except Exception:
        pass
    return ifaces


def _authorize_firewall() -> None:
    """Programmatically authorize the active Python interpreter under macOS Application Firewall."""
    if sys.platform == "darwin":
        fw_path = "/usr/libexec/applicationfirewall/socketfilterfw"
        if os.path.exists(fw_path):
            print(f"[launch] Programmatically authorizing python interpreter under macOS Application Firewall...", file=sys.stderr)
            try:
                subprocess.run(["sudo", fw_path, "--add", sys.executable], check=True)
                subprocess.run(["sudo", fw_path, "--unblockapp", sys.executable], check=True)
                print(f"[launch] macOS Application Firewall authorization successful.", file=sys.stderr)
            except Exception as e:
                print(f"[launch] WARNING: macOS Firewall authorization failed or user cancelled: {e}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-process launcher for cppmega_mlx",
        epilog="Args after `--` are forwarded to the target command.",
    )
    parser.add_argument("--np", type=int, default=2,
                         help="Number of processes (world_size).")
    parser.add_argument("--backend",
                         choices=["mlx", "mpi", "auto"],
                         default="auto")
    parser.add_argument("--hostfile", default=None,
                         help="MPI hostfile (mpirun mode).")
    parser.add_argument("--interface", default="auto",
                         help="Network interface to bind (e.g. en0, en1, lo0) or 'auto' (default).")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                         help="Command after `--`.")
    args = parser.parse_args()

    # Strip leading `--` if present.
    cmd = args.rest
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("no target command supplied after `--`")

    # 1. Programmatically authorize the firewall
    _authorize_firewall()

    # 2. NIC Device Bindings & Open MPI Environment Variable configuration
    ifaces = _get_active_interfaces()
    target_iface = args.interface
    if target_iface == "auto":
        non_lo = [k for k in ifaces if k != "lo0"]
        if non_lo:
            target_iface = non_lo[0]
        elif "lo0" in ifaces:
            target_iface = "lo0"
        else:
            target_iface = None

    if target_iface and target_iface in ifaces:
        ip_addr = ifaces[target_iface]
        print(f"[launch] Binding to network interface: {target_iface} (IP: {ip_addr})", file=sys.stderr)
        os.environ["OMPI_MCA_btl_tcp_if_include"] = target_iface
        os.environ["OMPI_MCA_oob_tcp_if_include"] = target_iface
        os.environ["MASTER_ADDR"] = ip_addr
    elif target_iface:
        print(f"[launch] WARNING: network interface '{target_iface}' is inactive or lacks IPv4 address.", file=sys.stderr)

    backend = args.backend
    if backend == "auto":
        if os.environ.get("MLX_USE_LAUNCHER") == "1" and _has_mlx_launch():
            backend = "mlx"
        elif shutil.which("mpirun"):
            backend = "mpi"
        elif _has_mlx_launch():
            backend = "mlx"
        else:
            print("ERROR: neither mlx.launch nor mpirun available.",
                   file=sys.stderr)
            return 2

    if backend == "mlx":
        # mlx.launch is a Python module with a CLI surface.
        launcher = [sys.executable, "-m", "mlx.launch",
                    "--num-processes", str(args.np), "--"]
        full = launcher + cmd
    else:
        # mpirun
        launcher = ["mpirun", "-n", str(args.np)]
        if args.hostfile:
            launcher.extend(["--hostfile", args.hostfile])
        full = launcher + cmd

    print(f"[launch] backend={backend} np={args.np}  cmd={' '.join(map(shlex.quote, full))}",
          file=sys.stderr)
    return subprocess.run(full).returncode


if __name__ == "__main__":
    raise SystemExit(main())
