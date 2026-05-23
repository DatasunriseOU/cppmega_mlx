"""V7-B-real: thin wrapper around mlx.launch for cppmega multi-process.

Honest-closure: docs/multimac_training.md described a multi-host
runbook but shipped no launcher. This script wraps mlx.launch (when
present) and mpirun (fallback) so a developer can do:

    python -m scripts.launch_multi --np 2 -- \\
        python -m scripts.bench_allreduce_distributed --buf-mb 1 4

without having to remember the exact mlx.launch flags. It picks
mlx.launch when MLX_USE_LAUNCHER=1 or mlx.launch is importable, and
falls back to a system mpirun.
"""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                         help="Command after `--`.")
    args = parser.parse_args()

    # Strip leading `--` if present.
    cmd = args.rest
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("no target command supplied after `--`")

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
