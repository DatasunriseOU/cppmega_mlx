"""V7-Q05: cppmega_mlx.cli.smoke_zero1 — ZeRO-1 distributed receipt.

Thin entrypoint that wraps scripts/bench_zero1_loopback.py under
``mlx.launch -n W --hosts H1,H2,...`` so a single command produces the
loopback / multi-mac receipt JSON. Closes
``docs/distributed_zero1_smoke_procedure.md`` Step 1 "not yet
implemented" placeholder.

Usage::

    # 2-process loopback on the same Mac (canonical receipt)
    python -m cppmega_mlx.cli.smoke_zero1 \\
        --hosts 127.0.0.1,127.0.0.1 \\
        --out bench/baselines/zero1_loopback_2proc_m4.json

    # Multi-Mac (requires mlx.launch ssh access to both hosts)
    python -m cppmega_mlx.cli.smoke_zero1 \\
        --hosts dev-128.local,peer-48.local \\
        --num-steps 20

    # In-process simulation fallback (no mlx.launch needed)
    python -m cppmega_mlx.cli.smoke_zero1 --simulate \\
        --num-steps 8

    # Show the launch command without running it
    python -m cppmega_mlx.cli.smoke_zero1 --hosts 127.0.0.1,127.0.0.1 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Sit at <repo>/cppmega_mlx/cli/smoke_zero1.py — two parents up = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "bench" / "baselines" / "zero1_smoke.json"
DEFAULT_SIMULATE_OUT = (
    REPO_ROOT / "bench" / "baselines" / "zero1_simulated_2proc_m4.json"
)
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_zero1_loopback.py"


def _resolve_world_size(hosts_csv: str, override: int) -> int:
    """Derive world size from the hosts CSV unless an override is given."""
    if override and override > 0:
        return int(override)
    return len([h for h in hosts_csv.split(",") if h.strip()])


def _build_launch_argv(
    *, mlx_launch: str, hosts_csv: str, world_size: int,
    num_steps: int, lr: float, out: Path,
) -> list[str]:
    """Construct the mlx.launch argv that runs the loopback bench."""
    return [
        mlx_launch, "-n", str(world_size),
        "--hosts", hosts_csv,
        sys.executable, str(BENCH_SCRIPT),
        "--steps", str(num_steps),
        "--lr", str(lr),
        "--out", str(out),
    ]


def _build_simulate_argv(
    *, num_steps: int, lr: float, out: Path,
) -> list[str]:
    """Construct the in-process simulation argv (no mlx.launch needed)."""
    return [
        sys.executable, str(BENCH_SCRIPT),
        "--simulate",
        "--steps", str(num_steps),
        "--lr", str(lr),
        "--out", str(out),
    ]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cppmega_mlx.cli.smoke_zero1",
        description="ZeRO-1 distributed smoke receipt via mlx.launch.",
    )
    p.add_argument(
        "--hosts",
        default="127.0.0.1,127.0.0.1",
        help=("Comma-separated host list passed to mlx.launch --hosts. "
              "Default: 127.0.0.1,127.0.0.1 (2-proc loopback)."),
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=0,
        help=("Explicit -n value for mlx.launch. Defaults to len(hosts)."),
    )
    p.add_argument("--num-steps", type=int, default=20,
                   help="Steps per rank + W=1 control (default 20).")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (default 1e-4).")
    p.add_argument("--out", type=Path, default=None,
                   help=(f"Output JSON. Defaults to {DEFAULT_OUT} "
                         f"(or {DEFAULT_SIMULATE_OUT} for --simulate)."))
    p.add_argument("--simulate", action="store_true",
                   help=("Run the in-process simulation fallback "
                         "(no mlx.launch needed)."))
    p.add_argument("--mlx-launch", default="mlx.launch",
                   help="Path to mlx.launch (default: looked up on PATH).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the argv that would be launched and exit 0.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.simulate:
        out = args.out or DEFAULT_SIMULATE_OUT
        cmd = _build_simulate_argv(
            num_steps=args.num_steps, lr=args.lr, out=out,
        )
        if args.dry_run:
            print(f"[smoke_zero1 dry-run] would run: {cmd}", file=sys.stderr)
            return 0
        return subprocess.call(cmd)

    # Loopback / multi-mac path: requires mlx.launch on PATH.
    world_size = _resolve_world_size(args.hosts, args.world_size)
    if world_size < 1:
        print("[smoke_zero1] world size resolved to 0; "
              "pass --hosts or --world-size", file=sys.stderr)
        return 64

    if not args.dry_run:
        # shutil.which honors PATH; explicit absolute paths bypass.
        if "/" not in args.mlx_launch:
            resolved = shutil.which(args.mlx_launch)
            if resolved is None:
                print(f"[smoke_zero1] mlx.launch not found on PATH "
                      f"(searched as {args.mlx_launch!r}). Install MLX "
                      f"or use --simulate.", file=sys.stderr)
                return 64
            mlx_launch_bin = resolved
        else:
            if not Path(args.mlx_launch).exists():
                print(f"[smoke_zero1] mlx.launch not found at "
                      f"{args.mlx_launch} (not on PATH either).",
                      file=sys.stderr)
                return 64
            mlx_launch_bin = args.mlx_launch
    else:
        mlx_launch_bin = args.mlx_launch

    out = args.out or DEFAULT_OUT
    cmd = _build_launch_argv(
        mlx_launch=mlx_launch_bin, hosts_csv=args.hosts,
        world_size=world_size, num_steps=args.num_steps,
        lr=args.lr, out=out,
    )
    if args.dry_run:
        print(f"[smoke_zero1 dry-run] would run: {cmd}", file=sys.stderr)
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
