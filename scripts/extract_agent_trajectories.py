#!/usr/bin/env python
"""Phase-8 CLI: extract C/C++ agent trajectories into a parquet dataset.

Enumerates local (and optionally remote) Claude Code + Codex sessions, keeps
only C/C++ sessions, walks them into ordered (obs, action, result, outcome)
transitions, and writes a parquet table.

RULE #1: rewards / exit codes are emitted ONLY when the session data carries a
REAL verifiable outcome; non-build/non-test steps always have reward=None.

Usage:
    extract_agent_trajectories.py --local-only --max-sessions 5 \
        --out outputs/agent_trajectories/transitions.parquet
    extract_agent_trajectories.py --with-remote --out <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega_mlx.data.agent_trajectory import (  # noqa: E402
    enumerate_local_sessions,
    enumerate_remote_sessions,
    extract_all,
    write_parquet,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument(
        "--local-only",
        action="store_true",
        help="enumerate only local sessions (default)",
    )
    grp.add_argument(
        "--with-remote",
        action="store_true",
        help="also enumerate remote sessions via ssh (read-only)",
    )
    ap.add_argument("--remote-host", default="dave@10.0.0.25")
    ap.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="cap number of sessions considered (after enumeration)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "outputs" / "agent_trajectories" / "transitions.parquet",
    )
    args = ap.parse_args(argv)

    refs = enumerate_local_sessions()
    if args.with_remote:
        refs = refs + enumerate_remote_sessions(args.remote_host)

    transitions, stats = extract_all(refs, max_sessions=args.max_sessions)

    print(f"sessions seen:    {stats.sessions_seen}")
    print(f"sessions kept:    {stats.sessions_kept} (C/C++)")
    print(f"sessions dropped: {stats.sessions_dropped} (non-C/C++)")
    print(f"  by source:      {stats.by_source}")
    print(f"transitions:      {stats.transitions}")
    print(f"build steps:      {stats.build_steps}")
    print(f"test steps:       {stats.test_steps}")
    print(f"build/test w/exit:{stats.build_test_with_exit}")
    print(f"rewards emitted:  {stats.rewards_emitted}")

    if not transitions:
        print("NO transitions extracted -- nothing written.", file=sys.stderr)
        return 1

    out = write_parquet(transitions, args.out)
    print(f"wrote {len(transitions)} transitions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
