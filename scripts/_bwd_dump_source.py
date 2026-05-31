#!/usr/bin/env python3
"""Dump the generated TileLang PrimFunc source for the mamba3 bwd segment.

Confirms the T.Kernel launch (threadgroup count + threads) and the dominant
loop structure, with NO GPU run.
"""
from __future__ import annotations

import linecache
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import m04_train_step as m  # noqa: E402
import cppmega_mlx.runtime.path_c_fusion_schedules as sched  # noqa: E402


def _find_mamba3_segment(chain):
    for seg in chain.segments:
        if any(n.op_name == "mamba3_mimo_bwd" for n in seg.region.nodes):
            return seg
    raise RuntimeError("no mamba3_mimo_bwd segment")


def main() -> int:
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    sched.MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH = window
    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    seg = _find_mamba3_segment(chain)
    # Build the descriptor prim_func for THIS segment so the source is cached.
    pf = m.build_path_c_descriptor_prim_func_for_segment(seg) if hasattr(
        m, "build_path_c_descriptor_prim_func_for_segment") else None
    # Fall back: compile the chain so all sources are emitted into linecache.
    _ = sched  # noqa
    m.compile_path_c_direct_fusion_chain_artifacts(chain)

    # Find the cached source whose entry name carries this segment.
    hits = []
    for fname, entry in list(linecache.cache.items()):
        if not fname.startswith("<path_c_descriptor_schedule:"):
            continue
        lines = entry[2]
        text = "".join(lines)
        if "mamba3" in text and ("time_rev" in text or "checkpoint" in text):
            hits.append((fname, text))
    if not hits:
        print("NO mamba3 bwd source found in linecache")
        return 1
    # Pick the one that mentions the bwd policy marker.
    chosen = None
    for fname, text in hits:
        if "mamba3_mimo_bwd_policy" in text:
            chosen = (fname, text)
            break
    if chosen is None:
        chosen = hits[0]
    fname, text = chosen
    lines = text.splitlines()
    print(f"=== SOURCE: {fname} ({len(lines)} lines) ===\n")
    # Print the T.Kernel line and a window around it.
    for i, ln in enumerate(lines):
        if "T.Kernel" in ln:
            print(f"  [L{i}] {ln.strip()}")
    print()
    # Count loop nesting / sync_threads / the dominant H-inner loops.
    n_sync = sum(1 for ln in lines if "sync_threads" in ln)
    n_serial = sum(1 for ln in lines if "T.serial" in ln)
    n_hidden_inner = sum(
        1 for ln in lines
        if "T.serial(0, 3584)" in ln
    )
    print(f"  total lines              = {len(lines)}")
    print(f"  T.sync_threads barriers  = {n_sync}")
    print(f"  T.serial loops           = {n_serial}")
    print(f"  inner H-loops T.serial(0, 3584) = {n_hidden_inner}")
    # Show the time_rev loop region.
    print("\n=== time_rev / replay region ===")
    for i, ln in enumerate(lines):
        if "time_rev" in ln or "replay_offset" in ln or "checkpoint_start" in ln:
            print(f"  [L{i}] {ln.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
