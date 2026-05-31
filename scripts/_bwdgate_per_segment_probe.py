#!/usr/bin/env python3
"""Per-segment timing probe for the full-scale direct-chain BACKWARD segments.

Runs EACH chain segment as its own one-segment sub-chain through
run_path_c_direct_fusion_chain_route, timing the command-buffer wall-time and
catching the GPU watchdog timeout per segment so we can attribute the failure to
an EXACT segment (rather than the bare traceback which only shows line 6826).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mlx.core as mx  # noqa: E402
import m04_train_step as m  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-index", type=int, default=None,
                        help="run ONLY this segment index (fresh-process isolation)")
    parser.add_argument("--max-segment-nodes", type=int, default=None)
    parser.add_argument("--backward-max-segment-nodes", type=int, default=None)
    args = parser.parse_args()
    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    plan_kwargs = {}
    if args.max_segment_nodes is not None:
        plan_kwargs["max_segment_nodes"] = args.max_segment_nodes
    if args.backward_max_segment_nodes is not None:
        plan_kwargs["backward_max_segment_nodes"] = args.backward_max_segment_nodes
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True, **plan_kwargs,
    )
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)

    # Allocate caller-owned buffers ONCE for the whole chain.
    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain)
    buffers: dict[str, Any] = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        buffers[name] = mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=dtype)
    mx.eval(*buffers.values())

    rows = []
    for seg in chain.segments:
        if args.only_index is not None and seg.index != args.only_index:
            continue
        ops = [n.op_name for n in seg.region.nodes]
        tgt = getattr(seg, "schedule_target", None)
        rd = getattr(tgt, "row_dispatch_mode", None) if tgt else None
        sub_chain = dataclasses.replace(chain, segments=(seg,))
        t0 = time.perf_counter()
        status = "ok"
        err = None
        nlaunch = None
        try:
            payload = m.run_path_c_direct_fusion_chain_route(
                chain=sub_chain,
                logical_buffers=buffers,
                artifacts=artifacts,
            )
            segres = payload.get("segments", [])
            if segres:
                nlaunch = segres[0].get("time_chunk_launch_count")
        except BaseException as exc:  # noqa: BLE001
            status = "FAIL"
            err = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        row = {
            "index": seg.index,
            "phase": seg.execution_phase,
            "region": seg.region.name,
            "ops": ops,
            "row_dispatch": rd,
            "time_chunk_launch_count": nlaunch,
            "wall_s": round(dt, 3),
            "status": status,
            "error": err,
        }
        rows.append(row)
        print(f"[probe] seg[{seg.index}] {seg.region.name} rd={rd} "
              f"ops={ops} -> {status} wall={dt:.3f}s "
              f"launches={nlaunch}"
              + (f"  ERR={err}" if err else ""), flush=True)
        if err:
            traceback.print_exc()
    print("\n[probe] JSON:\n" + json.dumps(rows, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
