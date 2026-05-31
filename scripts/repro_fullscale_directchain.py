#!/usr/bin/env python3
"""DECISIVE full-scale repro: compile + execute the local_gb10_quarter Path C
direct-chain BACKWARD segment shaders at FULL model scale (depth=13, dims=3584).

This drives the SAME path as scripts/m04_train_step.py
--use-path-c-direct-chain-runtime: it builds the full local_gb10_quarter regions,
plans the direct fusion chain (mamba3_mimo_bwd now monolithic grid_chunks,
m2rnn_bwd time-chunked launcher_chunks), NATIVE-COMPILES every segment shader via
TileLang -> Metal newComputePipelineState (this is where the prior
XPC_ERROR_CONNECTION_INTERRUPTED crash happened), then EXECUTES the route with
caller-owned zero buffers (this is where kIOGPUCommandBufferCallbackErrorTimeout
would fire if mamba3 monolithic trips the GPU watchdog).

No kernel patching, no model code swap; routes are the production direct-chain
schedule. RULE #1: the guarded runtime RAISES on failure (no silent fallback).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mlx.core as mx  # noqa: E402
import m04_train_step as m  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=None,
                        help="sequence length for the region shape env (default: region default)")
    parser.add_argument("--execute", action="store_true",
                        help="after compile, EXECUTE the route (drives the GPU watchdog)")
    args = parser.parse_args()

    print(f"[repro] mx default device: {mx.default_device()}", flush=True)
    print(f"[repro] PATH_C target: {m._path_c_default_target()}", flush=True)
    print(f"[repro] per-segment cmdbuf commit: "
          f"{m._path_c_per_segment_command_buffer_commit_enabled()}", flush=True)

    # 1) FULL-scale region (depth=13, hidden=3584) -- the real production route.
    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    print(f"[repro] profile={profile.name} depth={profile.depth} "
          f"hidden={profile.hidden_size} max_seq={profile.max_seq_length}", flush=True)
    sel = m._select_path_c_model_route_region(regions)
    assert sel is not None
    print(f"[repro] selected region={sel.name} nodes={len(sel.nodes)} "
          f"ops={[n.op_name for n in sel.nodes]}", flush=True)

    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    print(f"[repro] chain.status={chain.status} n_segments={len(chain.segments)}", flush=True)
    for seg in chain.segments:
        tgt = getattr(seg, "schedule_target", None)
        ops = [n.op_name for n in seg.region.nodes]
        rd = getattr(tgt, "row_dispatch_mode", None) if tgt else None
        print(f"[repro]   seg[{seg.index}] phase={seg.execution_phase} "
              f"status={seg.status} row_dispatch={rd} ops={ops}", flush=True)

    # 2) NATIVE COMPILE each segment shader -- the newComputePipelineState step.
    print("\n[repro] === COMPILING segment shaders (newComputePipelineState) ===", flush=True)
    t_compile = time.perf_counter()
    try:
        artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)
    except BaseException as exc:  # noqa: BLE001  surface the crash exactly
        print(f"[repro] !!! COMPILE FAILED after {time.perf_counter()-t_compile:.2f}s: "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 3
    compile_elapsed = time.perf_counter() - t_compile
    print(f"[repro] === ALL {len(artifacts)} segment shaders COMPILED ok in "
          f"{compile_elapsed:.2f}s ===", flush=True)

    if not args.execute:
        print(json.dumps({
            "phase": "compile_only",
            "compiled_segments": len(artifacts),
            "compile_elapsed_seconds": compile_elapsed,
            "mamba3_seg_dispatch": next(
                getattr(getattr(s, "schedule_target", None), "row_dispatch_mode", None)
                for s in chain.segments
                if any(n.op_name == "mamba3_mimo_bwd" for n in s.region.nodes)
            ),
            "m2rnn_seg_dispatch": next(
                getattr(getattr(s, "schedule_target", None), "row_dispatch_mode", None)
                for s in chain.segments
                if any(n.op_name == "m2rnn_bwd" for n in s.region.nodes)
            ),
        }, indent=2))
        return 0

    # 3) EXECUTE the route -- caller-owned zero buffers; drives the GPU watchdog.
    print("\n[repro] === EXECUTING direct-chain route (GPU watchdog window) ===", flush=True)
    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain)
    buffers: dict[str, Any] = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        buffers[name] = mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=dtype)
    mx.eval(*buffers.values())
    print(f"[repro] allocated {len(buffers)} caller-owned logical buffers", flush=True)

    t_run = time.perf_counter()
    try:
        payload = m.run_path_c_direct_fusion_chain_route(
            chain=chain,
            logical_buffers=buffers,
            artifacts=artifacts,
        )
    except BaseException as exc:  # noqa: BLE001
        print(f"[repro] !!! EXECUTE FAILED after {time.perf_counter()-t_run:.2f}s: "
              f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 4
    run_elapsed = time.perf_counter() - t_run
    print(f"[repro] === ROUTE EXECUTED status={payload.get('status')} in "
          f"{run_elapsed:.2f}s ===", flush=True)

    # Per-segment timing + status (focus on mamba3_mimo_bwd / m2rnn_bwd).
    seg_report = []
    for s in payload.get("segments", []):
        seg_report.append({
            "index": s.get("index"),
            "phase": s.get("execution_phase"),
            "status": s.get("status"),
            "ops": [n for n in s.get("op_signature", s.get("ops", []))]
                   if isinstance(s.get("op_signature", s.get("ops", [])), list) else None,
            "time_chunk_launch_count": s.get("time_chunk_launch_count"),
            "elapsed_seconds": s.get("elapsed_seconds"),
        })
    print(json.dumps({
        "phase": "executed",
        "route_status": payload.get("status"),
        "runtime_uses_direct_fusion_chain": payload.get("runtime_uses_direct_fusion_chain"),
        "segment_count": payload.get("segment_count"),
        "compile_elapsed_seconds": compile_elapsed,
        "run_elapsed_seconds": run_elapsed,
        "segments": seg_report,
    }, indent=2, default=str))
    return 0 if payload.get("status") == "ok" else 5


if __name__ == "__main__":
    raise SystemExit(main())
