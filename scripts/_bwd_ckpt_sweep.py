#!/usr/bin/env python3
"""Replay-fraction probe: steady per-launch time vs checkpoint interval.

Drives the SAME mamba3 bwd window=1 launches as the window sweep, but for a
given ``--checkpoint-interval`` (monkeypatched). Because window=1 fixes one
reverse time-step per launch and the replay loop re-runs the forward recompute
for ``[checkpoint_start, time_idx]`` (up to checkpoint_interval rows), the steady
per-launch time isolates the REPLAY cost. Smaller interval => fewer replays =>
less recompute => faster, IF replay dominates.

We time the FIRST launch (time_idx=4095) where the replay length == the worst
case for that interval, plus the steady launches. Reports per-launch time so we
can attribute the fraction that is checkpoint-replay vs the fixed backward math.
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
import cppmega_mlx.runtime.path_c_fusion_schedules as sched  # noqa: E402


def _find_mamba3_segment(chain):
    for seg in chain.segments:
        if any(n.op_name == "mamba3_mimo_bwd" for n in seg.region.nodes):
            return seg
    raise RuntimeError("no mamba3_mimo_bwd segment")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--max-launches", type=int, default=5)
    args = parser.parse_args()

    sched.MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH = 1
    sched.MAMBA3_BWD_REPLAY_CHECKPOINT_INTERVAL = int(args.checkpoint_interval)
    print(f"[ckpt] checkpoint_interval={args.checkpoint_interval} window=1", flush=True)

    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    mamba_seg = _find_mamba3_segment(chain)
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)
    sub_chain = dataclasses.replace(chain, segments=(mamba_seg,))

    launch_times: list[float] = []
    state: dict[str, Any] = {"last": None}
    real_sync = mx.synchronize

    def timed_sync(*a, **k):
        r = real_sync(*a, **k)
        now = time.perf_counter()
        if state["last"] is not None:
            launch_times.append(now - state["last"])
        state["last"] = now
        return r

    mx.synchronize = timed_sync

    real_launches_fn = m._path_c_segment_time_chunk_launches
    limit = int(args.max_launches)

    def limited_launches(pf):
        full = real_launches_fn(pf)
        return full[:limit] if full else full

    m._path_c_segment_time_chunk_launches = limited_launches

    specs = m._path_c_direct_chain_required_logical_buffer_specs(sub_chain)
    buffers = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        buffers[name] = mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=dtype)
    mx.eval(*buffers.values())

    status = "ok"
    err = None
    t0 = time.perf_counter()
    state["last"] = t0
    try:
        m.run_path_c_direct_fusion_chain_route(
            chain=sub_chain, logical_buffers=buffers, artifacts=artifacts,
        )
    except BaseException as exc:  # noqa: BLE001
        status = "FAIL"
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        mx.synchronize = real_sync
        m._path_c_segment_time_chunk_launches = real_launches_fn

    for i, dt in enumerate(launch_times):
        print(f"[ckpt]   launch[{i}] = {dt:.3f}s", flush=True)
    steady = (sum(launch_times[1:]) / max(1, len(launch_times) - 1)
              if len(launch_times) > 1 else None)
    out = {
        "checkpoint_interval": int(args.checkpoint_interval),
        "status": status,
        "error": err,
        "first_launch_s": round(launch_times[0], 4) if launch_times else None,
        "steady_per_launch_s": round(steady, 4) if steady is not None else None,
    }
    print("\n[ckpt] RESULT:\n" + json.dumps(out, indent=2), flush=True)
    return 0 if status == "ok" else 7


if __name__ == "__main__":
    raise SystemExit(main())
