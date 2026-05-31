#!/usr/bin/env python3
"""Isolation sweep for the mamba3_mimo_bwd time-chunk window at FULL scale.

Drives ONLY the mamba3_mimo_bwd backward segment of the full local_gb10_quarter
Path C direct-chain (depth=13, hidden=3584, max_seq=4096) -- NO 61-min forward.
For a given ``--rows-per-kernel-launch`` window it:

  1. monkeypatches MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH so the mamba3 segment is
     compiled with that per-launch time-chunk window,
  2. plans + compiles the chain, extracts the mamba3 segment,
  3. allocates caller-owned zero buffers,
  4. drives the segment's time-chunk launches ONE AT A TIME, timing EACH launch
     (eval + synchronize == one committed Metal command buffer), capturing the
     GPU watchdog timeout per-launch so we attribute the failure to the FIRST
     launch.

The FIRST launch (chunk 0, subchunk 0) is the watchdog gate: it processes the
LAST ``window`` reverse time-steps AND runs the checkpoint-replay for them.

By default only the first ``--max-launches`` launches are timed (enough to see
whether the first launch fits the watchdog and the steady per-launch cost);
pass ``--all-launches`` to run the ENTIRE mamba3 backward (4096/window launches)
and report the total mamba3-backward wall time.

RULE #1: no silent fallback. A watchdog timeout on the first launch is RAISED
and reported as that window being infeasible.
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


def _find_mamba3_segment(chain: Any) -> Any:
    for seg in chain.segments:
        if any(n.op_name == "mamba3_mimo_bwd" for n in seg.region.nodes):
            return seg
    raise RuntimeError("no mamba3_mimo_bwd segment found in direct-chain plan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-kernel-launch", type=int, required=True,
                        help="time-steps per launch for the mamba3 bwd window")
    parser.add_argument("--max-launches", type=int, default=4,
                        help="number of leading launches to drive+time (gate probe)")
    parser.add_argument("--all-launches", action="store_true",
                        help="drive ALL launches (full mamba3 backward); reports total")
    args = parser.parse_args()

    window = int(args.rows_per_kernel_launch)
    if window <= 0:
        raise ValueError("--rows-per-kernel-launch must be positive")

    # Force the mamba3 per-op time-chunk window for THIS run.
    sched.MAMBA3_BWD_ROWS_PER_KERNEL_LAUNCH = window

    print(f"[sweep] device={mx.default_device()} target={m._path_c_default_target()}",
          flush=True)
    print(f"[sweep] mamba3 rows_per_kernel_launch (window) = {window}", flush=True)

    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    print(f"[sweep] profile={profile.name} depth={profile.depth} "
          f"hidden={profile.hidden_size} max_seq={profile.max_seq_length}", flush=True)
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    mamba_seg = _find_mamba3_segment(chain)
    tgt = mamba_seg.schedule_target
    print(f"[sweep] mamba3 seg index={mamba_seg.index} region={mamba_seg.region.name} "
          f"row_dispatch={getattr(tgt, 'row_dispatch_mode', None)} "
          f"max_rows_per_launch={getattr(tgt, 'max_rows_per_launch', None)}", flush=True)

    # Compile ALL segments (the runtime route needs the full artifact set), then
    # drive ONLY the mamba3 sub-chain.
    t_c = time.perf_counter()
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)
    print(f"[sweep] compiled {len(artifacts)} segment shaders in "
          f"{time.perf_counter()-t_c:.1f}s", flush=True)

    # Inspect the compiled mamba3 prim_func to report the launch math.
    prim_func = None
    for art in artifacts:
        pf = getattr(art, "prim_func", None) or getattr(art, "_prim_func", None)
        # artifacts is a dict in the per-segment probe; handle both.
    # The route runner reads launch math off the prim_func; surface it via the
    # same helper the runtime uses.
    sub_chain = dataclasses.replace(chain, segments=(mamba_seg,))

    # Hook per-launch timing by wrapping mx.synchronize (each time-chunk launch
    # commits via eval()+synchronize() -- the synchronize is the command-buffer
    # boundary). We time the interval between consecutive synchronize() returns.
    launch_times: list[float] = []
    state: dict[str, Any] = {"last": None}
    real_sync = mx.synchronize

    def timed_sync(*a: Any, **k: Any) -> Any:
        r = real_sync(*a, **k)
        now = time.perf_counter()
        if state["last"] is not None:
            dt = now - state["last"]
            launch_times.append(dt)
            print(f"[sweep]   launch[{len(launch_times)-1}] committed in "
                  f"{dt:.3f}s", flush=True)
        state["last"] = now
        return r

    mx.synchronize = timed_sync  # type: ignore[assignment]

    # Limit the number of launches when not running the full backward, by
    # truncating the launch list the runtime iterates. We do that by patching
    # _path_c_segment_time_chunk_launches in m04 to slice to max_launches.
    real_launches_fn = m._path_c_segment_time_chunk_launches
    limit = None if args.all_launches else int(args.max_launches)

    def limited_launches(pf: Any) -> tuple:
        full = real_launches_fn(pf)
        if not full:
            return full
        print(f"[sweep] mamba3 total time-chunk launches (full) = {len(full)} "
              f"(window={window}, S/window)", flush=True)
        if limit is None:
            return full
        return full[:limit]

    m._path_c_segment_time_chunk_launches = limited_launches  # type: ignore[assignment]

    specs = m._path_c_direct_chain_required_logical_buffer_specs(sub_chain)
    buffers: dict[str, Any] = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        buffers[name] = mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=dtype)
    mx.eval(*buffers.values())
    print(f"[sweep] allocated {len(buffers)} caller-owned buffers; driving "
          f"{'ALL' if args.all_launches else limit} mamba3 launches", flush=True)

    status = "ok"
    err = None
    t0 = time.perf_counter()
    # Seed the per-launch clock so launch[0]'s window (run-start -> first
    # synchronize) is measured. The route runner's pre-launch buffer eval is
    # cheap (zero buffers already materialised); the first synchronize boundary
    # therefore measures launch[0]'s GPU command buffer.
    state["last"] = t0
    try:
        payload = m.run_path_c_direct_fusion_chain_route(
            chain=sub_chain,
            logical_buffers=buffers,
            artifacts=artifacts,
        )
    except BaseException as exc:  # noqa: BLE001
        status = "FAIL"
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        payload = None
    finally:
        mx.synchronize = real_sync  # type: ignore[assignment]
        m._path_c_segment_time_chunk_launches = real_launches_fn  # type: ignore[assignment]
    total = time.perf_counter() - t0

    first_launch = launch_times[0] if launch_times else None
    steady = (
        sum(launch_times[1:]) / max(1, len(launch_times) - 1)
        if len(launch_times) > 1 else None
    )
    out = {
        "window_rows_per_kernel_launch": window,
        "status": status,
        "error": err,
        "driven_launches": (None if args.all_launches else limit),
        "all_launches": bool(args.all_launches),
        "timed_launch_count": len(launch_times),
        "first_launch_s": round(first_launch, 4) if first_launch is not None else None,
        "steady_per_launch_s": round(steady, 4) if steady is not None else None,
        "wall_total_s": round(total, 3),
        "first_launch_fits_watchdog": (status == "ok"),
    }
    print("\n[sweep] RESULT:\n" + json.dumps(out, indent=2), flush=True)
    return 0 if status == "ok" else 7


if __name__ == "__main__":
    raise SystemExit(main())
