#!/usr/bin/env python3
"""Probe: force launcher_chunks (ROW-CHUNK) on a per-row-INDEPENDENT backward op
and time the per-launch wall-time + verify each window stays under the watchdog.

Targets seg index under max_segment_nodes=1 so each backward op is isolated.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mlx.core as mx  # noqa: E402
import m04_train_step as m  # noqa: E402
import cppmega_mlx.runtime.path_c_fusion_schedules as S  # noqa: E402

PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS = m.PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS
path_c_kernel_buffer_order = m.path_c_kernel_buffer_order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True,
                        help="backward op to isolate, e.g. attention_qkv_projection_bwd")
    parser.add_argument("--max-rows-per-launch", type=int, default=64)
    parser.add_argument("--max-launches", type=int, default=10)
    args = parser.parse_args()

    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True, max_segment_nodes=1,
    )
    seg = next(s for s in chain.segments
               if [n.op_name for n in s.region.nodes] == [args.op])
    base_target = seg.schedule_target
    # Force launcher_chunks (row-chunk) on this op's target.
    forced_target = S._target_with_max_rows_per_launch(
        base_target, seg.region, args.max_rows_per_launch,
        S.DESCRIPTOR_ROW_DISPATCH_LAUNCHER_CHUNKS,
    )
    prim_func = forced_target.schedule_template(seg.region)
    print(f"[rc] op={args.op} forced row_dispatch="
          f"{getattr(prim_func,'_cppmega_path_c_row_dispatch_mode',None)} "
          f"max_rows={getattr(prim_func,'_cppmega_path_c_max_rows_per_launch',None)} "
          f"chunk_count={getattr(prim_func,'_cppmega_path_c_row_chunk_count',None)} "
          f"subchunk_count={getattr(prim_func,'_cppmega_path_c_row_subchunk_count',None)}", flush=True)
    src = prim_func and None
    # Compile this single forced segment via compile_path_c_region path used by chain.
    import dataclasses
    seg2 = dataclasses.replace(seg, schedule_target=forced_target, index=0)
    chain2 = dataclasses.replace(chain, segments=(seg2,))
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain2)
    art = artifacts[0]
    ksrc = art.get_kernel_source()
    print(f"[rc] shader {len(ksrc.encode())//1024}KB {ksrc.count(chr(10))+1}L", flush=True)

    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain2)
    buffers: dict[str, Any] = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        buffers[name] = mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=dtype)
    mx.eval(*buffers.values())

    kernel_buffer_shapes = m._path_c_kernel_buffer_shapes(prim_func)
    kernel_param_order = path_c_kernel_buffer_order(prim_func)
    gate_param = str(getattr(prim_func, "_cppmega_path_c_backward_gate_param",
                             "path_c_run_backward") or "path_c_run_backward")
    chunk_param = str(getattr(prim_func, "_cppmega_path_c_row_chunk_index_param", "") or "")
    subchunk_param = str(getattr(prim_func, "_cppmega_path_c_row_subchunk_index_param", "") or "")
    kernel_call_args: list[Any] = []
    scalar_positions: dict[str, int] = {}
    for name in kernel_param_order:
        if name in buffers:
            kernel_call_args.append(m._path_c_exact_kernel_buffer(
                buffers[name], kernel_buffer_shapes.get(name), mx_module=mx))
        elif name == gate_param:
            kernel_call_args.append(1)
        elif name in PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS:
            scalar_positions[str(name)] = len(kernel_call_args)
            kernel_call_args.append(PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS[name])
        else:
            raise ValueError(f"param {name} unbound")
    arrays = tuple(kernel_call_args)
    buffer_arrays = tuple(a for a in arrays if hasattr(a, "shape"))
    mx.eval(*buffer_arrays)

    launches = m._path_c_segment_time_chunk_launches(prim_func)
    print(f"[rc] total launches={len(launches)}", flush=True)
    cpos = scalar_positions[chunk_param]
    spos = scalar_positions[subchunk_param]
    worst = 0.0
    n = min(len(launches), args.max_launches)
    for i, (ci, si) in enumerate(launches[:n]):
        la = list(arrays); la[cpos] = int(ci); la[spos] = int(si)
        t0 = time.perf_counter()
        try:
            art(*la); mx.eval(*buffer_arrays); mx.synchronize()
        except BaseException as exc:  # noqa: BLE001
            print(f"[rc] launch {i} (c={ci},s={si}) FAIL {time.perf_counter()-t0:.3f}s "
                  f"{type(exc).__name__}: {str(exc)[:140]}", flush=True)
            return 1
        dt = time.perf_counter() - t0; worst = max(worst, dt)
        if i < 4 or dt > 1.0 or i == n-1:
            print(f"[rc] launch {i} (c={ci},s={si}) ok {dt:.4f}s", flush=True)
    est_total = worst * len(launches)
    print(f"[rc] ran {n}/{len(launches)}, worst={worst:.4f}s, "
          f"est_full={est_total:.1f}s ({len(launches)} launches)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
