#!/usr/bin/env python3
"""Time INDIVIDUAL time-chunk launches for a launcher_chunks backward segment.

Replicates the per-(chunk,subchunk) launch loop from
run_path_c_direct_fusion_chain_route for ONE segment, timing each launch
(eval+synchronize) so we can see launch count + per-launch watchdog headroom.
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
PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS = m.PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS
path_c_kernel_buffer_order = m.path_c_kernel_buffer_order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-index", type=int, required=True)
    parser.add_argument("--max-launches", type=int, default=999999)
    args = parser.parse_args()

    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)
    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain)
    buffers: dict[str, Any] = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        buffers[name] = mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=dtype)
    mx.eval(*buffers.values())

    seg = next(s for s in chain.segments if s.index == args.only_index)
    target = seg.schedule_target
    artifact = m._path_c_direct_chain_artifact_for_segment(artifacts, seg)
    prim_func = target.schedule_template(seg.region)

    print(f"[lp] seg{seg.index} {seg.region.name} ops={[n.op_name for n in seg.region.nodes]}", flush=True)
    print(f"[lp] row_dispatch={getattr(prim_func,'_cppmega_path_c_row_dispatch_mode',None)} "
          f"max_rows_per_launch={getattr(prim_func,'_cppmega_path_c_max_rows_per_launch',None)} "
          f"row_chunk_count={getattr(prim_func,'_cppmega_path_c_row_chunk_count',None)} "
          f"row_subchunk_count={getattr(prim_func,'_cppmega_path_c_row_subchunk_count',None)}", flush=True)

    launches = m._path_c_segment_time_chunk_launches(prim_func)
    print(f"[lp] total launches={len(launches)}", flush=True)

    # Assemble kernel args (mirror the route)
    kernel_buffer_shapes = m._path_c_kernel_buffer_shapes(prim_func)
    kernel_param_order = path_c_kernel_buffer_order(prim_func)
    gate_param = str(getattr(prim_func, "_cppmega_path_c_backward_gate_param",
                             "path_c_run_backward") or "path_c_run_backward")
    run_backward_value = 1 if seg.execution_phase == "backward" else 0
    chunk_param = str(getattr(prim_func, "_cppmega_path_c_row_chunk_index_param", "") or "")
    subchunk_param = str(getattr(prim_func, "_cppmega_path_c_row_subchunk_index_param", "") or "")
    kernel_call_args: list[Any] = []
    scalar_positions: dict[str, int] = {}
    for name in kernel_param_order:
        if name in buffers:
            kernel_call_args.append(m._path_c_exact_kernel_buffer(
                buffers[name], kernel_buffer_shapes.get(name), mx_module=mx))
        elif name == gate_param:
            kernel_call_args.append(run_backward_value)
        elif name in PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS:
            scalar_positions[str(name)] = len(kernel_call_args)
            kernel_call_args.append(PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS[name])
        else:
            raise ValueError(f"param {name} unbound")
    arrays = tuple(kernel_call_args)
    buffer_arrays = tuple(a for a in arrays if hasattr(a, "shape"))
    mx.eval(*buffer_arrays)
    chunk_pos = scalar_positions[chunk_param]
    subchunk_pos = scalar_positions[subchunk_param]

    n = min(len(launches), args.max_launches)
    t_all = time.perf_counter()
    worst = 0.0
    for i, (ci, si) in enumerate(launches[:n]):
        la = list(arrays)
        la[chunk_pos] = int(ci)
        la[subchunk_pos] = int(si)
        t0 = time.perf_counter()
        try:
            artifact(*la)
            mx.eval(*buffer_arrays)
            mx.synchronize()
        except BaseException as exc:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(f"[lp] launch {i} (chunk={ci},sub={si}) FAIL after {dt:.3f}s: "
                  f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return 1
        dt = time.perf_counter() - t0
        worst = max(worst, dt)
        if i < 5 or dt > 1.0 or i == n - 1:
            print(f"[lp] launch {i} (chunk={ci},sub={si}) ok {dt:.4f}s", flush=True)
    print(f"[lp] ran {n}/{len(launches)} launches, worst={worst:.4f}s, "
          f"total={time.perf_counter()-t_all:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
