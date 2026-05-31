#!/usr/bin/env python3
"""FAST determinism check for row-windowed (launcher_chunks) independent backward.

Runs the FIRST N row-window launches of a per-row-INDEPENDENT backward op TWICE
on the same random inputs (re-initialising owner grads each time), and asserts
every DIRECT-WRITE grad buffer is bit-exact across the two runs. The only
non-deterministic buffer is rope_inv_freq_grad (reduced via relaxed T.atomic_add,
verified in the generated source) -- its run-to-run float-reassociation variance
is inherent to the op in ANY dispatch mode and is reported (relative) but not
required to be bit-exact. This proves the row-windowing + path_c_first_row_launch
one-time owner-grad zero is deterministic on the non-atomic grads.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    parser.add_argument("--op", default="attention_qkv_projection_bwd")
    parser.add_argument("--max-launches", type=int, default=24)
    args = parser.parse_args()

    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(scheduled.region, include_backward=True)
    seg = next(s for s in chain.segments if [n.op_name for n in s.region.nodes] == [args.op])
    seg0 = dataclasses.replace(seg, index=0)
    chain1 = dataclasses.replace(chain, segments=(seg0,))
    arts = m.compile_path_c_direct_fusion_chain_artifacts(chain1)
    art = arts[0]
    prim_func = seg0.schedule_target.schedule_template(seg0.region)

    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain1)
    rng = np.random.default_rng(0)
    init = {}
    for nm, sp in specs.items():
        shape = tuple(int(d) for d in sp["shape"]); dt = str(sp["dtype"])
        if dt in ("float16", "bfloat16", "float32"):
            arr = rng.standard_normal(shape) * 0.1
            init[nm] = arr.astype("float32") if dt == "float32" else arr.astype("float16")
        else:
            init[nm] = np.zeros(shape, dtype=dt)

    kbuf_shapes = m._path_c_kernel_buffer_shapes(prim_func)
    korder = path_c_kernel_buffer_order(prim_func)
    gate = str(getattr(prim_func, "_cppmega_path_c_backward_gate_param", "path_c_run_backward") or "path_c_run_backward")
    cparam = str(getattr(prim_func, "_cppmega_path_c_row_chunk_index_param", "") or "")
    sparam = str(getattr(prim_func, "_cppmega_path_c_row_subchunk_index_param", "") or "")
    launches = m._path_c_segment_time_chunk_launches(prim_func)
    n = min(len(launches), args.max_launches)

    def run() -> dict[str, np.ndarray]:
        bufs = {nm: mx.array(init[nm]) for nm in specs}
        mx.eval(*bufs.values())
        kargs: list[Any] = []; spos: dict[str, int] = {}
        for nm in korder:
            if nm in bufs:
                kargs.append(m._path_c_exact_kernel_buffer(bufs[nm], kbuf_shapes.get(nm), mx_module=mx))
            elif nm == gate:
                kargs.append(1)
            elif nm in PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS:
                spos[str(nm)] = len(kargs); kargs.append(PATH_C_SCALAR_KERNEL_PARAM_DEFAULTS[nm])
            else:
                raise ValueError(f"unbound {nm}")
        arrays = tuple(kargs); barrays = tuple(a for a in arrays if hasattr(a, "shape"))
        mx.eval(*barrays)
        cp, sp = spos[cparam], spos[sparam]
        for ci, si in launches[:n]:
            la = list(arrays); la[cp] = int(ci); la[sp] = int(si)
            art(*la); mx.eval(*barrays); mx.synchronize()
        return {nm: np.asarray(bufs[nm].astype(mx.float32)) for nm in bufs}

    # Buffers written via relaxed T.atomic_add are non-deterministic in float
    # summation order in ANY dispatch mode (dispatch-independent), so they are
    # reported (relative) but not required bit-exact. Every OTHER grad is a
    # direct indexed write and MUST be bit-exact across the two row-windowed runs.
    import re
    atomic_targets = {
        mt.group(1)
        for line in prim_func._cppmega_path_c_generated_source.split("\n")
        for mt in [re.search(r"T\.atomic_add\(([A-Za-z0-9_]+)", line)]
        if mt
    }

    a = run(); b = run()
    det_max = 0.0; det_worst = None; atomic_max_abs = 0.0; changed = 0
    for nm in a:
        if not np.allclose(a[nm], 0):
            changed += 1
        if nm in atomic_targets:
            atomic_max_abs = max(atomic_max_abs, float(np.max(np.abs(a[nm] - b[nm]))) if a[nm].size else 0.0)
            continue
        d = float(np.max(np.abs(a[nm] - b[nm]))) if a[nm].size else 0.0
        if d > det_max:
            det_max = d; det_worst = nm
    print(f"[det] op={args.op} launches={n}/{len(launches)} changed_out={changed} "
          f"atomic_targets={sorted(t.split('_')[-2]+'_'+t.split('_')[-1] for t in atomic_targets)} "
          f"direct_write_grads_max_abs_diff={det_max:.3e} (worst={det_worst}) "
          f"atomic_grads_max_abs_diff={atomic_max_abs:.3e}", flush=True)
    ok = det_max == 0.0 and changed > 0
    print("[det] PASS (direct-write grads bit-exact; atomic grads vary by float noise)"
          if ok else "[det] FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
