#!/usr/bin/env python3
"""Measure MSL kernel-source size for EVERY direct-chain segment + whether its
pipeline-state compiles (newComputePipelineState) at full scale. Calibrates the
real crash threshold by correlating MSL size with compile-success per segment."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mlx.core as mx  # noqa: E402
import m04_train_step as m  # noqa: E402
import cppmega_mlx.runtime.path_c_fusion as pcf  # noqa: E402


def main() -> int:
    profile, _syms, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    artifacts = m.compile_path_c_direct_fusion_chain_artifacts(chain)

    # Pull MSL size per segment by re-compiling with a capturing lowerer.
    rows = []
    for seg in chain.segments:
        target = seg.schedule_target
        tmpl = m.mark_path_c_schedule_template_for_region(
            target.schedule_template, seg.region,
            implementation_kind=target.implementation_kind,
            production_schedule_id=target.schedule_id
            if target.implementation_kind == "production" else "",
            required_real_abi_inputs=target.required_real_abi_inputs,
        )
        cap = {}

        def lw(func_or_mod, *, target, **kwargs):
            k = pcf.tilelang_single_entry_lowerer(
                func_or_mod, target=target, execution_backend="tvm_ffi", **kwargs)
            cap["k"] = k
            return k

        m.compile_path_c_region(
            seg.region, schedule_template=tmpl,
            schedule_name=target.schedule_name,
            schedule_status=target.schedule_status,
            tilelang_lowerer=lw, target="metal",
        )
        try:
            msl = cap["k"].get_kernel_source()
        except Exception:
            msl = ""
        msl = msl if isinstance(msl, str) else str(msl)
        rows.append((seg.index, [n.op_name for n in seg.region.nodes],
                     len(msl), msl.count("\n") + 1))

    # Try compiling+running each isolated segment's pipeline-state.
    specs = m._path_c_direct_chain_required_logical_buffer_specs(chain)
    buffers = {}
    for name, spec in specs.items():
        dtype = getattr(mx, str(spec["dtype"]))
        buffers[name] = mx.zeros(tuple(int(d) for d in spec["shape"]), dtype=dtype)
    mx.eval(*buffers.values())

    print(f"{'idx':>3} {'kb':>6} {'lines':>5} {'pipeline':>9}  ops")
    for (idx, ops, nbytes, nlines) in rows:
        seg = next(s for s in chain.segments if s.index == idx)
        sub = dataclasses.replace(chain, segments=(seg,))
        ok = "ok"
        try:
            m.run_path_c_direct_fusion_chain_route(
                chain=sub, logical_buffers=buffers, artifacts=artifacts)
        except BaseException as exc:  # noqa: BLE001
            ok = "CRASH" if "XPC" in str(exc) or "state" in str(exc) else "fail"
        print(f"{idx:>3} {nbytes/1024:>6.1f} {nlines:>5} {ok:>9}  {ops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
