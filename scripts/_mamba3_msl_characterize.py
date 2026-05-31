#!/usr/bin/env python3
"""Characterize the mamba3_mimo_bwd Metal kernel source: total MSL size + phase
attribution by counting generated lines belonging to each emitter phase marker.

Extracts the generated Metal (MSL) source from the compiled TileLang artifact for
the mamba3 backward segment and reports:
  - total MSL bytes / lines
  - the TileLang DSL source bytes/lines on the prim_func
  - rough per-phase line attribution within the MSL via heuristic markers
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import m04_train_step as m  # noqa: E402
import cppmega_mlx.runtime.path_c_fusion as pcf  # noqa: E402
import cppmega_mlx.runtime.path_c_fusion_schedules as pcs  # noqa: E402


def main() -> int:
    profile, route_symbols, regions = m._local_gb10_path_c_model_regions()
    sel = m._select_path_c_model_route_region(regions)
    scheduled = m.plan_path_c_fusion_schedule_for_region(sel, include_backward=True)
    chain = m.plan_path_c_direct_fusion_chain_for_region(
        scheduled.region, include_backward=True,
    )
    mamba_seg = next(
        s for s in chain.segments
        if any(n.op_name == "mamba3_mimo_bwd" for n in s.region.nodes)
    )
    print(f"mamba3 segment index={mamba_seg.index} region={mamba_seg.region.name}")
    target = mamba_seg.schedule_target

    schedule_template = m.mark_path_c_schedule_template_for_region(
        target.schedule_template,
        mamba_seg.region,
        implementation_kind=target.implementation_kind,
        production_schedule_id=target.schedule_id
        if target.implementation_kind == "production" else "",
        required_real_abi_inputs=target.required_real_abi_inputs,
    )

    # Compile just this segment and grab the artifact (JITKernel).
    captured = {}

    def capturing_lowerer(func_or_mod, *, target, **kwargs):
        kernel = pcf.tilelang_single_entry_lowerer(
            func_or_mod, target=target, execution_backend="tvm_ffi", **kwargs,
        )
        captured["kernel"] = kernel
        captured["prim_func"] = pcf._single_entry_prim_func(func_or_mod)
        return kernel

    compiled = m.compile_path_c_region(
        mamba_seg.region,
        schedule_template=schedule_template,
        schedule_name=target.schedule_name,
        schedule_status=target.schedule_status,
        tilelang_lowerer=capturing_lowerer,
        target="metal",
    )
    kernel = captured["kernel"]
    prim = captured.get("prim_func")
    dsl = getattr(prim, "_cppmega_path_c_generated_source", "") or ""
    print(f"DSL source: {len(dsl)} bytes, {dsl.count(chr(10))+1} lines")

    try:
        msl = kernel.get_kernel_source()
    except Exception as exc:  # noqa: BLE001
        print(f"get_kernel_source failed: {type(exc).__name__}: {exc}")
        msl = kernel.kernel_source if hasattr(kernel, "kernel_source") else ""
    if not isinstance(msl, str):
        msl = str(msl)
    lines = msl.splitlines()
    print(f"MSL source: {len(msl)} bytes ({len(msl)/1024:.1f} KB), {len(lines)} lines")

    # Count loops in MSL.
    n_for = sum(1 for ln in lines if "for (" in ln or "for(" in ln)
    n_if = sum(1 for ln in lines if ln.strip().startswith("if (") or ln.strip().startswith("if("))
    n_exp = msl.count("exp(") + msl.count("metal::exp")
    n_log = msl.count("log(") + msl.count("metal::log")
    n_atomic = msl.count("atomic")
    print(f"MSL loops(for)={n_for} ifs={n_if} exp={n_exp} log={n_log} atomic_tokens={n_atomic}")

    # Dump MSL to a file for inspection.
    out = ROOT / "scripts" / "_mamba3_bwd.metal"
    out.write_text(msl)
    print(f"wrote MSL -> {out}")

    # Also dump the DSL for phase inspection.
    out2 = ROOT / "scripts" / "_mamba3_bwd_dsl.txt"
    out2.write_text(dsl)
    print(f"wrote DSL -> {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
