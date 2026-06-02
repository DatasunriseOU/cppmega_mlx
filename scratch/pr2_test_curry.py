"""PR-2 test: does currying the scalar run_backward to a compile-time constant
remove mismatch #3 (the guarded-sync wall)? We specialize the real prim with
path_c_run_backward bound to 0 (fwd-only) and 1 (bwd), then re-run the s_tir
build to see whether 'Cannot insert syncs inside condition' goes away.

If specialization folds the `if(run_backward)` away, the fwd-only / bwd-only prim
might lower under generic s_tir. If it STILL raises (the T.Kernel/T.alloc_shared
body is fundamentally tilelang-only), that pins the decision: the adapter must use
an EXTERNAL (tilelang.compile'd packed) kernel boundary, not a generic-TIR inline.
No fabrication: we print the exact outcome of each path."""
from __future__ import annotations
import sys, traceback
import tvm
from tvm import tir, relax

from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region,
    build_path_c_model_region_from_route_symbols,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


def try_specialize_and_build(prim, run_backward_value: int):
    print(f"\n{'='*70}\nSPECIALIZE path_c_run_backward = {run_backward_value}\n{'='*70}")
    # Find the scalar param.
    scalar = None
    for p in prim.params:
        if p not in prim.buffer_map:
            scalar = p
            break
    if scalar is None:
        print("  no scalar param found")
        return
    try:
        sp = prim.specialize({scalar: tir.const(run_backward_value, scalar.dtype)})
        print("  specialize OK -> new params:", len(sp.params),
              "scalars:", [p.name for p in sp.params if p not in sp.buffer_map])
    except Exception:
        print("  specialize FAILED:")
        traceback.print_exc()
        return
    # Try to lower this specialized prim through s_tir build (LLVM).
    mod = tvm.IRModule({sp.attrs["global_symbol"] if sp.attrs else "main": sp})
    try:
        # Use tirx build path (same as relax uses internally)
        lib = tvm.tirx.build(mod, target=tvm.target.Target("llvm"))
        print("  tirx.build (llvm) OK -- specialized prim lowers under generic s_tir!")
    except Exception as exc:
        msg = str(exc)
        key = "Cannot insert syncs inside condition"
        print(f"  tirx.build FAILED. contains '{key}':", key in msg)
        # Print just the salient error line.
        for line in msg.splitlines():
            if "Check failed" in line or key in line or "Kernel" in line or "shared" in line.lower():
                print("    >>", line.strip()[:200])


def main() -> int:
    cfg = local_gb10_quarter_profile().hybrid_config()
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg,
    )
    autograd_region = build_path_c_aot_autograd_region(fwd_region)
    prim = path_c_fusion_schedule_template(autograd_region)
    try_specialize_and_build(prim, 0)
    try_specialize_and_build(prim, 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
