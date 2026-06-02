"""Compile the REAL path_c prim AS-IS (no curry, 17 params) via tilelang.compile
on Metal -- its real production path -- to confirm the kernel lowers to a callable
packed function (the external boundary the Relax adapter must use, since generic
s_tir RAISES). Print the callable + its declared out_idx / param count."""
from __future__ import annotations
import sys, time, traceback
import tvm
from tvm import tir

from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region,
    build_path_c_model_region_from_route_symbols,
    _compile_tilelang_prim_func,
    _tilelang_compile_pass_configs_for_prim_func,
    _path_c_default_target,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


def main() -> int:
    cfg = local_gb10_quarter_profile().hybrid_config()
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg,
    )
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(fwd_region))
    print("prim params:", len(prim.params), " out_idx:", list(prim.attrs.get("tilelang_out_idx")))
    target = _path_c_default_target()
    pass_configs = _tilelang_compile_pass_configs_for_prim_func(prim)
    t0 = time.time()
    try:
        kernel = _compile_tilelang_prim_func(
            prim, target=target, execution_backend="tvm_ffi",
            pass_configs=pass_configs or None,
        )
        print(f"tilelang.compile OK in {time.time()-t0:.1f}s ; kernel type:", type(kernel).__name__)
        # Inspect the callable surface.
        for attr in ("get_kernel_source", "params", "out_idx", "func", "prim_func"):
            print(f"  has {attr}:", hasattr(kernel, attr))
        src = None
        try:
            src = kernel.get_kernel_source()
        except Exception:
            pass
        if src:
            print("  kernel source bytes:", len(src), " (Metal MSL emitted)")
    except Exception:
        print(f"tilelang.compile FAILED after {time.time()-t0:.1f}s:")
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
