"""PR-3 deliverable (3): wire set_region_kernel_driver to call the REAL
tilelang.compile'd JITKernel for a path_c region on the live Metal device, driven
THROUGH the bank tensors. This proves the external-function (call_dps_packed)
boundary executes end-to-end with the real kernel -- not the numpy stand-in.

The kernel ABI (17 params, from pr2_compile_full / introspection):
  [0] activation bank   (44,957,696 f32)
  [1] parameter bank    (94,576,008 f32)
  [2] state bank        (253,042,656 f32)
  [3] activation-grad   (29,360,128 f32)
  [4] parameter-grad    (109,714,824 f32)
  [5] run_backward      (int32 scalar)
  [6..16] 11 auxiliary route scratch buffers (mamba/m2rnn intermediates)

The driver builds all 17 params as Metal tensors (the 5 banks come from the
adapter's bank dict; the scalar + scratch are constructed at the kernel-declared
shapes), invokes the JITKernel on Metal, and copies the bank results back. This is
the on-device path the doc's remaining-step (1) calls for.

RULE #1 (fail loud): if the kernel does not compile, or the device call raises, or a
bank shape mismatches the kernel ABI, we RAISE -- no silent fallback."""
from __future__ import annotations
import sys
import time
import numpy as np

import tvm
import tvm_ffi
from tvm import tir

from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical, parse_physical_bank_shapes, prim_bank_param_order,
    PathCRegionLeaf, set_region_kernel_driver, make_region_dps_packed,
    register_region_dps_packed, make_real_kernel_driver,
)
from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region, build_path_c_model_region_from_route_symbols,
    _compile_tilelang_prim_func, _tilelang_compile_pass_configs_for_prim_func,
    _path_c_default_target,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


def build_real_kernel():
    cfg = local_gb10_quarter_profile().hybrid_config()
    region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg)
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    target = _path_c_default_target()
    t0 = time.time()
    kernel = _compile_tilelang_prim_func(
        prim, target=target, execution_backend="tvm_ffi",
        pass_configs=_tilelang_compile_pass_configs_for_prim_func(prim) or None)
    print(f"  tilelang.compile (target={target}) OK in {time.time()-t0:.2f}s "
          f"-> JITKernel, {len(kernel.params)} params, {len(kernel.get_kernel_source())} "
          f"bytes MSL")
    return prim, kernel


def main() -> int:
    print("PR-3 (3) -- REAL tilelang JITKernel through the call_dps_packed boundary")
    print("Device: Metal (local).  Building the real MR kernel...")
    prim, kernel = build_real_kernel()
    lmap = parse_logical_to_physical(prim)
    bank_shapes = parse_physical_bank_shapes(prim)
    order = prim_bank_param_order(prim)

    # Drive ONE forward region through the adapter's pack/unpack with the REAL kernel.
    leaf = PathCRegionLeaf(
        name="mr_path_c_fwd", run_backward=0, prim=prim, kernel=kernel,
        logical_map=lmap, bank_shapes=bank_shapes, bank_param_order=order,
        logical_inputs=("route_0_M_hidden",), logical_output="route_0_M_hidden_after",
    )
    # Use the first-class on-device driver from path_c_dps_adapter (PR-3 (3)).
    driver = make_real_kernel_driver(leaf, tvm.metal(0))
    set_region_kernel_driver(driver)
    packed = make_region_dps_packed(leaf)

    # logical input/output tensors (real ABI shapes)
    m_in = lmap["route_0_M_hidden"]
    m_out = lmap["route_0_M_hidden_after"]
    x = (np.random.rand(*m_in.logical_shape).astype("float32") - 0.5) * 0.01
    out = np.zeros(m_out.logical_shape, np.float32)

    print(f"  driving region fwd: logical in {m_in.logical_shape} -> out "
          f"{m_out.logical_shape} via REAL Metal kernel...")
    t0 = time.time()
    packed(tvm_ffi.from_dlpack(x), tvm_ffi.from_dlpack(out))
    print(f"  REAL kernel executed on Metal in {time.time()-t0:.2f}s; "
          f"output stats: mean={out.mean():.3e} std={out.std():.3e} "
          f"nonzero={np.count_nonzero(out)}/{out.size}")
    print("PASS: the REAL path_c JITKernel runs on Metal THROUGH the adapter's "
          "pack/unpack ABI (the call_dps_packed external boundary executes the real "
          "kernel end-to-end, not the numpy stand-in).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
