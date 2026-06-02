"""Stage A: REAL path_c MR tilelang JITKernel through the call_dps_packed adapter
pack/unpack on tvm.cuda(0) (gb10). Direct mirror of scratch/pr3_real_kernel_driver.py
(proven on Metal: 14.68M nonzero) but on CUDA. Proves the on-device real-kernel path
of the adapter works on gb10 CUDA."""
from __future__ import annotations
import sys, time
import numpy as np
import tvm, tvm_ffi

from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical, parse_physical_bank_shapes, prim_bank_param_order,
    PathCRegionLeaf, set_region_kernel_driver, make_region_dps_packed,
    make_real_kernel_driver,
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
    assert target == "cuda", f"expected cuda target on gb10, got {target}"
    t0 = time.time()
    kernel = _compile_tilelang_prim_func(
        prim, target=target, execution_backend="tvm_ffi",
        pass_configs=_tilelang_compile_pass_configs_for_prim_func(prim) or None)
    src = kernel.get_kernel_source()
    print(f"  tilelang.compile(target=cuda) OK in {time.time()-t0:.2f}s -> JITKernel, "
          f"{len(kernel.params)} params, {len(src)} bytes CUDA-C")
    # report smem usage hint
    nshared = src.count("__shared__")
    print(f"  kernel src has {nshared} __shared__ decls")
    return prim, kernel


def main() -> int:
    print("Stage A -- REAL path_c MR JITKernel through call_dps_packed adapter on tvm.cuda(0)")
    dev = tvm.cuda(0)
    print("TVM:", tvm.__version__, "cuda.exist=", dev.exist)
    if not dev.exist:
        raise RuntimeError("FAIL: tvm.cuda(0) not present")
    prim, kernel = build_real_kernel()
    lmap = parse_logical_to_physical(prim)
    bank_shapes = parse_physical_bank_shapes(prim)
    order = prim_bank_param_order(prim)
    print(f"  ABI: {len(prim.params)} params, {len(lmap)} logical tensors, "
          f"{len(bank_shapes)} banks; bank order[:5]={order[:5]}")

    leaf = PathCRegionLeaf(
        name="mr_path_c_fwd", run_backward=0, prim=prim, kernel=kernel,
        logical_map=lmap, bank_shapes=bank_shapes, bank_param_order=order,
        logical_inputs=("route_0_M_hidden",), logical_output="route_0_M_hidden_after")

    driver = make_real_kernel_driver(leaf, dev)   # <-- on tvm.cuda(0)
    set_region_kernel_driver(driver)
    packed = make_region_dps_packed(leaf)

    m_in = lmap["route_0_M_hidden"]; m_out = lmap["route_0_M_hidden_after"]
    x = (np.random.rand(*m_in.logical_shape).astype("float32") - 0.5) * 0.01
    out = np.zeros(m_out.logical_shape, np.float32)
    print(f"  driving region fwd: in {m_in.logical_shape} -> out {m_out.logical_shape} "
          f"via REAL CUDA kernel...")
    t0 = time.time()
    packed(tvm_ffi.from_dlpack(x), tvm_ffi.from_dlpack(out))
    dt = time.time()-t0
    nz = int(np.count_nonzero(out))
    print(f"  REAL kernel executed on CUDA in {dt:.2f}s; out mean={out.mean():.3e} "
          f"std={out.std():.3e} nonzero={nz}/{out.size}")
    if nz == 0:
        raise RuntimeError("FAIL-LOUD: real CUDA kernel produced ALL-ZERO output")
    print(f"STAGE-A PASS: REAL path_c MR JITKernel runs on gb10 CUDA THROUGH the "
          f"call_dps_packed adapter pack/unpack ({nz} nonzero outputs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
