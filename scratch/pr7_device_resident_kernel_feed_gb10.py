"""PR-7 device-DLPack primitive validation (isolation, BEFORE the rework).

Proves a REAL path_c MR tilelang JITKernel can be driven on gb10 tvm.cuda(0) with
its INPUT/OUTPUT bank buffers DEVICE-RESIDENT (tvm.cuda(0) nd-arrays), with NO numpy
host copy in the per-call hot path -- and times it against the current numpy-staged
make_real_kernel_driver path (the per-region per-call speedup ceiling for the rework).

Reuses the proven scaffold of scratch/pr3_cuda_real_kernel_driver_gb10.py.
RULE #1: real measured results, fail-loud, device == numpy within fp tolerance.
"""
from __future__ import annotations
import sys, time
import numpy as np
import tvm

from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical, parse_physical_bank_shapes, prim_bank_param_order,
    PathCRegionLeaf, make_real_kernel_driver,
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
    kernel = _compile_tilelang_prim_func(
        prim, target=target, execution_backend="tvm_ffi",
        pass_configs=_tilelang_compile_pass_configs_for_prim_func(prim) or None)
    print(f"  tilelang.compile(cuda) OK -> JITKernel, {len(kernel.params)} params, "
          f"out_idx={list(kernel.out_idx)}")
    return prim, kernel


def main() -> int:
    print("PR-7 device-resident kernel feed -- REAL path_c MR JITKernel on tvm.cuda(0)")
    dev = tvm.cuda(0)
    print("TVM:", tvm.__version__, "cuda.exist=", dev.exist)
    if not dev.exist:
        raise RuntimeError("FAIL: tvm.cuda(0) not present")
    prim, kernel = build_real_kernel()
    lmap = parse_logical_to_physical(prim)
    bank_shapes = parse_physical_bank_shapes(prim)
    order = prim_bank_param_order(prim)
    leaf = PathCRegionLeaf(
        name="mr_path_c_fwd", run_backward=0, prim=prim, kernel=kernel,
        logical_map=lmap, bank_shapes=bank_shapes, bank_param_order=order,
        logical_inputs=("route_0_M_hidden",), logical_output="route_0_M_hidden_after")

    kparams = list(kernel.params)
    out_idx = set(int(x) for x in kernel.out_idx)
    bank_pos = {name: i for i, name in enumerate(order[:5])}
    gate_pos = next(i for i, p in enumerate(kparams) if len(list(p.shape)) == 0)
    m_in = lmap["route_0_M_hidden"]; m_out = lmap["route_0_M_hidden_after"]
    print(f"  CALL CONVENTION: kernel(*args), {len(kparams)} positional args; "
          f"banks->params {bank_pos}; scalar gate (run_backward) at idx {gate_pos}; "
          f"DPS out_idx={sorted(out_idx)} (kernel writes IN PLACE into these arg buffers)")

    # ---- seed input host bank state (the packed logical input -> its bank sub-range) ----
    x = (np.random.rand(*m_in.logical_shape).astype("float32") - 0.5) * 0.01
    host_banks = {b: np.zeros((n,), np.float32) for b, n in bank_shapes.items()}
    host_banks[m_in.bank][m_in.offset:m_in.offset + m_in.size] = x.reshape(-1)

    def out_of(banks_dict):
        m = m_out
        return banks_dict[m.bank][m.offset:m.offset + m.size].copy()

    N = 30

    # =================================================================== #
    # PATH A: NUMPY-STAGED (current make_real_kernel_driver) -- per call:
    #   5x tvm.runtime.tensor(np.ascontiguousarray(host_bank), dev)  [H2D]
    #   kernel(*args); dev.sync()
    #   args[pos].numpy()  for each out bank  [D2H]
    # =================================================================== #
    numpy_driver = make_real_kernel_driver(leaf, dev)
    banks_np = {k: v.copy() for k, v in host_banks.items()}
    numpy_driver(leaf, banks_np)  # warmup
    out_numpy = out_of(banks_np)
    t0 = time.time()
    for _ in range(N):
        b = {k: v.copy() for k, v in host_banks.items()}  # fresh host banks each step (as in adapter)
        numpy_driver(leaf, b)
    dev.sync()
    t_numpy = (time.time() - t0) / N

    # =================================================================== #
    # PATH B: DEVICE-RESIDENT -- banks allocated ONCE as tvm.cuda(0) tensors;
    #   per call: kernel(*dev_args); dev.sync().  NO H2D, NO D2H, NO numpy.
    # =================================================================== #
    dev_args = [None] * len(kparams)
    for name, pos in bank_pos.items():
        dev_args[pos] = tvm.runtime.tensor(
            np.ascontiguousarray(host_banks[name], np.float32), device=dev)  # ONCE
    dev_args[gate_pos] = int(leaf.run_backward)
    for i in range(len(kparams)):
        if dev_args[i] is not None or i == gate_pos:
            continue
        shp = [int(d) for d in kparams[i].shape]
        dev_args[i] = tvm.runtime.tensor(np.zeros(shp, np.float32), device=dev)  # ONCE

    kernel(*dev_args); dev.sync()  # warmup
    out_dev_bank = dev_args[bank_pos[m_out.bank]]
    out_device = out_dev_bank.numpy()[m_out.offset:m_out.offset + m_out.size].copy()  # readback ONLY for check
    t0 = time.time()
    for _ in range(N):
        kernel(*dev_args)   # device buffers fed directly, zero host traffic
    dev.sync()
    t_dev = (time.time() - t0) / N

    nz = int(np.count_nonzero(out_device))
    max_abs_diff = float(np.abs(out_numpy - out_device).max())
    print(f"\n  device-resident out: mean={out_device.mean():.3e} std={out_device.std():.3e} "
          f"nonzero={nz}/{out_device.size}")
    print(f"  numpy-staged    out: mean={out_numpy.mean():.3e} std={out_numpy.std():.3e}")
    print(f"  max|device - numpy| = {max_abs_diff:.3e}  (numerically equivalent if ~0)")
    print(f"\n  per-call NUMPY-STAGED : {t_numpy*1e3:9.3f} ms")
    print(f"  per-call DEVICE-RESID : {t_dev*1e3:9.3f} ms")
    print(f"  SPEEDUP CEILING       : {t_numpy/t_dev:9.2f}x  "
          f"(host-staging eliminated per region per call)")

    if nz == 0:
        raise RuntimeError("FAIL-LOUD: device-resident kernel produced ALL-ZERO output")
    if max_abs_diff > 1e-3:
        raise RuntimeError(f"FAIL-LOUD: device path != numpy path, max diff {max_abs_diff:.3e}")
    print("\nPR-7 PASS: REAL path_c MR JITKernel driven with DEVICE-RESIDENT tvm.cuda(0) "
          "bank buffers, NO numpy host copy in the call, numerically == numpy path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
