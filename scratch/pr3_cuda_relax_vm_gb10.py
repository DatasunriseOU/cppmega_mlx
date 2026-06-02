"""Stage B: a REAL path_c MR region as an R.call_dps_packed LEAF inside ONE Relax
@R.function, COMPILED FOR target=cuda and RUN ON relax.VirtualMachine(ex, tvm.cuda(0)).
The packed func runs the REAL tilelang-compiled CUDA JITKernel (via make_real_kernel_driver)
through the adapter pack/unpack. Proves the call_dps_packed external-function boundary
executes the real path_c-CUDA kernel end-to-end ON gb10 THROUGH the Relax graph + CUDA VM.

Reference check: the SAME real kernel run via the direct adapter driver (Stage A path,
CPU-imported input) -- the VM graph output must match it (the graph boundary must execute
the identical real compute, not a stand-in)."""
from __future__ import annotations
import sys, time
import numpy as np
import tvm, tvm_ffi
from tvm import relax

from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical, parse_physical_bank_shapes, prim_bank_param_order,
    PathCRegionLeaf, set_region_kernel_driver, make_region_dps_packed,
    register_region_dps_packed, make_real_kernel_driver, emit_region_call,
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
    print(f"  tilelang.compile(target=cuda) OK in {time.time()-t0:.2f}s -> "
          f"{len(kernel.params)} params, {len(kernel.get_kernel_source())} bytes CUDA-C")
    return prim, kernel


def main() -> int:
    print("Stage B -- REAL path_c MR call_dps_packed leaf on the CUDA Relax VM (gb10)")
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

    # Install the REAL on-device CUDA driver + register the region as a packed func.
    driver = make_real_kernel_driver(leaf, dev)
    set_region_kernel_driver(driver)
    packed_name = "pathc.mr_fwd.cuda"
    register_region_dps_packed(leaf, packed_name)

    m_in = lmap["route_0_M_hidden"]; m_out = lmap["route_0_M_hidden_after"]
    in_sinfo = relax.TensorStructInfo(tuple(int(d) for d in m_in.logical_shape), "float32")
    out_sinfo = relax.TensorStructInfo(tuple(int(d) for d in m_out.logical_shape), "float32")

    # Assemble the @R.function: ONE call_dps_packed leaf (the real region).
    bb = relax.BlockBuilder()
    xv = relax.Var("x", in_sinfo)
    with bb.function("train_step", [xv]):
        with bb.dataflow():
            y = emit_region_call(bb, packed_name, [xv], out_sinfo)
            out = bb.emit_output(y)
        bb.emit_func_output(out)
    mod = bb.get()
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: assembled CUDA call_dps_packed step not well-formed")
    print("  @R.function well_formed: True; compiling for target=cuda + CUDA VM...")

    ex = tvm.compile(mod, target=tvm.target.Target("cuda"))
    vm = relax.VirtualMachine(ex, dev)

    # Input: a CUDA device tensor fed to the VM (the real graph entry on gb10).
    x_np = ((np.random.default_rng(0).random(m_in.logical_shape, np.float32) - 0.5)
            * 0.01).astype(np.float32)
    x_cuda = tvm.runtime.tensor(x_np, device=dev)
    print(f"  running CUDA VM train_step: in {m_in.logical_shape} (device={x_cuda.device})...")
    t0 = time.time()
    res = vm["train_step"](x_cuda)
    dev.sync()
    out_vm = res.numpy()   # CUDA -> host
    dt = time.time()-t0
    nz = int(np.count_nonzero(out_vm))
    print(f"  CUDA VM executed in {dt:.2f}s; out device was {res.device}; "
          f"mean={out_vm.mean():.3e} std={out_vm.std():.3e} nonzero={nz}/{out_vm.size}")
    if nz == 0:
        raise RuntimeError("FAIL-LOUD: CUDA VM call_dps_packed produced ALL-ZERO output")

    # Reference: SAME real kernel via the direct adapter packed (host-imported input).
    set_region_kernel_driver(make_real_kernel_driver(leaf, dev))
    direct = make_region_dps_packed(leaf)
    ref_out = np.zeros(m_out.logical_shape, np.float32)
    direct(tvm_ffi.from_dlpack(x_np), tvm_ffi.from_dlpack(ref_out))
    maxdiff = float(np.abs(out_vm - ref_out).max())
    print(f"  REF (same real CUDA kernel via direct adapter): nonzero="
          f"{int(np.count_nonzero(ref_out))}; max|VM - REF| = {maxdiff:.3e}")
    if not np.allclose(out_vm, ref_out, rtol=1e-4, atol=1e-6):
        raise RuntimeError(f"FAIL-LOUD: CUDA VM graph output disagrees with the same "
                           f"real kernel run directly; max abs diff={maxdiff}")
    print(f"STAGE-B PASS: a REAL path_c MR region runs THROUGH the Relax R.call_dps_packed "
          f"graph on the gb10 CUDA VM ({nz} nonzero, byte-matching the direct real-kernel "
          f"run, max diff {maxdiff:.1e}). The external-function boundary executes the REAL "
          f"tilelang-CUDA kernel end-to-end on gb10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
