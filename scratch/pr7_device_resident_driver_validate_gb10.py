"""PR-7 VALIDATION of the DEVICE-RESIDENT DPS driver rework (AFTER the rework).

Proves on gb10 tvm.cuda(0), with the REAL path_c MR tilelang JITKernel:

  (A) NUMERIC EQUIVALENCE: the reworked DEVICE-RESIDENT bank-forward driver
      (make_real_bank_forward_driver device path: banks stay tvm.cuda(0) tensors,
      pack/unpack are device VIEW + device->device copies, kernel mutates banks in
      place, NO .numpy() in the hot path) produces the SAME act_out + state_out as the
      NUMPY-STAGED reference path -- max abs diff within fp tolerance.

  (B) E2E TRAIN STEP: the whole Relax train_step (fwd[device-resident real kernel] +
      remat + bwd + in-place Adam + loss) RUNS on tvm.cuda(0) and the loss is FINITE,
      compile-once / run-many over several steps, with the device-resident forward.

  (C) SPEEDUP: per-forward-call device-resident vs numpy-staged wall (the host-bounce
      removal at the bank-forward driver), reported.

RULE #1: real measured results, fail-loud, device == numpy within fp tolerance.
Single-run discipline: ONE process, banks freed at exit.
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import tvm

from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical, parse_physical_bank_shapes, prim_bank_param_order,
    PathCRegionLeaf, make_real_kernel_driver, make_device_resident_kernel_driver,
    alloc_device_banks,
)
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    BANK_ACT, BANK_ACTG, BANK_PARAM, BANK_PARAMG, BANK_STATE, real_bank_numels,
    bank_arg_is_device,
)
from cppmega_mlx.runtime.path_c_relax_train_step import (
    make_real_bank_forward_driver, register_real_forward_driver,
    build_train_step, register_loss_driver,
)
from cppmega_mlx.runtime.path_c_relax_step_optim import register_optim_driver
from cppmega_mlx.runtime.path_c_relax_step_banks import register_bank_drivers
from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region, build_path_c_model_region_from_route_symbols,
    _compile_tilelang_prim_func, _tilelang_compile_pass_configs_for_prim_func,
    _path_c_default_target,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile

import tvm_ffi
from tvm import relax

GB = 1024.0 ** 3


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
    print(f"  tilelang.compile(cuda) OK in {time.time()-t0:.1f}s -> {len(kernel.params)} "
          f"params, out_idx={list(kernel.out_idx)}", flush=True)
    return prim, kernel


def make_leaf(prim, kernel):
    return PathCRegionLeaf(
        name="mr_path_c_fwd", run_backward=0, prim=prim, kernel=kernel,
        logical_map=parse_logical_to_physical(prim),
        bank_shapes=parse_physical_bank_shapes(prim),
        bank_param_order=prim_bank_param_order(prim),
        logical_inputs=("route_0_M_hidden",),
        logical_output="route_0_M_hidden_after")


def part_A_numeric_equivalence(leaf, dev):
    """Run the device-resident bank-forward driver and the numpy-staged one on the SAME
    act/param SSA banks; assert act_out + state_out match within fp tolerance."""
    print("\n=== (A) NUMERIC EQUIVALENCE: device-resident vs numpy-staged forward ===",
          flush=True)
    bank_numels = dict(real_bank_numels())
    bs = leaf.bank_shapes
    for k in (BANK_ACT, BANK_PARAM, BANK_STATE, BANK_ACTG, BANK_PARAMG):
        bank_numels[k] = bs[k]
    act_n = bank_numels[BANK_ACT]
    param_n = bank_numels[BANK_PARAM]
    state_n = bank_numels[BANK_STATE]

    rng = np.random.default_rng(0)
    act_in = (rng.random(act_n, np.float32) - 0.5).astype(np.float32) * 0.01
    param_in = (rng.random(param_n, np.float32) - 0.5).astype(np.float32) * 0.01

    def _dev(x):
        return tvm.runtime.tensor(np.ascontiguousarray(x, np.float32), device=dev)

    # ---- numpy-staged path (reference) ----
    numpy_driver = make_real_kernel_driver(leaf, dev)
    fwd_numpy = make_real_bank_forward_driver(leaf, numpy_driver, bank_numels, device=None)
    # force the host path: feed it CPU tensors so bank_arg_is_device(act_in)==False.
    act_in_cpu = tvm.runtime.tensor(act_in, device=tvm.cpu())
    param_cpu = tvm.runtime.tensor(param_in, device=tvm.cpu())
    ao_cpu = tvm.runtime.empty((act_n,), "float32", device=tvm.cpu())
    so_cpu = tvm.runtime.empty((state_n,), "float32", device=tvm.cpu())
    assert not bank_arg_is_device(act_in_cpu)
    fwd_numpy(act_in_cpu, param_cpu, act_in_cpu, ao_cpu, so_cpu)
    ao_ref = ao_cpu.numpy().reshape(-1)
    so_ref = so_cpu.numpy().reshape(-1)

    # ---- device-resident path ----
    fwd_dev = make_real_bank_forward_driver(leaf, numpy_driver, bank_numels, device=dev)
    act_in_d = _dev(act_in)
    param_d = _dev(param_in)
    ao_d = tvm.runtime.empty((act_n,), "float32", device=dev)
    so_d = tvm.runtime.empty((state_n,), "float32", device=dev)
    assert bank_arg_is_device(act_in_d), "FAIL: device tensor not detected as device"
    fwd_dev(act_in_d, param_d, act_in_d, ao_d, so_d)
    dev.sync()
    ao_got = ao_d.numpy().reshape(-1)
    so_got = so_d.numpy().reshape(-1)

    d_ao = float(np.abs(ao_ref - ao_got).max())
    d_so = float(np.abs(so_ref - so_got).max())
    nz_ao = int(np.count_nonzero(ao_got))
    print(f"  act_out  : nonzero={nz_ao}/{ao_got.size}  max|dev-numpy|={d_ao:.3e}", flush=True)
    print(f"  state_out:                          max|dev-numpy|={d_so:.3e}", flush=True)
    if nz_ao == 0:
        raise RuntimeError("FAIL-LOUD: device-resident forward produced ALL-ZERO act_out")
    if d_ao > 1e-3 or d_so > 1e-3:
        raise RuntimeError(
            f"FAIL-LOUD: device-resident != numpy-staged (act {d_ao:.3e}, state {d_so:.3e})")
    print("  (A) PASS: device-resident forward == numpy-staged forward (within fp tol).",
          flush=True)
    return bank_numels


def part_C_speedup(leaf, dev, bank_numels, n=20):
    print("\n=== (C) PER-FORWARD-CALL SPEEDUP: device-resident vs numpy-staged ===",
          flush=True)
    act_n = bank_numels[BANK_ACT]; param_n = bank_numels[BANK_PARAM]
    state_n = bank_numels[BANK_STATE]
    rng = np.random.default_rng(1)
    act = (rng.random(act_n, np.float32) - 0.5).astype(np.float32) * 0.01
    param = (rng.random(param_n, np.float32) - 0.5).astype(np.float32) * 0.01

    numpy_driver = make_real_kernel_driver(leaf, dev)

    # numpy-staged (host path): feed CPU tensors each call.
    fwd_numpy = make_real_bank_forward_driver(leaf, numpy_driver, bank_numels, device=None)
    a_cpu = tvm.runtime.tensor(act, device=tvm.cpu())
    p_cpu = tvm.runtime.tensor(param, device=tvm.cpu())
    ao_cpu = tvm.runtime.empty((act_n,), "float32", device=tvm.cpu())
    so_cpu = tvm.runtime.empty((state_n,), "float32", device=tvm.cpu())
    fwd_numpy(a_cpu, p_cpu, a_cpu, ao_cpu, so_cpu)  # warmup
    t0 = time.time()
    for _ in range(n):
        fwd_numpy(a_cpu, p_cpu, a_cpu, ao_cpu, so_cpu)
    dev.sync()
    t_numpy = (time.time() - t0) / n

    # device-resident: feed device tensors (allocated once).
    fwd_dev = make_real_bank_forward_driver(leaf, numpy_driver, bank_numels, device=dev)
    a_d = tvm.runtime.tensor(act, device=dev)
    p_d = tvm.runtime.tensor(param, device=dev)
    ao_d = tvm.runtime.empty((act_n,), "float32", device=dev)
    so_d = tvm.runtime.empty((state_n,), "float32", device=dev)
    fwd_dev(a_d, p_d, a_d, ao_d, so_d); dev.sync()  # warmup
    t0 = time.time()
    for _ in range(n):
        fwd_dev(a_d, p_d, a_d, ao_d, so_d)
    dev.sync()
    t_dev = (time.time() - t0) / n

    print(f"  per-forward NUMPY-STAGED : {t_numpy*1e3:9.3f} ms", flush=True)
    print(f"  per-forward DEVICE-RESID : {t_dev*1e3:9.3f} ms", flush=True)
    print(f"  SPEEDUP                  : {t_numpy/max(t_dev,1e-9):9.2f}x", flush=True)
    return t_numpy, t_dev


def part_B_e2e(leaf, dev, bank_numels, n_layers, n_steps):
    """Whole train_step on CUDA with the DEVICE-RESIDENT forward, compile-once/run-many."""
    print(f"\n=== (B) E2E TRAIN STEP on CUDA (device-resident fwd), n_layers={n_layers}, "
          f"{n_steps} steps ===", flush=True)
    numpy_driver = make_real_kernel_driver(leaf, dev)
    # device=dev -> the registered fwd packed funcs take the DEVICE-RESIDENT path.
    register_real_forward_driver(leaf, numpy_driver, bank_numels, n_layers, device=dev)
    register_optim_driver(bank_numels)
    register_loss_driver()

    mod = build_train_step(bank_numels, n_layers)
    # build_train_step calls register_bank_drivers (abstract bwd/adam) + register_optim +
    # register_loss, then we RE-register the real device-resident forward to override.
    register_real_forward_driver(leaf, numpy_driver, bank_numels, n_layers, device=dev)
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: train_step IRModule not well-formed")

    t0 = time.time()
    ex = tvm.compile(mod, target=tvm.target.Target("cuda"))
    vm = relax.VirtualMachine(ex, dev)
    print(f"  compile-once: {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(0)
    def _dev(x):
        return tvm.runtime.tensor(np.ascontiguousarray(x, np.float32), device=dev)
    act0 = _dev((rng.random(bank_numels[BANK_ACT], np.float32) - 0.5) * 0.01)
    param = _dev((rng.random(bank_numels[BANK_PARAM], np.float32) - 0.5) * 0.01)
    paramg0 = _dev(np.zeros(bank_numels[BANK_PARAMG], np.float32))
    actg0 = _dev((rng.random(bank_numels[BANK_ACTG], np.float32) - 0.5) * 0.01)
    m = _dev((rng.random(bank_numels[BANK_PARAM], np.float32) * 0.1))
    v = _dev((rng.random(bank_numels[BANK_PARAM], np.float32) * 0.1))

    losses = []
    for s in range(n_steps):
        t0 = time.perf_counter()
        out = vm["train_step"](act0, param, paramg0, actg0, m, v)
        dev.sync()
        dt = time.perf_counter() - t0
        loss = float(np.asarray(out[0].numpy()).reshape(-1)[0])
        if not np.isfinite(loss):
            raise RuntimeError(f"FAIL-LOUD: step {s} loss not finite ({loss})")
        param, m, v = out[1], out[2], out[3]
        losses.append(loss)
        print(f"  step {s:>2}: {dt*1000:8.1f} ms  loss={loss:.6e}", flush=True)
    print(f"  (B) PASS: e2e train_step RUNS on CUDA with device-resident forward; "
          f"loss finite (last={losses[-1]:.6e}).", flush=True)
    return losses


def main() -> int:
    print("PR-7 VALIDATION: device-resident DPS driver rework on gb10 tvm.cuda(0)")
    print("TVM:", tvm.__version__, flush=True)
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("FAIL-LOUD: tvm.cuda(0) not present on gb10")

    prim, kernel = build_real_kernel()
    leaf = make_leaf(prim, kernel)

    bank_numels = part_A_numeric_equivalence(leaf, dev)
    part_C_speedup(leaf, dev, bank_numels, n=int(os.environ.get("PR7V_N", "20")))

    n_layers = int(os.environ.get("PR7V_LAYERS", "2"))
    n_steps = int(os.environ.get("PR7V_STEPS", "3"))
    part_B_e2e(leaf, dev, bank_numels, n_layers, n_steps)

    print("\nPR-7 VALIDATION PASS: the reworked DEVICE-RESIDENT DPS driver is numerically "
          "equivalent to the numpy-staged path, runs the whole train_step e2e on CUDA "
          "with a finite loss, and removes the per-region host bounce in the forward.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
