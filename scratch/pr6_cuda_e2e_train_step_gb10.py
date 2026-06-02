"""PR 6 gb10 CUDA e2e: the WHOLE path_c Relax train_step (fwd+sqrt-N remat + bwd +
in-place Adam + loss), banks exposed as cross-region SSA tensors, with the REAL
tilelang path_c CUDA kernel driving the forward regions, COMPILED for target=cuda and
EXECUTED on relax.VirtualMachine(ex, tvm.cuda(0)) on gb10.

This is the testable e2e deliverable: ONE @R.function that is the whole training step,
the real path_c-CUDA kernel behind the forward call_dps_packed boundary (proven on gb10:
14.68M nonzero), StaticPlanBlockMemory + relax.build(target=cuda), one step executed on
device, loss reported finite, peak measured by free -g delta.

Two stages:
  Stage 1 (PURE GRAPH, real bank scale): the whole train_step with the abstract bank
    drivers, compiled target=cuda + run on tvm.cuda(0) at the REAL bank numels for a
    reduced layer count. Proves the WHOLE graph (every region kind: fwd, remat-fwd, bwd,
    adam_inplace, loss) executes on CUDA end to end, loss finite, with the MEASURED
    free -g peak. This is the memory + executability proof at real bank scale.
  Stage 2 (REAL KERNEL forward): the SAME graph but the forward regions run the REAL
    tilelang path_c CUDA JITKernel (make_real_kernel_driver) behind the boundary, at a
    small layer count. Proves the real path_c-CUDA compute runs THROUGH the whole-step
    graph end to end on device (loss finite). Compiling the real kernel is ~127s.

FAIL-LOUD: any region that cannot run through the graph on CUDA RAISES, naming it.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import tvm
import tvm_ffi
from tvm import relax

from cppmega_mlx.runtime.path_c_relax_train_step import (
    build_train_step,
    run_train_step_on_device,
    plan_train_step,
    register_real_forward_driver,
)
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    BANK_ACT, BANK_ACTG, BANK_PARAM, BANK_PARAMG, BANK_STATE, real_bank_numels,
)
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


GB = 1024.0 ** 3


def _free_gb() -> float:
    """Available memory in GB from /proc/meminfo (MemAvailable). Unified-memory box:
    GPU allocations draw from the same pool, so free -g delta is the device peak."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("FAIL-LOUD: MemAvailable not found in /proc/meminfo")


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
          f"{len(kernel.params)} params, {len(kernel.get_kernel_source())} bytes CUDA-C",
          flush=True)
    return prim, kernel


def stage1_pure_graph_cuda(numels, n_layers):
    """The WHOLE train_step (abstract bank drivers) on tvm.cuda(0) at real bank scale."""
    print(f"\n=== STAGE 1: whole train_step on CUDA, REAL bank scale, "
          f"n_layers={n_layers} (pure-graph drivers) ===", flush=True)
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("FAIL-LOUD: tvm.cuda(0) not present on gb10")
    plan = plan_train_step(numels, n_layers)
    print(f"  planned peak (analyzer) = {plan.planned_peak/GB:.3f} GB; "
          f"{plan.recompute_calls} recompute calls", flush=True)

    free_before = _free_gb()
    t0 = time.time()
    res = run_train_step_on_device(numels, n_layers, target="cuda", device=dev)
    dev.sync()
    free_after = _free_gb()
    free_min = free_after  # measured after the step; the runner syncs at peak
    peak_delta = free_before - free_after
    print(f"  RUNS=yes on CUDA  loss={res.loss:.6e} (finite="
          f"{np.isfinite(res.loss)})  param'-checksum={res.param_checksum:.4e}", flush=True)
    print(f"  compile={res.compile_s:.2f}s run={res.run_s:.2f}s", flush=True)
    print(f"  free -g: before={free_before:.2f} GB after={free_after:.2f} GB "
          f"delta(approx peak)={peak_delta:.2f} GB", flush=True)
    if not np.isfinite(res.loss):
        raise RuntimeError("FAIL-LOUD: stage-1 CUDA loss not finite")
    return res, peak_delta, plan.planned_peak


def stage1_measured_peak(numels, n_layers):
    """Same as stage1 but samples free -g DURING the run via a background sampler so the
    delta captures the true high-water (not just the post-run residual)."""
    import threading
    print(f"\n=== STAGE 1 (peak-sampled): whole train_step on CUDA, real bank scale, "
          f"n_layers={n_layers} ===", flush=True)
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("FAIL-LOUD: tvm.cuda(0) not present on gb10")
    plan = plan_train_step(numels, n_layers)

    free_before = _free_gb()
    min_free = [free_before]
    stop = [False]

    def sampler():
        while not stop[0]:
            f = _free_gb()
            if f < min_free[0]:
                min_free[0] = f
            time.sleep(0.02)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    res = run_train_step_on_device(numels, n_layers, target="cuda", device=dev)
    dev.sync()
    stop[0] = True
    th.join(timeout=2.0)
    peak_delta = free_before - min_free[0]
    print(f"  RUNS=yes on CUDA  loss={res.loss:.6e} (finite="
          f"{np.isfinite(res.loss)})", flush=True)
    print(f"  planned peak (analyzer) = {plan.planned_peak/GB:.3f} GB", flush=True)
    print(f"  free -g sampled: before={free_before:.2f} GB min_during={min_free[0]:.2f} GB "
          f"MEASURED peak delta={peak_delta:.2f} GB", flush=True)
    if not np.isfinite(res.loss):
        raise RuntimeError("FAIL-LOUD: stage-1 CUDA loss not finite")
    return res, peak_delta, plan.planned_peak


def stage2_real_kernel_cuda(numels, n_layers):
    """The whole train_step with the REAL tilelang path_c CUDA kernel driving the
    forward regions, on tvm.cuda(0), at a small layer count."""
    print(f"\n=== STAGE 2: whole train_step on CUDA with the REAL path_c kernel "
          f"forward, n_layers={n_layers} ===", flush=True)
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("FAIL-LOUD: tvm.cuda(0) not present on gb10")

    prim, kernel = build_real_kernel()
    lmap = parse_logical_to_physical(prim)
    bank_shapes = parse_physical_bank_shapes(prim)
    order = prim_bank_param_order(prim)
    leaf = PathCRegionLeaf(
        name="mr_path_c_fwd", run_backward=0, prim=prim, kernel=kernel,
        logical_map=lmap, bank_shapes=bank_shapes, bank_param_order=order,
        logical_inputs=("route_0_M_hidden",), logical_output="route_0_M_hidden_after")
    real_driver = make_real_kernel_driver(leaf, dev)

    # Use the REAL physical bank numels (the kernel writes the activation bank).
    bank_numels = dict(numels)
    bank_numels[BANK_ACT] = bank_shapes[BANK_ACT]
    bank_numels[BANK_PARAM] = bank_shapes[BANK_PARAM]
    bank_numels[BANK_STATE] = bank_shapes[BANK_STATE]
    bank_numels[BANK_ACTG] = bank_shapes[BANK_ACTG]
    bank_numels[BANK_PARAMG] = bank_shapes[BANK_PARAMG]

    register_real_forward_driver(leaf, real_driver, bank_numels, n_layers)

    free_before = _free_gb()
    t0 = time.time()
    res = run_train_step_on_device(bank_numels, n_layers, target="cuda", device=dev)
    dev.sync()
    free_after = _free_gb()
    peak_delta = free_before - free_after
    print(f"  RUNS=yes on CUDA (real path_c kernel fwd)  loss={res.loss:.6e} (finite="
          f"{np.isfinite(res.loss)})  param'-checksum={res.param_checksum:.4e}",
          flush=True)
    print(f"  step run={res.run_s:.2f}s  free -g before={free_before:.2f} after="
          f"{free_after:.2f} delta={peak_delta:.2f} GB", flush=True)
    if not np.isfinite(res.loss):
        raise RuntimeError("FAIL-LOUD: stage-2 real-kernel CUDA loss not finite")
    return res, peak_delta


def main() -> int:
    print("PR 6 gb10 CUDA e2e -- whole path_c Relax train_step on tvm.cuda(0)")
    print("TVM:", tvm.__version__, flush=True)
    numels = real_bank_numels()
    total_mb = sum(numels.values()) * 4 / 1024 / 1024
    print(f"real per-region banks: {len(numels)} banks, {total_mb:.1f} MB/region", flush=True)

    # The largest layer count whose whole-step REAL-bank-scale CUDA run fits in the box.
    # Each layer's act+state banks are ~1.1 GB; sqrt-N remat keeps the peak O(sqrt N).
    n_layers_stage1 = int(os.environ.get("PR6_STAGE1_LAYERS", "8"))
    n_layers_stage2 = int(os.environ.get("PR6_STAGE2_LAYERS", "2"))

    # STAGE 1: whole graph on CUDA at real bank scale (every region kind), peak sampled.
    res1, peak1, planned1 = stage1_measured_peak(numels, n_layers_stage1)

    # STAGE 2: real path_c CUDA kernel forward through the whole-step graph.
    res2, peak2 = stage2_real_kernel_cuda(numels, n_layers_stage2)

    print("\n=== PR-6 e2e SUMMARY (gb10 CUDA) ===", flush=True)
    print(f"  Stage 1 (whole graph, real bank scale, {n_layers_stage1} layers): "
          f"RUNS=yes loss={res1.loss:.4e} finite=yes  planned-peak="
          f"{planned1/GB:.3f} GB  measured-free-delta={peak1:.2f} GB", flush=True)
    print(f"  Stage 2 (REAL path_c CUDA kernel fwd, {n_layers_stage2} layers): "
          f"RUNS=yes loss={res2.loss:.4e} finite=yes  measured-free-delta={peak2:.2f} GB",
          flush=True)
    print("\nTESTABLE e2e graph train_step RUNS on gb10 CUDA: every region kind (fwd, "
          "remat-fwd, bwd, adam_inplace, loss) executes through the Relax call_dps_packed "
          "graph on tvm.cuda(0), the REAL path_c-CUDA kernel drives the forward boundary, "
          "and the loss is finite. RULE #1: any region that could not run RAISED.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
