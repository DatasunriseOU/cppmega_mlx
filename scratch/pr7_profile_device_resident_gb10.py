"""PR 7 gb10 CUDA PROFILE -- the whole path_c Relax train_step WITH THE DEVICE-RESIDENT
FORWARD DRIVER (the host-staging-elimination lever, §4a of RELAX-GRAPH-VS-MEGATRON.md).

This is the device-resident counterpart of scratch/pr7_profile_train_step_gb10.py. The
PR-7 baseline profiled the ABSTRACT numpy host-staged forward driver (every bank
round-tripped to host numpy per region across the call_dps_packed boundary) and measured
8L: 162.2 s/step -> 25.2 tok/s, ~100% host-staging-bound, forward+remat = 96.9%. The
device-resident rework keeps the forward banks DEVICE-RESIDENT on tvm.cuda(0) end-to-end
(zero-copy device VIEW pack/unpack + device->device copies, the REAL tilelang MR JITKernel
mutating the device banks IN PLACE at out_idx -- no .numpy() in the per-region hot path).

This profiler:
  * builds the REAL MR path_c JITKernel + leaf ONCE (the same artifact the §4a validation
    proved numerically equivalent: max|device-numpy|=1.025e-06);
  * registers the DEVICE-RESIDENT forward driver (register_real_forward_driver(device=dev))
    for every pathc.bank_fwd_i region, WRAPPED with a per-region wall-time accumulator;
  * keeps the abstract numpy bwd/adam/loss drivers (timed via the same factory monkeypatch
    as the baseline) -- those are lever-4 (not yet device-resident), so the profile shows
    whether the step is now device-compute-bound (forward) or shifted to the remaining
    abstract bwd/adam staging;
  * compiles ONCE, runs N steps on the SAME VM, threads optimizer state step->step;
  * reports mean warm step -> tok/s, planned + measured peak, and the per-region breakdown.

TOKENS/STEP: 4096 (seq=4096, batch=1), identical to the baseline, so tok/s is directly
comparable to the 25.2 baseline. Megatron baseline: 3399 tok/s @ ~26 GB, 16384 tok/step.

FAIL-LOUD (RULE #1): any non-finite loss, any region that cannot run, or a device tensor
not detected as device RAISES, naming it. No fabrication -- measured only.
"""
from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass, field

import numpy as np

import tvm
import tvm_ffi
from tvm import relax

from cppmega_mlx.runtime.path_c_relax_train_step import (
    build_train_step, plan_train_step,
    make_real_bank_forward_driver, register_loss_driver,
)
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    BANK_ACT, BANK_ACTG, BANK_PARAM, BANK_PARAMG, BANK_STATE, real_bank_numels,
    bank_arg_is_device,
)
from cppmega_mlx.runtime.path_c_relax_step_optim import register_optim_driver
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
SEQ = 4096
TOKENS_PER_STEP = SEQ * 1


def _avail_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("FAIL-LOUD: MemAvailable not found")


# --------------------------------------------------------------------------- #
# Per-region-kind wall-time accumulators (same scheme as the baseline profiler).
# --------------------------------------------------------------------------- #
@dataclass
class RegionTimers:
    fwd_s: float = 0.0
    fwd_n: int = 0
    bwd_s: float = 0.0
    bwd_n: int = 0
    adam_s: float = 0.0
    adam_n: int = 0
    loss_s: float = 0.0
    loss_n: int = 0

    def reset(self):
        for f in ("fwd_s", "fwd_n", "bwd_s", "bwd_n", "adam_s", "adam_n",
                  "loss_s", "loss_n"):
            setattr(self, f, 0)


TIMERS = RegionTimers()


def _wrap(kind: str, fn):
    def wrapped(*args):
        t0 = time.perf_counter()
        r = fn(*args)
        dt = time.perf_counter() - t0
        if kind == "fwd":
            TIMERS.fwd_s += dt
            TIMERS.fwd_n += 1
        elif kind == "bwd":
            TIMERS.bwd_s += dt
            TIMERS.bwd_n += 1
        elif kind == "adam":
            TIMERS.adam_s += dt
            TIMERS.adam_n += 1
        elif kind == "loss":
            TIMERS.loss_s += dt
            TIMERS.loss_n += 1
        return r
    return wrapped


def install_timed_bwd_adam_loss_factories():
    """Time the abstract bwd/adam/loss drivers via their FACTORY functions (same robust
    interception point as the baseline). The FORWARD is NOT factory-patched here -- it is
    the device-resident driver, which we wrap directly at registration (see register_timed_
    device_resident_forward). Numerics UNCHANGED."""
    import cppmega_mlx.runtime.path_c_relax_step_banks as B
    import cppmega_mlx.runtime.path_c_relax_step_optim as O
    import cppmega_mlx.runtime.path_c_relax_train_step as T

    if getattr(B, "_PR7_TIMED_BWD", False):
        return
    _orig_bwd = B._region_bwd_driver
    _orig_adam = O._adam_inplace_driver
    _orig_loss = T._loss_driver

    def timed_bwd(numels):
        return _wrap("bwd", _orig_bwd(numels))

    def timed_adam(numels):
        return _wrap("adam", _orig_adam(numels))

    def timed_loss():
        return _wrap("loss", _orig_loss())

    B._region_bwd_driver = timed_bwd
    O._adam_inplace_driver = timed_adam
    T._loss_driver = timed_loss
    B._PR7_TIMED_BWD = True


def register_timed_device_resident_forward(leaf, real_driver, bank_numels, n_layers, dev):
    """Build the DEVICE-RESIDENT forward packed func ONCE (banks stay device tensors,
    real kernel mutates them in place -- no .numpy() in the hot path), WRAP it with the
    fwd timer, and register it for every pathc.bank_fwd_i. Equivalent to
    register_real_forward_driver(device=dev) but timed so the profiler sees the
    device-resident forward cost."""
    fwd = make_real_bank_forward_driver(leaf, real_driver, bank_numels, device=dev)
    timed = _wrap("fwd", fwd)
    for i in range(n_layers):
        tvm_ffi.register_global_func(f"pathc.bank_fwd_{i}", timed, override=True)


# --------------------------------------------------------------------------- #
# Build the REAL MR path_c JITKernel + leaf (the same artifact §4a validated).
# --------------------------------------------------------------------------- #
def build_real_kernel():
    cfg = local_gb10_quarter_profile().hybrid_config()
    region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg)
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    target = _path_c_default_target()
    if target != "cuda":
        raise RuntimeError(f"FAIL-LOUD: expected cuda target on gb10, got {target}")
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


# --------------------------------------------------------------------------- #
@dataclass
class ProfileResult:
    n_layers: int
    n_steps: int
    compile_s: float
    step_times: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    planned_peak: int = 0
    measured_peak_gb: float = 0.0
    timers: RegionTimers = None

    @property
    def mean_step_s(self):
        warm = self.step_times[1:] if len(self.step_times) > 1 else self.step_times
        return float(np.mean(warm))

    @property
    def median_step_s(self):
        warm = self.step_times[1:] if len(self.step_times) > 1 else self.step_times
        return float(np.median(warm))

    @property
    def tok_s(self):
        return TOKENS_PER_STEP / self.mean_step_s


def profile_steps(numels, n_layers, n_steps, leaf, dev):
    if not dev.exist:
        raise RuntimeError("FAIL-LOUD: tvm.cuda(0) not present")

    # The bank ABI for the device-resident forward must carry the REAL kernel's physical
    # bank shapes (the validate script overrides BANK_ACT/PARAM/STATE/ACTG/PARAMG from the
    # leaf). We do the same so the forward driver's device banks match the kernel ABI.
    bank_numels = dict(numels)
    bs = leaf.bank_shapes
    for k in (BANK_ACT, BANK_PARAM, BANK_STATE, BANK_ACTG, BANK_PARAMG):
        bank_numels[k] = bs[k]

    # Time bwd/adam/loss via their factories BEFORE any registration.
    install_timed_bwd_adam_loss_factories()

    real_driver = make_real_kernel_driver(leaf, dev)

    plan = plan_train_step(bank_numels, n_layers)

    mod = build_train_step(bank_numels, n_layers)
    # build_train_step re-registers the abstract bank drivers (incl. abstract fwd); now
    # OVERRIDE pathc.bank_fwd_i with the TIMED DEVICE-RESIDENT forward.
    register_timed_device_resident_forward(leaf, real_driver, bank_numels, n_layers, dev)
    register_optim_driver(bank_numels)
    register_loss_driver()
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: train_step IRModule not well-formed")

    t0 = time.time()
    ex = tvm.compile(mod, target=tvm.target.Target("cuda"))
    vm = relax.VirtualMachine(ex, dev)
    compile_s = time.time() - t0
    print(f"  [compile-once] {compile_s:.2f}s (amortizes to 0 over a loop)", flush=True)

    rng = np.random.default_rng(0)
    def _dev(x):
        return tvm.runtime.tensor(np.ascontiguousarray(x, np.float32), device=dev)

    act0 = _dev((rng.random(bank_numels[BANK_ACT], np.float32) - 0.5) * 0.01)
    param = _dev((rng.random(bank_numels[BANK_PARAM], np.float32) - 0.5) * 0.01)
    paramg0 = _dev(np.zeros(bank_numels[BANK_PARAMG], np.float32))
    actg0 = _dev((rng.random(bank_numels[BANK_ACTG], np.float32) - 0.5) * 0.01)
    m = _dev((rng.random(bank_numels[BANK_PARAM], np.float32) * 0.1))
    v = _dev((rng.random(bank_numels[BANK_PARAM], np.float32) * 0.1))

    # Sanity: the device-resident path must actually be taken (banks detected as device).
    if not bank_arg_is_device(act0):
        raise RuntimeError("FAIL-LOUD: device act bank not detected as device tensor")

    res = ProfileResult(n_layers, n_steps, compile_s, planned_peak=plan.planned_peak)

    free_before = _avail_gb()
    min_free = [free_before]
    stop = [False]
    def sampler():
        while not stop[0]:
            fr = _avail_gb()
            if fr < min_free[0]:
                min_free[0] = fr
            time.sleep(0.02)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    TIMERS.reset()
    for s in range(n_steps):
        t0 = time.perf_counter()
        out = vm["train_step"](act0, param, paramg0, actg0, m, v)
        dev.sync()
        dt = time.perf_counter() - t0
        loss = float(np.asarray(out[0].numpy()).reshape(-1)[0])
        if not np.isfinite(loss):
            stop[0] = True
            raise RuntimeError(f"FAIL-LOUD: step {s} loss not finite ({loss})")
        param, m, v = out[1], out[2], out[3]
        res.step_times.append(dt)
        res.losses.append(loss)
        print(f"  step {s:>2}: {dt*1000:8.1f} ms  loss={loss:.6e}", flush=True)

    stop[0] = True
    th.join(timeout=2.0)
    res.measured_peak_gb = free_before - min_free[0]
    res.timers = TIMERS
    return res


def report(res: ProfileResult, label: str):
    print(f"\n=== PROFILE (DEVICE-RESIDENT FWD) {label} "
          f"(n_layers={res.n_layers}, {res.n_steps} steps) ===")
    print(f"  compile-once     : {res.compile_s:.2f}s (amortized to ~0 over a loop)")
    print(f"  mean step (warm) : {res.mean_step_s*1000:.1f} ms  "
          f"median {res.median_step_s*1000:.1f} ms")
    print(f"  tokens/step      : {TOKENS_PER_STEP} (seq={SEQ}, batch=1)")
    print(f"  THROUGHPUT       : {res.tok_s:.2f} tok/s")
    print(f"  planned dev-peak : {res.planned_peak/GB:.3f} GB")
    print(f"  MEASURED peak    : {res.measured_peak_gb:.2f} GB free-delta high-water")
    t = res.timers
    total = t.fwd_s + t.bwd_s + t.adam_s + t.loss_s
    pct = lambda x: (100 * x / total) if total > 0 else 0.0
    print(f"  per-region wall over {res.n_steps} steps (host-driver time):")
    print(f"    fwd+remat : {t.fwd_s*1000:9.1f} ms  ({t.fwd_n} calls)  "
          f"{pct(t.fwd_s):5.1f}%  [DEVICE-RESIDENT]")
    print(f"    bwd       : {t.bwd_s*1000:9.1f} ms  ({t.bwd_n} calls)  "
          f"{pct(t.bwd_s):5.1f}%  [abstract numpy]")
    print(f"    adam      : {t.adam_s*1000:9.1f} ms  ({t.adam_n} calls)  "
          f"{pct(t.adam_s):5.1f}%  [abstract numpy]")
    print(f"    loss      : {t.loss_s*1000:9.1f} ms  ({t.loss_n} calls)  "
          f"{pct(t.loss_s):5.1f}%  [abstract numpy]")
    print(f"    driver-tot: {total*1000:9.1f} ms  (rest = VM overhead)")
    # per-call means
    if t.fwd_n:
        print(f"  per-call fwd  : {t.fwd_s*1000/t.fwd_n:9.1f} ms/call (device-resident)")
    if t.bwd_n:
        print(f"  per-call bwd  : {t.bwd_s*1000/t.bwd_n:9.1f} ms/call (abstract)")
    return res


def main() -> int:
    print("PR 7 gb10 CUDA PROFILE -- DEVICE-RESIDENT forward, compile-once/run-many")
    print("TVM:", tvm.__version__, flush=True)
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("FAIL-LOUD: tvm.cuda(0) not present on gb10")

    numels = real_bank_numels()
    total_mb = sum(numels.values()) * 4 / 1024 / 1024
    print(f"real per-region banks: {len(numels)} banks, {total_mb:.1f} MB/region",
          flush=True)

    prim, kernel = build_real_kernel()
    leaf = make_leaf(prim, kernel)

    n_layers = int(os.environ.get("PR7_LAYERS", "8"))
    n_steps = int(os.environ.get("PR7_STEPS", "10"))

    res = profile_steps(numels, n_layers, n_steps, leaf, dev)
    report(res, f"gb10 CUDA, {n_layers} layers")

    fwd_per_step = res.timers.fwd_n // res.n_steps
    bwd_per_step = res.timers.bwd_n // res.n_steps
    print("\n=== REGION COUNTS (for extrapolation) ===")
    print(f"  fwd+remat calls/step = {fwd_per_step}, bwd calls/step = {bwd_per_step}")
    print(f"  per-step mean = {res.mean_step_s:.3f}s; "
          f"fwd-share={(res.timers.fwd_s/(res.timers.fwd_s+res.timers.bwd_s+res.timers.adam_s+res.timers.loss_s+1e-9))*100:.1f}%")
    print("  See doc §13 for the 28-layer extrapolation + Megatron gap.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
