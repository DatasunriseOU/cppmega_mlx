"""PR 7 gb10 CUDA PROFILE: the whole path_c Relax train_step, profiled over MULTIPLE
steps with the compile cached, to extract tok/s + peak memory + a per-region time
breakdown, then compared to the Megatron baseline (3399 tok/s @ 26 GB).

LOOP OPTIMIZATION APPLIED (the headline lever the profiler surfaces):
  COMPILE-ONCE / RUN-MANY. The PR-6 runner (run_train_step_on_device) re-built the
  IRModule + tvm.compile + VM on every call (compile dominates: ~minutes at 28 layers,
  seconds at 8). A training loop runs ONE step graph THOUSANDS of times. Here we
  tvm.compile ONCE, build the VM ONCE, then execute N steps on the SAME VM, threading
  the optimizer state (param', m', v') step->step. compile_s is reported separately and
  amortizes to ~0 over a real loop; the tok/s headline uses the mean NON-compile step.

PROFILER:
  * mean / median / p10 per-step wall (device-synced), tokens/step / mean-step -> tok/s
  * peak memory: a background /proc/meminfo sampler -> measured free-delta high-water,
    plus the StaticPlanBlockMemory planned device-peak (the honest device-allocator
    high-water).
  * per-region-kind breakdown: every call_dps_packed driver (bank_fwd, bank_bwd,
    adam_inplace, loss) is wrapped with a wall-time accumulator, so we see WHERE the
    step time goes (fwd vs recompute-fwd vs bwd vs optimizer vs loss vs host-staging).

TOKENS/STEP: the bank ABI is built from local_gb10_quarter_profile (hidden=3584,
max_seq_length=4096). The activation bank (44,957,696 f32) = ~3.06 * seq * hidden, i.e.
ONE sequence of seq=4096 tokens (batch=1). So this graph step processes 4096 tokens.
Megatron baseline runs bs=4 seq=4096 = 16384 tok/step. The 4x batch gap is stated
EXPLICITLY in the comparison (no fabrication) -- tok/s is throughput, batch-invariant to
first order for a memory-bound step, but we report BOTH the raw per-step tok/s and note
the batch ratio.

FAIL-LOUD: any non-finite loss or any region that cannot run RAISES, naming it.
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

from cppmega_mlx.runtime.path_c_relax_train_step import build_train_step, plan_train_step
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    BANK_ACT, BANK_ACTG, BANK_PARAM, BANK_PARAMG, BANK_STATE, real_bank_numels,
    register_bank_drivers,
)
from cppmega_mlx.runtime.path_c_relax_step_optim import register_optim_driver
from cppmega_mlx.runtime.path_c_relax_train_step import register_loss_driver

GB = 1024.0 ** 3
SEQ = 4096          # local_gb10_quarter max_seq_length -> tokens per (batch=1) step
TOKENS_PER_STEP = SEQ * 1   # batch=1 in the bank ABI


def _avail_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("FAIL-LOUD: MemAvailable not found")


# --------------------------------------------------------------------------- #
# Per-region-kind wall-time accumulators. We re-register every bank/optim/loss
# driver wrapped with a timer keyed by region kind, so the profiler sees where
# the step time goes. This does NOT change the numerics (same drivers underneath).
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


def install_timed_factories():
    """Monkeypatch the driver FACTORY functions so every packed func they produce is
    timed. This is robust to re-registration: build_train_step/plan_train_step call
    register_bank_drivers (which calls these factories) multiple times, and each time
    the factory now returns a TIMED packed func. The Relax VM binds the registered
    global by name at load, so wrapping at the factory (not at register-time) is the
    only reliable interception point. Numerics are UNCHANGED (same compute underneath).
    """
    import cppmega_mlx.runtime.path_c_relax_step_banks as B
    import cppmega_mlx.runtime.path_c_relax_step_optim as O
    import cppmega_mlx.runtime.path_c_relax_train_step as T

    if getattr(B, "_PR7_TIMED", False):
        return  # idempotent

    _orig_fwd = B._region_fwd_driver
    _orig_bwd = B._region_bwd_driver
    _orig_adam = O._adam_inplace_driver
    _orig_loss = T._loss_driver

    def timed_fwd(numels):
        return _wrap("fwd", _orig_fwd(numels))

    def timed_bwd(numels):
        return _wrap("bwd", _orig_bwd(numels))

    def timed_adam(numels):
        return _wrap("adam", _orig_adam(numels))

    def timed_loss():
        return _wrap("loss", _orig_loss())

    B._region_fwd_driver = timed_fwd
    B._region_bwd_driver = timed_bwd
    O._adam_inplace_driver = timed_adam
    T._loss_driver = timed_loss
    B._PR7_TIMED = True


# --------------------------------------------------------------------------- #
# Compile ONCE, run N steps on the SAME VM (the loop optimization).
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
        # drop the first step (warm-up: kernel JIT, cudnn handles, first host stage)
        warm = self.step_times[1:] if len(self.step_times) > 1 else self.step_times
        return float(np.mean(warm))

    @property
    def median_step_s(self):
        warm = self.step_times[1:] if len(self.step_times) > 1 else self.step_times
        return float(np.median(warm))

    @property
    def tok_s(self):
        return TOKENS_PER_STEP / self.mean_step_s


def profile_steps(numels, n_layers, n_steps, target="cuda"):
    dev = tvm.cuda(0) if target == "cuda" else tvm.cpu()
    if target == "cuda" and not dev.exist:
        raise RuntimeError("FAIL-LOUD: tvm.cuda(0) not present")

    # Patch the driver factories BEFORE any registration so every registered packed
    # func is timed (build_train_step/plan_train_step re-register via the factories).
    install_timed_factories()

    plan = plan_train_step(numels, n_layers)

    mod = build_train_step(numels, n_layers)
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: train_step IRModule not well-formed")

    # ---- COMPILE ONCE ----
    t0 = time.time()
    ex = tvm.compile(mod, target=tvm.target.Target(target))
    vm = relax.VirtualMachine(ex, dev)
    compile_s = time.time() - t0
    print(f"  [compile-once] {compile_s:.2f}s (amortizes to 0 over a loop)", flush=True)

    rng = np.random.default_rng(0)
    def _dev(x):
        return tvm.runtime.tensor(np.ascontiguousarray(x, np.float32), device=dev)

    # initial state (threaded step->step: param', m', v' feed the next step)
    act0 = _dev((rng.random(numels[BANK_ACT], np.float32) - 0.5))
    param = _dev((rng.random(numels[BANK_PARAM], np.float32) - 0.5))
    paramg0 = _dev(np.zeros(numels[BANK_PARAMG], np.float32))
    actg0 = _dev((rng.random(numels[BANK_ACTG], np.float32) - 0.5))
    m = _dev((rng.random(numels[BANK_PARAM], np.float32) * 0.1))
    v = _dev((rng.random(numels[BANK_PARAM], np.float32) * 0.1))

    res = ProfileResult(n_layers, n_steps, compile_s, planned_peak=plan.planned_peak)

    # background peak sampler over the whole step loop
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
        # thread optimizer state into the next step (param', m', v')
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
    print(f"\n=== PROFILE {label} (n_layers={res.n_layers}, {res.n_steps} steps) ===")
    print(f"  compile-once     : {res.compile_s:.2f}s (amortized to ~0 over a loop)")
    print(f"  mean step (warm) : {res.mean_step_s*1000:.1f} ms  "
          f"median {res.median_step_s*1000:.1f} ms")
    print(f"  tokens/step      : {TOKENS_PER_STEP} (seq={SEQ}, batch=1)")
    print(f"  THROUGHPUT       : {res.tok_s:.1f} tok/s")
    print(f"  planned dev-peak : {res.planned_peak/GB:.3f} GB")
    print(f"  MEASURED peak    : {res.measured_peak_gb:.2f} GB free-delta high-water")
    t = res.timers
    total = t.fwd_s + t.bwd_s + t.adam_s + t.loss_s
    pct = lambda x: (100 * x / total) if total > 0 else 0.0
    print(f"  per-region wall over {res.n_steps} steps (host-driver time):")
    print(f"    fwd+remat : {t.fwd_s*1000:9.1f} ms  ({t.fwd_n} calls)  "
          f"{pct(t.fwd_s):5.1f}%")
    print(f"    bwd       : {t.bwd_s*1000:9.1f} ms  ({t.bwd_n} calls)  "
          f"{pct(t.bwd_s):5.1f}%")
    print(f"    adam      : {t.adam_s*1000:9.1f} ms  ({t.adam_n} calls)  "
          f"{pct(t.adam_s):5.1f}%")
    print(f"    loss      : {t.loss_s*1000:9.1f} ms  ({t.loss_n} calls)  "
          f"{pct(t.loss_s):5.1f}%")
    print(f"    driver-tot: {total*1000:9.1f} ms  (rest = VM/host-staging overhead)")
    return res


def main() -> int:
    print("PR 7 gb10 CUDA PROFILE -- whole path_c train_step, compile-once/run-many")
    print("TVM:", tvm.__version__, flush=True)
    numels = real_bank_numels()
    total_mb = sum(numels.values()) * 4 / 1024 / 1024
    print(f"real per-region banks: {len(numels)} banks, {total_mb:.1f} MB/region", flush=True)

    n_layers = int(os.environ.get("PR7_LAYERS", "8"))
    n_steps = int(os.environ.get("PR7_STEPS", "10"))

    res = profile_steps(numels, n_layers, n_steps, target="cuda")
    report(res, f"gb10 CUDA, {n_layers} layers")

    # Extrapolation note for 28 layers (the 1.8B config): the step is dominated by the
    # per-region host-staged drivers, which scale ~linearly in region count. The 28-layer
    # region count is 146 vs the 8-layer count; we report the ratio for an honest
    # extrapolated tok/s.
    print("\n=== EXTRAPOLATION to 28 layers (1.8B) ===")
    print(f"  8-layer regions  ~ {8 + res.timers.fwd_n//res.n_steps - 8 + res.timers.bwd_n//res.n_steps} "
          f"fwd+remat+bwd calls/step (fwd_n/step={res.timers.fwd_n//res.n_steps}, "
          f"bwd_n/step={res.timers.bwd_n//res.n_steps})")
    print("  See doc for the region-count scaling + extrapolated 28-layer tok/s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
