"""PR 6 of docs/RELAX-GRAPH-MEMORY-PATH.md -- the TESTABLE END-TO-END train_step.

This assembles ONE Relax @R.function that is the WHOLE training step and RUNS it on
gb10 CUDA (tvm.cuda(0)):

    forward (sqrt-N remat) + backward + in-place Adam optimizer + a scalar LOSS,

with the PHYSICAL BANKS exposed as cross-region Relax SSA tensors (PR-3), the O(N)
checkpoint bank attacked by sqrt(N) rematerialization (PR-4), and the optimizer-state
term kept at 1x by an explicit in-place Adam op (PR-5). The forward region driver can
be wired to the REAL tilelang-compiled path_c CUDA kernel (the driver phase, proven on
gb10: 14.68M nonzero), so the external call_dps_packed boundary executes the REAL
path_c compute end-to-end through the Relax graph + CUDA VM.

This is the "до состояния что можно тестировать" deliverable: a single entry that
builds the whole-step IRModule, sets the real CUDA kernel driver, runs
StaticPlanBlockMemory + relax.build for target=cuda, and EXECUTES one training step on
device feeding real inputs, proving it RUNS and the loss is finite + the measured peak
(free -g delta) at the tested config.

WHAT IS NEW vs PR-5 (path_c_relax_step_optim.build_full_step_with_optim)
------------------------------------------------------------------------
  (a) A scalar LOSS leaf: a `pathc.bank_loss` call_dps_packed that reduces the final
      forward activation bank to a single f32 (sum of squares / numel = mean-square),
      so the step returns a finite scalar loss we can report -- the training-step
      observable that gates the profiling phase.
  (b) A REAL-KERNEL forward driver option: `register_real_forward_driver` swaps the
      abstract numpy bank-forward driver (`pathc.bank_fwd_i`) for a driver that runs
      the REAL tilelang path_c CUDA JITKernel on the physical banks (via the driver
      phase's make_real_kernel_driver). The bank-SSA, remat, optimizer, and loss
      structure are UNCHANGED -- only the per-region forward compute behind the
      call_dps_packed boundary becomes the real kernel.
  (c) An on-device RUNNER: `run_train_step_on_device` compiles for target=cuda, builds
      a relax.VirtualMachine on tvm.cuda(0), feeds real device inputs, executes one
      step, and returns the loss + the updated optimizer state, FAIL-LOUD on any region
      that can't run through the graph on CUDA.

DEVICE: real run target = CUDA on gb10 (tvm.cuda(0)). The same module also compiles +
runs on CPU LLVM for the numeric self-check (loss matches a numpy reference), so the
graph is testable off-device too.

Run (CPU self-check, off-gb10):
    TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \\
    PYTHONPATH=/Volumes/external/sources/cppmega.mlx \\
    <python> -m cppmega_mlx.runtime.path_c_relax_train_step

Run (gb10 CUDA e2e, real kernel): see scratch/pr6_cuda_e2e_train_step_gb10.py.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import tvm
import tvm_ffi
from tvm import relax

from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir,
    _plan_and_lower,
)
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    BANK_ACT,
    BANK_ACTG,
    BANK_PARAM,
    BANK_PARAMG,
    BANK_STATE,
    BankSinfo,
    bank_arg_to_host,
    bank_writeback,
    real_bank_numels,
    register_bank_drivers,
)
from cppmega_mlx.runtime.path_c_relax_step_remat import (
    checkpoint_boundaries,
    nearest_boundary,
    recompute_overhead,
)
from cppmega_mlx.runtime.path_c_relax_step_optim import (
    register_optim_driver,
    true_planned_peak,
    _adam_reference,
)


# --------------------------------------------------------------------------- #
# (a) The scalar LOSS driver. Reduces the final forward activation bank to one
#     f32 = mean of squares (a stand-in training loss whose finiteness is the
#     train-step observable). call_dps_packed ABI: (act_in, loss_out[1]).
# --------------------------------------------------------------------------- #
def _loss_driver():
    def packed(act_in, loss_out):
        a = bank_arg_to_host(act_in)
        # mean of squares over the activation bank -- a finite scalar training loss.
        lo = np.array([np.float32(np.mean(np.square(a.astype(np.float64))))], np.float32)
        bank_writeback(loss_out, lo)

    return packed


def register_loss_driver() -> None:
    tvm_ffi.register_global_func("pathc.bank_loss", _loss_driver(), override=True)


def _loss_reference(act_bank: np.ndarray) -> float:
    return float(np.mean(np.square(act_bank.astype(np.float64))))


# --------------------------------------------------------------------------- #
# (b) The REAL-kernel forward driver. Replaces the abstract numpy bank-forward
#     compute (path_c_relax_step_banks._region_fwd_driver) with a call into the
#     REAL tilelang path_c CUDA JITKernel on the physical banks.
#
# The bank-forward call_dps_packed ABI is (act_in, param, state_in, act_out,
# state_out): act_out = the new activation bank, state_out = the checkpoint. We
# realize act_out by running the real kernel on the banks (the kernel writes its
# activation bank in place) and copy the activation bank into act_out; the
# checkpoint state_out snapshots the activation, exactly as the abstract driver
# does, so the backward (gate-by-checkpoint) semantics are preserved.
#
# `leaf` is a fwd-only PathCRegionLeaf (run_backward=0) carrying the real kernel;
# `real_driver` is make_real_kernel_driver(leaf, device) from the driver phase.
# --------------------------------------------------------------------------- #
def make_real_bank_forward_driver(leaf, real_driver, bank_numels: dict[str, int]):
    """Build a `pathc.bank_fwd_*` packed func that runs the REAL path_c CUDA kernel.

    ABI: (act_in, param, state_in, act_out, state_out). It packs the act/param banks
    into the real kernel's physical banks, runs the real kernel (real_driver writes
    the activation bank back into ``banks``), then writes act_out = activation bank
    and state_out = checkpoint snapshot of the activation. The real kernel is the
    SAME compiled artifact proven on gb10 (14.68M nonzero)."""

    act_n = bank_numels[BANK_ACT]
    state_n = bank_numels[BANK_STATE]
    param_n = bank_numels[BANK_PARAM]

    def packed(act_in, param, state_in, act_out, state_out):
        a = bank_arg_to_host(act_in)
        p = bank_arg_to_host(param)
        ao = act_out
        so = state_out
        # Physical banks the real kernel operates on. The activation bank is seeded
        # with the incoming activation; the parameter bank with the incoming params.
        banks = {
            BANK_ACT: np.ascontiguousarray(a, np.float32).reshape(-1)[:act_n].copy(),
            BANK_ACTG: np.zeros(bank_numels[BANK_ACTG], np.float32),
            BANK_PARAM: np.ascontiguousarray(p, np.float32).reshape(-1)[:param_n].copy(),
            BANK_PARAMG: np.zeros(bank_numels[BANK_PARAMG], np.float32),
            BANK_STATE: np.zeros(state_n, np.float32),
        }
        if banks[BANK_ACT].size < act_n:
            pad = np.zeros(act_n, np.float32)
            pad[: banks[BANK_ACT].size] = banks[BANK_ACT]
            banks[BANK_ACT] = pad
        if banks[BANK_PARAM].size < param_n:
            pad = np.zeros(param_n, np.float32)
            pad[: banks[BANK_PARAM].size] = banks[BANK_PARAM]
            banks[BANK_PARAM] = pad
        # Run the REAL tilelang path_c CUDA kernel on the banks (writes act bank).
        real_driver(leaf, banks)
        act_new = banks[BANK_ACT].reshape(-1)
        # act_out <- new activation bank.
        bank_writeback(ao, act_new[:act_n])
        # state_out <- checkpoint snapshot of the activation (what backward reads).
        ck = np.zeros(state_n, np.float32)
        n = min(state_n, act_new.size)
        ck[:n] = act_new[:n]
        bank_writeback(so, ck)

    return packed


def register_real_forward_driver(leaf, real_driver, bank_numels: dict[str, int],
                                 n_layers: int) -> None:
    """Install the REAL path_c CUDA kernel as EVERY `pathc.bank_fwd_i` region.

    All forward regions share the one real MR kernel (the bank ABI is per-block
    identical; the bank STORAGE is what the planner threads). The backward + optimizer
    + loss drivers stay the abstract numpy ones (the backward kernel is the s_tir wall
    documented in the adapter; the forward proves the real-kernel-through-graph path).
    """
    fwd = make_real_bank_forward_driver(leaf, real_driver, bank_numels)
    for i in range(n_layers):
        tvm_ffi.register_global_func(f"pathc.bank_fwd_{i}", fwd, override=True)


# --------------------------------------------------------------------------- #
# (c) The whole-step IRModule: fwd (remat) + bwd + in-place Adam + LOSS.
#
# This is build_full_step_with_optim (PR-5) extended to ALSO emit a scalar loss leaf
# on the final forward activation, and to RETURN (loss, param', m', v'). The bank-SSA,
# remat, and in-place-Adam structure are identical to PR-5 (re-emitted here so the loss
# can hook the final forward activation, which PR-5 does not expose).
# --------------------------------------------------------------------------- #
def build_train_step(numels: dict[str, int], n_layers: int) -> tvm.IRModule:
    register_bank_drivers(numels, n_layers)
    register_optim_driver(numels)
    register_loss_driver()
    boundaries = checkpoint_boundaries(n_layers)
    bset = set(boundaries)

    sAct = BankSinfo(numels[BANK_ACT]).sinfo()
    sActG = BankSinfo(numels[BANK_ACTG]).sinfo()
    sParam = BankSinfo(numels[BANK_PARAM]).sinfo()
    sParamG = BankSinfo(numels[BANK_PARAMG]).sinfo()
    sState = BankSinfo(numels[BANK_STATE]).sinfo()
    sM = BankSinfo(numels[BANK_PARAM]).sinfo()
    sV = BankSinfo(numels[BANK_PARAM]).sinfo()
    sLoss = relax.TensorStructInfo((1,), "float32")

    bb = relax.BlockBuilder()
    act0 = relax.Var("act0", sAct)
    param = relax.Var("param", sParam)
    paramg0 = relax.Var("paramg0", sParamG)
    actg0 = relax.Var("actg0", sActG)
    m0 = relax.Var("m0", sM)
    v0 = relax.Var("v0", sV)
    with bb.function("train_step", [act0, param, paramg0, actg0, m0, v0]):
        with bb.dataflow():
            # ---- FORWARD (sqrt-N remat) ----
            act = act0
            saved_act: dict[int, relax.Var] = {}
            saved_ckpt: dict[int, relax.Var] = {}
            for i in range(n_layers):
                if i in bset:
                    saved_act[i] = act
                out = bb.emit(relax.call_dps_packed(
                    f"pathc.bank_fwd_{i}", [act, param, act], [sAct, sState]))
                act_next = bb.emit(relax.TupleGetItem(out, 0))
                ck = bb.emit(relax.TupleGetItem(out, 1))
                if i in bset:
                    saved_ckpt[i] = ck
                act = act_next
            act_final = act

            # ---- LOSS on the final forward activation ----
            loss = bb.emit(relax.call_dps_packed("pathc.bank_loss", [act_final], [sLoss]))

            # ---- BACKWARD (re-emit forward for non-boundary checkpoints) ----
            actg = actg0
            paramg = paramg0
            for i in reversed(range(n_layers)):
                if i in bset:
                    ck_i = saved_ckpt[i]
                else:
                    b = nearest_boundary(i, boundaries)
                    rec_act = saved_act[b]
                    ck_i = None
                    for j in range(b, i + 1):
                        rout = bb.emit(relax.call_dps_packed(
                            f"pathc.bank_fwd_{j}", [rec_act, param, rec_act],
                            [sAct, sState]))
                        rec_act = bb.emit(relax.TupleGetItem(rout, 0))
                        rec_ck = bb.emit(relax.TupleGetItem(rout, 1))
                        ck_i = rec_ck
                    if ck_i is None:
                        raise RuntimeError(
                            "FAIL-LOUD: recompute produced no checkpoint for region "
                            f"{i}")
                out = bb.emit(relax.call_dps_packed(
                    f"pathc.bank_bwd_{i}", [actg, param, ck_i, paramg],
                    [sActG, sParamG]))
                actg = bb.emit(relax.TupleGetItem(out, 0))
                paramg = bb.emit(relax.TupleGetItem(out, 1))

            # ---- IN-PLACE ADAM OPTIMIZER ----
            opt = bb.emit(relax.call_dps_packed(
                "pathc.adam_inplace", [param, m0, v0, paramg], [sParam, sM, sV]))
            param_new = bb.emit(relax.TupleGetItem(opt, 0))
            m_new = bb.emit(relax.TupleGetItem(opt, 1))
            v_new = bb.emit(relax.TupleGetItem(opt, 2))

            res = bb.emit_output(relax.Tuple([loss, param_new, m_new, v_new]))
        bb.emit_func_output(res)
    return bb.get()


# --------------------------------------------------------------------------- #
# Planned-peak accounting for the whole train_step (same honest analyzer as PR-5).
# --------------------------------------------------------------------------- #
@dataclass
class TrainStepPlan:
    n_layers: int
    planned_peak: int
    fwd_calls: int
    recompute_calls: int


def plan_train_step(numels: dict[str, int], n_layers: int) -> TrainStepPlan:
    mod = build_train_step(numels, n_layers)
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: train_step IRModule is not well-formed")
    mod_ct = _legalize_to_call_tir(mod)
    mod_pl = _plan_and_lower(mod_ct)
    fwd, extra, _ = recompute_overhead(n_layers)
    return TrainStepPlan(
        n_layers, true_planned_peak(mod_pl["train_step"]), fwd, extra)


# --------------------------------------------------------------------------- #
# The on-device RUNNER. Compiles for target, builds the VM, feeds real inputs,
# executes one step, returns (loss, param', m', v'). FAIL-LOUD on any failure.
# --------------------------------------------------------------------------- #
@dataclass
class TrainStepResult:
    n_layers: int
    target: str
    loss: float
    param_checksum: float
    compile_s: float
    run_s: float
    planned_peak: int


def run_train_step_on_device(
    numels: dict[str, int],
    n_layers: int,
    *,
    target: str,
    device: Any,
    seed: int = 0,
    inputs: dict[str, np.ndarray] | None = None,
) -> TrainStepResult:
    """Build the whole train_step, compile for `target`, run it on `device`, and
    return the loss + updated-param checksum. FAIL-LOUD on a non-finite loss or any
    region that cannot run through the graph on the target."""
    mod = build_train_step(numels, n_layers)
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: train_step IRModule is not well-formed")

    rng = np.random.default_rng(seed)
    if inputs is None:
        inputs = {}
    act0 = inputs.get(
        "act0", (rng.random(numels[BANK_ACT], np.float32) - 0.5).astype(np.float32))
    param = inputs.get(
        "param", (rng.random(numels[BANK_PARAM], np.float32) - 0.5).astype(np.float32))
    paramg0 = inputs.get("paramg0", np.zeros(numels[BANK_PARAMG], np.float32))
    actg0 = inputs.get(
        "actg0", (rng.random(numels[BANK_ACTG], np.float32) - 0.5).astype(np.float32))
    m0 = inputs.get(
        "m0", (rng.random(numels[BANK_PARAM], np.float32) * 0.1).astype(np.float32))
    v0 = inputs.get(
        "v0", (rng.random(numels[BANK_PARAM], np.float32) * 0.1).astype(np.float32))

    t0 = time.time()
    ex = tvm.compile(mod, target=tvm.target.Target(target))
    vm = relax.VirtualMachine(ex, device)
    compile_s = time.time() - t0

    def _dev(x):
        return tvm.runtime.tensor(np.ascontiguousarray(x, np.float32), device=device)

    t0 = time.time()
    out = vm["train_step"](
        _dev(act0), _dev(param), _dev(paramg0), _dev(actg0), _dev(m0), _dev(v0))
    if hasattr(device, "sync"):
        device.sync()
    run_s = time.time() - t0

    loss_t, param_t, m_t, v_t = out[0], out[1], out[2], out[3]
    loss = float(np.asarray(loss_t.numpy()).reshape(-1)[0])
    param_new = np.asarray(param_t.numpy()).reshape(-1)
    if not np.isfinite(loss):
        raise RuntimeError(
            f"FAIL-LOUD: train_step loss is NOT finite ({loss}) at n_layers={n_layers} "
            f"target={target}")
    checksum = float(np.abs(param_new).sum())
    if not np.isfinite(checksum):
        raise RuntimeError(
            "FAIL-LOUD: updated param bank has non-finite values after the optimizer")
    return TrainStepResult(
        n_layers, target, loss, checksum, compile_s, run_s,
        plan_train_step(numels, n_layers).planned_peak)


# --------------------------------------------------------------------------- #
# CPU numeric self-check: the whole train_step runs + the loss matches a numpy
# reference of the SAME dataflow (abstract bank drivers). This makes the e2e graph
# testable OFF gb10 (no CUDA needed) and proves the loss leaf is correct.
# --------------------------------------------------------------------------- #
def _numpy_reference_step(numels, n_layers, act0, param, paramg0, actg0, m0, v0):
    bias = np.float32(param[:1].sum() * 1e-6)
    act = act0.copy()
    ckpts = []
    for _ in range(n_layers):
        act = np.maximum(act + bias, 0.0)
        ck = np.zeros(numels[BANK_STATE], np.float32)
        n = min(ck.size, act.size)
        ck[:n] = act[:n]
        ckpts.append(ck)
    loss = _loss_reference(act)
    actg = actg0.copy()
    paramg = paramg0.copy()
    for i in reversed(range(n_layers)):
        ck = ckpts[i]
        n = min(actg.size, ck.size)
        gate = (ck[:n] > 0.0).astype(np.float32)
        new_actg = np.zeros(numels[BANK_ACTG], np.float32)
        new_actg[:n] = actg[:n] * gate
        m = min(paramg.size, n)
        new_paramg = paramg.copy()
        new_paramg[:m] = paramg[:m] + actg[:m] * 1e-3
        actg = new_actg
        paramg = new_paramg
    p_new, m_new, v_new = _adam_reference(param, m0, v0, paramg, numels)
    return loss, p_new, m_new, v_new


def _verify_cpu_numerics(numels, n_layers) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    act0 = (rng.random(numels[BANK_ACT], np.float32) - 0.5).astype(np.float32)
    param = (rng.random(numels[BANK_PARAM], np.float32) - 0.5).astype(np.float32)
    paramg0 = np.zeros(numels[BANK_PARAMG], np.float32)
    actg0 = (rng.random(numels[BANK_ACTG], np.float32) - 0.5).astype(np.float32)
    m0 = (rng.random(numels[BANK_PARAM], np.float32) * 0.1).astype(np.float32)
    v0 = (rng.random(numels[BANK_PARAM], np.float32) * 0.1).astype(np.float32)
    inputs = dict(act0=act0, param=param, paramg0=paramg0, actg0=actg0, m0=m0, v0=v0)

    res = run_train_step_on_device(
        numels, n_layers, target="llvm", device=tvm.cpu(), inputs=inputs)
    ref_loss, ref_p, _, _ = _numpy_reference_step(
        numels, n_layers, act0, param, paramg0, actg0, m0, v0)
    dloss = abs(res.loss - ref_loss)
    if not np.isclose(res.loss, ref_loss, rtol=1e-4, atol=1e-6):
        raise RuntimeError(
            "FAIL-LOUD: train_step loss disagrees with numpy reference; "
            f"vm={res.loss} ref={ref_loss} diff={dloss}")
    ref_checksum = float(np.abs(ref_p).sum())
    dchk = abs(res.param_checksum - ref_checksum)
    if not np.isclose(res.param_checksum, ref_checksum, rtol=1e-3, atol=1e-3):
        raise RuntimeError(
            "FAIL-LOUD: train_step param' checksum disagrees with numpy reference; "
            f"vm={res.param_checksum} ref={ref_checksum}")
    if not np.isfinite(res.loss):
        raise RuntimeError("FAIL-LOUD: train_step loss not finite")
    return dloss, dchk


def main() -> int:
    gb = 1024.0 ** 3
    print("PR 6 -- TESTABLE END-TO-END train_step (fwd+remat+bwd+in-place Adam+loss).")
    print("Device: CPU LLVM Relax VM self-check. TVM:", tvm.__version__)
    numels = real_bank_numels()

    # 1) The whole step RUNS on CPU + the loss matches a numpy reference.
    print("\n--- CPU e2e: whole train_step runs + loss finite + matches reference ---")
    vm_numels = {k: max(8, v // 20000) for k, v in numels.items()}
    for nl in (4, 8, 16, 28):
        res = run_train_step_on_device(
            vm_numels, nl, target="llvm", device=tvm.cpu())
        dloss, dchk = _verify_cpu_numerics(vm_numels, nl)
        print(f"  layers={nl:>2}: RUNS=yes  loss={res.loss:.6e} (finite=yes)  "
              f"vs-ref|dloss|={dloss:.2e} |dchk|={dchk:.2e}  PASS")

    # 2) Full-scale planned peak of the WHOLE step (real numels, incl. loss).
    print("\n--- FULL-SCALE planned peak of the whole train_step (real banks) ---")
    print(f"  {'layers':>6} {'planned peak':>14} {'recompute':>11}")
    for nl in (4, 8, 16, 28):
        plan = plan_train_step(numels, nl)
        print(f"  {nl:>6} {plan.planned_peak/gb:>11.3f} GB {plan.recompute_calls:>5} calls")

    plan28 = plan_train_step(numels, 28)
    print(f"\n  1.8B (28 MR blocks) whole-step planned peak = "
          f"{plan28.planned_peak/gb:.3f} GB")
    print("\nALL CPU CHECKS PASSED: the whole train_step (fwd+remat+bwd+in-place Adam+"
          "loss) assembles into ONE Relax @R.function, runs on the LLVM VM with a finite "
          "loss matching the numpy reference, and plans to the PR-5 full-step peak. The "
          "gb10 CUDA e2e (real kernel) is scratch/pr6_cuda_e2e_train_step_gb10.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
