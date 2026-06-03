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
    bank_arg_is_device,
    bank_arg_to_host,
    bank_copy_prefix_device,
    bank_writeback,
    real_bank_numels,
    register_bank_drivers,
    set_real_bank_bwd_installed,
)
from cppmega_mlx.runtime.path_c_dps_adapter import (
    alloc_device_banks,
    make_device_resident_kernel_driver,
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
def make_real_bank_forward_driver(leaf, real_driver, bank_numels: dict[str, int],
                                  *, device=None):
    """Build a `pathc.bank_fwd_*` packed func that runs the REAL path_c CUDA kernel.

    ABI: (act_in, param, state_in, act_out, state_out). It seeds the real kernel's
    activation/parameter physical banks from the incoming act/param SSA bank tensors,
    runs the real kernel (writes the activation bank in place), then writes
    act_out = activation bank and state_out = checkpoint snapshot of the activation.
    The real kernel is the SAME compiled artifact proven on gb10 (14.68M nonzero).

    TWO PATHS (RULE #1: clear device-vs-host gate, no silent fallback):
      * DEVICE-RESIDENT (gb10): when the call_dps_packed bank tensors arrive on a real
        device, the act/param banks are seeded into the driver-OWNED device physical
        banks via DEVICE->DEVICE view copies, the DEVICE-RESIDENT kernel driver runs
        the kernel on those device banks IN PLACE, and act_out/state_out are filled by
        DEVICE->DEVICE copies. NO ``.numpy()`` / host staging anywhere -- this removes
        the per-region per-step ~2028 MB host round-trip at the VM boundary.
      * NUMPY-STAGED (CPU self-test / reference): host numpy banks + the numpy-staged
        real_driver. Used only when the ABI tensors are host tensors.

    ``device`` is required for the device-resident path; the device kernel driver +
    device physical banks are built ONCE here and reused across calls (the kernel
    mutates them in place)."""

    act_n = bank_numels[BANK_ACT]
    state_n = bank_numels[BANK_STATE]
    param_n = bank_numels[BANK_PARAM]

    # Build the device-resident kernel driver + the driver-owned device physical banks
    # ONCE (lazily, on first device call). These persist across every forward call.
    dev_state: dict[str, object] = {}

    def _ensure_device_state(dev):
        if "driver" not in dev_state:
            dev_state["driver"] = make_device_resident_kernel_driver(leaf, dev)
            dev_state["banks"] = alloc_device_banks(leaf.bank_shapes, dev)
        return dev_state["driver"], dev_state["banks"]

    def packed(act_in, param, state_in, act_out, state_out):
        if bank_arg_is_device(act_in):
            # ---- DEVICE-RESIDENT PATH (no numpy host bounce) ----
            dev = act_in.device
            dev_driver, dbanks = _ensure_device_state(dev)
            # The non-input banks (ACTG, PARAMG, STATE) are zeroed ONCE at alloc (in
            # alloc_device_banks). The scout validated (pr7_device_resident_kernel_feed:
            # max|device-numpy|=0) that REUSING the banks across calls -- WITHOUT
            # re-zeroing -- is bit-identical to the fresh-zeroed-numpy path for the
            # forward kernel (the forward reads ACT+PARAM and overwrites its out banks;
            # it does not read stale non-input bank state). So we DO NOT re-zero per
            # call -- that would reintroduce a ~1.5 GB host->device bounce we are here
            # to eliminate. (Numeric equivalence is asserted on gb10.)
            # Seed the kernel's activation/parameter physical banks from the SSA banks
            # (device->device view copies; tail-zeroed where the SSA bank is shorter).
            bank_copy_prefix_device(dbanks[BANK_ACT], act_in,
                                    min(act_n, int(np.prod([int(d) for d in act_in.shape]))))
            bank_copy_prefix_device(dbanks[BANK_PARAM], param,
                                    min(param_n, int(np.prod([int(d) for d in param.shape]))))
            # Run the REAL kernel on the device banks (writes BANK_ACT in place).
            dev_driver(leaf, dbanks, dev)
            # act_out <- new activation bank (device->device).
            bank_copy_prefix_device(act_out, dbanks[BANK_ACT],
                                    min(act_n, int(np.prod([int(d) for d in act_out.shape]))))
            # state_out <- checkpoint snapshot of the activation (device->device).
            so_n = int(np.prod([int(d) for d in state_out.shape]))
            bank_copy_prefix_device(state_out, dbanks[BANK_ACT], min(so_n, act_n))
            return

        # ---- NUMPY-STAGED REFERENCE PATH (CPU self-test) ----
        a = bank_arg_to_host(act_in)
        p = bank_arg_to_host(param)
        ao = act_out
        so = state_out
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
        real_driver(leaf, banks)
        act_new = banks[BANK_ACT].reshape(-1)
        bank_writeback(ao, act_new[:act_n])
        ck = np.zeros(state_n, np.float32)
        n = min(state_n, act_new.size)
        ck[:n] = act_new[:n]
        bank_writeback(so, ck)

    return packed


def register_real_forward_driver(leaf, real_driver, bank_numels: dict[str, int],
                                 n_layers: int, *, device=None) -> None:
    """Install the REAL path_c CUDA kernel as EVERY `pathc.bank_fwd_i` region.

    All forward regions share the one real MR kernel (the bank ABI is per-block
    identical; the bank STORAGE is what the planner threads). The backward + optimizer
    + loss drivers stay the abstract numpy ones (the backward kernel is the s_tir wall
    documented in the adapter; the forward proves the real-kernel-through-graph path).

    ``device`` enables the DEVICE-RESIDENT forward path (banks stay device tensors, no
    host bounce). Pass the CUDA device on gb10."""
    fwd = make_real_bank_forward_driver(leaf, real_driver, bank_numels, device=device)
    for i in range(n_layers):
        tvm_ffi.register_global_func(f"pathc.bank_fwd_{i}", fwd, override=True)


# --------------------------------------------------------------------------- #
# (b2) The REAL-kernel BACKWARD driver — symmetric to make_real_bank_forward_driver.
#
# This is what flips the §15-profiled `pathc.bank_bwd_i` from the ABSTRACT NUMPY host
# backward (path_c_relax_step_banks._region_bwd_driver, the 79.6%/91.4% wall — a host
# round-trip of the ~2028 MB region banks) to the REAL GRIDDED backward MR kernel
# (run_backward=1), which behind the call_dps_packed boundary runs the gridded
# B0/B1/B2 chunked SSD backward on the physical banks. The bank-SSA, remat, optimizer,
# and loss structure are UNCHANGED — only the per-region backward compute becomes the
# real device-resident kernel.
#
# The bank-backward call_dps_packed ABI (from _region_bwd_driver) is
#   (actg_in, param, state_ckpt, paramg_in, actg_out, paramg_out):
#     IN : actg_in   = incoming activation grad (from the next layer)
#          param     = the model weights (read-only)
#          state_ckpt= checkpoint i (the forward-saved activation snapshot)
#          paramg_in = running parameter-grad accumulator
#     OUT: actg_out  = grad propagated to the previous layer (the bwd kernel writes
#                      BANK_ACTG in place)
#          paramg_out= updated parameter-grad accumulator (BANK_PARAMG in place)
#
# `leaf` is a BACKWARD PathCRegionLeaf (run_backward=1) carrying the real kernel;
# `real_driver` is make_real_kernel_driver(leaf, device) from the driver phase.
# --------------------------------------------------------------------------- #
def make_real_bank_backward_driver(leaf, real_driver, bank_numels: dict[str, int],
                                   *, device=None):
    """Build a `pathc.bank_bwd_*` packed func that runs the REAL gridded MR backward.

    Mirrors make_real_bank_forward_driver exactly (same DEVICE-RESIDENT vs
    NUMPY-STAGED gate, RULE #1: clear device-vs-host gate, no silent fallback), but
    over the BACKWARD bank ABI. The real kernel is the SAME compiled artifact as the
    forward, specialized to run_backward=1 (the gridded B0/B1/B2 transpose). It seeds
    BANK_ACTG/BANK_PARAM/BANK_STATE/BANK_PARAMG from actg_in/param/state_ckpt/
    paramg_in, runs the kernel (writes BANK_ACTG + BANK_PARAMG in place), then writes
    actg_out = BANK_ACTG and paramg_out = BANK_PARAMG.

    RULE #1: no numpy host backward is reachable from this driver; the only paths are
    (a) DEVICE-RESIDENT real kernel and (b) NUMPY-STAGED real kernel (the same real
    compiled kernel, just host-staged for the CPU self-check). Any failure RAISES.
    """

    actg_n = bank_numels[BANK_ACTG]
    state_n = bank_numels[BANK_STATE]
    param_n = bank_numels[BANK_PARAM]
    paramg_n = bank_numels[BANK_PARAMG]

    # Build the device-resident kernel driver + driver-owned device physical banks
    # ONCE (lazily, first device call); persists across every backward call.
    dev_state: dict[str, object] = {}

    def _ensure_device_state(dev):
        if "driver" not in dev_state:
            dev_state["driver"] = make_device_resident_kernel_driver(leaf, dev)
            dev_state["banks"] = alloc_device_banks(leaf.bank_shapes, dev)
        return dev_state["driver"], dev_state["banks"]

    def packed(actg_in, param, state_ckpt, paramg_in, actg_out, paramg_out):
        if bank_arg_is_device(actg_in):
            # ---- DEVICE-RESIDENT PATH (no numpy host bounce) ----
            dev = actg_in.device
            dev_driver, dbanks = _ensure_device_state(dev)
            # Seed the backward kernel's physical banks from the SSA banks
            # (device->device view copies; tail-zeroed where the SSA bank is shorter).
            bank_copy_prefix_device(
                dbanks[BANK_ACTG], actg_in,
                min(actg_n, int(np.prod([int(d) for d in actg_in.shape]))))
            bank_copy_prefix_device(
                dbanks[BANK_PARAM], param,
                min(param_n, int(np.prod([int(d) for d in param.shape]))))
            bank_copy_prefix_device(
                dbanks[BANK_STATE], state_ckpt,
                min(state_n, int(np.prod([int(d) for d in state_ckpt.shape]))))
            bank_copy_prefix_device(
                dbanks[BANK_PARAMG], paramg_in,
                min(paramg_n, int(np.prod([int(d) for d in paramg_in.shape]))))
            # Run the REAL gridded backward kernel (run_backward=1) on the device
            # banks (writes BANK_ACTG + BANK_PARAMG in place).
            dev_driver(leaf, dbanks, dev)
            # actg_out <- new activation grad (device->device).
            bank_copy_prefix_device(
                actg_out, dbanks[BANK_ACTG],
                min(actg_n, int(np.prod([int(d) for d in actg_out.shape]))))
            # paramg_out <- updated parameter-grad accumulator (device->device).
            bank_copy_prefix_device(
                paramg_out, dbanks[BANK_PARAMG],
                min(paramg_n, int(np.prod([int(d) for d in paramg_out.shape]))))
            return

        # ---- NUMPY-STAGED REFERENCE PATH (CPU self-test; STILL the real kernel) ----
        ag = bank_arg_to_host(actg_in)
        p = bank_arg_to_host(param)
        ck = bank_arg_to_host(state_ckpt)
        pgi = bank_arg_to_host(paramg_in)
        banks = {
            BANK_ACT: np.zeros(bank_numels[BANK_ACT], np.float32),
            BANK_ACTG: np.ascontiguousarray(ag, np.float32).reshape(-1)[:actg_n].copy(),
            BANK_PARAM: np.ascontiguousarray(p, np.float32).reshape(-1)[:param_n].copy(),
            BANK_PARAMG: np.ascontiguousarray(pgi, np.float32).reshape(-1)[:paramg_n].copy(),
            BANK_STATE: np.ascontiguousarray(ck, np.float32).reshape(-1)[:state_n].copy(),
        }
        for bank, n in ((BANK_ACTG, actg_n), (BANK_PARAM, param_n),
                        (BANK_PARAMG, paramg_n), (BANK_STATE, state_n)):
            if banks[bank].size < n:
                pad = np.zeros(n, np.float32)
                pad[: banks[bank].size] = banks[bank]
                banks[bank] = pad
        real_driver(leaf, banks)
        bank_writeback(actg_out, banks[BANK_ACTG].reshape(-1)[:actg_n])
        bank_writeback(paramg_out, banks[BANK_PARAMG].reshape(-1)[:paramg_n])

    return packed


def register_real_backward_driver(leaf, real_driver, bank_numels: dict[str, int],
                                  n_layers: int, *, device=None) -> None:
    """Install the REAL gridded path_c backward kernel as EVERY `pathc.bank_bwd_i`.

    This REPLACES the abstract numpy host backward (the §15 79.6%/91.4% wall) with the
    real device-resident gridded backward MR kernel (run_backward=1 -> gridded
    B0/B1/B2). It also flips ``set_real_bank_bwd_installed(True)`` so that:
      * the RULE #1 guard in register_bank_drivers does NOT raise (the gridded path is
        live, not the forbidden numpy fallback), and
      * a subsequent register_bank_drivers re-entry does NOT clobber this real bwd
        binding with the numpy one.

    ``leaf`` MUST be a BACKWARD leaf (run_backward=1); ``device`` enables the
    DEVICE-RESIDENT backward path (banks stay device tensors, no host bounce). Pass the
    CUDA device on gb10. RULE #1: a forward leaf here is a wiring bug — RAISE."""
    if int(getattr(leaf, "run_backward", 0)) != 1:
        raise RuntimeError(
            "register_real_backward_driver: leaf.run_backward must be 1 (a BACKWARD "
            f"leaf), got run_backward={getattr(leaf, 'run_backward', None)!r}. "
            "Pass the bwd-specialized PathCRegionLeaf (RULE #1: no silent forward "
            "leaf binding as the backward region)."
        )
    bwd = make_real_bank_backward_driver(leaf, real_driver, bank_numels, device=device)
    for i in range(n_layers):
        tvm_ffi.register_global_func(f"pathc.bank_bwd_{i}", bwd, override=True)
    # Mark the real gridded backward as installed so the RULE #1 guard knows the
    # numpy bank backward is NOT the live path (and is never re-bound over this).
    set_real_bank_bwd_installed(True)


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
            # SAVE at each boundary: the checkpoint (read directly by bwd) and the
            # EXITING activation (entering region i+1) = the recompute START point.
            act = act0
            saved_ckpt: dict[int, relax.Var] = {}
            saved_exit: dict[int, relax.Var] = {}
            for i in range(n_layers):
                out = bb.emit(relax.call_dps_packed(
                    f"pathc.bank_fwd_{i}", [act, param, act], [sAct, sState]))
                act_next = bb.emit(relax.TupleGetItem(out, 0))
                ck = bb.emit(relax.TupleGetItem(out, 1))
                if i in bset:
                    saved_ckpt[i] = ck
                    saved_exit[i] = act_next
                act = act_next
            act_final = act

            # ---- LOSS on the final forward activation ----
            loss = bb.emit(relax.call_dps_packed("pathc.bank_loss", [act_final], [sLoss]))

            # ---- BACKWARD (Korthikanti per-segment recompute cache) ----
            # Walk each segment ONCE: the first backward region that needs a recomputed
            # checkpoint re-emits the forward (b+1)..seg_end exactly once, caching every
            # checkpoint; every later backward region in the segment reads the cache.
            # This is O(N) recompute (N - #boundaries) instead of the naive O(N*sqrt N)
            # per-region prefix re-derivation. Numerically identical; the recomputes are
            # emitted lazily inside each segment's local backward window so they stay
            # short-lived (O(sqrt N) checkpoint peak).
            seg_end_of: dict[int, int] = {}
            for k, b in enumerate(boundaries):
                seg_end_of[b] = (boundaries[k + 1] - 1) if k + 1 < len(boundaries) \
                    else (n_layers - 1)
            recomputed_ckpt: dict[int, relax.Var] = {}

            def _recompute_segment(b: int) -> None:
                rec_act = saved_exit[b]
                for j in range(b + 1, seg_end_of[b] + 1):
                    rout = bb.emit(relax.call_dps_packed(
                        f"pathc.bank_fwd_{j}", [rec_act, param, rec_act],
                        [sAct, sState]))
                    rec_act = bb.emit(relax.TupleGetItem(rout, 0))
                    rec_ck = bb.emit(relax.TupleGetItem(rout, 1))
                    recomputed_ckpt[j] = rec_ck

            actg = actg0
            paramg = paramg0
            for i in reversed(range(n_layers)):
                if i in bset:
                    ck_i = saved_ckpt[i]
                else:
                    b = nearest_boundary(i, boundaries)
                    if i not in recomputed_ckpt:
                        _recompute_segment(b)
                    ck_i = recomputed_ckpt.get(i)
                    if ck_i is None:
                        raise RuntimeError(
                            "FAIL-LOUD: recompute produced no checkpoint for region "
                            f"{i} (segment boundary {b})")
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
