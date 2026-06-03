"""PR 4 of docs/RELAX-GRAPH-MEMORY-PATH.md -- REMATERIALIZATION on the O(N)
checkpoint term (the measured highest-leverage remaining step).

PR 3 (path_c_relax_step_banks.py) exposed the physical banks as cross-region Relax
SSA tensors, so StaticPlanBlockMemory collapses EVERY bank to a constant working set
EXCEPT the O(N) state/checkpoint bank: forward region i WRITES checkpoint i and
backward region i READS it, so in a depth-N step ALL N forward checkpoints are
simultaneously live across the whole backward pass = O(N) * 0.943 GB/layer. The
MEASURED PR-3 28-layer (1.8B) planned peak is 27.38 GB, and the honest finding was
that liveness reuse CANNOT beat the checkpoint term -- only rematerialization can.

PR 4 implements sqrt(N) gradient checkpointing (Chen 2016 / the sqrt-N rule) ON the
real bank SSA graph, turning the PR-3 27.38 GB *projection* of ~7 GB into a MEASURED
peak:

  * Keep checkpoint BOUNDARIES every ceil(sqrt(N)) layers (the saved activation
    snapshots act[boundary]).
  * For NON-boundary backward regions, RE-EMIT the forward call_dps_packed
    (recompute) from the nearest saved boundary, regenerating that region's
    checkpoint LOCALLY inside the backward step. The recomputed checkpoint is born
    and killed within the local backward segment -> short-lived -> NOT live across
    the whole backward pass.

So the checkpoint term drops from O(N) saved-across-backward to O(sqrt N) saved
boundaries + O(sqrt N) transiently-live recompute checkpoints, at the cost of ~1
extra forward pass (the recompute calls). The activation/grad/param/grad-accum banks
keep their PR-3 cross-region SSA threading unchanged.

WHY THE RE-EMISSION MAKES THE CHECKPOINT SHORT-LIVED (the load-bearing point):
StaticPlanBlockMemory plans on textual alloc..last-use liveness. In PR-3 the
checkpoint tensor's alloc (fwd i) and last-use (bwd i) straddle the ENTIRE backward
pass, so all N coexist. In PR-4 the recomputed checkpoint's alloc (a recompute call
emitted INSIDE the backward step of region i) and its last-use (the bwd i call, the
very next binding) are ADJACENT -- the planner sees it die immediately and reuses the
storage for the next segment's recompute. Only the O(sqrt N) saved boundary
activations span the backward pass. This is the explicit remat the doc requires,
because StaticPlanBlockMemory cannot see through call_dps_packed to insert remat
itself (it must be EXPLICIT in the assembled graph).

NUMERIC EQUIVALENCE: recompute is mathematically identical (the recomputed checkpoint
== the forward-computed checkpoint, deterministic same op on the same saved
activation), so the remat assembly produces the SAME (actg, paramg) as PR-3. Verified
on the LLVM VM, max abs diff (RULE #1: any mismatch RAISES).

DEVICE: CPU LLVM Relax VM (planning is target-independent IR-level), same as PR-3.

Run:
    TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \\
    PYTHONPATH=/Volumes/external/sources/cppmega.mlx \\
    <python> -m cppmega_mlx.runtime.path_c_relax_step_remat
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

import tvm
import tvm_ffi
from tvm import relax

from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir,
    _plan_and_lower,
    _sum_alloc_bytes,
    _sum_storage_bytes,
)
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    BANK_ACT,
    BANK_ACTG,
    BANK_PARAM,
    BANK_PARAMG,
    BANK_STATE,
    BankSinfo,
    BankResult,
    real_bank_numels,
    register_bank_drivers,
    build_bank_chain,
    measure_banks,
    true_planned_peak,
)


# --------------------------------------------------------------------------- #
# sqrt(N) checkpoint schedule (Chen 2016 / sqrt-N rule).
# --------------------------------------------------------------------------- #
def checkpoint_boundaries(n_layers: int) -> list[int]:
    """The saved checkpoint boundaries: layer 0, then every ceil(sqrt(N)) layers.

    Returns the forward-region indices whose ENTERING activation act[i] is SAVED
    (a snapshot kept live across the backward pass). Non-boundary regions recompute
    their checkpoint during backward from the nearest preceding boundary.
    """
    if n_layers <= 0:
        raise ValueError(f"FAIL-LOUD: n_layers must be >0, got {n_layers}")
    seg = max(1, math.ceil(math.sqrt(n_layers)))
    return list(range(0, n_layers, seg))


def nearest_boundary(i: int, boundaries: list[int]) -> int:
    """The greatest boundary <= i (the segment start that region i recomputes from)."""
    chosen = boundaries[0]
    for b in boundaries:
        if b <= i:
            chosen = b
        else:
            break
    return chosen


# --------------------------------------------------------------------------- #
# Remat bank-SSA assembly.
#
# The forward driver (pathc.bank_fwd_i, registered by PR-3's register_bank_drivers)
# computes (act_out, state_out=checkpoint_i) from (act_in, param, state_in). PR-3
# threads act forward and SAVES every checkpoint. PR-4 instead:
#
#   FORWARD: thread act forward as PR-3, but SAVE act[i] ONLY at boundaries (the
#            saved activation snapshots) and SAVE checkpoint_i ONLY at boundaries.
#            (Both come from the same fwd call -- we keep the boundary ones live and
#            let the non-boundary act/checkpoint die at last use, exactly as the
#            forward-flowing activation bank already does in PR-3.)
#
#   BACKWARD region i (non-boundary): RE-EMIT forward calls from the nearest saved
#            boundary up to i to REGENERATE checkpoint_i LOCALLY, immediately before
#            the bwd i call that consumes it. The recompute call's output checkpoint
#            is killed right after bwd i -> short-lived.
#
#   BACKWARD region i (boundary): its checkpoint was saved in forward -> no recompute.
# --------------------------------------------------------------------------- #
def build_remat_bank_chain(numels: dict[str, int], n_layers: int) -> tvm.IRModule:
    """Assemble the whole fwd+bwd step with sqrt(N) remat on the checkpoint bank.

    Banks are Relax-level SSA tensors (PR-3). The ONLY change vs build_bank_chain is
    the checkpoint liveness: saved checkpoints exist ONLY at boundaries; every other
    backward region RE-EMITS the forward (recompute) to regenerate its checkpoint
    locally. Numerically identical; the planner sees the recomputed checkpoints as
    short-lived (alloc..use adjacent), so the peak checkpoint term is O(sqrt N).
    """
    register_bank_drivers(numels, n_layers)
    boundaries = checkpoint_boundaries(n_layers)
    bset = set(boundaries)

    sAct = BankSinfo(numels[BANK_ACT]).sinfo()
    sActG = BankSinfo(numels[BANK_ACTG]).sinfo()
    sParam = BankSinfo(numels[BANK_PARAM]).sinfo()
    sParamG = BankSinfo(numels[BANK_PARAMG]).sinfo()
    sState = BankSinfo(numels[BANK_STATE]).sinfo()

    bb = relax.BlockBuilder()
    act0 = relax.Var("act0", sAct)
    param = relax.Var("param", sParam)
    paramg0 = relax.Var("paramg0", sParamG)
    actg0 = relax.Var("actg0", sActG)
    with bb.function("train_step", [act0, param, paramg0, actg0]):
        with bb.dataflow():
            # ---- FORWARD ----
            # Thread act forward through every region. At each boundary SAVE (a) the
            # checkpoint_i (read directly by bwd i, no recompute) and (b) the EXITING
            # activation act[i+1] (the activation ENTERING region i+1), which is the
            # recompute START point for that segment's backward pass. Non-boundary
            # act/checkpoint outputs die at last use (forward-flowing). Only O(sqrt N)
            # boundary checkpoints + O(sqrt N) boundary exit-activations span backward.
            act = act0
            saved_ckpt: dict[int, relax.Var] = {}   # boundary -> saved checkpoint
            saved_exit: dict[int, relax.Var] = {}   # boundary -> activation EXITING boundary
            for i in range(n_layers):
                out = bb.emit(relax.call_dps_packed(
                    f"pathc.bank_fwd_{i}", [act, param, act],
                    [sAct, sState]))
                act_next = bb.emit(relax.TupleGetItem(out, 0))
                ck = bb.emit(relax.TupleGetItem(out, 1))
                if i in bset:
                    saved_ckpt[i] = ck  # boundary checkpoint: live until bwd i
                    saved_exit[i] = act_next  # activation ENTERING region i+1 (recompute start)
                act = act_next

            # ---- BACKWARD (Korthikanti per-segment recompute cache) ----
            # For each region i (reverse): obtain checkpoint_i. If i is a boundary,
            # use the saved checkpoint. Else the checkpoint must be RECOMPUTED from the
            # nearest saved boundary. The KEY anti-pattern fix vs the naive remat: do
            # NOT re-derive the prefix b..i independently for every non-boundary i
            # (that is the O(N*sqrt N) checkpointing anti-pattern -- region b+1
            # recomputes [b..b+1], b+2 recomputes [b..b+2] from scratch, etc., each
            # re-running the same prefix). Instead WALK EACH SEGMENT ONCE: the first
            # time the backward pass needs a recomputed checkpoint in a segment,
            # re-emit the forward b..segment_end exactly once and CACHE every
            # checkpoint b+1..segment_end. Every later backward region in that segment
            # reads its checkpoint straight from the cache -- zero extra forward calls.
            #
            # Recompute count drops from O(N*sqrt N) (sum 2+3+..+L per segment) to
            # O(N) (each non-boundary region recomputed exactly once: N - #boundaries
            # total). Numerically IDENTICAL: the cached ck_j is the same op on the same
            # boundary activation as before, just emitted once instead of redundantly.
            #
            # Liveness is preserved: a segment's recompute is emitted lazily, the FIRST
            # time (in reverse order) that segment is entered -- i.e. immediately before
            # that segment's highest backward region consumes its checkpoint -- so the
            # recomputed checkpoints are born inside the segment's local backward window
            # and the planner still sees them as short-lived (O(sqrt N) peak), exactly
            # as the naive remat. We only remove the REDUNDANT re-emissions.
            seg_end_of: dict[int, int] = {}  # boundary b -> last region index in its segment
            for k, b in enumerate(boundaries):
                seg_end_of[b] = (boundaries[k + 1] - 1) if k + 1 < len(boundaries) \
                    else (n_layers - 1)
            recomputed_ckpt: dict[int, relax.Var] = {}  # region j -> its recomputed checkpoint

            def _recompute_segment(b: int) -> None:
                """Re-emit the forward (b+1)..seg_end ONCE, caching every checkpoint.
                Starts from saved_exit[b] (the activation entering region b+1), so the
                boundary region b's own forward is NOT re-run (its checkpoint is already
                saved). Lazy + idempotent: called the first time the backward pass needs
                a recomputed checkpoint in segment b, then never re-emitted. Emits
                exactly (seg_end - b) forward calls = one per non-boundary region."""
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
                        # first backward region in this segment -> recompute it ONCE.
                        _recompute_segment(b)
                    ck_i = recomputed_ckpt.get(i)
                    if ck_i is None:
                        raise RuntimeError(
                            "FAIL-LOUD: recompute produced no checkpoint for "
                            f"region {i} (segment boundary {b})")
                out = bb.emit(relax.call_dps_packed(
                    f"pathc.bank_bwd_{i}", [actg, param, ck_i, paramg],
                    [sActG, sParamG]))
                actg = bb.emit(relax.TupleGetItem(out, 0))
                paramg = bb.emit(relax.TupleGetItem(out, 1))
            res = bb.emit_output(relax.Tuple([actg, paramg]))
        bb.emit_func_output(res)
    return bb.get()


# --------------------------------------------------------------------------- #
# Recompute-overhead accounting: how many EXTRA forward region calls remat issues.
# --------------------------------------------------------------------------- #
def recompute_overhead(n_layers: int) -> tuple[int, int, float]:
    """Returns (n_forward_calls_baseline, n_extra_recompute_calls, overhead_factor).

    Baseline forward calls = n_layers (one per region). With the Korthikanti
    per-segment recompute cache, each segment is walked exactly ONCE during backward
    (re-emitting (b+1)..seg_end), so each NON-boundary region is recomputed exactly
    once -> extra recompute calls = n_layers - #boundaries. This is the O(N) lower
    bound for sqrt-N checkpointing; the naive remat's redundant per-region prefix
    re-derivation (O(N*sqrt N), sum 2+3+..+L per segment) is eliminated.
    """
    boundaries = checkpoint_boundaries(n_layers)
    extra = n_layers - len(boundaries)
    return n_layers, extra, extra / max(1, n_layers)


# --------------------------------------------------------------------------- #
# Measure the REMAT assembly: planned peak via the SAME true_planned_peak analyzer.
# --------------------------------------------------------------------------- #
@dataclass
class RematResult:
    n_layers: int
    all_live: int
    planned_ws: int
    planned_peak: int
    fwd_calls: int
    recompute_calls: int


def measure_remat(numels: dict[str, int], n_layers: int,
                  *, run_vm: bool, scale: float = 1.0) -> RematResult:
    scaled = {k: max(1, int(v * scale)) for k, v in numels.items()}
    mod = build_remat_bank_chain(scaled, n_layers)
    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: remat bank-SSA step is not well-formed")
    mod_ct = _legalize_to_call_tir(mod)
    mod_pl = _plan_and_lower(mod_ct)
    fwd, extra, _ = recompute_overhead(n_layers)
    res = RematResult(
        n_layers,
        _sum_alloc_bytes(mod_ct["train_step"]),
        _sum_storage_bytes(mod_pl["train_step"]),
        true_planned_peak(mod_pl["train_step"]),
        fwd, extra,
    )
    if run_vm:
        _verify_remat_numerics(scaled, n_layers, mod)
    return res


def _numpy_reference(numels: dict[str, int], n_layers: int,
                     act0, param, paramg0, actg0):
    """Independent numpy reference of the FULL (non-remat) bank-SSA dataflow.
    The remat assembly must match this byte-for-byte (recompute is identical)."""
    bias = np.float32(param[:1].sum() * 1e-6)
    act = act0.copy()
    ckpts = []
    for _ in range(n_layers):
        act = np.maximum(act + bias, 0.0)
        ck = np.zeros(numels[BANK_STATE], np.float32)
        n = min(ck.size, act.size)
        ck[:n] = act[:n]
        ckpts.append(ck)
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
    return actg, paramg


def _verify_remat_numerics(numels: dict[str, int], n_layers: int,
                           mod: tvm.IRModule) -> float:
    """Run the planned REMAT VM and check it matches the independent numpy reference
    of the FULL (non-remat) dataflow. recompute is mathematically identical, so they
    MUST agree. RULE #1: RAISE on any mismatch. Returns the max abs diff."""
    rng = np.random.default_rng(0)
    act0 = (rng.random(numels[BANK_ACT], np.float32) - 0.5).astype(np.float32)
    param = (rng.random(numels[BANK_PARAM], np.float32) - 0.5).astype(np.float32)
    paramg0 = np.zeros(numels[BANK_PARAMG], np.float32)
    actg0 = (rng.random(numels[BANK_ACTG], np.float32) - 0.5).astype(np.float32)

    ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out_actg, out_paramg = vm["train_step"](
        tvm_ffi.from_dlpack(act0), tvm_ffi.from_dlpack(param),
        tvm_ffi.from_dlpack(paramg0), tvm_ffi.from_dlpack(actg0))
    got_actg = np.from_dlpack(out_actg)
    got_paramg = np.from_dlpack(out_paramg)

    ref_actg, ref_paramg = _numpy_reference(numels, n_layers, act0, param,
                                            paramg0, actg0)
    d_actg = float(np.abs(got_actg - ref_actg).max())
    d_paramg = float(np.abs(got_paramg - ref_paramg).max())
    if not np.allclose(got_actg, ref_actg, rtol=1e-3, atol=1e-4):
        raise RuntimeError(
            "FAIL-LOUD: remat planned VM actg disagrees with FULL numpy reference; "
            f"max abs diff={d_actg} -- recompute is NOT numerically equivalent")
    if not np.allclose(got_paramg, ref_paramg, rtol=1e-3, atol=1e-4):
        raise RuntimeError(
            "FAIL-LOUD: remat planned VM paramg disagrees with FULL numpy reference; "
            f"max abs diff={d_paramg} -- recompute is NOT numerically equivalent")
    return max(d_actg, d_paramg)


def _verify_remat_equals_nonremat(numels: dict[str, int], n_layers: int) -> float:
    """Strongest equivalence: run BOTH the PR-3 non-remat AND the PR-4 remat planned
    VMs on the SAME inputs and check identical outputs. RULE #1: RAISE on mismatch."""
    rng = np.random.default_rng(7)
    act0 = (rng.random(numels[BANK_ACT], np.float32) - 0.5).astype(np.float32)
    param = (rng.random(numels[BANK_PARAM], np.float32) - 0.5).astype(np.float32)
    paramg0 = np.zeros(numels[BANK_PARAMG], np.float32)
    actg0 = (rng.random(numels[BANK_ACTG], np.float32) - 0.5).astype(np.float32)
    args = (tvm_ffi.from_dlpack(act0), tvm_ffi.from_dlpack(param),
            tvm_ffi.from_dlpack(paramg0), tvm_ffi.from_dlpack(actg0))

    mod_full = build_bank_chain(numels, n_layers)
    ex_full = tvm.compile(mod_full, target=tvm.target.Target("llvm"))
    vm_full = relax.VirtualMachine(ex_full, tvm.cpu())
    f_actg, f_paramg = vm_full["train_step"](*args)

    mod_remat = build_remat_bank_chain(numels, n_layers)
    ex_remat = tvm.compile(mod_remat, target=tvm.target.Target("llvm"))
    vm_remat = relax.VirtualMachine(ex_remat, tvm.cpu())
    r_actg, r_paramg = vm_remat["train_step"](*args)

    da = float(np.abs(np.from_dlpack(f_actg) - np.from_dlpack(r_actg)).max())
    dp = float(np.abs(np.from_dlpack(f_paramg) - np.from_dlpack(r_paramg)).max())
    if da != 0.0 or dp != 0.0:
        raise RuntimeError(
            "FAIL-LOUD: remat output differs from non-remat output: "
            f"max abs diff actg={da} paramg={dp}")
    return max(da, dp)


def report_remat(r: RematResult, base: BankResult, *, label: str) -> None:
    gb = 1024.0 ** 3
    print(f"\n=== {label}  layers={r.n_layers} (sqrt-N remat on checkpoint bank) ===")
    boundaries = checkpoint_boundaries(r.n_layers)
    print(f"  saved checkpoint boundaries = {boundaries} "
          f"({len(boundaries)} of {r.n_layers})")
    print(f"  PR-3 banks-only planned peak = {base.planned_peak/gb:8.3f} GB")
    print(f"  PR-4 remat   planned peak    = {r.planned_peak/gb:8.3f} GB  "
          f"({base.planned_peak/max(1,r.planned_peak):5.2f}x lower than PR-3)")
    print(f"  vs eager all-live {base.all_live/gb:8.3f} GB -> "
          f"{base.all_live/max(1,r.planned_peak):5.2f}x lower")
    print(f"  recompute overhead = {r.recompute_calls} extra fwd calls "
          f"on {r.fwd_calls} baseline ({r.recompute_calls/max(1,r.fwd_calls):.2f}x "
          f"extra forward work)")
    # FAIL-LOUD: remat MUST lower the PR-3 banks-only peak (that is the whole point).
    if not r.planned_peak < base.planned_peak:
        raise RuntimeError(
            "FAIL-LOUD: remat did NOT lower the PR-3 banks-only peak: "
            f"remat={r.planned_peak} pr3={base.planned_peak}")


def main() -> int:
    gb = 1024.0 ** 3
    print("PR 4 -- REMATERIALIZATION (sqrt-N) ON THE O(N) CHECKPOINT BANK.")
    print("Device: CPU LLVM Relax VM. TVM:", tvm.__version__)
    numels = real_bank_numels()
    total_mb = sum(numels.values()) * 4 / 1024 / 1024
    print(f"real per-region banks parsed: {len(numels)} banks, {total_mb:.1f} MB/region")
    print(f"  state/checkpoint bank = {numels[BANK_STATE]*4/gb:.3f} GB/layer "
          f"(the O(N) term remat attacks)")

    # 1) NUMERIC VALIDATION (downscaled, VM runs on CPU): remat == full (non-remat).
    print("\n--- numeric validation: remat == non-remat (recompute is identical) ---")
    vm_numels = {k: max(8, v // 20000) for k, v in numels.items()}
    for nl in (4, 8, 28):
        d_ref = _verify_remat_numerics(vm_numels, nl, build_remat_bank_chain(vm_numels, nl))
        d_eq = _verify_remat_equals_nonremat(vm_numels, nl)
        print(f"  layers={nl:>2}: remat vs numpy-ref max|diff|={d_ref:.1e}; "
              f"remat vs non-remat VM max|diff|={d_eq:.1e}  PASS")
    # stress downscale (denser banks) at 8 layers
    vm_stress = {k: max(8, v // 2000) for k, v in numels.items()}
    d_eq = _verify_remat_equals_nonremat(vm_stress, 8)
    print(f"  stress /2000 layers=8: remat vs non-remat VM max|diff|={d_eq:.1e}  PASS")

    # 2) FULL-SCALE peak: PR-3 banks-only vs PR-4 remat, at 4/8/28 layers (+16 for slope).
    print("\n--- FULL-SCALE peak: PR-3 banks-only vs PR-4 remat (real numels) ---")
    rows = []
    for nl in (4, 8, 16, 28):
        base = measure_banks(numels, nl, run_vm=False, scale=1.0)
        r = measure_remat(numels, nl, run_vm=False, scale=1.0)
        rows.append((nl, base, r))
        report_remat(r, base, label="REAL banks")

    print("\n--- scaling table: eager / PR-3 banks / PR-4 remat (real banks) ---")
    print(f"  {'layers':>6} {'eager all-live':>15} {'PR-3 banks':>12} "
          f"{'PR-4 remat':>12} {'remat vs eager':>15} {'recompute':>11}")
    for nl, base, r in rows:
        print(f"  {nl:>6} {base.all_live/gb:>12.2f} GB {base.planned_peak/gb:>9.2f} GB "
              f"{r.planned_peak/gb:>9.2f} GB {base.all_live/max(1,r.planned_peak):>13.2f}x "
              f"{r.recompute_calls:>5} calls")

    # 3) The 1.8B (28-block) headline + the in-place-optimizer note.
    nl28 = [row for row in rows if row[0] == 28][0]
    _, base28, r28 = nl28
    print("\n--- 1.8B STEP HEADLINE (28 MR blocks) ---")
    print(f"  eager all-live (activation/grad term) : {base28.all_live/gb:7.2f} GB")
    print(f"  PR-3 banks-exposed planned peak       : {base28.planned_peak/gb:7.2f} GB")
    print(f"  PR-4 sqrt-N remat MEASURED peak       : {r28.planned_peak/gb:7.2f} GB")
    print(f"  recompute overhead                    : {r28.recompute_calls} extra "
          f"fwd calls on {r28.fwd_calls} ({r28.recompute_calls/r28.fwd_calls:.2f}x)")

    # In-place optimizer note (lever 5): the Adam m/v static term, NOT in this graph.
    # Adam state = m + v = 2x params. param bank = 360.8 MB/layer here is the WEIGHTS
    # bank; the optimizer m/v are a SEPARATE 2x-params static term added at update.
    p_gb = numels[BANK_PARAM] * 4 / gb
    print("\n--- in-place optimizer note (lever 5, the doc's last lever) ---")
    print(f"  Adam m/v state = 2x params. Per the parameter bank ({p_gb:.2f} GB/layer "
          f"WEIGHTS), the optimizer m,v add ~{2*p_gb:.2f} GB/layer of STATIC state.")
    print("  In-place Adam (param<-update, m,v updated in place) keeps that term at 1x")
    print("  instead of allocating fresh m',v'. Across the external call_dps_packed")
    print("  boundary StaticPlanBlockMemory does NOT alias in place (PR-3 finding), so")
    print("  this MUST be an explicit in-place op, not a planner freebie -- specified,")
    print("  not yet wired (it attacks the optimizer-state term, orthogonal to remat).")

    print("\nALL CHECKS PASSED: sqrt-N remat re-emits the forward call_dps_packed for "
          "non-boundary backward regions, making their checkpoints short-lived; the "
          "MEASURED 28-block planned peak drops from the PR-3 O(N) "
          f"{base28.planned_peak/gb:.2f} GB to O(sqrt N) {r28.planned_peak/gb:.2f} GB, "
          "numerically equivalent to non-remat (max abs diff 0.0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
