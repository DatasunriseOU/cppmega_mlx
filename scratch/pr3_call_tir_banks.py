"""PR-3 critical test: is StaticPlanBlockMemory's mis-planning of the bank tensors
caused by the call_dps_packed EXTERNAL boundary being opaque to the planner's
liveness (it can't see which banks the packed func WRITES, so it kills the output
storage immediately)? Re-assemble the SAME bank-SSA chain with call_tir DPS leaves
(transparent to the planner: the output buffer is a real DPS param the planner
tracks) and check whether the checkpoint banks are then kept live O(N) correctly.

This isolates the limitation: call_dps_packed outputs are NOT properly co-planned;
only call_tir DPS outputs are. The real path_c kernel can ONLY be call_dps_packed
(PR-2 mismatch #3), so this bounds what the graph path can plan for the real kernel."""
from __future__ import annotations
import sys
import numpy as np
import tvm
from tvm import relax, tir
from tvm.script import tir as T
from cppmega_mlx.runtime.path_c_relax_step_banks import (
    real_bank_numels, BANK_ACT, BANK_ACTG, BANK_PARAM, BANK_PARAMG, BANK_STATE,
)
from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir, _plan_and_lower, _sum_alloc_bytes, _sum_storage_bytes,
    eager_peak_bytes, planned_peak_bytes,
)


def make_fwd_prim(na, ns):
    @T.prim_func
    def fwd(act_in: T.Buffer((na,), "float32"),
            param: T.Buffer((1,), "float32"),
            act_out: T.Buffer((na,), "float32"),
            ckpt_out: T.Buffer((ns,), "float32")):
        T.func_attr({"global_symbol": "fwd", "tir.noalias": True})
        for i in range(na):
            with T.sblock("a"):
                vi = T.axis.remap("S", [i])
                act_out[vi] = T.max(act_in[vi] + param[0] * T.float32(1e-6), T.float32(0))
        for j in range(ns):
            with T.sblock("c"):
                vj = T.axis.remap("S", [j])
                ckpt_out[vj] = act_out[vj] if vj < na else T.float32(0)
    return fwd


def make_bwd_prim(na, nag, npg):
    @T.prim_func
    def bwd(actg_in: T.Buffer((nag,), "float32"),
            ckpt: T.Buffer((na,), "float32"),
            paramg_in: T.Buffer((npg,), "float32"),
            actg_out: T.Buffer((nag,), "float32"),
            paramg_out: T.Buffer((npg,), "float32")):
        T.func_attr({"global_symbol": "bwd", "tir.noalias": True})
        for i in range(nag):
            with T.sblock("g"):
                vi = T.axis.remap("S", [i])
                gate = T.if_then_else(ckpt[vi] > T.float32(0), T.float32(1), T.float32(0))
                actg_out[vi] = actg_in[vi] * gate
        for j in range(npg):
            with T.sblock("p"):
                vj = T.axis.remap("S", [j])
                paramg_out[vj] = paramg_in[vj] + T.float32(1e-3)
    return bwd


def build_call_tir_bank_chain(numels, n_layers):
    na = numels[BANK_ACT]
    ns = numels[BANK_STATE]
    nag = numels[BANK_ACTG]
    npg = numels[BANK_PARAMG]
    bb = relax.BlockBuilder()
    sAct = relax.TensorStructInfo((na,), "float32")
    sState = relax.TensorStructInfo((ns,), "float32")
    sActG = relax.TensorStructInfo((nag,), "float32")
    sParamG = relax.TensorStructInfo((npg,), "float32")
    sParam1 = relax.TensorStructInfo((1,), "float32")
    act0 = relax.Var("act0", sAct)
    param = relax.Var("param", sParam1)
    paramg0 = relax.Var("paramg0", sParamG)
    actg0 = relax.Var("actg0", sActG)
    gfwd = []
    gbwd = []
    for i in range(n_layers):
        gfwd.append(bb.add_func(make_fwd_prim(na, ns), f"fwd_{i}"))
        gbwd.append(bb.add_func(make_bwd_prim(na, nag, npg), f"bwd_{i}"))
    with bb.function("train_step", [act0, param, paramg0, actg0]):
        with bb.dataflow():
            act = act0
            ckpts = []
            for i in range(n_layers):
                out = bb.emit(relax.call_tir(
                    gfwd[i], [act, param],
                    [sAct, sState]))
                act = bb.emit(relax.TupleGetItem(out, 0))
                ck = bb.emit(relax.TupleGetItem(out, 1))
                ckpts.append(ck)
            actg = actg0
            paramg = paramg0
            for i in reversed(range(n_layers)):
                # ckpt is (ns,) but bwd wants (na,) view of its prefix; downscale: ns>=na here? use na-sized slice via a separate ckpt prim. For test, pass act-sized ckpt.
                out = bb.emit(relax.call_tir(
                    gbwd[i], [actg, ckpts[i], paramg],
                    [sActG, sParamG]))
                actg = bb.emit(relax.TupleGetItem(out, 0))
                paramg = bb.emit(relax.TupleGetItem(out, 1))
            res = bb.emit_output(relax.Tuple([actg, paramg]))
        bb.emit_func_output(res)
    return bb.get()


def main():
    numels = real_bank_numels()
    # bwd prim reads ckpt as (na,) -- but ckpt bank is (ns,). To keep the call_tir
    # buffer shapes consistent we use ns==na for the ckpt in THIS isolation test
    # (the point is the PLANNER liveness, not the exact compute), so shrink state
    # to the activation size for the call_tir variant.
    numels = dict(numels)
    numels[BANK_STATE] = numels[BANK_ACT]
    gb = 1024 ** 3
    print(f"{'L':>3} {'all-live GB':>14} {'planned-WS GB':>14} "
          f"{'strict-peak GB':>15} {'planned-peak GB':>16}")
    for nl in (2, 4, 8, 16, 28):
        mod = build_call_tir_bank_chain(numels, nl)
        if not relax.analysis.well_formed(mod):
            raise RuntimeError("FAIL-LOUD: call_tir bank chain not well-formed")
        ct = _legalize_to_call_tir(mod)
        pl = _plan_and_lower(ct)
        al = _sum_alloc_bytes(ct["train_step"])
        ws = _sum_storage_bytes(pl["train_step"])
        sp = eager_peak_bytes(ct["train_step"])
        pp = planned_peak_bytes(pl["train_step"])
        print(f"{nl:>3} {al/gb:>14.2f} {ws/gb:>14.2f} {sp/gb:>15.2f} {pp/gb:>16.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
