"""PR-3 probe: can a R.call_dps_packed leaf take MULTIPLE bank tensors as inputs and
return MULTIPLE (a TUPLE of) updated bank tensors as outputs, with StaticPlanBlockMemory
co-planning the banks across a chain of such leaves? This is the core mechanism of the
bank-as-Relax-tensor SSA assembly: a region reads bank tensors, writes updated bank
tensors, and the planner aliases dead bank storage across regions.

Downscaled banks so it runs fast on CPU; the mechanism (tuple-out DPS + cross-region
bank planning) is what PR-3 needs to validate before scaling to the real bank sizes."""
from __future__ import annotations
import sys
import numpy as np
import tvm, tvm_ffi
from tvm import relax
from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir, _plan_and_lower, _sum_alloc_bytes, _sum_storage_bytes,
    eager_peak_bytes, planned_peak_bytes,
)

# Two banks: an "activation" bank A and a "state" bank S (downscaled).
NA, NS = 4096, 8192


@tvm_ffi.register_global_func("pr3.region_dps", override=True)
def _region(actA, stateS, outA, outS):
    a = np.from_dlpack(actA); s = np.from_dlpack(stateS)
    oa = np.from_dlpack(outA); os_ = np.from_dlpack(outS)
    # region "compute": new activation bank = relu(a + 0.01), new state bank = s rotated
    oa[...] = np.maximum(a + 0.01, 0.0)
    os_[...] = s * 0.999 + 0.001


def main() -> int:
    bb = relax.BlockBuilder()
    sA = relax.TensorStructInfo((NA,), "float32")
    sS = relax.TensorStructInfo((NS,), "float32")
    A0 = relax.Var("A0", sA)
    S0 = relax.Var("S0", sS)
    n_layers = 4
    with bb.function("train_step", [A0, S0]):
        with bb.dataflow():
            a, s = A0, S0
            for i in range(n_layers):
                out = bb.emit(relax.call_dps_packed(
                    "pr3.region_dps", [a, s],
                    [sA, sS]))
                a = bb.emit(relax.TupleGetItem(out, 0))
                s = bb.emit(relax.TupleGetItem(out, 1))
            res = bb.emit_output(relax.Tuple([a, s]))
        bb.emit_func_output(res)
    mod = bb.get()
    print("well_formed:", relax.analysis.well_formed(mod))
    assert relax.analysis.well_formed(mod)
    mod_ct = _legalize_to_call_tir(mod)
    mod_pl = _plan_and_lower(mod_ct)
    print("eager all-live :", _sum_alloc_bytes(mod_ct["train_step"]))
    print("planned WS     :", _sum_storage_bytes(mod_pl["train_step"]))
    print("strict peak    :", eager_peak_bytes(mod_ct["train_step"]))
    print("planned peak   :", planned_peak_bytes(mod_pl["train_step"]))
    ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    A = (np.random.rand(NA).astype("float32") - 0.5)
    S = (np.random.rand(NS).astype("float32") - 0.5)
    gotA, gotS = vm["train_step"](tvm_ffi.from_dlpack(A), tvm_ffi.from_dlpack(S))
    gotA = np.from_dlpack(gotA); gotS = np.from_dlpack(gotS)
    # numpy ref
    a, s = A.copy(), S.copy()
    for _ in range(n_layers):
        a = np.maximum(a + 0.01, 0.0); s = s * 0.999 + 0.001
    print("max abs diff A:", float(np.abs(gotA - a).max()),
          " S:", float(np.abs(gotS - s).max()))
    assert np.allclose(gotA, a, atol=1e-5) and np.allclose(gotS, s, atol=1e-5)
    print("PASS: tuple-out DPS bank leaf plans + builds + runs + correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
