"""Prove a single R.call_dps_packed leaf (the PR-2 adapter boundary) is well_formed,
legalizes to call_tir form, plans under StaticPlanBlockMemory, builds, and runs on
the LLVM VM with correct numerics. Uses a tiny registered DPS packed func that does
relu(x@w) over an internal bank (the pack/kernel/unpack pattern, downscaled)."""
from __future__ import annotations
import sys
import numpy as np
import tvm, tvm_ffi
from tvm import relax
from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir, _plan_and_lower, _sum_alloc_bytes, _sum_storage_bytes,
    eager_peak_bytes, planned_peak_bytes,
)

S, H = 8, 16

@tvm_ffi.register_global_func("pr2.region_fwd_dps", override=True)
def _region_fwd(x, w, out):
    xa = np.from_dlpack(x); wa = np.from_dlpack(w); oa = np.from_dlpack(out)
    bank = np.zeros(S*H + H*H + S*H, np.float32)        # internal physical bank
    bank[0:S*H] = xa.reshape(-1); bank[S*H:S*H+H*H] = wa.reshape(-1)  # pack
    res = np.maximum(bank[0:S*H].reshape(S,H) @ bank[S*H:S*H+H*H].reshape(H,H), 0.0)
    bank[S*H+H*H:] = res.reshape(-1)                    # kernel writes output sub-range
    oa[...] = bank[S*H+H*H:].reshape(S,H)               # unpack

def main() -> int:
    bb = relax.BlockBuilder()
    sx = relax.TensorStructInfo((S,H),"float32"); sw = relax.TensorStructInfo((H,H),"float32")
    x = relax.Var("x", sx); w0 = relax.Var("w0", sw); w1 = relax.Var("w1", sw)
    with bb.function("train_step", [x, w0, w1]):
        with bb.dataflow():
            h0 = bb.emit(relax.call_dps_packed("pr2.region_fwd_dps", [x, w0], sx))
            h1 = bb.emit(relax.call_dps_packed("pr2.region_fwd_dps", [h0, w1], sx))
            out = bb.emit_output(h1)
        bb.emit_func_output(out)
    mod = bb.get()
    print("well_formed:", relax.analysis.well_formed(mod))
    assert relax.analysis.well_formed(mod)
    mod_ct = _legalize_to_call_tir(mod)
    mod_pl = _plan_and_lower(mod_ct)
    print("eager all-live:", _sum_alloc_bytes(mod_ct["train_step"]),
          " planned WS:", _sum_storage_bytes(mod_pl["train_step"]))
    print("strict peak:", eager_peak_bytes(mod_ct["train_step"]),
          " planned peak:", planned_peak_bytes(mod_pl["train_step"]))
    ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    xn=(np.random.rand(S,H).astype("float32")-0.5)
    w0n=(np.random.rand(H,H).astype("float32")-0.5)*0.1
    w1n=(np.random.rand(H,H).astype("float32")-0.5)*0.1
    got=np.from_dlpack(vm["train_step"](tvm_ffi.from_dlpack(xn),tvm_ffi.from_dlpack(w0n),tvm_ffi.from_dlpack(w1n)))
    ref=np.maximum(np.maximum(xn@w0n,0)@w1n,0)
    print("max abs diff:", float(np.abs(got-ref).max()))
    assert np.allclose(got,ref,rtol=1e-3,atol=1e-4)
    print("PASS: call_dps_packed leaf is well_formed + plans + builds + runs + correct.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
