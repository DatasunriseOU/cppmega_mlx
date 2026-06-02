"""Confirm a DPS-clean TIR PrimFunc (output buffer LAST, void return) works as a
R.call_tir leaf and builds + runs on the LLVM Relax VM. This is the leaf shape
the real path_c regions must adopt (logical-buffer ABI) to be call_tir leaves."""
from __future__ import annotations

import sys
import numpy as np

import tvm
import tvm_ffi
from tvm import relax, tir
from tvm.script import tir as T


@T.prim_func
def mlp_region(
    x: T.Buffer((128, 128), "float32"),
    w: T.Buffer((128, 128), "float32"),
    out: T.Buffer((128, 128), "float32"),   # OUTPUT LAST -- DPS
):
    T.func_attr({"global_symbol": "mlp_region", "tir.noalias": True})
    for i, j in T.grid(128, 128):
        with T.sblock("acc"):
            vi, vj = T.axis.remap("SS", [i, j])
            with T.init():
                out[vi, vj] = T.float32(0)
            for k in range(128):
                with T.sblock("mm"):
                    vk = T.axis.reduce(128, k)
                    out[vi, vj] = out[vi, vj] + x[vi, vk] * w[vk, vj]


def main() -> int:
    bb = relax.BlockBuilder()
    gv = bb.add_func(mlp_region, "mlp_region")
    n = 128
    x = relax.Var("x", relax.TensorStructInfo((n, n), "float32"))
    w0 = relax.Var("w0", relax.TensorStructInfo((n, n), "float32"))
    w1 = relax.Var("w1", relax.TensorStructInfo((n, n), "float32"))
    out_sinfo = relax.TensorStructInfo((n, n), "float32")
    with bb.function("train_step", [x, w0, w1]):
        with bb.dataflow():
            h0 = bb.emit(relax.call_tir(gv, relax.Tuple([x, w0]), out_sinfo))
            h1 = bb.emit(relax.call_tir(gv, relax.Tuple([h0, w1]), out_sinfo))
            out = bb.emit_output(h1)
        bb.emit_func_output(out)
    mod = bb.get()
    print("well_formed:", relax.analysis.well_formed(mod))

    ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    xv = np.random.rand(n, n).astype("float32") * 0.01
    w0v = np.random.rand(n, n).astype("float32") * 0.01
    w1v = np.random.rand(n, n).astype("float32") * 0.01
    out = vm["train_step"](tvm_ffi.from_dlpack(xv), tvm_ffi.from_dlpack(w0v),
                           tvm_ffi.from_dlpack(w1v))
    got = np.from_dlpack(out)
    ref = (xv @ w0v) @ w1v
    print("max abs diff:", np.abs(got - ref).max())
    assert np.allclose(got, ref, rtol=1e-3, atol=1e-3), "DPS prim VM output mismatch"
    print("DPS-clean prim builds + runs + matches reference. OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
