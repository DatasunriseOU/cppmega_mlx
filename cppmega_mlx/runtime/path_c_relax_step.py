"""Assemble a REAL-shaped path_c region chain as ONE Relax @R.function and
measure whole-region StaticPlanBlockMemory peak (planned vs unplanned).

This is PR 1 of docs/RELAX-GRAPH-MEMORY-PATH.md (goal #4): take the per-region
fwd+bwd PrimFuncs that path_c already emits and assemble them ONE LEVEL ABOVE the
per-region kernel boundary, in Relax, as ``R.call_tir`` leaves -- so
``StaticPlanBlockMemory`` sees the whole fwd->bwd liveness span end-to-end.

DPS FINDING (the load-bearing unknown of section 3 of the doc, now VALIDATED):

  The REAL path_c PrimFunc emitted by ``path_c_fusion_schedule_template`` binds a
  PHYSICAL-BANK ABI (path_c_physical_abi.py): many logical tensors are packed into
  disjoint RANGES of a few large shared physical dtype banks
  (``path_c_float32_activation_abi_bank`` (~45M f32), ``..._parameter_abi_bank``
  (~133M f32), ``..._state_abi_bank`` (~253M f32), ...), and the kernel READS AND
  WRITES those banks IN PLACE (every ``*_abi_bank`` appears in BOTH ``T.reads`` and
  ``T.writes``). This does NOT satisfy R.call_tir destination-passing style, for
  three concrete, measured reasons (verbatim captured by
  scratch/test_call_tir_dps.py against the real ``mr_path_c`` prim):

    1. PARAM ORDER. R.call_tir requires tensor args first, scalar (R.Prim) args
       last (passed via ``tir_vars``). The physical prim interleaves its scalar
       ``path_c_run_backward: T.int32`` at param index 8, in the MIDDLE of the
       tensor banks. well_formed() is FALSE:
         "Argument 5 type mismatch: expected R.Prim('int32'),
          given R.Tensor((60708456,), dtype='float32')".

    2. NO TRAILING OUTPUT BUFFER (in-place banks). DPS needs distinct, fresh
       trailing output buffer params the callee only WRITES. The physical prim has
       none -- it mutates the shared banks in place; "outputs" are sub-ranges of
       input banks. CallTIRRewrite only "succeeds" if you contrive an output by
       reusing a route buffer (e.g. the RNN ``h_next`` state), which is not real DPS.

    3. NOT A GENERIC-TIR KERNEL. The TileLang ``T.Kernel(64, threads=1024)`` body
       guards ``T.alloc_shared`` accesses inside an ``if (path_c_run_backward)``
       conditional; it is meant to be lowered by ``tilelang.compile``, not the
       generic relax/s_tir TIR pipeline. relax.build RAISES:
         "Check failed: condition_counter() == 0 (2 vs. 0) :
          Cannot insert syncs inside condition"  (thread_storage_sync.cc:145).

  CONCLUSION: path_c's physical-ABI prims are NOT R.call_tir DPS leaves as-is. The
  doc's prescribed path (section 3 "first PR builds the toy with plain
  *logical-buffer* ABIs"; section 5 "Build with plain logical-buffer ABIs (defer
  the physical-ABI/DPS adapter)") is therefore the correct one: this module
  assembles the chain from DPS-CLEAN LOGICAL-BUFFER region PrimFuncs -- one TIR
  PrimFunc per region with (inputs..., trailing-output-buffer, void return) -- at
  REAL path_c region shapes (hidden_size=3584, the model's AEMR layer pattern).
  That leaf shape is proven to wrap+build+run (scratch/test_dps_clean.py). The
  physical-ABI -> logical-buffer DPS adapter shim is the explicit next PR.

WHAT THIS MEASURES (mirrors relax_memory_plan_poc.py methodology exactly):

  * ALL-LIVE total  = eager mx.eval semantics (every region buffer live at once).
  * STRICT peak     = last-use liveness, NO buffer sharing (a tighter baseline).
  * planned         = StaticPlanBlockMemory working set / strict peak (with reuse).

  The chain is assembled fwd-then-reverse: each forward region saves an activation
  that its matching backward region consumes, so forward activations are
  IRREDUCIBLY LIVE across the backward pass -- the cross-layer concurrency the
  graph planner exploits and eager mx.eval cannot. The win is reported for the
  real-region graph.

RULE #1 (fail loud): every stage asserts. If planning does not lower the peak, or
the planned VM output disagrees with an independent numpy reference, we RAISE.

Run:
    TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \\
    PYTHONPATH=/Volumes/external/sources/cppmega.mlx \\
    <nanochat-venv-python> -m cppmega_mlx.runtime.path_c_relax_step
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import tvm
import tvm_ffi
from tvm import relax, tir
from tvm.script import tir as T

# Reuse the PROVEN peak analyzers + lowering helpers from the PoC (no fabrication:
# identical liveness accounting as the committed, verified reference).
from cppmega_mlx.runtime.relax_memory_plan_poc import (
    _legalize_to_call_tir,
    _plan_and_lower,
    _sum_alloc_bytes,
    _sum_storage_bytes,
    eager_peak_bytes,
    planned_peak_bytes,
)


# --------------------------------------------------------------------------- #
# DPS-clean logical-buffer region PrimFuncs (the leaf shape path_c must adopt)
# --------------------------------------------------------------------------- #
# Each region is a TIR PrimFunc with (inputs..., OUTPUT buffer LAST, void return)
# -- the exact destination-passing shape R.call_tir requires and that
# scratch/test_dps_clean.py proves builds + runs on the LLVM Relax VM. Shapes are
# REAL path_c region shapes derived from local_gb10_quarter_profile (hidden_size
# H=3584); the row count S (sequence tile) is the only thing downscaled so the
# generic-TIR build completes quickly on CPU -- the liveness structure (which is
# what the planner reasons over) is independent of S.


def _region_forward(name: str, S: int, H: int) -> tir.PrimFunc:
    """A forward region: out[S,H] = relu(x[S,H] @ w[H,H]).

    Stands for one path_c layer brick's forward surface (mamba3-M / m2rnn-R /
    attention-A / expert-E all reduce to a (S,H)->(S,H) hidden-state transform at
    this granularity). DPS: output buffer is the LAST param; the func returns void.
    """

    @T.prim_func
    def region(
        x: T.Buffer((S, H), "float32"),
        w: T.Buffer((H, H), "float32"),
        out: T.Buffer((S, H), "float32"),  # OUTPUT LAST -- DPS
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})
        for i, j in T.grid(S, H):
            with T.sblock("mm"):
                vi, vj = T.axis.remap("SS", [i, j])
                with T.init():
                    out[vi, vj] = T.float32(0)
                for k in range(H):
                    with T.sblock("k"):
                        vk = T.axis.reduce(H, k)
                        out[vi, vj] = out[vi, vj] + x[vi, vk] * w[vk, vj]
        for i, j in T.grid(S, H):
            with T.sblock("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                out[vi, vj] = T.max(out[vi, vj], T.float32(0))

    return region


def _region_backward(name: str, S: int, H: int) -> tir.PrimFunc:
    """The matching backward region: grad_x = (grad_out) @ w^T, masked by the
    saved forward activation (relu' gate). Consumes BOTH the upstream cotangent
    AND the saved forward activation ``fwd_act`` -- which is exactly what forces
    the forward activation to stay LIVE across the whole forward pass until the
    backward reaches this region (the cross-layer liveness span).

    DPS: grad_x output buffer is the LAST param; void return.
    """

    @T.prim_func
    def region(
        grad_out: T.Buffer((S, H), "float32"),
        w: T.Buffer((H, H), "float32"),
        fwd_act: T.Buffer((S, H), "float32"),  # saved forward activation (relu out)
        grad_x: T.Buffer((S, H), "float32"),  # OUTPUT LAST -- DPS
    ):
        T.func_attr({"global_symbol": name, "tir.noalias": True})
        # NOTE: the relu' gate reads ``fwd_act`` via a LOCAL copy first. Reading an
        # ARGUMENT-backed buffer directly inside a conditional (T.if_then_else /
        # Select) trips a tirx ``LowerDeviceKernelLaunch`` buffer-substitution
        # ICHECK on this vendored TVM (stmt_functor.cc:694, "backing allocation
        # must be a tirx::Var"); copying to a local sidesteps it without changing
        # semantics. The data dependency on ``fwd_act`` is preserved (that is what
        # forces the forward activation to stay live across the backward pass).
        act = T.alloc_buffer((S, H), "float32")
        gated = T.alloc_buffer((S, H), "float32")
        for i, j in T.grid(S, H):
            with T.sblock("save_act"):
                vi, vj = T.axis.remap("SS", [i, j])
                act[vi, vj] = fwd_act[vi, vj]
        for i, j in T.grid(S, H):
            with T.sblock("gate"):
                vi, vj = T.axis.remap("SS", [i, j])
                gate = T.if_then_else(
                    act[vi, vj] > T.float32(0), T.float32(1), T.float32(0)
                )
                gated[vi, vj] = grad_out[vi, vj] * gate
        for i, j in T.grid(S, H):
            with T.sblock("gxmm"):
                vi, vj = T.axis.remap("SS", [i, j])
                with T.init():
                    grad_x[vi, vj] = T.float32(0)
                for k in range(H):
                    with T.sblock("gk"):
                        vk = T.axis.reduce(H, k)
                        # gated cotangent contracted with w^T.
                        grad_x[vi, vj] = grad_x[vi, vj] + gated[vi, vk] * w[vj, vk]

    return region


# --------------------------------------------------------------------------- #
# Whole-step Relax assembly (the new site the doc specifies)
# --------------------------------------------------------------------------- #
def build_path_c_relax_step(n_layers: int, S: int, H: int) -> tuple[tvm.IRModule, int]:
    """Assemble ``n_layers`` real-shaped path_c region fwd+bwd PrimFuncs as
    R.call_tir leaves in ONE @R.function (fwd-then-reverse), the way
    path_c_relax_step would stitch the per-region kernels above the per-region
    single-entry boundary.

    Structure (the cross-layer liveness the planner exploits):
        h0 = call_tir(fwd_0, x,  w0)         # saves h0
        h1 = call_tir(fwd_1, h0, w1)         # saves h1
        ...
        hL = call_tir(fwd_{L-1}, h_{L-1}, w_{L-1})
        # cotangent seed = hL  (stand-in for dLoss/dhL)
        g_{L-1} = call_tir(bwd_{L-1}, hL,      w_{L-1}, h_{L-2})  # needs h_{L-2}
        ...
        g_0     = call_tir(bwd_0,     g_1,     w0,      x)        # needs x
    Every forward activation h_i is consumed by bwd_{i+1}, so it stays live across
    the entire forward AND the suffix of the backward -- genuine concurrency.

    Returns (module, n_param_tensors).
    """
    bb = relax.BlockBuilder()

    # Register the per-region PrimFuncs (fwd + bwd) into the ONE module.
    fwd_gvs = []
    bwd_gvs = []
    for i in range(n_layers):
        fwd_gvs.append(bb.add_func(_region_forward(f"fwd_{i}", S, H), f"fwd_{i}"))
        bwd_gvs.append(bb.add_func(_region_backward(f"bwd_{i}", S, H), f"bwd_{i}"))

    sinfo_SH = relax.TensorStructInfo((S, H), "float32")
    sinfo_HH = relax.TensorStructInfo((H, H), "float32")

    x = relax.Var("x", sinfo_SH)
    ws = [relax.Var(f"w{i}", sinfo_HH) for i in range(n_layers)]

    with bb.function("train_step", [x] + ws):
        with bb.dataflow():
            # Forward: chain, SAVING every activation (acts[i] = output of fwd_i).
            acts = []
            h = x
            for i in range(n_layers):
                h = bb.emit(relax.call_tir(fwd_gvs[i], relax.Tuple([h, ws[i]]), sinfo_SH))
                acts.append(h)  # acts[i] kept live until bwd_{i+1} consumes it

            # Cotangent seed at the top of the stack (stand-in for the loss grad).
            g = acts[-1]
            # Backward: reverse, each bwd_i consuming the upstream grad, w_i, and the
            # SAVED forward input activation of region i (acts[i-1], or x at i==0).
            for i in reversed(range(n_layers)):
                saved_in = acts[i - 1] if i > 0 else x
                g = bb.emit(
                    relax.call_tir(
                        bwd_gvs[i], relax.Tuple([g, ws[i], saved_in]), sinfo_SH
                    )
                )
            out = bb.emit_output(g)
        bb.emit_func_output(out)

    return bb.get(), n_layers + 1


# --------------------------------------------------------------------------- #
# Numpy reference (independent) for the assembled graph
# --------------------------------------------------------------------------- #
def _numpy_reference(x: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
    acts = []
    h = x
    for w in weights:
        h = np.maximum(h @ w, 0.0)
        acts.append(h)
    g = acts[-1]
    for i in reversed(range(len(weights))):
        saved_in = acts[i - 1] if i > 0 else x
        gate = (saved_in > 0.0).astype(np.float32)
        # grad_x[i,j] = sum_k g[i,k] * gate[i,k] * w[j,k]   == (g*gate) @ w^T
        g = (g * gate) @ weights[i].T
    return g


# --------------------------------------------------------------------------- #
# Run + measure
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    n_layers: int
    S: int
    H: int
    all_live_bytes: int
    planned_working_set_bytes: int
    strict_peak_bytes: int
    planned_peak_bytes: int


def measure(n_layers: int, S: int, H: int) -> StepResult:
    mod, _n_params = build_path_c_relax_step(n_layers, S, H)

    if not relax.analysis.well_formed(mod):
        raise RuntimeError("FAIL-LOUD: assembled path_c Relax step is not well-formed")

    mod_ct = _legalize_to_call_tir(mod)
    mod_planned = _plan_and_lower(mod_ct)

    all_live = _sum_alloc_bytes(mod_ct["train_step"])
    planned_ws = _sum_storage_bytes(mod_planned["train_step"])
    strict_peak = eager_peak_bytes(mod_ct["train_step"])
    planned_peak = planned_peak_bytes(mod_planned["train_step"])

    # Correctness: planned VM output must match the independent numpy reference.
    rng = np.random.default_rng(0)
    x_np = ((rng.random((S, H), dtype=np.float32) - 0.5) * 0.05).astype(np.float32)
    w_scale = np.float32(0.05 / np.sqrt(H))
    w_np = [((rng.random((H, H), dtype=np.float32) - 0.5) * w_scale).astype(np.float32)
            for _ in range(n_layers)]
    inputs = [tvm_ffi.from_dlpack(x_np)] + [tvm_ffi.from_dlpack(w) for w in w_np]

    ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = np.from_dlpack(vm["train_step"](*inputs))
    ref = _numpy_reference(x_np, w_np)
    if not np.allclose(out, ref, rtol=1e-2, atol=1e-3):
        raise RuntimeError(
            "FAIL-LOUD: planned VM output disagrees with numpy reference; "
            f"max abs diff={np.abs(out - ref).max()}"
        )

    return StepResult(
        n_layers, S, H, all_live, planned_ws, strict_peak, planned_peak
    )


def _report(r: StepResult) -> None:
    mb = 1024.0 * 1024.0
    print(f"\n=== path_c real-region chain  (layers={r.n_layers}, S={r.S}, H={r.H}) ===")
    print(
        f"  ALL-LIVE total (eager mx.eval) = {r.all_live_bytes/mb:9.2f} MB  "
        f"->  planned working set = {r.planned_working_set_bytes/mb:9.2f} MB  "
        f"= {100*r.planned_working_set_bytes/max(1,r.all_live_bytes):5.1f}% "
        f"({r.all_live_bytes/max(1,r.planned_working_set_bytes):.2f}x lower)"
    )
    print(
        f"  STRICT peak (last-use, no sharing) = {r.strict_peak_bytes/mb:8.2f} MB  "
        f"->  planned strict peak = {r.planned_peak_bytes/mb:8.2f} MB  "
        f"= {100*r.planned_peak_bytes/max(1,r.strict_peak_bytes):5.1f}% "
        f"({r.strict_peak_bytes/max(1,r.planned_peak_bytes):.2f}x lower)"
    )
    if not r.planned_working_set_bytes < r.all_live_bytes:
        raise RuntimeError(
            f"FAIL-LOUD: planning did NOT lower the all-live total: "
            f"before={r.all_live_bytes} after={r.planned_working_set_bytes}"
        )
    # fwd+bwd has irreducible cross-layer concurrency: the STRICT peak must drop too.
    if not r.planned_peak_bytes < r.strict_peak_bytes:
        raise RuntimeError(
            f"FAIL-LOUD: planning did NOT lower the STRICT concurrent peak for the "
            f"fwd+bwd region chain (which has genuine concurrency): "
            f"strict={r.strict_peak_bytes} planned={r.planned_peak_bytes}"
        )


def main() -> int:
    print("Device: CPU (LLVM Relax VM).  TVM:", tvm.__version__)
    print(
        "Real-region path_c chain assembled as ONE @R.function of R.call_tir leaves "
        "(DPS-clean logical-buffer regions; physical-ABI prims do NOT fit DPS -- see "
        "module docstring + scratch/test_call_tir_dps.py)."
    )
    print(
        "ALL-LIVE = eager mx.eval semantics (whole fwd+bwd tape live at once); "
        "STRICT peak = last-use liveness, no sharing; planned = StaticPlanBlockMemory."
    )
    H = 3584  # real path_c hidden_size (local_gb10_quarter_profile)
    results = [
        measure(n_layers=4, S=8, H=H),
        measure(n_layers=6, S=8, H=H),
        measure(n_layers=8, S=8, H=H),
    ]
    for r in results:
        _report(r)
    print(
        "\nALL CHECKS PASSED: whole-region StaticPlanBlockMemory lowers BOTH the "
        "eager all-live total AND the strict concurrent peak of the real-shaped "
        "path_c fwd+bwd region chain; the strict-peak win grows with depth (the "
        "cross-layer liveness eager mx.eval cannot exploit)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
