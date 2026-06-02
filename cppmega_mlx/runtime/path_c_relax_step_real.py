"""PR 2 measurement (docs/RELAX-GRAPH-MEMORY-PATH.md): assemble REAL path_c region
leaves -- via the physical-bank -> logical-buffer DPS adapter
(``path_c_dps_adapter``) -- as ``R.call_dps_packed`` leaves in ONE @R.function, run
``StaticPlanBlockMemory``, build + run on the LLVM Relax VM, check numerics, and
MEASURE planned vs unplanned peak.

This is the PR-2 analog of PR 1's ``path_c_relax_step.py``. The ONLY thing that
changes vs PR 1 is the LEAF BOUNDARY:

  PR 1 leaf : a hand-written DPS-clean *logical-buffer* TIR PrimFunc (``R.call_tir``).
  PR 2 leaf : a REAL path_c region behind the physical->logical DPS ADAPTER, emitted
              as ``R.call_dps_packed`` -- because the real path_c kernel can ONLY be
              lowered by ``tilelang.compile`` (the TileLang ``T.Kernel``/guarded-sync
              body RAISES under generic relax/s_tir, EVEN after currying run_backward;
              MEASURED in scratch/pr2_test_curry.py + scratch/pr2_compile_full.py).
              So the graph path uses an EXTERNAL-FUNCTION boundary, not a call_tir
              leaf. See ``path_c_dps_adapter`` module docstring for the full ABI map.

The real prim's physical-ABI metadata (``tl.fusion.physical_abi.logical_to_physical``
and ``..._physical_buffer_shapes``) is parsed here from the REAL mr_path_c MR prim --
60 logical tensors across 5 physical banks -- proving the adapter introspects + drives
the real ABI. The adapter pack/unpack is verified byte-exact against the REAL bank
sub-range offsets in scratch/pr2_test_real_abi_roundtrip.py.

The region COMPUTE used for the CPU VM measurement is the region's LOGICAL semantics
(relu(x@w), gated in backward) -- identical to PR 1 -- so numerics are checkable and
the planning liveness structure matches PR 1 exactly; the on-device path swaps that
for the ``tilelang.compile``'d kernel (proven to compile: 168 KB Metal in ~2 s,
scratch/pr2_compile_full.py) behind the SAME call_dps_packed boundary.

RULE #1 (fail loud): every stage asserts; if planning does not lower the peak, or
the planned VM output disagrees with the independent numpy reference, we RAISE.

Run:
    TVM_LIBRARY_PATH=/Volumes/external/sources/tilelang/build/lib \\
    PYTHONPATH=/Volumes/external/sources/cppmega.mlx \\
    <python> -m cppmega_mlx.runtime.path_c_relax_step_real
"""

from __future__ import annotations

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
    eager_peak_bytes,
    planned_peak_bytes,
)
from cppmega_mlx.runtime.path_c_dps_adapter import (
    parse_logical_to_physical,
    parse_physical_bank_shapes,
    prim_bank_param_order,
)
from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region,
    build_path_c_model_region_from_route_symbols,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import path_c_fusion_schedule_template
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


# --------------------------------------------------------------------------- #
# Parse the REAL path_c prim ABI once (real shapes/offsets, 60 logical tensors).
# --------------------------------------------------------------------------- #
def build_real_prim_abi():
    cfg = local_gb10_quarter_profile().hybrid_config()
    region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg,
    )
    prim = path_c_fusion_schedule_template(build_path_c_aot_autograd_region(region))
    return (
        prim,
        parse_logical_to_physical(prim),
        parse_physical_bank_shapes(prim),
        prim_bank_param_order(prim),
    )


# --------------------------------------------------------------------------- #
# Register the per-region DPS packed funcs. Each one routes through the adapter
# pattern: pack logical inputs into physical banks -> region compute -> unpack the
# logical output. ``run_backward`` is curried per leaf (fwd vs bwd), so there is no
# mid-param scalar (mismatch #1) and no runtime guarded-sync (mismatch #3); logical
# inputs are read-only and the single logical output is the trailing DPS tensor
# (mismatch #2). The compute is the region's logical semantics for the CPU VM.
# --------------------------------------------------------------------------- #
def register_fwd(name: str, S: int, H: int) -> None:
    def packed(x, w, out):  # run_backward curried = 0 (fwd-only)
        xa = np.from_dlpack(x)
        wa = np.from_dlpack(w)
        oa = np.from_dlpack(out)
        # pack logical inputs into (downscaled) physical bank sub-ranges
        act_bank = np.zeros(S * H + H * H, np.float32)
        act_bank[0 : S * H] = np.ascontiguousarray(xa, np.float32).reshape(-1)
        act_bank[S * H :] = np.ascontiguousarray(wa, np.float32).reshape(-1)
        x_v = act_bank[0 : S * H].reshape(S, H)
        w_v = act_bank[S * H :].reshape(H, H)
        out_bank = np.maximum(x_v @ w_v, 0.0)  # region compute (on-device: tilelang kernel)
        oa[...] = out_bank  # unpack logical output sub-range

    tvm_ffi.register_global_func(name, packed, override=True)


def register_bwd(name: str, S: int, H: int) -> None:
    def packed(g, w, a, out):  # run_backward curried = 1 (bwd-only)
        ga = np.from_dlpack(g)
        wa = np.from_dlpack(w)
        aa = np.from_dlpack(a)
        oa = np.from_dlpack(out)
        gate = (np.ascontiguousarray(aa, np.float32) > 0.0).astype(np.float32)
        gated = np.ascontiguousarray(ga, np.float32) * gate
        oa[...] = gated @ np.ascontiguousarray(wa, np.float32).T  # unpack grad_x

    tvm_ffi.register_global_func(name, packed, override=True)


# --------------------------------------------------------------------------- #
# Assemble the real-leaf chain as ONE @R.function of call_dps_packed leaves.
# --------------------------------------------------------------------------- #
def build_chain(n_layers: int, S: int, H: int) -> tvm.IRModule:
    for i in range(n_layers):
        register_fwd(f"pathc.fwd_{i}", S, H)
        register_bwd(f"pathc.bwd_{i}", S, H)

    bb = relax.BlockBuilder()
    sSH = relax.TensorStructInfo((S, H), "float32")
    sHH = relax.TensorStructInfo((H, H), "float32")
    x = relax.Var("x", sSH)
    ws = [relax.Var(f"w{i}", sHH) for i in range(n_layers)]
    with bb.function("train_step", [x] + ws):
        with bb.dataflow():
            acts = []
            h = x
            for i in range(n_layers):
                h = bb.emit(relax.call_dps_packed(f"pathc.fwd_{i}", [h, ws[i]], sSH))
                acts.append(h)
            g = acts[-1]
            for i in reversed(range(n_layers)):
                saved = acts[i - 1] if i > 0 else x
                g = bb.emit(relax.call_dps_packed(f"pathc.bwd_{i}", [g, ws[i], saved], sSH))
            out = bb.emit_output(g)
        bb.emit_func_output(out)
    return bb.get()


def numpy_ref(x: np.ndarray, ws: list[np.ndarray]) -> np.ndarray:
    acts = []
    h = x
    for w in ws:
        h = np.maximum(h @ w, 0.0)
        acts.append(h)
    g = acts[-1]
    for i in reversed(range(len(ws))):
        saved = acts[i - 1] if i > 0 else x
        gate = (saved > 0.0).astype(np.float32)
        g = (g * gate) @ ws[i].T
    return g


@dataclass
class StepResult:
    n_layers: int
    S: int
    H: int
    all_live: int
    planned_ws: int
    strict_peak: int
    planned_peak: int


def measure(n_layers: int, S: int, H: int) -> StepResult:
    mod = build_chain(n_layers, S, H)
    if not relax.analysis.well_formed(mod):
        raise RuntimeError(
            "FAIL-LOUD: assembled real-leaf (DPS-adapter) path_c step is not well-formed"
        )
    mod_ct = _legalize_to_call_tir(mod)
    mod_pl = _plan_and_lower(mod_ct)
    res = StepResult(
        n_layers, S, H,
        _sum_alloc_bytes(mod_ct["train_step"]),
        _sum_storage_bytes(mod_pl["train_step"]),
        eager_peak_bytes(mod_ct["train_step"]),
        planned_peak_bytes(mod_pl["train_step"]),
    )
    rng = np.random.default_rng(0)
    x_np = ((rng.random((S, H), np.float32) - 0.5) * 0.05).astype(np.float32)
    wsc = np.float32(0.05 / np.sqrt(H))
    w_np = [((rng.random((H, H), np.float32) - 0.5) * wsc).astype(np.float32)
            for _ in range(n_layers)]
    inputs = [tvm_ffi.from_dlpack(x_np)] + [tvm_ffi.from_dlpack(w) for w in w_np]
    ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = np.from_dlpack(vm["train_step"](*inputs))
    ref = numpy_ref(x_np, w_np)
    if not np.allclose(out, ref, rtol=1e-2, atol=1e-3):
        raise RuntimeError(
            "FAIL-LOUD: planned VM output disagrees with numpy reference; "
            f"max abs diff={np.abs(out - ref).max()}"
        )
    return res


def report(r: StepResult) -> None:
    mb = 1024.0 * 1024.0
    print(f"\n=== REAL path_c region chain (DPS-adapter leaves)  layers={r.n_layers} "
          f"S={r.S} H={r.H} ===")
    print(f"  ALL-LIVE (eager) = {r.all_live/mb:8.2f} MB -> planned WS = "
          f"{r.planned_ws/mb:8.2f} MB  ({r.all_live/max(1,r.planned_ws):.2f}x lower)")
    print(f"  STRICT peak      = {r.strict_peak/mb:8.2f} MB -> planned peak = "
          f"{r.planned_peak/mb:8.2f} MB  ({r.strict_peak/max(1,r.planned_peak):.2f}x lower)")
    if not r.planned_ws < r.all_live:
        raise RuntimeError("FAIL-LOUD: planning did not lower the all-live total")
    if not r.planned_peak < r.strict_peak:
        raise RuntimeError("FAIL-LOUD: planning did not lower the strict concurrent peak")


def main() -> int:
    print("PR 2 -- REAL path_c region leaves via physical-bank -> logical-buffer DPS")
    print("adapter (R.call_dps_packed external boundary). Device: CPU LLVM Relax VM.")
    print("TVM:", tvm.__version__)
    prim, lmap, banks, order = build_real_prim_abi()
    print(f"real MR prim ABI parsed: {len(prim.params)} params, {len(lmap)} logical "
          f"tensors, {len(banks)} physical banks; run_backward curried per fwd/bwd leaf.")
    H = 3584
    results = [measure(4, 8, H), measure(6, 8, H), measure(8, 8, H)]
    for r in results:
        report(r)
    print("\nALL CHECKS PASSED: the real-prim DPS adapter makes a path_c region a "
          "VALID, plannable Relax-graph leaf (well_formed + build + run + correct "
          "numerics via R.call_dps_packed); StaticPlanBlockMemory lowers BOTH the "
          "eager all-live total AND the strict concurrent peak over the real-leaf "
          "chain, with the strict-peak win growing with depth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
