"""Concretely TEST whether a REAL path_c PrimFunc can be a R.call_tir leaf.

We attach the real path_c PrimFunc to a BlockBuilder IRModule and emit a
@R.function that issues R.call_tir(prim, [inputs], out_sinfo). We then run the
standard legalize->plan pipeline and relax.build. Whatever fails (well-formed
check, CallTIRRewrite arity, build) is captured verbatim -- that is the exact
ABI mismatch the doc asks us to report.

DPS contract (from CallTIRRewrite): call_tir appends the allocated OUTPUT
buffers as the LAST args to the prim. So prim.params must be
  [input-buffer params ...] ++ [output-buffer params ...]
with outputs last and matching out_sinfo, and the prim must return void and
WRITE only the trailing output params. We test the real prim against that.

VALIDATED RESULT (2026-06-02, this script): NO -- path_c's physical-ABI prim does
NOT fit R.call_tir DPS, for three concrete reasons captured below verbatim:
  1. PARAM ORDER: scalar `path_c_run_backward: T.int32` is at param index 8 (the
     MIDDLE), not last -> well_formed() FALSE:
       "Argument 5 type mismatch: expected R.Prim('int32'),
        given R.Tensor((60708456,), dtype='float32')".
  2. NO TRAILING OUTPUT BUFFER: the prim packs logical tensors into shared physical
     dtype banks and reads+writes them IN PLACE (every `*_abi_bank` in BOTH
     T.reads and T.writes); CallTIRRewrite only "succeeds" by contriving the last
     route buffer (RNN h_next) as a fake output -- not real DPS.
  3. NOT A GENERIC-TIR KERNEL: relax.build RAISES inside s_tir ThreadSync,
       "Cannot insert syncs inside condition" (thread_storage_sync.cc:145),
     because the TileLang T.Kernel body guards shared-mem accesses inside an
     `if (path_c_run_backward)` -- it is meant for tilelang.compile, not relax/s_tir.
Conclusion: PR 1 (cppmega_mlx/runtime/path_c_relax_step.py) uses DPS-clean
logical-buffer leaves; PR 2 writes the physical->logical DPS adapter shim.
See docs/RELAX-GRAPH-MEMORY-PATH.md section 3.
"""
from __future__ import annotations

import sys
import traceback

import tvm
from tvm import relax, tir
from tvm.script import relax as R

from cppmega_mlx.runtime.path_c_fusion import (
    build_path_c_aot_autograd_region,
    build_path_c_model_region_from_route_symbols,
)
from cppmega_mlx.runtime.path_c_fusion_schedules import (
    path_c_fusion_schedule_template,
)
from cppmega_mlx.recipes.model_factory import local_gb10_quarter_profile


def _strip_to_tir(prim):
    """Return a plain tir.PrimFunc (TileLang prims may need lowering first)."""
    # The schedule_template already returns a tir.PrimFunc-like object.
    return prim


def attempt_call_tir_wrap(prim, *, label: str) -> None:
    print(f"\n{'='*70}\nATTEMPT R.call_tir wrap: {label}\n{'='*70}")
    params = list(prim.params)
    buffer_map = prim.buffer_map
    # Partition params into buffer params and scalar params.
    buf_params = [p for p in params if p in buffer_map]
    scalar_params = [p for p in params if p not in buffer_map]
    print(f"buffer params: {len(buf_params)}   scalar params: {len(scalar_params)} "
          f"-> {[str(getattr(p,'name',p)) for p in scalar_params]}")

    # Build the input/output StructInfos from the buffer_map (treat ALL buffers as
    # inputs and the LAST buffer as the nominal output, which is the optimistic
    # DPS interpretation -- we want to see whether the build accepts it).
    def sinfo_of(p):
        b = buffer_map[p]
        shape = [int(d) for d in b.shape]
        return relax.TensorStructInfo(shape, str(b.dtype))

    in_sinfos = [sinfo_of(p) for p in buf_params[:-1]]
    out_sinfo = sinfo_of(buf_params[-1])

    bb = relax.BlockBuilder()
    gv = bb.add_func(prim, "path_c_region")
    # Relax inputs: one per input buffer param.
    relax_ins = [relax.Var(f"in{i}", s) for i, s in enumerate(in_sinfos)]
    try:
        with bb.function("train_step", relax_ins):
            with bb.dataflow():
                # tir_vars for the scalar params (run_backward = 1).
                tir_vars = [tir.const(1, "int32") for _ in scalar_params] or None
                out = bb.emit(
                    relax.call_tir(gv, relax.Tuple(relax_ins), out_sinfo,
                                   tir_vars=tir_vars)
                )
                out = bb.emit_output(out)
            bb.emit_func_output(out)
        mod = bb.get()
        print("BlockBuilder assembled R.call_tir OK (structural).")
    except Exception as exc:  # noqa: BLE001  -- we WANT the verbatim failure
        print("FAILED at BlockBuilder/call_tir construction:")
        traceback.print_exc()
        return

    # Try the well-formed check + legalize + plan + build.
    from tvm.relax.transform import (
        CallTIRRewrite, LegalizeOps, ToNonDataflow, RemovePurityChecking,
        StaticPlanBlockMemory, LowerAllocTensor, KillAfterLastUse,
    )
    try:
        ok = relax.analysis.well_formed(mod)
        print("well_formed:", ok)
    except Exception:
        print("well_formed check raised:")
        traceback.print_exc()

    try:
        mod_ct = tvm.transform.Sequential(
            [LegalizeOps(), ToNonDataflow(), RemovePurityChecking(), CallTIRRewrite()]
        )(mod)
        print("CallTIRRewrite OK -- inspecting how outputs were appended:")
        # Show the rewritten call to the prim.
        print(mod_ct["train_step"])
    except Exception:
        print("FAILED at legalize/CallTIRRewrite:")
        traceback.print_exc()
        return

    try:
        ex = tvm.compile(mod, target=tvm.target.Target("llvm"))
        print("relax.build (llvm) OK")
    except Exception:
        print("FAILED at relax.build:")
        traceback.print_exc()


def main() -> int:
    cfg = local_gb10_quarter_profile().hybrid_config()
    fwd_region = build_path_c_model_region_from_route_symbols(
        region_name="mr_path_c", route_symbols=("M", "R"), model_config=cfg,
    )
    autograd_region = build_path_c_aot_autograd_region(fwd_region)
    prim = path_c_fusion_schedule_template(autograd_region)
    print("prim type:", type(prim).__name__,
          "  is tir.PrimFunc:", isinstance(prim, tir.PrimFunc))
    attempt_call_tir_wrap(prim, label="MR joint fwd+bwd real path_c prim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
