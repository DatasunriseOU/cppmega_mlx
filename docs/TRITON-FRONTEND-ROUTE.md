# Tri-Dao mamba_ssm backward Triton kernels routed through OUR tilelang stack

Status: 3 of 7 backward kernels route end-to-end through
`tilelang triton frontend -> tilelang lower -> tvm CUDA codegen -> sm_121`
to runnable tensor-core (HMMA / TF32) machine code on GB10. The other 4
break at precisely-named, deeper boundaries (no silent fallback; RULE #1).

Measured on GB10 (sm_121a), venv `/home/dave/cppmega-venv`, tilelang from
source `/home/dave/source/tilelang`. Date 2026-06-06.

## The route (verbatim path)

```
@triton.jit kernel (mamba_ssm.ops.triton.ssd_chunk_scan / ssd_chunk_state)
  -> Triton compile -> TTIR (custom tt.* dialect text)
  -> poc.triton_frontend.triton_native_parse.parse_ttir_via_triton
       (Triton's OWN libtriton.ir validates + canonically re-prints TTIR;
        we parse that text into mlir.ir-shaped adapter ops)
  -> poc.triton_frontend.from_ttir  (TTIR -> tvm.tir.PrimFunc)
  -> tilelang.engine.lower / tilelang.compile -> tvm CUDA codegen
  -> nvcc/ptxas -> sm_121a cubin (HMMA.1688.F32.TF32 SASS)
```

Entry points used:
- `tilelang.frontends.triton.compile_ttir(ttir, name=..., target="cuda")`
  -> `JITKernel` (the production wrapper; dispatches to `from_ttir`).
- `poc.triton_frontend.from_ttir(...)` -> `PrimFunc` (lower-only path).

## Which kernels route (3/7)

| kernel | stage reached | CUDA src | tensor-core |
|---|---|---|---|
| `_chunk_scan_bwd_dstates` | runnable JITKernel | 33586 B | 32x HMMA.1688.F32.TF32 |
| `_chunk_scan_bwd_dc`      | runnable JITKernel | 34128 B | mma_sync |
| `_chunk_scan_bwd_dcb`     | runnable JITKernel | 47676 B | mma_sync |

### Proof it is OUR stack (tvm), not a native-triton fallback

`compile_ttir(_chunk_scan_bwd_dstates)` -> JITKernel -> `export_ptx` / `export_sass`:

- PTX header: `.version 9.3`, **`.target sm_121a`** (GB10 native).
- PTX: **32 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`** + 8 `ldmatrix`.
- SASS (ptxas): **32 `HMMA.1688.F32.TF32`** + 8 `LDSM.16.M88.4`.
- Generated `.cu` includes `tl_templates/cuda/instruction/mma.h`,
  `tl_templates/cuda/gemm.h` (TileLang/TVM template headers); 0 triton tokens.

That chain (tvm CUDA codegen -> ptxas HMMA for sm_121) cannot be produced by
the native Triton runtime; it is our tilelang->tvm path.

## Which kernels do NOT route yet (4/7) and exactly where

| kernel | break stage | precise cause |
|---|---|---|
| `_chunk_scan_bwd_dx` | tilelang_lower | GEMM `m_warp * n_warp == num_warps` ICHECK (CUDA op/gemm.cc) fails for the reduce+atomic-surrounded dot |
| `_chunk_state_bwd_ddAcs_stable` | tilelang_lower | same `m_warp * n_warp == num_warps` ICHECK |
| `_chunk_state_bwd_db` | tilelang_lower | `variables (arg4,) are used, but are not passed in as API arguments` -- a scalar-load of `dA_cumsum_ptr` re-declares a duplicate `arg4` buffer (shape `(1,)`) that aliases the param buffer instead of reusing it |
| `_chunk_state_bwd_dx` | from_ttir | `Can't cast a handle to other types` during emission (pointer/handle used in a scalar arithmetic context) |

All four FAIL LOUD at the named boundary -- no degraded/empty kernel is emitted.

## OUR-stack fixes made (this round)

All in `/Volumes/external/sources/tilelang/poc/triton_frontend/triton_native_parse.py`:

1. **Generic-form inline region parsing for `tt.reduce` / `tt.scan` combiners.**
   Triton prints the combiner region generically:
   `%0 = "tt.reduce"(%in) <{axis = 1 : i32}> ({ ^bb0(%a,%b): arith.addf ... tt.reduce.return ... }) : (...) -> R`.
   The text parser previously RAISED at the closing `}) : (...) -> ...` line
   ("hit the close of a GENERIC-FORM inline region"). Added:
   - `opens_generic_region` detection (header ends with `({`) distinct from the
     scf bare-`{` region opener.
   - `_build_generic_region`: consumes `^bbN(args):` block args + inner ops into
     `op.regions[].blocks[].operations[]` (the exact structure
     `op_emitters.reduction._detect_via_mlir` walks to find the `arith.*`
     combiner), and parses the closing `}) : (...) -> R` line for the result type
     (which is NOT on the header), rewriting the result Value's type via the new
     `_Value.set_type`.
   - `_parse_props_attrs` for the `<{axis = ...}>` properties block (surfaces
     `axis` to the emitter).
   - `_OPNAME_RE` broadened to match multi-dotted op names (`tt.reduce.return`).

2. **`tt.atomic_rmw` custom-form RMW op keyword.** TTIR prints
   `tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %val, %mask : ...`; the leading
   `fadd` keyword is now mapped to the `rmw_op` attribute the
   `map_tt_atomic_rmw` emitter requires (added `tt.atomic_rmw -> rmw_op` to
   `_POSITIONAL_KEYWORD_ATTRS`).

Effect: the 4 non-routing kernels moved PAST the TTIR-parse barrier (reduce
region + atomic_rmw now parse) and now break deeper in tilelang lowering /
emission -- a strictly more advanced boundary than before.

Prior-round fixes (commit 5cfb5bc2 / gb10 b0004c30) still in force:
NEW `triton_native_parse.py` provider; `from_ttir` RAISES instead of emitting a
`T.evaluate(0)` stub; `map_tt_dot` allocates the CUDA MMA accumulator C in
`local.fragment`; `_emit_region` propagates `ctx.target`.

## Measured ms (GB10)

- Native Tri-Dao `mamba_chunk_scan_combined` full fwd+bwd (b1 s2048 h8 d64 n64
  chunk256): **3.78 ms** on GB10 (the "10ms-class" reference; faster here for
  this size on GB10). vs path_c v1 905 ms.
- Routed kernels: COMPILE to runnable sm_121 tensor-core JITKernels
  (verified PTX+SASS). A *numerically-correct timed run* of the routed kernels
  is BLOCKED on (a) the PtrAnalysis C++ shim being unbuilt on GB10 -- the
  frontend uses the MVP scalar-load path, which does not reproduce Tri-Dao's
  strided tile addressing -- and (b) the full Tri-Dao strided launch harness.
  A degenerate (zero-stride) launch core-dumps (illegal address), confirming
  the addressing gap. We report this honestly rather than fabricate a number.

## To finish the remaining 4 + get parity/ms

- Build the PtrAnalysis C++ shim on GB10 (`_triton_frontend_cxx`) in a fresh
  process (libtriton-LLVM double-registration aborts otherwise) so tile loads
  use the real multi-element pointer analysis -> enables correct numeric parity.
- `_chunk_state_bwd_db`: fix the scalar-pointer-load buffer aliasing in
  `op_emitters/memory.py` (`_redeclare_ctx_buffer_1d` must rewrite ALL prior
  references, including ones already emitted into statements, or the scalar load
  must reuse `ctx.buffers[key]`).
- `_chunk_scan_bwd_dx` / `_chunk_state_bwd_ddAcs_stable`: the GEMM warp-partition
  ICHECK in `src/backend/cuda/op/gemm.cc` (`isSquare` yields `best_m=best_n=1`
  when no `m*n==num_warps` partition keeps `m_per_warp,n_per_warp >= 1`).
  Needs a Python-side GEMM policy/num_warps choice that factors the tile, or a
  C++ fallback in the partitioner (a tilelang rebuild).
- `_chunk_state_bwd_dx`: trace the handle-cast in emission (a `!tt.ptr` used in a
  scalar arith op needs an address-of/load instead of a value cast).
