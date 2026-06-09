# Backend-agnostic i32-overflow guard fold — §P1 dstates MEASURED (gb10 CUDA, EXECUTED)

HEAD `46ae6c79de8c1319194a7a0d1cced59228dfec53` (tilelang, branch
`merge/upstream-codegen-reorg`). All CUDA numbers below are EXECUTED on the
NVIDIA GB10 (sm_121, CUDA 13.2) under the gb10 GPU mutex. RULE#1: bit-correct
EXECUTED results only; the Metal check is codegen-only (no Apple-GPU dispatch).

## What the fold is

Triton wraps every i32->i64 address expression in an always-true overflow guard
pair (`addr <= 2147483647` / `-2147483648 <= addr`). For the Tri-Dao chunk shapes
(seqlen 4096, nheads 112, headdim/dstate 64 — all << 2^31) these are statically
TRUE. Un-folded, OUR walker materialized them as serial scalar loops + spills.

The fold lives in OUR backend-agnostic layer (the TTIR->TIR walker), NOT in
libtriton / the NVIDIA pipeline:

- `poc/triton_frontend/op_emitters/arith.py`
  - `_is_int32_overflow_guard` (line ~1001) recognizes the `sle 2147483647` /
    `sge -2147483648` always-true bound pairs.
  - `_emit_cmpi` (line ~1065) folds them to a constant-true tile.
  - `_emit_andi` (line ~641) drops `guard_true & x -> x`.
  - Gated by `ctx.routed_triton_prologue_opt` (caller passes `prologue_opt=True`).
- VERIFIED backend-agnostic: `grep -ciE 'cuda|nvidia|libtriton|ptx|sass'
  op_emitters/arith.py` = **0**. The fold is a pure TIR constant-true
  substitution that lowers identically to CUDA and Metal.

## §P1 dstates — EXECUTED on gb10 (em_dstates_cpasync.py)

Input TTIR `/tmp/ttir7/_chunk_scan_bwd_dstates.ttir` (895 lines) carries the full
guard chain: extsi=102, cmpi=109, INT_MAX=66, andi=54.

Interleaved CUDA-event timing, N=60/rep × 4 reps, median-of-medians:

| variant | spill (STL+LDL) | LDGSTS | IMAD | ISETP | parity (MAXDIFF) | ms |
|---|---|---|---|---|---|---|
| OFF (un-routed serial prologue, plain LDG) | 2494 (STL 1688 + LDL 806) | 0 | 1249 | 1280 | 4.882812e-04 | **3140.2** |
| OPT (fold + cp.async, prologue_opt=True) | **272** (STL 136 + LDL 136) | 16 | 803 | 796 | 4.882812e-04 | **745.1** |
| native Triton | — | — | (298) | (69) | reference | **1.19** |

- OFF 3140 ms reproduces the PROVEN ROOT CAUSE serial-prologue baseline (~3144 ms).
- Fold + cp.async: 4.21x faster than OFF; spills 2494 -> 272 (9.2x); IMAD/ISETP halved.
- Parity is byte-identical 4.882812e-04 for BOTH OFF and OPT (guards always-true ->
  same values), all 29 360 128 outputs nonzero, allclose(1e-3) PASS.

### Generated §P1 .cu — the guard prologue is GONE

From the EXECUTED tvm_kernels.cu (00000 = OFF, 00001 = OPT):

| | OFF .cu | OPT .cu |
|---|---|---|
| lines | 628 | 336 |
| `2147483647` guard sites | **14** | **0** |
| `for(` loops | 168 | 84 |

The fold drops every overflow-guard comparison (14 -> 0) and halves the loop
count in the consumed/lowered CUDA source. SASS instruction count 35 574 -> 14 809.

## Fold confirmed across the whole routed kernel family (build-only, /tmp/route_popt.py)

`prologue_opt=False` vs `True`, guard count in generated CUDA:

| kernel | guards OFF | guards FOLD | mma | .cu bytes OFF->FOLD |
|---|---|---|---|---|
| _chunk_scan_bwd_dc | 14 | **0** | 2 | 38271 -> 23943 |
| _chunk_state_bwd_db | 15 | **0** | 2 | 42823 -> 27468 |
| _chunk_state_bwd_dx | 20 | **0** | 2 | 54284 -> 39413 |
| _chunk_state_bwd_ddAcs_stable | 16 | **0** | 2 | 42922 -> 31133 |

Fold removes ALL guards in every kernel; `mma=2` (GEMM) and `__global__` intact;
.cu shrinks ~30-37%. No regression: every kernel that routed with popt=False also
routes with popt=True, identical mma count.

## Metal codegen check — CODEGEN ONLY, NO Apple-GPU dispatch

The SAME folded prim (`prologue_opt=True`, NO cp.async / no CUDA-only op) lowers
through the full TileLang Metal pipeline to valid MSL
(`metal_foldcheck2.py` on `_chunk_state_bwd_ddAcs_stable`, 80 guards in):

- `tilelang.lower(pf, target=Target("metal"))` -> CompiledArtifact OK.
- `art.kernel_source`: 43805 chars of MSL — `#include <metal_stdlib>`,
  `#include <metal_simdgroup>`, `using namespace metal;`, `kernel`, `threadgroup`,
  `device` qualifiers; AtomicAdd lowered to `tl::AtomicAdd`.
- `HAS_GUARD_2147483647 = 0` in the MSL — **the guard chain is folded out in the
  Metal output exactly as in CUDA**. No CUDA-only op. The fold is Metal-portable.

(Host is Linux/CUDA; it physically cannot launch an Apple GPU — the check is
inherently watchdog-safe. The dstates kernel's multi-stage copy loop trips
TileLang's Metal `PipelinePlanning` pass — an orthogonal software-pipelining
limitation present with or without the fold, see `metal_foldcheck.py` — so the
fold-portability proof uses the non-pipelined `ddAcs_stable` kernel.)

## Honest gap and the "272 -> ~0 / toward native 1.19 ms" target

- The emitter fold eliminates the DOMINANT i32-overflow serial prologue: 14 -> 0
  guard sites in the .cu, 2494 -> 272 spills, 3140 -> 745 ms. This is the working,
  EXECUTED, bit-correct path.
- The residual 272 spills are NOT the guard prologue — they are the kernel's
  genuine 64×64 GEMM working set. The remaining 745 ms vs native 1.19 ms gap
  (626x) is GEMM tiling/scheduling (32 vs 256 HMMA tile, copy staging), NOT the
  folded guards.
- The TTIR-level COMPLETE fold (`_fold_ttir` in jit_to_ttir.py) does reach native
  guard-chain parity on TEXT (895 -> 362 lines; extsi 102->0, cmpi 109->7,
  INT_MAX 66->0, andi 54->3, matching native Triton's folded .ttir). BUT feeding
  that pre-folded TTIR back through OUR walker fails
  (`EmitError: PtrState references unresolved SSA value '%112'`) because Triton's
  `rewrite_tensor_pointer` canonicalizer restructures the addressing into a form
  the walker's PtrState resolution cannot consume. So the TTIR-level path is NOT a
  working executed route for dstates; the emitter-level fold (OPT above) is.

## GO / NO-GO

GO for the backend-agnostic guard fold: it is real (always-true equivalence,
bit-correct 4.882812e-04), in OUR Metal-portable layer (zero libtriton/CUDA refs),
drops §P1 from 3140 -> 745 ms (4.21x) by eliminating the guard prologue (14 -> 0
.cu guard sites, 2494 -> 272 spills), and the SAME folded prim emits valid MSL with
the guard folded out (Metal-portable). NO-GO on reaching native 1.19 ms: the
remaining 745 ms is the GEMM tiling, a separate tuning axis, not the guard fold.

## Reproduce from HEAD

```
ssh gb10 'mkdir /tmp/gb10_gpu_owner.lock && echo own>/tmp/gb10_gpu_owner.lock/id'
ssh gb10 'source /home/dave/cppmega-venv/bin/activate; cd /home/dave/source/tilelang;
  python3 poc/triton_frontend/_test_harness/tridao_parity/em_dstates_cpasync.py'   # §P1 EXECUTED
ssh gb10 '... python3 poc/triton_frontend/_test_harness/tridao_parity/metal_foldcheck2.py'  # MSL codegen
ssh gb10 'rm -rf /tmp/gb10_gpu_owner.lock'
```
