# TTIR canonicalize/fold — §P1 dstates MEASURE (gb10, sm_121, triton 3.7.0)

Status: **FoldTTIR DONE+VERIFIED. Fold ROUTING = NO-GO (honest, RULE #1).**

## What was proven

The captured TTIR for `_chunk_scan_bwd_dstates_kernel` was snapshotted too early
in Triton's pipeline (`ASTSource.make_ir` returns PRE-optimization TTIR), so it
carried the i32->i64 overflow-guard chain that native Triton folds in `make_ttir`
via the MLIR canonicalizer (int-range analysis proves the guards always-true for
the kernel's i32 index ranges).

`_fold_ttir()` (poc/triton_frontend/_test_harness/jit_to_ttir.py, commit
0a01caac) replicates Triton's nvidia `make_ttir` pass list on the captured module
before serialization. EXECUTED on gb10:

| TTIR                | lines | extsi | cmpi | 2147483647 | andi |
|---------------------|------:|------:|-----:|-----------:|-----:|
| UNFOLD (pre-fold)   |   895 |   102 |  109 |         66 |   54 |
| **FOLD (from HEAD)**|   367 | **0** |**7** |      **0** |**3** |
| native Triton cache |   299*|     0 |    7 |          0 |    3 |

The folded TTIR matches native Triton's TTIR counts exactly (7 remaining cmpi are
legit `slt` bounds masks; zero overflow guards). The fold is a real semantic-
preserving MLIR-pass equivalence, NOT a dropped check.

(*native cache .ttir is normalized/shorter; the guard COUNTS match, which is the
input-matching invariant the diff proves.)

## MEASURE result — routing the folded TTIR

Routed both TTIRs through the frontend with `TL_FORCE_CP_ASYNC=1` (cp.async live),
PtrAnalysis C++ shim ON (PYTHONPATH=poc/triton_frontend/_cxx/build), executed
cubins, CUDA-event timing N=60/rep x4 reps median-of-medians, parity vs native.

| Routed kernel | for-loops (.cu) | LDGSTS | HMMA | spill STL+LDL | EXEC ms | parity MAXDIFF |
|---------------|----------------:|-------:|-----:|--------------:|--------:|----------------|
| UNFOLD        |              84 |     16 |   32 |       **272** | **733.29** | **4.882812e-04 PASS** |
| FOLD (raw)    |              49 |     16 |   16 |           192 |  638.51 | **1.515449e+03 FAIL** |
| native        |               — |      — |    — |             — |   **1.16–1.20** | reference |

- The fold DID reduce the routed kernel (for-loops 84->49, spills 272->192,
  HMMA 32->16, cu_len 22564->16104) and was ~13% faster wall-clock (733->638 ms).
- **But the folded routing is NUMERICALLY WRONG (MAXDIFF 1.5e3, ~half nonzero).**
  Per RULE #1 a faster-but-wrong path is forbidden; it is now made to FAIL LOUD.

## Root cause of the fold-routing failure (verifiable)

Triton's canonicalizer FUSES `acc = acc + dot(...)` so the GEMM accumulates
DIRECTLY into the loop-carried `acc` tile (`tensor<64x64xf32>`). The unfolded TTIR
keeps the GEMM writing a FRESH fragment (`dot_c_logical`) and does the accumulation
as a SEPARATE add into a linearly-copied shared carry — which our frontend lowers
correctly.

For the fused (folded) form, the loop-carried accumulator surfaces as a
`LazyTileExpr` and is allocated as a SHARED `carry_tile` (control.py). On CUDA the
tensor-core MMA store layout (`make_mma_store_layout`) requires C in a swizzled
`local.fragment`. Two real frontend bugs were exposed and the third is structural:

1. **lb-name collision (FIXED — control.py `_bound_name`).** The loop-carry
   `lb` was stored as a positional SSA name (`%0`). The canonicalizer/CSE renames
   the K-loop lower bound to a named constant (`%c0_i32`) and reassigns `%0` to an
   unrelated `tt.splat` tile, so `lb` resolved to a float32 tile and aborted with
   `Cast(... ffi.OpaquePyObject)`. Fix: bind lb/step by their CONSTANT VALUE
   (numbering-independent), identical semantics. Also fixes the standalone matmul
   fold-route which hit the same bug.

2. **shared-as-fragment over-accept (FIXED — reduction.py).**
   `_is_fragment_scope` lumped `shared` in with fragment scopes (correct for Metal),
   so a SHARED carry passed straight to the CUDA MMA as C and aborted at
   LayoutInference (`carry_tile must be a fragment, but got shared`). Added
   `_is_strict_fragment_scope` and gated the CUDA-fragment copy on it.

3. **swizzle-vs-linear carry round-trip (STRUCTURAL — not safely fixable here).**
   With (1)+(2) the fold COMPILES, but the loop-carry snapshot/commit
   (`_append_loop_carry_copies` -> `_copy_buffer_stmt`) and the C-seed copy use
   SERIAL SCALAR element copies that index the tile LINEARLY. The MMA fragment is
   SWIZZLED. So each K-trip reads the carry in mma-swizzle layout but the carry is
   written/stored linearly -> the accumulator is silently corrupted
   (MAXDIFF ~1.5e3). Correctly lowering the fused form needs a layout-aware
   `T.copy` (or a swizzle-correct fragment-resident carry across the K-loop),
   which the current frontend does not emit.

## Decision (RULE #1)

- Keep fixes (1) and (2): real correctness improvements, no regression to the
  proven UNFOLD path (still 4.882812e-04, 733 ms) or the unit suite (the only
  reduction/control test failures are PRE-EXISTING `inspect.getsource` OSErrors,
  identical with edits reverted).
- The fused-GEMM-accumulate-into-shared-carry case now RAISES a clear `EmitError`
  instead of silently emitting 1.5e3-garbage. **Fold routing = NO-GO** until a
  layout-aware loop-carried-fragment lowering lands.

## Remaining gap to native

UNFOLD routed §P1 = 733 ms vs native 1.16–1.20 ms (~611x). The overflow-guard
prologue is NOT the dominant cost on the CURRENT HEAD (the cp.async work already
removed the 64-serial-thread prologue; routed UNFOLD has threadIdx_for=0, 272
spills). The fold's TTIR is correct and matches native, but our frontend cannot
yet exploit the fused accumulate form, so it does not (yet) move §P1 toward native.
The next lever is the layout-aware fragment-resident loop-carry (issue #3 above),
not further TTIR folding.

## Reproduce (gb10, GPU mutex)

```
ssh gb10
source /home/dave/cppmega-venv/bin/activate
cd /home/dave/source/tilelang && git checkout <HEAD>
export PYTHONPATH=/home/dave/source/tilelang/poc/triton_frontend/_cxx/build:$PYTHONPATH
# 1) fold the captured TTIR (BK=32, matching the snapshot):
python /tmp/cap_folded_bk32.py        # -> extsi=0 cmpi=7 2147483647=0 andi=3 lines=367
# 2) build/measure UNFOLD (PASS) and FOLD (now fails loud):
python /tmp/em_fold_measure.py unfold # PARITY 4.882812e-04, 733 ms
python /tmp/em_fold_one.py fold        # EmitError: CUDA MMA accumulator C is a SHARED loop-carried tile ... (RULE #1)
```
