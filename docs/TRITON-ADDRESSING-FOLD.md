# Triton Addressing-Fold (FULL transform-1, Coalesce-style)

Kill the spilled `[64]`/`[2048]`/`[4096]` index/mask arrays in the routed-Triton
prologue by folding addressing/mask tiles into the strided `T.copy` load body
instead of materializing them in any scope.

- tilelang HEAD: `67ab579a` (local `merge/upstream-codegen-reorg`) == gb10 `f5640a3a`
  (six fold files byte-identical between the two: `mlir_walker.py`, `op_mapping.py`,
  `__init__.py`, `op_emitters/{arith,memory,control}.py`).
- Gate: `from_ttir(..., prologue_opt=True)` (default ON for the routed path;
  `prologue_opt=False` reproduces the pre-fold serial prologue for A/B).
- Hardware: NVIDIA GB10, `sm_121a`, aarch64-linux.
- All numbers below are REGENERATED FRESH FROM HEAD: both fold-OFF and fold-ON
  PrimFunc JSONs are re-emitted from the `_chunk_scan_bwd_dstates` TTIR via
  `from_ttir(prologue_opt=False|True)`, then compiled and run interleaved.
  Fold-OFF JSON = 497670 bytes, fold-ON JSON = 270624 bytes (the materialized
  index/mask arrays disappear from the serialized PrimFunc).

## What folds

`build_addressing_fold_set` (`poc/triton_frontend/mlir_walker.py:582`) reconstructs
the MLIR use-def graph by **Value identity** (the native parser leaves
`value.uses=None` and prints the same value differently as operand vs result, so a
name-keyed map silently fails to connect a load's mask operand back to its
`arith.cmpi`). A producer result is fold-eligible by fixpoint iff EVERY use is
either a load/store addressing/mask SINK or an addressing PRODUCER whose own
results are eligible. `scf.for`/`scf.yield` are transparent for non-accumulator
(pointer/int) loop-carried slots; the f32 accumulator slot and any `tt.dot` /
store-VALUE / reduce / return use disqualify (RULE #1: never fold a data tile).

Eligible `tile_binop` / `bcast` / `make_range` / `expand_dims` / bool-mask tiles
are kept as `LazyTileExpr` and consumed per-lane inside the copy loop body
(`_resolve_lane_operand` / `_scalarize_tile_index_base` / `_read_lane` /
strided-store rhs already read a `LazyTileExpr` per-lane). The arrays are never
materialized -> no local spill, no shared overflow, nothing to thread-distribute.
The cooperative GEMM half (`dot_a_*`/`ptx_ldmatrix`/`mma_sync<TF32,...>`) is untouched.

## Correctness (§P1 + small real-strided multi-K-trip)

Controlled A/B, native Triton `_chunk_scan_bwd_dstates_kernel` as reference.

### §P1 production config (b1 nh112 hd64 ds64 nc64 cs64, grid (1,64,112), numel 29,360,128, multi-K-trip)

| build | MAXDIFF vs native | ALLCLOSE 1e-3 |
|-------|-------------------|---------------|
| fold OFF | 4.882812e-04 | PASS |
| fold ON  | 4.882812e-04 | PASS |
| OFF vs ON | **0.000000e+00 (byte-identical)** | — |

### Small real-strided multi-K-trip (b1 nh4 hd64 ds64 nc4 cs64, 2 K-trips)

Real non-default strides via padded-then-sliced tensors:
`dout.stride=[73728,288,72,1]`, `C.stride=[36864,144,72,1]` (note 72 != hd=64).

| build | MAXDIFF vs native | ALLCLOSE 1e-3 |
|-------|-------------------|---------------|
| fold ON | 3.051758e-05 | PASS |

## Performance (CUDA events, N=50/round, 6 interleaved OFF/OPT rounds)

| build | median ms/kernel |
|-------|------------------|
| fold OFF (pre-fold serial prologue) | **1467.97** |
| fold ON (addressing-fold)           | **1103.26** |
| native Triton (reference)           | **1.11844** |

- Fold speedup: **1.331x**, **-364.71 ms** (per-round spread <2 ms; very stable).
- Remaining gap to native: OPT/native = **~986x** (was ~1312x at fold-OFF).
- The drop is real but the kernel is still ~1000x off native: the residual cost
  is the un-coalesced loads (LDGSTS=0) + the tensor-core under-utilization
  (HMMA 32 vs Triton 256), NOT the prologue arrays anymore.

## SASS (cuobjdump `sm_121a`, regenerated from HEAD)

| metric | fold OFF | fold ON | Triton (ref, prior MEASURED) |
|--------|----------|---------|------------------------------|
| local spill STL+LDL | 658 (412+246) | **306 (168+138)** | 0 |
| coalesced LDGSTS    | 0 | 0 | 75 |
| HMMA                | 32 | 32 | 256 |
| total instr (this counter) | 17528 | 8932 | 1981 |

- **Local spill halved: 658 -> 306 (-53%).** The fold removes the index/mask
  arrays that were spilling; the residual 306 is the GEMM accumulator/fragment
  traffic, not addressing arrays.
- HMMA unchanged (32) OFF==ON: the tensor-core GEMM path is byte-intact through
  the fold.

## Other-paths no-regression (routed family, codegen A/B OFF vs ON)

| kernel | OFF | ON | note |
|--------|-----|----|------|
| `_chunk_scan_bwd_dstates` | OK len37174 mma2 | OK len22883 mma2 | folded (shorter, GEMM intact) |
| `_chunk_scan_bwd_dc` | OK len38271 mma2 | OK len23943 mma2 | folded |
| `_chunk_state_bwd_db` | OK len42823 mma2 | OK len27468 mma2 | folded |
| `_chunk_state_bwd_dx` | OK len54284 mma2 | OK len39413 mma2 | folded |
| `_chunk_state_bwd_ddAcs_stable` | OK len42922 mma2 | OK len31133 mma2 | folded |
| `_chunk_scan_bwd_dcb` | FAIL (PtrState unresolved SSA) | FAIL (same) | **pre-existing at HEAD, identical OFF==ON** |
| `_chunk_scan_bwd_dx` | FAIL (undefined.size()==0) | FAIL (same) | **pre-existing at HEAD, identical OFF==ON** |

5/7 codegen OK in both gates; `mma=2` preserved in every OK kernel (GEMM
cooperative half intact); the 2 failures are pre-existing and identical in both
gates (NOT introduced by the fold). Native dstates baseline reproduced at
1.11844 ms.

## Honest GO / NO-GO

**GO (partial, as designed).** The fold landed and is correct + reproducible from
HEAD: byte-identical output OFF==ON, real 1.331x / -365 ms speedup, local spill
658 -> 306. What did NOT land in this transform (explicitly out of scope here and
flagged for follow-up):

- **LDGSTS still 0.** Coalesced `cp.async` global->shared is a separate rewrite
  (the loads still go through per-lane strided `T.copy`, not a coalesced LDGSTS).
  This is the bulk of the remaining ~986x gap.
- **HMMA still 32 vs Triton 256.** Tensor-core under-utilization is independent of
  the prologue fold.
- **Spill 306 not 0.** The residual is GEMM accumulator/fragment traffic, not the
  eliminated addressing arrays.

RULE #1 honored: the fold either applies or the kernel keeps the intact GEMM
cooperative path; no silent serial fallback was introduced; the 2 routed failures
are pre-existing and surfaced (not papered over).
