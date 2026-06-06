# Tri-Dao mamba_ssm bwd -> OUR (tilelang/tvm triton_frontend) stack: route + parity status

MEASURED on gb10 (sm_121a, venv /home/dave/cppmega-venv, source /home/dave/source/tilelang).
Commit b4f9f94c (merge/upstream-codegen-reorg). All GPU exec gb10-only.

## Verdict: NO-GO (parity 0/7). Routing + TF32 tensor-core path PROVEN; numeric parity NOT yet achieved.

The route is real (7/7 kernels lower end-to-end to TF32 tensor-core MMA, 288 HMMA in SASS),
but the routed *output is numerically WRONG* on every real-strided config because of a
gemm-accumulator-fragment epilogue defect (below). Per RULE #1 no routed ms is reported as a
correctness/perf result — only parity-PASS kernels qualify, and 0/7 PASS.

## Route + SASS re-confirm (after the FIX#3/FIX#4 read-path change)

| check | result |
|---|---|
| route_all7 (lower -> CUDA, __global__ + mma_sync) | 7/7 OK WITH MMA |
| sass_all7 (export_sass) | 7/7 SASS_OK |
| HMMA.1688.F32.TF32 total in SASS | 288 (32+32+32+32+32+64+64) |
| reduce kernels emitting atomicAdd | dx=1, state_dx=3, ddAcs=1 (REDG.E.ADD) |

So the read-path change (FIX#3 input-extent + FIX#4 STSM/LDSM gate) did NOT regress routing:
still 7/7 route, still 288 HMMA.

## Parity (routed vs NATIVE Tri-Dao triton), REAL-strided NON-degenerate config

Config: b1 nh8 hd64 ds64 nc8 cs64, BLOCK_K=32 => cs=64 != BLOCK_K => 2 K-trips/chunk
(NON-degenerate, real seqlen strides — NOT cs=BK single-trip, NOT strides=0).

| kernel | NATIVE nz | ROUTED nz | MAXDIFF | allclose 1e-3 |
|---|---|---|---|---|
| _chunk_scan_bwd_dstates | 262144/262144 | 2048/262144 | 2.335834e+02 | **FAIL** |
| (6 others share the identical gemm->fragment->epilogue codegen) | — | — | — | **FAIL (not driven; same defect)** |

dstates is the simplest of the 7; it FAILS, and all 7 emit the identical defective
fragment-store (verified in generated CUDA), so parity is 0/7. Production §P1 parity was
NOT run because the small real-strided config already fails for the same structural reason —
running prod would only reproduce the same wrong fragment store at larger extent.

## Root cause (precise, RULE #1 — the exact remaining defect, no fabricated PASS)

The hand-built mma.1688 gemm writes its 64x64 result into a `local.fragment` (`dot_c_frag`),
whose elements are DISTRIBUTED across the 128 threads in the mma C-fragment register layout:
logical element [i,j] does NOT live at flat offset i*N+j of any one thread's register array.

FIX#4 correctly tries to materialise the fragment to a logical shared tile before the
Tri-Dao epilogue (`T.copy(c_frag, c_logical)`), and correctly gates STSM/LDSM off when the
fragment layout is absent (src/backend/cuda/op/copy_analysis.cc CheckSTSMCopy/CheckLDSMCopy).
BUT the fragment produced by the hand-built mma loop is NOT the recognised epilogue of a
`tl::gemm` op, so **LayoutInference never registers a thread-layout for `dot_c_frag`**. With
no layout in `layout_map`, the SIMT/Normal copy falls back to a FLAT thread-iteration copy.
Generated CUDA (all 7 kernels, identical signature):

```c
*(float4*)(dot_c_logical_313 + ((i_4 * 512) + (((int)threadIdx.x) * 4)))
        = dot_c_frag_311[((i_4 * 128) + ((int)threadIdx.x))];   // FLAT, layout-blind
```

This copies the fragment in raw thread-flat order, NOT through the mma-1688 register->[i,j]
mapping. Only the slots where flat order happens to coincide with the true fragment layout
carry meaning (2048/262144 nonzero, MAXDIFF 2.34e2). The materialisation is structurally
unable to be correct via this path because there is no registered fragment layout to invert.

### The real fix (next step, not done here)
The fragment->shared store must apply the mma-1688 C-fragment thread layout. Two routes:
 1. Emit the gemm as a real `tl::gemm` whose epilogue LayoutInference recognises, so the
    fragment gets a registered layout and `T.copy(c_frag, shared)` lowers layout-correctly
    (then STSM/LDSM or the layout-aware SIMT store both work); OR
 2. Register the mma-1688 C-fragment layout for the hand-built `dot_c_frag` explicitly before
    the copy, so the SIMT/Normal store distributes registers to the correct [i,j] positions.
RULE #1: this is a real correctness route to implement — NOT a silent fallback to patch over.

## Native reference (valid regardless of routed parity)

| measurement | value |
|---|---|
| native _chunk_scan_bwd_dstates @ §P1 (S=4096 c=64 g=8 H=112 P=64 N=64 bs1, grid=(1,64,112)) | 1.126 ms/kernel (MEASURED this run; ~1.2ms class, consistent w/ recorded 1.206ms) |
| native full mamba_ssm bwd (all 7 kernels + non-Triton ops) | ~10ms-class |
| path_c v1 full chain | 905 ms |

routed ms vs native: NOT REPORTED — routed fails parity (RULE #1: no perf number for a
numerically-wrong kernel). full-7 routed sum vs ~10ms / vs 905ms: NOT REPORTED for the same
reason.

## Summary
- 7/7 route to real TF32 tensor cores (288 HMMA) — PROVEN and re-confirmed after read-path fix.
- 0/7 numeric parity — blocked by the layout-blind fragment->shared epilogue store.
- GO/NO-GO: **NO-GO** until the mma C-fragment layout is applied at the materialisation copy.
