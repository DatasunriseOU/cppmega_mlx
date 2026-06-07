# TRITON-ROUTE-FAST — MEASURE phase (gb10 sm_121, mutex held)

Status: **NO-GO** on the goal (correct + native-fast Tri-Dao bwd through our stack).
All numbers below are MEASURED live on gb10 (aarch64-linux, sm_121) at HEAD
`eade89be` (StructuredGemm: cooperative gemm operand staging + deterministic C
layout, BUG1/BUG2). RULE#1 honored: routed ms reported, parity is fabrication-free,
the exact remaining defect is named.

## Headline

| metric | MEASURED | target |
|---|---|---|
| parity PASS (prod §P1) | **0/7** | 7/7 @ 1e-3 |
| parity PASS (small, real 2-K-trip) | **0/7** | 7/7 @ 1e-3 |
| routed ms/kernel (dstates, prod §P1) | **2547.6 ms** | ~1.16 ms |
| native ms/kernel (dstates, prod §P1) | **1.14228 ms** | (baseline) |
| routed vs native | **~2230x SLOWER** | ~1x |
| 7/7 route + MMA in CUDA | **YES** (mma/ldmatrix present all 7) | yes |

The serial-staging bottleneck the StructuredGemm fix targeted is NOT the live
bottleneck in this environment: routed ms is ~2547 ms (vs the prior 2419/2535 ms),
i.e. essentially UNCHANGED. The dominant cost is a SEPARATE upstream defect (below).

## Root cause (verified at source, not inferred)

`poc.triton_frontend.ptr_analysis.shim_available() == False` on gb10. The C++
`_triton_frontend_cxx` PtrAnalysis shim is UNBUILT for aarch64-linux (only macOS
`-darwin.so` artifacts exist; final link of vendored `libTriton*.a` macOS archives
fails on aarch64). The frontend emits a RuntimeWarning and falls back to the MVP
scalar path, which lowers multi-element tile operand LOADS as per-element
masked-GATHER (`carry_index_*`) sequences. That gather PERMUTES the `[hd,BK]`
operand → structural permutation (MAXDIFF 2.6–2.8e2 small / 1.0–1.17e3 prod — a
permutation, not a transpose/scale/precision miss), AND the per-element gather +
serial epilogue are the ~2547 ms cost.

The BUG1 cooperative `T.copy` operand staging (eade89be) IS present in the routed
CUDA, but the operand still ARRIVES permuted from the scalar gather upstream of the
staging copy, so parity stays broken and the gather cost dominates. The real fix is
to BUILD the C++ shim on gb10 so multi-element tile loads use the vendored
triton-shared PtrAnalysis (no gather) — a separate, pre-existing infra defect.

## Per-kernel parity (small cfg: b1 nh8 hd64 ds64 nc4 cs64 ng2 s256, REAL 2-K-trip, cs64≠BK32, strides≠0)

| kernel | status | MAXDIFF | note |
|---|---|---|---|
| `_chunk_scan_bwd_dstates` | FAIL | 2.78e2 | correctly marshalled (== proven prod harness map); full tile written, values permuted |
| `_chunk_scan_bwd_dc` | SEGV | — | generic positional marshaller OOB: walker reorders params (carry_index gather buffers FIRST), positional packing → wrong buffers |
| `_chunk_scan_bwd_dcb` | SEGV | — | same marshaller defect |
| `_chunk_scan_bwd_dx` | SEGV | — | same marshaller defect |
| `_chunk_state_bwd_db` | SEGV | — | same marshaller defect |
| `_chunk_state_bwd_dx` | SEGV | — | same marshaller defect |
| `_chunk_state_bwd_ddAcs_stable` | N/A | — | atomic ddA accumulator output (no directly-comparable buffer) |

prod §P1 (S4096 c64 g8 H112 P64 N64): `_chunk_scan_bwd_dstates` FAIL MAXDIFF=1.165e3,
full tile written (routed nz 29360128/29360128 == native nz), values permuted.

The 6 non-dstates kernels share the same underlying defect; the SEGV is a harness
marshalling limitation (param reorder), not a new fault — each needs per-kernel
hand-mapping like dstates' prod harness, after which they'd hit the SAME gather
permutation.

## ms table (parity-PASS kernels only — there are NONE; dstates ms shown for the record)

| path | ms/kernel @ prod §P1 | source |
|---|---|---|
| native triton `_chunk_scan_bwd_dstates` | 1.14228 | MEASURED gb10 |
| routed (our stack) `_chunk_scan_bwd_dstates` | 2547.6 | MEASURED gb10 (parity FAIL — informational only) |
| path_c v1 (full chain) | 905 (prior) | EXTRAPOLATION / prior run |

RULE#1: routed ms is informational here because NO kernel passes parity. There is no
honest "routed-our-stack fast" number to report as a result.

## NO PERF REGRESSION on other frontend GEMM paths (user CRITICAL ask)

The StructuredGemm change (eade89be) touches exactly 3 python files
(`op_emitters/reduction.py`, `frame_register.py`, `__init__.py`), inside
`map_tt_dot`'s operand-staging closure.

- Live CUDA-ms before/after for `fla_dot_exp2` / `matmul` / `dot_reduce_atomic` /
  `dot_reduce_atomic_trans_b` is **NOT obtainable on gb10**: the full TTIR→PrimFunc
  lowering for ANY `tt.dot` kernel is BLOCKED here because `_lower_ttir` needs the
  same UNBUILT C++ shim (`round_trip_through_cxx_shim`; in-process MLIR has no `tt`
  dialect → "Dialect `tt' not found"). The block is IDENTICAL before and after.
- Measurable surface = frontend unit suite. BEFORE/AFTER by file-swap (the 3 files
  reverted to `88106531` vs HEAD `eade89be`), identical command:
  `pytest test_dot_reduce_atomic test_op_emitters_reduction test_op_mapping
  test_op_emitters_memory test_reducer_corpus test_builder_context`:

  | build | result |
  |---|---|
  | AFTER (eade89be) | **121 passed, 10 skipped, 3 failed** |
  | BEFORE (88106531) | **121 passed, 10 skipped, 3 failed** |

  IDENTICAL — same 3 failures both ways (pre-existing: 2× ranked-offset store bounds,
  1× tma-fallback-when-triton-available). ZERO correctness regression from the change.
- 7/7 Tri-Dao bwd kernels still compile to CUDA with MMA/ldmatrix tensor-core
  instructions (route intact).

Honest verdict: NO correctness regression on other frontend paths (unit suite
byte-identical pass/fail before/after). Live CUDA-ms before/after for the named GEMM
kernels could not be measured because their lowering is shim-gated in this env
(blocked identically pre/post change) — reported as a measurement gap, not a PASS.

## GO/NO-GO

**NO-GO.** Routed Tri-Dao bwd is neither correct (0/7 parity) nor fast (2547 ms vs
1.14 ms native). The blocker is the UNBUILT aarch64 C++ PtrAnalysis shim forcing the
scalar masked-gather path; the StructuredGemm operand-staging fix is in place and
correct in isolation but is downstream of the permuting gather, so it cannot rescue
parity or perf until the shim is built on gb10.
