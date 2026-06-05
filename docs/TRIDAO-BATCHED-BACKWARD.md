# §TB1 — Tri-Dao Batched Large-Tile B2 Backward (CUDA sm_121 + Metal simdgroup)

Status: **MEASURED, honest mixed verdict.** Metal = GO (in-budget). CUDA = NO-GO
at prod (smem launch ceiling blocks the head-amortization that the recipe needs).

## The recipe (P1 / Tri-Dao lever)

path_c B2 backward ran its 4 dominant contractions (DYX, dC_off, dC_diag, and
dchunk_states) either single-thread scalar (the prod Metal prim) or threaded (the
v1 CUDA prim), or as a per-64-tile `T.gemm` (the §27 twin, 0.749x — staging+sync
exceeded the threaded serial at M=64). The Tri-Dao recipe batches the (chunk,head)
dimension so each tensor-core GEMM amortizes the ldmatrix/operand-staging/`sync`
fixed cost over a tall M (HEADS_PER_CTA heads' rows), plus bs>1 CTA parallelism to
hide per-GEMM latency.

Two NEW flag-gated prims (the ONE path when on, RAISE on failure — RULE #1; the
serial/§27 prims stay byte-identical when off):
- `chunk_scan_combine_bwd_cuda_prim_gemm_batched` (sm_121 mma.sync, fragment-C)
- `chunk_scan_combine_bwd_metal_gemm_prim_batched` (simdgroup, C-in-frag; batches
  DYX + dchunk_states only — Apple 32KB pool)

Flags: `CPPMEGA_PATH_C_B2_GEMM_BATCHED` / `CPPMEGA_PATH_C_METAL_GEMM_BATCHED`;
`CPPMEGA_PATH_C_B2_HEADS_PER_CTA` / `CPPMEGA_PATH_C_METAL_HEADS_PER_CTA`.

## §TB1.0 The offset-0 GEMM-staging constraint (load-bearing implementation fact)

The original batched design sliced a tall `(HPC*L, *)` shared band into per-head
sub-GEMMs (`T.gemm(opA[hh*L : hh*L+L, ...], ...)`). **This tilelang's `T.gemm`
ASSERTS the A operand's first-dim offset == 0** (`gemm_op.py:104`), so head hh>0
fails — caught at trace (CUDA) and MSL-compile (Metal). FIX: `dY16`/`opA` are now
OFFSET-0 head-sized staging tiles (`(L,headdim)` / `(maxLP,maxLP)`) REUSED across
the serial head loop; per-head `dY`/`DYX` results live in `(HPC,L,*)` bands. The
amortization that SURVIVES = one staging alloc + band-level syncs shared over the
head loop — NOT a literal tall-M MMA. (A real tall-M dense MMA would compute
HPC² cross-head garbage; block-diagonal per-head GEMMs are correct but each MMA is
still M=L=64.)

## §TB1.1 CUDA (gb10 sm_121) — MEASURED NO-GO

Lean A/B (`scratch/probe_b2_batched_cuda_ab_only.py`), prod cfg b=1 S=4096 c=64
G=8 H=112 P=64 N=64, NVIDIA GB10, fp16 operands / fp32 fragment:

| prim                       | HPC | ms (incl. output-zero) | vs v1   | verdict |
|----------------------------|-----|------------------------|---------|---------|
| v1 threaded                | —   | 882.6                  | 1.000x  | base    |
| §27 single-tile GEMM       | —   | 1215.5                 | 0.726x  | NO-GO   |
| batched-large-tile         | 1   | 1234.3                 | 0.715x  | NO-GO   |
| batched-large-tile         | 2+  | **fails to launch**    | —       | NO-GO   |

(ms includes the per-iter zeroing of 7 outputs incl. the 7.5GB dinp; the RELATIVE
ratios are valid — same overhead for all three.)

**WHY NO-GO (the honest root cause):**
1. **HPC>=2 cannot launch.** The batched prim requests **66560 B dynamic shared
   memory** at HPC=2; the runtime `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE)`
   FAILS (`tvm.error.InternalError: Failed to set the allowed dynamic shared
   memory size to 66560`). GB10 `shared_per_block`=49152, `_optin`=101376; the
   kernel's static+dynamic total exceeds what the TVM launch path enables. The
   per-head fp32 `dY` (HPC·64·64·4) + `DYX` (HPC·64·64·4) bands — which the serial
   dinp/dseg tail must read for ALL heads — are what blow the budget. They cannot
   be fp16 (the dinp/dseg accumulation needs fp32).
2. **HPC=1 is the only one that fits, and it is 0.715x** — i.e. ESSENTIALLY the §27
   single-tile twin (0.726x). With one head per CTA there is NO head band to
   amortize over, so the ldmatrix/staging/sync overhead is paid per single 64-tile
   GEMM exactly as in §27 — the defect the recipe was meant to cure.

So the lever (HEADS_PER_CTA>=2 amortization) is **blocked by the smem launch
ceiling** in this tilelang/TVM build; the only launchable config degenerates to
the already-rejected §27 twin. bs4 (4x CTA supply) does NOT rescue this: it
multiplies CTA supply for v1 and batched equally, so the per-GEMM amortization
deficit (the actual bottleneck) is unchanged. **The batched-GEMM B2 does NOT beat
the threaded v1 on CUDA at any launchable config.**

Self-consistency (batched HPC=1 vs v1, all 7 GEMM-able outputs): worst **1.98e-5**
(dC=1.98e-5, dchunk=1.10e-5, dinp=0, dx=dz=0, dA_y=6.78e-6, dD=8.34e-7) — the
batched math is CORRECT; the problem is purely performance/occupancy.

## §TB1.2 Metal (local Apple, simdgroup) — MEASURED GO (in-budget)

`scratch/probe_b2_batched_metal_local.py`, MPS, fp16 operands / fp32 accum. The
prod L=P=N=64 batched Metal prim is a **SMEM-NO-GO** (49664 B > Apple's HARD 32768
B threadgroup pool at HPC=1; the dispatcher gate RAISES). At the IN-BUDGET
L=P=N=32 config (the largest that fits Apple's 32KB) it BUILDS+RUNS:

| HPC | serial ms | batched ms | speedup | parity worst | verdict |
|-----|-----------|------------|---------|--------------|---------|
| 1   | 10.82     | 2.17       | **4.99x** | 6.03e-6 PASS | **GO** |
| 2   | 10.82     | 4.35       | 2.49x   | dchunk 6.62e-3 | GO (perf) / parity FAIL |

- **HPC=1: 4.99x GO, full parity PASS** (worst 6.03e-6 over all 7 grads). The
  simdgroup DYX + dchunk_states GEMMs cleanly beat the scalar serial prod.
- **HPC=2 has a cross-head bug**: only `dchunk` breaks (6.62e-3), the other 6
  grads stay ~1e-6. The offset-0 staging-tile reuse across heads needs a stronger
  inter-head sync in the dchunk phase; HPC=1 is correct and is the shipping config.

Metal verdict: the batched-GEMM recipe WORKS and is a strong GO at in-budget dims
(HPC=1, 4.99x vs scalar); prod L=P=N=64 is the honest SMEM-NO-GO (Apple 32KB).

## §TB1.3 z3 / TLA proof (non-vacuous)

`scratch/proof_b2_batched_driver.py` (run on gb10, z3 live):
- **3 positives PROVED** (z3_used=z3_proved=True): `dchunk` (transpose_A +
  decay-fold scale), `dcoff` (dense), `dcdiag` (lower-tri mask) — every obligation
  (operand_maps_match, mask_equiv, scale_equiv, single_writer, k_covered) unsat.
- **2 negatives FAIL correctly (non-vacuity)**: overlapping head-bands
  (m_stride<tile_m) → single_writer=False via counter-witness; transpose-bugged A
  map (i·K+k vs the real k·P+i) → operand_maps_match=False via counter-witness.

Verdict `ALL_POSITIVES_PROVED_AND_NON_VACUOUS`. The rewrite is z3-correct and
race-free; the build dispatcher gates emission behind `require_gemm_rewrite_proof`
(RAISES on sat/unknown/disabled — fail-closed, RULE #1).

## §TB1.4 Bottom line / attribution

- The batched-GEMM rewrite is z3-PROVEN correct and MEASURED numerically correct
  (CUDA 1.98e-5, Metal 6.03e-6 at HPC=1) on both backends.
- **CUDA: MEASURED NO-GO.** The HEADS_PER_CTA>=2 amortization that is the actual
  lever cannot launch (66560 B dynamic smem ceiling in this TVM build); the only
  launchable config (HPC=1) degenerates to the §27 twin and is 0.715x vs the
  threaded v1. The backward chain (447.8ms) and step tok/s are NOT improved by
  this path on CUDA — the v1 threaded B2 remains the best CUDA B2.
- **Metal: MEASURED GO in-budget** (4.99x vs scalar at HPC=1, full parity), but a
  **prod SMEM-NO-GO** (Apple 32KB pool); usable only at <=32-dim chunks.
- Honest: the batched recipe is necessary-but-not-sufficient on this toolchain. To
  make CUDA GO one must (a) get TVM to opt into >48KB dynamic smem (the GB10 optin
  is 101376 B, so HPC=2's 66560 B SHOULD be settable — a TVM launch-path gap), or
  (b) drop the resident fp32 dY/DYX bands (recompute in the dinp/dseg tail) to fit
  HPC>=2 under 48KB. Neither is done here; the measured verdict stands.

## §TB1.5 RE-VERIFICATION (2026-06-05, this session, gb10-only run)

Independent re-run on NVIDIA GB10 (CUDA 13.2, tilelang 0.1.9+cuda.git8385a23d).
The §TB1.1–1.3 verdicts REPRODUCE. NO local Apple GPU was dispatched this session
(SoC watchdog-panic safety) — Metal is CODEGEN-VERIFIED only, numeric Metal timing
+ parity are DEFERRED (the §TB1.2 4.99x is from a PRIOR local run, not re-measured).

- **CUDA HPC=2 launch failure REPRODUCED VERBATIM**: `tvm.error.InternalError:
  Failed to set the allowed dynamic shared memory size to 66560` (the lever cannot
  launch). v1 + §27 built fine; only batched HPC=2 dies at launch.
- **CUDA HPC=1 timing REPRODUCED** (`probe_b2_batched_cuda_ab_only.py`, prod cfg):
  v1_threaded=905.5ms, §27_single_tile=1231.6ms, batched=1250.1ms →
  batched_vs_v1=**0.724x (NO-GO)**, batched_vs_§27=**0.985x** (degenerates to §27).
  batched-vs-v1 self-consistency worst **1.98e-5** over 7 GEMM-able outputs (math
  correct; performance NO-GO).
- **CUDA real-tensor-core SASS PROVEN**: lowered the batched prim to CUDA, compiled
  the cubin (`nvcc -arch=sm_121a`), `cuobjdump -sass` shows **128 `HMMA.16816.F32`**
  (m16n8k16 fp16→fp32 tensor-core) + **64 `LDSM`** (ldmatrix). C++ emits
  `tl::mma_sync<...,16,8,16,...>` x8 (4 contractions × 2 n-tiles). HMMA.16816 = m16
  per step, 4 m-steps per per-head 64-tile — confirms §TB1.0: per-head M=64 tiles,
  NOT a tall HPC·64 dense MMA. Real tensor-core, but no head-amortization survives.
- **z3 proof REPRODUCED**: 3 positives PROVED (z3_used=z3_proved=True, all
  obligations unsat), 2 negatives FAIL correctly (overlap_band→single_writer=False
  counter-witness; transpose_bug→operand_maps_match=False counter-witness).
  VERDICT `ALL_POSITIVES_PROVED_AND_NON_VACUOUS` — non-vacuous.
- **Metal CODEGEN-VERIFIED (lowered on gb10, ZERO Apple GPU exec)**: lowered the
  batched Metal prim to MSL at the in-budget L=P=N=32 / HPC=1 config; the MSL emits
  **`simdgroup_multiply_accumulate`** (the DYX + dchunk GEMMs), **`simdgroup_load`/
  `simdgroup_store`**, and **`simdgroup_matrix<float, 8, 8>`** (Apple 8×8 fragment).
  Genuine large-tile simdgroup tensor-op codegen confirmed. Numeric Metal timing
  + 7-grad parity are DEFERRED (not run locally this session, watchdog safety).
