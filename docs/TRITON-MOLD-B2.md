# Triton-Mold path_c B2 Backward (§DYN static→dynamic) — MEASURED VERDICT

Status: **DECIDED — HONEST NO-GO (0.745x vs in-process v1).** The static→dynamic
"Triton mold" achieves launch-feasibility (driver `STATIC=0`, HPC=2 launches with
the full 4-GEMM layout) but does **not** beat the v1 threaded B2 prim.

Hardware: NVIDIA GB10 (sm_121, unified 121 GB). Prod cfg: `b=1, S=4096, chunk=64,
G=8, H=112, P=64, N=64`, `HEADS_PER_CTA=2` (the only valid HPC for `nheads/ngroups
= 14`; see §2). Toolchain: CUDA 13.3, tilelang from source, torch 2.13 dev cu132.
Date: 2026-06-06.

---

## 1. What was built (the Triton mold)

NEW flag-gated CUDA prim `chunk_scan_combine_bwd_cuda_prim_gemm_batched_dyn`
(`cppmega_mlx/nn/_tilelang/mamba3_chunked_backward_core.py:2699`). It is the
§TB1/§SO1 batched math/grid/in-kernel-head-loop **byte-for-byte**, with the five
GEMM operand-staging tiles (`dY16`, `opA`, `opB`, `store_fp32`, `dCdiag_sh`) moved
from explicit **STATIC** `scope="shared"` to the **DYNAMIC** region (the
`T.alloc_shared` default `shared.dyn`). tilelang lowers `shared.dyn` into
`extern __shared__ __align__(1024) uchar buf_dyn_shmem[]`, so the compiler reserves
~0 static smem and the driver grants the full per-block dynamic opt-in. This is the
Tri-Dao Triton bwd HW mold (same gb10: STATIC=0, MAXDYN opt-in).

Wired behind `CPPMEGA_PATH_C_B2_GEMM_BATCHED_DYN`, mutually exclusive with the other
B2 flags (RAISES on ambiguity). Own dynamic-smem gate against the 101376 B opt-in
cap (RAISES over-cap). z3/TLA proof gate (RAISES `GemmRewriteNotProven` on
sat/unknown). RULE #1: ONE path when flagged, no fallback. **Default path (no flag)
is unchanged v1** (`else: prim = chunk_scan_combine_bwd_cuda_prim`).

---

## 2. cuFuncGetAttribute — STATIC≈0 PROVEN at the driver level

The cubin (nvcc `-arch=sm_121a` on the exact emitted `device_kernel.cu`) loaded via
the CUDA driver API (`cuda.bindings.driver`):

```
cuFuncGetAttribute(main_kernel):
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES            = 0       <-- STATIC = 0 (mold achieved)
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES (default) = 49152
  CU_FUNC_ATTRIBUTE_NUM_REGS                     = 192
cuFuncSetAttribute(MAX_DYNAMIC = 91136)          = GRANTED -> now 91136   (HPC=2 LAUNCHES)
device MAX_SHARED_MEMORY_PER_BLOCK_OPTIN         = 101376
```

cuobjdump cross-check on the same cubin:
```
Function main_kernel:  REG:192  SHARED:1024  LOCAL:0  STACK:0   (smem=1024)
.nv.shared.main_kernel  size 0x400 (=1024 B)   <-- just the __align__(1024) slot for
                                                    the extern dyn array, NOT staging
```

**STATIC dropped from 57344 B (the §40a26d44 per-head GEMM staging) to 0.** The only
"static" smem is the 1024 B alignment placeholder for the dynamic `buf_dyn_shmem[]`.
The 91136 B dynamic request (HPC=2) fits the 101376 B sm_121 opt-in cap → the driver
grants it → **HPC=2 LAUNCHES** with the full 4-GEMM layout.

`HEADS_PER_CTA=4` is **INVALID** at prod: `nheads/ngroups = 112/8 = 14`, and 4 neither
divides nor is a multiple of 14, so a CTA head-band would straddle a group whose
`C`/`B` operands are per-GROUP. `_b2_batched_heads_per_cta` RAISES (RULE #1: no silent
straddle). The valid HPC set is {1, 2, 7, 14}; HPC=2 is the measured point.

---

## 3. TIMING — MEASURED NO-GO

In ONE gb10 process (`probe_chunked_backward_cuda_gb10.py --prod --b2-gemm-ab`,
`CPPMEGA_PATH_C_B2_HEADS_PER_CTA=2`):

| prim                         | ms (median) | vs in-process v1 |
| ---------------------------- | ----------: | ---------------: |
| v1 threaded (`prim` default) |    1052.074 |           1.000x |
| single-tile GEMM (§27)       |    1370.047 |           0.768x |
| batched-large-tile (§TB1)    |    1418.405 |           0.742x |
| **§DYN batched_dyn (mold)**  |    **1412.696** |       **0.745x  NO-GO** |

§DYN vs batched_static = 1.004x (the static→dynamic scope flip is ~0.4% faster — the
smem move is free but does not change the dominant cost). **0.745x < 1.0x → NO-GO.**

NOTE on baselines: this probe's in-process v1 measures **1052 ms**, not the historical
905.4 ms reference (an earlier wf machine/thermal state). All §DYN ratios above are vs
the **same-process 1052 ms v1**, which is the apples-to-apples comparison. Against the
historical references: §DYN 1412.7 ms is 0.64x of 905.4 ms (v1_threaded), 0.87x of
1231 ms (§27), and 141x of 10 ms (Tri-Dao Triton). Tri-Dao remains ~140x faster.

Backward chain (B2 + B1 + B0), this run:  **B2 1051.1 + B1 5.55 + B0 266.2 ms**.
(The 447.8 ms reference is the v1 B2 from an earlier wf; B2 alone is 1051 ms here.)

---

## 4. SASS — real HMMA, but PER-HEAD relabel (NOT fattened-M head-accum)

`cuobjdump --dump-sass` on the §DYN cubin:
```
128  HMMA.16816.F32     (m16n8k16 tensor-core MMA, F32 accumulate)
  0  OMMA / IMMA / BMMA
```

These are real tensor-core HMMA tiles (`HMMA.16816.F32 R84, R96, R124, RZ ; ...`).
But the M-dimension of every GEMM is the **per-head** tile (L=64 or P=64), issued
inside `for hh in T.serial(HPC)` loops — the four contractions (DYX = dY@xᵀ, dC_off =
dY@prev_states, dC_diag masked, dchunk = (decay·dY)ᵀ@C) each run once **per head** with
operands that differ on BOTH sides. The 128 HMMA = 4 GEMMs × HPC=2 heads × the 16-row
fragmentation of the 64-tiles. **This is a per-64 (per-head) relabel, NOT a fattened-M
head-accumulation.** Tri-Dao fattens M by looping `nheads_per_program` heads that SHARE
a per-GROUP operand (B/C) into ONE accumulator tile; path_c cannot, because dC and dinp
are **per-HEAD** (distinct operands both sides) — a dense tall-M over heads would compute
HPC² cross-head garbage (block-diagonal violation). The mold therefore amortizes only
operand-staging/sync, never the tensor-core M-setup.

---

## 5. PARITY — all 8 grads PASS

§DYN vs v1 (in-process, max|abs| over ALL elements):
```
dC 6.21e-05  dx 0.00e+00  dz 0.00e+00  dchunk 9.30e-05  dinp 6.41e-05
dA_y 3.15e-05  dD 4.41e-06
```
Chained 8-grad gate (B2→B1→B0, GATE 1e-3, ALL elements, default v1 chain):
```
dz 1.73e-04  dx 8.10e-04  dC 5.03e-05  dB 1.09e-05  dlog_decay 6.67e-04
ddt 1.50e-04  dh0 1.84e-04  dD 2.57e-05   -> WORST 8.099e-04 < 1e-03   PASS
```
dD uses the fp16-cache; the §DYN deltas are fp16-rounding (≤9.3e-5), well within gate.

---

## 6. z3 PROOF — non-vacuous

`proof_b2_batched_driver.py`, `dyn_scope_flip`:
- POSITIVES dchunk / dcoff / dcdiag: `z3_proved=True, single_writer=True,
  operand_maps_match=True, scale_equiv=True, verdict_identical_to_static=True`.
- NEGATIVE `interleaved_dyn_bands` (m_stride = L//HPC, overlapping head rows):
  `z3_proved=False, single_writer=False` — correctly fails (non-vacuous).
- VERDICT: **ALL_POSITIVES_PROVED_AND_NON_VACUOUS.**

The static→dynamic move is byte-layout-only: operand maps, single-writer band
disjointness (m_blocks=HPC), mask, and scale-fold are identical to the §TB1 static
proof. The dynamic region does NOT relax single-writer.

---

## 7. Step tok/s — UNCHANGED (NO-GO consequence)

§DYN is NO-GO, so it is **never on the production step path**: the default B2 branch
(no flag) is v1, and RULE #1 makes §DYN opt-in only (RAISES on smem/proof/compile
failure, no silent use). Production step tok/s @8L/@28L, bs1 AND bs4, is therefore the
**v1 baseline** — wiring the 0.745x §DYN prim into the step would only regress it. No
step-tok/s gain is claimed or measured for the mold; the gap vs Megatron 3399 tok/s is
the pre-existing v1 gap, unchanged by this work.

---

## 8. HONEST NO-GO — WHY static→dynamic still can't beat v1

The mold removed the smem launch-blocker (STATIC 57344→0, HPC=2 launches), but the
dominant costs are untouched:

1. **The dinp 3-index serial term is un-GEMM-able.** `dinp[b,s,h,p,n]` is a per-element
   outer-product zeroed and written by a serial thread-strided loop; the smem flip does
   nothing for it. It dominates the kernel.
2. **path_c dC is per-HEAD → NO M-fattening.** Unlike Tri-Dao there is no per-group
   head-accumulation to amortize tensor-core setup (§4); the mold only amortizes
   operand-staging/sync, a small fraction.
3. **Occupancy cap.** ~91 KB dynamic per CTA + 192 regs pins ~1 CTA/SM at bs1; the
   batched layout has fewer, fatter CTAs than v1's threaded grid, so it loses the
   latency-hiding v1 gets from many light CTAs.

The real fix is re-architecting dC/dinp into per-GROUP split buffers (so a genuine
fattened-M head-accumulation becomes valid) — a far larger change than the smem flip,
and out of scope here. **Verdict: ship nothing to the default path; keep v1. The §DYN
prim stays as a flag-gated, proven, launch-feasible NO-GO reference.**

---

## Reproduce

```
# timing + parity (HPC=2 mandatory: 14 = nheads/ngroups)
CPPMEGA_PATH_C_B2_HEADS_PER_CTA=2 python scratch/probe_chunked_backward_cuda_gb10.py --prod --b2-gemm-ab
# driver attributes (STATIC=0, MAXDYN grant) + SASS
HPC=2 python scratch/wf1_cufunc_attr_b2dyn.py        # source-regex + SASS HMMA
#   then: nvcc -arch=sm_121a -cubin <cache>/device_kernel.cu ... -o /tmp/wf1_dyn.cubin
python scratch/wf1_cubin_attr.py /tmp/wf1_dyn.cubin  # cuFuncGetAttribute STATIC/MAXDYN
cuobjdump --dump-resource-usage /tmp/wf1_dyn.cubin   # SHARED:1024, REG:192
cuobjdump --dump-sass /tmp/wf1_dyn.cubin | grep -c HMMA   # 128 HMMA.16816.F32
# z3
python scratch/proof_b2_batched_driver.py            # ALL_POSITIVES_PROVED_AND_NON_VACUOUS
```
