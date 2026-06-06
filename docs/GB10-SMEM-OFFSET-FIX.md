# GB10 SMEM / Offset Fix — §SO1

MEASURED on NVIDIA GB10 DGX-Spark (sm_121, Grace-Blackwell, unified 121 GB),
2026-06-06. All GPU execution gb10-only (Mac SoC-watchdog safety: no local Metal
GPU dispatch this run). Config: bs=1 S=4096 chunk=64 G=8 H=112 P=64 N=64.

## §SO0 — Core question: which tilelang/tvm does gb10 ACTUALLY load?

The user asked: does the gb10 `path_c` env load OUR source fork (with the fix) or a
pip/stale wheel (without it)? **ANSWER: it loads OUR source fork. Confirmed live.**

The cppmega_mlx probes run with:
```
PYTHONPATH=/home/dave/source/cppmega_mlx:/home/dave/source/tilelang:\
  /home/dave/source/tilelang/3rdparty/tvm/python:\
  /home/dave/source/tilelang/3rdparty/tvm/3rdparty/tvm-ffi/python
TVM_LIBRARY_PATH=/home/dave/source/tilelang/build/lib
/home/dave/cppmega-venv/bin/python ...
```
Resolved live:
- `tvm`      -> `/home/dave/source/tilelang/3rdparty/tvm/python/tvm/__init__.py` (OUR fork)
- `tilelang` -> `/home/dave/source/tilelang/tilelang/__init__.py` (OUR fork)
- `libtvm_runtime.so` -> `/home/dave/source/tilelang/build/lib/` (dev root),
  mtime **2026-06-06 02:04:25**, and it CARRIES the new RULE#1 cap string
  `"... B of shared memory, which exceeds the device per-block opt-in cap of ..."`.
  The old opaque `"Failed to set the allowed dynamic shared memory size to"` is
  retained as the second-tier raise.

There is NO pip `tilelang`/`tvm`/`apache-tvm` installed in the venv. The "wrong
tilelang" hypothesis is **falsified** — gb10 runs the fixed fork.

## §SO1 — What gb10 was using, the diagnosis, the fix, measured deltas

### The original blocker (reproduced live, HPC=2)
`chunk_scan_combine_bwd_cuda_prim_gemm_batched` at HEADS_PER_CTA=2 (the amortization
lever) FAILED TO LAUNCH. With the rebuilt fork the failure is now a CLEAR named-cap
raise (was an opaque driver error):
```
tvm.error.InternalError: Kernel 'main_kernel' requires static=57344 + dynamic=66560
= 123904 B of shared memory, which exceeds the device per-block opt-in cap of
101376 B (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN). ... (RULE#1 fail-fast.)
```
**Diagnosis (b) CONFIRMED:** static (57344) + dynamic (66560) = 123904 B overflows the
sm_121 per-block opt-in cap (101376 B live). The request is GENUINELY too big — it is
NOT a missing opt-in (the cuFuncSetAttribute opt-in was always attempted) and NOT a
carveout issue. Moving the 5 static `__shared__` tiles to `scope="shared.dyn"` does
NOT cure it (same total). HPC=4 does not even build for H=112/G=8 (heads_per_group=14;
4 neither divides nor is a multiple of 14 — a clean config raise, not a smem issue).

### The runtime fix (tvm cuda_module.cc:212-246)
RULE#1 static-aware check: query `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` (static) +
`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (cap) and RAISE naming
static+dynamic+cap BEFORE the opaque driver error. No silent clamp. (Already in the
loaded binary; verified.)

### The smem-fit kernel reductions (§SO1, this run) — make HPC>=2 LAUNCH
Two precision-disciplined reductions to `chunk_scan_combine_bwd_cuda_prim_gemm_batched`
ONLY (v1-threaded + §27 single-tile prims byte-identical — flag-gated, untouched):

1. **DYX band -> single reused (L,L) tile.** DYX is a per-head intermediate consumed by
   dC_diag AND the dseg dA-grad. Fusing the dseg accumulation UP into the dC_diag
   head-loop lets DYX be one (L,L) tile, not an (HPC,L,L) band. Precision-NEUTRAL (same
   fp32 DYX_frag, same fp32 atomic_add; dseg/dC_diag atomic_adds commute). Saves
   (HPC-1)*L*L*4 = 16384 B at HPC=2. dynamic 66560 -> 50176, total 107520 B (still over).

2. **dY result band fp32 -> fp16.** Every consumer downcasts dY to an fp16 MMA operand
   or multiplies it by an fp32 scalar (dC_off, dchunk, dinp, DYX). dD uses the fp32
   LOCAL dy_v (not the band) so dD is unaffected. Saves HPC*L*P*2 = 16384 B at HPC=2.
   The per-grad 1e-3 gate is the hard arbiter (RULE#1: explicit operand dtype + fail-
   fast gate, NOT a silent downgrade). dynamic 50176 -> 33792, total ~91136 B < cap.

**RESULT: HPC=2 NOW LAUNCHES** (no InternalError; kernel runs; smem ~91 KB < 101376 cap).

### MEASURED deltas (prod shape S=4096 H=112, B2-AB probe)
| variant                         | ms      | vs v1 (905.2 ms) | vs §27 (1231 ms) |
|---------------------------------|---------|------------------|------------------|
| v1_threaded (prod prim)         | 905.2   | 1.000x           | —                |
| §27 single-tile GEMM            | 1231.1  | 0.735x           | 1.000x           |
| batched HPC=1 (pre-fit)         | 1250.0  | 0.724x           | 0.985x           |
| **batched HPC=2 (smem-fit, A)** | **1279.3** | **0.708x**    | 0.962x           |

Small-shape full-chain probe (S=256/512, H=2): batched HPC=2 = 0.671x / 0.727x vs v1;
at S=512 batched (9.15 ms) beats §27 single-tile (13.28 ms) at **1.452x** — the batching
DOES help vs §27, but neither beats v1-threaded.

### (B) Tall-M (offset-relax) — NOT a valid single large MMA here
`gemm_op.py` carries the `allow_first_dim_offset` relax (gated; prod prims never set it,
stay byte-identical) and z3 proves the tall-M offset-band map (bands @ 0/64/128/192
tile-aligned + disjoint; misaligned offset=65 rejected). BUT the four B2 contractions are
**block-diagonal over heads**: DYX (dY@x^T), dC_off (dY@prev_states), dchunk ((decay*dY)^T@C)
have per-head DISTINCT operands on BOTH sides — a dense tall M=HPC*L MMA would compute
HPC^2 cross-head GARBAGE blocks. Only dC_diag/dseg share a per-GROUP operand (B/C), so a
genuine tall-M is valid for AT MOST 1 of 4 GEMMs and only across same-group heads. Since
smem-fit-only is already 0.708x, a tall-M on 1/4 GEMMs cannot flip it to >1.0x. **SASS
confirms the launched HPC=2 kernel is per-64 MMA in a serial head loop** (`tl::mma_sync<
...,16,8,16,...>` with M-loop `i<2` INSIDE `for hh<2` — two M=64 waves per CTA, NOT one
M=128 wave). The tall-M lever is therefore a NON-STARTER for this kernel's algebra.

### Parity (RULE#1 gate, RAISES on miss)
- B2-AB (7 grads, prod shape): worst max|abs| = **2.04e-05** < 1e-3. All pass.
- Full 8-grad chain probe (--b2-gemm-ab, dD fp16-cache, ALL elements): **overall_pass=true**,
  worst = 5.46e-04 < 1e-3 (dz/dx/dC/dB/dlog_decay/ddt/dh0/dD). The fp16 dY band + DYX
  fusion did NOT break end-to-end parity.

### z3 proof (re-run)
`scratch/proof_b2_batched_driver.py` -> **ALL_POSITIVES_PROVED_AND_NON_VACUOUS**: dchunk
+ tallM_offset_band positives proved; negatives rejected (overlap_band single-writer fail,
transpose_bug counter-witness, tallM_misaligned_offset @65, metal dropped-carry sat).

## §SO1-Metal — L=32 sub-chunk retile (codegen status, NUMERIC DEFERRED)
Lowering the Metal batched prim at sub_chunks=2 (L_sub=32) **RAISES NotImplementedError**:
the sub-chunk-loop kernel BODY (inter-sub-chunk state carry) is GATED pending Apple-GPU
numeric validation (watchdog safety — no Apple GPU exec this run). Only the SIZING +
z3 associativity are done, so MSL/simdgroup-MMA emission **cannot be verified** (no
lowerable kernel). MEASURED sizing (`_metal_subchunk_smem_bytes`, Apple cap 32768 B):

| HPC | sub | L_sub | all_static | store_dynamic | fits 32KB? |
|-----|-----|-------|-----------|---------------|------------|
| 1   | 1   | 64    | 57856     | 41472         | no / no    |
| 1   | 2   | 32    | 43264     | **26880**     | no / **YES** (store_f32 -> dynamic) |
| 1   | 4   | 16    | 37504     | 21120         | no / YES   |
| 2   | 2   | 32    | 49664     | 33280         | no / no (512 B over) |
| 2   | 4   | 16    | 40192     | 23808         | no / YES   |

So the L=32 retile fits Apple's 32 KB ONLY at HPC=1 with store_f32 moved to
threadgroup-dynamic (26880 B). The L=32-alone all-static layout never fits (43264 B).
The prior "4.99x usable at prod" is NOT realizable this run: the sub-chunk body is
unimplemented and Apple-GPU numeric is deferred.

## VERDICT — honest NO-GO with WHY
The §SO1 smem-fit fix achieves the stated mechanical goal: **HPC=2 now LAUNCHES** on
sm_121 (66560B/123904B no longer rejected; clear named-cap raise replaces the opaque
error; no silent clamp). All 8 grads pass the 1e-3 gate. BUT the amortization is **still
NO-GO**: batched HPC=2 = 1279 ms = **0.708x vs v1-threaded 905 ms**. ROOT CAUSE: the
gemm_op.py offset-0 staging forces per-head M=64 MMAs in a serial head loop (SASS-
confirmed), and the four contractions are block-diagonal so a genuine tall-M band MMA is
mathematically invalid for 3/4 GEMMs — the staging/sync amortization over HPC heads does
not recover the per-64-tile tensor-core setup cost. v1-threaded remains the prod floor;
the Tri-Dao ~10 ms gap is NOT closed by this rewrite. The batched B2 stays flag-gated and
OFF in prod (prod prims byte-identical), so the prod backward chain / step tok/s are
UNCHANGED from the v1 baseline (447.8 ms chain reference; batched would make the chain
WORSE, so it must not be wired in).
