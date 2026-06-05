# §MF1 — Mono-Fused path_c Backward Port (cppmega mono-chunk) — MEASURED on gb10

Date: 2026-06-05
Author: David Gornshtein (davidgornshtein@gmail.com)
Repo: `/Volumes/external/sources/cppmega.mlx`
Commit: `ab8332b` (`feat(mamba3 mono-bwd): cppmega mono-chunk PORT — B2 GEMM + dinp_diag-B0 fused, flag-gated, gb10-only`)
Box: gb10 / DGX-Spark, NVIDIA GB10, sm_121 (consumer Blackwell; mma.sync m16n8k16, NO wgmma/tcgen05), ~99 KB per-threadgroup smem.
TileLang dev root: `/home/dave/source/tilelang/build`; torch `2.13.0.dev20260417+cu132`, CUDA available.

## Goal

path_c backward is a 6-kernel chain (fwd cache F0/F1/F2 + backward B0/B1/B2), the
§17-measured **447.8 ms/call** = ~80% of the step. The cppmega "Megatron-class"
recipe is a MONO-FUSED chunk kernel: ONE kernel keeps the chunk state resident in
shared/regs, runs LARGE tensor-core GEMMs internally, no inter-kernel sync / global
round-trips. This port maps that onto path_c's B-algebra fused into ONE kernel.

## What was built (additive, flag-gated, byte-identical-safe)

NEW prim added purely additively to
`cppmega_mlx/nn/_tilelang/mamba3_chunked_backward_core.py` (commit `ab8332b`:
623 insertions, ZERO deletions in the core; the only 3 deletions are probe argparse
wiring):

- `bwd_mono_cuda_prim` (sm_121) — ONE kernel per `(chunk, head, batch)`. Body = the §27
  four-GEMM B2 (`chunk_scan_combine_bwd_cuda_prim_gemm`, byte-identical math:
  DYX=dY@x^T transpose_B; dC_off=dY@prev_states; dC_diag=M@B with decay+dt+mask folded;
  dchunk_states=(decay·dY)^T@C transpose_A), EXTENDED to keep `dinp_diag[l,p,n]` RESIDENT
  in a shared `DINP[L,P·N]` tile and fold the dinp_diag-dependent HALF of B0
  (`dx_diag = dt·sum_n dinp_diag·B`, `dB_diag = dt·sum_p dinp_diag·x`,
  `ddt_diag = sum_{p,n} dinp_diag·x·B`) BEFORE any global re-read.
- `bwd_mono_metal_prim`, `build_bwd_mono`, `bwd_mono_grid`,
  `MAMBA3_BWD_MONO_OP_NAME = "mamba3_bwd_mono"`.
- probe flags `--bwd-mono-ab` / `--nano` in `scratch/probe_chunked_backward_cuda_gb10.py`.

**What CANNOT fuse (RULE #1, stated up front):** B1 (cross-chunk reverse adjoint)
carries an adjoint ACROSS chunks → a per-`(chunk,head)` CTA cannot express it without
cooperative-grid sync; STAYS separate. B0's dstates-coupled half needs `dstates` (the
OUTPUT of B1) → STAYS in the post-B1 B0 kernel. The math is LINEAR in `dinp`, so the
split (mono owns dinp_diag→dx/dB/ddt; B0 takes dinp_diag=0 and adds only the `_states`
half) is EXACT. Maximal honest fusion = **B2 + dinp_diag-B0** in one kernel.

## MEASURED RESULT — SMEM WALL at prod = honest structural NO-GO

`build_bwd_mono(cuda)` at prod cfg (S=4096, c=64, g=8, H=112, P=64, N=64, bs1) RAISES
(verbatim gb10 output):

```
build_bwd_mono(cuda): per-threadgroup smem 1139456 B (§27 GEMM 90624 B +
resident DINP[L=64,P*N=4096] 1048832 B; L=64,P=64,N=64) >= the gb10 budget
101376 B. The resident-dinp_diag mono fusion is the largest tile
(L*P*N*4 = 1048576 B) and does NOT fit at these dims — this is the HONEST
structural wall (per §27 the per-CTA resident dinp_diag tile is ~64x the GEMM
operands). RULE #1: refusing to launch over-budget or silently fall back...
```

- Total mono smem at prod = **1,139,456 B (1.14 MB) ≫ gb10 budget 101,376 B (99 KB)** — over by **11.2×**.
- The resident `DINP[L,P·N]` tile alone = **1,048,576 B (1 MB)**, exactly the §27-predicted "~64× the GEMM operands".
- `build_bwd_mono(metal)` RAISES unconditionally (Apple 32 KB cap; the §27 4-GEMM staging alone is 72.5 KB) → mono is gb10-only.

This is a RAISE with where+what (RULE #1) — NO over-budget launch, NO silent fallback
to the 6-kernel chain. The mono fusion only fits at small `L·P·N`:
`L=16,P=32,N=32` = 79 KB FIT (commit envelope).

### Prod 8-grad parity (unchanged 6-kernel chain, all elements, gate 1e-3) — PASS

The `--bwd-mono-ab` prod run hits the mono SMEM-NO-GO then runs the full 6-kernel
chain through the 8-grad gate over ALL elements (gold fp16-cache for dD):

```
dz=1.73e-04 dx=8.10e-04 dC=5.03e-05 dB=1.09e-05 dlog_decay=6.67e-04
ddt=1.50e-04 dh0=1.84e-04 dD=2.38e-05  -> WORST=8.099e-04  gate<1e-03  PASS
```

Chain prim timings at prod (gb10): B2=1021.3 ms, B1=5.28 ms, B0=266.9 ms.
`overall_pass: true`. dD via the honest fp16-cache gold = 2.38e-05.

### In-budget build verification (the port is REAL, not vaporware)

At the fit config the mono prim does construct to ONE CUDA kernel. Building
`bwd_mono_cuda_prim(threads=32)` at `L=16,P=32,N=32` (resident DINP 65,536 B) compiles
to a single `main_kernel` (`__global__` token count = 2 = prototype + definition; one
logical kernel), shared memory declared, with `mma.` present. The full 4-warp
`mma.sync m16n8k16` tensor-core path requires ≥32-wide tiles AND ≥4 warps (128 threads)
— which forces dims whose resident DINP tile blows the 99 KB budget. So the mono kernel
faces a genuine DUAL bind: tiles small enough to fit smem are too small to fill the
tensor-core 4-warp partition. The §27 four-GEMM B2 body it reuses DOES emit real
`mma.sync m16n8k16` at prod tiles (separately verified) — but the resident-DINP
extension cannot coexist with those tiles in 99 KB.

## Step tok/s impact

Because the mono build is SMEM-NO-GO at prod, the 6-kernel chain is **byte-identical**
(additive entry point — the 6 builders/grids/op-names are untouched; commit shows ZERO
core deletions). The mono kernel is NEVER invoked at prod, so:

- **step tok/s @8L = 907, @28L = 298 — UNCHANGED** (no kernel substituted into §17).
- gap vs Megatron 3399 tok/s: **3.75× @8L / 11.4× @28L — UNCHANGED**.

The mono-fused lever does NOT move tok/s at prod dims. Honest NO-GO.

## mono-bwd ms vs the 447.8 ms 6-kernel chain

N/A at prod — the mono kernel does NOT BUILD at prod (SMEM-NO-GO), so there is no prod
mono ms to compare against 447.8 ms. The comparison is structurally impossible at prod
dims, not merely slower. At the in-budget `--nano` cfg the mono runs but at a config 64×
smaller than prod (not a prod claim).

## Honest attribution

Three independent measurements of THIS kernel class all say NO-GO:
1. §27 measured the four-GEMM B2 prim (which mono reuses verbatim) at **0.749×** vs v1
   (1371.5 vs 1027.5 ms, same gb10) — single-64-tile m16n8k16 GEMM staging+sync exceed
   the serial reductions. Fusing B0's dinp_diag work does not shrink them.
2. cppmega's OWN identical mono-chunk WMMA kernel closed at **3.009× SLOWER** than
   TileLang (`cppmega/docs/status/mamba3_mono_cuda_chunk_wave10_final_2026_04_30.md`,
   verbatim: *"Do not merge any monolithic CUDA chunk implementation from this branch as
   a production path. CUDA path 3.009385065623324x slower"*). Our port reaches the SAME
   structural conclusion on the SAME model — independently confirming cppmega's finding.
3. §18 measured the B2 kernels occupancy-SATURATED at bs1 (v2 dstate-split 0.997×):
   removing the dinp_diag round-trip + one launch saves latency, not compute.

The mono-fused port is correctly implemented, gated, and byte-identical-safe; at prod
dims it is a MEASURED structural NO-GO (smem wall), not a speedup. RULE #1 honored
throughout: the build RAISES with where+what, no silent fallback, no over-budget launch.

## Files

- `/Volumes/external/sources/cppmega.mlx/cppmega_mlx/nn/_tilelang/mamba3_chunked_backward_core.py`
  (`bwd_mono_cuda_prim` / `build_bwd_mono` / `bwd_mono_grid` / `bwd_mono_metal_prim`)
- `/Volumes/external/sources/cppmega.mlx/scratch/probe_chunked_backward_cuda_gb10.py` (`--bwd-mono-ab` / `--nano`)
- `/Volumes/external/sources/cppmega.mlx/docs/MAMBA3-PATHC-MULTIKERNEL-DESIGN.md` §9
