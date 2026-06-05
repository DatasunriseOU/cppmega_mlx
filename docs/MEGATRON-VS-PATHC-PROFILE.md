# §P1 — Megatron/cppmega vs path_c BACKWARD: step-level profiler + code diff (gb10 sm_121)

Date: 2026-06-05
Author: David Gornshtein (davidgornshtein@gmail.com)
Repo: `/Volumes/external/sources/cppmega.mlx`
Box: gb10 / DGX-Spark, NVIDIA GB10, sm_121 (consumer Blackwell; `mma.sync m16n8k16`,
NO wgmma/tcgen05, ~99 KB per-threadgroup dyn-smem cap).
TileLang dev root (path_c): `/home/dave/source/tilelang` on branch
`merge/upstream-codegen-reorg`.

## TL;DR — the "120x / fundamental" framing is REFUTED with measurement

The user was right: "fundamental" was unproven. Here is the MEASURED answer.

1. **The "120x on the same engine" (cppmega 3.707 ms vs path_c 447.8 ms) is
   APPLES-TO-ORANGES on THREE axes and is NOT real.**
   - The `3.70674 ms` is the cppmega TileLang **MIMO** `mamba_mimo_bwd_bwd` (the
     gradient/VJP kernel of the `mamba_mimo_bwd_combined` first-order backward) at
     cppmega's **productionish** shape `B=4,S=4096,H=32,G=1,N=64,P=128,R=4,chunk=16`,
     measured on **H200** (Hopper, wgmma tensor cores). Source: cppmega
     `docs/status/mamba3_cuda_bwd_bwd_10wave_summary_2026_04_30.md:72` and
     `mamba3_bwd_bwd_owner_rewrite_wave1_2026_04_29.md:61` (baseline
     `bwd_fwd 1.8718 + bwd_bwd 3.7084 = chain 5.5628 ms`).
   - The `447.8 ms` was path_c's B2→B1→B0 VJP chain at path_c **prod**
     `bs1,S=4096,c=64,G=8,H=112,P=64,N=64` — but at a RE-GRIDDED / reduced-state
     substitution (§17), because the full-N=64 scan tile does not fit the gb10 smem
     cap (see below). The HONEST fresh full-config number is ~1300 ms (this doc).
   - Differences: (a) different HARDWARE (H200 wgmma vs gb10 sm_121 mma.sync);
     (b) different SHAPE (H32/G1/P128/chunk16 vs H112/G8/P64/chunk64 — ~3.5x heads,
     8x groups, 4x chunk); (c) the path_c number was a reduced-state substitution,
     not the full kernel.

2. **cppmega's shipped MIMO TileLang backward DOES NOT COMPILE on the current gb10
   dev TileLang** (`merge/upstream-codegen-reorg`): SEGFAULT in the tirx
   `ExprFunctor/AccessPath` visitor at BOTH productionish and representative shapes
   (see "cppmega backward on gb10" below). So a like-for-like same-box cppmega-MIMO
   number cannot be produced on this branch; its authoritative numbers are H200.

3. **What Megatron/cppmega does CONCRETELY faster (the real lever, MEASURED):**
   cppmega's full-model training step does NOT use the TileLang MIMO kernels for the
   hot mamba path at all — it runs the **fused Triton SSD** `mamba_chunk_scan_combined`
   (one smem-resident kernel: cb, dA-cumsum, chunk-states, scan, combine), measured on
   gb10 at **3.11 ms fwd / ~10 ms bwd**. path_c splits the SAME SSD math into separate
   gridded kernels F0/F1/F2 (fwd) and B2/B1/B0 (bwd), each round-tripping the chunk
   tensors through global memory, and **F0 precompute ALONE = 16.37 ms ≥ 5.7x cppmega's
   ENTIRE fused fwd**. The lever is FUSION + smem-residency of the chunk state, not a
   different algorithm and not host/launch staging.

## Config matched / reconciliation

| stack | what it is | shape | hardware | source |
| --- | --- | --- | --- | --- |
| cppmega MIMO bwd (`mamba_mimo_bwd_bwd`) | TileLang VJP kernel | B4 S4096 H32 G1 N64 P128 R4 c16 | **H200** | cppmega status docs |
| cppmega MIMO `bwd_combined` chain | `bwd_fwd` + `bwd_bwd` (first-order) | same | H200 | 1.87 + 3.71 = **5.56 ms** |
| cppmega Triton SSD (in the real step) | `mamba_chunk_scan_combined` fused | path_c prod tile S4096 c64 G8 H112 P64 N64 | **gb10** | MAMBA3-PATHC-VS-CPPMEGA.md: **3.11 ms fwd / ~10 ms bwd** |
| path_c B2/B1/B0 VJP chain | 3 gridded TileLang kernels | bs1 S4096 c64 G8 H112 P64 N64 | **gb10** | THIS doc: **~1300 ms** |

The only TRUE same-box (gb10), same-config, same-SSD-math comparison available is the
last two rows: **cppmega fused Triton fwd 3.11 ms vs path_c F0 alone 16.37 ms (F2 won't
even launch).** That is the real, honest gap.

## MEASURED — path_c prod backward chain (gb10, fresh, full config)

`scratch/probe_chunked_backward_cuda_gb10.py --prod`, cfg
`bs1,S=4096,c=64,G=8,H=112,P=64,N=64,nchunks=64`, CUDA-event medians:

| prim | what it computes | median ms/call |
| --- | --- | ---: |
| **B2** `chunk_scan_combine_bwd` | the VJP scan-combine (dC/dx/dz/dchunk_states/dinp_diag/dA_y/dD) | **1026.2** |
| **B1** `inter_chunk_recur_bwd` | cross-chunk reverse adjoint (O(S/c)) | **5.30** |
| **B0** `chunk_precompute_bwd` | dinp split (dx/dB/ddt + dstates-coupled) | **269.2** |
| **chain total** | B2→B1→B0 | **~1300.8** |

8-grad parity PASS (gate 1e-3, all elements): `dz=1.73e-04 dx=8.10e-04 dC=5.03e-05
dB=1.09e-05 dlog_decay=6.68e-04 ddt=1.50e-04 dh0=1.84e-04 dD=2.19e-05`, WORST=8.10e-04.

### Isolated tilelang-only B2 (no reference, no resets) — gb10 CUDA events

To remove the probe's reference-VJP and the 4.7 GB `dinp_diag.zero_()` per call from the
measurement, the B2 prim was timed in isolation feeding only the kernel:
**`B2_ISOLATED_KERNEL_ONLY median_ms = 827.3`** (`/tmp/b2_isolated.py`, 10-iter median,
min 826.6 / max 828.7). So the genuine tilelang B2 device time is **~827 ms**; the
probe's 1026 ms includes the per-call zero-fills of the giant grad scratch tensors.

## gap LOCALIZED — it is B2, and B2 is COMPUTE-bound serial reduction

The backward gap is concentrated in **B2 (~827 ms isolated / ~1026 ms in-chain)**; B0 is
secondary (~266 ms) and B1 is negligible (~5 ms). The WHY, from the code:

- path_c's **default** prod B2 prim `chunk_scan_combine_bwd_cuda_prim`
  (`mamba3_chunked_backward_core.py:413`, the one prod actually runs with
  `CPPMEGA_PATH_C_B2_V2` unset) computes every backward contraction as **per-thread
  `T.serial` scalar reductions** over the lower-triangular `(s,l)` chunk pairs and over
  `headdim`/`dstate` — `grep`: 0 `T.gemm` in its body, many `T.serial`/`T.Parallel`
  scalar loops. It is `T.Kernel(grid)`-parallel across chunks/heads but each thread does
  scalar MACs. This is compute-bound: the isolated 827 ms is pure GPU device time on a
  single long-running grid (§18 already found these B2 kernels occupancy-SATURATED at
  bs1, v2 dstate-split 0.997x → not launch-bound, not occupancy-starved).
- cppmega's `mamba_mimo_bwd_bwd` (`state-spaces-mamba/.../mamba3_mimo_bwd.py`) does the
  SAME contractions as **15 `T.gemm` tensor-core matmuls** (dk, dq, dPsiV, dqk, dstates;
  lines 811/822/846/905/930/952/980/1026/1050/1148 …) with the chunk state RESIDENT in
  shared memory, grid `T.Kernel(H, B, threads=256)` — ONE block per (head, batch) that
  loops over `nchunks` INTERNALLY with no global round-trip per chunk.

### Why path_c can't just adopt cppmega's tensor-core GEMM B2 at prod (MEASURED NO-GO)

path_c already HAS a tensor-core GEMM B2 prim (`chunk_scan_combine_bwd_cuda_prim_gemm`,
the §27 four-GEMM body). At prod dims, single-process A/B on gb10:

```
[B2-GEMM-AB] MEASURED v1(serial)=1020.3 ms  gemm(tensor-core)=1352.9 ms
             speedup=0.754x  NO-GO   (math-equivalent, max|abs| dC 6.2e-5 / dA_y 3.2e-5)
```

The GEMM B2 is **0.754x — SLOWER** than the serial B2 at path_c's prod tile. Reason
(§MF1 / §27): at `c=64,P=64,N=64` the per-CTA smem to stage the GEMM operands AND keep
the chunk state resident blows the gb10 99 KB cap (the resident `dinp_diag[L,P·N]` tile
alone = 1 MB), so the GEMM variant cannot fill the 4-warp `mma.sync` partition AND fit
smem simultaneously → it pays the staging/sync cost without the tensor-core throughput.
cppmega avoids this because its productionish uses **chunk=16** (4x smaller chunk → the
resident chunk tile fits smem) on **H200** (wgmma + larger smem). So "port cppmega's
gridded GEMM backward" does NOT close the gap at path_c's gb10 prod tile — MEASURED.

## cppmega backward on gb10 — SEGFAULT on the current dev TileLang (honest NO-RUN)

`mamba_mimo_bwd_combined` (the shipped cppmega MIMO first-order backward =
`mamba_mimo_bwd_fwd` recompute + `mamba_mimo_bwd_bwd` VJP) was driven on gb10 at BOTH
productionish (B4 S4096 H32 G1 N64 P128 R4 c16) and representative (B2 S1024 H16 …):

```
!!!!!!! Segfault encountered !!!!!!!
  in tvm::tirx::ExprFunctor<void (PrimExpr const&, AccessPath)>::VisitExpr(...)
  in tvm::tirx::TIRVisitorWithPath::VisitStmt_(AttrStmtNode const*, AccessPath)
(exit 139, before the kernel even runs)
```

The segfault is in the **tirx AccessPath visitor** of the `merge/upstream-codegen-reorg`
dev TileLang that path_c uses — a COMPILER regression on this branch, not a kernel bug.
cppmega's MIMO backward was authored/measured against an older stable TileLang on H200;
the stable build on gb10 (`/home/dave/tilelang-build`) is not fully wired into the venv
(missing `libz3.so.4.16` / tvm_ffi bindings) so a same-box cppmega-MIMO number is not
producible on this branch. Recorded as MEASURED NO-RUN (RULE #1: surfaced, not faked).

## GPU-busy% / launch-bound vs compute-bound (MEASURED, with an honest caveat)

- **path_c B2 is COMPUTE-bound.** The isolated tilelang-only device time is **827 ms**
  (CUDA events, on-stream) for a single grid — there is no host gap inside it; §18
  measured it occupancy-saturated (dstate-split 0.997x) and batch-invariant at bs1. It
  is slow because it does the contractions as scalar serial MACs, not because of
  launch/occupancy.
- **nsys CAVEAT (honest):** `nsys profile --trace=cuda` on gb10 sm_121 does NOT capture
  the TileLang `main_kernel` launches (the TVM runtime launch path is not symbolized by
  nsys's CUDA injection on this box) — the `cuda_gpu_kern_sum` for the isolated B2 run is
  empty of tilelang kernels (`"does not contain CUDA kernel data"`), and the full-probe
  trace shows only the torch reference-VJP `at::native` elementwise/`gemvx` kernels +
  the 4.7 GB `dinp_diag` zero-fills. So GPU-busy% via nsys is unavailable for the
  tilelang kernels on this branch; the compute-bound conclusion rests on the on-stream
  CUDA-event device time (827 ms, no host bubble) + the §18 occupancy receipts.
- **At the STEP level the cppmega advantage is ALSO launch-amortization:** cppmega runs
  the full step under CUDA-graph, fully device-resident, where mamba+m2rnn are only
  ~24% of the iter (nam56r nsys: MIMO 17.5% + M2RNN 6.2%) at ~3692 tok/s / 4437 ms-iter
  on gb10 — but that is the ENGINE wrapper, secondary to the per-kernel fusion gap above.

## Concrete WHAT Megatron/cppmega does differently (NOT fundamental)

The SAME Mamba3/SSD math is faster in cppmega because of CONCRETE engineering choices,
each reproducible:

1. **Fused, smem-resident SSD kernel** (the dominant lever): cppmega's hot path is ONE
   fused Triton kernel (`mamba_chunk_scan_combined`, 3.11 ms fwd gb10) keeping the chunk
   state in shared memory; path_c splits it into F0/F1/F2 + B2/B1/B0 gridded kernels with
   global round-trips, and F0 ALONE is 16.37 ms (≥5.7x). FUSION, not algorithm.
2. **Tensor-core GEMM contractions** in the VJP (cppmega MIMO bwd_bwd: 15 `T.gemm`)
   vs path_c's default serial scalar reductions — BUT this only pays off when the chunk
   tile fits smem (cppmega chunk=16 / H200), which it does NOT at path_c's gb10 prod tile
   (chunk=64, GEMM B2 = 0.754x SLOWER, MEASURED). So this lever is HW/shape-gated, not
   free.
3. **Smaller chunk (16 vs 64)** so the resident chunk state fits the smem cap and the
   tensor-core partition stays full — the structural reason path_c's GEMM/mono B2 hit the
   99 KB wall at chunk=64,N=P=64 (§MF1).
4. **CUDA-graph + device-resident step** amortizing launch across the ~24%-mamba iter
   (engine wrapper, secondary).

None of these is "fundamental." The honest path forward for path_c on gb10 is (a) FUSE
F0/F1/F2 and B2/B1/B0 into smem-resident kernels and (b) drop chunk to 16-class so a
tensor-core GEMM VJP fits the 99 KB cap — exactly what cppmega/Megatron does. The
already-tried mono-fusion at chunk=64 is a measured smem NO-GO (§MF1); the lever is the
chunk-size + fusion REDESIGN, not a one-kernel port.

## Files / commands (all gb10, single-owner mutex)

- path_c prod backward chain + per-grad gate:
  `scratch/probe_chunked_backward_cuda_gb10.py --prod`
- B2 serial-vs-GEMM A/B at prod: `... --prod --b2-gemm-ab`
- isolated tilelang-only B2 timer: `/tmp/b2_isolated.py` (gb10)
- cppmega MIMO backward driver (SEGFAULTs on this branch): `/tmp/cppmega_bwd_bench.py`
  + `state-spaces-mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py`
  (`mamba_mimo_bwd_combined` / `mamba_mimo_bwd_fwd` / `mamba_mimo_bwd_bwd`)
- cppmega Triton SSD fused (the real-step kernel):
  `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`
- per-region same-box compare (already MEASURED): `docs/MAMBA3-PATHC-VS-CPPMEGA.md`,
  `scratch/mamba3_m2rnn_compare.py --prod`
- path_c B2 default serial prim: `cppmega_mlx/nn/_tilelang/mamba3_chunked_backward_core.py:413`
  (`chunk_scan_combine_bwd_cuda_prim`); GEMM prim same file
  (`chunk_scan_combine_bwd_cuda_prim_gemm`).
