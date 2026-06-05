# TileLang AUTO-MONO-FUSION — multi-kernel chunk region -> ONE kernel

Extension of the `AutoGemmifyReductions` auto-GEMM pass (docs/APPLE-METAL-GEMM-
AUTOPASS-Z3.md, commit 5b30bb10) ONE level up: at the **dataflow-region** level.
Where the auto-GEMM pass detects a single serial reduction and proves it is a
`T.gemm`, this pass detects a multi-kernel producer-consumer region that
round-trips an intermediate through GLOBAL memory and AUTO-FUSES it into ONE
persistent kernel that keeps the intermediate RESIDENT in shared memory.

Files:
- `/Volumes/external/sources/tilelang/tilelang/transform/auto_fuse_chunk_region.py`
  — detector + z3 fusion-safety prover + dispatcher (commit a576c9d6).
- `/Volumes/external/sources/tilelang/testing/python/transform/test_auto_fuse_chunk_region.py`
  — 17 analysis tests (graph/proof/decline). **17/17 PASS**.
- `/Volumes/external/sources/tilelang/testing/python/transform/bench_auto_fuse_metal.py`
  — the COMPILE+RUN harness that actually fuses + measures on the Apple GPU.

Environment of record: this M-series Mac, macOS arm64, tilelang built+loadable
from `/Volumes/external/sources/tilelang/build` (the live `build/` is UNTOUCHED —
the pass is a Python TIR/graph module, no C++ rebuild). z3 4.15.4.

---

## What the pass does (three stages, mirroring the auto-GEMM pass)

1. **DETECT** (`match_fusion_region`). Reuses the production `path_c_fusion`
   dataflow graph — `_infer_edges` producer-consumer matching + Kahn topo-sort
   `_nodes_in_dependency_order` (imports the real `cppmega_mlx.runtime` module
   when present, else a byte-identical local twin). Recognizes a connected
   linear/tree producer-consumer chain, classifies INTERNAL (privatizable)
   buffers vs region inputs/outputs, and flags ESCAPING buffers.

2. **PROVE** (`prove_fusion`, z3). Extends the auto-GEMM z3 obligation set with
   the NEW fusion-safety obligations, each checked NON-VACUOUSLY over the actual
   `nchunks` x `state_cells` extents (negate the property, require UNSAT):
   - **privatization** — single-writer + single-reader per (chunk, cell), so
     promoting the buffer global->smem introduces no cross-threadgroup race;
   - **carry-domination** — producer write[c] precedes consumer read[c] in the
     fused serial schedule;
   - **escape** — no internal buffer is also a region output.
   z3 disabled / unavailable / UNKNOWN / SAT -> DECLINE (never fuse without a
   proof). Non-vacuity is locked by the disabled-z3 and degenerate-extent tests.

3. **DISPATCH** (`dispatch_region`). On ACCEPT (proved + in-budget + GEMMs kept +
   nothing escapes) selects ONE mono kernel to replace the N per-node builds. On
   DECLINE leaves the region MULTI-KERNEL (RULE #1, no wrong fusion).

---

## MEASURED on the Apple GPU — the auto-fusion ACTUALLY fires + wins

`bench_auto_fuse_metal.py` closes the loop end-to-end. It builds the smallest
demonstrable producer-consumer GEMM chain that round-trips an intermediate
through GLOBAL memory (the structural skeleton of the mamba3 F0->F1 chunk
hand-off):

```
K_A:  Y = X @ W1     (producer — writes Y to GLOBAL)
K_B:  Z = Y @ W2     (consumer — re-reads Y from GLOBAL)
```

The harness first consults the pass (`dispatch_region`). It returns
`fused=True replaced=('K_A','K_B')`, privatizing `Y`. Only then does it emit the
fused kernel. The fused kernel keeps `Y` RESIDENT in shared memory across both
`T.gemm` calls (no second metal dispatch, Y never hits global), and **both GEMMs
stay `T.gemm`** (`keeps_gemms=True`) — the cppmega-class recipe, NOT a scalar
fusion.

Compiled via `tilelang.compile(target="metal")`, run on the Apple GPU through the
native torch.mps boundary, 100-iter timing, M=K=N=P=64, fp32:

| path                         | metal dispatches | ms (typical) | max\|abs vs ref\| |
|------------------------------|-----------------:|-------------:|------------------:|
| MULTI-KERNEL (K_A + K_B)     | 2                | 0.079–0.108  | 0.000e+00         |
| **AUTO-FUSED (ONE kernel)**  | **1**            | **~0.055**   | **0.000e+00**     |

- **AUTO-fused 2 kernels -> 1** (pass-decided, z3-proved, Y privatized to smem).
- **Speedup multi/fused = 1.45x – 1.96x** across runs (fused is rock-steady at
  ~0.055 ms; the multi-kernel time varies with the second-dispatch + global-Y
  round-trip overhead it removes).
- **Parity bit-exact over EVERY element**: fused-vs-multi = 0.000e+00 and
  fused-vs-ref = 0.000e+00. The two paths are algorithmically identical (Y in
  fp32), differing ONLY in whether Y lives in global (multi) or smem (fused).

This is a genuine AUTO-fusion win: the pass removed one of two metal dispatches
and the inter-kernel global round-trip of `Y`, with zero numeric change.

---

## Honest scope (RULE #1) — prototype vs general

| claim | status |
|-------|--------|
| detector recognizes a producer-consumer region + topo-sorts + classifies internal/escaping buffers | **REAL** prototype, 17/17 tests |
| z3 fusion-safety proof (privatization + carry-domination + escape), non-vacuous | **REAL** — declines on disabled-z3 / degenerate extents / dropped GEMMs / escape / over-budget |
| AUTO-fuse a **2-kernel** producer-consumer region into ONE compiled Metal kernel, keep the intermediate smem-resident, measure on the Apple GPU | **REAL** — 1.45–1.96x, bit-exact parity, 1 dispatch vs 2 |
| general N-kernel stateful **mamba3 F0/F1/F2 SSD chunk** mono-fusion with the full `state[headdim,dstate]` chunk-axis carry | **DEFERRED** — dispatcher selects the hand-written mono builder name; it does NOT auto-synthesize the full stateful chunk-carry TIR. The 2-kernel GEMM-chain bench is the demonstrated bar. |

The PROTOTYPE BAR (a working 2-kernel auto-fusion + parity) is MET and MEASURED.
The full stateful SSD chunk mono-kernel auto-synthesis is honestly DEFERRED to the
proven hand-written builder — the same honest deferral the auto-GEMM pass makes
(`_emit_gemm_for_match` returns None rather than fabricating a brittle raw-TIR
splice). On a region it cannot safely fuse (backward B0/B2 — B2 per-contraction
GEMM is a measured 0.749x NO-GO, B0 is a reverse-cumsum scatter), the pass
DECLINES and leaves it multi-kernel (`decline_reason=fused_body_drops_gemms_perf_nogo`).

### Two engineering walls hit + how they were handled (honest)

1. The `mx.fast.metal_kernel` MSL-rewrite bridge (`_mlx_runtime.wrap_tilelang_metal_kernel`)
   does NOT correctly launch an arbitrary compiled `T.gemm` Metal kernel — it
   produced near-zero output (verified: producer GEMM in isolation returned a
   ~99.8%-zero buffer). The bench instead uses the native torch.mps boundary,
   where the fused kernel is **bit-exact** (diff 0.0). Reported honestly rather
   than shipping wrong-output timings.
2. The native torch.mps adapter has an fp16 fragment->`half4` C-style-cast
   codegen bug (`SyntaxError: cast from simdgroup_float8x8 to half4`) when an
   fp32 accumulator fragment is copied to an fp16 shared/output buffer. The bench
   uses all-fp32 buffers, which sidesteps it and keeps the result bit-exact. (The
   cppmega F0 Track-A path avoids this by storing summary_states fp32.)

No C++ rebuild; the live `/Volumes/external/sources/tilelang/build` is untouched.
The pass is default-OFF (env `TILELANG_ENABLE_AUTO_FUSE_CHUNK` /
PassConfig `tl.auto_fuse_chunk_region`).
