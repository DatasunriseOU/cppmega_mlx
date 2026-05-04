# Path C blockers — what TileLang upstream needs for full Apple Silicon parity

Date: 2026-05-04

This document tracks what TileLang DSL features are still missing so Path C
(`@T.prim_func` lowered through patched apple-head TileLang on Metal target)
can match Path B (hand-written MSL via `mx.fast.metal_kernel`) for the
remaining ops in `cppmega_mlx/nn/_tilelang/`.

**Why this matters:** every Path C port we add may surface a new TileLang
scheduler/codegen gap. When that happens, we file a fresh upstream patch
under `docs/upstream/`, stack it on top of the existing `jorgecurious/tilelang:metal-gemm-upstream-rebase`
chain (PR #2130), and update this tracker.

The Path B/C bench table in `docs/production_kernel_routing.md` is the
authoritative scoreboard. This doc explains the **why** for each ❌.

---

## Currently shipped Path C (1 of 11 ops)

| Op | Bench (M4 Max) | Status | Notes |
|---|---|---|---|
| **Mamba3 main fwd+bwd** (B=2 T=512 H=4 P=32 N=64 FP32) | C 7.707 ms vs B 7.823 ms | ✅ shipped | Bit-exact parity with B, 1.5% faster (within noise). Path C kept as DSL-path-works reproducibility receipt. |

---

## Ops where Path C is blocked

### Sparse-MLA family (4 ops)

The sparse-MLA kernels need 32×32 fragment tiles for performance. Our
existing `tilelang_metal_pipelined` patch (Agent D's 3D-buffer fix)
unblocked T.Pipelined num_stages>1 only at **16×16** fragments — sparse-MLA
wants 32×32. The path forward is a follow-up patch that extends the 3D
buffer fix to larger fragment sizes.

| Op | Path C blocker | What patch unblocks it |
|---|---|---|
| Sparse-MLA fwd BF16 | T.Pipelined num_stages>1 only works at 16×16 fragments | **NEW patch needed**: extend `tilelang_metal_pipelined` to 32×32 fragments. Likely touches `tilelang/intrinsics/metal_macro_generator.py` (3D handling) and `src/op/copy.cc` (fragment size dispatch). |
| Sparse-MLA fwd FP8 | (a) same T.Pipelined 32×32 issue, (b) scheduler-glue between `T.fp8_scaled_matmul` macro and `GemmMetalScalar` so per-load scale fuses into K-loop | Two patches: same as above, plus a scheduler pass that fuses scale broadcast at gemm fragment-load time. The scaffold for the macro is already in `tilelang_metal_fp8_scaled_matmul`; a follow-up scheduler patch is needed. |
| Sparse-MLA fwd blockscaled (e8m0) | TileLang DSL has no block-scale (e8m0) layout primitive | **NEW patch**: introduce `T.BlockScaledLayout(scale_dtype="e8m0", block_size=16)` API and lower it through GemmMetalScalar with the matching scale tile bookkeeping. Larger work — likely 200-400 LOC. |
| Sparse-MLA bwd (chunked dQ/dK/dV) | DSL doesn't compose chunked backward flow with VJP via `mx.custom_function`-like glue. Path B does manual chunking outside autograd. | **DSL feature**: scheduler-level support for "chunked backward with explicit dKV accumulation". Not blocked on a single patch — may require a TileLang language-level extension. Defer. |

### Topk + selector (1 op)

| Op | Path C blocker | What patch unblocks it |
|---|---|---|
| `topk_selector` | No profitable Metal schedule for argpartition + per-row index gather. Path B uses a hand-tuned simdgroup reduction; DSL can't currently emit that pattern. | **DSL feature**: T.simdgroup_reduce primitive with custom comparator + per-row stable sort. Larger work; not on the critical path for M0. Defer. |

### FP8 ops (2 ops, slower than B)

These technically work via Path C but are 3-6× slower than Path B's
audiohacking-style hand-written MSL. The win path is **fused scheduler**.

| Op | Current Path C state | What's needed |
|---|---|---|
| FP8 scaled matmul (128³ e4m3 per-tensor) | ✅ 0.555 ms (Path B 0.172 ms — 3.16× slower) | Fuse the per-load scale broadcast into the GemmMetalScalar K-loop. The audiohacking kernel does this; our DSL path emits a post-loop multiply. **NEW patch**: `tilelang_metal_fp8_scaled_matmul_fused_scheduler`. |
| FP8 scaled vecmat (M=1 N=K=4096 e4m3) | ✅ 1.098 ms (Path B 0.182 ms — 6.01× slower) | Same as matmul + simdgroup_sum reduction for M=1 specialization. Path B uses hand-written `simd_sum`; DSL needs a `T.simdgroup_reduce_sum` primitive. **NEW patch**: extend the fused scheduler with simdgroup-sum specialization. |

---

## Suggested upstream patch roadmap (ordered by ROI)

When we file these as new `docs/upstream/` artifacts, all stack on top of
`jorgecurious/tilelang:metal-gemm-upstream-rebase` (PR #2130) plus our existing
`tilelang_metal_pipelined`, `tilelang_metal_fp8`, `tilelang_metal_fp8_scaled_matmul`
patches. Each new patch adds another link in the dependency chain.

| # | Patch | Unblocks | Effort | ROI |
|---|---|---|---|---|
| **A** | `tilelang_metal_pipelined_32x32` (extend 3D buffer fix to 32×32 fragments) | Sparse-MLA fwd BF16 + FP8 (Path C parity with B) | medium (~150 LOC, touches metal_macro_generator + dispatch) | **HIGH** — unblocks 2 sparse-MLA ports |
| **B** | `tilelang_metal_fp8_scaled_matmul_fused_scheduler` (fuse per-load scale into GemmMetalScalar K-loop) | FP8 matmul/vecmat speed parity with audiohacking | medium-high (~250 LOC, scheduler pass) | **MEDIUM** — quality-of-life, Path B already wins |
| **C** | `tilelang_metal_blockscaled_e8m0` (DSL primitive for e8m0 block-scale layout) | Sparse-MLA blockscaled fwd via Path C | high (~350 LOC, language + lowering) | **MEDIUM** — only useful for FP8 path |
| **D** | `tilelang_metal_simdgroup_reduce` (T.simdgroup_reduce_sum primitive) | FP8 vecmat M=1 specialization, topk_selector | medium (~200 LOC) | **LOW** — Path B already optimized, M0 doesn't need topk via DSL |
| **E** | `tilelang_metal_chunked_bwd` (chunked backward flow with explicit accumulation) | Sparse-MLA bwd via Path C | very high (language-level extension) | **LOW** — Path B already chunks outside autograd; DSL would just match it |

Total estimated effort: A is the only must-do for Path C parity on the
production-shape kernels. B-D are quality-of-life. E is language-design
work and shouldn't block the M0 timeline.

---

## What this means for current PR filing

The 3 rebase agents (mixed_dtype, fp8_gemm, fp8_vector) are putting our
existing patches on top of `jorgecurious/tilelang:metal-gemm-upstream-rebase`.
**These are not the end of the story** — patches A through E above are
expected follow-ups when we extend Path C beyond Mamba3.

The filing pack should explicitly say:
- "Current 8 patches are the foundation; future Path C ports surface new gaps"
- "Each follow-up will be a fresh `docs/upstream/<name>/` directory with its own README + patch file"
- "All of them stack on PR #2130 (or whatever upstream Apple Metal PR series eventually merges)"

---

## How to update this tracker

When a new Path C port hits a TileLang gap:

1. Add a row under "Ops where Path C is blocked" with the specific blocker
2. If the unblock is identifiable as a single patch, add to the roadmap with
   estimated effort and ROI
3. When the patch lands locally, update `docs/production_kernel_routing.md`
   bench scoreboard and move the row from "blocked" to "shipped"
4. If the patch ends up filed upstream, link it from the README in
   `docs/upstream/<patch_name>/`

The tracker exists so we don't lose track of what TileLang DSL needs as we
extend coverage — every Path C "❌" in the production table should map back
to a specific gap explained here.

---

## Currently expected dependency chain (top to bottom)

```
upstream tile-ai/tilelang main
  ↑
  PR #1869 (oraluben metal-gemm)
  ↑
  PR #2118 (cklxx metal-gemm-scalar-fallback)
  ↑
  PR #2121 (SiriusNEO multi-backend codegen refactor)
  ↑
  PR #2130 (jorgecurious metal-gemm-upstream-rebase)  ← REBASE TARGET
  ↑
  Our: tilelang_gemm_mixed_dtype       (rebased 2026-05-04)
  Our: tilelang_metal_pipelined        (clean apply on PR #2130)
  Our: tilelang_metal_fp8              (clean apply on PR #2130)
  Our: tilelang_metal_fp8_gemm         (rebased 2026-05-04)
  Our: tilelang_metal_fp8_vector       (rebased 2026-05-04, depends on tilelang_metal_fp8)
  Our: tilelang_metal_fp8_scaled_matmul (clean apply on PR #2130, frontend macro)
  Our: tilelang_metal_shared_dyn       (no-op investigation artifact, not a code PR)
  ↑ FUTURE
  Patch A: tilelang_metal_pipelined_32x32           (when sparse-MLA Path C lands)
  Patch B: tilelang_metal_fp8_scaled_matmul_fused_scheduler  (when FP8 perf parity matters)
  Patch C: tilelang_metal_blockscaled_e8m0          (when blockscaled Path C lands)
  Patch D: tilelang_metal_simdgroup_reduce          (when topk Path C lands)
  Patch E: tilelang_metal_chunked_bwd               (DSL language extension, deferred)
```
