# Path C blockers — what TileLang upstream needs for full Apple Silicon parity

Date: 2026-05-04

This document tracks what TileLang DSL features are still missing so Path C
(@T.prim_func lowered through patched apple-head TileLang on Metal target)
can match Path B (hand-written MSL via mx.fast.metal_kernel) for the
remaining ops in cppmega_mlx/nn/_tilelang/.

**Why this matters:** every Path C port we add may surface a new TileLang
scheduler/codegen gap. When that happens, we file a fresh upstream patch
under docs/upstream/, stack it on top of the existing jorgecurious/tilelang:metal-gemm-upstream-rebase
chain (PR #2130), and update this tracker.

The Path B/C bench table in docs/production_kernel_routing.md is the
authoritative scoreboard. This doc explains the **why** for each ❌.

---

## Current Path C dispatch status

<table>
  <thead>
    <tr>
      <th>Op</th>
      <th>Bench receipt (M4 Max)</th>
      <th>Dispatch status</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>**topk_selector**</td>
      <td>C/B ratios 0.510-0.880 across the checked shapes</td>
      <td>✅ production AUTO</td>
      <td>topk_selector(..., backend="auto") prefers Path C where<br>
      available, then falls back to Path B and pure MLX.</td>
    </tr>
    <tr>
      <td>**Mamba3 main fwd+bwd** (B=2 T=512 H=4 P=32 N=64 FP32)</td>
      <td>C 7.707 ms vs B 7.823 ms</td>
      <td>✅ proof/override</td>
      <td>Bit-exact parity with B, 1.5% faster (within noise). Path B<br>
      remains the production route; Path C is kept as a DSL-path<br>
      reproducibility receipt.</td>
    </tr>
    <tr>
      <td>**Sparse-MLA BF16 fwd+bwd**</td>
      <td>paired C/B: B2_S128_H8_D64 fwd 0.993 / bwd 0.913;<br>
      B4_S512_H8_D64 fwd 0.993 / bwd 0.975; B4_S1024_H8_D64 fwd<br>
      0.994 / bwd 0.997</td>
      <td>⚠️ per-shape AUTO</td>
      <td>sparse_mla_path_c.py lowers TileLang DSL to scalar MSL and<br>
      mirrors Path B lane loops. AUTO promotes checked rows only<br>
      when the receipt shape matches and all required fwd+bwd no-<br>
      worse gates are true; today all three checked BF16 rows<br>
      promote to Path C.</td>
    </tr>
  </tbody>
</table>

---

## Ops where Path C is blocked

### Sparse-MLA family (BF16 AUTO is per-shape; FP8/e8m0 still blocked)

The old "no BF16 Path C exists" blocker is closed in cppmega.mlx: the in-tree
Path C port does not wait on upstream T.gemm/T.Pipelined Sparse-MLA
templates. It uses scalar TileLang DSL, lowers through apple-head Metal,
postprocesses the MSL back to Path-B-style lane loops, and keeps the same
dkv_partial backward contract.

AUTO is intentionally per-shape and fail-closed. The checked-in
bench/tilelang_ports/sparse_mla.json receipt covers forward and backward
(strict.phase="all", fwd_only=false), so it promotes only rows that match
the requested shape and have every required fwd+bwd no_worse_than_path_b gate
set. The current receipt promotes B2_S128_H8_D64, B4_S512_H8_D64, and
B4_S1024_H8_D64; unreceipted shapes stay on Path B.

<table>
  <thead>
    <tr>
      <th>Op</th>
      <th>Path C blocker</th>
      <th>What patch unblocks it</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sparse-MLA BF16 unreceipted shapes</td>
      <td>The three checked fwd+bwd rows promote, but no receipt<br>
      proves other shape/topk combinations.</td>
      <td>Regenerate bench/tilelang_ports/sparse_mla.json with the new<br>
      shape, then update dispatch tests and docs before widening<br>
      AUTO promotion.</td>
    </tr>
    <tr>
      <td>Sparse-MLA FP8 fwd/bwd</td>
      <td>Partial qk reducers have parity/timing receipts (QK-reduce<br>
      C/B 0.864, indexed QK-reduce C/B 0.696), but they are not<br>
      the strict full-dispatch gate. Full Path C QK remains<br>
      unavailable, so sparse_mla_fp8.json keeps the full Path C<br>
      gate red.</td>
      <td>Extend the FP8 scaled-matmul scheduler so per-load scale<br>
      fuses into the K loop, then add a full Sparse-MLA FP8 Path C<br>
      wrapper/test pair that mirrors BF16 dkv_partial.</td>
    </tr>
    <tr>
      <td>Sparse-MLA blockscaled fwd/bwd (e8m0)</td>
      <td>Partial e8m0 QK-reduce has a parity/timing receipt (C/B<br>
      0.4364), but it is not the strict full-dispatch gate. Full<br>
      Path C QK remains unavailable, so<br>
      sparse_mla_blockscaled.json keeps the full Path C gate red.</td>
      <td>Introduce T.BlockScaledLayout.e8m0_k32() /<br>
      scale_format="e8m0_block_k32" API and lower block_size=32<br>
      through the Metal scalar path with matching scale-tile bookkeeping; then mirror the<br>
      BF16 partial-backward contract. Larger work — likely 200-400<br>
      LOC.</td>
    </tr>
  </tbody>
</table>

### Topk + selector follow-up (not a dispatch blocker)

topk_selector is not blocked for production dispatch: the current AUTO path
uses Path C when available, and the checked-in receipt keeps every C/B ratio
<= 1.0. A future T.simdgroup_reduce primitive can still improve reduction
ergonomics and FP8 vecmat specialization, but it is not required to select
Path C for topk today.

### FP8 ops (2 ops, slower than B)

These technically work via Path C but are 3-6× slower than Path B's
audiohacking-style hand-written MSL. The win path is **fused scheduler**.

<table>
  <thead>
    <tr>
      <th>Op</th>
      <th>Current Path C state</th>
      <th>What's needed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FP8 scaled matmul (128³ e4m3 per-tensor)</td>
      <td>✅ 0.555 ms (Path B 0.172 ms — 3.16× slower)</td>
      <td>Fuse the per-load scale broadcast into the GemmMetalScalar<br>
      K-loop. The audiohacking kernel does this; our DSL path<br>
      emits a post-loop multiply. **NEW patch**:<br>
      tilelang_metal_fp8_scaled_matmul_fused_scheduler.</td>
    </tr>
    <tr>
      <td>FP8 scaled vecmat (M=1 N=K=4096 e4m3)</td>
      <td>✅ 1.098 ms (Path B 0.182 ms — 6.01× slower)</td>
      <td>Same as matmul + simdgroup_sum reduction for M=1<br>
      specialization. Path B uses hand-written simd_sum; DSL needs<br>
      a T.simdgroup_reduce_sum primitive. **NEW patch**: extend<br>
      the fused scheduler with simdgroup-sum specialization.</td>
    </tr>
  </tbody>
</table>

---

## Suggested upstream patch roadmap (artifact order)

The local docs/upstream/ artifact and probe directories already exist for the
three Path C follow-ups below. They still need apply-to-apple-head verification
and upstream filing. All stack on top of
jorgecurious/tilelang:metal-gemm-upstream-rebase (PR #2130) plus our existing
tilelang_metal_pipelined, tilelang_metal_fp8, tilelang_metal_fp8_scaled_matmul
patches. Each patch adds another link in the dependency chain.

<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Patch</th>
      <th>Unblocks</th>
      <th>Effort</th>
      <th>ROI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>**A**</td>
      <td>tilelang_metal_pipelined_32x32 (keep simdgroup storage<br>
      scalar for 32x32 pipelined Metal kernels)</td>
      <td>32x32 pipelined TileLang Metal lowering for Path C follow-<br>
      up probes</td>
      <td>medium (~100 LOC, storage rewrite)</td>
      <td>**HIGH** — first artifact/probe in the Path C follow-up chain</td>
    </tr>
    <tr>
      <td>**B**</td>
      <td>tilelang_metal_fp8_scaled_matmul_fused_scheduler (fuse per-<br>
      load scale into GemmMetalScalar K-loop)</td>
      <td>FP8 matmul/vecmat speed parity and Sparse-MLA FP8 Path C<br>
      composition</td>
      <td>medium-high (~250 LOC, scheduler pass)</td>
      <td>**HIGH** — remaining Sparse-MLA FP8 blocker</td>
    </tr>
    <tr>
      <td>**C**</td>
      <td>tilelang_metal_blockscaled_e8m0 (DSL primitive for<br>
      e8m0_block_k32 block-scale layout)</td>
      <td>Sparse-MLA blockscaled fwd/bwd via Path C</td>
      <td>high (~350 LOC, language + lowering)</td>
      <td>**MEDIUM** — needed for blockscaled Path C</td>
    </tr>
  </tbody>
</table>

BF16 Sparse-MLA no longer needs a TileLang language extension in this repo for
the checked fwd+bwd rows, and those rows already have Path C AUTO coverage. The
remaining BF16 work is widening the receipt matrix without dropping the
fail-closed row gate. The remaining upstream work is FP8 scheduler/layout
coverage plus optional reduction ergonomics.

---

## What this means for current PR filing

The 3 rebase agents (mixed_dtype, fp8_gemm, fp8_vector) are putting our
existing patches on top of jorgecurious/tilelang:metal-gemm-upstream-rebase.
**These are not the end of the story** — local artifact/probe directories for
patches A through C above already exist, but they still need apply-to-apple-head
verification and upstream filing before they become upstreamed Path C coverage.

The filing pack should explicitly say:
- "Current 8 patches are the foundation; Path C ports surfaced A/B/C follow-up gaps"
- "Each A/B/C follow-up has a local docs/upstream/<name>/ artifact/probe directory, but still needs apply-to-apple-head verification and upstream filing"
- "All of them stack on PR #2130 (or whatever upstream Apple Metal PR series eventually merges)"

---

## How to update this tracker

When a new Path C port hits a TileLang gap:

1. Add a row under "Ops where Path C is blocked" with the specific blocker
2. If the unblock is identifiable as a single patch, add to the roadmap with
   estimated effort and ROI
3. When the patch lands locally, update docs/production_kernel_routing.md
   bench scoreboard and move the row from "blocked" to "shipped"
4. If the patch ends up filed upstream, link it from the README in
   docs/upstream/<patch_name>/

The tracker exists so we don't lose track of what TileLang DSL needs as we
extend coverage — every Path C "❌" in the production table should map back
to a specific gap explained here.

---

## Currently expected dependency chain (top to bottom)


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
  ↑ LOCAL ARTIFACTS / PROBES EXIST; NEED APPLY-TO-APPLE-HEAD + UPSTREAM FILING
  Patch A: tilelang_metal_pipelined_32x32
  Patch B: tilelang_metal_fp8_scaled_matmul_fused_scheduler  (when FP8 sparse-MLA Path C lands)
  Patch C: tilelang_metal_blockscaled_e8m0          (when blockscaled Path C lands)
