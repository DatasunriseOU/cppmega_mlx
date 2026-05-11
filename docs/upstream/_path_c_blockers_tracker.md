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
      available, then falls back to Path B and pure MLX.<br>
      The direct float32/float16 <code>topk_selector_tilelang_direct(..., out=...)</code>
      route is tvm-ffi owner-output and mutates caller-owned
      <code>mx.int32</code>; direct bf16 fails closed.</td>
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
      C/B 0.975, indexed QK-reduce C/B 0.655), but they are not<br>
      the strict full-dispatch gate. Standalone scaled matmul/vecmat<br>
      parity is now closed in the TileLang Metal worktree. Full<br>
      Sparse-MLA FP8 forward composition is wired as a prepared-buffer<br>
      DSA A-layer route; the strict training gate remains red until<br>
      prepared-buffer backward/update coverage lands.</td>
      <td>Keep the real packed-dot4 QK reducer as the dispatchable partial<br>
      path, then add full Path C FP8 fwd/bwd composition using the<br>
      direct-global packed-dot4 scaled-matmul lowering as the<br>
      building block.</td>
    </tr>
    <tr>
      <td>FP8 Path C end-to-end training dtype route</td>
      <td>scripts/m04_train_step.py accepts dtype=fp8_path_c as an<br>
      explicit optimizer-matrix route. DSA A-layers now produce<br>
      prepared q_fp8/q_scale/kv_fp8/kv_scale tensors before calling<br>
      Sparse-MLA Path C. The remaining public Path C gap is M&gt;1<br>
      T.fp8_scaled_matmul as an MLX custom_function/VJP training op.<br>
      Adapters must not quantize/copy large bf16/fp32 tensors into<br>
      temporary FP8 staging buffers.</td>
      <td>Add a differentiated MLX-callable M&gt;1 FP8 Path C matmul that<br>
      consumes existing FP8 GPU buffers, then extend the prepared-buffer<br>
      ownership pattern from DSA activations to trainable parameters,<br>
      absorbed MLA KV layout, and backward/update kernels.</td>
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

The owner-output closure claim is now narrow but real:
`topk_selector_tilelang_direct(..., out=...)` lowers the TileLang DSL route
through tvm-ffi for float32/float16 scores, mutates the caller-provided
`mx.int32` output, and returns that same owner array. Direct bf16 dispatch
fails closed to avoid hidden casts. This does not mean public
`topk_selector(..., backend="auto")` is a generic owner-output route: AUTO is
still receipt-gated and compatibility wrapper paths remain in the public Path C
surface.

### FP8 ops (standalone correctness useful; current production speed blocked)

These technically work via Path C, but the current TL-W production gate is
performance, not just compile/parity. The hand-written Metal reference split
matters: `/tmp/fp8-mps-metal` provides byte FP8 decode, 4-way K unroll,
scale-after-accumulated-dot, and the M==1 `simd_sum` vecmat reducer; the
cppmega.mlx/AppMana fast path adds packed `uint32` loads plus LUT-backed dot4
decode. Older packed-dot4 probe receipts showed parity or speed parity in
isolated harnesses. Do not use those as the current production gate.

As of the 2026-05-11 TL-W owner-output/tvm-ffi path, MLX Metal tensors can
cross DLPack as `kDLMetal:0` and a standalone TVM Metal kernel can read/write
the same buffers. That proves the bridge substrate. The current production
timing is still red: FP8 matmul is about 14x slower than the shipped
Path B/audiohacking-style MSL route, and vecmat is about 1.7x slower. The m04
end-to-end FP8 training route also still needs graph wiring over prepared FP8
buffers with no hidden large tensor staging.

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
      <td>❌ current TL-W owner-output/tvm-ffi route is about 14x slower<br>
      than the shipped Path B MSL route.</td>
      <td>Keep Path B for production. Rework the real packed-dot4<br>
      scheduler/codegen path until the current strict receipt is no worse,<br>
      then reconnect it to full Sparse-MLA FP8 fwd/bwd composition.</td>
    </tr>
    <tr>
      <td>FP8 scaled vecmat (M=1 N=K=4096 e4m3)</td>
      <td>❌ current TL-W owner-output/tvm-ffi route is about 1.7x slower<br>
      than the shipped M=1 Path B simd_sum reducer.</td>
      <td>Keep Path B for production. Rework the direct-global vecmat<br>
      lowering and wrapper boundary until strict current receipts are no<br>
      worse and stable across paired p90/p99 samples.</td>
    </tr>
  </tbody>
</table>

---

## Suggested upstream patch roadmap (artifact order)

The local docs/upstream/ artifact and probe directories track the three Path C
follow-ups below. Patch A shipped locally; patches B/C were filed as #2146/#2147,
but B's original performance framing was wrong. The local B artifact is now a
documentation-only tombstone until a real scheduler/codegen implementation
replaces it. All real upstream code patches stack on top of
jorgecurious/tilelang:metal-gemm-upstream-rebase (PR #2130) plus our existing
tilelang_metal_pipelined, tilelang_metal_fp8, tilelang_metal_fp8_scaled_matmul
patches. Each patch adds another link in the dependency chain.

2026-05-05 update: the local sparse FP8 QK reducer no longer depends on the
legacy scaled-matmul macro probe for M==1/topk. It lowers through the active
packed `__tvm_fp8_e4m3_dot4_packed` Metal path, applies scales after the dot,
uses a finite fp32-min sentinel for invalid indices because current TileLang
Metal cannot lower `T.infinity`, and the refreshed M4 Max receipt reports
QK-reduce C/B=0.975 and indexed-QK C/B=0.655 with zero invalid-index mismatch.
2026-05-11 TL-W update: those older standalone scaled-matmul/vecmat receipts
are superseded for production planning by the owner-output/tvm-ffi gate, where
FP8 matmul is still about 14x slow and vecmat about 1.7x slow. The
full-dispatch sparse FP8 gate intentionally stays red until fwd/bwd composition
is implemented and current strict receipts are green.

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
      <td>tilelang_metal_fp8_scaled_matmul_fused_scheduler (currently a<br>
      tombstone for the retired algebraic patch; replacement is the<br>
      active packed FP8 Metal scheduler/codegen path)</td>
      <td>Standalone FP8 matmul/vecmat speed parity and the building<br>
      block for Sparse-MLA FP8 Path C composition</td>
      <td>medium/high (first recover current owner-output/tvm-ffi speed,<br>
      then continue into full Sparse-MLA FP8 fwd/bwd composition)</td>
      <td>**HIGH** — remaining full-composition Sparse-MLA FP8 blocker</td>
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

The 3 rebase agents (mixed_dtype, fp8_gemm, fp8_vector) put the foundation
patches on top of jorgecurious/tilelang:metal-gemm-upstream-rebase. **These are
not the end of the story** — local artifact/probe directories for patches A
through C track the remaining Path C coverage. A shipped locally; B/C were filed,
but B's local artifact is now only a tombstone for the retired algebraic
scaled-operands patch. It is not applyable scheduler/codegen work.

The filing pack should explicitly say:
- "Current filed patches are the foundation; Path C ports surfaced A/B/C follow-up gaps"
- "Each A/B/C follow-up has a local docs/upstream/<name>/ artifact/probe directory with current status"
- "Patch B's #2146 macro rewrite is retired; packed-dot4 probes remain useful diagnostics, but current owner-output/tvm-ffi FP8 speed parity is still red and full Sparse-MLA FP8 composition still needs filing"
- "All of them stack on PR #2130 / #2142 (or whatever upstream Apple Metal PR series eventually merges)"

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
  ↑ LOCAL ARTIFACTS / PROBES EXIST; TRACK CURRENT FOLLOW-UP STATUS
  Patch A: tilelang_metal_pipelined_32x32           (local shipped)
  Patch B: tilelang_metal_fp8_scaled_matmul_fused_scheduler  (FILED #2146, RETIRED LOCALLY 2026-05-04 — the algebraic scaled-operands patch is not applyable here and did not close the local Path B FP8 MSL gap. The replacement local TileLang worktree path is scale-after-dot plus packed uint32/LUT dot4 decode, 4-way K stride, direct-global full matmul, and simd_sum vecmat reduction. As of TL-W, the owner-output/tvm-ffi production gate is still red: matmul about 14x slow and vecmat about 1.7x slow. Recover that speed before composing it into Sparse-MLA FP8 fwd/bwd.)
  Patch C: tilelang_metal_blockscaled_e8m0          (FILED #2147; keep apply checks against the active prereq stack)
