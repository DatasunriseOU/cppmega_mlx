# Path C follow-up upstream PR candidates

Date: 2026-05-04

The Path C TileLang DSL ports (`sparse_mla_path_c.py`, `sparse_mla_blockscaled_path_c.py`, `sparse_mla_fp8_path_c.py`, `fp8_vecmat_path_c.py`, `mamba3_path_c.py`) surfaced **TileLang Metal codegen workarounds** that cppmega.mlx currently implements as MSL string-substitution post-processing in `cppmega_mlx/nn/_tilelang/_msl_transform.py`. These workarounds are evidence of upstream codegen gaps — fixing them in TileLang itself removes the post-processing pass and makes the same DSL produce correct MSL for every consumer.

## Filed already (2026-05-04)

These were filed as PRs #2139-#2142:
- `tilelang_gemm_mixed_dtype` (#2139)
- `tilelang_metal_fp8_gemm` (#2140)
- `tilelang_metal_pipelined` (#2141)
- `tilelang_metal_fp8_scaled_matmul` (#2142)

## New candidates surfaced from Path C work (NOT yet filed)

### Candidate #1 — `[Metal] emit kernel body directly instead of inline void wrapper`

**Bug:** TileLang's metal codegen emits the prim_func body as an `inline void` helper called from the entry-point kernel. Apple's MSL **forbids** `threadgroup` allocations in non-kernel functions. Any prim_func with `T.alloc_shared` ends up unable to compile through `xcrun --sdk macosx metal -c`.

**cppmega workaround:** `cppmega_mlx/nn/_tilelang/_msl_transform.py::_inline_tilelang_kernel_body`. Strips the `inline void` wrapper and inlines the body verbatim into the kernel function. Module docstring (lines 13-15) explicitly calls this out:

> "Body extraction is done inline rather than wrapping into ``inline void`` because Apple's MSL forbids ``threadgroup`` allocations in non-kernel functions."

**Upstream fix:** in TileLang metal codegen (`src/target/codegen_metal.{cc,h}` and/or its `3rdparty/tvm/src/target/source/codegen_metal.{cc,h}` mirror), when the prim_func body contains `T.alloc_shared` / `T.alloc_fragment` declarations, emit the body inline in the kernel function instead of wrapping it. Detect threadgroup-using bodies during PrintFunctionSignature and short-circuit the inline-void path.

**Estimated effort:** ~80-150 LOC C++ change. Localized to PrintFunctionSignature/PrintStmt in `codegen_metal.cc`. Tested via any prim_func with `T.alloc_shared` followed by `xcrun --sdk macosx metal -c`.

**ROI:** **HIGH** — every Apple Silicon TileLang user hits this. cppmega has 5 Path C ports that rely on the workaround.

**Filing target:** `tile-ai/tilelang` main, stacked on PR #2130 (jorgecurious metal-gemm-upstream-rebase) like our other tilelang PRs.

### Candidate #2 — `[Metal] emit Metal builtins directly instead of TileLang threadIdx/blockIdx aliases`

**Bug:** TileLang's metal codegen emits intermediate aliases like:

```cpp
int blockIdx_x = ((int)threadgroup_position_in_grid.x);
int threadIdx_x = ((int)thread_position_in_threadgroup.x);
```

These aliases mirror CUDA's `blockIdx.x` / `threadIdx.x` notation but aren't required on Metal — the kernel can just reference `threadgroup_position_in_grid.x` directly. The aliases:
- Add unused declarations when the alias is dead (TileLang doesn't strip them)
- Force `(int)` casts that mask `uint` semantics from MSL
- Complicate downstream MSL passes that need to identify thread-position references

**cppmega workaround:** 4 helpers in `_msl_transform.py`:
- `_metal_builtin_for_tilelang_alias` — maps alias → Metal builtin
- `_rewrite_tilelang_builtin_axis` — rewrites unbracketed reference
- `_rewrite_tilelang_builtin_axis_cast` — rewrites cast form
- `_canonicalize_tilelang_builtin_aliases` — orchestrates both
- `_drop_alias_decl_if_unused` — removes unused alias declarations

**Upstream fix:** in TileLang metal codegen, when emitting references to threadIdx / blockIdx, emit the Metal builtin directly without the intermediate alias. The CUDA-compat alias is only needed for codegen targets that mirror CUDA syntax — Metal isn't one.

**Estimated effort:** ~50-100 LOC C++ change. Touches the threadIdx/blockIdx emit path in `codegen_metal.cc`.

**ROI:** **MEDIUM** — quality-of-life. Doesn't unblock new functionality; just removes need for downstream string post-processing. Worth filing because the fix is small.

**Filing target:** `tile-ai/tilelang` main, stacked on PR #2130.

### Candidate #3 — `[Metal] elide redundant bounds-check on masked gather loads`

**Bug:** TileLang generates `if (0 <= idx && idx < N) { load } else { 0.0 }` guards around gather loads even when the producer guarantees masked-invalid indices. For sparse-MLA hot loops this adds branchy code in inner loops.

**cppmega workaround:** `_remove_redundant_kv_bounds_checks` and `_remove_redundant_flat_kv_bounds_checks` regex passes in `sparse_mla_path_c.py` (lines 38-83). They detect the guard pattern and replace with unconditional load, asserting that the producer (cppmega's index masking) has already filtered invalid entries.

**Upstream fix:** This is **algorithm-specific** (sparse-MLA's index-mask contract). Not a clean generic codegen fix. **Don't file** as upstream — it's downstream consumer optimization. Documented here for completeness.

### Candidate #4 — partial-receipt gaps (track for later)

The Path C blocker tracker (`docs/upstream/_path_c_blockers_tracker.md`) already lists 5 future patches (A-E). Three of them are concrete and worth filing once we have a working scheduler/layout implementation:

- **A** `tilelang_metal_pipelined_32x32` — extend our PR #2141 fix to 32×32 fragments. Needed for full sparse-MLA Path C parity.
- **B** `tilelang_metal_fp8_scaled_matmul_fused_scheduler` — scheduler pass that fuses per-load scale into the GemmMetalScalar K-loop, closing the 3-6× gap to audiohacking MSL.
- **C** `tilelang_metal_blockscaled_e8m0` — DSL primitive for e8m0 block-scale layout.

These are larger pieces of work than candidates #1 and #2 — they need real implementation, not just patches.

## Filing recommendation

**Now:** Candidates #1 and #2 are ready-to-write — the cppmega workaround code is the rosetta stone for what the upstream fix should look like. Each is ~100 LOC C++ in TileLang's metal codegen.

**Later:** Candidates A, B, C from the tracker need design + implementation work, not just patch translation.

**Don't file:** Candidate #3 (sparse-MLA bounds-check) is too algorithm-specific.

## Why these aren't already filed

The 6 PRs we filed today (`mlx_from_dlpack`, `tvm_shared_storage`, mixed-dtype, fp8_gemm, pipelined, fp8_scaled_matmul) cover the **codegen blockers we discovered before Path C work**. The Path C ports were mostly added in commits `62b11c5..6d100c6` — after the original filing pack was assembled. Candidates #1 and #2 are the new artifacts from Path C ports surfacing additional gaps.

When filed, they should reference cppmega.mlx's `_msl_transform.py` as the workaround proof, and link to the Path C ports (sparse_mla_path_c.py, etc.) as consumers that demonstrate the gap matters in practice.

## Tracker entry

When candidates #1 and #2 are filed:

1. Create `docs/upstream/tilelang_metal_inline_kernel_body/` and `tilelang_metal_emit_metal_builtins/` directories with patch + README each.
2. Update `_filed_prs_2026_05_04.md` (or new dated receipt) with PR URLs.
3. Update `_pr_filing_pack.md` reality-check header to add the 2 new PRs.
4. After both land, remove the corresponding workarounds from `_msl_transform.py` and update Path C ports accordingly.
