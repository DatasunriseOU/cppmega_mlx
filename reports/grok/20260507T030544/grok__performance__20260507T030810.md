---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 5b83522a14065d99a23788c3ac47f4e8daa70be6
timestamp: 2026-05-07T03:08:10.808812+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/test_tilelang_topk.py']
---
**Performance review of the wave-3 topk_selector changes (fix-round-3: 6b74ec7 + 5b83522) and related Path C / MSL wiring in the bundle.** Focus is on regressions vs. the buggy-but-fast prior state, hot-path issues, and concrete opportunities. The insertion-sort early-break fix in Path C is **correct** (standard `else: break` after shift; no other corruptions visible in the lowered body or tests). The prior heuristic (`elif (K<=8) or (K>=64): break`) skipped the break for K=16/32, allowing over-shifting and corrupting the maintained sorted invariant of `local_vals` (ascending, root = smallest of top-K). This produced wrong outputs on some inputs—**correctness win**, but the ~1.28–1.5× slowdown at K=32 (relative to Path B) is expected and now properly gated by `_PATH_C_AUTO_PROFITABLE_RECEIPTS` (excludes the 1,512,32 case). No evidence of other "fast because broken" optimizations in the reviewed files.<grok:render card_id="c76bea" card_type="citation_card" type="render_inline_citation"><argument name="citation_id">16</argument></grok:render>

### 1. Critical Performance Regressions / Concerns (Path C topk)

**topk_selector.py:1001–1013 (insertion loop in `_path_c_kernel_for` / TileLang PrimFunc)**  
The fixed loop is now a true linear scan in the worst case for each of the `seq_len / threads` candidate insertions:

```python
for p in T.serial(1, _TOPK_C_K):  # K up to 256
    if value > local_vals[p]:
        ... shift ...
        pos = p
    else:
        break
```

- **Impact**: For K=256 and adversarial (or near-sorted descending) inputs, this is O(K) per insertion → O(seq_len * K / threads) per row in the worst case. With `threads ≈ 32–64` and seq_len=4096, this is noticeable vs. the prior buggy early-exit. The buggy version accidentally limited shifts for mid-K, making it faster on the bench shapes.  
- The Path B direct-MSL kernel (`_TOPK_SOURCE:422–438`) uses an **identical** insertion loop (fixed K unroll comment claims compiler unrolling, but still O(K) shifts). Both suffer the same asymptotic.  
- **Quantified from bundle**: topk_selector.json shows Path C still wins overall (0.56–0.92× Path B ratio across receipts), but the K=32 regression is real. The 4096×256 row (4.96 ms Path C vs. 8.78 ms Path B) is dominated by the merge phase, not insertion.  

**Recommendation (high priority)**: Replace the insertion with a **binary search + memmove-style shift** (or maintain a max-heap of size K). For register-resident K≤256 on Apple Silicon, even a simple reverse scan from the end (finding insertion point in O(log K) then shift) would help. Or switch to a priority-queue pattern using registers + conditional moves. This would recover most of the "lost" perf without re-introducing the bug. Test with adversarial inputs (reverse-sorted or many near-boundary values).

**topk_selector.py:1022–1051 (tree-merge loop)** and **_path_c_rewrite_merge_round:845–915**  
- The rewrite removes one pre-write `threadgroup_barrier` and tightens the active-lane condition (`% (stride*2) == 0` → bitwise `& ((stride*2)-1) == 0`). This is a **good micro-optimization** for the log(threads) rounds.  
- **Fragility risk (performance regression vector)**: The regex-based rewrite (`body.count`, `replace`, `re.sub` with hardcoded patterns like `for (int i_2 = 0;`) is extremely brittle. If TileLang's MSL emission changes whitespace, variable naming, or barrier placement (very likely with Z3 passes or scheduler tweaks), the rewrite silently no-ops (`if body == lowering.body: return lowering`). You fall back to TileLang's more conservative emission (extra barriers), introducing **silent performance regression** on future TileLang updates.  
- The comment at 874–875 acknowledges this. The `test_path_c_lowering_uses_single_merge_active_branch` pins the current emission but won't catch emission drift.  
- **Impact**: Extra barriers in the inner merge rounds (especially at threads=64) hurt on Apple TBDR GPUs, where threadgroup sync has non-trivial cost even if lightweight.<grok:render card_id="334dca" card_type="citation_card" type="render_inline_citation"><argument name="citation_id">9</argument></grok:render>

**Actionable**: 
1. Add a loud assertion in `_path_c_rewrite_merge_round` (and in the test) that at least one replacement occurred, or compute a checksum of the targeted pattern.
2. Long-term: push the tighter merge scheduling into TileLang (intra-warp barrier elision or a custom TIR pass via the `pass_configs` hook already wired). The Z3 roadmap mentions related ideas (#11).

**topk_selector.py:735–748 (`_path_c_threads_for`)** and Path B cap logic (`topk_selector_metal:628–637`)  
Both cap threadgroup size based on `max_shared_bytes = 32 * 1024` and `K*8` bytes for the pair/merge buffers. This is correct for current M-series (32 KiB threadgroup memory limit). However:  
- It uses a conservative static 32 KiB; real devices report via `MTLDevice.maxThreadgroupMemoryLength` (often 32 KiB, sometimes higher on certain families).  
- Power-of-two preference is good for the tree reduction, but the cap `min(64, ...)` for K=256 forces smaller groups than theoretically possible if spilling is managed.  
- No dynamic query of device limits inside the hot path (good, but the lru_cache on `_path_c_kernel_for` already amortizes).  

Minor: the pair buffer sizing in Path B (`PAIR_BUF = block_size * K`) and Path C shared allocs are duplicated logic—risk of divergence.

### 2. Sparse-MLA Path C & MSL Wiring (sparse_mla.json + _msl_transform.py)

**sparse_mla.json rows (esp. B4_S1024_H8_D64)**:  
- Forward Path C is now ~1.00–1.03× Path B (paired ratio 1.0018 on the failing row); backward mixed (paired 0.95–1.08, some kernel-only ratios >1.0). The "strict gate failed" entries and `bwd_blocker: "mixed overhead"` indicate Path C is no longer strictly better everywhere.  
- This is **not a regression** per se (strict policy allows it), but the wiring (including any canonicalize/hoist changes) has not delivered the expected win on the largest shape. The comment in sparse_mla.json line 22/23 flags the exact paired ratios.

**_msl_transform.py:460–471 (`_canonicalize_tilelang_builtin_aliases`)** and related rewrites:  
- The alias rewriting + decl stripping is solid for compatibility. No obvious hot-path cost (done once at lowering time).  
- The `_rewrite_msl_code_segments` (preserves comments/strings) and comment-masking helpers are O(N) but on small kernel source—negligible.  

No new O(n²) or allocation-in-loops visible. The Z3 `pass_configs` threading (`tl.drop_provable_bound_checks`, `tl.simd_lift_reductions`) is correctly filtered and cached—good.

**_msl_transform.py:844– (dispatch / lowering)**:  
- `metal_grid_for_lowering` multiplies grid × threadgroup extents. This is correct but can produce very large grids for high-occupancy kernels; ensure MLX `mx.fast.metal_kernel` handles it without host-side overhead (it should).

**General MSL / TileLang Path C concerns**:  
- Multiple lowers (`_path_c_kernel_for` lru_cache(maxsize=128)`) and shape specialization are excellent for avoiding repeated TVM lowering cost.  
- FP8 intrinsic registration and `_assert_path_c_metal_fp8_intrinsics_registered` are defensive and off the hot path.  
- No redundant I/O, blocking calls, or memory growth patterns in the provided files. The `dsa_splitk_indexer_loss.py` (truncated) mentions similar block-size tuning for Metal's 32 KiB limit—consistent and correct.

### 3. Other Hot-Path / Regression Risks

- **Caching & lru**: Excellent use (`_path_c_kernel_for`, `_tilelang_available`, status funcs). No N+1 lowering issues.
- **Reference fallback**: Pure-MLX `topk_selector_reference` uses `mx.argpartition` / `argsort`—fine for small shapes, but the AUTO routing correctly prefers fast paths.
- **Memory**: All kernels stay well under threadgroup limits by design. No large synchronous payloads or tight-loop allocations.
- **Test impact**: `test_tilelang_topk.py` has good coverage (including the K=32 insertion fix test and receipt gating). The smoke test now uses `max_ratio=1.5`—pragmatic.

### Prioritized Performance Suggestions (Concrete)

1. **topk_selector.py:1001–1013 & _TOPK_SOURCE:422–438**: Implement binary-search insertion point + bulk shift (or register heap) for the local top-K maintenance. Expected win: recover most of the K=16/32 regression; bigger uplift at K=256. Measure with the existing bench script on reverse-sorted inputs.
2. **topk_selector.py:878–915 (`_path_c_rewrite_merge_round`)**: Harden or upstream the barrier elision. Add a post-rewrite assertion that the pattern was found (e.g., count of `if (lane & mask) == 0` increased or barrier count dropped). This prevents silent perf regression on TileLang updates.
3. **Unify threadgroup sizing logic** between Path B (`topk_selector_metal`) and Path C (`_path_c_threads_for`). Expose a shared helper that optionally queries `mx.metal.get_device().max_threadgroup_memory_length` if available.
4. **Sparse-MLA BWD (largest shape)**: The mixed blocker suggests overhead in reduce/fresh_reduce phases. Profile the generated MSL (or use Metal counters) to see if extra synchronization or suboptimal tiling from TileLang is the culprit. The `pass_configs` already include bound-check dropping—ensure it's firing for the reduce kernels.
5. **Minor**: In Path B `_TOPK_SOURCE`, the comment claims "Compiler unrolls" the K-loop—verify with `metal` compiler flags or disassembly if needed; explicit unroll pragmas could help small K.

**Summary**: The fix-round-3 changes correctly addressed the **critical correctness regression** in Path C topk at the cost of expected perf at K=32 (now properly not auto-routed). No other "fast-because-broken" patterns found. The main remaining perf opportunities are in the O(K) insertion cost (both paths) and hardening the fragile merge rewrite. The MSL transformation and sparse-MLA wiring look solid with no new hot-path regressions. The AUTO routing and strict_policy gating prevent shipping regressions to users.

These kernels are already in a strong state for Apple Silicon—focus on the insertion improvement and rewrite robustness for the next wave.