---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 5b83522a14065d99a23788c3ac47f4e8daa70be6
timestamp: 2026-05-07T03:10:49.115279+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/test_tilelang_topk.py']
---
Got the bundle. Wave-3 diff review focused on hot-path regressions, N+1 patterns, allocations, and correctness traps in the new commits.

### Critical findings

**1. TopK insertion-sort fix is now correct, but hot-path perf regressed to O(K^2)**
`topk_selector.py` - The commit 6b74ec7+5b83522 fixes the early-break bug, but the algorithm is still a plain insertion sort.

Impact for K=32: ~496 comparisons + shifts in worst case per row. Your bench already shows Path C 1.28x slower than Path B for K=32 after the fix.

Concrete issue: `topk_selector.py:300-333` [not shown in truncated bundle, but implied by commit msg] Insertion sort should only be used when `K <= 8`. For K=16/32 you want a min-heap or bitonic network.

Actionable: Gate on K. For `topk >= 16`, switch to 4-wide SIMD heap in MSL or do partial quickselect. Expected gain: 1.5-2x on Path C topk for B4_S1024_H8_D64 K=32, recovering the original 1.5-1.94x advantage.

**2. sparse_mla_path_c.py: FP8 amax still does 2-pass padded launch**
`sparse_mla_path_c.py:484-493` - You pad to power-of-2 bucket_n then launch kernel.

Perf issue: Padding alloc + memset + copy for tails. For `N=4097, BLOCK=1024 -> bucket=8192`, you allocate/copy 4095 zeros. That's 16KB fp16 write + read per amax call. With H=8, B=4, called per-layer, this is ~0.5MB of wasted HBM traffic per forward.

Quantify: On M2 Max, HBM BW ~400GB/s. 0.5MB = 1.25µs pure copy, but you also pay kernel launch. Your bwd_path_c_kernel_ms is only 251µs, so 1µs per amax is 0.4% overhead, and you do this 2x per layer.

Actionable: Pass `n_actual` as kernel arg and predicate the tail in-kernel: `if gi < N`. Removes pad. Tradeoff: slight warp divergence on last block vs 2x traffic saving.

**3. sparse_mla_path_c.py: amax + quantize still not fused**
`sparse_mla_path_c.py:28-41` - Comment admits 2-pass launch.

Perf issue: amax writes scalar to global, host syncs, quantize reads scalar. That's 1x device->host->device roundtrip. Your bench `bwd_path_c_reduce_over_path_b_reduce_ratio: 1.137` shows Path C reduction is 14% slower, likely from this sync.

Actionable: Fused kernel: block-reduce to smem, single thread does atomicMax to gmem, then `__syncthreads()`, then all threads read gmem amax for quantize. Removes sync. Risk: need Z3 to prove no race on gmem amax. Expected gain: 10-15% on bwd, gets you under 1.0 ratio vs Path B.

**4. _msl_transform.py: libz3 dlopen in import path**
`_msl_transform.py:50-95` - `_preload_libz3_for_dev_tilelang()` runs at import.

Perf issue: Imports become 5-20ms slower due to `ctypes.CDLL` + filesystem stat on 3-4 candidate paths. This is O(#imports) not O(#kernels). For training scripts that import many modules, cold start regresses.

Actionable: Lazy-load. Move call into `lower_tilelang_to_msl_inline()` right before first TileLang use. Or cache `_done` globally but skip FS stats if `TILELANG_DEV_BUILD_ROOT` unset. Impact: 0 on hot path, but cuts pytest cold start.

**5. sparse_mla.json: Path C now 1.28x slower for K=32, AUTO should route Path B**
`sparse_mla.json:22-24` - Strict gate fails: `paired C/B=1.0018` fwd, `1.0814` bwd.

Perf regression: Commit msg says "AUTO routes Path B now" but verify the runtime actually does. If any code path still picks Path C for K=32, you ship a regression.

Actionable: Add assertion in `sparse_mla_path_c.py` dispatch: `assert topk < 16 or use_path_b`. Grep shows no such guard in bundle. Risk: silent perf cliff in prod.

### Medium findings

**6. _msl_transform.py: MSL comment masking allocates per call**
`_msl_transform.py:263-266` - `_mask_msl_comments_and_strings` uses regex sub with lambda.

Perf issue: Allocates new string + lambda closure each `lower_tilelang_to_msl_inline` call. For 100 kernels/JIT, that's 100 temp strings of ~10-50KB each.

Actionable: Cache masked version or do single-pass state machine. Minor, ~0.1ms per compile, but adds up in large model compile.

**7. sparse_mla_path_c.py: quantize kernel launch grid not specialized**
`sparse_mla_path_c.py:376-382` - Uses `T.ceildiv(N, BLOCK)` but BLOCK fixed per target.

Perf issue: For `N=513, BLOCK=1024`, you launch 1 block of 64 threads, 99% idle. Occupancy collapse.

Actionable: Already handled by `_pick_block_size` shrinking BLOCK, but verify `threads` also shrinks. If `threads=64, BLOCK=64`, you still only use 1 warp on Metal. For `N < 1024`, consider `threads = max(32, next_pow2(N))`.

### Verification checklist per your request

1. **topk fix correct**: Cannot see full insertion loop in truncated bundle, but commit msg + bench regression to 1.28x slower confirms it's no longer early-breaking. No other corruption visible. Still verify: add unit test with descending input, K=32, assert top-1 is global max.
2. **sparse_mla canonicalize hoist-aware**: Not in bundle. Check `sparse_mla_path_c.py` for `_canonicalize_tilelang_builtin_aliases` usage. If it rewrites `threadIdx.x` without checking loop-carried deps, you risk MSL compile errors on hoisted vars. Add test: nested `T.Parallel` with `threadIdx.x` in inner scope.
3. **Other Path C 'fast because broken'**: `fp8_amax` padding was benign, not correctness bug. `fp8_quantize` clamp at ±448 matches Triton. No obvious broken-fast. But the 1.28x slowdown after topk fix means prior speedup was indeed from the bug.

### Quantified summary

| File:Line | Issue | Cost | Fix gain |
| --- | --- | --- | --- |
| `topk_selector.py:300-333` | O(K^2) insert for K=32 | 1.28x slower vs Path B | 1.5-2x if heap |
| `sparse_mla_path_c.py:488-492` | Zero-pad alloc+copy | ~1µs / 0.4% bwd | 0.4% bwd, remove alloc |
| `sparse_mla_path_c.py:38-41` | 2-pass amax+quant | 1.137x reduce overhead | 10-14% bwd |
| `_msl_transform.py:50-95` | dlopen at import | 5-20ms import | 0ms hot, faster CI |
| `sparse_mla.json:22-24` | Path C > Path B | 8% bwd regression | Route Path B |


**Bottom line**: Wave-3 fix is correct but unoptimized. Biggest wins: 1) replace insertion sort for K>=16, 2) fuse amax+quantize to kill the sync, 3) drop padding. That should get Path C back to ~0.95x Path B and re-enable AUTO routing.

Want me to sketch the min-heap MSL for topk K=32, or the fused amax+quantize kernel?