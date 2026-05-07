---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 285234fb38719459c57ee272eaa374b346e2876b
timestamp: 2026-05-07T04:04:25.863262+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/conftest.py', 'tests/test_tilelang_topk.py']
---
**No findings.**

This chunk (3/3) of the mlx-z3-wiring diff introduces no performance regressions or hot-path concerns.

### Summary of changes reviewed (restricted to visible code):
- **fp8_vecmat_path_c.py**: Extensive threading/RLock usage around caches (`_FP8_VECMAT_KERNEL_CACHE`, `_FP8_VECMAT_FIRST_RUN_LOCK`), one-shot warning helpers, manual dict-based kernel caching (replacing `@lru_cache`), intrinsics check caching, and `_resolve_vecmat_scale` fast-path. All changes add synchronization for correctness under concurrent MLX worker threads and fix prior cache-key fragility (vectorized_loads). No new allocations in tight loops, no redundant I/O, no N+1, no blocking calls. The manual lock+dict cache is equivalent or better than the previous lru_cache for the hot path.
- **sparse_mla_path_c.py**: String-based post-processing regexes for MSL canonicalization (hoisted decl stripping, tautological alias removal, infinity normalization). Pure post-lowering text transforms with no runtime cost on the inference hot path. No allocation or loop changes.
- **topk_selector.py**: Similar lock guarding for pass-config cache (now explicit `threading.Lock`), minor insertion-sort logic tweak (early-break fix for correctness), and one tightened test expectation. Cache population remains O(1) amortized.
- **tests/**: Only env var setup for the new `CPPMEGA_ALLOW_UNSAFE_LIBZ3` gate and test expectation updates. No production code impact.

All hot-path concerns from prior waves (duplicate first-run probes, cache key mismatches, silent fallback degradation) have been addressed with minimal-overhead guards. No O(n²), no memory growth vectors, no synchronous payloads, and no new blocking behavior introduced in this visible chunk. The libz3 /tmp gate is correctly isolated behind the env flag as specified.

The orchestrator can merge this chunk with no performance issues flagged from this portion.