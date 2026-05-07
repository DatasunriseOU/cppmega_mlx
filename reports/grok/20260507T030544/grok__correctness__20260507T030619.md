---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 5b83522a14065d99a23788c3ac47f4e8daa70be6
timestamp: 2026-05-07T03:06:19.438049+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/test_tilelang_topk.py']
---
**No findings.**

This chunk (1/3) of the diff is limited to:

- Benchmark JSON result updates (`sparse_mla.json`, `topk_selector.json`) reflecting new timing data, changed iters/warmup counts, updated MLX version, and adjusted strict-policy outcomes after Path C became dispatchable. These are data-only changes with no executable code.
- A new helper `_preload_libz3_for_dev_tilelang()` + its eager call in `_msl_transform.py`.

The preload function is defensive, idempotent, uses safe fallbacks, swallows only `OSError` from `CDLL` (expected for missing files), and explicitly documents the macOS dyld basename-resolution problem it solves. It has no impact on production paths, no shared state beyond a sentinel attribute, and no effect on correctness of kernel dispatch or numerical results.

No off-by-one errors, no exception swallowing that hides bugs, no type mismatches, no race conditions, no null/None mishandling, no edge-case regressions, and no behavioral changes to existing kernels visible in this chunk. The topk insertion-sort early-break fix and sparse_mla canonicalize hoist-aware changes are not present in chunk 1.

No correctness bugs introduced by the visible portion of this diff.

**No findings.**

This chunk (new files `dsa_splitk_indexer_loss.py` and `fp8_amax.py`) introduces no correctness bugs, off-by-one errors, incorrect error handling, swallowed exceptions, mismatched types, race conditions, broken null/None handling, or regressions to existing behaviour within the visible code.

### Key verified areas (restricted to this chunk):

- **TopK insertion-sort early-break fix**: Not present in this chunk (lives in `topk_selector.py` per files-touched list; wave-3 note acknowledges prior K=16/32 regression is already addressed in 6b74ec7+5b83522). No related code here.
- **_metal_block_overrides**: Shape-aware AH > 64 logic for stage-2 is careful; register budget math is sound and preserves wave-2 behaviour for AH ≤ 64. No silent deletion of overrides.
- **Causal trim / _active_sk_tiles**: The `T.max(..., 1)` clamp is explicitly justified in comments for edge cases (ASq=0, non-divisible last block) and prevents skipped loop body / uninitialized accumulators. Correct.
- **IndexMask / sparse_loss handling**: Bounds predicates (`in_bounds`, `valid`) are consistently applied before loads; empty-tensor path uses `torch.empty` to avoid unnecessary zero-fill. Good.
- **USE_MD_PRE budget gate**: Correctly falls back to HBM reload on register pressure; placeholder `(1,1)` fragments ensure constant-folding safety.
- **fp8_amax / quantize**:
  - `_pick_block_size` + `_bucket_n` logic is defensive (pow2 bucketing, divisibility enforcement, tiny-shape snapping).
  - Padding with zeros for amax is semantically safe (`max(0, x) == max(x)`).
  - Non-finite `amax_val` guard raises `FloatingPointError` (improvement over silent NaN poisoning).
  - `block_size % threads == 0` invariant is enforced at kernel-build time.
- **Error handling**: All dimension/parameter validation paths raise early with clear messages. No swallowed exceptions.
- **Contiguity / device parity**: Explicit checks and `.contiguous()` calls where needed; no assumption on input layout.
- **Metal vs CUDA differences**: Handled explicitly via `_metal_block_overrides` and `_resolve_target`; no mismatched assumptions.

The code is defensively written with extensive comments explaining wave-2/3 trade-offs, edge-case handling, and numerical equivalence to the Triton reference. No regressions or new correctness issues are introduced in the visible diff.

The orchestrator can merge this chunk's result with the other two.