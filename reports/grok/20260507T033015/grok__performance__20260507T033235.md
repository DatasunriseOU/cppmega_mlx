---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 888d368715dee062c804309c99cd15d4ee55468f
timestamp: 2026-05-07T03:32:35.046956+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/test_tilelang_topk.py']
---
**No performance regressions or hot-path concerns found in this chunk.**

The visible portion of the diff (chunk 2/3) consists of two new files:

- `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py` (full 1192-line implementation)
- `cppmega_mlx/nn/_tilelang/fp8_amax.py` (full 638-line implementation)

Both are carefully engineered TileLang ports of existing Triton kernels with explicit attention to memory budgets, register pressure, redundant HBM traffic elimination, and target-specific tuning (CUDA vs Metal).

### Key observations from the visible code:

- **Memory-aware tuning**: `_metal_block_overrides` and `USE_MD_PRE` budget calculation (`_MD_PRE_BYTES <= 32*1024`) correctly gate large fragments to prevent register spilling on Apple Silicon (32 KB threadgroup limit). Stage-2 `M_pre`/`D_pre` prefetch is disabled when it would exceed budget. Good.
- **Hoisting & reuse**: Stage 1 fully hoists Q tile (`Q_full`) outside the sk_tile loop. Stage 2 does partial hoist inside `(sk_tile, h)` to cut d_tile-level reloads. Explicit comments reference prior wave findings. No obvious redundant I/O left.
- **Causal trim**: `_active_sk_tiles` computation with clamp-to-1 avoids useless iterations and uninitialized output paths. Sound.
- **No O(n²) or N+1**: All loops are properly blocked/tiled with `T.Pipelined`, `T.gemm`, and bounded reductions. No quadratic behavior.
- **Allocations**: All `alloc_shared`/`alloc_fragment` sizes are static per-block and tuned per-target. No allocations inside tight inner loops.
- **Caching**: Heavy use of `@lru_cache(maxsize=64/256)` on kernel builders keyed on shape+target+scale_bits. `_bucket_n` further reduces JIT pressure for amax. Appropriate for hot training paths.
- **Contiguity & copies**: Minimal contiguous() calls with "already contiguous" fast-paths. Sparse mask uses `empty` (not `zeros`) when possible.
- **Atomic / reduction**: `T.atomic_max` + block `T.reduce_max` is standard and necessary for amax; no worse than the original Triton path.
- **No blocking calls in async paths**: This is pure synchronous kernel launch code (no torch async or CUDA streams visible here).
- **No large synchronous payloads**: All data movement is through well-sized tiles.

The changes introduce no new hot-path regressions relative to the Triton baseline they replace. Memory tuning for Metal is notably more conservative and explicit than many GPU kernels. The `libz3 preload _failed_attempts cap=3` and hard assert mentioned in the commit note are outside this chunk.

No findings.