---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 85470b2a15362a4232215e436e6be7a005991c62
timestamp: 2026-05-07T02:15:17.503798+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
**No performance regressions or hot-path concerns introduced by this diff.**

The changes consist of three large new TileLang-based kernel modules (`dsa_splitk_indexer_loss.py`, `fp8_amax.py`) plus targeted fixes in the two existing ones (`fp8_vecmat_path_c.py`, `topk_selector.py`). All hot code lives inside `@T.prim_func` definitions that lower to CUDA/Metal kernels; the Python wrappers are cold-path JIT + dispatch logic.

### Summary of relevant patterns checked
- **No O(n²) loops** or quadratic behavior in either Python or generated kernels. All tiling is fixed-block (e.g. BLOCK_SQ/BLOCK_SK/BLOCK_D) with linear SK tiling.
- **No redundant I/O or N+1 queries**. Tensor accesses are properly predicated and coalesced via `T.Parallel` + `T.gemm`. Boundary guards prevent OOB reads (explicit `if` predicates on loads, matching prior Triton `mask=`).
- **No blocking calls in async code**. No async paths present.
- **Allocation behavior**: All `T.alloc_shared` / `T.alloc_fragment` are static per-kernel and sized to small fixed blocks (e.g. 32×32×16 on Metal, 128×128×64 on CUDA). No allocations inside inner loops.
- **Memory growth**: Stage-1/2 temporaries are kernel-local (shared + registers). Host-side masks use `torch.empty` (not `zeros`) when `sparse_loss=False` to avoid unnecessary zero-fill.
- **Caching**: `_stage1_kernel_for` / `_stage2_kernel_for` (and equivalents in other files) use `@lru_cache(maxsize=64)`. The new `threading.Lock` guards in `fp8_vecmat_path_c.py:111` and `topk_selector.py:205` correctly protect the pass-config probe cache, eliminating potential races on first use without introducing contention on the hot path.
- **Z3-related wiring**: Conservative-by-default, with only fp8 dot4 auto and barrier elision enabled. Fixes mentioned (ScopedBVMode, Bind BV clamp, Reset wiring, predicate_fusion null guards on BufferLoad, vectorize iter_var_size<=1 skip, butterfly guard, K>0 guard, etc.) are defensive and do not alter generated kernel structure or introduce new loops/allocations.
- **MLX-side**: PassConfig opt-in + lock is hot-path safe (lock held only on cache miss).

The largest addition (`dsa_splitk_indexer_loss.py`, 959 lines) implements a **two-stage split-K online-softmax + KL reduction** with careful tiling for both CUDA (96 KB shmem) and Metal (32 KB threadgroup) memory budgets. Double-buffering is requested via `T.Pipelined`, loads are bounds-guarded, and reductions use efficient `T.reduce_max`/`T.reduce_sum`. No large synchronous payloads or tight-loop allocations.

Minor notes (none rise to regression level):
- Metal uses smaller tiles (32/32/16) to stay under register pressure — expected and documented.
- Sparse mask construction on host for `sparse_loss=True` is unchanged from prior Triton path.
- `IndexMask`/`IndexScores` loads remain predicated (lines ~300, ~340, ~620, ~680 in new file).

All changes are either performance-neutral or represent strict improvements (better Metal support, race-free caching, defensive Z3 guards, empty vs zeros for non-sparse mask). No hot-path regressions detected.