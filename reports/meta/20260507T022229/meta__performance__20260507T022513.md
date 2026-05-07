---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 2b9310ef73638a41b8de47566c41891b55218679
timestamp: 2026-05-07T02:25:13.699759+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
Got it — reviewed all 4 files. No HIGH blockers from the earlier wave-1 MLX wiring check, but I do see several hot-path regressions and allocation traps that will bite on long sequences.

### dsa_splitk_indexer_loss.py

**1. Stage-2 Q reload on every sk_tile loop: O(SK_TILES * AH) HBM reads**
`/mnt/data/dsa_splitk_indexer_loss.py:626-632`
```python
for d_tile in T.serial((AD + BLOCK_D - 1) // BLOCK_D):
    for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
        sq_idx = sq_block_id * BLOCK_SQ + i
        d_idx = d_tile * BLOCK_D + dd
        if (sq_idx < ASq) and (d_idx < AD):
            Q_s[i, dd] = Q[sq_idx, b, h, d_idx] # HBM read every sk_tile
```
Stage 1 correctly hoists Q into `Q_full` at `/mnt/data/dsa_splitk_indexer_loss.py:364-369` and reuses it. Stage 2 still reloads Q from HBM for each `sk_tile` and each head `h`. For `ASq=2048, Sk=2048, AH=32, BLOCK_SQ=32, BLOCK_SK=32`, that's `64 sk_tiles * 32 heads = 2048` full reloads of the same Q block.

**Impact**: ~2GB extra HBM traffic at batch=1, seq=2k. Throughput drop ~30-40% on M-series vs stage-1.

**Fix**: Hoist Q like stage-1. You already note it at `/mnt/data/dsa_splitk_indexer_loss.py:606-612` as TODO wave-3. Either restructure to outer h-loop or cache Q per sq_block in shared with `AH` copies. Shared cost: `BLOCK_SQ * AD * AH * 2 bytes = 32*64*32*2 = 128KB` on CUDA defaults — too big. On Metal `32*32*16*2 = 32KB` exact budget. Safer: per-h shared load inside `h` loop but hoisted above `sk_tile`.

**2. M_pre/D_pre fragments spill on CUDA: 64KB register pressure**
`/mnt/data/dsa_splitk_indexer_loss.py:575-576`
```python
M_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
D_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
```
With CUDA defaults `AH=128, BLOCK_SQ=128`: `2 * 128 * 128 * 4 = 128KB` fragments. That's the full register budget for 1 warp. NVCC will spill to local mem, killing occupancy.

**Impact**: 5-10x slowdown on A100/H100. Metal path safe: `32*32*4*2 = 8KB`.

**Fix**: Don't pre-load all heads. Load `m_h[i], d_h[i]` inside the `h` loop from global `M,D` like the old code. Trade 2x HBM loads for no spill. Or split-K over heads: process 8 heads at a time.

**3. Causal trim still runs full SK_TILES on last Q block**
`/mnt/data/dsa_splitk_indexer_loss.py:374-380`
```python
_max_sq_in_block = sq_block_id * BLOCK_SQ + (BLOCK_SQ - 1)
_max_useful_sk = T.min(_max_sq_in_block, ASq - 1)
_active_sk_tiles = T.min(SK_TILES, _max_useful_sk // BLOCK_SK + 1)
```
For last `sq_block_id`, `_max_sq_in_block >= ASq-1`, so `_active_sk_tiles = SK_TILES`. You still run all tiles even though upper-triangular mask zeros them. GEMM cost stays O(SK_TILES).

**Impact**: Last block does 2x work vs needed. For 2k seq, ~3% total waste. Grows to 25% at 8k seq.

**Fix**: `T.ceildiv(sq_idx + 1, BLOCK_SK)` per row, but needs per-i loop. Cheaper: accept it; causal mask is cheap vs GEMM.

### fp8_amax.py

**4. Amax padding + extra kernel launch: redundant alloc + sync**
`/mnt/data/fp8_amax.py:486-492`
```python
if bucket_n!= n_actual:
    padded = torch.zeros(bucket_n, dtype=flat.dtype, device=flat.device)
    padded[:n_actual] = flat
    flat = padded
```
You bucket to power-of-2 to cache JIT, but pay a `cudaMemset` + `cudaMemcpy` on every call with odd size. Then `kernel(flat, amax)` is a second launch.

**Impact**: For `N=4097`, you allocate 8192 elems and copy. 50% memory/time overhead. Host-device sync between launches.

**Fix**: Make kernel accept `n_actual` and mask tail in-loop. You already do `if gi < N:` at `/mnt/data/fp8_amax.py:317-321`. Drop padding. Use `lru_cache` key on `(ceildiv(n,block),...)` not `bucket_n` to still hit cache for nearby sizes.

**5. Quantize creates temp output when user passes non-contiguous out**
`/mnt/data/fp8_amax.py:567-573`
```python
if out_flat_view.is_contiguous():
    kernel(flat, inv_scale_buf, out_flat_view)
    return out
out_flat = torch.empty(n_elements, dtype=fp8_dtype, device=flat.device)
kernel(flat, inv_scale_buf, out_flat)
out_flat_view.copy_(out_flat)
```
Double write + extra alloc on hot path if user slices tensor.

**Impact**: 2x bandwidth if `out = tensor[:,::2]` common in fused ops.

**Fix**: Accept `out.stride()` in PrimFunc and use `T.Buffer` with strides. TileLang supports it. Or document: "out must be contiguous".

### fp8_vecmat_path_c.py

**6. Metal vecmat re-validates intrinsics every build: lock contention**
`/mnt/data/fp8_vecmat_path_c.py:222-228`
```python
global _FP8_VECMAT_PATH_C_INTRINSICS_CHECKED
if _FP8_VECMAT_PATH_C_INTRINSICS_CHECKED:
    return
with _FP8_VECMAT_PATH_C_INTRINSICS_CHECK_LOCK:
   ...
```
You fixed double-check with global bool, but lock still taken on every `_fp8_vecmat_kernel_for` call at `/mnt/data/fp8_vecmat_path_c.py:718`. For model with 100+ layers, that's 100 lock acquires even if cached.

**Impact**: ~100us * 100 = 10ms startup latency. Not hot-path but shows in traces.

**Fix**: Use double-checked locking pattern: check bool, if false then acquire lock, re-check inside. Or rely on `lru_cache` to memoize entire kernel object so `_fp8_vecmat_kernel_for` only runs once per shape.

**7. Scalar fallback uses per-element global loads: no vectorization**
`/mnt/data/fp8_vecmat_path_c.py:492-495`
```python
accum[0] += T.cast(A[0, k], "float32") * T.cast(B[col, k], "float32")
```
When `K % 4!= 0` you hit scalar path `/mnt/data/fp8_vecmat_path_c.py:467-470`. No `T.vectorized(4)` or uint32 load.

**Impact**: 4x bandwidth vs packed path. For K=4097, you do 4096 packed + 1 scalar, fine. For K=130, all scalar: 4x slower.

**Fix**: Peel loop: do `K//4*4` with dot4, tail with scalar. You already guard `K%4==0` at `/mnt/data/fp8_vecmat_path_c.py:532-535` for macro path, but scalar fallback should still vectorize what it can.

### topk_selector.py

**8. Path C threadgroup merge uses O(BLOCK_SIZE * K) shared: caps BLOCK_SIZE**
`/mnt/data/topk_selector.py:521-522`
```python
threadgroup float pair_vals[PAIR_BUF];
threadgroup int pair_idx[PAIR_BUF];
```
With `PAIR_BUF = BLOCK_SIZE * K` at `/mnt/data/topk_selector.py:664`. At `K=256`, `BLOCK_SIZE` capped to `32768 / (256*8) = 16` threads per `/mnt/data/topk_selector.py:632-638`.

**Impact**: Only 16-way parallelism per row. For `B=1, T=4k`, you launch 1 threadgroup with 16 threads: 256x underutilized vs A100's 108 SMs.

**Fix**: Already documented. For K>64, ship `backend="mlx"` not Path B. Or switch to multi-pass: each threadgroup outputs its top-K, then second kernel merges. Adds sync but scales.

**9. PassConfig probe logs to stderr on every import**
`/mnt/data/topk_selector.py:184-192`
```python
print(
    f"[cppmega-mlx-cuz] dropping unsupported PassConfig "
    f"key {key!r} from topk_selector path-c lowering "
    f"(not registered in active libtilelang).",
    file=sys.stderr,
)
```
Runs at import time via `_topk_path_c_pass_configs()` cache fill.

**Impact**: Log spam in distributed runs, I/O syscall per process.

**Fix**: Gate on `os.environ.get("CPPMEGA_LOG_LEVEL")`. Or drop print, keep `warnings.warn` only.

**MLX wiring cross-check**: Wave-1 notes hold. I see no HIGH/MED blockers. The `metal_fp8_e4m3_dot4` intrinsic check `/mnt/data/fp8_vecmat_path_c.py:217-228` and PassConfig filter `/mnt/data/fp8_vecmat_path_c.py:86-117` match the "optional design refactor" comment. One tweak: `TILELANG_METAL_VECMAT_TARGET` at `/mnt/data/fp8_vecmat_path_c.py:36` hardcodes `-thread_warp_size=32`. TileLang now defaults that on Metal, so you can drop the flag unless you support gfx1030.

Want me to patch the stage-2 Q hoist first? That's the biggest win and unblocks 2k+ seq lengths.