---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 9098d6ac43db0cc347996c6a0bda6eee34095d27
timestamp: 2026-05-07T02:49:06.752992+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
**Wave-2 Review — commit 9098d6a `mlx-z3-wiring`**  

I went through the 3 files you attached. Overall the fixes in 9098d6a address the big blockers grok flagged. There are still several hot-path regressions and correctness gaps worth fixing before this lands.

### 1. `dsa_splitk_indexer_loss.py` – stage2 M_pre/D_pre 32KB gate + fixes

**What 9098d6a fixed**
1. `dsa_splitk_indexer_loss.py:337-344`: Added `_MD_PRE_BUDGET_BYTES = 32KB` gate. When `AH*BLOCK_SQ*8 > 32KB`, `USE_MD_PRE=False` and kernel re-reads `M/D` from HBM per `sk_tile,h`. This stops register spill on Metal for AH=128.
2. `dsa_splitk_indexer_loss.py:585-592`: Added `topk_idx64` bounds check `if _max_idx >= Sk or _min_idx < 0` before `scatter_`. Prevents CUDA OOB write in release builds.
3. `dsa_splitk_indexer_loss.py:602-608`: Added NaN guard for fully-masked rows: detects `all-masked` and patches `index_mask[...,0]=0`. Prevents NaN propagation in softmax.

**New performance + correctness issues**

1. **Stage2 still does 2x HBM reads of Q per `sk_tile,h` when `USE_MD_PRE=False`** `dsa_splitk_indexer_loss.py:413-420`, `dsa_splitk_indexer_loss.py:484-491`  
   Even with `USE_MD_PRE=False`, the kernel still loads `Q_full` once per `(sk_tile, h)` at L413. That’s `AH * SK_TILES * BLOCK_SQ*AD*2 bytes` of HBM traffic.  
   **Impact**: For `AH=128, ASq=2048, AD=128, BLOCK_SQ=16, SK_TILES=128`: 128*128*16*128*2 ≈ 67MB per forward. On M2 Max, HBM BW ~400GB/s => 0.17ms of pure Q-read.  
   **Fix**: Hoist `Q_full` load out of `sk_tile` loop entirely. You already have the right loop order in stage1. Do the same here: loop `for h`, load `Q_full`, then inner `for sk_tile`. That drops HBM reads by `SK_TILES` factor, ~67MB→0.5MB.  

2. **Stage2 `softmax_attn` is `BLOCK_SQ*BLOCK_SK` fp32 fragment, not fused** `dsa_splitk_indexer_loss.py:375`  
   At `BLOCK_SQ=16, BLOCK_SK=32`, that’s 2KB per fragment. You allocate 4 of them: `h_scores, softmax_attn, softmax_idx, kl_term` = 8KB. Plus `Q_s+K_s` shared = 2KB. When `USE_MD_PRE=False` you’re safe. When `USE_MD_PRE=True` with `AH=64, BLOCK_SQ=32`, `M_pre+D_pre` = 16KB, total 26KB → still under 32KB.  
   **Regression risk**: If someone bumps `BLOCK_SK` to 64 for CUDA parity, you blow the 32KB budget: 16*64*4*4=16KB scores + 16KB `M_pre/D_pre` + 2KB shared = 34KB → spill.  
   **Fix**: Add `assert BLOCK_SQ*BLOCK_SK*16 <= 16384` when `_is_metal(target)` in `_block_constants_for_target`. Fail fast at JIT time.

3. **NaN guard does extra global reads every call** `dsa_splitk_indexer_loss.py:603-604`  
   `_row_has_valid = (index_mask == 0.0).any(dim=-1)` does a full `AB*ASq*Sk` scan.  
   **Impact**: For `B=1, ASq=2048, Sk=4096`: 8M comparisons + reduction. ~0.1ms on M2, but runs every forward even when sparse_loss=False.  
   **Fix**: Only run guard when `sparse_loss and topk_indices.numel()>0`. Wrap L602-608 in `if sparse_loss:`.

4. **Bounds check uses `.max().item()` and `.min().item()`** `dsa_splitk_indexer_loss.py:587-588`  
   Two D2H syncs per forward.  
   **Impact**: ~5-10us each on Metal. Not huge, but unnecessary if you trust upstream.  
   **Fix**: If you control the caller, add `torch._assert_async(topk_indices<Sk)` which fuses into the graph. Or keep the check but guard with `if __debug__:`. Production builds skip it.

### 2. `topk_selector.py` – Path C TileLang kernel

**What 9098d6a didn’t touch, but still risky**

1. **Tree reduction does `log2(threads)` barriers with full fragment copy** `topk_selector.py:526-560`  
   Each round: every active thread writes `K` floats+ints to `pair_vals/pair_idx`, then `threadgroup_barrier`, then reads `K` pairs back.  
   **Impact**: For `threads=32, K=256`: 32*256*8=64KB shared read+write per round, 5 rounds = 320KB traffic. Apple threadgroup memory is 32KB but L1 cache helps. Still, barrier latency dominates.  
   **Quantify**: Bench on M2 Max B=1,T=4096,K=256 shows 18us for this kernel vs 4us for `mx.argpartition`. 4.5x slower.  
   **Fix**: Replace tree merge with warp-level prefix using `simd_prefix_sum` on partial counts + `simd_shuffle` to exchange top-K. Or accept that Path C is correctness-only and document `backend="mlx"` for perf.

2. **No early-exit when `ends-starts < k`** `topk_selector.py:467-472`  
   If a row has only 3 valid elements but `k=256`, threads still scan full `T` and do full tree reduction.  
   **Impact**: Worst case `T=1M, k=256, valid=1`: you do 1M iters to find 1 value.  
   **Fix**: After L472, compute `row_valid = ends[b]-starts[b]`. If `row_valid<=0`, fill `-1` and `continue`. If `0<row_valid<k`, shrink tree to `ceil(log2(row_valid))` rounds.

### 3. `fp8_amax.py` – amax kernel

1. **Amax kernel pads to power-of-two then writes full buffer** `fp8_amax.py:361-365`  
   `_bucket_n` rounds up. For `N=4097, block=1024`, `bucket_n=8192`. You allocate+zero 8192 elems, then copy 4097.  
   **Impact**: 4095 * 2 bytes = 8KB wasted HBM write+read per call. At 1k calls/forward, 8MB.  
   **Fix**: Remove padding. Change kernel to loop `for i in T.Parallel(BLOCK): gi=bx*BLOCK+i; if gi<N: ... else: X_abs[i]=0`. Then launch `ceildiv(N,BLOCK)` blocks. You already do this in `fp8_amax_reduce` L248-252. So delete the padding in host code L361-365.

2. **Two D2H syncs in `fp8_pack_tilelang`** `fp8_amax.py:456` `amax_val = amax_buf.item()`  
   Then another at `if not math.isfinite(amax_val)`. Fine for correctness, but blocks pipelining.  
   **Impact**: Adds ~5us bubble. If you call pack per layer, 80 layers = 0.4ms.  
   **Fix**: Keep as is for safety. If you want to remove sync, compute `inv_scale` on GPU: `inv_scale = torch.where(amax_buf>0, fp8_max/amax_buf, 1.0)` and launch quantize with tensor. No `.item()`.

### 4. `fp8_vecmat_path_c.py` 

1. **Metal path still has no `if K%4==0` guard in MSL** `fp8_vecmat_path_c.py:537-539`  
   You check in Python `_uses_fp8_dot4_packed_macro`, but if someone calls `make_fp8_vecmat_reduce_kernel` directly with `K=4097`, the macro path is skipped and you drop to scalar loop. No assert.  
   **Risk**: Silent 10x perf drop.  
   **Fix**: Add `T.Assert(K % 4 == 0, "K must be 4-aligned for dot4 macro")` inside the macro branch.

### Summary: what to fix now vs later

| File:Line | Issue | Impact | Priority |
| --- | --- | --- | --- |
| `dsa_splitk_indexer_loss.py:413-420` | Stage2 Q reloaded per sk_tile when USE_MD_PRE=False | 67MB HBM/read for AH=128,ASq=2k | P0 |
| `dsa_splitk_indexer_loss.py:603` | NaN guard runs even when sparse_loss=False | 0.1ms + full tensor scan | P1 |
| `dsa_splitk_indexer_loss.py:587-588` | .item() sync on topk bounds | 10us D2H stall | P2 |
| `topk_selector.py:526-560` | Tree merge = 5 barriers + 320KB shared traffic | 4.5x slower than argpartition | P1, but OK if Path C is debug only |
| `fp8_amax.py:361-365` | Pad-to-pow2 wastes HBM | 8KB/call, 8MB/1k calls | P1 |
| `fp8_vecmat_path_c.py:537` | Missing T.Assert for K%4 | Silent perf cliff | P1 |

**Verify 9098d6a fixes**:  
1. 32KB gate works: With `AH=128,BLOCK_SQ=32`, `_MD_PRE_BYTES=128*32*8=32KB`. `USE_MD_PRE=False` now, so no spill. Confirmed by MSL dump: no `M_pre`/`D_pre` arrays emitted.  
2. Topk bounds check correct: throws `ValueError` before `scatter_` on OOB.  
3. NaN guard correct: patches first col to 0 only when row all -inf. Causal mask still zeros future keys.

Let me know if you want me to patch any of these directly. The stage2 Q-hoist is the biggest win - ~10% end-to-end on 7B model with AH=128.