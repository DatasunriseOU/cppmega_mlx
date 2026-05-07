---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: HEAD~1..HEAD
base_ref: 44f4f888bb8f22bc9526658ac87168ee11e677cb
head_ref: 2b9310ef73638a41b8de47566c41891b55218679
timestamp: 2026-05-07T02:20:58.209169+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py']
---
**P1 correctness bugs (high severity: can cause numerical mismatch, NaNs, or silent wrong results vs. Triton reference / upstream Megatron PR #4039).**

### dsa_splitk_indexer_loss_tilelang.py

1. **Incorrect output layout for stage-2 `C` in the packed-dot4 path of `make_fp8_vecmat_reduce_kernel` (but wait—no, that's in the second file; for DSA:)**

   Focus on DSA:

   **dsa_splitk_indexer_loss_tilelang.py: ~lines 312-320 (stage1 Q-hoist load) and ~lines 425-430 (stage2 Q load):**

   In stage1:
   ```python
   for i, dd in T.Parallel(BLOCK_SQ, AD):
       sq_idx = sq_block_id * BLOCK_SQ + i
       if sq_idx < ASq:
           Q_full[i, dd] = Q[sq_idx, b, h, dd]
       else:
           Q_full[i, dd] = T.cast(0, in_dtype)
   ```
   Then inside sk_tile / d_tile:
   ```python
   for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
       d_idx = d_tile * BLOCK_D + dd
       if d_idx < AD:
           Q_s[i, dd] = Q_full[i, d_idx]   # <-- correct reuse
   ```

   **But in stage2 (no hoist yet, as noted in TODO):**
   ```python
   for i, dd in T.Parallel(BLOCK_SQ, BLOCK_D):
       sq_idx = sq_block_id * BLOCK_SQ + i
       d_idx = d_tile * BLOCK_D + dd
       if (sq_idx < ASq) and (d_idx < AD):
           Q_s[i, dd] = Q[sq_idx, b, h, d_idx]  # <-- direct HBM read every sk_tile * h * d_tile
   ```

   The **wave-2 perf comment** claims a hoist was done only for stage1. This is **not** a regression (the TODO at line ~510 acknowledges it), but it introduces a **performance regression** vs. a full hoist (and vs. Triton which can cache better). More critically for **correctness**: on Metal with small BLOCK_D=16 and large AH/SK, the repeated loads + no shared Q staging per-head can lead to **subtle scheduling differences** in pipelined loops, potentially exposing **uninitialized Q_s fragments** on boundary d_tiles if the `if` predicate is mis-evaluated in SIMDgroup emission. Not a hard bug, but fragile.

2. **P1: Mishandled edge case in stage1 causal trim when ASq == 0 or Sk == 0 (or very small).**

   **dsa_splitk_indexer_loss_tilelang.py: lines 278-285 (stage1) and ~lines 500-507 (stage2):**

   ```python
   _max_sq_in_block = sq_block_id * BLOCK_SQ + (BLOCK_SQ - 1)
   _max_useful_sk = T.min(_max_sq_in_block, ASq - 1)
   _active_sk_tiles = T.min(SK_TILES, _max_useful_sk // BLOCK_SK + 1)
   ```

   - If `ASq == 0`, `ASq-1 == -1`, `_max_useful_sk = min(..., -1)`. Integer division and `T.min` with negative can produce **negative or zero _active_sk_tiles** (TileLang/TVM may treat as 0 or wrap).
   - If `sq_block_id == 0` and `BLOCK_SQ > ASq`, the trim is overly aggressive or produces **off-by-one** (e.g., `_max_useful_sk // BLOCK_SK + 1` under-counts the last partial tile).
   - **Regression risk**: Triton reference (original in cppmega/cppmega/megatron/dsa_splitk_indexer_loss.py) uses `tl.arange` + masks without this early-exit trim; the trim was added in wave-2 for perf. On `ASq % BLOCK_SQ != 0` or tiny shapes, it can **skip the final sk_tile** that still has valid `sq_idx >= sk_idx` positions → **wrong softmax stats / loss under-count**.

   Same pattern in stage2. **Actionable fix**: Add explicit `if ASq <= 0 or Sk <= 0: return` early in both `make_*_kernel`, and guard trim with `max(0, ...)` + `T.max(1, ...)` for the +1 term when `_max_useful_sk >= 0`.

3. **P1: IndexMask handling in sparse_loss=True when IndexMask is empty/zero-length tensor (as documented in the wrapper).**

   **dsa_splitk_indexer_loss_tilelang.py: lines 92-100 (docstring) and ~lines 250-255, ~lines 580-585:**

   The code creates:
   ```python
   if sparse_loss:
       index_mask = ... .scatter_(...)
   else:
       index_mask = torch.empty(...)  # shape (AB, ASq, Sk)
   ```
   Then passes it unconditionally.

   In kernel:
   ```python
   if SPARSE and in_bounds:
       s = s + IndexMask[b, sq_idx, sk_idx]
   ```
   - When `sparse_loss=False`, `IndexMask` is **uninitialized** (empty allocates without zero-fill, per comment). The `if SPARSE` guards the **read**, so no UB in CUDA, but **Metal SIMDgroup emission** or boundary tiles (`in_bounds=False` but predicate evaluation) may still touch memory → **undefined behavior** or NaNs if compiler reorders.
   - The docstring claims "zero-length when not sparse" but code always allocates full shape. **Mismatch** with "to keep the PrimFunc signature stable".

   **Regression**: Triton wrapper likely passes a properly masked/zero tensor or uses `mask=` in loads. Here, uninitialized memory when `not sparse_loss` is a **silent correctness hazard** on Metal (where shared mem / register pressure is tighter).

4. **P2: Swallowed exceptions / poor error handling in `_stage1_kernel_for` / `_stage2_kernel_for` and JIT cache.**

   **dsa_splitk_indexer_loss_tilelang.py: lines 650-670 (`_stage*_kernel_for`):**

   ```python
   prim = make_..._kernel(...)
   return tilelang.compile(...)
   ```
   No `try/except` around compile (unlike fp8_vecmat which has some MSLDispatch handling). If TileLang lowering fails (e.g., Metal block size OOM after overrides, or shape >128 AH check bypassed), the exception is raised from the cached call site in `dsa_splitk_indexer_loss_tilelang`, but **lru_cache** can cache the failure state poorly on retry. Also, `softmax_scale_bits` unpack assumes valid float bits—no validation.

   The top-level function has `if AH > 128: raise`, but this is **after** shape extraction and **before** kernel build—good, but the Metal overrides (32x32x16) are applied unconditionally via `_block_constants_for_target`, even on CUDA when shapes would support larger tiles → **perf regression** on CUDA (not correctness, but violates "CUDA defaults" claim).

5. **P2: In stage1 head-0 index softmax path: `idx_scores_f` is zeroed only on first sk_tile? No—re-zeroed every tile inside the if h==0 block? Wait:**

   Look carefully: the `for sk_tile` loop has:
   - scores_f zeroed every tile (good).
   - Then **after** matmul/scale/mask for attention.
   - Then **if h == 0**: the idx_scores_f block **does NOT zero `idx_scores_f`** before the per-tile index_scores load/exp/reduce.

   **Bug**: `idx_scores_f[i,j] = v` or `-inf` happens, but if previous tile left garbage (or on Metal register reuse), the `T.reduce_max` / exp can pick stale values. **Fix**: move `for i,j in Parallel: idx_scores_f[i,j] = ...` or explicit zero before the if h==0 block, mirroring scores_f.

   This is a **clear regression** introduced by the wave-2 structure (the index path was inside the sk_tile but not symmetrically zeroed like the main attention path).

### fp8_vecmat_path_c.py (wave-1 fixes noted, but remaining issues)

6. **P1: `_uses_fp8_dot4_packed_macro` called with runtime `K` but the packed branch in `make_fp8_vecmat_reduce_kernel` hard-codes assumptions.**

   **fp8_vecmat_path_c.py: lines 312-320 (`_uses...`)** and the PrimFunc at ~lines 340+ (packed branch uses `_FP8_VM_K_WORDS = K//4` but the dot4 call is `T.metal_fp8_e4m3_dot4(..., i, i)` with `i < _FP8_VM_K_WORDS`).

   The guard `K % 4 == 0` is there, but `_normalize_vecmat_inputs` already raises on `k % 4 != 0`. However, the **C tensor shape** differs:
   - Packed: `C: T.Tensor((1, N), "float32")`
   - Others: `(N,)`

   In `_fp8_vecmat_kernel_for` (~lines 480+): it checks `output_shape in ((n,), (1,n))` and reshapes at the end. **But** the dispatch path has:
   ```python
   if input_names == [...] and output_shape in ((n,), (1,n)):
       ... kernel(..., output_shapes=(output_shape,))
   else:
       _msl_transform.dispatch(...)
   ```
   This is brittle. If lowering changes the exact shape tuple due to TileLang version, **output reshape fails** or wrong data is returned (shape mismatch in MLX metal_kernel call). The comment acknowledges the difference, but no robust canonicalization.

7. **P2: Race condition / caching in `_FP8_VECMAT_PATH_C_INTRINSICS_CHECKED` and PassConfig filtering.**

   The global `with _FP8_VECMAT_PATH_C_INTRINSICS_CHECK_LOCK:` protects the check, but `_FP8_VECMAT_PATH_C_PASS_CONFIGS_CACHE` uses a different lock (`_fp8_vecmat_path_c_pass_configs_cache_lock`). Concurrent calls from multiple MLX threads (possible in training) can still race the first-time `PassContext` probe, leading to **duplicate warnings** or partially-filtered configs (swallowed AttributeError in `_filter_supported_pass_configs`).

   Also, `_warn_path_c_unavailable` uses a set for de-dupe, but logs to stderr/warnings—**good** (as per wave-1), but if called from exception paths, the reason string may include stack traces making the set ineffective.

8. **P2: Edge case in `_resolve_vecmat_scale` when `scale_w` is scalar float but `scale_w_per_row=True` logic.**

   It correctly sets `scale_w_per_row = arr.size == n`, but if input `scale_w` is a 0-d or wrong-shape mx.array, the reshape + size check can raise **after** partial normalization, leaving inconsistent state. Minor, but swallows the exact shape error message sometimes.

**Performance suggestions (correctness-adjacent, as they affect numerical stability under register pressure on Metal):**

- **Metal block overrides (32/32/16)**: The comment at lines ~140-160 in DSA file is good, but stage2 still allocates `M_pre/D_pre = (AH, BLOCK_SQ)` fp32 fragments **per sq_block** (128*32*4 ~16KB on CUDA, but on Metal with AH=128 it's 16KB+ shared → **register spilling** likely, causing slowdown or compile failure). Consider hoisting only when `AH * BLOCK_SQ * 4 < 24KB` or add a Metal-specific smaller AH pre-load.
- Stage1 Q-hoist is excellent (saves SK_TILES * HBM loads); stage2 TODO should be prioritized—**Q is independent of sk_tile**.
- In DSA stage2 per-head loop: the `softmax_attn` accumulation over AH heads inside sk_tile can cause **fp32 overflow** or precision loss on long SK (many additions before final *= INV_AH). Consider accumulating in a higher-precision temp or per-head normalize earlier (but that changes algo—match Triton exactly first).
- For FP8 vecmat: the packed dot4 path is fastest, but the `unroll_factor=4` in `T.unroll` + `explicit=False` may not emit optimal MSL on all TileLang versions—test against canonical_vecmat_runtime_body.

**General recommendations (actionable):**

- Add **unit tests** comparing against the original Triton kernels for edge cases: ASq=1, Sk=1, ASq % BLOCK !=0, sparse_loss=True/False, AH=1/128, Metal vs CUDA.
- Run with `torch.compile` mode or deterministic flags to catch NaN propagation.
- The `pg_collection` param is accepted but ignored—document or remove for clarity.
- In both files, the `T = cast(Any, T)` + `import tilelang.language as T` inside functions is fine for pyright, but consider a top-level `if TYPE_CHECKING:` for better static analysis.

The wave-2 changes (SK trim, Q hoist, warning improvements) are mostly solid, but the **index_scores_f zeroing omission** and **causal trim edge cases** are the most likely sources of numerical divergence from the Triton reference. Fix those first for P0/P1 correctness.