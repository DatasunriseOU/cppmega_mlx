---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 9098d6ac43db0cc347996c6a0bda6eee34095d27
timestamp: 2026-05-07T02:46:50.598735+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
**Review of attached files (post Wave-2 / commit 9098d6a "mlx-z3-wiring")**

Focus: **correctness bugs** introduced or remaining in this diff, especially regressions, edge cases, error handling, and the new changes mentioned (M_pre/D_pre budget gate, topk_indices bounds-check + NaN guard). I reference specific files and approximate line numbers based on the provided sources.

### 1. `dsa_splitk_indexer_loss.py` — Critical correctness issues (new in this port)

**High severity — potential NaN / wrong loss values**

* **File**: `dsa_splitk_indexer_loss.py`
* **Location**: `make_dsa_splitk_stage2_kernel` (the truncated section + full stage2 logic around the `if USE_MD_PRE` block and M_pre/D_pre handling)

The new Wave-2 "M_pre/D_pre 32KB budget gate with HBM fallback" is **incorrectly implemented** in the provided code.

In the stage2 kernel builder (around the comment "Wave-1b fix-round-2 (HIGH perf)"):
```python
if USE_MD_PRE:
    M_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
    D_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
else:
    M_pre = T.alloc_fragment((1, 1), "float32")   # !!!
    D_pre = T.alloc_fragment((1, 1), "float32")
```

**Bug**: When `USE_MD_PRE=False` (i.e., AH*BLOCK_SQ*4 > ~16-24KB on Metal), the code still **declares** tiny placeholder fragments but the *later read path* does:
```python
if USE_MD_PRE:
    ... read from M_pre[h, i] ...
else:
    ... read from HBM M[b, h, sq_idx] ...
```

However, the **declaration** of the small `(1,1)` fragments is still present in the PrimFunc IR. More critically, the **pre-load loop** for M_pre/D_pre is *conditionally executed only if USE_MD_PRE*, but the variable `M_pre`/`D_pre` in the `else` branch is a **different shape**. This creates a **name shadowing / type mismatch** in the generated IR that TileLang/TVM lowering may not handle cleanly across CUDA vs Metal, or when the constant-folder doesn't fully eliminate the dead `(1,1)` allocation.

**Worse**: If the `if USE_MD_PRE:` guard is ever elided or the condition is evaluated at a point where the shape is inconsistent, you get **undefined behavior** or **register spill / crash** on Metal. The comment claims "the constant-folder elides the dead array" — this is **not guaranteed** for all TileLang versions / targets.

**Recommendation (actionable)**: Make the allocation shape a **constexpr** that is always safe. Use:
```python
M_pre_shape = (AH, BLOCK_SQ) if USE_MD_PRE else (1, 1)
M_pre = T.alloc_fragment(M_pre_shape, "float32")
```
(or compute it once at Python level and pass as a global). Same for D_pre. This eliminates the conditional allocation entirely.

**Related regression**:
- The Wave-2 comment mentions "when AH*BLOCK_SQ*8 exceeds the per-block register budget" → fallback to HBM. But the actual condition for `USE_MD_PRE` is **not shown** in the provided snippet (it's truncated). If the condition is computed using Metal-specific block sizes but the kernel is compiled for CUDA (where 128x128 is fine), you may unnecessarily fall back to slower HBM path on CUDA. Verify the Python-side `_metal_block_overrides` vs CUDA path.

**NaN / denom handling in stage2** (lines around softmax_idx and KL term):
```python
denom1 = d1_local[i]
if denom1 <= T.cast(0, "float32"):
    denom1 = T.cast(1, "float32")
```
This is good, but the equivalent for attention denom (`d_h[i]`) should be audited for identical handling. Also, the `EPS` constant in the KL term (`p * (log(p + EPS) - log(q_ + EPS))`) — ensure `EPS` is defined and positive (e.g. `1e-8` or `1e-12`). Missing definition would be a hard error.

**IndexMask sparse path**:
The new NaN guard for fully-masked rows (added in Wave-2) is present and looks correct:
```python
_row_has_valid = (index_mask == 0.0).any(dim=-1)
...
if not bool(_row_has_valid.all()):
    ... patch index_mask[..., 0] ...
```
Good. However, this runs **every** call when `sparse_loss=True`. For dense case it skips allocation of the full mask — correct.

**topk_indices validation** (new in Wave-2, around `dsa_splitk_indexer_loss_tilelang`):
- Good bounds check on `max()` / `min()`.
- Promotion to `int64` for scatter_ is correct.
- Contiguity enforcement is good.
- **Minor**: `if topk_indices.numel() > 0:` before `.max()` — safe, but `torch.max` on empty tensor raises; the guard prevents it.

**Causal trim logic** (`_active_sk_tiles`):
```python
_max_useful_sk = T.min(_max_sq_in_block, ASq - 1)
_active_sk_tiles = T.max(
    T.min(SK_TILES, _max_useful_sk // BLOCK_SK + 1), 1
)
```
The `T.max(..., 1)` clamp is explicitly called out as a "Wave-3 self-audit" to prevent skipping the loop entirely on boundary blocks. This looks correct and prevents uninitialized accumulator bugs. Good fix.

### 2. `fp8_amax.py` — Minor issues

* **File**: `fp8_amax.py`
* **Location**: `_pick_block_size` and `_bucket_n`

No major correctness bugs introduced. The power-of-two bucketing for amax is correct (`amax(0) == 0` is identity). The `block_size % threads == 0` enforcement is good.

**Nit (potential future footgun)**: `tilelang_supports_with_reason` returns a tuple, but the thin `tilelang_supports` wrapper is used in some places. Ensure all call sites in `cppmega/megatron/fp8_activations.py` were updated to handle the reason when needed. Not a regression in this file.

### 3. `fp8_vecmat_path_c.py` — No new correctness bugs found in this diff

The Z3 PassConfig filtering, intrinsics registration cache, and `_canonicalize_macro_output_shape` (new robustness) look solid. The `MSLDispatchUnsupported` handling with warning is a good improvement over silent fallback.

The `vectorized_loads=True` probe warning is correctly emitted once.

### 4. `topk_selector.py` — Serious correctness regression in Path C merge

**High severity — incorrect top-k on Metal**

* **File**: `topk_selector.py`
* **Location**: `_path_c_kernel_for` (the TileLang DSL kernel), specifically the local heap insertion loop and the merge phase.

**Bug 1: Local heap insertion is broken for K > 1**

In the sweep loop:
```python
if value > local_vals[0]:
    pos = 0
    for p in T.serial(1, _TOPK_C_K):
        if value > local_vals[p]:
            local_vals[p - 1] = local_vals[p]
            local_idx[p - 1] = local_idx[p]
            pos = p
        elif (_TOPK_C_K <= 8) or (_TOPK_C_K >= 64):
            break   # <--- This is extremely suspicious
    local_vals[pos] = value
    local_idx[pos] = j
```

The `elif` break condition looks like a **debug/tuning artifact** that was left in. For medium K (e.g. 16-32) it may early-break the shift, leaving the insertion **incorrect** (wrong position or partial shift). This is a **clear regression** from a cleaner insertion sort.

The comment "Compiler unrolls" suggests this was an attempt to help unrolling, but it breaks correctness.

**Bug 2: Merge phase has off-by-one / indexing issues**

In the tree reduction:
```python
for pick in T.serial(_TOPK_C_K):
    ...
    if a_val >= b_val:
        merged_vals[_TOPK_C_K - 1 - pick] = a_val
        ...
        ap -= 1
    else:
        ...
        bp -= 1
```

This tries to build a descending list. Combined with the final write:
```python
indices[bx, i] = pair_idx[0, _TOPK_C_K - 1 - i]
```

It *should* be okay if the merge is correct, but the early `break` in the insertion phase means the local lists going into the merge are already corrupted for many K values.

**Bug 3: `_path_c_rewrite_merge_round` is fragile**

The regex-based post-processing of the lowered MSL (`_path_c_rewrite_merge_round`) is extremely brittle. It hard-codes strings like `thread_position_in_threadgroup.x` and specific patterns. Any change in TileLang's Metal codegen (e.g. variable naming, whitespace, or new barriers) will silently break the rewrite or produce malformed MSL.

This is a maintenance hazard and source of subtle correctness bugs.

**Path B vs Path C divergence**:
The direct-MSL Path B (`topk_selector_metal`) uses a different (also complex) merge strategy with `PAIR_BUF` etc. Ensure the reference (`topk_selector_reference`) and both paths agree on the *set* of returned indices (tests already do set equality, which is good). However, with the insertion bug in Path C, even set equality may fail on some inputs.

**Recommendation**: The Path C kernel is too complex for the current TileLang Metal target. Strongly consider **disabling Path C** for topk_selector until the insertion + merge logic is proven equivalent to Path B / reference on a broad test suite (including K=1, K=seq_len, masked rows, all -inf, etc.). Or simplify the kernel dramatically.

### 5. Cross-file / integration issues

* **Error handling**: Several places swallow exceptions with `except Exception: return None` (e.g. in `topk_selector_tilelang`, `topk_selector_metal`). This is acceptable for optional fast paths but the caller (in `cppmega`) should have clear fallback logging when Path C/B returns None. The one-shot warnings added in Wave-2 help, but make sure they don't get suppressed in production training runs.
* **Device / dtype consistency**: The DSA kernel does good contiguity enforcement. The topk paths promote to float32 internally — correct for comparison stability.
* **Sparse loss NaN guard**: Good addition, but verify that patching `index_mask[..., 0]` doesn't break the causal mask logic downstream (it shouldn't, because causal already zeros invalid positions).

### Summary of most critical correctness findings (post-fix-round-2)

1. **dsa_splitk_indexer_loss.py**: Conditional `(1,1)` vs `(AH, BLOCK_SQ)` fragment allocation in stage2 when `USE_MD_PRE=False` — **shape/type mismatch risk**. Fix by making shape a single constexpr expression.
2. **topk_selector.py**: Broken local top-K insertion loop in Path C (`_path_c_kernel_for`) due to suspicious `elif` break condition — **incorrect top-k results**. Also fragile regex rewrite of merge round.
3. **dsa_splitk_indexer_loss.py**: Verify that `USE_MD_PRE` condition is computed correctly for **both** CUDA and Metal (not Metal-only logic leaking to CUDA path).
4. **General**: Ensure `EPS` is defined in the KL term. Audit all `denom <= 0` guards for identical handling between attention and index softmax paths.

The Wave-2 changes (topk_indices bounds check, fully-masked row NaN guard, M_pre/D_pre budget gate intent) are mostly in the right direction, but the implementation details (especially the placeholder fragments and topk Path C kernel) introduced or left correctness risks.

These should be addressed before declaring the DSA / topk ports production-ready. The fp8 paths look cleaner by comparison.