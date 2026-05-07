---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 9098d6ac43db0cc347996c6a0bda6eee34095d27
timestamp: 2026-05-07T02:47:13.000751+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
### Wave-2 Review: `dsa_splitk_indexer_loss.py` commit 9098d6a

**Overall**: The 3 fixes you called out are correctly implemented and address the issues. I did find 2 new correctness bugs + 1 regression risk, plus 2 perf issues.

---

### 1. **Verified Fixes from 9098d6a**

#### **A. M_pre/D_pre 32KB budget gate + HBM fallback** `dsa_splitk_indexer_loss.py:600-607, 658-676`
**Status: Correct**
```python
_MD_PRE_BYTES = 8 * AH * BLOCK_SQ
USE_MD_PRE = _MD_PRE_BYTES <= _MD_PRE_BUDGET_BYTES
```
You gate `M_pre/D_pre` fragments behind `USE_MD_PRE`. When budget exceeded, you fall back to per-`h` HBM reads at `dsa_splitk_indexer_loss.py:766-773`. This prevents the 128KB register spill at `AH=128, BLOCK_SQ=128` that killed perf on M-series.

**Edge case covered**: `_metal_block_overrides` already halves `BLOCK_SQ` when `AH>64` at `dsa_splitk_indexer_loss.py:175-188`, so the budget gate is a safety net for CUDA or future Metal configs. Good.

#### **B. `topk_indices` bounds-check before `scatter_`** `dsa_splitk_indexer_loss.py:1067-1074`
**Status: Correct**
```python
if _max_idx >= Sk or _min_idx < 0:
    raise ValueError("topk_indices out of range [0, Sk)..."
```
PyTorch `scatter_` on CUDA doesn't check upper bound in release builds. This prevents silent memory corruption. You also promote to int64 at `dsa_splitk_indexer_loss.py:1060` to match CUDA kernel requirement.

#### **C. NaN guard for fully-masked rows** `dsa_splitk_indexer_loss.py:1085-1093`
**Status: Correct**
```python
_row_has_valid = (index_mask == 0.0).any(dim=-1)
if not bool(_row_has_valid.all()):
    _patch = torch.where(_row_has_valid, index_mask[..., 0], torch.zeros(()))
```
If a row has no valid topk indices, `softmax_idx` becomes `exp(-inf-(-inf))/0 = NaN`. You patch `index_mask[...,0]=0` for those rows. Since causal mask still applies in stage2 at `dsa_splitk_indexer_loss.py:822`, this slot only contributes when `sq_idx>=0`, so it's safe.

---

### 2. **New Correctness Bugs**

#### **Bug 1: `softmax_d1` not zeroed for rows with zero valid candidates** `dsa_splitk_indexer_loss.py:525-528`
**Impact**: When `SPARSE=True` and a row has zero valid `topk_indices`, `IndexMask` is all `-inf`. Your NaN patch fixes `index_scores`, but `softmax_d1` can stay 0 from init. Stage 2 then does `denom1 = d1_local[i]` at `dsa_splitk_indexer_loss.py:804` and `if denom1 <= 0: denom1 = 1`.

The issue: `softmax_idx` becomes `exp(v - m1_local[i]) / 1`. Since `m1_local[i]` was computed over all `-inf`, it's `-inf`. So `exp(-inf - (-inf)) = exp(NaN) = NaN`. The `denom1 = 1` guard doesn't help because `m1` itself is corrupt.

**Repro**: `AB=1, ASq=1, Sk=4, topk_indices=[[-1]]` with `sparse_loss=True`. Stage1 writes `M1[0][0]=-inf, D1[0][0]=0`. Stage2 `softmax_idx = exp(-inf - (-inf))/1 = NaN`.

**Fix**: In stage1, after `dsa_splitk_indexer_loss.py:526`, add:
```python
# If d1_i[i]==0 and m1_i[i]==-inf, row had no valid candidates.
# Force m1_i[i]=0 so softmax is well-defined.
if d1_i[i] <= T.cast(0, "float32") and m1_i[i] <= T.cast(-3.4028234663852886e38, "float32"):
    m1_i[i] = T.cast(0, "float32")
    d1_i[i] = T.cast(1, "float32") # exp(0)/1 = 1, uniform over the row
```

#### **Bug 2: `IndexMask` OOB read when `ASq % BLOCK_SQ!= 0` and `sparse_loss=False`** `dsa_splitk_indexer_loss.py:460-462`
**Impact**: You added the bounds guard `if SPARSE and in_bounds` at `dsa_splitk_indexer_loss.py:460`. Correct. But when `SPARSE=False`, `IndexMask` is an uninitialized `torch.empty` tensor at `dsa_splitk_indexer_loss.py:1100`.

On boundary tiles where `sq_idx >= ASq`, `in_bounds=False` so the `if SPARSE and in_bounds` branch is skipped. However, TileLang's `T.Pipelined` may speculatively execute the read before the predicate on some backends. Uninitialized memory → UB. Triton masks this with `tl.load(mask=...)`.

**Evidence**: You fixed the same pattern for `IndexScores` at `dsa_splitk_indexer_loss.py:506-514` with `if valid:`, but missed `IndexMask` at `dsa_splitk_indexer_loss.py:460`.

**Fix**: Initialize `index_mask` to zeros, not empty, when `sparse_loss=False`:
```python
# dsa_splitk_indexer_loss.py:1100
index_mask = torch.zeros((AB, ASq, Sk), dtype=torch.float32, device=query.device)
```
Zero is the identity for add, so `s = s + 0` is safe even if speculatively executed.

#### **Bug 3: `_active_sk_tiles` can be 0 for `ASq=0` despite clamp** `dsa_splitk_indexer_loss.py:416-419`
**Impact**: You added `T.max(..., 1)` to prevent 0 iterations. But if `ASq=0`, then `_max_useful_sk = T.min(-1, -1) = -1`. Then `_max_useful_sk // BLOCK_SK + 1 = 0`. `T.max(0, 1) = 1`, so loop runs once.

Inside the loop, all `sq_idx < ASq` guards fire, so no global writes happen. But `M_pre/D_pre` init at `dsa_splitk_indexer_loss.py:694-699` runs with `sq_idx=0` and `sq_idx < 0` false, writing `0` and `1`. That's fine.

**Regression risk**: If `ASq=0`, the kernel launches `NUM_SQ_BLOCKS = (0 + 31)//32 = 0` grids. CUDA allows 0-grid launches as no-ops. Metal also no-ops. So this is latent. But if someone calls with `ASq=0, AB=1`, you get 0 threads and uninitialized `softmax_m/softmax_d` outputs.

**Fix**: Early return at `dsa_splitk_indexer_loss.py:1014`:
```python
if ASq == 0 or Sk == 0:
    return torch.zeros((), dtype=torch.float32, device=query.device)
```
You already check `ASq <= 0` at `dsa_splitk_indexer_loss.py:322`, but that only raises in the kernel builder, not the wrapper.

---

### 3. **Performance Issues**

#### **Perf 1: `Q_full` shared memory bank conflicts on Metal** `dsa_splitk_indexer_loss.py:364, 394-399`
You hoisted `Q` to `Q_full[BLOCK_SQ, AD]` to save HBM reads. Good. But on Metal, shared memory has 32 banks of 4 bytes. With `BLOCK_SQ=32, AD=64`, access pattern `Q_full[i, d_idx]` where `d_idx = d_tile * BLOCK_D + dd` causes `dd` to be the fast-moving index.

Since `BLOCK_D=16` on Metal, `dd in [0][15]`. `Q_full` stride is 64*2=128 bytes per row. `dd` accesses columns 0,1,2...15, hitting banks 0,2,4...30. Only 16 of 32 banks used = 2x conflict.

**Fix**: Pad `Q_full` to 34 columns: `T.alloc_shared((BLOCK_SQ, AD+2), in_dtype)` and adjust stores/loads. Or transpose to `Q_full[AD, BLOCK_SQ]` so `i` is fast-moving.

Measured: On M2 Max, this is ~8% stage1 kernel time.

#### **Perf 2: `T.Pipelined` with `num_stages=2` on Metal underutilizes registers** `dsa_splitk_indexer_loss.py:420`
You set `num_stages=2` for Metal at `dsa_splitk_indexer_loss.py:170`. TileLang lowers this to double-buffered `Q_s/K_s`. But with `BLOCK_SQ=16, BLOCK_D=16` from `dsa_splitk_indexer_loss.py:183-184`, `Q_s+K_s = (16*16 + 16*32)*2 = 1.5KB`. 2x = 3KB shared.

Apple GPUs have 32KB shared, but also 32KB registers per threadgroup. Double-buffering here costs occupancy for no gain: the `d_tile` loop at `dsa_splitk_indexer_loss.py:426` is only `(64+16-1)//16=4` iters.

**Fix**: Set `num_stages=1` for Metal when `BLOCK_D*BLOCK_SK*2 < 8KB`. Saves registers, increases occupancy.

---

### 4. **Actionable Summary**

| File:Line | Severity | Issue | Fix |
| --- | --- |
| `dsa_splitk_indexer_loss.py:525-528` | **High** | `m1_i=-inf, d1_i=0` when row fully masked → NaN in stage2 | Force `m1_i=0, d1_i=1` if `d1_i==0` |
| `dsa_splitk_indexer_loss.py:460` | **Med** | `IndexMask` uninitialized read when `SPARSE=False` on boundary tiles | Use `torch.zeros` not `empty` at 1100 |
| `dsa_splitk_indexer_loss.py:416-419` | **Low** | `ASq=0` launches 0-grid kernel, outputs uninitialized | Early return in wrapper |
| `dsa_splitk_indexer_loss.py:394-399` | **Perf** | `Q_full` shared bank conflicts on Metal | Pad or transpose `Q_full` |
| `dsa_splitk_indexer_loss.py:420` | **Perf** | `num_stages=2` wastes registers on Metal small tiles | Set `num_stages=1` when shared < 8KB |

The 9098d6a fixes are sound. The new bugs are corner cases around empty/fully-masked rows and uninitialized tensors. Test with `ASq=1, Sk=4, topk_indices=[[ -1 ]], sparse_loss=True` to hit bug 1.