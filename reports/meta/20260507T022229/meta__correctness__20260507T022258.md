---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 2b9310ef73638a41b8de47566c41891b55218679
timestamp: 2026-05-07T02:22:58.881921+00:00
files: ['cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
I tried to pull the code from your uploads but wasn't able to retrieve the file contents just now, so I can't point to specific `file:line` bugs yet.

To give you a correctness-focused review with concrete `file:line` cites, I need to actually see the diff or the relevant sections. Could you paste the key snippets or the diff itself here? 

**What I'll check once I have the code:**
1. **Off-by-one / indexing**: Split-K, top-k selection, and indexer logic are classic spots. I’ll trace loop bounds, buffer sizes, and `arange` vs `range` usage.
2. **FP8 amax / scaling**: Look for `NaN/Inf` propagation, zero amax leading to div-by-zero, and mismatched dtypes when downcasting. Check if `amax` is computed per-tensor vs per-channel and whether the scale is applied before/after the op.
3. **Null/None handling**: Any place you index, cast, or reduce—will verify guards for `None`, empty tensors, and `shape[0] == 0` edge cases.
4. **Error handling**: Python/C++ boundary for `fp8_vecmat_path_c.py`—check if C++ exceptions are converted to Python exceptions or silently swallowed. Also verify return-code checks.
5. **Race conditions**: In `dsa_splitk_indexer_loss.py`, I’ll look for shared state across threads/tiles, atomic vs non-atomic accumulations, and missing barriers in MLX kernels.
6. **Regressions**: Compare default values and argument parsing vs previous behavior. Flag if `topk_selector.py` changed `k > n` handling or if stable-sort guarantees changed.

**MLX wiring cross-check**  
You mentioned Wave-1 grok found no HIGH/MED blockers. I’ll verify:
- Correct `mx.array` vs `mx.custom_function` usage for gradients
- Stream/device placement matches inputs
- Shape/dtype checks before calling into C++/metal kernels in `fp8_vecmat_path_c.py`
- Whether `compile` is used correctly and doesn’t hide shape errors

Drop the diff or the 4 files’ contents and I’ll go line-by-line for bugs and cite exact locations. What’s the most critical file you want me to start with?