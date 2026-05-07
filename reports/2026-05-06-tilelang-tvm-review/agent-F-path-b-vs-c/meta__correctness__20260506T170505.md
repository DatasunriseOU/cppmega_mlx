---
aspect: correctness
provider: meta
model: meta-muse-spark
range: (uncommitted)
base_ref: 
head_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
timestamp: 2026-05-06T17:05:05.358257+00:00
files: ['cppmega_mlx/nn/_tilelang/sparse_mla.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_fp8_path_c.py', 'cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/mamba3.py', 'cppmega_mlx/nn/_tilelang/mamba3_path_c.py', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/_path_b_lowering.py', 'cppmega_mlx/nn/_tilelang/_mamba3_helpers.py', 'cppmega_mlx/nn/_tilelang/_mamba3_helpers_tilelang.py', 'cppmega_mlx/nn/_tilelang/__init__.py', 'docs/production_kernel_routing.md', 'bench/tilelang_ports/fp8_path_c_vs_path_b.json', 'bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'docs/upstream/_path_c_blockers_tracker.md', 'scripts/bench_tilelang_fp8_path_c.py', 'tests/test_tilelang_bench_harness.py', 'tests/test_tilelang_fp8_vecmat_path_c.py', 'tests/test_tilelang_msl_transform.py']
---
I tried to pull specific lines from `review_bundle.md` to do the cross-comparison, but the file search didn’t return any excerpts from the bundle. I can’t access the code contents directly right now, so I can’t give the concrete `file:line` citations you need.

To give you the focused, line-cited audit you asked for, I need the actual code excerpts. Can you either:

**Option 1: Paste the critical sections**
For each pair, paste:
1. The public function signature from both Path B and Path C files
2. The softmax/scale/accumulator loop from both
3. The routing block from `__init__.py` and the relevant part of `docs/production_kernel_routing.md`
4. Any `import _path_b_lowering` lines
5. The gating check in `_msl_transform.py`

**Option 2: Re-upload**
If you re-upload the bundle as separate files instead of one markdown, I can parse them and run the exact diff analysis.

### What I’ll check once I have the code
So you know what to expect, here’s the checklist I’ll run for each pair:

**1. INTERFACE divergence**
- Compare `def` signatures: args, `*kwargs`, defaults, `->` return annotations
- Check `dtype` and `shape` in docstrings vs actual `return`
- Verify `docs/production_kernel_routing.md` description matches `__init__.py` exports

**2. ALGORITHMIC divergence**
- Softmax: `max -> exp -> sum -> div` order, masking applied pre or post `max`
- Scale: `Q @ K.T * scale` vs `Q * scale @ K.T` - impacts fp16 overflow
- Accumulators: fp32 vs fp16/bf16 for `qk`, `pv` intermediates
- Masked tiles: `if mask.all()`: skip vs compute 0s. Off-by-one on tile boundaries common here

**3. PERFORMANCE-SHAPE divergence**
- Look for `if M == 1`, `if K % 128 == 0`, `if d_v <= 64` branches only in Path C
- Path B usually generic `for i in range(ceil_div(M, TM))`

**4. Dead-code `_path_b_lowering.py`**
- Grep all Path B files for `from _path_b_lowering import` or `import _path_b_lowering`
- If 0 hits, flag as dead-code SUSPICION, not deletion

**5. `_msl_transform.py` gating**
- Search for `if 'path_c' in kernel_name:` or decorator checks
- Risk: runs on both but no-ops for C, hiding bugs in B

**6. Routing vs exports**
- Map `docs/production_kernel_routing.md` table to `__all__` in `__init__.py`
- Flag kernels documented but not exported

**7. LOC delta root cause**
- If Path C >> Path B: likely inlined what B keeps in helpers. Check for duplicate matmul prologues
- If Path C << Path B: likely deleted legacy checks. Risk of regression on edge cases like `N=0`, `K=0`

If you paste just the 5 pairs of function signatures + `__init__.py` exports, I can start with item 1 right now and give concrete findings. Which option works best?