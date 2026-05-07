---
aspect: performance
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
timestamp: 2026-05-06T17:11:02.055834+00:00
files: ['cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py', 'cppmega_mlx/nn/_tilelang/mamba3.py', 'cppmega_mlx/nn/_tilelang/mamba3_path_c.py', 'cppmega_mlx/nn/_tilelang/_mamba3_helpers.py', 'cppmega_mlx/nn/_tilelang/_mamba3_helpers_tilelang.py', 'cppmega_mlx/nn/_tilelang/m2rnn.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/_path_b_lowering.py', 'cppmega_mlx/nn/_tilelang/__init__.py', 'tests/test_tilelang_fp8_vecmat_path_c.py', 'tests/test_tilelang_mamba3.py', 'tests/test_tilelang_mamba3_path_c.py', 'tests/test_tilelang_m2rnn.py', 'tests/test_tilelang_topk.py', 'tests/test_tilelang_mamba3_helpers.py', 'docs/production_kernel_routing.md', 'bench/tilelang_ports/fp8_path_c_vs_path_b.json', 'bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'docs/upstream/_path_c_blockers_tracker.md', 'scripts/bench_tilelang_fp8_path_c.py', 'tests/test_tilelang_bench_harness.py', 'tests/test_tilelang_msl_transform.py']
---
## Finding 1
**Severity**: high  
**File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:620`  
**Issue**: The public hot-path entrypoint `fp8_scaled_vecmat_path_c` (and its helper `_fp8_vecmat_kernel_for`) performs full TileLang prim_func construction + `lower_tilelang_to_msl_inline` + `mx.fast.metal_kernel` instantiation on every call. No `@lru_cache` or module-level memoization is present in the diff. For recurrent use in Mamba3/M2RNN forward/backward passes (where FP8 vecmat is invoked per-token or per-layer with fixed N/K), this introduces repeated lowering/compilation overhead and can cause memory growth from accumulating kernel objects. The macro switch (new `T.fp8_scaled_matmul` fast path) does not mitigate it.  
**Fix**:  
```diff
+ from functools import lru_cache
...
+ @lru_cache(maxsize=128)
 def _fp8_vecmat_kernel_for(
     N: int, K: int, outputs_per_block: int = 4, ...
 ) -> tuple[...]:
```

## Finding 2
**Severity**: medium  
**File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:634`  
**Issue**: After the new macro fast-path (`_uses_fp8_scaled_matmul_vecmat_macro` branch), the prim_func now declares `C: T.Tensor((1, _FP8_VM_N), "float32")` (was flat `(N,)`). The runtime wrapper always does `out if out.ndim == 1 else out.reshape((n,))` on the returned array. This reshape is now executed on every FP8 vecmat call (visible hot-path change vs prior flat-output design) and introduces a potential view/copy/sync point in the critical path even though MLX reshape is usually cheap.  
**Fix**:  
```diff
-        out = outputs[0]
-        return out if out.ndim == 1 else out.reshape((n,))
+        out = outputs[0]
+        return out.reshape((n,)) if out.ndim == 2 else out  # explicit, zero-cost view when possible
```

## Finding 3
**Severity**: medium  
**File**: `cppmega_mlx/nn/_tilelang/_msl_transform.py:426`  
**Issue**: The `_inline_tilelang_kernel_body` rewrite (used by FP8 vecmat Path C lowering) now unconditionally calls the new `_tilelang_inline_scalar_builtin_decls` which performs two regex passes (`_uses_code_identifier` + `_declares_code_identifier` + `_strip_msl_comments_and_strings`) over the entire kernel body string for every kernel in the bundle. This string-processing overhead occurs at every lowering (including the hot-path macro case); fragile identifier detection can silently inject wrong decls or miss cases when kernels use `gridThreadIdx`/`simdLaneIdx` in complex ways. No bypass for non-vecmat kernels in the visible code.  
**Fix**:  
```diff
 def _inline_tilelang_kernel_body(inner: str) -> str:
     body = (
         "    uint3 blockIdx = threadgroup_position_in_grid;\n"
         "    uint3 threadIdx = thread_position_in_threadgroup;\n"
-        + _tilelang_inline_scalar_builtin_decls(inner)
+        # move builtin decls into TileLang DSL when possible; keep string rewrite only for legacy
+        + (_tilelang_inline_scalar_builtin_decls(inner) if "gridThreadIdx" in inner or "simdLaneIdx" in inner else "")
     )
```

## Finding 4
**Severity**: info  
**File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:480`  
**Issue**: `canonical_vecmat_runtime_body` (and its export in `__all__`) is now documented as "Reference Path B-style body kept for source comparisons, not runtime" after the macro switch. It is dead for production Path C but still present and parseable; minor maintenance burden and potential for accidental reuse in tests/benches.  
**Fix**: Move to a test-only helper or docstring; remove from `__all__` if no longer referenced by visible tests (e.g. `test_tilelang_fp8_vecmat_path_c.py`).