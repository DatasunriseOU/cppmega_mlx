---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
timestamp: 2026-05-06T17:12:18.377404+00:00
files: ['cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/fp8_msl_kernels.py', 'cppmega_mlx/nn/_tilelang/mamba3.py', 'cppmega_mlx/nn/_tilelang/mamba3_path_c.py', 'cppmega_mlx/nn/_tilelang/_mamba3_helpers.py', 'cppmega_mlx/nn/_tilelang/_mamba3_helpers_tilelang.py', 'cppmega_mlx/nn/_tilelang/m2rnn.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/_path_b_lowering.py', 'cppmega_mlx/nn/_tilelang/__init__.py', 'tests/test_tilelang_fp8_vecmat_path_c.py', 'tests/test_tilelang_mamba3.py', 'tests/test_tilelang_mamba3_path_c.py', 'tests/test_tilelang_m2rnn.py', 'tests/test_tilelang_topk.py', 'tests/test_tilelang_mamba3_helpers.py', 'docs/production_kernel_routing.md', 'bench/tilelang_ports/fp8_path_c_vs_path_b.json', 'bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'docs/upstream/_path_c_blockers_tracker.md', 'scripts/bench_tilelang_fp8_path_c.py', 'tests/test_tilelang_bench_harness.py', 'tests/test_tilelang_msl_transform.py']
---
## Finding 1
**Severity**: high  
**File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:203`  
**Issue**: The fast-path branch (`_uses_fp8_scaled_matmul_vecmat_macro` → `vec==4`, `K%4==0`, `reduce_threads==32`) now emits a `T.fp8_scaled_matmul` `PrimFunc` that declares `C: T.Tensor((1, _FP8_VM_N), "float32")` (2-D) while `_fp8_vecmat_kernel_for` still hard-codes `output_shape=(N,)` (1-D flat) and passes it to `mx.fast.metal_kernel`. The runtime `fp8_scaled_vecmat_path_c` then applies a post-hoc `reshape` only when `out.ndim != 1`. This introduced a buffer-allocation / stride / indexing mismatch between the lowered MSL kernel and the Metal runtime buffer (semantic delta vs. the previous hand-written flat-`C` fast path in the same file). Visible regression risk for any `N` that hits the macro path.  
**Fix**:
```diff
diff --git a/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py b/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py
index caf49a7..0000000 100644
--- a/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py
+++ b/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py
@@ -429,7 +429,11 @@ def _fp8_vecmat_kernel_for(
         )
     input_names = ["A", "A_scale", "B", "B_scale"]
     source = lowering.body
     threadgroup = lowering.threadgroup
-    grid = _msl_transform.metal_grid_for_lowering(lowering)
+    grid = _msl_transform.metal_grid_for_lowering(lowering)
+    output_shape: tuple[int, ...] = (
+        (1, N) if _uses_fp8_scaled_matmul_vecmat_macro(...) else (N,)
+    )
     kernel = mx.fast.metal_kernel(...)
```

## Finding 2
**Severity**: medium  
**File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:620`  
**Issue**: The fast-path `if input_names == ["A", "A_scale", "B", "B_scale"]:` block in `fp8_scaled_vecmat_path_c` now receives a 2-D output from the macro `PrimFunc` but unconditionally does `return out if out.ndim == 1 else out.reshape((n,))`. This band-aid (introduced together with the macro switch) silently masks the shape mismatch from Finding 1 and can produce wrong views or copy overhead on every fast-path call; non-fast-path kernels still expect flat output. Regression to the previous always-flat contract.  
**Fix**:
```diff
diff --git a/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py b/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py
index caf49a7..0000000 100644
--- a/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py
+++ b/cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py
@@ -620,7 +620,7 @@ def fp8_scaled_vecmat_path_c(
         outputs = cast(...)(
             ...
         )
-        out = outputs[0]
-        return out if out.ndim == 1 else out.reshape((n,))
+        out = outputs[0].reshape((n,)) if out.ndim != 1 else outputs[0]
+        return out
     input_map = ...
```

## Finding 3
**Severity**: low  
**File**: `cppmega_mlx/nn/_tilelang/_msl_transform.py:410`  
**Issue**: New `_tilelang_inline_scalar_builtin_decls` (and callers in `_inline_tilelang_kernel_body`) relies on regex-based `_uses_code_identifier` / `_declares_code_identifier` + a post-strip `re.search(r"\bgridThreadIdx\s*\.", ...)` heuristic to decide `uint` vs `uint3`. This rewrite pass now applies to every kernel lowered via `_inline_tilelang_kernel_body` (including the fp8 vecmat macro path). Fragile for future kernels that use the identifier in non-standard ways (e.g. qualified names, strings that survive stripping, or conflicting manual decls); could introduce undeclared-identifier errors or wrong-type declarations.  
**Fix**:
```diff
diff --git a/cppmega_mlx/nn/_tilelang/_msl_transform.py b/cppmega_mlx/nn/_tilelang/_msl_transform.py
index 896ac90..0000000 100644
--- a/cppmega_mlx/nn/_tilelang/_msl_transform.py
+++ b/cppmega_mlx/nn/_tilelang/_msl_transform.py
@@ -410,7 +410,7 @@ def _tilelang_inline_scalar_builtin_decls(inner: str) -> str:
         code = _strip_msl_comments_and_strings(inner)
-        if re.search(r"\bgridThreadIdx\s*\.", code):
+        if re.search(r"\bgridThreadIdx\s*\.\w", code):  # require member access
             decls.append("    uint3 gridThreadIdx = thread_position_in_grid;\n")
         else:
             decls.append("    uint gridThreadIdx = thread_position_in_grid.x;\n")
```

**No findings.**

All code changes visible in this chunk (scripts/bench_tilelang_fp8_path_c.py, tests/test_tilelang_bench_harness.py, tests/test_tilelang_fp8_vecmat_path_c.py, and tests/test_tilelang_msl_transform.py) introduce no off-by-one errors, incorrect error handling, swallowed exceptions, mismatched types, race conditions, incorrect defaults, broken None handling, mishandled edge cases, or regressions to existing behaviour.

- Bench statistics, paired timing, tokens/s checks, and strict-policy logic (_tokens_per_s_no_worse, _shape_row_strict_ok, _worst_paired_steps) are consistent with the new data shapes and always write the receipt on failure.  
- Runtime-source vs diagnostic-source separation in vecmat metadata tests is correctly enforced.  
- MSL transform alias-rewrite tests pass and match the updated inline-body behaviour.  
- No dead-code paths, dispatch fall-throughs, or untested public APIs are present in the visible diff.  

(Path C source delta vs fp8_msl_kernels.py, mamba3 equivalence, _msl_transform application to the full bundle, m2rnn hot paths, topk dispatch table, __init__.py exports, and _path_b_lowering status are outside this chunk and will be covered by the orchestrator.)