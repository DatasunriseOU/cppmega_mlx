---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: HEAD~1..HEAD
base_ref: 80c94135568038f8c15e36a6ed76251e1046c401
head_ref: ab17d216746312e902018bbe9c793365c1879c19
timestamp: 2026-05-07T09:11:26.989282+00:00
files: ['cppmega_mlx/nn/_tilelang/_msl_transform.py', 'scripts/bench_tilelang_fp8_path_c.py']
---
**P0/P1 correctness bugs (must-fix; introduced or exposed by recent Wave 3 hardening changes):**

1. **_msl_transform.py:464-472 (_parse_buffer_param_names) — incorrect buffer name extraction for TileLang-emitted signatures (breaks dispatch input count validation in Path C vecmat/matmul paths)**

   The parser does:
   ```python
   clean = re.sub(r"\[\[.*?\]\]", "", decl).strip()
   m = re.search(r"(\w+)\s*$", clean)
   ...
   if "device " in clean:
       names.append(var)
   ```
   It skips `blockIdx`/`threadIdx` (good) but relies on a fragile regex that only matches the *last* word before end-of-decl. TileLang Metal codegen (post-#799) often emits qualifiers like `device const float8_e4m3*` or `const device half*` with attributes, and the `[[.*?\]]` strip is too greedy/non-anchored. More critically, some parameters (scales, offsets, or packed buffers) may appear without a literal `"device "` substring if TileLang uses `device` as a type qualifier or in pointer syntax.

   This causes `parsed = lowering.buffer_param_names` to under-count or mis-order inputs. Then in `dispatch()` (lines 178-192):
   ```python
   expected_inputs = len(parsed) - len(output_dtypes)
   if ... or len(inputs) != expected_inputs:
       raise MSLDispatchUnsupported(...)
   ```
   → silent fallback to pure-MLX Path B (or garbage if count coincidentally matches), breaking the "full Path C dispatch" contract for vecmat (see bench: `fp8_scaled_vecmat_path_c`).

   **Regression note**: Pre-Wave 3 the validation wasn't there; the hardening exposed this. Test with `make_fp8_vecmat_reduce_kernel` or a matmul PrimFunc.

   Fix: Use a more robust parser (e.g., split on commas, then regex for `device\s+\w+\s*(\w+)` or leverage TVM/TileLang signature introspection if exposed). Or expose buffer names directly from the lowering artifact.

2. **_msl_transform.py:312-318 (dispatch when lowering=None) + line 185 (lowering validation) — mismatched error paths and swallowed None kernel**

   `dispatch()` calls `kernel(...)` without checking if `kernel is None` (from `make_metal_kernel` returning None on `not can_run_metal()` or missing `mx.fast.metal_kernel`). The early `if any(...size == 0...)` raise happens, but the main path assumes `kernel` is callable.

   In Path C callers (e.g., `fp8_vecmat_path_c.py` not shown but implied by bench), this can lead to `TypeError: 'NoneType' is not callable` instead of the clean `MSLDispatchUnsupported` that `msl_dispatch_status` and `_metal_kernel_constructor` intend.

   Also, when `lowering is not None` the input-count check can raise before reaching the kernel call, but the docstring and `MSLDispatchUnsupported` contract assume all failures funnel through that exception. Inconsistent.

   **Severity**: P1 in production inference (crashes instead of graceful Path B fallback).

3. **bench_tilelang_fp8_path_c.py: ~1200-1210 (_bench_paired_vecmat_mlx and _make_path_c_vecmat_runner) + line  ~650 (_require_runtime) — None-handling regression in Path C vecmat dispatch**

   `_make_path_c_vecmat_runner` does:
   ```python
   out = fp8_scaled_vecmat_path_c(...)
   if out is None:
       raise RuntimeError("... returned None")
   ```
   But the paired `_bench_paired_callables` and single-path `_bench_path_c_vecmat_mlx` now propagate this as a hard failure in the stats (ok=False). Pre-Wave 3 changes to `_msl_transform` (lazy preload, narrow excepts, cache) made TileLang lowering more likely to return None or raise `MSLDispatchUnsupported` early (e.g., missing intrinsics, target mismatch, libz3 preload failure).

   In `_bench_sparse_status` and other status probes, similar None checks exist, but the strict gate (`_shape_row_strict_ok`) and `_tokens_per_s_no_worse` do not robustly handle the new "unavailable" states from lowered kernels.

   Edge case: empty tensors or unsupported dtypes now hit the new validation in `dispatch()` (good) but bench assumes success path.

**P2 correctness issues (important but lower urgency):**

4. **_msl_transform.py: ~550-560 (_lowering_cache_key) — unhashable pass_configs values silently skip caching (potential perf regression + subtle correctness drift across calls)**

   ```python
   try:
       cfg = frozenset((pass_configs or {}).items())
   except TypeError:
       return None  # no caching
   ```
   If any value in `pass_configs` (e.g., nested dict from Z3 options or lists) is unhashable, the entire lowering bypasses the `_LOWERING_CACHE`. Since Wave 3 added `pass_configs` support and the cache uses `id(prim_func)`, repeated calls with the same PrimFunc + config now re-lower (slow) instead of hitting cache. Not a correctness bug per se, but breaks the "idempotent lowering" assumption and can cause non-deterministic behavior if TVM lowering is non-deterministic (rare but possible with passes).

   Also, `_LOWERING_CACHE_KEEPALIVE.append(prim_func)` only happens on cache hit path — if skipped, GC pressure increases.

5. **_msl_transform.py: ~400 (_drop_alias_decl_if_unused) + _canonicalize_tilelang_builtin_aliases — comment/string masking is insufficient for decl removal**

   `_mask_msl_comments_and_strings` replaces content with spaces but keeps delimiters for offset preservation. Then `_strip_msl_comments_and_strings` fully removes them for the "used?" check. However, the decl regexes (`_TILELANG_BUILTIN_ALIAS_DECLS`) are applied to the *original* `body` (before full strip in some paths), and re.sub can leave partial lines or whitespace that affects subsequent parses.

   More critically: if a comment contains `blockIdx` or `threadIdx` (possible in TileLang debug output), the strip removes it, so the "unused" check may incorrectly drop the decl when the alias *is* used inside a commented block (unlikely but possible). Or vice versa.

   Edge case: multi-line strings/comments spanning decls.

6. **bench_tilelang_fp8_path_c.py: ~300-320 (_prepare_tilelang_import_environment) + reset_env_preparation — thread-safety vs. global state reset race in tests**

   The `_IMPORT_ENV_LOCK` protects the once-per-process guard, but `reset_env_preparation()` (test-only) clears globals under the same lock. If a bench thread is mid-preparation and a test calls reset, or multiple tests race, you can get partial env (e.g., stale `sys.path` or missing `TILELANG_DEV_BUILD_ROOT`).

   Also, `_purge_stale_imported_modules` and `_disable_stale_editable_import_finders` mutate `sys.modules`/`sys.meta_path` globally without per-test isolation beyond the lock. Fine for production (idempotent), but fragile in pytest with `--tb=no` or parallel workers.

**Performance suggestions (secondary to correctness; all low-risk):**

- **Lowering cache hit rate**: The key uses `id(prim_func)` + frozenset(pass_configs). Since callers (bench + inference kernels) often hold module-level PrimFunc objects, this is fine, but add a small LRU eviction or size limit if many distinct shapes/configs are probed (bench does shape sweeps).

- In `_as_metal_target_cached` (lru_cache(maxsize=4)) — good, but the uncached path still does repeated `tvm.target.Target(...)` construction for non-string inputs. Consider widening the cache or memoizing the dict-form parsing.

- `_preload_libz3_for_dev_tilelang`: The failed-attempts counter is process-global and never reset (except via test `reset_env_preparation`). On a host where libz3 is genuinely missing, every Path C dispatch pays the small retry cost up to `_MAX_FAILED_ATTEMPTS=3`. Acceptable, but could be per-kernel or tied to the lowering cache key.

- Bench: `_bench_paired_callables` alternates order to reduce bias — excellent. But the `samples_by_step` dicts grow with `--iters`; for very large iters consider sampling or summarizing on-the-fly.

**No regressions found in**:
- `metal_grid_for_lowering` (explicitly excluded per query; multiplication is correct).
- libz3 preload TOCTOU hardening (Wave 3 fix looks solid; direct `CDLL` + exception discrimination is better than exists()+dlopen).
- Narrowed except blocks (good; `ImportError`/`ModuleNotFoundError` now properly become `MSLDispatchUnsupported`).
- `_ensure_single_libtvm_ffi_image` guard.

**Recommendations**:
- Add unit tests for `_parse_buffer_param_names` with real TileLang-emitted MSL snippets (including FP8 vecmat signatures with `device const ...*` and attributes).
- In `dispatch()`, add explicit `if kernel is None: raise MSLDispatchUnsupported("metal_kernel unavailable")` before the call.
- For the buffer parser, consider falling back to TVM's `prim_func.params` or `buffer_map` introspection inside `lower_tilelang_to_msl_inline` and storing the exact ordered list.

These are the concrete, actionable issues tied to the attached files and recent changes. The code is otherwise quite robust (great caching, lazy preload, status helpers). Fix the buffer parsing first — it's the root cause of potential silent Path C fallbacks.