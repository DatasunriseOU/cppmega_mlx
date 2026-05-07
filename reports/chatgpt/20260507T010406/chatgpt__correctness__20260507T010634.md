---
aspect: correctness
provider: chatgpt
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
head_ref: 920264654224121c8f55da9173f03ff6929d3e69
timestamp: 2026-05-07T01:06:34.444081+00:00
files: ['cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
## Finding 1

- **Severity**: medium  
- **File**: `cppmega_mlx/nn/_tilelang/_msl_transform.py:452`  
- **Issue**: The `pass_configs` dict is passed directly to `tvm.transform.PassContext` via `dict(pass_configs)`. If `pass_configs` contains mutable values or unexpected types, TVM may raise runtime exceptions. There is no type validation or defensive copying beyond `dict()`.  
- **Fix**: Validate or sanitize `pass_configs` entries before passing them to `PassContext`. Example:

```python
safe_pass_configs = {k: v for k, v in pass_configs.items() if isinstance(k, str)}
with tvm.transform.PassContext(opt_level=3, config=safe_pass_configs):
    artifact = tl_lower(prim_func, target=metal_target)
```

---

## Finding 2

- **Severity**: medium  
- **File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:123`  
- **Issue**: `_filter_supported_pass_configs` swallows all exceptions during `PassContext` construction. If TVM throws an unexpected error (not `AttributeError`) due to misconfiguration, it will silently drop the pass config, potentially hiding correctness bugs in the lowered kernel.  
- **Fix**: Narrow the exception catch to `AttributeError` only and let other exceptions propagate:

```python
try:
    with tvm.transform.PassContext(opt_level=3, config={key: value}):
        pass
except AttributeError:
    if key not in _FP8_VECMAT_PATH_C_FILTERED_KEYS_LOGGED:
        _FP8_VECMAT_PATH_C_FILTERED_KEYS_LOGGED.add(key)
        print(f"[cppmega-mlx-cuz] dropping unsupported PassConfig key {key!r}", file=sys.stderr)
    continue
```

---

## Finding 3

- **Severity**: high  
- **File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:235`  
- **Issue**: `_uses_fp8_dot4_packed_macro` uses `K % 4 == 0` check for runtime alignment. If `K` is negative, this check still passes for multiples of 4 (e.g., `K=-4`), which could trigger unsafe calls to `T.metal_fp8_e4m3_dot4`. Negative K values are not guarded.  
- **Fix**: Add explicit non-negative check:

```python
k_aligned = (K >= 0) and (K % 4 == 0)
```

---

## Finding 4

- **Severity**: medium  
- **File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:281`  
- **Issue**: `lower_fp8_vecmat_msl` silently falls back if `artifact` does not have `kernel_source` or `rt_mod.get_source`. If TVM returns an unexpected artifact type, the function returns `None`, breaking downstream code that expects a string.  
- **Fix**: Add explicit exception or fallback string:

```python
if hasattr(artifact, "kernel_source"):
    return str(artifact.kernel_source)
rt_mod = getattr(artifact, "rt_mod", None)
if rt_mod is not None and hasattr(rt_mod, "get_source"):
    return str(rt_mod.get_source())
raise RuntimeError("Failed to extract kernel source from TileLang artifact")
```

---

## Finding 5

- **Severity**: low  
- **File**: `cppmega_mlx/nn/_tilelang/topk_selector.py:1012`  
- **Issue**: `_topk_path_c_pass_configs` caches results in `_TOPK_PATH_C_PASS_CONFIGS_CACHE`, but no thread-safety is provided. In multi-threaded MLX lowering, two threads could race on the first access, potentially calling `_topk_filter_supported_pass_configs` twice.  
- **Fix**: Protect cache with a simple lock:

```python
import threading
_cache_lock = threading.Lock()

def _topk_path_c_pass_configs() -> dict[str, Any]:
    ...
    global _TOPK_PATH_C_PASS_CONFIGS_CACHE
    if _TOPK_PATH_C_PASS_CONFIGS_CACHE is None:
        with _cache_lock:
            if _TOPK_PATH_C_PASS_CONFIGS_CACHE is None:
                _TOPK_PATH_C_PASS_CONFIGS_CACHE = _topk_filter_supported_pass_configs(
                    _TOPK_PATH_C_CANDIDATE_PASS_CONFIGS
                )
    return dict(_TOPK_PATH_C_PASS_CONFIGS_CACHE)
```

---

No off-by-one arithmetic errors or swallowed critical exceptions were found outside of the noted issues. The rest of the Z3 wiring logic appears consistent with the intended TileLang lowering contracts.