---
aspect: performance
provider: chatgpt
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
head_ref: 920264654224121c8f55da9173f03ff6929d3e69
timestamp: 2026-05-07T01:10:12.689823+00:00
files: ['cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py']
---
## Finding 1

- **Severity**: medium  
- **File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:88`  
- **Issue**: `_filter_supported_pass_configs` probes each candidate key by instantiating a `tvm.transform.PassContext` in a loop. This happens at runtime the first time `_fp8_vecmat_pass_configs()` is called. Although cached, the initial probe involves repeated FFI calls to TVM for every candidate key, which can be slow (especially if the number of candidate keys grows).  
- **Fix**: Consider precomputing supported keys at module load or using a single `PassContext` probe with all candidates at once to reduce repeated FFI instantiations.

---

## Finding 2

- **Severity**: medium  
- **File**: `cppmega_mlx/nn/_tilelang/topk_selector.py:177`  
- **Issue**: Similar to the FP8 vecmat case, `_topk_filter_supported_pass_configs` probes each candidate key individually using `with tvm.transform.PassContext(...)` inside a loop. This incurs a runtime FFI call per key, which may add noticeable latency on first kernel lowering.  
- **Fix**: Batch all candidate keys into one `PassContext` probe or memoize results at module load to avoid repeated FFI calls.

---

## Finding 3

- **Severity**: low  
- **File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py:350`  
- **Issue**: `_uses_fp8_dot4_packed_macro` checks `K % 4 == 0` every time it's called. While trivial, in hot paths with millions of iterations (unlikely here, but possible in meta-programming loops) this could be slightly optimized by precomputing alignment for known buffer shapes.  
- **Fix**: For large batch kernels, compute `K % 4` once and pass as a flag to avoid repeated modulo operations.

---

## Finding 4

- **Severity**: low  
- **File**: `cppmega_mlx/nn/_tilelang/_msl_transform.py:452`  
- **Issue**: `lower_tilelang_to_msl_inline` now wraps the lowering in a `PassContext` per call. If called repeatedly for small kernels, the context creation and teardown may add overhead.  
- **Fix**: For kernels that are lowered repeatedly with the same `pass_configs`, consider memoizing the `PassContext` or the lowered artifact to avoid redundant lowering work.

---

## Finding 5

- **Severity**: info  
- **File**: `cppmega_mlx/nn/_tilelang/sparse_mla.py:71`  
- **Issue**: Z3 PassConfig wiring notes are informational; no performance regression exists because these kernels are direct MSL and do not interact with TileLang lowering.  
- **Fix**: None needed; this is a performance-neutral comment.

---

No critical performance regressions were introduced. Main potential hot-path concern is the per-key FFI probing in `_filter_supported_pass_configs` and `_topk_filter_supported_pass_configs` during first lowering. Subsequent calls are cached.