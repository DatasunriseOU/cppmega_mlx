---
aspect: correctness
provider: grok
model: gpt-5-5-pro
range: HEAD~1..HEAD
base_ref: ebbb84b61ea744483351f1e06ca543903aa90aba
head_ref: 60f0005fd74aedeb0a6039b1e4f74a8c970ee7a4
timestamp: 2026-05-07T01:24:48.116316+00:00
files: ['cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py']
---
Here is a focused correctness review of the provided `fp8_vecmat_path_c.py` file (the TileLang Path C implementation). I prioritized **bugs introduced or exposed by this diff/wave** (eb bb84b on top of prior MLX-wiring), especially around shape handling, macro vs. fallback paths, dispatch, error swallowing, edge cases, and regressions vs. the hand-written Path B `fp8_scaled_vecmat` contract. Severity is P0 (crash/incorrect output always), P1 (incorrect in common cases or silent wrong path), P2 (edge/rare but breaks contract or future-proofing).

I reference **file:line** based on the attached file content (lines as provided; approximate for long functions). Findings are concrete and actionable.

### P0/P1 Critical Correctness Bugs

1. **P0: Incorrect output shape handling in dot4 packed macro path — leads to wrong results or shape mismatch on dispatch**  
   **Location**: `make_fp8_vecmat_reduce_kernel` (around lines 180-200 for the `@T.prim_func` dot4 branch) and `_fp8_vecmat_kernel_for` (lines ~480-500, specifically the comment and `if _uses_fp8_dot4_packed_macro`).  
   In the dot4 branch, `C: T.Tensor((1, _FP8_VM_N), "float32")` (note the extra leading dim). In non-dot4 branches, it's `(N,)`.  
   Then in `_fp8_vecmat_kernel_for`:
   ```python
   if _uses_fp8_dot4_packed_macro(...):
       output_shape = (1, N)
   else:
       output_shape = (N,)
   ```
   Later dispatch (`fp8_scaled_vecmat_path_c` lines ~550-580) does:
   ```python
   out = outputs[0]
   return out if out.ndim == 1 else out.reshape((n,))
   ```
   **Issue**: The `mx.fast.metal_kernel` call (or `_msl_transform.dispatch`) receives `output_shapes=(output_shape,)` where shape can be `(1, N)`. MLX Metal kernels expect consistent buffer shapes; the reshape hack assumes the *returned* tensor has ndim==2 only for the macro case, but the lowering + kernel binding may produce a 1D view or mismatched strides. This breaks when the kernel writes to `C[0, col]` vs. `C[col]`.  
   **Regression**: Path B hand-written kernel uses flat `(N,)` everywhere. This introduces a shape divergence that wasn't present before the TileLang path.  
   **Fix**: Make the PrimFunc *always* declare `C` as `(N,)` (or consistently `(1, N)` and handle reshape upstream). Or normalize inside the kernel body via `T.reshape` or pointer arithmetic. Test with small `N,K` where dot4 path activates (K%4==0).

2. **P1: `_uses_fp8_dot4_packed_macro` called with runtime `K` but decision baked into cached kernel**  
   **Location**: `make_fp8_vecmat_reduce_kernel` (line ~140: `if _uses_fp8_dot4_packed_macro(vec=vec, K=K)` — note `K` from globals/dataclass) and `_fp8_vecmat_kernel_for` + `@lru_cache` (lines ~460-490).  
   The lru_cache on `_fp8_vecmat_kernel_for` keys on `(N, K, ...)` so different K%4 cases get different kernels — good. But the helper itself (lines 290-320) has a debug assertion only when env var set, otherwise silent fallback.  
   **Issue**: `_normalize_vecmat_inputs` already rejects K%4 !=0 (line ~520), so the assertion path is dead except for direct `make_fp8_vecmat_reduce_kernel` calls. But the dot4 branch uses `T.metal_fp8_e4m3_dot4` with packed `uint32` access assuming 4-byte words. If any misalignment sneaks through (e.g., future shape changes), it silently produces garbage (wrong dot products).  
   **Regression risk**: Prior wiring (9202646/9a668ea) likely didn't have this packed macro split.  
   **Fix**: Remove the debug-only assertion or make it always raise in non-debug (or tie it to a stronger runtime check). Document that dot4 path requires K%4==0 *and* vec==4.

3. **P1: C tensor indexing mismatch in dot4 vs. non-dot4 kernels**  
   **Location**: Dot4 PrimFunc (lines ~190-200): `C[0, col] = ...`  
   Non-dot4 branches (lines ~220-230 and ~260-270): `C[col] = ...` (or with leading dim in some).  
   **Issue**: When TileLang lowers the dot4 PrimFunc, the generated MSL writes `C[0 * stride + col]` (or similar), while fallback writes flat. Combined with the shape difference above, this can cause out-of-bounds writes or incorrect values if the Metal buffer is bound as 1D. The `canonical_vecmat_runtime_body` (lines ~400-430) shows the *desired* Path-B-like flat `C[row]` — the generated code diverges from this.  
   **Fix**: Standardize the PrimFunc signature and indexing for C to always be flat `(N,)`; use `T.reshape` or adjust the access expression inside the if-branch.

4. **P1: Swallowed exceptions and silent fallback in lowering/dispatch**  
   **Location**: `_fp8_vecmat_kernel_for` (lines ~470-480: `try: ... except MSLDispatchUnsupported: return None` in caller `fp8_scaled_vecmat_path_c`), `lower_fp8_vecmat_msl` (lines ~350-370: broad `except Exception` in simplify), and `_filter_supported_pass_configs` (lines ~80-100: catches `AttributeError/KeyError/TypeError` silently per-key).  
   Also in `make_fp8_vecmat_reduce_kernel` (lines ~280-285: `try: apply_simplify except: return original`).  
   **Issue**: `MSLDispatchUnsupported` is caught and turns the fast path into `None` (fallback to slower path, presumably), but no logging/warning. PassConfig filtering logs only once but drops keys silently. Simplify failure swallows and uses un-simplified IR (potentially slower or incorrect on Metal). This hides bugs in Z3 passes or intrinsic registration (`_assert_path_c_metal_fp8_intrinsics_registered`).  
   **Regression**: Prior hand-written Path B had no such silent fallbacks.  
   **Fix**: Log warnings (or raise in debug) on fallback. Make simplify failure at least log the exception. Propagate more specific errors instead of broad `except`.

5. **P1: `_grid_for_lowering` multiplies grid * threadgroup incorrectly for Metal dispatch**  
   **Location**: `_grid_for_lowering` (lines ~510-520) and usage in `_fp8_vecmat_kernel_for` + dispatch (lines ~550+).  
   ```python
   return (
       max(1, lowering.grid[0] * lowering.threadgroup[0]),
       ...
   )
   ```
   **Issue**: In MLX `mx.fast.metal_kernel` and standard Metal, you typically pass `grid` as total threads (or threadgroups-per-grid) and `threadgroup` as threads-per-threadgroup. Multiplying them here computes *total threads* but may overcount if `lowering.grid` already represents groups. TileLang's Metal lowering likely emits grid in threadgroups (common in TVM/TileLang). This can launch too many / too few threadgroups, causing incomplete coverage (missed outputs) or over-launch (waste/crashes on bounds).  
   **Evidence from Metal docs**: Grid is often computed with ceildiv for coverage; thread_position_in_grid assumes proper sizing.  
   **Fix**: Clarify what `lowering.grid` represents (threadgroups or total threads). Use `grid = lowering.grid` directly if it's already total, or compute ceildiv properly matching `T.ceildiv(_FP8_VM_N, _FP8_VM_NP)`. Test with `N` not multiple of `outputs_per_block*reduce_threads`.

### P2 Issues (Edge Cases, Regressions, Maintainability)

- **P2: `C` shape in dot4 PrimFunc vs. dispatch input_map fallback** (lines ~560-580): The `if input_names == [...] and output_shape in ((n,), (1,n))` branch uses direct `kernel(...)`, else falls back to `_msl_transform.dispatch` with `input_map` that reshapes A to `(1,K)`. Inconsistent paths can cause shape mismatches for A/B when dot4 path is active.  
  **Fix**: Unify to always use the same dispatch path or normalize all inputs to match the PrimFunc signature.

- **P2: Hard-coded globals update in `make_fp8_vecmat_reduce_kernel`** (lines ~130-140: `g.update(...)` with `_FP8_VM_*`). This mutates module globals for each call (even cached). Race-prone in multi-threaded or concurrent kernel builds. Also, `T = cast(Any, T)` after import.  
  **Fix**: Pass params explicitly into nested function or use a closure/dataclass instead of globals.

- **P2: `vectorized_loads=True` path is incomplete / probe-only** (lines ~150-170): Uses per-thread `A_local/B_local` + scalar casts, no packed dot. But default is False, and fast path assumes dot4. If ever enabled, the reduction uses `tvm_thread_allreduce` which may not match `simd_sum` in Path B.  
  **Fix**: Either remove or fully align with canonical body.

- **P2: No handling for `scale_w_per_row=False` edge in some paths** (globals `_FP8_VM_SW`, used in indexing). If `scale_w` is scalar but code assumes array, or vice versa. `_resolve_vecmat_scale` is decent but called after shape checks.  
  **Edge**: N=1, K=4 (minimal dot4), scalar scales.

- **P2: `fp8_vecmat_path_c_status` and `can_run_metal` — silent None return on failure** (lines ~540+). Caller must check for None, but no doc on when/why it falls back. Swallows TileLang import or lowering errors upstream.  
  **Fix**: Return a richer status or raise with details.

- **Minor: Off-by-one risk in unroll/ceildiv** (dot4 loop: `T.unroll(0, T.ceildiv(_FP8_VM_K_WORDS, _FP8_VM_RT), ...)` and `i = ko * _FP8_VM_RT + kr`). If `K_WORDS % RT !=0`, bounds `if i < _FP8_VM_K_WORDS` is present but depends on TileLang codegen proving it. Matches Path B intent but verify generated MSL has no extra iterations.

- **Performance note (not correctness)**: The dot4 path aims for LUT + packed loads + `simd_sum`, matching Path B. But if lowering doesn't emit `__tvm_fp8_e4m3fn_lut` + reinterpret_cast<uint> (check via `fp8_vecmat_msl_blockers`), it falls back to slower scalar decodes — this is already diagnosed, but the blockers dict is post-lowering only. Consider forcing intrinsic registration earlier.

**Overall Recommendation**: The dot4 packed macro path introduces the most fragility (shape/indexing divergence from Path B). Prioritize making all PrimFunc branches produce *identical* MSL structure to the `canonical_vecmat_runtime_body` (flat C, packed uint loads, simd_sum). Add unit tests that compare Path C output bitwise to Path B for small N/K (including N=1, K=4, K=128, various scale_w shapes). Run with `CPPMEGA_FP8_VECMAT_PATH_C_DEBUG=1` to surface issues.

These are the high-severity items from a correctness-focused pass. No obvious races in the hot path (Metal dispatch is single-stream), but the caching + globals + silent excepts warrant cleanup. Let me know if you want deeper dive on specific functions or generated MSL diffs.