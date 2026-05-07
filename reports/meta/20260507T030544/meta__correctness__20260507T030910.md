---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 5b83522a14065d99a23788c3ac47f4e8daa70be6
timestamp: 2026-05-07T03:09:10.070453+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/test_tilelang_topk.py']
---
## Finding 1
- **Severity**: critical
- **File**: `cppmega_mlx/nn/_tilelang/_msl_transform.py:67`
- **Issue**: `_preload_libz3_for_dev_tilelang` sets `_done` only on successful `CDLL` load, but never sets it on failure. If none of the `candidates` exist or all `CDLL` calls raise `OSError`, the function returns with `_done` unset. The module-level call at line 98-99 runs on import, so every subsequent import of `_msl_transform` will re-scan the filesystem and attempt to `CDLL` every candidate again. This introduces unbounded import-time overhead and risks re-raising `OSError` logs on every dependent module import, especially under multi-process MLX worker pools.
- **Fix**: Set `_done = True` unconditionally after the first attempt, regardless of success, to make it idempotent. Also log a single warning on failure instead of silent retry:
```python
for candidate in candidates:
    try:
        if candidate.exists():
            ctypes.CDLL(str(candidate), ctypes.RTLD_GLOBAL)
            _preload_libz3_for_dev_tilelang._done = True  # type: ignore[attr-defined]
            return
    except OSError:
        continue
# Mark done even on failure to avoid re-scanning
_preload_libz3_for_dev_tilelang._done = True  # type: ignore[attr-defined]
```

## Finding 2
- **Severity**: high  
- **File**: `cppmega_mlx/nn/_tilelang/_msl_transform.py:73`
- **Issue**: Hardcoded fallback path `"/tmp/tl_apache_tvm_swap/build/lib/libz3.dylib"` is checked in production code. This path is specific to one developer bench environment and will not exist on CI or user machines. The `candidate.exists()` check means it silently does nothing elsewhere, but the presence of environment-specific paths in code is a correctness hazard: if this path existed with a stale/incorrect `libz3.dylib`, it would be loaded preferentially over brew/system versions and could cause ABI mismatch crashes in `libtilelang.dylib`.
- **Fix**: Remove the `/tmp/...` hardcoded path. Rely only on `TILELANG_DEV_BUILD_ROOT`, `TILELANG_ROOT`, and standard system locations. If a dev-only override is needed, gate it behind `if os.environ.get("CPPMEGA_TILELANG_ALLOW_DEV_PATHS"):`.

## Finding 3
- **Severity**: medium
- **File**: `bench/tilelang_ports/sparse_mla.json:54`
- **Issue**: Benchmark harness changed `warmup` from 5 to 3 and `iters` from 20 to 10 for `sparse_mla` forward measurements, while topk kept `warmup: 3, iters: 10`. The diff comment says "correct version is now 1.28x slower for K=32", but the benchmark now uses fewer iterations and warmup steps than wave-2. Reduced samples increase variance and risk false positives on the strict gate `path_c_over_path_b_max_ratio: 1.0`. For B4_S1024_H8_D64, `fwd_path_c_over_path_b_ratio: 1.0351` and `bwd: 1.0115` already exceed the strict ratio, and with only 10 iters the median is unstable. This can cause nondeterministic CI failures or mask regressions.
- **Fix**: Restore `warmup >= 5` and `iters >= 20` for strict-gated kernels, or add `stddev` to JSON and fail CI if coefficient of variation > 5%. Document the statistical significance requirement for the 1.0 ratio gate.

## Finding 4
- **Severity**: medium
- **File**: `cppmega_mlx/nn/_tilelang/_msl_transform.py:95-99`
- **Issue**: Eager `dlopen` of `libz3.dylib` runs at module import time via `if sys.platform == "darwin": _preload_libz3_for_dev_tilelang()`. If `import tilelang` later fails for an unrelated reason, the process will already have `libz3` loaded with `RTLD_GLOBAL`. This can poison symbol resolution for other libraries that bundle a different Z3 version. `RTLD_GLOBAL` makes Z3 symbols visible to all subsequent `dlopen` calls, creating potential ABI conflicts. The comment says "silent on success" but does not handle load errors, so a bad `libz3.dylib` will crash import before user code can catch it.
- **Fix**: Defer preload to first Path C dispatch, not import. Wrap in try/except and use `RTLD_LOCAL` unless global is strictly required. Emit a single `warnings.warn` if preload fails instead of crashing:
```python
def ensure_libz3():
    if getattr(ensure_libz3, "_done", False): return
    try:
        # ... load logic with RTLD_LOCAL
    except OSError as e:
        warnings.warn(f"libz3 preload failed: {e}")
    ensure_libz3._done = True
```

## Finding 5
- **Severity**: low
- **File**: `bench/tilelang_ports/sparse_mla.json:21-22`
- **Issue**: Strict gate failures changed from `paired C/B=inf path_b_ok=False path_c_ok=False` to concrete ratios `C/B=1.0018` and `1.0814` with `path_b_ok=True path_c_ok=True`. This confirms Path C now dispatches, which matches the claimed topk fix. However, `bwd_path_c_over_path_b_ratio: 1.1959` for B2_S128_H8_D64 exceeds the `path_c_over_path_b_max_ratio: 1.0` yet `bwd_path_c_no_worse_than_path_b: true`. This is a logical inconsistency: the boolean should be `false` when ratio > 1.0. Indicates a bug in strict-gate evaluation logic not shown in this chunk, which could allow perf regressions to pass CI.
- **Fix**: Verify in the harness that `no_worse_than_path_b` is computed as `ratio <= max_ratio`. Add unit test for boundary case `ratio == 1.0000001`.

## Finding 6
- **Severity**: info
- **File**: `bench/tilelang_ports/sparse_mla.json:5`
- **Issue**: `mlx_version` bumped from `0.31.1` to `0.31.2`. The diff does not show whether MSL kernel APIs changed between these versions. The prompt mentions "sparse_mla canonicalize hoist-aware fix didn't introduce new MSL compile issues", but this chunk has no code for `sparse_mla_path_c.py`. Without seeing that file, we cannot verify the hoist fix. The version bump alone is not a bug, but it invalidates comparisons against wave-2 numbers unless MLX 0.31.2 is confirmed perf-neutral.
- **Fix**: Include `sparse_mla_path_c.py` diff in review. Pin `mlx_version` in CI and re-run wave-2 baselines on 0.31.2 to isolate topk fix impact.

No findings related to topk insertion-sort correctness or sparse_mla canonicalize hoist in this chunk. Those files are not present in chunk 1 of 3.

## Finding 1
- **Severity**: critical
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:375`
- **Issue**: `T.reduce_max` on `idx_scores_f` uses a register-allocated fragment that was only primed with `-inf` inside `if h == 0`. For heads `h != 0`, `idx_scores_f` is never written before the reduce, so it contains uninitialized/undefined values. On Metal SIMDgroup this can yield NaN or garbage and corrupt `softmax_idx`, then `kl_term`. The priming loop at 368-370 is inside the `if h == 0` block.
- **Fix**: Move the `-inf` initialization of `idx_scores_f` outside the `if h == 0` guard, or hoist it to the top of the `sk_tile` loop before any per-head work:
```diff
-                    if h == 0:
-                        for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
-                            idx_scores_f[i, j] = T.cast(-3.4028234663852886e38, "float32")
+                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
+                        idx_scores_f[i, j] = T.cast(-3.4028234663852886e38, "float32")
+                    if h == 0:
                         for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
```

## Finding 2
- **Severity**: high
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:310-312`
- **Issue**: `_active_sk_tiles` clamps to `min(SK_TILES, ...)` but the subsequent `for sk_tile in T.Pipelined(_active_sk_tiles, ...)` still indexes `sk_idx = sk_tile * BLOCK_SK + j` and accesses `K[sk_idx, ...]` at line 328-329 without an `sk_idx < Sk` guard inside the `K_s` load. For `Sk` not divisible by `BLOCK_SK`, the last tile will read past `Sk` bounds. The `if (sk_idx < Sk)` guard exists at 328 but only gates the assignment, not the earlier `K[sk_idx, b, h, d_idx]` load in the same expression, so on CUDA this is an OOB load before the predicate. TileLang may not mask it.
- **Fix**: Guard the `K` load itself, not just the assignment:
```diff
-                            K_s[dd, j] = K[sk_idx, b, h, d_idx]
+                            if (sk_idx < Sk) and (d_idx < AD):
+                                K_s[dd, j] = K[sk_idx, b, h, d_idx]
+                            else:
+                                K_s[dd, j] = T.cast(0, in_dtype)
```

## Finding 3
- **Severity**: high
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:420-423`
- **Issue**: `M_pre` and `D_pre` are allocated as `(AH, BLOCK_SQ)` fragments unconditionally at 420-421, even when `USE_MD_PRE` is `False`. For `AH=128, BLOCK_SQ=128` this is 128KB of registers per thread, exceeding CUDA per-block register limits and causing spills or compile failure. The `else` at 422 only allocates `(1,1)` but the `if USE_MD_PRE` block still instantiates the large fragments because TileLang elaborates both branches at compile time.
- **Fix**: Use `T.alloc_buffer` with `scope="local"` gated by `T.If` or make the large allocation conditional via Python, not a runtime `if`. Since `USE_MD_PRE` is a Python constant, wrap allocation in Python:
```diff
-            if USE_MD_PRE:
-                M_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
-                D_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32")
-            else:
-                M_pre = T.alloc_fragment((1, 1), "float32")
-                D_pre = T.alloc_fragment((1, 1), "float32")
+            M_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32") if USE_MD_PRE else None
+            D_pre = T.alloc_fragment((AH, BLOCK_SQ), "float32") if USE_MD_PRE else None
```
And guard all uses with `if USE_MD_PRE:` at Python level, not runtime T.If.

## Finding 4
- **Severity**: high  
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:351-353`
- **Issue**: `IndexMask[b, sq_idx, sk_idx]` is read at line 351 inside `if SPARSE and in_bounds`, but `IndexMask` is `torch.empty` for `sparse_loss=False` per line 857-858. On CUDA, reading uninitialized memory is UB and may return NaNs that propagate into `s` and then `exp(s - m_i)`. The guard `in_bounds` prevents OOB but not uninitialized access when `SPARSE=False`.
- **Fix**: Only add `IndexMask` when `SPARSE` and the buffer is initialized. Either fill with zeros when not sparse, or split the kernel:
```diff
-        index_mask = torch.empty((AB, ASq, Sk), dtype=torch.float32, device=query.device)
+        index_mask = torch.zeros((AB, ASq, Sk), dtype=torch.float32, device=query.device)
```
Zero is safe because `SPARSE=False` branch never uses it, and zero add is a no-op if the constexpr path isn't eliminated.

## Finding 5
- **Severity**: medium
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:513-514`
- **Issue**: `_active_sk_tiles = T.max(T.min(SK_TILES, _max_useful_sk // BLOCK_SK + 1), 1)` can underflow when `_max_useful_sk` is negative. `_max_useful_sk = T.min(_max_sq_in_block, ASq - 1)` where `_max_sq_in_block = sq_block_id * BLOCK_SQ + (BLOCK_SQ - 1)`. If `ASq=0`, then `_max_useful_sk=-1`, `// BLOCK_SK + 1 = 0`, `min(SK_TILES, 0)=0`, `max(0,1)=1`. Loop runs once with all `sq_idx >= ASq`, writing to `M[b,h,sq_idx]` OOB at line 395-396 because the `if sq_idx < ASq` guard prevents writes but the earlier `m_i[i]` is never initialized for those lanes. Violates precondition `ASq <= 0` but function only checks `ASq <= 0` at line 224 and raises. However `ASq=0` would already error. Edge case: `ASq=1, BLOCK_SQ=32, sq_block_id=0` gives `_max_useful_sk=0`, `_active_sk_tiles=1`, correct. But `ASq=1, sq_block_id=1` gives `_max_sq_in_block=63`, `_max_useful_sk=0`, still 1 tile, and `sq_idx` loop at 395 skips due to guard. Not a bug, but the clamp is redundant and masks logic errors.
- **Fix**: Assert `ASq > 0` already exists at 224. Remove `T.max(..., 1)` and let `0` tiles be handled, or add explicit `if ASq == 0: return` before kernel launch. Current code is defensive but obscures intent.

## Finding 6
- **Severity**: medium
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:605-606`
- **Issue**: `d_h[i] = D[b, h, sq_idx]` is loaded per `(sk_tile, h)` iteration when `USE_MD_PRE=False`. For `AH=128, SK_TILES=16`, this is 2048 HBM loads of the same `D` values. With `USE_MD_PRE` disabled due to `_MD_PRE_BUDGET_BYTES`, this regresses perf vs Triton which loads once per `(b,h,sq_block)`. The `M_pre/D_pre` budget check at 413-415 is too conservative for CUDA: 128*128*8=128KB fits in registers for 1 warp, but the kernel uses 256 threads = 8 warps, so 16KB per warp, which is borderline. Forcing `USE_MD_PRE=True` on CUDA would fix it.
- **Fix**: Make `USE_MD_PRE` depend on target: `USE_MD_PRE = _MD_PRE_BYTES <= _MD_PRE_BUDGET_BYTES or not _is_metal(target)`. Or increase budget for CUDA to 128KB.

## Finding 7
- **Severity**: low
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:823-827`
- **Issue**: `topk_indices.to(dtype=torch.int64, copy=False)` with `copy=False` means if input is already `int64`, `topk_idx64` aliases it. Later `topk_idx64.max()` and `min()` at 831-832 will sync if `topk_indices` is on a different stream, but no explicit sync. If caller used a non-default stream, this can read uninitialized values. PyTorch ops usually insert syncs, but explicit `.contiguous()` at 829 forces a copy and sync. Only risk if input is `int64` and non-contiguous, but `is_contiguous()` check handles it. Still, document that inputs must be on same stream.
- **Fix**: Add comment or force `torch.cuda.current_stream().synchronize()` before `max/min` when not contiguous.

## Finding 8
- **Severity**: low
- **File**: `cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py:847-850`
- **Issue**: `_patch = torch.where(_row_has_valid, index_mask[..., 0], zeros)` reads `index_mask[..., 0]` which is uninitialized `-inf` from `torch.full` at 824 for invalid rows. `where` may evaluate both branches on CUDA, causing NaNs if `-inf` propagates. The intent is to keep valid rows unchanged, but `where` is not lazy. If any element of `index_mask[...,0]` is NaN, `_patch` for valid rows becomes NaN.
- **Fix**: Use `index_mask[..., 0] = torch.where(_row_has_valid, index_mask[..., 0], 0.0)` directly, or mask the read: `torch.where(_row_has_valid, index_mask[...,0].clone(), 0.0)`.

## Finding 9
- **Severity**: info
- **File**: `cppmega_mlx/nn/_tilelang/fp8_amax.py:212`
- **Issue**: `_BLOCK_SIZE_TABLE` lists `metal: (256, 64)` but Metal warp is 32, so 64 threads = 2 simdgroups. Comment says "2 simdgroups" which is correct. No bug, but inconsistent with `dsa_splitk_indexer_loss.py` which uses 128 threads for Metal stage1 at line 138. If TileLang maps 128 threads to 4 simdgroups, register pressure doubles vs amax kernel. Not a correctness issue, just inconsistency.
- **Fix**: Align Metal thread counts across kernels or document rationale.

No other correctness bugs visible in chunk 2.

## Finding 1
- **Severity**: critical
- **File**: `cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py` near `_strip_z3_hoisted_address_decls`
- **Issue**: The regex deletes any line matching `int h = 0;`, `int b = 0;`, `int g = 0;`, etc, regardless of whether it was actually hoisted by z3-final. Pattern `r"(?m)^[ \t]*int (?:h|b|g|q_row_base|kv_b_base|idx_base|out_row|d_out_row|dkv_partial_base) = (?:0|\([^\n]*(?:threadgroup_position_in_grid\.x|gid)[^\n]*\));\n"` includes `0` as an alternative RHS. If the generated MSL legitimately declares `int h = 0;` for its own logic, this pass will silently remove it, breaking later uses of `h`. This corrupts kernels that happen to use those common single-letter names.
- **Fix**: Restrict the deletion to only lines whose RHS references `threadgroup_position_in_grid` or the already-injected `gid`. Remove the `0` alternative, or require the RHS to contain `gid`/`threadgroup_position_in_grid.x`. Example:
```python
hoisted_pattern = re.compile(
    r"(?m)^[ \t]*int (?:" + "|".join(hoisted_names) + r") = "
    r"\([^\n]*(?:threadgroup_position_in_grid\.x|gid)[^\n]*\);\n"
)
```

## Finding 2
- **Severity**: high
- **File**: `cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py` in `_canonicalize_fwd_base_indexing` and `_canonicalize_bwd_base_indexing`
- **Issue**: Promoting `gather_idx = indices[...];` to `int gather_idx = indices[...];` changes the statement from an assignment to a declaration. If `gather_idx` was declared earlier in the function scope, this creates a new inner-scope variable that shadows the outer one. Any later uses of `gather_idx` outside the loop will read the unmodified outer variable, breaking the gather index data flow. In the worst case, if there was no prior declaration, but the assignment occurs in a loop, each iteration now declares a fresh variable, so code after the loop referencing `gather_idx` will see an undeclared identifier.
- **Fix**: Only inject the type if you can prove there is no existing declaration in scope. Safer: do not add `int`; instead ensure the declaration exists once at function top. If you must rewrite, track whether `gather_idx` was already declared and skip the `int` prefix when it was:
```python
if "int gather_idx" not in msl and "thread int gather_idx" not in msl:
    msl = re.sub(r"(gather_idx = indices\[[^;]+;)", r"int \1", msl, count=1)  # at first use
else:
    # leave as assignment
    pass
```

## Finding 3
- **Severity**: high  
- **File**: `cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py` in `_canonicalize_fwd_base_indexing`
- **Issue**: `msl = re.sub(r"\buint kv_row_base_1 = ", "uint kv_row_base = ", msl)` unconditionally renames the declaration. If `uint kv_row_base` already exists earlier in the function, this creates a duplicate definition and the Metal compiler will reject the kernel. If it does not exist, the rename is fine, but the subsequent `msl = re.sub(r"\bkv\[kv_row_base_1 \+ d\]", "kv[kv_row_base + d]", msl)` assumes all uses were renamed, yet there may be other uses like `kv_row_base_1` in address math that are missed, leaving undefined identifiers.
- **Fix**: Guard the rename. Only perform it when `kv_row_base` is not already present, and replace *all* occurrences atomically:
```python
if "uint kv_row_base " not in msl and "uint kv_row_base_1 " in msl:
    msl = msl.replace("uint kv_row_base_1", "uint kv_row_base")
    msl = msl.replace("kv_row_base_1", "kv_row_base")
```

## Finding 4
- **Severity**: medium
- **File**: `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py` in `_uses_fp8_dot4_packed_macro`
- **Issue**: New guard `if K <= 0: return False` prevents `% 4` on non-positive `K`, but the function is still called with `K` coming from user shapes. If `K == 0`, the caller `_fp8_vecmat_kernel_for` will still build a kernel and later the Metal grid may be `(N, 0, 1)`, leading to a zero-sized launch or divide-by-zero in `K // 4` used by `_FP8_VM_K_WORDS`. The kernel should fail fast instead of generating degenerate code.
- **Fix**: Validate `K > 0` and `K % 4 == 0` in `_fp8_vecmat_kernel_for` before building the PrimFunc, and raise `ValueError` with a clear message. Remove the silent `return False` path for `K <= 0` since that shape is unsupported by the packed macro anyway.

## Finding 5
- **Severity**: low
- **File**: `cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py` in `_canonicalize_fwd_lane_indexing`
- **Issue**: `msl = re.sub(r"(?m)^[ \t]*int k = k;\n", "", msl)` and similar deletions of `int d_N = d;` remove tautological assignments. If TileLang ever emits `int k = k;` as a way to inhibit a compiler warning about unused `k`, deleting it reintroduces the warning. More importantly, the regex is line-anchored and will miss `int k=k;` without spaces, leaving some tautologies in place and making the canonicalization non-deterministic.
- **Fix**: Make the pattern whitespace-tolerant or skip the removal entirely. If the goal is just to silence diffs, keep the tautologies; they are harmless to the compiler. If removal is required, use `r"(?m)^[ \t]*int\s+k\s*=\s*k\s*;\s*\n"`.

No other correctness bugs found in this chunk. The topk insertion-sort fix now correctly uses `else: break` for all `K`, and the cache key now includes `vectorized_loads`, resolving the prior stale-kernel issue.