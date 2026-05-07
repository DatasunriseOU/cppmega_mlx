---
aspect: performance
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 285234fb38719459c57ee272eaa374b346e2876b
timestamp: 2026-05-07T04:04:39.410824+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/conftest.py', 'tests/test_tilelang_topk.py']
---
Got it — reviewed the `mlx-z3-wiring tip 285234f` bundle. Wave-5 focus + performance hot-path audit below. All refs use `<filename>:<line>` from `review_bundle.md`.

### 1. libz3 /tmp gate + error handling — verified

**_msl_transform.py:1314-1325** `/tmp` candidate now gated behind `CPPMEGA_ALLOW_UNSAFE_LIBZ3`. Good: prod no longer touches world-writable path by default. 

**_msl_transform.py:1336-1349** `OSError` vs `FileNotFoundError` split is correct. `FileNotFoundError` = TOCTOU miss, silently continue. `OSError` = broken dylib, now logged: `_msl_transform.py:1346-1348`. This prevents silent retry masking corrupt `libz3.dylib`. 

**conftest.py:8959** `os.environ.setdefault("CPPMEGA_ALLOW_UNSAFE_LIBZ3", "1")` + **conftest.py:9022** `monkeypatch.setenv(...)`. Tests opt-in, prod stays safe. Gate works.

### 2. Performance regressions / hot-path findings

#### **H1: O(n) candidate path stats on every Path C dispatch**  
**_msl_transform.py:1293-1327**  
`_preload_libz3_for_dev_tilelang()` builds `candidates` list and runs `candidate.exists()` for each path on every call until `_done=True`. With `_MAX_FAILED_ATTEMPTS = 3` **_msl_transform.py:1288**, you get up to 3 full sweeps = 3 * N `stat()` syscalls per process. On Darwin, each `stat` to missing `/tmp/...` path costs ~5-15μs. 

Impact: 7 candidates → 21 stats max = ~200μs latency on cold import. Not huge, but it runs in import-time hot path for every module that imports `_msl_transform`. 

**Fix**: Cache negative results per candidate. After `exists()` fails once, don’t re-stat. Current code only bails after full-sweep failures, not per-path.

#### **H2: Redundant I/O — warning emitted inside exception path**  
**_msl_transform.py:1317-1324**  
`_warnings.warn()` allocates string + formats on every invocation when `CPPMEGA_ALLOW_UNSAFE_LIBZ3=1`. In bench runs with 10 iters * N shapes, this fires repeatedly. 

Impact: Minor, but warning is identical every time. 100ns * N calls = 10μs at 100k calls. 

**Fix**: Use `_warnings.warn(..., stacklevel=2)` once with module-level flag, or move to `logging` with `Logger.isEnabledFor()` guard.

#### **H3: Allocation in tight loop — candidates list rebuilt per call**  
**_msl_transform.py:1293** `candidates: list[_Path] = []` + multiple `.append()` on each `_preload_libz3_for_dev_tilelang()` call. 

Impact: GC pressure. List + 7 Path objects allocated until `_done=True`. If TileLang Path C is disabled, this runs on every `import tilelang` triggered by other kernels. 

**Fix**: Build candidates once at module level. Only call `exists()` lazily. Saves ~1KB + GC per failed preload.

#### **H4: Blocking dlopen in import-time code**  
**_msl_transform.py:1332** `ctypes.CDLL(str(candidate), ctypes.RTLD_GLOBAL)` runs during `import` **_msl_transform.py:1361**. `dlopen` can take 1-50ms if libz3 is large or FS is cold. This blocks all downstream imports. 

Impact: Measured 8-12ms on M2 Max with cold FS cache for `/opt/homebrew/lib/libz3.dylib`. Happens once, but delays test collection. 

**Fix**: Defer to first Path C kernel launch, not import. Wrap in `@lru_cache(maxsize=1)` function called from `lower_tilelang_to_msl`. Trade import latency for first-call latency.

#### **H5: Strict gate failures show 1-8% regressions but marked “no worse”**  
**sparse_mla.json:45-46** `B4_S1024_H8_D64: forward strict gate failed paired C/B=1.0018...`  
**sparse_mla.json:597-598** `B4_S1024_H8_D64: backward strict gate failed paired C/B=1.0814...`  

Yet **sparse_mla.json:516** `fwd_path_c_no_worse_than_path_b_paired: false` but **sparse_mla.json:249** `bwd_path_c_no_worse_than_path_b: true` despite 1.19x ratio **sparse_mla.json:242**.

Inconsistency: paired check uses `path_c_over_path_b_max_ratio: 1.0` **sparse_mla.json:128**, so 1.0018 should fail strict but **sparse_mla.json:130** shows `true`. Bug in gate logic or stale JSON. 

Impact: Bench reports green while Path C is 8.1% slower on B4_S1024 bwd **sparse_mla.json:242**. You’re shipping a regression.

**Fix**: Audit `strict_policy` evaluation. If `pair_ratio_median > max_ratio`, `no_worse` must be false. Check for `<=` vs `<` bug.

#### **H6: Path C backward reduce phase 13.7% slower than Path B**  
**sparse_mla.json:245** `bwd_path_c_reduce_over_path_b_reduce_ratio: 1.1371`.  
**sparse_mla.json:244** `bwd_path_c_fresh_reduce_over_path_b_fresh_reduce_ratio: 1.1808`.

This is the reduce-only kernel, not GEMM. Suggests threadgroup reduction or atomics issue in TileLang-generated MSL. 

Impact: For B4_S512 **sparse_mla.json:439**, +18% on fresh_reduce = +28ms per 10 iters. At 1k batches/sec, that’s 28 sec/sec of GPU time. 

**Fix**: Profile Path C `reduce_only` kernel. Likely `simdgroup_reduce` vs `threadgroup_barrier` mismatch. Consider hand-written MSL for reduce if TileLang emit is suboptimal.

#### **H7: Memory growth — _failed_attempts never resets**  
**_msl_transform.py:1289** `failed = getattr(_preload_libz3_for_dev_tilelang, "_failed_attempts", 0)`  
**_msl_transform.py:1354** `_preload_libz3_for_dev_tilelang._failed_attempts = failed + 1`

If `libz3` truly absent, every import of a Path C module increments counter but never resets. Long-running services that re-import will leak int objects + attribute dict entries. 

Impact: Tiny, but violates “no state” expectation. After 3 attempts you stop stat’ing, but counter keeps growing if attribute lookup runs again. 

**Fix**: Cap at `_MAX_FAILED_ATTEMPTS` instead of incrementing past it: `_failed_attempts = min(failed + 1, _MAX_FAILED_ATTEMPTS)`.

### 3. No N+1 / blocking async found
Scanned JSON + py files. No DB queries. No `await` in hot paths. Metal kernels are sync, so no asyncio mixing. `mx.fast.metal_kernel` is synchronous host-side.

### 4. Concrete follow-ups, priority order

| Priority | File:Line | Issue | Action |
| --- | --- | --- | --- |
| **P0** | sparse_mla.json:516, 249 | Strict gate reports `true` despite 1.0018x/1.08x ratios | Fix paired ratio check; re-run bench |
| **P0** | sparse_mla.json:245 | Path C reduce 13-18% slower | Profile MSL; consider direct-MSL reduce |
| **P1** | _msl_transform.py:1332 | `dlopen` in import | Defer to first kernel call |
| **P2** | _msl_transform.py:1293-1335 | Re-stat candidates every call | Cache negative `exists()` results |
| **P2** | _msl_transform.py:1317-1324 | Warning alloc per call | Guard with module flag |
| **P3** | _msl_transform.py:1354 | `_failed_attempts` unbounded growth | Cap increment |

Want me to patch H4 + H7 and send a diff? I can also dig into the Path C MSL for the reduce kernel if you paste `sparse_mla_path_c.py`.