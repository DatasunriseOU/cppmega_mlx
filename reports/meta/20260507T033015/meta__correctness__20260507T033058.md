---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 888d368715dee062c804309c99cd15d4ee55468f
timestamp: 2026-05-07T03:30:58.801562+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/test_tilelang_topk.py']
---
Here’s the Wave-4 correctness review for tip `888d3687` + `2bef81d3`. I scanned the 9-file bundle for the 2 new commits only.

### 1. `cppmega_mlx/nn/_tilelang/_msl_transform.py`

**1.1 Race condition on `_failed_attempts` counter**  
`_msl_transform.py:1288`  
```python
failed = getattr(_preload_libz3_for_dev_tilelang, "_failed_attempts", 0)
if failed >= _MAX_FAILED_ATTEMPTS:
    return
...
_preload_libz3_for_dev_tilelang._failed_attempts = failed + 1  # type: ignore[attr-defined]
```
**Bug**: `failed` is read locally, then written back non-atomically. If multiple threads/importers call `_preload_libz3_for_dev_tilelang` concurrently, two threads can both read `failed=2`, both increment, and both write `3`. You’ll exceed `cap=3` and keep retrying. Python attribute access isn���t thread-safe.  

**Fix**: Use a lock, or make the counter process-global with `itertools.count` guarded by `threading.Lock`. Since this runs at import time on Darwin, the simplest fix is a module-level `threading.Lock()` around the whole function body.

**1.2 Swallowed `OSError` hides real dlopen failures**  
`_msl_transform.py:1314`  
```python
except OSError:
    continue
```
**Bug**: If `ctypes.CDLL` fails due to `mach-o file, but is an incompatible architecture` or `Symbol not found`, you silently try the next candidate and then bump `_failed_attempts`. The caller gets no signal that libz3 was found but unusable. This regresses debuggability: CRITICALs you wanted to clear will reappear as generic “Path C did not dispatch” with no stderr.  

**Fix**: Log the `OSError` at `warning` level before `continue`. Only swallow `FileNotFoundError`. Distinguish “not found” vs “found but broken”.

**1.3 `_done` flag not set on early return from cap**  
`_msl_transform.py:1289`  
```python
if failed >= _MAX_FAILED_ATTEMPTS:
    return
```
**Bug**: After 3 failures, every subsequent call returns immediately but `_done` is never set. If another code path checks `_done` to decide whether to log “libz3 unavailable”, it will see `False` forever and keep spamming checks. `_done` should mean “we’re finished trying”, not “we succeeded”.  

**Fix**: Set `_done = True` in the cap branch, or rename flag to `_attempt_complete` and set it regardless of success.

**1.4 `cap=3` can be exhausted by transient FS errors**  
`_msl_transform.py:1287`  
**Edge case**: If `/Volumes/external/...` is a network mount that returns `EIO` once, you increment `failed`. After 3 transient errors, you permanently give up for the process lifetime even if libz3 becomes available later.  

**Impact**: Violates “do not regress existing behavior”. Before fix-round-4, every dispatch retried. Now you may permanently disable Path C after 3 blips.  

**Fix**: Don’t count `OSError` where `e.errno in (EAGAIN, EINTR, EIO)`. Only increment on `FileNotFoundError` or successful `stat` + failed `dlopen`.

### 2. `cppmega_mlx/nn/_tilelang/topk_selector.py`

**2.1 Hard assert will crash production if TileLang IR changes**  
`topk_selector.py:8550`  
```python
assert body != lowering.body, (
    "_path_c_rewrite_merge_round: MSL pattern not found — TileLang "
    "merge-round emission shape has changed..."
)
```
**Bug**: This is a `hard assert`, not a graceful fallback. If TileLang 0.9.2 ships a whitespace change, every import of `topk_selector` will raise `AssertionError` and kill the process. The comment says “fail loud so a future TileLang version... forces a deliberate regex update”, but that breaks semver: users can’t `pip install -U tilelang` without your code breaking.  

**Regression**: Prior behavior was silent degradation. Now you introduced a crash on valid upgrade paths.  

**Fix**: Replace `assert` with `if body == lowering.body: raise RuntimeError(...)`. Then catch it at the callsite `_path_c_kernel_for` and fall back to unoptimized lowering with a `warnings.warn`. Failing import is worse than slower kernel.

**2.2 Assert compares strings but `replace` may return same object**  
`topk_selector.py:8550`  
**Edge case**: `dataclasses.replace` returns a new object only if fields change. If `body` is unchanged, `replace(lowering, body=body)` returns a new `TileLangMSLLowering` instance, so `body != lowering.body` is always `True` even when no rewrite happened. The assert will never fire, defeating the guard.  

**Proof**: `dataclasses.replace(x, field=x.field)` => `id(new) != id(x)`. Your assert checks identity of `body` strings, but you assigned `body = lowering.body` at line 850, then maybe mutated it. If no mutation, you still call `replace` and the assert passes incorrectly.  

**Fix**: Track `did_rewrite: bool = False` and set it when you do a replacement. Assert `did_rewrite, "pattern not found"`.

### 3. `sparse_mla.json` - Verification of CRITICALs

**3.1 `strict.enabled=false` but failures logged**  
`sparse_mla.json:16`  
`strict.enabled` is `false`, yet `strict.failures` lists 2 entries at `:22` and `:23`.  

**Bug**: If CRITICALs were supposed to be cleared by the preload cap, the harness is still collecting strict failures. That means either:  
1. Path C still returned `None` and the bench fell back, or  
2. The ratio check ran despite `strict.enabled=false`.  

Either way, the commit message “Verify CRITICALs cleared” is not satisfied. The failures show `path_c_over_path_b_max_ratio=1.0018` and `1.0814`, both >1.0, so Path C is slower. The preload fix doesn’t affect kernel speed.  

**Action**: The libz3 preload only fixes “did not dispatch”. It won’t clear perf CRITICALs. You still need to address Path C speed vs Path B.

### 4. Missing bugs: no off-by-one or None handling issues found

I checked all 9 files for `.shape[-1] - 1`, `range(len(...))`, `arr[:-1]`, and `None` returns. No new off-by-ones introduced by these 2 commits. Existing code already guards `None` from `_path_c_kernel_for` at `topk_selector.py:8704`.

### Summary of actionable findings

| File:Line | Severity | Issue | Fix |
| --- | --- | --- | --- |
| `_msl_transform.py:1288` | **Critical** | Race on `_failed_attempts` lets counter exceed cap | Add `threading.Lock()` around read-modify-write |
| `_msl_transform.py:1314` | **High** | `except OSError` swallows broken-dylib errors, loses diagnostics | Log exception, only swallow `FileNotFoundError` |
| `_msl_transform.py:1289` | **Medium** | `_done` never set when cap hit, causes repeated checks | Set `_done=True` on cap return |
| `_msl_transform.py:1287` | **Medium** | Transient FS errors permanently disable Path C | Don’t increment on `EIO/EAGAIN` |
| `topk_selector.py:8550` | **Critical** | `assert` crashes import on TileLang upgrade; regression | Raise + catch, fallback to unoptimized kernel with warning |
| `topk_selector.py:8550` | **High** | `dataclasses.replace` makes assert vacuous | Track `did_rewrite` boolean instead |
| `sparse_mla.json:22` | **Info** | CRITICALs not cleared; perf ratio >1.0 remains | Preload fix unrelated to speed; separate issue |

The libz3 preload with `cap=3` is correct in spirit but needs thread safety and better error triage. The hard assert in the merge-round rewrite is the biggest correctness regression: it converts a perf fallback into a hard crash. Change it to a recoverable exception before shipping.