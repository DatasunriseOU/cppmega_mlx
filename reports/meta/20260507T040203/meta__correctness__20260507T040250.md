---
aspect: correctness
provider: meta
model: gpt-5-5-pro
range: main..mlx-z3-wiring
base_ref: aef115bb0fa97b68b541580d957ff6b043e76176
head_ref: 285234fb38719459c57ee272eaa374b346e2876b
timestamp: 2026-05-07T04:02:50.552956+00:00
files: ['bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/dsa_splitk_indexer_loss.py', 'cppmega_mlx/nn/_tilelang/fp8_amax.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'tests/conftest.py', 'tests/test_tilelang_topk.py']
---
Wave-5 correctness review for tip 285234f. Checked the attached bundle. Here are concrete correctness findings:

### 1. Libz3 /tmp gating + exception split - mostly correct, but one correctness risk

**cppmega_mlx/nn/_tilelang/_msl_transform.py:98-109**
The `/tmp` candidate is correctly gated behind `CPPMEGA_ALLOW_UNSAFE_LIBZ3=="1"` and emits a warning. Good.

**cppmega_mlx/nn/_tilelang/_msl_transform.py:120-133**
The `FileNotFoundError` vs `OSError` split is implemented:
- `FileNotFoundError`: silent continue, handles TOCTOU where file vanished between `exists()` and `dlopen`.
- `OSError`: logged warning, handles broken dylib/arch mismatch.

**Bug: Exception masking on non-Darwin**
**cppmega_mlx/nn/_tilelang/_msl_transform.py:144-145**
```python
if sys.platform == "darwin":
    _preload_libz3_for_dev_tilelang()
```
The preload only runs on Darwin. But the comment on line 59 says "every Path C kernel then silently returns `None`". On Linux, `import tilelang` can still fail with `OSError: libz3.so: cannot open`. Since the preload never runs, you never hit the attempt-cap logic at lines 74-75.

**Impact**: On Linux CI without libz3, every Path C dispatch will stat all candidates on every call instead of bailing after 3 attempts. Not a security bug, but a correctness regression vs Darwin where you added the bail-out.

**Fix**: Remove the Darwin guard, or make the attempt-cap check platform-agnostic. The `ctypes.CDLL` will raise `OSError` on Linux for missing `.so`, which you already catch.

### 2. Conftest env override breaks user opt-out

**tests/conftest.py:35**
```python
os.environ.setdefault("CPPMEGA_ALLOW_UNSAFE_LIBZ3", "1")
```
This runs at conftest import time, before any test imports `_msl_transform.py`.

**Bug: Incorrect default + swallowed user intent**
`setdefault` means: if the user/CI explicitly sets `CPPMEGA_ALLOW_UNSAFE_LIBZ3=0` in the environment, you overwrite it with `1`. The comment says "secure default (gated, off)" but the code forces it ON for all tests.

**Impact**: Tests cannot verify the safe behavior where `/tmp` is rejected. If production code accidentally depends on `/tmp`, tests won't catch it.

**Fix**: Remove the `setdefault`. Tests that need `/tmp` should `monkeypatch.setenv` explicitly. Or use `setdefault("CPPMEGA_ALLOW_UNSAFE_LIBZ3", "0")` to match the claimed "secure default".

### 3. Attempt counter never resets on success path

**cppmega_mlx/nn/_tilelang/_msl_transform.py:118**
```python
_preload_libz3_for_dev_tilelang._done = True
```
**cppmega_mlx/nn/_tilelang/_msl_transform.py:137-138**
```python
_preload_libz3_for_dev_tilelang._failed_attempts = failed + 1
```

**Bug: Stale failure count after transient errors**
If the first 3 calls fail due to transient NFS hiccup or race, `_failed_attempts` hits 3. Even if libz3 later becomes available, line 74-75 will early-return forever because `_done` never got set and `_failed_attempts` never resets.

**Impact**: Flaky CI where a one-time env glitch permanently disables Path C for the whole process.

**Fix**: Reset `_failed_attempts = 0` on successful dlopen before setting `_done = True`.

### 4. Swallowed exception in intrinsic registration

**cppmega_mlx/nn/_tilelang/_msl_transform.py:835-838**
```python
try:
    _register_path_c_metal_fp8_intrinsics()
except Exception:
    pass
```
**cppmega_mlx/nn/_tilelang/_msl_transform.py:727-731**
Inside `_register_path_c_metal_fp8_intrinsics`, individual op registration failures are also `except Exception: continue`.

**Bug: Silent failure to register required ops**
If `register_op` fails due to TVM API change or permission issue, you swallow it. Later, `_assert_path_c_metal_fp8_intrinsics_registered` at line 734 will raise, but only when a kernel actually calls it. Import-time registration can fail silently and you won't know until runtime.

**Impact**: Violates "fail-closed error paths" from the module docstring line 9-10. A kernel could import successfully but crash later in lowering with an opaque TVM error.

**Fix**: At minimum log at WARNING level in the except blocks. Better: collect errors and raise at end of `_register_path_c_metal_fp8_intrinsics` if any op failed.

### 5. Off-by-one in attempt cap check

**cppmega_mlx/nn/_tilelang/_msl_transform.py:74-75**
```python
if failed >= _MAX_FAILED_ATTEMPTS:
    return
```

**Correctness**: This is correct. With `_MAX_FAILED_ATTEMPTS = 3`, you allow attempts 0,1,2 and bail on 3. Matches the comment "bail after a small number of full-sweep failures".

No off-by-one here.

### 6. Race condition documented but unhandled - acceptable

**cppmega_mlx/nn/_tilelang/_msl_transform.py:91-97**
TOCTOU between `exists()` and `dlopen` is called out explicitly. You decided not to fix because "dlopen has no by-fd variant on macOS". For dev-only paths behind an opt-in flag, this is a reasonable correctness tradeoff. Documenting it prevents regressions where someone tries to "fix" it later.

### 7. Null/None handling in dispatch - correct

**cppmega_mlx/nn/_tilelang/_msl_transform.py:230-232**
```python
if any(isinstance(x, mx.array) and x.size == 0 for x in inputs):
    raise MSLDispatchUnsupported("empty tensors must use the pure MLX fallback")
```
Properly rejects empty inputs before launching metal kernel. Prevents Metal undefined behavior on zero-sized buffers.

**cppmega_mlx/nn/_tilelang/_msl_transform.py:201-202**
```python
if not can_run_metal():
    return None
```
Correct fail-closed: returns `None` instead of throwing, so callers can fall back. Matches docstring line 9.

### 8. Type mismatch risk in _as_metal_target

**cppmega_mlx/nn/_tilelang/_msl_transform.py:815-817**
```python
if value.lstrip("-").isdigit():
    config[key] = int(value)
```
**Bug: Mishandled floats**
If someone passes `target="metal -max_threads_per_threadgroup=1024.0"`, `isdigit()` returns False, so you keep it as string `"1024.0"`. TVM expects int and will `TypeError`.

**Impact**: Low - not idiomatic to pass floats, but violates robustness principle for config parsing.

**Fix**: Try `int(float(value))` or use `ast.literal_eval` with try/except.

### Summary of actionable correctness fixes

1. **tests/conftest.py:35** - Change `setdefault("...","1")` to `"0"` or delete. You’re forcing unsafe ON.
2. **_msl_transform.py:144** - Remove `if sys.platform == "darwin":` guard so attempt-cap works on Linux.
3. **_msl_transform.py:118** - Add `_preload_libz3_for_dev_tilelang._failed_attempts = 0` before setting `_done = True`.
4. **_msl_transform.py:835-838, 727-731** - Replace bare `except Exception: pass` with logging or raise.
5. **_msl_transform.py:815-817** - Handle float strings in target parsing.

The `OSError` vs `FileNotFoundError` split at lines 120-133 is correct and addresses the fix-round-5 finding. The `/tmp` gate at line 98 is correct.