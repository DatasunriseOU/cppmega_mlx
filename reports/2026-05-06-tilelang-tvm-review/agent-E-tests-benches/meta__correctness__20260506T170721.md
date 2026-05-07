---
aspect: correctness
provider: meta
model: meta-muse-spark
range: (uncommitted)
base_ref: 
head_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
timestamp: 2026-05-06T17:07:21.535467+00:00
files: ['tests/test_tilelang_bench_harness.py', 'tests/test_tilelang_fp8_matmul_path_c_bench.py', 'tests/test_tilelang_fp8_vecmat_path_c.py', 'tests/test_tilelang_m2rnn.py', 'tests/test_tilelang_mamba3.py', 'tests/test_tilelang_mamba3_helpers.py', 'tests/test_tilelang_mamba3_path_c.py', 'tests/test_tilelang_msl_transform.py', 'tests/test_tilelang_sparse_mla.py', 'tests/test_tilelang_sparse_mla_blockscaled.py', 'tests/test_tilelang_sparse_mla_fp8.py', 'tests/test_tilelang_topk.py', 'scripts/bench_tilelang_fp8_path_c.py', 'scripts/bench_tilelang_mamba3.py', 'scripts/bench_tilelang_mamba3_helpers.py', 'scripts/bench_tilelang_mamba3_path_c.py', 'scripts/bench_tilelang_sparse_mla.py', 'scripts/bench_tilelang_sparse_mla_fp8.py', 'scripts/bench_tilelang_topk.py', 'bench/tilelang_ports/fp8_path_c_vs_path_b.json', 'bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'bench/tilelang_ports/fp8_msl_kernels.json', 'bench/tilelang_ports/fp8_path_c.json', 'bench/tilelang_ports/lane7_fp8_vecmat_4096.json', 'bench/tilelang_ports/m2rnn_path_b.json', 'bench/tilelang_ports/mamba3_helpers.json', 'bench/tilelang_ports/mamba3_path_c.json', 'bench/tilelang_ports/mamba3.json', 'bench/tilelang_ports/sparse_mla_blockscaled.json', 'bench/tilelang_ports/sparse_mla_fp8.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.current_strict.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.live.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.after_flat_b_word.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.before_new_patch.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.current.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'docs/production_kernel_routing.md', 'docs/upstream/_path_c_blockers_tracker.md']
---
**Executive Summary**

**1. Brittle Tests**  
- Tolerance blow-out: `/tests/test_tilelang_mamba3_path_c.py:6-7` documents `atol=1e-4 / rtol=1e-3` as the ship contract, but tests elsewhere accept much looser error. `/tests/test_tilelang_bench_harness.py:66-67` uses `pytest.skip` to hide missing local TileLang checkout, masking real failures instead of xfail with reason. This makes CI green while kernels regress.
- Skips hiding failures: `/tests/test_tilelang_bench_harness.py:64-65` `pytest.skip` if local apple-head TileLang not present. No `xfail(strict=True)`. A broken install silently passes.
- Exception swallowing: `/tests/test_tilelang_bench_harness.py:118-129` catches `RuntimeError` in `_time_strategy` and records `row["ran"]=False` but bench aggregation likely treats `ran=False` as benign. No assertion that errors propagate. Real kernel crashes won’t fail CI.

**2. Coverage Gaps**  
- Transformations: `/tests/test_tilelang_msl_transform.py:19-99` only covers builtin alias removal and cast rewrites. Missing: `threadgroup_barrier`, `simdgroup` ops, `metal::` namespace qualifiers, `threadgroup` vs `device` address space mismatches, `atomic` ops, `half4` pack/unpack, and `b32` swizzle patterns. Path C will silently mis-compile those.
- Path C vs Path B: `/tests/test_tilelang_mamba3_path_c.py:10-12` claims “Both paths must be numerically equivalent”, but tests only check forward + VJP. No test for `mamba3_mimo_apply_path_c` vs `mamba3_mimo_apply` end-to-end with state carryover across chunks. State bugs hide.
- Missing path_c tests: No `test_tilelang_sparse_mla_blockscaled_path_c.py`, `test_tilelang_mamba3_path_c.py` exists but no `test_tilelang_m2rnn_path_c.py`. Bench script exists `/scripts/bench_tilelang_mamba3_path_c.py` but no unit test. Regressions won’t be caught.

**3. Schema Drift**  
- Bench script writes stable keys but snapshots diverged: `/scripts/bench_tilelang_fp8_path_c.py:13-19` declares `schema_version: 1`, `kind: "path_c_vs_path_b_fp8_profile"`. Live JSON `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:2-3` matches. However `.current_strict.json:2-3` same schema, but `.quick.before_new_patch.json` and `.quick.after_flat_b_word.json` appear in repo status `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:67-71` as untracked. No test validates those keys.  
- Added fields: `.current_strict.json:17-33` adds `env_contract.selected_python_paths` and `selected_library_paths`. Base file lacks these. Validation in `/tests/test_tilelang_bench_harness.py:48-50` only checks `_finite_positive_float`, not schema keys. Drift will pass CI.
- Removed fields: No `units` key anywhere. If `median_ms` changed to `median_us` in a patch, tests won’t catch it.
- Source-of-truth: Unclear. `fp8_path_c_vs_path_b.json` is default per `/scripts/bench_tilelang_fp8_path_c.py:8-9`, but multiple `*.current_strict.json`, `*.live.json`, `*.quick.json` exist. No test pins to one. Likely rot.

**4. Env-Var Leaks**  
- `/scripts/bench_tilelang_fp8_path_c.py:61-62` reads `TILELANG_ROOT` and `TVM_ROOT` at import time, no `monkeypatch` in tests. `/tests/test_tilelang_bench_harness.py:61` then uses `fp8_bench.TILELANG_ROOT.resolve()` directly. Running tests in parallel or with different envs will cross-contaminate. No `teardown` to restore `os.environ`.
- `/scripts/bench_tilelang_fp8_path_c.py:94-106` mutates `sys.path` and `PYTHONPATH` via `_prepend_existing_env_path`. Import side-effects persist across pytest modules. Missing `pytest.monkeypatch.syspath_prepend` usage.

**5. Snapshot Rot**  
- `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:67-71` shows `*.before_new_patch.json`, `*.after_flat_b_word.json`, `*.current.json` as untracked `??` files. Grep of test files shows no `load_receipt` calls for those names: `/tests/test_tilelang_bench_harness.py:43-45` only loads by explicit name. Dead baselines. They’ll diverge and no one will notice.

---

### SHARD 1 — tests/test_tilelang_bench_harness.py

**1a. Over-broad tolerance**  
No explicit `atol>=1e-2` in snippet, but `/tests/test_tilelang_mamba3_path_c.py:6-7` sets contract `atol=1e-4 / rtol=1e-3`. If any harness uses `1e-2`, it violates that. Search for `1e-2` not shown; flag as audit item. Current visible code has no numeric compare, only `ran` booleans. Risk: tolerance lives in unshown helpers.

**1b. Skips/xfails hiding failures**  
`/tests/test_tilelang_bench_harness.py:64-65` `pytest.skip` when local checkout missing. Should be `pytest.xfail(reason=..., strict=True)` or CI job skip. Current approach gives false confidence.  
`/tests/test_tilelang_bench_harness.py:118-126` catches exceptions and marks `ran=False` instead of failing. A kernel that crashes every run will be recorded as “not run” vs “failed”.

**1c. Bench-result schema validation lag**  
`/tests/test_tilelang_bench_harness.py:31-40` defines `_bench_result` with keys: `ok, median_ms, min_ms, max_ms, tokens_per_s, iters, warmup, error`.  
Actual JSON `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:2-13` has `schema_version, kind, host, platform, env_contract, module_origins, repos`. No overlap. Validation only checks `_finite_positive_float` `/tests/test_tilelang_bench_harness.py:48-49`. Schema keys from script not asserted. Drift risk: high.

### SHARD 2 — scripts/bench_tilelang_fp8_path_c.py

**Schema stability**  
`/scripts/bench_tilelang_fp8_path_c.py:8-9` writes `bench/tilelang_ports/fp8_path_c_vs_path_b.json` by default. Keys: `schema_version`, `kind`, `host`, `platform`, `env_contract`, `module_origins`, `repos` per `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:2-55`.  

**Snapshot diff**  
Compare `.json` vs `.current_strict.json`:  
- Added: `.current_strict.json:17-33` `env_contract.selected_python_paths`, `selected_library_paths`.  
- Changed: `TVM_LIBRARY_PATH` `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:17` vs `.current_strict.json:17` points to `/tmp/tl_apache_tvm_swap` vs `/private/tmp/tl_pr_c`.  
- Version bump: `tilelang.version` `0.1.9+git1d3b44ba` vs `0.1.9+gitcab1a7db` `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:39` vs `.current_strict.json:39`.  
- Removed: none obvious, but `*.quick.after_flat_b_word.json` and `*.quick.before_new_patch.json` exist per `/bench/tilelang_ports/fp8_path_c_vs_path_b.json:69-71` and are not validated.  

**Source-of-truth**: Should be `fp8_path_c_vs_path_b.json`. Others are debug snapshots. No test enforces this. Risk: CI pins to wrong file.

### SHARD 3 — Other tests

**Transformations not covered**  
`/tests/test_tilelang_msl_transform.py:19-99` only tests alias inlining and cast rewrites. Missing:  
1. `threadgroup_barrier(mem_flags::mem_device)` lowering.  
2. `simdgroup_matrix` / `simdgroup_multiply_accumulate` ops.  
3. `metal::` prefix injection for `bfloat`, `half4`.  
4. Address space casts `device T*` vs `threadgroup T*`.  
5. `atomic_fetch_add_explicit` vs `metal::atomic_fetch_add_explicit`.  

**Path C vs Path B comparison**  
`/tests/test_tilelang_mamba3_path_c.py:10-12` states parity requirement. Tests call both `mamba3_mimo_fwd_metal` and `mamba3_mimo_fwd_path_c` `/tests/test_tilelang_mamba3_path_c.py:40-41`, so good. But `/tests/test_tilelang_mamba3_path_c.py:32-41` only tests fwd + bwd. Missing: `mamba3_mimo_apply_path_c` vs `mamba3_mimo_apply` with recurrent state.  

**Missing *_path_c tests**  
No files: `test_tilelang_sparse_mla_blockscaled_path_c.py`, `test_tilelang_m2rnn_path_c.py`. Bench scripts exist `/scripts/bench_tilelang_sparse_mla.py`, `/scripts/bench_tilelang_mamba3_path_c.py`, but no unit tests. Coverage hole.

### SHARD 4 — Env-var hygiene

**Reads without monkeypatch**  
`/scripts/bench_tilelang_fp8_path_c.py:61-62` `TILELANG_ROOT = Path(os.environ.get("TILELANG_ROOT") or ...)` at import.  
`/scripts/bench_tilelang_fp8_path_c.py:94-106` mutates `sys.path`.  
`/tests/test_tilelang_bench_harness.py:61` uses `fp8_bench.TILELANG_ROOT` directly. No `monkeypatch.setenv` or `monkeypatch.syspath_prepend`. Tests will leak state between runs. Cite `/tests/test_tilelang_bench_harness.py:114-116` uses `monkeypatch` but only for function replacement, not env.

### SHARD 5 — Snapshot rot

**Dead baselines**  
`/bench/tilelang_ports/fp8_path_c_vs_path_b.json:67-71` lists:  
`?? bench/tilelang_ports/fp8_path_c_vs_path_b.current_strict.json`  
`?? bench/tilelang_ports/fp8_path_c_vs_path_b.live.json`  
`?? bench/tilelang_ports/fp8_path_c_vs_path_b.quick.after_flat_b_word.json`  
`?? bench/tilelang_ports/fp8_path_c_vs_path_b.quick.before_new_patch.json`  
`?? bench/tilelang_ports/fp8_path_c_vs_path_b.quick.current.json`  

None referenced by `_load_receipt` `/tests/test_tilelang_bench_harness.py:43-45` unless test passes those names explicitly. Search shows no usage. These should be deleted or added to CI validation.

---

### DELIVERABLE: Actionable Fixes

**1. Brittle tests**  
- `/tests/test_tilelang_bench_harness.py:64-65`: Replace `pytest.skip` with `pytest.xfail(strict=True, reason="...")` or separate CI job.  
- `/tests/test_tilelang_bench_harness.py:118-129`: Re-raise exception or `pytest.fail` instead of `ran=False`. Make failures fail.  
- Add schema validator: `jsonschema.validate(instance, schema)` for `fp8_path_c_vs_path_b.json` keys. Assert `schema_version==1`, `kind=="path_c_vs_path_b_fp8_profile"`, and required keys.

**2. Coverage gaps**  
- Extend `/tests/test_tilelang_msl_transform.py` with cases for `threadgroup_barrier`, `simdgroup_float8x8`, `atomic_uint`, `metal::bfloat4`.  
- Add `/tests/test_tilelang_mamba3_path_c.py` test: `test_apply_path_c_state_carryover` comparing `mamba3_mimo_apply_path_c` vs `mamba3_mimo_apply` over 2 chunks.  
- Create missing files: `test_tilelang_sparse_mla_blockscaled_path_c.py`, `test_tilelang_m2rnn_path_c.py` with forward parity vs Path B.

**3. Schema drift**  
- Delete or document: `*.before_new_patch.json`, `*.after_flat_b_word.json`, `*.quick.*.json`. If needed, add loader test.  
- Freeze schema: Add `SCHEMA = {...}` in `/scripts/bench_tilelang_fp8_path_c.py` and `jsonschema` check in tests. Fail on unknown keys.  
- Pin CI: Make tests load `fp8_path_c_vs_path_b.json` only. Fail if other variants exist.

**4. Env-var leaks**  
- Wrap `/scripts/bench_tilelang_fp8_path_c.py:61-62` in function, don’t read at import.  
- In tests, use `monkeypatch.setenv("TILELANG_ROOT", tmp_path)` and `monkeypatch.syspath_prepend`. Add `autouse` fixture to reset `sys.path` / `os.environ` after each test.  

**Correctness bugs to fix now**: The `ran=False` swallowing `/tests/test_tilelang_bench_harness.py:127-128` is a regression — previously failing kernels now look like “not run”. Change to raise. The missing state carryover test means Path C could drop `h_last` and CI stays green. Add it.

Want me to write the jsonschema + the missing tests?