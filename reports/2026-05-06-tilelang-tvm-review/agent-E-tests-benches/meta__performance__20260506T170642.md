---
aspect: performance
provider: meta
model: meta-muse-spark
range: (uncommitted)
base_ref: 
head_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
timestamp: 2026-05-06T17:06:42.286014+00:00
files: ['tests/test_tilelang_bench_harness.py', 'tests/test_tilelang_fp8_matmul_path_c_bench.py', 'tests/test_tilelang_fp8_vecmat_path_c.py', 'tests/test_tilelang_m2rnn.py', 'tests/test_tilelang_mamba3.py', 'tests/test_tilelang_mamba3_helpers.py', 'tests/test_tilelang_mamba3_path_c.py', 'tests/test_tilelang_msl_transform.py', 'tests/test_tilelang_sparse_mla.py', 'tests/test_tilelang_sparse_mla_blockscaled.py', 'tests/test_tilelang_sparse_mla_fp8.py', 'tests/test_tilelang_topk.py', 'scripts/bench_tilelang_fp8_path_c.py', 'scripts/bench_tilelang_mamba3.py', 'scripts/bench_tilelang_mamba3_helpers.py', 'scripts/bench_tilelang_mamba3_path_c.py', 'scripts/bench_tilelang_sparse_mla.py', 'scripts/bench_tilelang_sparse_mla_fp8.py', 'scripts/bench_tilelang_topk.py', 'bench/tilelang_ports/fp8_path_c_vs_path_b.json', 'bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'bench/tilelang_ports/fp8_msl_kernels.json', 'bench/tilelang_ports/fp8_path_c.json', 'bench/tilelang_ports/lane7_fp8_vecmat_4096.json', 'bench/tilelang_ports/m2rnn_path_b.json', 'bench/tilelang_ports/mamba3_helpers.json', 'bench/tilelang_ports/mamba3_path_c.json', 'bench/tilelang_ports/mamba3.json', 'bench/tilelang_ports/sparse_mla_blockscaled.json', 'bench/tilelang_ports/sparse_mla_fp8.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.current_strict.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.live.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.after_flat_b_word.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.before_new_patch.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.quick.current.json', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'docs/production_kernel_routing.md', 'docs/upstream/_path_c_blockers_tracker.md']
---
I tried to pull the specific sections from `review_bundle.md` to do a file:line analysis, but my file search didn’t return any excerpts from the bundle — looks like I can't access the content directly right now. 

To give you the performance regression + hot-path review you asked for, I’ll need to see the actual code blocks. Can you either:

1. **Paste the key sections** for each shard you listed, or 
2. **Break the bundle into smaller files** per shard and re-upload, or
3. **Tell me which specific functions/lines** to focus on first

### What I’ll do once I have the code

**SHARD 1 — tests/test_tilelang_bench_harness.py**  
I'll scan for: 
- `np.allclose`, `torch.allclose`, `assert_close` with `atol>=1e-2` or `rtol>=5e-2` on fp32 
- `@pytest.mark.skip`, `@pytest.mark.xfail` decorators hiding perf regressions
- JSON schema validation: compare `jsonschema.validate()` keys vs what `scripts/bench_tilelang_fp8_path_c.py` actually writes

**SHARD 2 — scripts/bench_tilelang_fp8_path_c.py**  
I'll diff the output dict keys against the 6 snapshot files you mentioned to flag: added/removed/renamed keys, unit changes `ms` vs `us`, and identify which snapshot is the source-of-truth. Plus look for O(n^2) loops, list/dict reallocation in the bench loop, and sync `device.synchronize()` in the wrong place.

**SHARD 3 — Coverage gaps**  
I’ll grep for `path_c` vs `path_b` comparisons in tests, and flag missing `*_path_c` test files for `sparse_mla_blockscaled`, `mamba3`, `m2rnn`.

**SHARD 4 — Env-var leaks**  
Find `os.environ['TVM_*']`, `TILELANG_*`, `METAL_DEVICE_WRAPPER_TYPE` reads without `monkeypatch.setenv` + teardown.

**SHARD 5 — Snapshot rot**  
Cross-ref `*.before_new_patch.json` etc against `pytest` collection to see if any test actually loads them.

### Quick performance heuristics I’ll apply

| Anti-pattern | Impact | Where I’ll look |
| --- | --- | --- |
| O(n^2) nested loops over tensors | 100x-1000x slowdown on L>4k | Bench loops, reduction code |
| Python list `append` in tight loop | GC pressure, 2-5x slower | Result collection in benchmarks |
| Missing `@torch.no_grad()` | 2x memory, autograd overhead | FP8 path_c kernel calls |
| `torch.cuda.synchronize()` per-iter | Kills async pipeline | Benchmark timing code |
| Re-reading JSON per test | N+1 I/O, 100ms+ per test | `json.load()` in test setup |
| Large dict copies for schema | 10-50MB alloc per run | Result serialization |

If you can drop even just `tests/test_tilelang_bench_harness.py` and `scripts/bench_tilelang_fp8_path_c.py` as separate files, I can start SHARD 1 + 2 right now with file:line citations and quantify the hot-path cost.

Want to start with one shard?