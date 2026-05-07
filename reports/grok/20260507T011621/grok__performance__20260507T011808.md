---
aspect: performance
provider: grok
model: gpt-5-5-pro
range: HEAD~1..HEAD
base_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
head_ref: aef115bb0fa97b68b541580d957ff6b043e76176
timestamp: 2026-05-07T01:18:08.076972+00:00
files: ['bench/tilelang_ports/_archive/README.md', 'bench/tilelang_ports/_archive/fp8_path_c_vs_path_b.current_strict.json', 'bench/tilelang_ports/_archive/fp8_path_c_vs_path_b.live.json', 'bench/tilelang_ports/_archive/fp8_path_c_vs_path_b.quick.after_flat_b_word.json', 'bench/tilelang_ports/_archive/fp8_path_c_vs_path_b.quick.before_new_patch.json', 'bench/tilelang_ports/_archive/fp8_path_c_vs_path_b.quick.current.json', 'bench/tilelang_ports/_archive/fp8_path_c_vs_path_b.quick.json', 'bench/tilelang_ports/fp8_path_c_vs_path_b.json', 'bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json', 'cppmega_mlx/nn/_tilelang/__init__.py', 'cppmega_mlx/nn/_tilelang/_experimental.py', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'cppmega_mlx/nn/_tilelang/mamba3.py', 'cppmega_mlx/nn/_tilelang/mamba3_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_fp8_path_c.py', 'cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py', 'cppmega_mlx/nn/_tilelang/topk_selector.py', 'docs/production_kernel_routing.md', 'docs/upstream/_path_c_blockers_tracker.md', 'reports/2026-05-06-tilelang-tvm-review/agent-C-fp8-mamba/grok__all__20260506T170449.md', 'reports/2026-05-06-tilelang-tvm-review/agent-C-fp8-mamba/grok__correctness__20260506T170226.md', 'reports/2026-05-06-tilelang-tvm-review/agent-C-fp8-mamba/grok__correctness__20260506T171218.md', 'reports/2026-05-06-tilelang-tvm-review/agent-C-fp8-mamba/grok__design__20260506T170227.md', 'reports/2026-05-06-tilelang-tvm-review/agent-C-fp8-mamba/grok__design__20260506T171239.md', 'reports/2026-05-06-tilelang-tvm-review/agent-C-fp8-mamba/grok__performance__20260506T171102.md', 'reports/2026-05-06-tilelang-tvm-review/agent-D-planning-vs-reality/grok__all__20260506T171424.md', 'reports/2026-05-06-tilelang-tvm-review/agent-D-planning-vs-reality/grok__correctness__20260506T171414.md', 'reports/2026-05-06-tilelang-tvm-review/agent-D-planning-vs-reality/grok__design__20260506T171408.md', 'reports/2026-05-06-tilelang-tvm-review/agent-E-tests-benches/meta__all__20260506T170334.md', 'reports/2026-05-06-tilelang-tvm-review/agent-E-tests-benches/meta__all__20260506T170743.md', 'reports/2026-05-06-tilelang-tvm-review/agent-E-tests-benches/meta__correctness__20260506T170721.md', 'reports/2026-05-06-tilelang-tvm-review/agent-E-tests-benches/meta__performance__20260506T170642.md', 'reports/2026-05-06-tilelang-tvm-review/agent-E-tests-benches/meta__tests__20260506T170346.md', 'reports/2026-05-06-tilelang-tvm-review/agent-E-tests-benches/meta__tests__20260506T170633.md', 'reports/2026-05-06-tilelang-tvm-review/agent-F-path-b-vs-c/meta__all__20260506T170603.md', 'reports/2026-05-06-tilelang-tvm-review/agent-F-path-b-vs-c/meta__correctness__20260506T170505.md', 'reports/2026-05-06-tilelang-tvm-review/agent-F-path-b-vs-c/meta__design__20260506T170611.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G-gpt55pro-fixwave/chatgpt__all__20260506T221347.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G-gpt55pro-fixwave/chatgpt__correctness__20260506T221324.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G-gpt55pro-fixwave/chatgpt__design__20260506T221409.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G-gpt55pro-fixwave/chatgpt__performance__20260506T221530.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G-gpt55pro-fixwave/chatgpt__security__20260506T221429.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G-gpt55pro-fixwave/chatgpt__tests__20260506T221511.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G1-correctness-fix1-fix5/chatgpt__correctness__20260506T222948.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G1-correctness-fix1-fix5/chatgpt__correctness__20260506T223129.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G1cli-correctness-fix1-fix5/chatgpt__correctness__20260506T223840.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G1cli-correctness-fix1-fix5/chatgpt__correctness__cli__20260506T2241.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G2-design-fix2/chatgpt__design__20260506T222914.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G2cli-design-fix2/chatgpt__design__20260506T223902.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G3-tests-fix3-fix4/chatgpt__tests__20260506T223018.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G3-tests-fix3-fix4/chatgpt__tests__20260506T223056.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G3cli-tests-fix3-fix4/chatgpt__tests__20260506T223934.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G3cli-tests-fix3-fix4/chatgpt__tests__cli__20260506T2243.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G4-correctness-fix-a-c/chatgpt__correctness__cli__20260506T2310.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G5-design-fix-f-g/chatgpt__design__cli__20260506T2312.md', 'reports/2026-05-06-tilelang-tvm-review/agent-G6-tests-fix-b-d-e/chatgpt__tests__cli__20260506T2316.md', 'reports/chatgpt/20260507T010406/chatgpt__correctness__20260507T010634.md', 'reports/chatgpt/20260507T010406/chatgpt__design__20260507T010747.md', 'reports/chatgpt/20260507T010406/chatgpt__performance__20260507T011012.md', 'reports/chatgpt/20260507T010406/chatgpt__security__20260507T011112.md', 'reports/grok/20260507T010406/grok__correctness__20260507T010413.md', 'reports/grok/20260507T010406/grok__design__20260507T010416.md', 'reports/grok/20260507T010406/grok__performance__20260507T010426.md', 'reports/grok/20260507T010406/grok__security__20260507T010429.md', 'scripts/bench_tilelang_fp8_path_c.py', 'tests/conftest.py', 'tests/test_tilelang_bench_harness.py', 'tests/test_tilelang_fp8_vecmat_path_c.py', 'tests/test_tilelang_m2rnn_path_c.py', 'tests/test_tilelang_msl_transform.py', 'tests/test_tilelang_path_c_vs_b_parity.py', 'tests/test_tilelang_sparse_mla_blockscaled_path_c.py']
---
**Performance-focused review of the TileLang Path C fix-wave (commit ebbb84b atop MLX-wiring).**  
The bundle shows a mature integration with good test coverage and strict gating, but the live/strict bench snapshots reveal lingering **hot-path performance regressions** in Path C vs. hand-tuned Path B (vendored MSL LUT + integer-bit encode kernels). I prioritized **quantifiable regressions**, allocation/loop issues, and hot-path concerns over generic advice. I cross-checked against the archived `.current_strict.json`, `.live.json`, and test files.

### 1. Critical Performance Regressions (P0/P1) — Quantified from Benches

**fp8_vecmat_path_c.py + _msl_transform.py: Path C vecmat is 11–115% slower than Path B on canonical shapes (P0 regression).**  
- **bench/tilelang_ports/fp8_path_c_vs_path_b.live.json** (and `.current_strict.json:1025`):  
  - `vecmat_4096`: `path_c_mlx_tilelang_fp8_scaled_vecmat_over_path_b` **median ratio = 1.11555** (p90=1.377, max=4.10).  
  - Worst paired steps show 3–4× spikes (e.g., step 23: 0.968ms vs 0.236ms Path B).  
  - `matmul_128` is marginally better (~0.90 median) but has tail latency spikes up to 2.85×.  
- **Root cause in hot loop (vecmat_4096 is the production path for many M=1 contracts):**  
  - `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py: canonical_vecmat_runtime_body` (and lowering via `_msl_transform.py:_inline_tilelang_kernel_body`) emits **packed `uint32` loads + LUT decode + `simd_sum`**, but still incurs more overhead than Path B’s hand-tuned LUT + direct integer-bit path.  
  - `tests/test_tilelang_fp8_vecmat_path_c.py:56–70` (and `fp8_vecmat_msl_features`) confirm `reinterpret_cast<device const uint*>`, `metal_fp8_dot4_packed` (9×), `metal_fp8_dot4_helper` (20×), but **no evidence of full unrolling or register pressure tuning** that Path B has.  
  - `scripts/bench_tilelang_fp8_path_c.py` (and harness in `tests/test_tilelang_bench_harness.py`) uses paired timing, which is correct for fairness but hides that Path C’s Metal kernel launch + transform overhead dominates on small/medium tiles.  

**Impact:** On real workloads (vecmat-heavy routing in production_kernel_routing.md), this is a **~10–15% end-to-end regression** for FP8 inference paths. Tail latency (p99 4×) risks jitter in latency-sensitive serving.

**Recommendation (actionable):**  
- Profile the generated MSL (`--dump-dir` in bench script) with Metal GPU counters or `xctrace`. Look for excess threadgroup barriers, suboptimal `simd_sum` usage, or missing `[[max_total_threads_per_threadgroup]]` / occupancy hints.  
- Consider **inline assembly or tighter macro** for the dot4 helper in TileLang’s `tilelang/language/fp8_op.py` or Metal codegen (see also `tilelang/tileop/metal_quant.py`).  
- Gate stricter in `fp8_path_c_vs_path_b.json` strict_policy (currently allows 1.0 max_ratio but live data violates it). Add a dedicated `vecmat_path_c_over_path_b_p90_max` < 1.05 policy.

### 2. Hot-Path Concerns & Allocation/Loop Issues (P1)

**_msl_transform.py: _inline_tilelang_kernel_body does repeated string operations on every kernel dispatch.**  
- Lines ~19–130 (the whole inliner + `_split_kernel_msl`, `_parse_buffer_param_names`): heavy regex-free string scans (`"threadIdx" in body`, splits, replacements) for every lowered kernel.  
- Called from `fp8_vecmat_path_c.py`, `sparse_mla_*_path_c.py`, `mamba3_path_c.py`, etc.  
- In tight dispatch loops (e.g., auto-routing in production_kernel_routing.md or repeated calls in `nn/_tilelang/_experimental.py`), this becomes **O(source_length) per call** with allocations. Not O(n²), but unnecessary for hot kernels.  
- `test_tilelang_msl_transform.py:194–217` (split_kernel_msl) shows the parser is careful with comments/strings, but still allocates new strings repeatedly.  

**Impact:** Minor on cold paths; measurable on high-QPS inference (extra CPU time before Metal launch).  

**Fix:** Cache lowered MSL per kernel signature (use a simple `@lru_cache` on canonical params in the dispatch surface) or move more transforms into TVM/Metal codegen pass (preferred long-term).

**Allocation in bench harness & tests (tight-loop risk):**  
- `tests/test_tilelang_bench_harness.py` and `scripts/bench_tilelang_fp8_path_c.py`: repeated `mx.array` creation + `mx.eval` in paired benches (see live.json sample_ms arrays).  
- `test_tilelang_path_c_vs_b_parity.py:224–232` (the parametrized sweep) does full forward passes with random data on every test run — fine for CI, but if run in hot CI loops or with large shapes, watch memory.  
- `_drive_fp8_vecmat` etc. create fresh RNG + arrays each time.

**No major O(n²) or N+1 seen**, but the paired bench alternates paths (`_bench_paired_callables` in harness) which is good for fairness yet doubles warmup/sync cost.

### 3. Correctness Bugs & Regressions (Skeptical Cross-Check vs. GPT Review)

**Intrinsic registration trip-wire is duplicated and fragile (P1 correctness hole).**  
- `tests/test_tilelang_fp8_vecmat_path_c.py:125–154` (and duplicated in `test_tilelang_path_c_vs_b_parity.py:302–330`).  
- `_try_get_global_func` walks multiple import paths (`tvm_ffi`, `tilelang.tvm`, legacy `_ffi`) with broad `except Exception`. If registration fails silently on some hosts (e.g., partial TVM build), Path C falls back to scalar decode → **massive perf regression + correctness drift** (as noted in Grok-D P0).  
- The test uses `pytest.importorskip("tvm")` but still risks false-green if `Op.get` or global func lookup partially succeeds.  

**Shape contract & back-compat in fp8_vecmat_path_c.py (P2 regression risk).**  
- `fp8_scaled_vecmat_path_c` always reshapes output back to flat `(n,)` (see test:119–121 comment on Fix-1 + Fix-A).  
- `tests/test_tilelang_fp8_vecmat_path_c.py:223–227` enforces K % 4 == 0 for packed dot4, but runtime body (`canonical_vecmat_runtime_body`) has per-row vs scalar scale paths that could diverge if scale layout changes.  
- `_experimental.py` (mentioned in bundle) — I didn’t see breaking changes, but ensure any new TileLang eager paths don’t bypass the MSL transform.

**conftest.py env scrub & schema deprecation:**  
- `tests/conftest.py` (and env_contract in JSONs) scrubs PYTHONPATH/TVM_LIBRARY_PATH etc. — looks reasonably complete, but `removed_import_finders` handling could leak if multiple editable installs coexist (see module_origins in live.json).  
- No obvious schema deprecation path issues in the JSON receipts (schema_version=1 stable), but `sparse_mla*.json` and blockscaled receipts still gate full_fwd_bwd behind "dispatch_surface" checks — good, but the xfail in `test_tilelang_sparse_mla_blockscaled_path_c.py:183–189` and `test_tilelang_path_c_vs_b_parity.py` will hide real regressions until the apply lands.

**_msl_transform.py address-space & namespace fixes (G4/G6) are xfailed but critical for future perf.**  
- The xfailed tests (e.g., `test_inline_body_prefixes_threadgroup_barrier...`) target real Metal canonicalization gaps that could cause subtle register spills or incorrect lowering if not fixed before wider adoption.

### 4. Other Observations (No Major Regressions Found Here)

- Sparse-MLA / Mamba3 / blockscaled paths correctly expose only partial surfaces today → xfail strategy is appropriate (strict=True prevents silent drift).  
- No redundant I/O, blocking calls in async paths, or obvious memory growth in the provided sources.  
- Bench JSONs include excellent diagnostics (`source_metrics`, `path_c_blockers`, `worst_paired_steps`) — keep this; it makes regressions debuggable.  
- TileLang version pinning and TVM submodule state look healthy (cab1a7db head).

### Summary & Prioritized Action Items

**P0 (fix before merge):**  
- Vecmat Path C slowdown (1.11× median, 4× tails) on `vecmat_4096` — root in lowering/hot-loop vs Path B. Profile + tighten or fallback selectively in production routing. Update strict_policy max_ratio or add p90 gate.

**P1:**  
- Cache or move `_inline_tilelang_kernel_body` work out of hot dispatch.  
- Harden FP8 dot4 intrinsic test (narrow exceptions, add version check).  
- Ensure packed dot4 path is always taken when K%4==0 and vec=4 (Fix-1/A already helped; confirm in generated MSL).

**P2:**  
- Land the missing `sparse_mla_blockscaled_path_c_apply` (and m2rnn) so xfailed parity tests become real gates.  
- Resolve the xfailed MSL canonicalization items (namespace qualifiers, address spaces) — they prevent future perf cliffs.

The fix-wave cleaned up many correctness holes (good test surface, blockers tracking), but **the core performance promise of Path C (TileLang-generated kernels matching or beating hand-written Path B) is not yet fully met on the vecmat hot path**. The paired benches and strict receipts make this visible — use them.

If you share Metal profiler output for the vecmat kernel or the exact generated MSL source for the slow cases, I can drill deeper. Overall, solid engineering; the regressions are fixable with targeted lowering tweaks.