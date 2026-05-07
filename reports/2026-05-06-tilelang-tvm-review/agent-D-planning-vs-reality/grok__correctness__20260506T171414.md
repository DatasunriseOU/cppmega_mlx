---
aspect: correctness
provider: grok
model: grok-4
range: (uncommitted)
base_ref: 
head_ref: ad5179ee8e20dc3e1651986fddece9dc6504de59
timestamp: 2026-05-06T17:14:14.724315+00:00
files: ['docs/upstream/_pr_filing_pack.md', 'docs/upstream/_path_c_blockers_tracker.md', 'docs/upstream/_path_c_followup_pr_candidates.md', 'docs/upstream/_filed_prs_2026_05_04.md', 'docs/upstream/local_build_status.md', 'docs/upstream/tvm_shared_storage/README.md', 'docs/upstream/tilelang_metal_pipelined/README.md', 'docs/upstream/tilelang_metal_pipelined_32x32/README.md', 'docs/upstream/tilelang_metal_fp8/README.md', 'docs/upstream/tilelang_metal_fp8_gemm/README.md', 'docs/upstream/tilelang_metal_fp8_scaled_matmul/README.md', 'docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/README.md', 'docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/PROBE_FAILED_2026-05-04.md', 'docs/upstream/tilelang_metal_fp8_vector/README.md', 'docs/upstream/tilelang_metal_blockscaled_e8m0/README.md', 'docs/upstream/tilelang_metal_emit_metal_builtins/README.md', 'docs/upstream/tilelang_metal_inline_kernel_body/README.md', 'docs/upstream/tilelang_metal_mamba3/README.md', 'docs/upstream/tilelang_metal_shared_dyn/README.md', 'docs/upstream/tilelang_gemm_mixed_dtype/README.md', 'docs/upstream/mlx_from_dlpack/README.md', 'cppmega_mlx/nn/_tilelang/_msl_transform.py', 'cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py', 'docs/production_kernel_routing.md', 'scripts/bench_tilelang_fp8_path_c.py', 'tests/test_tilelang_bench_harness.py', 'tests/test_tilelang_fp8_vecmat_path_c.py', 'tests/test_tilelang_msl_transform.py', 'bench/tilelang_ports/fp8_path_c_vs_path_b.json', 'bench/tilelang_ports/sparse_mla.json', 'bench/tilelang_ports/topk_selector.json']
---
**PLANNING-vs-REALITY AUDIT + CORRECTNESS REVIEW**  
**review_bundle.md** (31 files, focus: upstream PR coherence + Path C FP8 regression)

Ranked by severity (1 = highest impact on correctness / dispatch / filing readiness). All cite `<filename>:<line-range>` from the bundle. Tagged **(a)–(f)** per instructions.

### 1. (e) **CRITICAL REGRESSION: unregistered `tirx.metal.fp8_e4m3_dot4` blocks all FP8 Path C**  
`bench/tilelang_ports/fp8_path_c_vs_path_b.json:166-171, 361-371` (and traceback at 171)  
`bench/tilelang_ports/fp8_path_c_vs_path_b.json:333` (target parse error)  
`tests/test_tilelang_fp8_vecmat_path_c.py:217-221` (rejection test still present)  

**Bug**: After the uncommitted changes to `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py` + `_msl_transform.py` (both `M` in git status at json:69), the packed-dot4 intrinsic is no longer registered. Every `T.fp8_scaled_matmul` / vecmat Path C now fails with `AttributeError: Check failed: (reg != nullptr)`. This is **not** the pre-patch state (tracker claimed parity closed). Path C dispatch surface is broken; AUTO falls back to Path B (or crashes). **Introduced by recent diff** — exact registration call site in `fp8_vecmat_path_c.py` (not shown in bundle snippet) is mismatched with TileLang lowering after #2144/#2145.

### 2. (e) **CRITICAL: invalid target string passed to lowering**  
`bench/tilelang_ports/fp8_path_c_vs_path_b.json:333`  
`tests/test_tilelang_msl_transform.py:155-167` (`metal_grid_for_lowering`)  

**Bug**: `TileLangMSLLowering` (or caller in `fp8_vecmat_path_c.py`) still emits the legacy CLI string `"metal -thread_warp_size=32"`. TileLang now rejects it ("Cannot parse target string… use JSON dict"). **Regression** from the metal builtins / target refactor (Candidate #2 → #2143). Breaks vecmat Path C entirely; matmul_128 also dead.

### 3. (c) **TRACKER-STALE + (e) half-finished FP8 refactor**  
`_path_c_blockers_tracker.md:299-305` (2026-05-06 edit)  
`bench/tilelang_ports/fp8_path_c_vs_path_b.json:350-376` (sparse_mla FP8 section)  
`bench/tilelang_ports/sparse_mla.json:22-27, 85, 281, 374` (Infinity ratios + "did not dispatch")  

**Bug**: Tracker still calls #2146 a "tombstone" and says FP8 Path C blocked by unregistered dot4 — **exactly what the live bench JSON shows**. The uncommitted mods to `_msl_transform.py`, `fp8_vecmat_path_c.py`, and bench scripts were an attempt to fix it, but left the code in a broken state (no commit, no updated receipt). Sparse-MLA FP8 QK reducer and full dispatch gate are red. **Planning vs reality mismatch**: tracker claims "standalone matmul/vecmat speed parity closed locally" — reality is total dispatch failure.

### 4. (a) **PLANNED-BUT-MISSING: incomplete PR artifacts**  
`_pr_filing_pack.md:882-896` (explicit note)  
TOC + `_path_c_followup_pr_candidates.md:17-31, 98`  

- `tilelang_metal_inline_kernel_body/` → only `README.md` (and .metal?) **no 0001-*.patch**  
- `tilelang_metal_mamba3/` → only diff/metal/json, no patch  
- `tilelang_metal_fp8_scaled_matmul_fused_scheduler/` → `PROBE_FAILED_2026-05-04.md` instead of standard README+patch  

**Gap**: Filing pack claims "four PR-ready upstream artifacts" + "additional tilelang patches", but these directories are incomplete. Candidate #1 (inline body) was closed as false-alarm in ac80716 yet artifact was never cleaned.

### 5. (d) **FOLLOWUP-CANDIDATES doc stale**  
`_path_c_followup_pr_candidates.md:17-31, 80-86` (Candidate #1 + #2)  
`_filed_prs_2026_05_04.md:66-77` (#2143 builtins)  

**Bug**: Doc still lists Candidate #1 (inline_kernel_body) and #2 (emit builtins) as "new candidates NOT yet filed". Reality: #2143/#2147 filed (and Candidate #1 closed). No update after ac80716 / d86f337. Same for Candidate #2 (superseded by filed PR).

### 6. (e) **UNCOMMITTED-INCOHERENT: bench JSONs + test/code drift**  
`bench/tilelang_ports/fp8_path_c_vs_path_b.*.json` (multiple hand-edited variants)  
`bench/tilelang_ports/sparse_mla.json:21-28` (strict failures)  
`bench/tilelang_ports/topk_selector.json:134-142, 200, etc.` (Path C "did not dispatch")  
`tests/test_tilelang_msl_transform.py:19-142` (tests for `_inline_tilelang_kernel_body` etc.)  

**Bug**: 11 modified files (git status in fp8 json:69) include half-finished refactors. Bench JSONs contain stale Infinity ratios, old errors, and uncommitted `.current_strict.json` / `.live.json` artifacts. Tests expect Path C or `skip`, but real dispatch fails → **regressions to existing behaviour** (Path B was working). Orphaned helpers likely in `_msl_transform.py`.

### 7. (f) **CROSS-REF mismatch with production routing**  
`docs/production_kernel_routing.md` (M file)  
`cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py` (M) + `_msl_transform.py` (M)  

**Issue**: Routing doc (and `topk_selector.json` AUTO policy) claims Path C promotion for certain shapes, but live FP8 vecmat/sparse_mla benches show dispatch failure. Doc is now inaccurate after the uncommitted changes.

### 8. (e) **Incorrect error handling in MSL transform**  
`tests/test_tilelang_msl_transform.py:257-276` (`_ensure_single_libtvm_ffi_image`)  
`_msl_transform.py` (implied by test + dispatch calls)  

**Bug**: Raises `MSLDispatchUnsupported` on mixed libtvm_ffi.dylib — good in test, but if this bubbles up in production dispatch (e.g. during `dispatch(...)` in vecmat path) it can swallow the real root cause (unregistered op). No recovery / fallback logging visible in bench traces.

### 9. (e) **Potential alias-dropping edge case (off-by-one / comment handling)**  
`tests/test_tilelang_msl_transform.py:89-108` (`test_inline_body_ignores_comments_and_strings`)  
`_inline_tilelang_kernel_body` implementation (in `_msl_transform.py`)  

**Risk**: The regex/comment/string stripping in the inline helper is brittle. If a real kernel body has a comment containing `threadIdx` that the test doesn't cover, alias declarations may be incorrectly kept or dropped → malformed MSL (compile error at `xcrun metal -c`).

### 10. (b) **FILED-BUT-UNDOCUMENTED minor drift**  
`_filed_prs_2026_05_04.md:1-121` (covers #2139-#2147 + TVM #37-39)  
`_pr_filing_pack.md:3-39` (2026-05-04 reality check)  

**Minor**: Filed list looks consistent with recent commits (d86f337 etc.), but `_pr_filing_pack.md` header still references the old 4-PR plan. No mention of the later #2143-#2147 filings or the inline-body false-alarm close. Documentation lag, not a correctness bug.

---

**Summary of correctness impact**  
The **core regression** (findings 1-3,6) is in the FP8 Path C path: unregistered intrinsic + target parsing + stale tracker = broken dispatch for `fp8_scaled_vecmat_path_c`, sparse-MLA FP8, and anything using the new packed dot4 lowering. This was introduced by the uncommitted changes to `_msl_transform.py` / `fp8_vecmat_path_c.py` after the filed PRs. All other Path C ports (Mamba3, topk, BF16 sparse-MLA) are now at risk of silent fallback or failure.

**Recommended immediate actions** (actionable):  
1. Revert or fix the FP8 registration/target in `fp8_vecmat_path_c.py` + `_msl_transform.py`.  
2. Regenerate bench JSONs and update `_path_c_blockers_tracker.md`.  
3. Complete or delete the incomplete PR directories (a).  
4. Sync followup-candidates doc.  
5. Run full test suite (`test_tilelang_*`) before any commit.

No performance suggestions until correctness is restored — current "Path C" is slower than Path B because it doesn't run at all.  

**Output ready for `output_dir` as `planning_vs_reality_audit.md`.**