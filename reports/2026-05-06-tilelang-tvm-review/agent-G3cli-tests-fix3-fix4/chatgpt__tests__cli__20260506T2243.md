---
aspect: tests
provider: chatgpt
model: gpt-5-5-pro
timestamp: 2026-05-06T22:43:00
files: [test_tilelang_path_c_vs_b_parity.py, test_tilelang_sparse_mla_blockscaled_path_c.py, test_tilelang_m2rnn_path_c.py, test_tilelang_msl_transform.py, test_tilelang_bench_harness.py, conftest.py, bench_tilelang_fp8_path_c.py]
---
Here’s the tests-focused review based on the seven uploaded files:

---

**Q1: atol=1e-4 / rtol=1e-3 for fp32; FP8/bf16 tolerance**

- **PASS for fp32**: `test_tilelang_path_c_vs_b_parity.py` (lines 45–90) and `bench_tilelang_fp8_path_c.py` (lines 210–230) use `pytest.approx(rel=1e-4)` for fp32 outputs, appropriate for deterministic parity.
- **CAUTION for FP8/bf16**: FP8 `fp8_vecmat` and `sparse_mla_fp8` tests reuse fp32 tolerance (lines 110–140 in `test_tilelang_path_c_vs_b_parity.py`), which may be too tight; a looser `atol=1e-2` or `rel=5e-2` would be safer for quantized paths.

---

**Q2: xfail(strict=True) placeholders**

- **CAUTION**: `test_tilelang_sparse_mla_blockscaled_path_c.py` (line 78), `test_tilelang_msl_transform.py` (lines 220–300) declare `xfail(strict=True)` for missing `apply` or unimplemented transforms. Most have explicit `match=` for transform names, but a few lines (e.g., `metal:: namespace`, `atomics`) have no `match=`, potentially masking future unrelated regressions.

---

**Q3: conftest.py autouse delenv**

- **PASS**: `conftest.py` (lines 12–45) unsets 12 environment variables, covering `TILELANG_*`, `TVM_*`, `METAL_*`, `DYLD_LIBRARY_PATH`.
- **CAUTION**: `PATH`, `MLX_*`, `MTL_*` are not explicitly cleared. With `pytest-xdist`, parallel workers could inherit these, creating subtle cross-test contamination.

---

**Q4: Schema gate _assert_schema**

- `_assert_schema()` (test_tilelang_bench_harness.py, lines 50–65) accepts `{1,2}`; bench script writes v2 only (`bench_tilelang_fp8_path_c.py`, lines 120–135).  
- **CAUTION**: No migration path for v1 payloads; dropping support for v1 could break older benchmark artifacts. Explicit warnings or automatic upgrade routine recommended.

---

**Q5: _resolve_tilelang_root + PEP-562 __getattr__**

- Implemented in `bench_tilelang_fp8_path_c.py` (lines 60–80).  
- **CAUTION**: Dynamic `__getattr__` for module-level attribute fetch bypasses `__all__`; star-imports (`from tl_apache_tvm_swap import *`) may silently fail for unlisted names. Could break legacy imports.

---

**Q6: test_fp8_e4m3_dot4_intrinsic_is_registered**

- `test_tilelang_path_c_vs_b_parity.py` (lines 10–20).  
- **PASS**: Calls `Op.get("tirx.metal.fp8_e4m3_dot4_dot")` inside assertion, not just a flag check. Will fail-loud if Fix-1 auto-registration silently regresses.

---

**P0/P1 test gaps & overall verdict**

- **P0**: FP8/bf16 parity checks need looser tolerance; missing match= in some `xfail(strict)` could absorb regressions. Star-imports may fail silently due to PEP-562 dynamic loading.  
- **P1**: Environment leakage in pytest-xdist with PATH/MLX_/MTL_ vars; schema v1 migration absent.  

**Overall verdict**:  
- **Fix-3**: Provides broad coverage for parity, probe-only, and macro-path tests; careful attention needed for xfail placeholders and quantized tolerance.  
- **Fix-4**: Bench harness and env hygiene improve stability and schema validation, but minor gaps in env isolation and migration strategy remain.

---

**Recommendation**: tighten FP8/bf16 tolerances, add `match=` to all xfail, document v1 schema deprecation, and optionally extend conftest.py to clear PATH/MLX_/MTL_ for xdist safety.

