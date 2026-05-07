---
aspect: design
provider: chatgpt
model: gpt-5-5-pro
timestamp: 2026-05-06T23:12
files: [__init__.py, _experimental.py, bench_tilelang_fp8_path_c.py]
---
Here’s the detailed DESIGN review based on the three uploaded files (`__init__.py`, `_experimental.py`, `bench_tilelang_fp8_path_c.py`):

---

### 1. Fix-G — `from ._experimental import *` idiom

**Assessment:** **CAUTION**  
**Evidence:** `_experimental.py` (file_000000009358720a8e50d9afcc9033ab) defines `__all__` explicitly:  

```python
__all__ = [
    "FP8VecmatPathCStatus",
    "fp8_vecmat_msl_features",
    "fp8_vecmat_path_c_status",
    ...
]
```

- Because `__all__` is explicitly set, the star import in `__init__.py` (`from ._experimental import *`) will **not re-export private names**.  
- Without `__all__`, star-import could accidentally expose `_`-prefixed internals. Here it is safe.  

**File evidence:**  
`__init__.py` line ~70:  

```python
from ._experimental import *
```

- ✅ `__all__` exists in `_experimental.py`, so this is currently safe.  

---

### 2. Fix-G — Backwards compatibility

**Assessment:** **PASS**  
- Public symbols like `blockscaled_sparse_mla_qk_reduce_path_c` remain in `_experimental.py` and are imported via `__init__.py`.  
- Callers can still do:  

```python
from cppmega_mlx.nn._tilelang import blockscaled_sparse_mla_qk_reduce_path_c
```

**Caveats:**  
- Callers relying on the **old physical file location** (`_tilelang/blockscaled_sparse_mla_qk_reduce_path_c.py`) now see a logical indirection via `_experimental.py`.  
- No LegacyAlias or deprecation note was added; if tooling inspects file paths, it could trip, but runtime import works.  

**File evidence:** `_experimental.py` exports:  

```python
"blockscaled_sparse_mla_qk_reduce_path_c",
```

---

### 3. Fix-G — Separation of concerns

**Assessment:** **PASS**  
- `sparse_mla_fp8_path_c.py` symbols remain **not imported** in `__init__.py`, consistent with Fix-2 routing policy: only PROBE/REDUCERS surface goes through `_experimental`.  
- Moving them would break the separation; leaving them outside `_experimental.py` is consistent.  

**File evidence:** `_experimental.py` contains only the 20 moved PROBE/REDUCERS symbols; `sparse_mla_fp8_path_c` not touched.  

---

### 4. Fix-F — `__all__` completeness

**Assessment:** **CAUTION**  
**File:** `bench_tilelang_fp8_path_c.py` (file_000000000e04720aaf6c614fa99a4ab3) line ~10:  

```python
__all__ = [
    "BenchStats",
    "DEFAULT_PARITY_MAX_ABS",
    "DEFAULT_PARITY_MAX_REL",
    ...
    "main"
]
```

- The 15 names include all **functions** used by tests and constants: BenchStats, default parity thresholds, TILELANG/TVM roots.  
- **Potential missing exports:**  
  - `SCHEMA_VERSION` — appears defined in module but not in `__all__`.  
  - Paired-bench dataclasses (PairedBenchResult) — **included**.  
- ✅ Other constants and labels included.  

So `SCHEMA_VERSION` may need inclusion to match the test and public API expectations.  

---

### 5. Cross-impact — `__init__.py` __all__

**Assessment:** **CAUTION**  
- `__init__.py` does **not declare its own `__all__`**; it relies on star-imports (`from ._experimental import *`) plus top-level definitions.  
- This works for runtime, but star-import chaining can be fragile: static analyzers or tooling that introspects `__all__` may **miss top-level stable functions** like `sparse_mla_apply`.  

**File evidence:** `__init__.py` ~lines 70–90: no `__all__` declared.

- Recommendation: define explicit `__all__` in `__init__.py` including:  
  - top-level stable functions (sparse_mla_apply, m2rnn_apply, etc.)  
  - all `_experimental` symbols if intended to be public  

---

### ✅ New design concerns introduced by Fix-F + Fix-G

1. **Star-import chaining fragility:** relying on `_experimental.__all__` and top-level `__init__.py` exports may confuse static analysis or IDE autocomplete.  
2. **Backwards path inspection:** users or tooling expecting old file paths may be misled; no deprecation notice added.  
3. **__all__ incompleteness in bench_tilelang_fp8_path_c.py:** `SCHEMA_VERSION` not exported; may break some test or API expectations.  
4. **Implicit exposure of future private symbols:** any new symbol added to `_experimental.py` not in `__all__` remains private; developers must maintain this discipline.  

---

### Overall Verdict

**PASS with CAUTION**  

- Runtime behavior is safe and consistent with Fix-2 design.  
- Minor gaps: explicit `__all__` in `__init__.py` recommended, export `SCHEMA_VERSION` in bench_tilelang_fp8_path_c.py.  
- Star-import chaining is convenient but fragile; documentation or linter guidance should caution maintainers.  

---

Design summary word count: 291  

