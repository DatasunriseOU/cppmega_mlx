# Probe failure receipt — 2026-05-04

**Verifier**: agent run on 2026-05-04 against `jorgecurious/tilelang:metal-gemm-upstream-rebase` HEAD `971c17b6`.

**Outcome**: probe failed. PR was **not** filed.

## Source-level probe (PASS)

```
.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/test_fp8_scaled_matmul_fused_scheduler_probe.py -v
# 9 passed in 0.04s
```

All 9 source-level assertions pass: hunk-count self-consistency, README markers, declared MSL fingerprints, K-loop scale fusion shape, vecmat reducer shape, and Path-C runtime-body markers in `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py`.

## Apply-to-apple-head probe (FAIL)

Cloned `jorgecurious/tilelang:metal-gemm-upstream-rebase` (HEAD `971c17b6`), applied the 5 prereq patches that exist on disk in correct order:

1. `tilelang_metal_fp8/0001-tilelang-metal-fp8-storage-only.patch` — applies clean
2. `tilelang_metal_fp8_vector/0001-tilelang-metal-fp8-vector-cast.patch` — applies clean (after 1)
3. `tilelang_metal_fp8_gemm/0001-metal-fp8-gemm-software-path.patch` — applies clean
4. `tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch` — applies clean
5. `tilelang_metal_fp8_scaled_matmul/0001-tilelang-fp8-scaled-matmul-intrinsic.patch` — applies clean (creates `tilelang/language/fp8_op.py` as a `@T.macro`)

(`tilelang_gemm_mixed_dtype` conflicts with `tilelang_metal_fp8_gemm` on `__init__.py` and `metal_fragment_to_simdgroup.py`; setting it aside since it's not on the dependency path of patch B.)

Then attempting the patch B apply:

```
git apply 0001-metal-fuse-fp8-scaled-matmul-scheduler.patch
# error: corrupt patch at line 43
git apply --recount 0001-metal-fuse-fp8-scaled-matmul-scheduler.patch
# error: patch failed: tilelang/tileop/gemm/__init__.py:6
# error: tilelang/tileop/gemm/__init__.py: patch does not apply
# error: patch failed: tilelang/tileop/gemm/gemm_metal.py:28
# error: tilelang/tileop/gemm/gemm_metal.py: patch does not apply
# error: tilelang/tileop/gemm/gemm_schedule.py: No such file or directory
```

## Root causes

Two distinct problems, both blocking:

### 1. Architectural mismatch (primary)

The patch wants to:

- Add `from .gemm_metal_fp8 import GemmMetalFP8` and `from .gemm_metal_fp8_scaled import GemmMetalFP8ScaledScheduler` to `tilelang/tileop/gemm/__init__.py`.
- Insert an `if self.op_name == "fp8_scaled_matmul"` dispatch hook into a `GemmMetal` class in `tilelang/tileop/gemm/gemm_metal.py`.
- Add a `from_fp8_scaled_matmul(cls, op)` classmethod to a `GemmSchedule` class in `tilelang/tileop/gemm/gemm_schedule.py`.

None of those exist on the apple-head branch:

- `tilelang/tileop/gemm/gemm_metal_fp8.py` — does not exist (no prereq creates it).
- `GemmMetalFP8` — symbol does not exist.
- `tilelang/tileop/gemm/gemm_schedule.py` — file does not exist.
- `GemmSchedule` — class does not exist.
- `op_name == "fp8_scaled_matmul"` dispatch — `GemmMetal.lower(...)` on the apple-head branch does not look at any `op_name` attribute. The class lowers a `Gemm` op (`tl.Gemm`), not a `T.fp8_scaled_matmul` op.

The actual shape of `T.fp8_scaled_matmul` on the prereq stack is purely the `@T.macro` body in `tilelang/language/fp8_op.py` (created by PR #2142). It expands inline into a TIR loop — there is no tileop/gemm scheduler hook to extend.

### 2. Malformed hunk headers (secondary)

The patch's first two hunks have line counts that don't match their bodies:

```
@@ -6,6 +6,7 @@ from .gemm_mma import GemmMMA  # noqa: F401
 from .gemm_metal import GemmMetal  # noqa: F401
 from .gemm_metal_fp8 import GemmMetalFP8  # noqa: F401
+from .gemm_metal_fp8_scaled import GemmMetalFP8ScaledScheduler  # noqa: F401
 from .gemm_schedule import GemmSchedule  # noqa: F401
```

Header claims `-6,6 +6,7` (6 old context lines, 7 new) but only 3 context lines + 1 added line are shown → `git apply` reports `corrupt patch at line 43`. Same for the gemm_metal.py @@ -28,6 +28,7 @@ hunk. `git apply --recount` rescues the line-count mismatch but cannot rescue the missing context lines.

## Recommendation

The patch must be re-drafted. Two paths forward:

- **Option A (lighter)**: extend the existing `@T.macro` body in `tilelang/language/fp8_op.py` (PR #2142) to fuse the per-load scale broadcast directly into the K-loop accumulation. No tileop hierarchy work needed; this is just rewriting the `_fp8_scaled_matmul_macro` / `_fp8_scaled_matmul_macro_trans_b` body. Probably 30-60 LOC.
- **Option B (heavier, matches original intent)**: first land an upstream refactor that lifts `T.fp8_scaled_matmul` from a `@T.macro` into a registered `Gemm`-like op with a `tilelang/tileop/gemm/gemm_metal_fp8_scaled.py` scheduler, then stack the fused scheduler on top. This is closer to ~250 LOC and a separate prereq PR.

Option A is the smaller delta and matches the existing PR #2142 architecture; Option B is the more invasive but architecturally cleaner shape that the current draft assumes.

## Status

- Patch artifact: kept in place for reference (not deleted).
- Probe (`test_fp8_scaled_matmul_fused_scheduler_probe.py`): unchanged; still passes against the artifact.
- `_filed_prs_2026_05_04.md` row 13: marked **drafted, probe-failed**.
- `_path_c_blockers_tracker.md` patch B line: marked DRAFTED, PROBE-FAILED 2026-05-04.
