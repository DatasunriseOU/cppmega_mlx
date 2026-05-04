# Metal FP8 scaled-matmul fused-scale macro prototype

## Status

Local upstream artifact for Path C patch **B**.

This is a macro-level prototype against the real current TileLang surface. After
the `tilelang_metal_fp8_scaled_matmul` prereq, `T.fp8_scaled_matmul(...)` lives
in `tilelang/language/fp8_op.py` as two `@T.macro` bodies:

- `_fp8_scaled_matmul_macro`
- `_fp8_scaled_matmul_macro_trans_b`

The earlier draft targeted a nonexistent `tilelang/tileop/gemm` scheduler stack.
This replacement does not add `GemmMetalFP8ScaledScheduler`, `GemmSchedule`, or a
fake Metal GEMM dispatch hook.

## What Patch B Does

- Patches only `tilelang/language/fp8_op.py`.
- Keeps scale selection inside the contracted-K loop.
- Rewrites both macro bodies from `a_val * b_val * sa * sb` into explicit
  `a_scaled = a_val * sa`, `b_scaled = b_val * sb`, then
  `C_local += a_scaled * b_scaled`.
- Covers the `transpose_B=True` macro that maps to the current M == 1 Path C
  vecmat shape.
- Avoids a post-loop `C *= scale` epilogue.

This is intentionally not CUDA/H200 acceptance and not a final performance
scheduler. The packed `uint32` FP8 loads, LUT-backed dot4 decode, and `simd_sum`
row reducer still live in cppmega.mlx's local MLX/Metal runtime body and remain
the reference shape for a future real TileLang scheduler.

## Patch

- `0001-metal-fuse-fp8-scaled-matmul-scheduler.patch`

The patch stacks after:

- PR #2130 (`jorgecurious/tilelang:metal-gemm-upstream-rebase`)
- `tilelang_metal_fp8`
- `tilelang_metal_fp8_gemm`
- `tilelang_metal_fp8_scaled_matmul`
- optional `tilelang_metal_blockscaled_e8m0`, if the block-scale prereq is part
  of the local replay

## Local Probe

```bash
./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/test_fp8_scaled_matmul_fused_scheduler_probe.py \
  -q
```

The local probe is source-level by default. It verifies that the artifact targets
`tilelang/language/fp8_op.py`, references both real macro names, keeps scale
multiplication inside each K loop, and no longer references the impossible
`tilelang/tileop/gemm` scheduler classes.

To run the real apply check, point `TILELANG_CHECKOUT` at a TileLang checkout
with the prereq patches already applied:

```bash
TILELANG_CHECKOUT=/path/to/tilelang \
  ./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/test_fp8_scaled_matmul_fused_scheduler_probe.py \
  -q
```

When `TILELANG_CHECKOUT` is unset, the `git apply --check` test is skipped.

Current validation recipe used for the local disposable apply check:

```bash
rm -rf /tmp/tilelang-path-b-apply
git clone /Volumes/external/sources/cppmega.backup/.tmp/tilelang-build \
  /tmp/tilelang-path-b-apply
git -C /tmp/tilelang-path-b-apply apply \
  /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_scaled_matmul/0001-tilelang-fp8-scaled-matmul-intrinsic.patch
git -C /tmp/tilelang-path-b-apply apply --check \
  /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/0001-metal-fuse-fp8-scaled-matmul-scheduler.patch
TILELANG_CHECKOUT=/tmp/tilelang-path-b-apply \
  ./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/test_fp8_scaled_matmul_fused_scheduler_probe.py \
  -q
```

The expected result is that the prereq applies cleanly, `git apply --check` on
patch B exits 0, and the env-gated probe reports all tests passing instead of a
source-only skip.

## Remaining Risk

This artifact is an upstream patch, not a vendored TileLang subtree. Runtime
acceptance still requires applying it to the current TileLang stack after
prereqs and running the relevant Apple Silicon Metal checks.
