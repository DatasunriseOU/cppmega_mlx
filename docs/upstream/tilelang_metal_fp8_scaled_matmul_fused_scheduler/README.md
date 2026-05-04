# Metal FP8 scaled-matmul fused scheduler

## Status

Local upstream artifact for Path C patch **B**.

This patch replaces the older scalar `T.fp8_scaled_matmul` macro expansion with a
Metal scheduler path that fuses FP8 decode and scale application inside the
contracted-K loop. It is the TileLang-side counterpart to cppmega.mlx's current
Path C FP8 vecmat runtime body:

- packed `uint32` FP8 loads via `reinterpret_cast<device const uint*>`
- LUT-backed e4m3 dot4 decode
- `simd_sum` for the M=1 vecmat reduction
- per-tensor and per-row scale loads applied before the accumulation is stored

## Why This Exists

The historical `tilelang_metal_fp8_scaled_matmul` patch was a correctness-first
frontend macro. It made `T.fp8_scaled_matmul(...)` usable, but the generated
Metal path still left too much work outside the hot loop for the M=1/top-k Path
C sparse-MLA surfaces. cppmega.mlx now carries local runtime MSL that proves the
desired Apple Silicon shape: decode packed FP8 bytes, multiply by the selected
scale for the current K/block, reduce with a SIMD-group primitive, then store.

This upstream patch moves that shape into TileLang proper so downstream code no
longer needs a bespoke MSL replacement for the FP8 vecmat/`M == 1` case.

## Patch

- `0001-metal-fuse-fp8-scaled-matmul-scheduler.patch`

The patch stacks after:

- PR #2130 (`jorgecurious/tilelang:metal-gemm-upstream-rebase`)
- `tilelang_metal_fp8`
- `tilelang_metal_fp8_gemm`
- `tilelang_metal_fp8_scaled_matmul`
- optional Path C patch A, `tilelang_metal_pipelined_32x32`, for the wider
  Sparse-MLA 32x32 fragment route

## Local Probe

```bash
./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/test_fp8_scaled_matmul_fused_scheduler_probe.py \
  -q
```

The probe is intentionally source-level. It validates that the artifact contains
the concrete scheduler API and that the in-tree cppmega.mlx Path C runtime still
has the packed-load, LUT dot4, `simd_sum`, and per-row-scale markers this patch
is meant to upstream.

## Remaining Risk

This artifact is an upstream patch, not a vendored TileLang subtree. Runtime
acceptance still requires applying it to the apple-head TileLang checkout and
rerunning the strict FP8 Path C benchmarks on M4 Max.
