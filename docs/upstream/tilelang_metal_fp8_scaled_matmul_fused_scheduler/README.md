# Metal FP8 scaled matmul Path C scheduler story

## Status

Local artifact for Path C patch **B**, corrected on 2026-05-04.

This directory no longer carries an applyable upstream patch. The previous
patch-B artifact was a false performance story: rewriting the scalar multiply
into a scaled-operands multiply did not implement the local MLX/Metal fast path.
That algebraic scaled-operands form is retired and must not be used as evidence
for FP8 performance work.

The honest local story is narrower:

- `cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py` is the local MLX/Metal Path C
  reference.
- The dispatchable M == 1 vecmat runtime body applies scales after the
  accumulated dot: `C[row] = sum * sx * sw`.
- The hot loop uses packed `uint32` loads via
  `reinterpret_cast<device const uint*>`.
- Each packed word covers four FP8 K elements, so the loop is the local 4-way K
  unroll/stride shape.
- The dot uses `fp8_e4m3fn_lut[...]` lookups for all four bytes, matching the
  local packed uint32/LUT dot4 path.
- The vecmat specialization reduces one output row with `sum = simd_sum(sum)`.

This is explicitly **not CUDA/H200 acceptance**. It is a local Apple Silicon
MLX/Metal receipt and source contract. It also makes **no claim that an upstream
TileLang tileop scheduler exists** for `T.fp8_scaled_matmul`; a real upstream
patch still needs scheduler/codegen work that emits this Metal hot-loop shape.

## Artifact

- `0001-metal-fuse-fp8-scaled-matmul-scheduler.patch`

The retained filename is a tombstone for the retired patch. It is
documentation-only and intentionally not applyable. The probe skip-gates any
`git apply --check` attempt with a clear message.

## Local Probe

```bash
./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_fp8_scaled_matmul_fused_scheduler/test_fp8_scaled_matmul_fused_scheduler_probe.py \
  -q
```

The probe verifies:

- README and tombstone markers disclaim the bogus algebraic scaled-operands
  patch.
- The artifact does not include an applyable `diff --git` patch.
- The local Path C source contains scale-after-dot, 4-way K stride, packed
  `uint32` loads, LUT dot4 decode, and `simd_sum` vecmat specialization markers.
- The source/docs do not claim CUDA/H200 acceptance or an existing upstream
  TileLang tileop scheduler.

The `TILELANG_CHECKOUT` environment variable is deliberately ignored for this artifact because there is
no real upstream patch to apply. If a future real scheduler/codegen patch
replaces the tombstone, the probe should be updated to require
`git apply --check` against a checkout with the correct prereqs.

## Remaining Risk

The local Path C runtime body is a cppmega.mlx MLX/Metal implementation detail,
not an upstream TileLang scheduler. Upstream parity still requires a new
TileLang scheduler/codegen patch that lowers `T.fp8_scaled_matmul` or an
equivalent operation to the packed Metal hot loop described here.
