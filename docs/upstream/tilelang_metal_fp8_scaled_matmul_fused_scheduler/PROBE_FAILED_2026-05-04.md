# Probe correction receipt - 2026-05-04

**Verifier**: local agent run in `/Volumes/external/sources/cppmega.mlx`.

**Outcome**: the old patch-B story is retired. The artifact is now a
documentation-only tombstone, and the probe asserts the real local MLX/Metal
Path C source markers instead of algebraic scaled-operands patch markers.

## What was wrong

The previous patch rewrote:

```text
a_val * b_val * sa * sb
```

into:

```text
(a_val * sa) * (b_val * sb)
```

That does not move scale after the accumulated dot, does not produce packed
`uint32` loads, does not use LUT-backed dot4 decode, does not create a 4-way K
unroll, and does not specialize M == 1 vecmat with `simd_sum`. It was not a
valid performance patch and should not be described as upstream TileLang
scheduler work.

## Current local contract

The actual local reference is
`cppmega_mlx/nn/_tilelang/fp8_vecmat_path_c.py`.

The probe now checks that the source contains:

- scale-after-dot output: `C[row] = sum * sx * sw`
- packed loads: `reinterpret_cast<device const uint*>`
- four FP8 bytes per packed word: byte lanes 0, 8, 16, and 24
- LUT dot4 decode: `fp8_e4m3fn_lut[...]`
- M == 1 vecmat specialization: one SIMD-group per output row
- Metal reduction: `sum = simd_sum(sum)`

## Upstream/apply status

`0001-metal-fuse-fp8-scaled-matmul-scheduler.patch` is intentionally not
applyable. It remains under the historical filename only so existing artifact
paths do not break. The probe skip-gates apply checks even when
`TILELANG_CHECKOUT` is set because there is no real upstream patch in this
directory.

This receipt makes no CUDA/H200 acceptance claim and no claim that a current
upstream TileLang tileop scheduler exists for this operation. A future real
upstream patch must replace the tombstone and add scheduler/codegen support for
the packed MLX/Metal hot-loop shape.
