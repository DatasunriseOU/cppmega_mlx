# TileLang E8M0 Block-Scale Layout Primitive

## Status

Local upstream artifact for Path C patch **C**.

This patch adds a TileLang DSL primitive for the MXFP8/E8M0 block-scale layout
used by the cppmega.mlx Sparse-MLA Path C QK reducer. It does not change local
runtime code in this repository; it is a PR-ready patch artifact for the
upstream TileLang tree.

## Why This Exists

The existing Path C FP8 scaled-matmul artifacts cover the FP8 storage path and
the fused Metal scheduler. The missing piece is a first-class way to say that
the scale operands are E8M0 bytes, laid out as one scale per 32 contracted-K
values:

- A scale shape: `(K / 32,)`
- B scale shape: `(N, K / 32)`, with broadcast `(K / 32,)` accepted for local
  probes
- scale index: `kb = k // 32`
- decode: `0` and `0xFF` map to zero; normal bytes decode as `2 ** (byte - 127)`

The patch exposes that contract as `T.BlockScaledLayout.e8m0_k32()` and
`T.e8m0_to_float(...)`, then lets `T.fp8_scaled_matmul(...)` accept either the
new layout object or the existing metadata spelling:

```python
T.fp8_scaled_matmul(
    A_fp8,
    A_scale,
    B_fp8,
    B_scale,
    C,
    transpose_B=True,
    block_scale_layout=T.BlockScaledLayout.e8m0_k32(),
)
```

Equivalent metadata:

```python
scale_format = "e8m0_block_k32"
scale_block_size = 32
```

## Patch

- `0001-tilelang-add-e8m0-blockscaled-layout-primitive.patch`

The patch stacks after:

- `tilelang_metal_fp8`
- `tilelang_metal_fp8_scaled_matmul`
- Path C patch **B**, `tilelang_metal_fp8_scaled_matmul_fused_scheduler`

## Local Probe

```bash
./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_blockscaled_e8m0/test_blockscaled_e8m0_probe.py \
  -q
```

The probe is source-level and focused on the artifact contract. It validates
that the patch carries the DSL surface, E8M0 decode semantics, K/32 shape rules,
contracted-K indexing, and a Mac MLX/Metal-only acceptance boundary. It also
rereads the current local Path C source to confirm the runtime evidence still
exposes the same constants and reducer markers.

## Remaining Risk

This directory is an upstream patch artifact, not a vendored TileLang checkout.
Runtime acceptance still requires applying the patch stack to the apple-head
TileLang tree and rerunning the strict Path C probes on Mac M4 Max with
MLX/Metal.
