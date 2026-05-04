# T.fp8_scaled_matmul for TileLang Metal

## Status

This directory still carries the original upstream patch artifact:

- `0001-tilelang-fp8-scaled-matmul-intrinsic.patch`

That patch is the historical scalar macro baseline. It is useful provenance for
the `T.fp8_scaled_matmul(...)` API surface and tests, but it is no longer the
performance story used by cppmega.mlx Lane 6.

The current local TileLang apple-head tree used by
`scripts/bench_tilelang_fp8_path_c.py` has a Metal FP8 GEMM lowering at
`tilelang/tileop/gemm/gemm_metal_fp8.py`. Apple simdgroup MMA still cannot
consume FP8 operands directly, so the lowering keeps FP8 as storage: it decodes
the shared FP8 A/B tiles once into `threadgroup half`, synchronizes the
threadgroup, and reuses the existing FP16 `simdgroup_matrix` MMA sequence.

## Why This Matters

For the target Lane 6 shape, `128x128x128` e4m3 per-tensor scaled matmul, the
current Path C is no longer the old scalar fallback:

- The emitted MSL contains `simdgroup_multiply_accumulate=1`,
  `simdgroup_load=2`, `simdgroup_store=1`, and `threadgroup_half=2`.
- It does not use the Path B LUT table (`fp8_e4m3_lut=0`), because FP8 decode
  happens through TileLang/TVM FP8 cast helpers into half threadgroup tiles.
- It remains storage-only FP8, not native FP8 MMA. The MMA operands are half.

This route matches the feasible Metal option found in local TileLang lowering
research: there is no `simdgroup_matrix<float8>` route to tune, so the practical
knob is how efficiently TileLang stages FP8 storage into half tiles before the
existing FP16 MMA sequence.

## Current Receipts

Run command for the live Lane 6 receipt:

```bash
./.venv/bin/python scripts/bench_tilelang_fp8_path_c.py \
  --shapes matmul_128 \
  --warmup 3 \
  --iters 8 \
  --skip-sparse \
  --skip-xcrun \
  --out /tmp/fp8_path_c_matmul128_before.json
```

Result on Davids-Mac-Studio.local:

| Shape | Path B median ms | Path C median ms | Paired Path C / Path B | Parity vs Path B |
| --- | ---: | ---: | ---: | --- |
| `matmul_128` | 0.2068 | 0.1738 | 0.835x | max abs 0.0 / max rel 0.0 |

Checked-in receipts agree that Path C is not worse than Path B for this shape:

| Receipt | Path B median ms | Path C median ms | Paired Path C / Path B | Parity vs Path B |
| --- | ---: | ---: | ---: | --- |
| `bench/tilelang_ports/fp8_path_c_vs_path_b.json` | 0.1227 | 0.1076 | 0.890x | 0.0 / 0.0 |
| `bench/tilelang_ports/fp8_path_c.json` | 0.1521 | 0.1362 | 0.896x | 0.0 / 0.0 |

The strict bench gate requires:

- Path B and Path C rows both run successfully.
- Path C / Path B paired median ratio is `<= 1.0`.
- Path C vs Path B parity is within `1e-5` max abs and max rel for
  parity-enabled shapes.

## Local Verification

The focused test surface for this lane is:

```bash
./.venv/bin/python -m pytest \
  tests/test_tilelang_fp8_matmul_path_c_bench.py \
  tests/test_fp8_msl_kernels.py \
  -q
```

The bench smoke should be run with `--strict` for the exact target shape:

```bash
./.venv/bin/python scripts/bench_tilelang_fp8_path_c.py \
  --shapes matmul_128 \
  --warmup 3 \
  --iters 8 \
  --skip-sparse \
  --skip-xcrun \
  --strict \
  --out /tmp/fp8_path_c_matmul128_after.json
```

## Remaining Limitations

The current fast path is proven only for the receipted shape/scale layouts. Do
not generalize the `128^3` result to larger matmuls, per-row scale, block-scale,
or vecmat without fresh strict receipts.

The remaining performance blocker is not "native FP8 simdgroup MMA" exposure in
Metal. The available route is still FP8 storage decode into half followed by
FP16 MMA. Further wins need TileLang scheduling work around staging, tile sizes,
pipeline overlap, and scale indexing while preserving the compact MSL markers
above.

## Attribution

- `audiohacking/fp8-mps-metal` (commit
  d4fbd40c48aa2a243e600d06627c7dd818150636, MIT): original scalar scaled
  matmul algorithm and reference semantics.
- `AppMana/mps-fp8-for-torch-and-comfyui-python-package` (commit
  a902571eca5362f5e2496cf33dcce52c8bac6a15, Apache 2.0): LUT decode and
  integer-bit encoder used by cppmega.mlx Path B.
- Local TileLang apple-head `tilelang/tileop/gemm/gemm_metal_fp8.py`: current
  FP8-storage-to-half-staging plus FP16 simdgroup MMA lowering used by Path C
  receipts.
