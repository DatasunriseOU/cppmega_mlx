# TileLang Metal pipelined 32x32 fragments

This artifact stacks on:

```bash
docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch
```

The earlier patch fixes the first Metal software-pipeline blocker by carrying
the leading `num_stages` buffer dimension into `MPSIntrinEmitter` access
pointers.  With that applied, 16x16 pipelined fragments lower, but 32x32
fragments can still fail when `StorageRewrite` rewrites a
`metal.simdgroup` buffer pointer element from scalar `float` to vector
`float32x4`.

`0001-metal-keep-simdgroup-storage-scalar-for-pipelined-32x32.patch` is the
second, C++ lowering fix.  It keeps `metal.simdgroup` storage scalar inside
`VectorTypeRewriter`, because Metal codegen lowers that scope to
`simdgroup_matrix<T, 8, 8>` and accepts scalar matrix element types only.

## Files

- `0001-metal-keep-simdgroup-storage-scalar-for-pipelined-32x32.patch`: upstream
  patch for `src/transform/storage_rewrite.cc`.
- `test_pipelined_32x32_probe.py`: runnable regression/probe for 32x32
  `T.Pipelined(..., num_stages=2)` Metal lowering.

## Apply

From the TileLang apple-head checkout:

```bash
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch
git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined_32x32/0001-metal-keep-simdgroup-storage-scalar-for-pipelined-32x32.patch
```

The patch is intentionally narrow and independent of Sparse-MLA scheduler
changes.  It only prevents an invalid vector pointer element rewrite for the
Metal simdgroup storage scope.

## Probe

Run from this repo, pointing Python at the TileLang checkout under test:

```bash
PYTHONPATH=/private/tmp/tilelang_apple_head/tilelang:/private/tmp/tilelang_apple_head/tilelang/3rdparty/tvm/python \
  ./.venv/bin/python -m pytest \
  docs/upstream/tilelang_metal_pipelined_32x32/test_pipelined_32x32_probe.py -q
```

The probe lowers two kernels to Metal:

- `gemm_32x32_pipe2`: minimal 32x32 FP16xFP16->FP32 GEMM with two-stage shared
  buffering and a 32x32 accumulator fragment.
- `sparse_mla_32x32_pipe2`: Sparse-MLA-shaped forward fragment using 32x32
  shared tiles, two pipelined K/V input stages, two score GEMMs, score staging,
  and a second GEMM into the value tile.

Both lowering tests assert that generated MSL contains simdgroup MMA, contains
threadgroup pipeline storage, does not contain the old
`simdgroup_matrix<float32x4, ...>` class of failure.  When the macOS Metal SDK
is present, the minimal 32x32 GEMM MSL also compiles with
`xcrun --sdk macosx metal -c`.

The same file can be run directly to print source metrics:

```bash
PYTHONPATH=/private/tmp/tilelang_apple_head/tilelang:/private/tmp/tilelang_apple_head/tilelang/3rdparty/tvm/python \
  ./.venv/bin/python docs/upstream/tilelang_metal_pipelined_32x32/test_pipelined_32x32_probe.py
```
