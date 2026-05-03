# Metal pipelined-shared-buffer 3D-region fix

## Blocker

`T.Pipelined(..., num_stages=N)` with `N > 1` lowers fine on the CUDA target
but fails to lower on the Metal target with:

    IndexError: Buffer A_shared is 3-dimensional (shape=[2, 32, 32]),
    but 2 index(es) were provided: (row_idx, col_idx).
    Please provide exactly 3 index/indices or slice(s).

## Root cause

`tilelang/src/transform/inject_pipeline.cc::RewriteAllocBuffer` prepends a
"version" dimension of size `num_stages` to every shared buffer participating
in a multi-stage pipeline (this is how double / triple buffering is realised).
A 2D shared buffer `[M, N]` becomes a 3D buffer `[num_stages, M, N]`, and
`RewritePipelineBufferRegion` inserts the per-iteration version index at
`region[0]`.

The CUDA macro generators (`mma_macro_generator.py`,
`wmma_macro_generator.py`, etc.) handle this correctly: they collect the
leading region dims via `[r.min for r in region[:-2]]` and pass them as
prefix indices into `T.access_ptr`.

The Metal macro generator
(`tilelang/intrinsics/metal_macro_generator.py::MPSIntrinEmitter`) was
written for the 2D-only case. Its `_parse_buffer_2d` helper only extracted
`(row, col)` offsets and the `ldmatrix_a` / `ldmatrix_b` / `simdgroup_copy`
methods called `T.access_ptr(buffer[row_idx, col_idx], ...)` — which trips
TileLang's `Buffer.__getitem__` arity check.

## Fix

Single-file Python edit (`tilelang/intrinsics/metal_macro_generator.py`):

1. `_parse_buffer_2d` returns an extra `leading_indices` tuple containing
   the `.min` of every region dimension preceding the last two.
2. `ldmatrix_a` / `ldmatrix_b` / `simdgroup_copy` propagate
   `leading + (row, col)` into their `T.access_ptr(buffer[...])` call.

The change mirrors the existing CUDA pattern in `mma_macro_generator.py`
(see `A_other = [r.min for r in A_region.region[:-2]]` at line 265) and
keeps the 2D-only fast path identical: when `region` has length 2,
`leading == ()` and `(leading + (row, col)) == (row, col)`.

## Diff stat

    tilelang/intrinsics/metal_macro_generator.py  | 29 ++++++++++++++++++-----
    1 file changed, 22 insertions(+), 7 deletions(-)

## Test results

After the patch, `test_pipelined_probe.py` (in this directory):

| Kernel | Status |
|---|---|
| `k_pipe_2` (num_stages=2 + 16x16 fragment) | OK |
| `k_pipe_3` (num_stages=3 + 16x16 fragment) | OK |
| `k_attn` (pipelined Q*K^T) | OK |

`docs/upstream/test_sparse_mla_pipeline.py`:

| Kernel | Pre-patch | Post-patch |
|---|---|---|
| `k1_simple_gemm` | OK | OK |
| `k2_pipelined_gemm` (32x32 fragment) | IndexError 3D buffer | float32x4 (separate baseline issue) |
| `k3_multi_gemm` | "Unknown storage scope `metal.simdgroup`" | OK |

The IndexError specifically caused by pipeline 3D buffer expansion is gone in
all probes. `k2_pipelined_gemm` with the original 32x32 fragment still fails,
but on a *different*, pre-existing baseline bug:
`StorageRewrite::PointerValueTypeRewrite` vectorises a `metal.simdgroup`
buffer's element dtype to `float32x4` when the kernel's output `T.copy`
emits a `T.Cast("float16x4", ramp_load_4)` pattern. Codegen then ICHECK-fails
at `codegen_metal.cc:454`. This is reproducible **without** pipelining (just
use an explicit `for ko in range(...)` K-loop) and is unrelated to this fix.
A clean upstream fix needs the codegen to either reject the ramp-load on
metal.simdgroup buffers earlier (T.copy fallback to `simdgroup_store`) or
handle vectorised simdgroup access via lane shuffle. That work is out of scope
for this patch.

## Apply

Cherry-pick onto `tilelang` checkout:

    cd /tmp/tilelang_apple_head/tilelang
    git apply ../docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch

The change is Python-only — no rebuild required.

## Upstream-PR readiness

Ready. The fix:

* Mirrors the existing CUDA implementation pattern (1:1 with
  `mma_macro_generator.py`).
* Has zero behaviour change for 2D buffers (the pre-patch hot path).
* Adds documentation explaining *why* leading dims must be threaded
  through, citing the `inject_pipeline.cc::RewritePipelineBufferRegion`
  call site.
* Is testable via the included `test_pipelined_probe.py` — three
  pipelined kernels at varying num_stages, all expected to lower.

Recommend filing as a PR on `apache/tilelang` with the title
"fix(metal): propagate pipelined version index in MPSIntrinEmitter access_ptr".
