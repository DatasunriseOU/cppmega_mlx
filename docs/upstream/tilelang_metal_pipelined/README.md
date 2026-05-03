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

## Performance / profiling

This patch changes Python-side Metal macro indexing only. It threads the
software-pipeline version index into `T.access_ptr`; it does not add an extra
runtime MSL operation beyond the double / triple buffering already requested by
`T.Pipelined(..., num_stages=N)`.

`test_pipelined_probe.py` now prints deterministic generated-source metrics and
asserts the key buffer sizes and simdgroup operation counts under pytest. On the
Apple-head Metal checkout used for this artifact:

| Kernel | Source lines | Source bytes | Threadgroup buffers | simdgroup_load | simdgroup_multiply_accumulate | simdgroup_store |
|---|---:|---:|---|---:|---:|---:|
| `k_pipe_2` | 56 | 3065 | `A_shared[1024]`, `B_shared[1024]` | 4 | 2 | 2 |
| `k_pipe_3` | 69 | 3942 | `A_shared[1536]`, `B_shared[1536]` | 6 | 3 | 2 |
| `k_attn` | 55 | 2758 | `K_shared[512]`, `Q_shared[256]` | 4 | 2 | 2 |

Generated MSL also compiles with:

    xcrun --sdk macosx metal -c <generated>.metal -o <generated>.air

A Torch/MPS launch smoke using `tilelang.compile(..., execution_backend="torch",
target="metal")` completed for `k_pipe_2`, `k_pipe_3`, and `k_attn`, but this is
not yet a PR-grade latency benchmark: the current Metal adapter logs a
non-fatal cache-save error (`MetalKernelAdapter` has no `libpath`). Keep runtime
latency checks manual until that adapter noise is fixed.

Manual lowering timing is available without making pytest flaky:

    TILELANG_PIPELINED_PROFILE=1 .venv/bin/python docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py

Manual Torch/MPS launch timing is also available and disables the TileLang disk
cache path to avoid the current Metal `libpath` cache-save noise:

    TILELANG_PIPELINED_RUNTIME_PROFILE=1 TILELANG_PIPELINED_RUNTIME_REPS=50 .venv/bin/python docs/upstream/tilelang_metal_pipelined/test_pipelined_probe.py

On the local MPS smoke run (`reps=50`, `warmups=10`, `rounds=3`), the observed
`tilelang.compile` wall times were about `257 ms`, `249 ms`, and `132 ms` for
`k_pipe_2`, `k_pipe_3`, and `k_attn`; launch medians were about `0.010 ms`,
`0.015 ms`, and `0.008 ms`. Treat those numbers as a local health check only,
not an upstream latency threshold.

Backed by the generated MSL, the only clear kernel-level tuning opportunity is
to choose `num_stages` deliberately: shared-memory use grows linearly with the
requested stage count (`1024` -> `1536` half elements per GEMM operand from
2-stage to 3-stage), while the generated MMA count grows with the loop body that
is already present. For the attention-shaped probe, only `K_shared` is
double-buffered and `Q_shared` stays single-loaded (`Q_shared[256]`), which is
the right source pattern for reusing Q across the K loop. No redundant
`simdgroup_multiply_accumulate` or `simdgroup_store` operations were introduced
by the 3D-buffer indexing fix.

## Apply

Cherry-pick onto the Apple-head / Metal-dev `tilelang` checkout that still has
`tilelang/intrinsics/metal_macro_generator.py`:

    cd /tmp/tilelang_apple_head/tilelang
    git apply /Volumes/external/sources/cppmega.mlx/docs/upstream/tilelang_metal_pipelined/0001-metal-pipeline-3d-buffer.patch

The change is Python-only — no rebuild required.

Public-main drift note: a fresh `tile-ai/tilelang` main checkout at `2eec5f0`
does not contain `tilelang/intrinsics/metal_macro_generator.py`, so this
artifact does not apply there as-is (`git apply --check` reports
`No such file or directory`). Refresh the patch against the branch that owns the
Metal macro emitter before opening a public upstream PR.

## Upstream-PR readiness

Ready on the Apple-head / Metal-dev branch. The fix:

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
