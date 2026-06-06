# Tri-Dao mamba_ssm bwd through OUR tilelang->tvm stack: route + MMA proof + remaining parity blocker

MEASURED on NVIDIA GB10 (sm_121a, Blackwell). All GPU exec under the gb10 mutex.
tilelang @ `merge/upstream-codegen-reorg` HEAD `838b83b3` (FIX#1 `46b8115d` + FIX#2 `838b83b3`).
venv `/home/dave/cppmega-venv`, source `/home/dave/source/tilelang`.

## GO / NO-GO: NO-GO for end-to-end numeric parity (HONEST)

- ROUTING + MMA codegen: **GO** — all 7/7 Tri-Dao bwd kernels compile end-to-end
  tilelang->tvm to real TF32 tensor-core MMA, on the production shape.
- OUTPUT grid-scaling (FIX#1): **GO at IR level** — dprev output buffer is sized to the
  FULL grid extent (proven below), not a single tile.
- End-to-end numeric PARITY (routed-vs-native, all elements 1e-3): **NO-GO** — blocked by a
  REMAINING frontend defect in INPUT read-buffer grid-scaling (root-caused below with a
  reproducing experiment). NO parity PASS is claimed. RULE #1: no strides=0 / single-tile
  number is reported as a result.

## 1. Routing: 7/7 to real HMMA TF32 MMA  (MEASURED)

`route_all7.py` (compile each captured TTIR via `from_ttir(..., _allow_text_ttir=True)`):

```
ROUTE _chunk_scan_bwd_dstates            len= 33776 global=True mma=2 atomicAdd=0
ROUTE _chunk_scan_bwd_dc                 len= 34395 global=True mma=2 atomicAdd=0
ROUTE _chunk_scan_bwd_dcb                len= 48488 global=True mma=2 atomicAdd=0
ROUTE _chunk_scan_bwd_dx                 len= 55251 global=True mma=2 atomicAdd=1
ROUTE _chunk_state_bwd_db                len= 38231 global=True mma=2 atomicAdd=0
ROUTE _chunk_state_bwd_dx                len= 49304 global=True mma=2 atomicAdd=3
ROUTE _chunk_state_bwd_ddAcs_stable      len= 38991 global=True mma=2 atomicAdd=1
TOTAL_OK_WITH_MMA=7/7
```

CUDA carries `tl::mma_sync<kTensorFloat32,...,16,8,8,...>` (m16n8k8 TF32) +
`#include <tl_templates/cuda/instruction/mma.h>`, `extern "C" __global__ __launch_bounds__(128,1)`.

## 2. SASS: real tensor-core MMA + atomics on the production-shape cubin  (MEASURED)

The cubin is shape-agnostic (all dims/strides are runtime kernel args; the same cubin serves any
grid), built for `.target sm_121a` / `.elftype @"ET_EXEC"`. `export_sass` then grep:

```
288 HMMA.1688.F32.TF32        # total across all 7 kernels (real TF32 m16n8k8 tensor-core)
per-kernel HMMA: dstates 32, dc 32, dcb 32, dx 32, db 32, ddAcs 64, state_dx 64
reduce kernels also emit: REDG.E.ADD.F32.FTZ.RN.STRONG.GPU   (real global atomic-add, 64 in dx)
```

This is OUR codegen emitting genuine Blackwell tensor-core MMA on the production shape.

## 3. MEASURE: native dstates on the REAL production config  (MEASURED)

Config §P1: S=4096, c=64, g=8, H=112, P=64, N=64, bs=1.
`native_prod.py`, grid=(1,64,112), dprev_numel=29,360,128:

```
NATIVE_PROD_MS_PER_KERNEL = 1.20584 ms   (200-iter timed avg)
dprev sum=-203350.219  nz=29360127/29360128 (full output)
```

Small representative config (b1 nh8 hd64 ds64 nc8 cs64): native = 0.01470 ms/kernel.

Note on the "~10ms-class": that figure is the FULL mamba_ssm backward (all 7 kernels + the
non-Triton ops), not one kernel. The single dstates kernel native cost is ~1.2 ms at production
shape. The routed kernel ms cannot be reported until parity is fixed (see §5); reporting a routed
ms for a kernel that produces wrong/truncated output would violate RULE #1.

## 4. OUTPUT grid-scaling (FIX#1) — proven at IR level  (MEASURED)

Routed PrimFunc declared param shapes (`from_ttir` -> `tvm.ir.save_json`):

```
arg0 (dout)  = (2048 * gridDim_1 * gridDim_2 * gridDim_0 * gridDim_0_1,)
arg1 (C)     = (2048 * gridDim_1 * gridDim_2 * gridDim_0 * gridDim_0_1,)
arg2 (dprev) = (4096 * gridDim_1 * gridDim_2 * gridDim_0 * gridDim_0_1,)   <-- OUTPUT, full grid
arg3 (dA)    = (32   * gridDim_1 * gridDim_2 * gridDim_0 * gridDim_0_1,)
```

For the small config grid=(1,8,8): arg2 = 4096*1*1*8*8 = 262144 = exactly
b*nc*nh*hd*ds = 1*8*8*64*64. The single-tile `(4096,)` truncation is gone; the in-kernel store
guard is `idx < grid*4096` (matches the dense output). FIX#1 is correct for the OUTPUT.

## 5. REMAINING PARITY BLOCKER (root-caused, reproducing experiment) — RULE #1 RAISE

The end-to-end routed-vs-native parity FAILS. Honest root cause, proven by a controlled
experiment (`parity_tiny.py`, NOT a strides=0 shortcut):

- **Defect**: INPUT read buffers (arg0 dout, arg3 dA) are grid-scaled with the **per-K-tile-load
  footprint** (`flat_tile_extent`: dout 64x32=2048, dA 32) times the grid, in
  `poc/triton_frontend/__init__.py::_flat_extent_for_indices` (the
  `isinstance(entry, tir.Buffer/LazyTileExpr)` branch, ~line 218:
  `extent_expr = flat_tile_extent; for ... : extent_expr *= program_id_extent`).
- For a STRIDED block-pointer input read whose per-block base uses the **seqlen stride** and that
  is read across a **K-loop** (cs/BK trips, advancing by `stride*BK` each trip), the true reachable
  flat index is `(grid-1)*block_stride + chunk_footprint`, which **exceeds** `flat_tile_extent*grid`.
  i.e. the declared input extent UNDER-counts by ~chunk_size/BLOCK_K.
- The lowering then synthesises a matching in-kernel read guard
  `idx < gridDim_1*gridDim_2*gridDim_0_1*gridDim_0 * 2048` (and `*32` for dA). Valid strided reads
  beyond that bound are **masked to 0** -> the GEMM consumes zeros -> the routed output is
  near-empty (wrong), NOT truncated-by-one-tile.
- This is the symmetric twin of FIX#1: FIX#1 fixed the OUTPUT (whose per-program footprint IS one
  dense tile, so `grid*tile_numel` is exact); the INPUTS need the per-program **strided** footprint,
  which `grid*tile_numel` does not bound.

Reproducing experiment (degenerate config chosen so cs==BLOCK_K==32, a single K-trip, declared
extent == real extent: dout=8192=8192, dA=128=128):

```
=== PARITY DEGENERATE (cs=BK=32, single K-trip) ===
NATIVE nz=16384/16384 sum=-1240.0947
ROUTED nz=128/16384 sum=219.3126
MAXDIFF=1.023045e+02 ALLCLOSE_1e-3=False
FAIL
```

Even here the routed output is ~empty (128/16384 nonzero) because for chunk c>0 the dout per-block
base `c*cs*stride_seqlen` already reaches the `grid*2048` guard boundary and the upper reads mask
to 0. This isolates the bug to the input read-buffer extent / guard — NOT the MMA, NOT the output
store (both proven correct above).

### Why not papered over
A sound fix requires threading the per-block **stride span** of each input block-pointer (available
as PtrState/scalar-param stride Vars when the offset tile is materialized) into
`_flat_extent_for_indices`, so the input read buffer is declared to span
`(grid-1)*block_stride + per_program_footprint` (a symbolic extent over the stride Vars). The offset
tile reaching `_flat_extent_for_indices` is currently an OPAQUE buffer (`carry_index_223` etc.) with
no stride metadata attached, so the bound cannot be derived at that point today. Over-declaring with
a guessed constant could mask a real OOB or silently change results, which RULE #1 forbids. This is
reported as the exact remaining stage rather than shipped as a fabricated PASS.

## Environment gotcha (recorded for the next run)
`from_ttir` (libtriton MLIR parse) and `import torch` (its own static LLVM) **cannot coexist in one
process** — whichever loads second segfaults the interpreter. The parity harness therefore splits:
process A runs `from_ttir` and dumps the PrimFunc via `tvm.ir.save_json`; process B (`import torch`
first, then `import tilelang`, then `tvm.ir.load_json` + `tilelang.compile`) drives the kernel.
TVM-script round-trip is lossy for tilelang tile ops (`T.copy` not in the tirx parser) — use
`save_json`/`load_json`, which preserve the full IR (35 params verified).

## Files (absolute)
- `/Volumes/external/sources/tilelang/poc/triton_frontend/_test_harness/tridao_parity/route_all7.py`
- `/Volumes/external/sources/tilelang/poc/triton_frontend/_test_harness/tridao_parity/phaseA_native.py`
- `/Volumes/external/sources/tilelang/poc/triton_frontend/_test_harness/tridao_parity/native_prod.py`
- `/Volumes/external/sources/tilelang/poc/triton_frontend/_test_harness/tridao_parity/parity_stage2b.py`
- `/Volumes/external/sources/tilelang/poc/triton_frontend/_test_harness/tridao_parity/parity_tiny.py`
- `/Volumes/external/sources/tilelang/poc/triton_frontend/_test_harness/tridao_parity/emit_pf_json.py`
- `/Volumes/external/sources/tilelang/poc/triton_frontend/_test_harness/tridao_parity/sass_all7.py`
- Frontend defect site: `/Volumes/external/sources/tilelang/poc/triton_frontend/__init__.py`
  `_flat_extent_for_indices` (~line 218) + `_redecl_input_buffer` (~line 314).
