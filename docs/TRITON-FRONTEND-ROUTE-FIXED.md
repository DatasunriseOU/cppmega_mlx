# Tri-Dao mamba_ssm bwd Triton kernels through tilelang -> tvm on gb10 (STEP 3 route+measure)

Date: 2026-06-06. Host: NVIDIA GB10 (sm_121), CUDA 13.2, driver 595.71.05.
venv: `/home/dave/cppmega-venv` loading tilelang+tvm from SOURCE `/home/dave/source/tilelang`
(commit `b5e4fed3`). triton 3.7.0, mamba_ssm 2.2.x.

## Honest GO / NO-GO

PARTIAL GO on COMPILE; **NO-GO on PARITY** (output is single-tile truncated — see bug below).

- 3/7 group-A bwd kernels COMPILE END-TO-END through OUR frontend
  (`from_ttir`/`compile_ttir`) -> tilelang lower -> **TVM CUDA codegen** to a real
  `__global__` tensor-core kernel. NOT a stub, NOT native-triton fallback.
- The routed `_chunk_scan_bwd_dstates` kernel **LAUNCHES on gb10** (executes, returns).
- BUT the routed output is WRONG: only one program tile (4096 = hdim*dstate = 64*64) of
  the full `dprev_states` tensor is written. Parity vs native triton FAILS. Root cause is
  a real frontend output-buffer size-inference defect (below). Per RULE#1 this is reported
  loudly, not papered over.

## Evidence the lowered PrimFunc is NON-EMPTY (real tt.dot body)

`from_ttir(_chunk_scan_bwd_dstates, _allow_text_ttir=True, target=cuda)`:
- PrimFunc length: **29448 chars**; `T.evaluate(0)` count: **0**; has `T.gemm(...)`: YES; has loops: YES.
- Real body contains: `T.gemm(T.region(dot_a_shared...), T.region(tile_load_305...), T.region(dot_c_frag...))`,
  masked `tile_load_*[i]=T.if_then_else(mask, arg[idx], 0.0)`, `T.exp(...)` for dA_cumsum,
  full int64 index arithmetic with overflow guards.

This comes from the NATIVE-PARSE provider (`triton_native_parse.py`, Provider 0), which uses
the already-installed `triton._C.libtriton.ir.parse_mlir_module` to validate + canonically
re-print the TTIR and adapt it for the existing OP_TABLE walker. NO C++ shim, NO mlir.ir/IREE
build needed (the C++ PtrAnalysis shim is unavailable in-process because libtriton's static LLVM
is already loaded; the frontend warns and uses the MVP scalar load/store path).

## Evidence of REAL tilelang->tvm CUDA codegen (real HMMA from OUR codegen)

`compile_ttir(...).get_kernel_source()` for `_chunk_scan_bwd_dstates_kernel` (33586 chars):
```
#include <tl_templates/cuda/instruction/mma.h>
...
tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32,
             tl::DataType::kFloat32, 16, 8, 8, false, true>(
    reinterpret_cast<float*>(dot_c_frag_308 + ...),
    reinterpret_cast<const uint32_t*>(A_local + ...),
    reinterpret_cast<const uint32_t*>(B_local + ...));
```
`tl::mma_sync<...kTensorFloat32...16,8,8...>` = m16n8k8 TF32 HMMA emitted by OUR tilelang
CUDA template (`tl_templates/cuda/instruction/mma.h`), driven by TVM codegen. This is OUR
codegen, NOT the triton runtime, NOT a stub.

3/7 compile to this form: each `mma_sync`x2, `__global__`x2:
- `_chunk_scan_bwd_dstates` (33586 chars), `_chunk_scan_bwd_dc` (34128), `_chunk_scan_bwd_dcb` (47676).

## N / 7 status (group-A first)

| kernel | TTIR capture | tilelang->tvm CUDA (real MMA) | launches on gb10 | correct output |
|---|---|---|---|---|
| _chunk_scan_bwd_dstates (A) | YES | YES (mma=2) | YES | NO (1-tile truncation) |
| _chunk_scan_bwd_dc (A)      | YES | YES (mma=2) | not run | NO (same bug) |
| _chunk_scan_bwd_dcb (A)     | YES | YES (mma=2) | not run | NO (same bug) |
| _chunk_state_bwd_db (A)     | YES | FAIL: `variables (arg4,) are used but not passed in` (tilelang var-binding) | - | - |
| _chunk_scan_bwd_dx (B)      | YES | FAIL: `m_warp*n_warp==num_warps` num_warps=0 | - | - |
| _chunk_state_bwd_dx (B)     | YES | FAIL: `Can't cast a handle to other types` | - | - |
| _chunk_state_bwd_ddAcs_stable (B) | YES | FAIL: num_warps=0 | - | - |

**3/7 compile to real tensor-core MMA CUDA through OUR stack. 1/3 confirmed to launch.
0/7 confirmed correct (output truncation bug blocks parity).**

## MEASURED (gb10, representative SMALL config)

Config: batch=1, nheads=8, headdim=64, dstate=64, nchunks=8, chunk_size=64, ngroups=1,
seqlen=512, BLOCK_M=BLOCK_N=64, BLOCK_K=32, HAS_SEQ_IDX=False.

- NATIVE triton `_chunk_scan_bwd_dstates` (inner JITFunction, autotune stripped, BLOCKS pinned),
  per-kernel wall time: **0.0167 ms** (this small config is tiny; the manifest's ~10ms-class
  figure is for the full production-size workload, not this representative slice).
- ROUTED tilelang->tvm kernel, per-launch wall time: **8.19 ms** — dominated by Python/FFI
  per-call overhead + the MVP SCALAR load/store path (C++ PtrAnalysis shim is disabled
  in-process), NOT a tensor-core-bound number. This is the launch of a kernel that only
  writes ONE tile, so it is NOT a fair perf comparison and MUST NOT be read as "the routed
  kernel is 8ms".
- path_c v1 905ms: not a meaningful comparison at this stage (different scope; routed kernel
  is per-kernel and output-incorrect).

## Parity vs native (1e-3, all elements): FAIL

Routed output `out_buf`: 32 / 4096 elements nonzero; native fills the full
`(batch, nchunks, nheads, headdim, dstate) = 262144`-element tensor. allclose(1e-3) = False.
The routed kernel cannot write past one 64x64 tile.

## ROOT CAUSE (real frontend defect — reported loudly, NOT patched here)

The output buffer `dprev_states_ptr` is size-inferred by the walker as a SINGLE program tile
`T.Buffer((4096,))` (4096 = hdim*dstate), whereas the INPUT buffers are correctly grid-scaled
`(2048 * gridDim_0*gridDim_0_1*gridDim_0_2*gridDim_0_3,)`. Two consequences in the generated CUDA:
1. A spurious store guard `if (idx < 4096) arg2[idx] = ...` (mma.cu line ~607) clamps every
   write whose per-block base offset (`blockIdx.x*arg22 + ...`) lands beyond the first tile —
   so all blocks except the first tile are silently dropped.
2. The undersized `[4096]` output param vs the real 262144-element tensor causes an FFI/launch
   segfault when a correctly-sized output buffer is passed; passing exactly 4096 avoids the
   crash but yields the truncated result above.

Native triton stores with mask `(offs_m < hdim) & (offs_n < dstate)` ONLY and a per-block base
pointer offset — it has NO `< 4096` whole-buffer clamp. The clamp is purely an artifact of OUR
output-buffer size inference treating a masked store's decl_buffer tile-extent as the whole-buffer
extent. Fix belongs in the frontend output-buffer shape inference (grid-scale the store target
like the load sources), then re-verify parity. This is a substantive frontend change, out of
scope for this measure step, and is left UNPATCHED rather than worked around.

## Reproduce

- Capture TTIR (pre-tvm, isolates libtriton): `/tmp/ttir7/*.ttir` (7 kernels).
- PrimFunc dump: `/tmp/verify_route.py <name>`.
- Compile + CUDA dump: `/tmp/run_route.py <name>` -> `/tmp/route_cuda/<name>_compiled.cu`.
- Native ref (isolated proc): `/tmp/phaseA_native.py` -> `/tmp/dstates_io.pt`.
- Routed launch + measure (isolated proc): `/tmp/phaseB2.py`.

NOTE: native-triton and the routed tvm path CANNOT coexist in one process (both link LLVM
statically; loading both -> duplicate cl::opt -> segfault). All measurement is two-phase:
native in one subprocess (saves tensors), routed in a fresh subprocess.
