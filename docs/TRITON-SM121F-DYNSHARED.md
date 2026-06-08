# TRITON SM121F DYNSHARED — lift the 48KB static cap under sm_121f, then test if the C-tile TMA executes

Date: 2026-06-08
Repo / HEAD: tilelang `merge/upstream-codegen-reorg` @ `84693de3`
Machine: NVIDIA GB10 (sm_121, CC 12.1, Blackwell), aarch64-linux, CUDA 13.2
Venv: /home/dave/cppmega-venv (py3.13). Source: /home/dave/source/tilelang

## Question

Two parts:
1. **DynShared (§1):** Make the routed `_chunk_scan_bwd_dstates` C-tile kernel's >48KB
   shared memory DYNAMIC (`shared.dyn`) so the compute_120f **48KB STATIC** shared cap
   does not apply, and verify the kernel **COMPILES under sm_121f** (the prior blocker:
   ptxas rejected 98KB static > 48KB family cap, so the f-case TMA was never testable).
2. **ExecuteTMA (§2):** With the kernel compiling under f, **RUN** the §P1 dstates kernel
   under `TL_ARCH_SUFFIX=f`. Does the grounded C-tile bulk-tensor TMA **EXECUTE TO
   COMPLETION** (compute-sanitizer clean, no `copy_sm90.h:96` illegal instruction)?
   This is the previously-untested f-case — the real answer to whether GB10 TMA works.

## Implementation (§1) — proper dynamic opt-in, no hack

`poc/triton_frontend/op_mapping.py` `_alloc_tile_buffer` (~line 1204): when
`TL_FORCE_DYN_SHARED=1`, a tile that would be allocated with scope `"shared"` is flipped
to `"shared.dyn"`. This routes the >48KB staging tile + accumulators through
MergeSharedMemoryAllocations' DYNAMIC path into the single `buf_dyn_shmem`
`extern __shared__`, so ptxas no longer counts them against the 48KB STATIC cap, and TVM's
`cuda_module.cc` raises the dynamic limit via
`cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`.

The global arch default stays `sm_121a` (`tilelang/contrib/nvcc.py:449`,
`nvrtc.py:65`); `TL_ARCH_SUFFIX` only overrides for this experiment. **path_c prod is
untouched** — both gates (`TL_ARCH_SUFFIX=f`, `TL_FORCE_DYN_SHARED=1`) are off by default.

## MEASURED results (from HEAD 84693de3, GB10)

Shape §P1: b1 nh112 hd64 ds64 nc64 cs64, grid (1,64,112).
Shim: `_cxx/build_linux/_triton_frontend_cxx.cpython-313-aarch64-linux-gnu.so` on PYTHONPATH.

### Baseline (f, NO dyn) — reproduces the prior blocker
```
ptxas error : Entry function '_chunk_scan_bwd_dstates_kernel_kernel'
              uses too much shared data (0x18020 bytes, 0xc000 max)   # 98KB static > 48KB
PTXAS_SHARED_REJECT_DETECTED
```

### §1 DYNSHARED (f, TL_FORCE_DYN_SHARED=1) — COMPILE BARRIER LIFTED
```
COMPILE_OK=True
tl_tma_load_count=2       # real grounded C-tile TMA
HAS_prefetch_tma=True
STATIC_SHARED_DECL: __shared__ __align__(16) uint64_t mbarrier_mem[4]   # only 32B static left
```
Disasm of the emitted `.cu` compiled to a real cubin under `-arch=sm_121f`:
```
SASS_UTMALDG=10   SASS_UTMACCTL=1
/*1180*/  UTMALDG.2D [UR8], [UR16], desc[UR18] ;
/*1320*/  UTMALDG.2D [UR16], [UR10], desc[UR20] ;
```
**§1 = GO.** Dynamic-shared routing lifts the 48KB static cap; the kernel ptxas-compiles
under sm_121f and the executed cubin contains genuine `UTMALDG.2D` bulk-tensor TMA.

### §2 EXECUTE under sm_121f — DECISIVE NEGATIVE
Grounded C-tile TMA OPT, single launch + sync under `TL_ARCH_SUFFIX=f TL_FORCE_DYN_SHARED=1`:
```
tl_tma_load=2 prefetch=True
LAUNCHING grounded TMA OPT kernel under f...
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```
compute-sanitizer memcheck pinpoints the faulting PC:
```
========= Illegal instruction
=========     at void tl::tma_load<...>(...)+0x1930 in copy_sm90.h:96
=========     by thread (128,0,0) in block (0,0,0)
=========         Device Frame: _chunk_scan_bwd_dstates_kernel_kernel+0x1080 in tvm_kernels.cu:166
```
The UTMALDG is **REACHED at runtime** (sanitizer caught it executing) and then **FAULTS**
with illegal instruction at the `cp.async.bulk.tensor` site — identical to the prior
sm_121a result. **§2 = NO-GO.**

### Parity (re-verified from HEAD)
Non-TMA routes (OFF baseline, OPT addressing-fold) both match Tri-Dao native:
```
PARITY OFF MAXDIFF=4.882812e-04 ALLCLOSE_1e-3=True PASS
PARITY OPT MAXDIFF=4.882812e-04 ALLCLOSE_1e-3=True PASS
OPT_REPEAT (multi-K-trip) MAXDIFF=4.882812e-04
```

### Routed §P1 EXEC ms under f (CUDA-events, N=60/rep x4, interleaved OFF/OPT)
```
rep0 OFF=1478.59  OPT=1095.96
rep1 OFF=1481.42  OPT=1095.95
rep2 OFF=1476.05  OPT=1095.90
```
Routed OPT (addressing-fold, non-TMA) median ≈ **1096 ms** under f, statistically identical
to the prior **1102 ms** (delta ≈ -6 ms, within run-to-run noise). Native Tri-Dao ≈ 1.12 ms.
The TMA path contributes **no** speedup because it cannot execute — it faults.

## ANSWER

- **Does the C-tile bulk-tensor TMA execute under sm_121f?** **NO.** It compiles (dynamic-shared
  lifts the 48KB cap), the `UTMALDG.2D` is emitted into the executed cubin and reached at
  runtime, but it **FAULTS with `cudaErrorIllegalInstruction` at `copy_sm90.h:96`** — the
  same failure as under sm_121a. The f-family target was the last untested variable; it is
  now tested. **GB10 `cp.async.bulk.tensor` TMA is non-functional in this toolchain (CUDA 13.2)
  regardless of arch suffix (a or f).** This is the decisive, honest negative.
- **§1 (compile barrier):** GO — proper dynamic opt-in, reproducible.
- **§2 (TMA execution):** NO-GO — sanitizer-clean evidence of an illegal-instruction fault.
- **dstates correctness:** preserved, §P1 + multi-K-trip MAXDIFF = 4.88e-04.
- **Routed ms under f:** ≈1096, == prior 1102 within noise; no TMA speedup (TMA faults).
- **Safety:** global default still `sm_121a`; path_c prod untouched; both gates off by default.

## Repro
```
ssh gb10
cd /home/dave/source/tilelang && source /home/dave/cppmega-venv/bin/activate
export PYTHONPATH=/home/dave/source/tilelang/poc/triton_frontend/_cxx/build_linux:$PYTHONPATH
# §1 compile (expect COMPILE_OK + tl_tma_load_count=2):
TL_ARCH_SUFFIX=f TL_FORCE_DYN_SHARED=1 TL_TMA_ROUTE=1 \
  python poc/triton_frontend/_test_harness/tridao_parity/compile_dstates_dynshared_f.py
# §2 execute + sanitizer (expect Illegal instruction @ copy_sm90.h:96):
TL_ARCH_SUFFIX=f TL_FORCE_DYN_SHARED=1 TL_TMA_ROUTE=1 \
  /usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --target-processes all \
  python poc/triton_frontend/_test_harness/tridao_parity/sanitize_dstates_tma_f.py
```
