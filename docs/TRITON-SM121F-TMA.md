# sm_121f ArchFix + C-tile TMA EXECUTE attempt on GB10 (Blackwell sm_121)

Date: 2026-06-08. Host: NVIDIA GB10 (CC 12.1, aarch64-linux), CUDA 13.2, NVRTC/nvcc 13.2.
tilelang HEAD: `a80e4fa8` (merge/upstream-codegen-reorg). All GPU exec on gb10 under mutex.

## Goal
After the ArchFix (CC 12.x -> sm_121f FAMILY arch, commit a80e4fa8), re-enable the
iter-6 routed C-tile TMA load and measure whether the coalesced TMA EXECUTES and
drops the dstates kernel §P1 ms vs the prior routed best (~1102 ms).

## VERDICT: NO-GO. The C-tile TMA still does NOT execute on GB10.

The diagnosis ("sm_121 fully supports cp.async.bulk.tensor TMA; sm_121f is the fix")
is FALSIFIED on this hardware by executed evidence:

1. The grounded C-tile TMA lowers to REAL `tl::tma_load` / `UTMALDG.2D` SASS
   (tma_load_count=2; SASS shows `UTMALDG.2D`, `UTMACCTL.PF`, `USETMAXREG`).
2. Under **sm_121f** (the ArchFix family target) the TMA kernel **fails to COMPILE**:
   `ptxas error: uses too much shared data (0x18020=98KB, 0xc000=48KB max)`.
   The `f` family target caps STATIC shared at 48KB; the double-buffered TMA tile
   plan needs ~98KB. It never reaches execution.
3. Under **sm_121a** (arch-specific; the documented alternative) the SAME kernel
   COMPILES (sm_121a permits the full Blackwell shared), but at runtime the
   executed `UTMALDG.2D` **FAULTS** with `cudaErrorIllegalInstruction`
   ("an illegal instruction was encountered") on the FIRST launch -- the same
   copy_sm90.h:96 class fault as iter-6. compute-sanitizer could not even flush a
   report: the illegal instruction hard-aborts the CUDA context/process.
   (OFF baseline parity passed 4.882812e-04 immediately before the OPT fault.)

So: sm_121f = won't compile (48KB cap); sm_121a = compiles but UTMALDG faults.
Neither makes the C-tile TMA execute. The coalescing-pays-off question is therefore
UNANSWERABLE from an executing TMA -- there is no executing TMA on GB10.

## The instruction
`copy_sm90.h:96` emits (CUDA >= 12.8 CTA variant):
`cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint`
-> compiles to `UTMALDG.2D ... desc[...]`. The descriptor is a by-value
`__grid_constant__ const CUtensorMap arg1_desc`. The executed UTMALDG traps illegal
on GB10 regardless of f/a suffix.

## ArchFix regression (sm_121f 48KB static-shared cap)
The ArchFix is NOT free: the family `f` target caps static shared at 48KB whereas
the prior `a` suffix permitted the full Blackwell shared. Any tilelang kernel
needing >48KB static shared that compiled under the pre-ArchFix sm_121a now FAILS
to compile under sm_121f. Observed directly: the routed dstates kernel's 80KB-shared
variant (`0x14000`) ptxas-fails under sm_121f / sm_120f / plain sm_121 but compiles
clean under sm_121a. This is a real global regression introduced by a80e4fa8.

## No-regression checks under sm_121f (clean committed HEAD)
- path_c F0 `chunk_precompute_fwd_cuda_prim` (the tensor-core GEMM): **PASS**.
  COMPILE ok; codegen `tl::mma_sync x4` + ldmatrix x4 (real mma.sync); per-call
  median 6.10 ms vs serial 63.76 ms = 10.4x; parity cb=1.22e-04 ss=3.74e-06
  dac=4.885e-04 (gate 5e-4) PASS. The production path_c GEMM executes correctly
  under sm_121f -- mma.sync HMMA is fine.
- Trivial SIMT vadd kernel: PASS (maxdiff 0) under sm_121f.
- Committed dstates SIMT route (HEAD default, no TMA): parity OFF/OPT 4.882812e-04
  PASS; OPT median ~1101-1102 ms (== prior routed best), OFF ~1478 ms. Unchanged.
- Standalone fp16 GEMM (128x128 tile, HMMA.16816.F32, NOT path_c): FAULTS illegal
  instruction at the HMMA (tvm_kernels.cu:45) under BOTH sm_121f AND sm_121a AND
  with disable_tma. This is config-specific (the path_c F0 mma.sync path works);
  reproduces at CLEAN committed HEAD -> a pre-existing GB10/HMMA-tile condition,
  NOT caused by the ArchFix. Flagged honestly; does not affect path_c.

## §P1 ms verdict
- routed EXEC §P1 ms now: ~1101-1102 ms (committed SIMT route, no TMA). Same as 1102.
- coalescing-pays-off: UNANSWERED. The TMA never executes, so no executed
  coalesced-TMA ms exists to compare. ms_delta vs 1102 = 0 (no change; TMA dead).
- gap to native 1.12 ms: still ~1.0x10^3x off; unchanged. The routed best stays
  banked-addressing-fold 1102 ms (native ~1.12 ms).

## Reproduce (gb10, under mutex)
Shim on PYTHONPATH so PtrAnalysis subprocess seeds ptr-state and grounding fires:
```
PYTHONPATH=poc/triton_frontend/_cxx/build:. \
  TL_ARCH_SUFFIX=a TL_TMA_ROUTE=1 \
  python poc/triton_frontend/_test_harness/tridao_parity/measure_dstates_tma_exec.py
# -> tma_load_count=2; OFF parity PASS; OPT launch -> illegal instruction
```
(TL_ARCH_SUFFIX / TL_TMA_ROUTE are experiment-only env gates added during this
investigation; the committed HEAD source is unmodified -- the TMA route is NOT the
committed default. The committed default is the SIMT route at ~1102 ms.)

## Recommendation
1. REVERT or re-scope ArchFix a80e4fa8: the sm_121f family target's 48KB static-shared
   cap is a real regression for >48KB kernels. If TMA is wanted, sm_121a (arch-specific)
   is required for the shared capacity -- but the UTMALDG still faults, so TMA is not
   usable on this GB10 regardless. Keep CC 12.x on sm_121a (the pre-ArchFix behavior)
   to preserve full shared for the SIMT/GEMM kernels that DO work.
2. The C-tile TMA is a dead end on this GB10 silicon/driver (UTMALDG illegal).
   Stay on the committed SIMT route (~1102 ms). The next real lever is 128-bit
   vectorization of the SIMT loads (the swizzled-shared-dest float4 blocker), NOT TMA.
