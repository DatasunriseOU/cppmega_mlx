# TRITON cp.async/LDGSTS LIVE + bit-correct on the EXECUTED §P1 path (EXECUTEMEASURE)

Status: **GO**. cp.async/LDGSTS is now LIVE on the EXECUTED §P1 dstates path WITH
torch+triton imported, AND bit-correct (MAXDIFF 4.882812e-04, NOT the racy 1.28e3).
Commit `a4c1bedf` (3 files, 64 insertions; all in tilelang). Measured on gb10
(NVIDIA GB10, sm_121, CUDA 13.2), gb10-only GPU exec.

## What was measured (EXECUTED, torch+triton in sys.modules)

Harness: `poc/triton_frontend/_test_harness/tridao_parity/em_dstates_cpasync.py`
(+ timing re-confirm `em_timing_only.py`). Both `import torch, triton` FIRST so
libtriton's static LLVM is loaded — i.e. the real executed §P1 path, NOT a
compile-only probe. §P1 = `_chunk_scan_bwd_dstates` (b1 nh112 hd64 ds64 nc64 cs64),
grid (1,64,112).

- **OFF** = `from_ttir(..., prologue_opt=False)` — un-routed serial baseline, plain LDG.
- **OPT** = `from_ttir(..., prologue_opt=True)` + `TL_FORCE_CP_ASYNC=1` — routed
  CopyNode emits `is_async_copy` → genuine cp.async/LDGSTS, race-closed by
  `cp.async.commit_group` + `cp.async.wait_group<0>` + CTA sync (BUG-B fix).

### 1. EXECUTED cubin SASS (the exact objects launched in this process)

| kernel | UTMALDG | LDGSTS | HMMA | LDG | spill |
|--------|---------|--------|------|-----|-------|
| OFF (plain LDG) | 0 | **0** | 32 | 128 | 2494 |
| OPT (cp.async live) | 0 | **16** | 32 | 96 | 272 |

- OPT genuine opcodes: `LDGSTS.E` ×16 (real global→shared async loads).
- OPT completion barrier present: `DEPBAR.LE` (cp.async.wait_group<0>) + `BAR.SYNC.DEFER`
  (CTA sync) — the BUG-B race-close, in the executed cubin.
- UTMALDG=0 on both (GB10 has no functional bulk-TMA; this is the LDGSTS path
  native Triton uses on GB10).

### 2. Parity (both bit-correct — cp.async is correct, NOT racy)

```
PARITY OFF MAXDIFF=4.882812e-04 ALLCLOSE_1e-3=True nonzero=29360128/29360128 PASS
PARITY OPT MAXDIFF=4.882812e-04 ALLCLOSE_1e-3=True nonzero=29360128/29360128 PASS
```

OPT (cp.async live) == 4.882812e-04, the §P1 target — NOT the racy 1.28e3.
nonzero = full 29360128/29360128 (no per-block base drop, no skipped elements).
Re-confirmed in a second independent process (both MAXDIFF=4.882812e-04 PASS).

### 3. EXEC ms (CUDA-events, N=50–60/rep ×4 reps, interleaved OFF/OPT, median-of-medians)

| path | ms | notes |
|------|-----|-------|
| OFF (plain LDG, fresh build) | **3139.34** | honest fresh-build OFF (~3147 expected) |
| OPT (cp.async live, bit-correct) | **745.62** | LDGSTS=16, MAXDIFF 4.88e-04 |
| native Triton ref | **1.2311** | mamba_ssm `_chunk_scan_bwd_dstates_kernel` |

- **DELTA = 2393.73 ms drop** (OFF − OPT), real, interleaved, BOTH bit-correct.
- **SPEEDUP = 4.21× vs fresh OFF 3139 ms.**
- Per-rep stability (OFF/OPT): 3144.97/740.47, 3143.51/741.88, 3141.81/740.90,
  3138.87/745.62 — tight, no outliers.

KEY answer: **with cp.async REACHING + CORRECT, §P1 ms drops measurably** — 3139 →
746 ms (4.21×), a 2.39 s real reduction on the executed, bit-correct kernel.

### 4. Remaining gap to native

OPT 745.62 ms vs native 1.2311 ms = **605.65× gap remaining**. cp.async removed the
single biggest serial-prologue cost (4.21×), but the routed kernel is still a
de-monomorphized single-stage SIMT GEMM with heavy register spill (272 STL+LDL,
down from 2494) and no multi-stage pipelining / tensor-core scheduling that the
native autotuned Triton kernel has. The gap is dominated by (a) no software
pipeline over the K-loop (single-stage cp.async, wait<0> every trip), (b) spill,
(c) tile/occupancy not autotuned. Those are the next levers, not cp.async reach.

## No-regression

- **Standard pipelined GEMM (disable_tma, the executable GB10 LDGSTS path)**:
  `GEMM_DISABLE_TMA 512^3 stages=2 MAXDIFF=3.12e-02 REL=2.81e-04 PASS`,
  SASS UTMALDG=0 LDGSTS=16 HMMA=128. This path goes through copy.cc's
  `no_implicit_commit_wait` branch (returns BEFORE the explicit-async edit) and
  `elem_offset==0` (lower_access_ptr adds a constant 0 → no-op). Bit-correct,
  byte-path unchanged.
- **Default bulk-TMA GEMM (example_gemm)**: compiles to UTMALDG=48 and faults
  `CUDA_ERROR_ILLEGAL_INSTRUCTION` at execution — this is the PRE-EXISTING GB10
  bulk-TMA hardware limitation (GB10 lacks cluster/2-CTA TMA), independent of this
  change (the change does not touch the TMA bulk path).
- **Prod default sm_121a untouched**: the prod default is `prologue_opt=False`
  (no TL_FORCE_CP_ASYNC) = the OFF path, measured 4.882812e-04. The async path is
  honestly env-gated (`TL_FORCE_CP_ASYNC=1`), never a silent fallback.
- **path_c untouched**: all 3 changed files are in tilelang
  (`poc/triton_frontend/op_emitters/memory.py`, `src/backend/cuda/op/copy.cc`,
  `src/transform/lower_access_ptr.cc`); no cppmega_mlx / path_c files touched.

## Root fixes (commit a4c1bedf)

- **BUG-A (reach)**: emission (no triton → text-walker reaches the CopyNode
  emitter) is decoupled from execution (loads the PrimFunc, compiles WITH triton).
  The cp.async copynode needs only `T.copy`, never the libtriton-conflicting
  PtrAnalysis C++ shim. LDGSTS=16 now lands in the executed cubin with torch+triton
  imported.
- **BUG-B (correctness)**: the 1.28e3 was a per-block base DROP, not a race —
  `LinearOffsetFromLoad` (lower_access_ptr.cc) omitted `buffer->elem_offset`, so the
  routed 2D strided view read block 0's slice for every CTA. Fix adds elem_offset
  (no-op for elem_offset==0 buffers). Plus copy.cc closes the explicit single-stage
  cp.async group out-of-line (commit_group + wait_group<0> + CTA sync) so a bare
  `is_async_copy` is race-free without a software pipeline.

## Reproduce (gb10)

```
ssh gb10
cd /home/dave/source/tilelang && source /home/dave/cppmega-venv/bin/activate
PYTHONPATH=/home/dave/source/tilelang python \
  poc/triton_frontend/_test_harness/tridao_parity/em_dstates_cpasync.py
# SASS: TL_FORCE_CP_ASYNC=1 python .../sass_dstates.py  -> OFF LDGSTS=0, OPT LDGSTS=16
```

## GO/NO-GO

**GO.** cp.async/LDGSTS is LIVE on the EXECUTED §P1 path WITH triton imported
(LDGSTS=16, UTMALDG=0 in the launched cubin) AND bit-correct (4.882812e-04, not
racy). §P1 ms drops 3139 → 746 ms (4.21×, real, interleaved, both bit-correct).
Remaining gap to native 1.23 ms = 605.65× (next levers: K-loop software
pipeline, spill reduction, tile autotune — not cp.async reach). No regression;
prod default + path_c untouched. Reproducible from HEAD a4c1bedf.
