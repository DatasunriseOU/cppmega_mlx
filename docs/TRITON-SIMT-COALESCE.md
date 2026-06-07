# TRITON-SIMT-COALESCE — iteration 7: SIMT-coalesced de-risk, the EXECUTED ms answer

**Date:** 2026-06-08
**Branch/HEAD:** `merge/upstream-codegen-reorg` @ `c2796a70`
**Target:** NVIDIA GB10, sm_121a (aarch64-linux), CUDA 13.x
**Kernel under test:** routed `_chunk_scan_bwd_dstates` at §P1
(`b1 nh112 hd64 ds64 nc64 cs64`, s=4096, grid `(1,64,112)`)

## The question (4 iterations could not answer)

Does routing the dstates K-loop `dout`/`C` global→shared `T.copy` as a **coalesced
load** measurably **drop §P1 ms**? Iterations 3–6 kept hitting TMA: iter-6 got the C
tile to lower to a real `UTMALDG` and the kernel LAUNCHED, but the **sm_90 TMA template
(`cp.async.bulk.tensor.2d`) FAULTS at runtime on sm_121** (compute-sanitizer: illegal
instruction at `tl::tma_load … copy_sm90.h:96`). So no executed ms ever came out.

## What iteration 7 did — SIDESTEP TMA, route SIMT

Set `disable_tma=True` on the routed C/dout CopyNodes so `Copy::Lower`
(`src/backend/cuda/op/copy.cc`, gated by `SelectCopyInstForLowering` on `!disable_tma`)
takes the **SIMT branch** (plain `LDG`), never `LowerBulk`/TMA. No tensorMap, no
`cp.async.bulk.tensor`, no sm_90 template — the sm_121 fault is gone.

### The lever (no hack): restore the per-Call annotations channel

In this fork `tirx.call_intrin` did `del annotations, kwargs` and the FFI `tirx::Call`
factory took only 4 args, so `T.copy(..., disable_tma=True)` **silently dropped** the
annotation and the copy was force-classified TMA-bulk. Fixed in `3rdparty/tvm`
(committed source, byte-identical on gb10):

- `python/tvm/tirx/op.py`: `del kwargs` only; `return Call(dtype, func_name, args, span, annotations=annotations)`
- `src/tirx/ir/expr.cc`: 5-arg `tirx::Call` ctor threads `annotations` → `CallNode::annotations`; absent ⇒ unchanged 4-arg path
- `python/tvm/tirx/expr.py`, `src/target/llvm/codegen_cpu.cc`: supporting

This is a *restore* of a real channel (the old `del annotations` was itself a silent
fallback), not a special-case. `disable_tma` now reaches `CopyNode`.

Route lever: `poc/triton_frontend/op_emitters/memory.py`
`_emit_ptrstate_tile_load_copynode` now emits `T.copy(src2d, tile_buf, disable_tma=True)`
unconditionally for the routed C/dout tiles.

## MEASURED FROM HEAD (c2796a70) on gb10 — all EXECUTED, runs-to-completion

Harness: committed `poc/triton_frontend/_test_harness/tridao_parity/measure_dstates_simt_coalesced.py`
(sha256 `4a63d05…`), N=60/rep × 4 reps, interleaved OFF/OPT, CUDA-event median-of-medians.

| metric | value |
|---|---|
| **compute-sanitizer memcheck** | **0 errors** — LAUNCHES + RUNS TO COMPLETION, no TMA fault |
| PARITY OFF | MAXDIFF **4.882812e-04** ALLCLOSE PASS |
| PARITY OPT | MAXDIFF **4.882812e-04** ALLCLOSE PASS (bit-identical to HEAD) |
| OPT cubin TMA | UTMALDG=0, no `tma_load`/`tensorMap`/`cp.async.bulk.tensor`/LDGSTS |
| **OPT_MS** (SIMT route) | **1101.97 ms** |
| OFF_MS (un-routed serial) | 3145.47 ms |
| speedup vs OFF | 2.85x |
| HMMA (GEMM cooperative) | **32** (intact) |
| native triton §P1 | **1.146 ms** (median N=100; min 1.139) |

### EXECUTED SASS of the sm_121a cubin (fresh cuobjdump -sass, this run)

```
LDG.E.CONSTANT   38     # scalar 32-bit global loads (the routed C/dout loads)
LDG.E.128         0     # NO vectorized coalesced global loads
LDGSTS / UTMALDG  0     # no cp.async, no TMA
STS.128 / LDS.128 324/108   # wide ops are SHARED-side only
STL.128 / LDL.128 168/136   # wide ops are LOCAL-side only
HMMA             32     # GEMM tensor-core path intact
STG.E             4
```

## THE COALESCING VERDICT (RULE #1 honest)

**Coalescing does NOT measurably drop §P1 ms — because the loads did not actually
become coalesced 128-bit loads.** The executed cubin shows **LDG.E.128 = 0 /
LDG.E.CONSTANT = 38**: the routed C/dout global loads stayed **scalar**. OPT_MS =
1101.97 ms == the prior *routed* best (~1102 ms); the SIMT route ran to completion and
TMA is fully sidestepped, but it delivered no coalescing speedup.

**Root cause:** the **swizzled shared-memory destination layout** breaks float4
vectorizability of the *global* load. The wide 128-bit traffic is all shared/local
(`STS.128`/`LDS.128`/`STL.128`/`LDL.128`) — the SIMT loop vectorizer cannot prove a
contiguous 128-bit window on the global side under the swizzle, so it emits scalar
`LDG.E.CONSTANT`. Contiguity of the ds/hd axis (stride 1) is necessary but not
sufficient: the swizzle on the destination defeats the source-side vectorizer.

The 2.85x vs OFF is the **addressing-fold / route restructuring** win (un-routed serial
→ routed), NOT a coalescing win. It is identical to the prior routed best with TMA
removed.

## Remaining gap to native (honest)

OPT **1101.97 ms** vs native **1.146 ms** ⇒ **~961x slower than native**. The 2.85x is
only vs the un-routed serial OFF baseline. Coalescing was the lever to start closing the
961x gap; it did not fire because the loads stayed scalar. The dominant cost is
elsewhere (per-element scalar global traffic + the routed serial structure), not the
load width alone.

## Separate blocker (not the coalescing question)

`async_loads=True` (pipelining the masked routed copy into cp.async) hits a distinct
`PipelinePlanning` overlapping-write conflict (CopyNode + mask epilogue both write the
tile). Out of scope here; the coalescing property is a load-width property independent of
pipelining and is answered above.

## GO / NO-GO

**NO-GO on "SIMT-coalesced loads drop §P1 ms" as a speedup lever — on the merits, with
the executed number that 4 iterations could not produce.** The kernel finally runs to
completion (sanitizer-clean, parity bit-correct), TMA is sidestepped, but the loads do
not vectorize to LDG.E.128 under the swizzled shared destination, so ms does not drop.
To actually realize coalescing the **shared-destination swizzle must be removed/aligned**
so the global load can be proven 128-bit-contiguous — that is the next lever, not more
TMA work.

## Reproduce from HEAD (c2796a70)

gb10 working tree byte-identical to committed sources (sha256-verified):
`memory.py` `a6c7218…`; tvm `expr.py/op.py/expr.cc/codegen_cpu.cc` all match `4418b2fe`.
`libtvm_compiler.so` newer than the touched `.cc` (fix compiled in).

```
ssh gb10 'cd /home/dave/source/tilelang && \
  /home/dave/cppmega-venv/bin/python \
  poc/triton_frontend/_test_harness/tridao_parity/measure_dstates_simt_coalesced.py'
# compute-sanitizer:
ssh gb10 'cd /home/dave/source/tilelang && /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck /home/dave/cppmega-venv/bin/python /tmp/ttir7/sanitizer_dstates.py'
# SASS:
ssh gb10 '/usr/local/cuda/bin/cuobjdump -sass \
  /tmp/tvm-debug-mode-tempdirs/<OPT-tempdir>/00000/tvm_kernels.cubin'
```

(`/tmp/ttir7/_chunk_scan_bwd_dstates.ttir` is the routed TTIR input.)
