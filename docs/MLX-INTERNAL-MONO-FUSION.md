# MLX-Internal Mono-Fusion of the SSD Chunk-Scan — Measured Evidence

Date: 2026-06-05
Author: David Gornshtein (davidgornshtein@gmail.com)
Artifact: `/Volumes/external/sources/mlx/python/mlx/ssd_fused.py`
MLX commit (Mac/Metal tree): `c9a393906` (branch `upstream-integration`)
MLX-CUDA build (gb10): `mlx 0.32.0.dev20260602+6680d75a`, editable in `cppmega-venv`, sm_121

## Goal

Make MLX emit ONE fused kernel body for the SSD chunk-scan forward
(F0 precompute + F1 inter-chunk recurrence + F2 scan/combine), replacing the
6-kernel path (F0/F1/F2 fwd + B0/B1/B2 bwd via tilelang) that round-trips state
through global memory with inter-kernel sync.

## HONEST ATTRIBUTION — custom-primitive, NOT auto-fusion

MLX's `mx.compile` auto-fusion is **elementwise-only**. In `mlx/compile.cpp`
the `is_fusable` predicate is a typeid allowlist of unary/binary/ternary/
broadcast ops; a `Matmul`/`Reduce`/`Scan` node is a HARD fusion boundary, and
`Compiled::eval_gpu` only emits a per-element contiguous/strided loop (no shared
memory, no tensor cores, no carried sequential state). Therefore **MLX cannot
auto-fuse a stateful tensor-core chunk-scan into one kernel body.**

The reachable mono-fusion is a **custom-primitive**: author the whole SSD chunk
region as ONE user kernel body and register it as ONE MLX op via the existing
escape hatch — `mx.fast.cuda_kernel` (CUDA JIT, sm_121 mma.sync-capable) /
`mx.fast.metal_kernel` (MSL threadgroup memory). One `CustomKernel` primitive =>
ONE launch => ONE graph node; chunk state stays resident in shared/threadgroup
memory across the chunk axis, never round-tripped through global memory.

This is **distinct from cudaGraph launch-batching** (MLX-CUDA `device.cpp`),
which orders the 6 separate kernels without merging their bodies. Here the body
is genuinely one kernel.

Label surfaced by the module: `auto_fusion_status()` returns
"custom-primitive (one mx.fast.{cuda,metal}_kernel op, mono kernel body, state
resident in shared mem across chunks) — NOT mx.compile auto-fusion".

## ONE-KERNEL PROOF (graph export, both backends)

`mx.export_to_dot(path, Y, final_state)` on `ssd_chunk_scan_fused(...)` yields a
graph with exactly ONE primitive node and zero GEMM/Scan nodes:

```
digraph {
  { rank=source; "A".."G"; }        // 7 inputs: x,B,C,A,dt,D,h0
  { 42529702104 [label ="CustomKernel", shape=rectangle]; }
  "A".."G" -> 42529702104           // ONE node
  42529702104 -> "H"                // Y
  42529702104 -> "I"                // final_state
}
```

- Metal (Mac): node labels = `['CustomKernel']`, CustomKernel=1, Matmul/GEMM=0, Scan=0
- CUDA (gb10): node labels = `['CustomKernel']`, CustomKernel=1, Matmul=0

=> MLX now emits **ONE fused kernel body** for the chunk (vs 6 separate kernels).
Confirmed independently on both backends.

## PARITY (vs pure-MLX serial per-timestep diagonal recurrence)

Reference = elementwise serial recurrence in fp32 (no GEMM tricks), the
unambiguous ground truth.

| backend | dtype | Y max\|abs\| | final_state max\|abs\| | gate | verdict |
|---------|-------|------------|----------------------|------|---------|
| Metal   | fp32  | 1.609e-06  | 6.855e-07            | 1e-5 | **PASS** |
| Metal   | fp16  | 1.796e-03  | 7.153e-07            | 5e-4 | ULP-limited* |
| CUDA    | fp32  | 5.364e-07  | 2.682e-07            | 1e-5 | **PASS** |
| CUDA    | fp16  | 1.796e-03  | 3.576e-07            | 5e-4 | ULP-limited* |

\* The fp16 absolute Y error is below **one fp16 ULP at the output magnitude**:
measured max|Y| = 5.011, fp16 ULP there = 3.906e-03 > the 1.796e-03 observed
error. fp16 **max RELATIVE error = 4.83e-4**, mean abs = 8.39e-5. The kernel
computes everything in fp32 internally and only casts the final Y to fp16, so
the residual is pure output-quantization, not a kernel bug. The fp32 path (state
and accumulators always fp32) passes the strict 1e-5 gate by >6x margin on both
backends. **Numerically correct.**

## TIMING — fused 1-kernel vs 6-kernel-style baseline

Shape: batch=4, seqlen=2048, nheads=8, headdim=64, dstate=16, chunk=64, fp16.
50 warm iters, full `mx.eval`. The "6-kernel-style baseline" reproduces the
structural F0/F1/F2 decomposition as separate MLX matmul/einsum ops that
round-trip through global memory (the structural analogue of the multi-kernel
chain).

| backend | fused 1-kernel | 6-kernel-style | fused vs 6-kernel |
|---------|---------------:|---------------:|------------------:|
| **CUDA (gb10, sm_121)** | **2.328 ms** | 7.312 ms | **3.14x FASTER** |
| Metal (Apple) | 14.885 ms | 4.887 ms | 0.328x (3x slower) |

## GO / NO-GO

- **CUDA / sm_121 (gb10): GO.** The mono-fused custom primitive is **3.14x
  faster** than the 6-kernel-style decomposition. Eliminating the global-memory
  round-trips and inter-kernel sync, with state resident in `__shared__` across
  the chunk axis, wins decisively even with a scalar contraction body (no
  mma.sync yet). This is the recipe the GOAL calls for.

- **Metal / Apple: NO-GO (this round).** The mono-fused body is **0.328x**
  (3x slower) than MLX's vectorized GEMM baseline. ROOT CAUSE: the kernel runs
  one threadgroup per (batch,head) with only `headdim=64` threads and a fully
  **scalar** inner contraction (the `cb`, `C·state`, and state-update loops are
  per-element). MLX's Metal `matmul` baseline uses tuned simdgroup GEMMs that
  hugely outperform the scalar loop at this size. The fusion is structurally
  correct (1 kernel, resident state) but the BODY needs simdgroup-matrix
  (`simdgroup_float8x8`) contractions to be competitive on Metal. Same fix the
  CUDA body needs for its own next step (mma.sync m16n8k16).

## tok/s EFFECT (honest scope)

The fused op is **default OFF** behind `MLX_SSD_FUSED`; the existing 6-kernel
path in `cppmega.mlx` path_c stays byte-identical unless opted in. Wiring this
into the full training step and measuring end-to-end tok/s was **NOT executed
this round** (the op replaces forward F0/F1/F2 only; B0/B1/B2 backward is still
the 6-kernel path, so a step-level tok/s number would not isolate this change).

What IS measured: on gb10 the fused forward chunk is **3.14x** faster than the
decomposed forward. Backward is ~80% of the step, so a forward-only fusion caps
the step-level upside; the honest next lever is to extend the same mono-kernel
body to the backward (B0/B1/B2) so the whole SSD chunk — fwd+bwd — is ONE
resident-state kernel. Extrapolating the forward-only 3.14x to a step-level
tok/s gain would be dishonest and is explicitly NOT claimed here.

## NEXT (to close the Megatron gap)

1. Replace the scalar contractions in `_CUDA_SRC` with mma.sync m16n8k16
   fragments (cb, C·state, state-update) — sm_121 has regular tensor cores.
2. Replace `_METAL_SRC` scalar loops with `simdgroup_float8x8` — required for
   Metal to even break even.
3. Author the mono-fused BACKWARD body (B0/B1/B2) with the same resident state,
   so fwd+bwd is one kernel (backward is 80% of the step).
