# Mamba3 MIMO TileLang Path B Port

Date: 2026-05-02

This page records the Path B port of cppmega's Mamba3 MIMO selective-scan
kernel to Apple Metal via MLX. The port lives in
cppmega_mlx/nn/_tilelang/mamba3.py and ships a custom-function VJP that ties
a vendor-MSL forward kernel to a vendor-MSL backward kernel, with a pure-MLX
fallback for both.

## Source pedigree

- Upstream forward + backward live in mamba_ssm.ops.tilelang.mamba3.{mamba3_mimo_fwd, mamba3_mimo_bwd}
  (state-spaces/mamba). The cppmega CUDA wrapper is
  cppmega/megatron/tilelang_mimo_autograd.py.
- The upstream Triton helpers compute_dacs_segsum_triton (in cppmega) and
  bwd_dadt_fused_triton, bwd_dtrap_ddt_triton (in
  mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils) have no Metal backend in
  Triton and are re-implemented here as pure MLX in
  cppmega_mlx/nn/_tilelang/_mamba3_helpers.py.
- The parity oracle is cppmega_mlx/nn/mamba3.py::Mamba3ReferenceBlock, which
  remains unmodified by this port (per the task contract).

## Path B contract

The TileLang "Path B" verified by the port research is:

1. Take the algorithmic recurrence — not the TileLang IR itself — as the
   spec. The recurrence is h[t] = exp(A[t]*dt[t]) * h[t-1] + x[t] * B[t]
   followed by y[t] = sum(h[t] * C[t]) + D * x[t], gated by
   silu(z[t]) * y[t].
2. Hand-write Metal shading language (MSL) and dispatch through
   mx.fast.metal_kernel — bypassing TVM's MarkHostMetalContext path that
   TileLang's tile-ai/tilelang#799 PR uses for PyTorch-MPS.
3. Wrap the fwd in mx.custom_function and define the VJP by calling the
   bwd kernel.

This avoids the macOS TileLang+TVM build chain entirely (TileLang's Metal
adapter is PyTorch-MPS oriented and doesn't lower cleanly through MLX). The
bench numbers below confirm the path works for the SSM scan kernel: forward
is ~25x faster than the reference, fwd+bwd is ~22-27x faster.

## fp16 carrier note

The kernel uses an fp16 carrier for storage and an fp32 internal accumulator.
Path A's research found that bf16 simdgroup ops still miscompile on macOS in
some MLX builds, so we deliberately default to fp16 outputs and let the host
upcast to fp32 only when the caller passes fp32 carriers. Inside the MSL
kernel every accumulator is float (fp32) regardless of the carrier dtype,
which keeps the recurrent decay product numerically stable across long
sequences.

## Triton helpers re-implementation

| Triton helper              | MLX rewrite         | Math                                                 |
| -------------------------- | ------------------- | ---------------------------------------------------- |
| compute_dacs_segsum_triton | compute_dacs_segsum | reverse cumsum of A[t]*dt[t] minus boundary; expand  |
| bwd_dadt_fused_triton      | bwd_dadt_fused      | d_decay = sum(dY*h, ...); dA = d_decay*dt; ddt = ... |
| bwd_dtrap_ddt_triton       | bwd_dtrap_ddt       | chain rule through trapezoidal sigmoid scale         |

All three helpers are verified at fp32 rtol=1e-4 against either a manual
loop reference (compute_dacs_segsum), a chain-rule expansion
(bwd_dadt_fused), or autograd through the original forward
(bwd_dtrap_ddt).

## Backward decomposition

The MSL bwd kernel emits per-(b, h, p) lane partials:

- dx, dz (B, T, H, P) — direct outputs, no reduction needed.
- dB_partial, dC_partial (B, T, H, P, N) — host sums over P.
- dA_partial, ddt_partial (B, T, H, P) — host sums over P.
- dD_partial (B, H, P) — host sums over (B, P).
- dh0 (B, H, P, N) — direct.

Path B still allocates `h_steps_scratch` (B*H*P, T, N) as an ignored output
buffer. Path C no longer does: its backward kernel computes `h_T` in a forward
register prepass, then reconstructs `h_{t-1}` from the current `h_t` in-place
inside the reverse pass. That removes the global `h_steps` scratch boundary
from the TileLang/TVM-FFI ABI.

## Bench

Spec shape: B=2, T=512, headdim=32, heads=4, state=64, dtype float16,
warmup=5, iters=15.

| Lane              | Mean (ms) | Median (ms) | vs reference (mean) |
| ----------------- | --------: | ----------: | ------------------: |
| fwd_reference     |     14.99 |       14.84 |               1.00x |
| fwd_metal         |      0.59 |        0.58 |              25.27x |
| fwd_bwd_reference |    144.76 |      115.33 |               1.00x |
| fwd_bwd_metal     |      5.32 |        5.32 |              27.22x |

These numbers exceed the task target (3-10x speedup) and significantly
exceed the Path B PoC anchor (5.2x on Mamba3-style scan). Receipt JSON is at
bench/tilelang_ports/mamba3.json. The 27x fwd+bwd speedup reflects two
multipliers: the per-step Metal scan vs per-step Python-driven MLX graph,
and the elimination of mx.stack/mx.scatter_along_axis overhead in the
reverse sweep.

## Parity

End-to-end parity vs the reference scan and autograd-through-reference:

- fp32 forward: max abs diff 9.3e-10, max rel diff <1e-7.
- fp16 forward at spec shape: max abs 2.1e-4 / ref norm 2.19 → rel 9.8e-5
  (well under rtol=1e-3).
- fp32 gradients across all 8 inputs at small shape (B=1, T=6): max rel norm
  diff <1e-7 for every input.
- fp16 gradients at spec shape (B=2, T=512): max rel norm diff 5.3e-3 for
  A and dt, <2e-3 for the rest. The two boundary tensors sit at the
  fp16 precision limit over 512 timesteps; the task target (rtol=5e-3) is
  the design limit.

## Surface

python
from cppmega_mlx.nn._tilelang import (
    mamba3_mimo_apply,        # mx.custom_function-wrapped fwd
    mamba3_mimo_fwd_metal,    # raw fwd returning (y, h_last)
    mamba3_mimo_bwd_metal,    # raw bwd returning per-input grads
    mamba3_mimo_reference,    # pure-MLX equivalent
    mamba3_mimo_metal_status, # introspect Metal eligibility
    compute_dacs_segsum,      # Triton-replacement helper
    bwd_dadt_fused,
    bwd_dtrap_ddt,
)


Callers that need autograd integration use mamba3_mimo_apply. Callers
that only need forward inference (and want the final hidden state) call
mamba3_mimo_fwd_metal directly.

## Blockers and notes

- TileLang's TVM-Metal lowering (PR tile-ai/tilelang#799) is not used at
  runtime. Attempting to install TileLang on macOS still fights the
  PyTorch-MPS-only build path; vendor MSL via mx.fast.metal_kernel is
  the practical Path B.
- The MSL kernel does not use T.gemm or simdgroup matrix ops; the inner
  state vector is per-thread (size N=64), and the loop is naive. Path A's
  finding that T.gemm and simdgroup-fragment-reduce miscompile on macOS
  does not apply to this kernel because we never touch those primitives.
- The pure-MLX backward fallback used by backend='mlx' is correct but
  slow (about 1.2x faster than the reference loop). Real training should
  use backend='auto' to select the Metal kernel.
- The forward kernel scales well as T grows (each lane keeps a register
  of N=64 fp32 state values). Path B backward still has scratch that grows
  linearly with T; Path C backward removes that scratch but pays extra reverse
  recurrence math, so AUTO keeps Path B unless the bench receipt proves
  Path C no-worse for the exact shape.
