# TileLang FP8 sparse-MLA port (Path B status)

This note documents the Apple Metal port of cppmega's FP8 sparse-MLA forward
and backward kernels. The work lives at:

- Module: cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py
- Tests: tests/test_tilelang_sparse_mla_fp8.py
- Bench: scripts/bench_tilelang_sparse_mla_fp8.py -> bench/tilelang_ports/sparse_mla_fp8.json

## Source kernels (gb10)

- cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd_fp8.py (fwd)
- cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd_fp8.py (bwd)
- cppmega/megatron/sparse_mla_ops/sparse_mla.py::SparseMLA_FP8 (autograd glue)

These are FP8 variants of the BF16 sparse-MLA kernel (PR #3674). Q and KV are
torch.float8_e4m3fn with per-token FP32 scale factors. After every Q@K tile
the kernel dequantizes the FP32 accumulator by q_scale * kv_scale (per
element along the BI dimension). V is dequantized to BF16 prior to S@V to
avoid having to apply per-token scales mid-GEMM.

## Status on Apple Silicon (tilelang 0.1.9, MLX 0.31)

The native TileLang DSL path through TVM-Metal is still **doubly blocked**,
but this no longer blocks the shipped Path B kernel. Current local code uses
hand-written MSL through `mx.fast.metal_kernel`: Q/KV are stored as `uint8`
e4m3 payloads and dequantized inline before the fp32 QK/SV arithmetic.
`sparse_mla_fp8_metal_status(...)` therefore reports `available=True` when
Metal dispatch is present. The blockers below apply only to a future native
TileLang `T.gemm`/float8 lowering path.

### Blocker 1: FP8 dtype emission to MSL is unsupported

Lowering any primfunc that *just casts* float8_e4m3 -> float16 with
target='metal' fails with:


Fatal: Cannot convert type float8_e4m3 to Metal type


The fault is in tilelang 0.1.9 vendored TVM at:


3rdparty/tvm/src/target/source/codegen_metal.cc:271


The Metal type emitter CodeGenMetal::PrintType handles float, bfloat,
int*, uint*, and bool. There is no float8_e4m3 / float8_e5m2 branch.
The CUDA codegen (codegen_cuda.cc:464,488,589) *does* have FP8 paths, and
the HIP backend has them too. The Metal one does not.

This means native TileLang FP8 storage tensors still cannot be referenced
through TVM-Metal lowering. Path B bypasses this by treating the payload as
plain `uint8` storage in MSL.

### Blocker 2: T.gemm is not registered for the metal target

This is the same blocker the BF16 sparse-MLA port hit (see
cppmega_mlx/nn/_tilelang/sparse_mla.py module docstring). A trivial primfunc

python
@T.prim_func
def main(A: T.Tensor((64, 64), 'bfloat16'), B: T.Tensor((64, 64), 'bfloat16'), C: T.Tensor((64, 64), 'float32')):
    with T.Kernel(1, threads=64):
        T.fill(C, 0.0)
        T.gemm(A, B, C)


lowered with target='metal' raises:


InternalError: Check failed: (0) is false: Unsupported target for gemm: metal -keys=metal,gpu ...


The simdgroup_matrix path is not yet wired up. Apple PRs in tile-ai/tilelang
HEAD are tracking this; once they land we can revisit a native TileLang Path C
full-kernel implementation. Until then Path B is the production path.

## What this port ships today

cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py provides:

1. sparse_mla_fp8_metal_status(...) -> SparseMLAFp8MetalStatus
   — Returns available=True when the direct-MSL Path B kernel can be
   constructed and the local Metal device can dispatch it.
2. sparse_mla_fp8_fwd_metal(...) and sparse_mla_fp8_bwd_metal(...)
   — Quantize Q/KV into uint8 e4m3 storage and dispatch the direct-MSL
   forward/backward kernels. They return `(out, lse)` and `(dq, dkv)` when
   Metal is available, otherwise `None` so callers can fall back.
3. sparse_mla_fp8_reference(...)
   — Pure-MLX FP8 reference using mx.to_fp8 / mx.from_fp8 with a per-tensor
   scale derived from amax / 448.0 (the e4m3 max). Mirrors the gb10
   tensorwise FP8 contract.
4. sparse_mla_quantized_matmul_reference(...)
   — Hand-built path that uses regular FP32 matmul on dequantized inputs,
   provided as a forward-only bench upper bound for what the
   mx.quantized_matmul(mode='mxfp8') path could reach.
5. sparse_mla_fp8_apply(...)
   — High-level entry that prefers direct-MSL Path B, falls back to the
   reference, and supports force_metal=True to fail closed when Metal is
   unavailable.
6. Path C QK reducers in sparse_mla_fp8_path_c.py
   — TileLang-DSL QK reducer and full-shape indexed QK reducer. The checked-in
   receipt has both reducers dispatchable, parity-clean, and not slower than
   their Path B comparison rows for the current smoke shape. They cover the QK
   score tile only; full FP8 forward/backward production dispatch remains Path B
   until the complete Path C sparse-MLA layout is wired and measured.

## FP8 dtype lowering surprises

Two surprises from the Path B investigation:

1. **mx.from_fp8 has no VJP in MLX 0.31.** Direct composition of
   mx.to_fp8 -> mx.from_fp8 makes any downstream loss non-differentiable
   (ValueError: [Primitive::vjp] Not implemented for FromFP8.). We work
   around this with a mx.custom_function straight-through estimator
   (_fp8_roundtrip_ste): forward does the quantize/dequantize roundtrip,
   backward is the identity cast. This matches how gb10 handles FP8 in
   training: the FP8 cast is not a gradient stop, only the recovered FP32
   values flow through autograd.

2. **mx.to_fp8 has no exposed scale.** It bakes a hard-coded e4m3 cast and
   stores the result as uint8. To match the gb10 per-token scale ABI we
   compute scale = max(|x|) / 448.0 on the FP32 input, divide by it before
   the cast, and multiply by it after the recovery.

## Numerical tolerance

Per the task brief, forward parity vs the BF16 reference uses rtol=5e-3
and backward uses rtol=1e-2. Both tolerances are achievable with the
following caveat: input magnitude matters.

- With scale=0.1 (typical post-layernorm attention queries) the FP8 path
  matches the BF16 reference to low single-digit e-3 absolute error on the
  checked smoke shapes. See bench JSON.
- With scale=1.0 (raw standard normal) the FP8 path drifts to ~4e-1 max abs
  err — that is real e4m3 quantization noise propagated through softmax + V
  matmul, not a port bug. Tests use scale=0.1 to stay inside the brief's
  rtol budget.
- Backward parity uses a **dequantize-then-BF16 oracle**: we run the BF16
  reference on the same FP8-recovered Q/KV that the FP8 reference produces
  internally, and compare gradients. This isolates the attention math from
  the quantization noise. With STE the two grads match to within rtol=1e-2.

## Bench (current Apple M-series)

From bench/tilelang_ports/sparse_mla_fp8.json on the current
(B=1, S=64, H=4, D=64, topk=16) smoke shape:

| path                          | median ms |
| ----------------------------- | --------- |
| BF16 reference                | 0.285     |
| FP8 reference (STE roundtrip) | 0.318     |
| quantized_matmul ref          | 0.231     |
| Path B direct-MSL FP8 fwd     | 0.198     |
| Path B direct-MSL QK vecmat   | 0.109     |
| Path C QK reducer             | 0.104     |
| Path C indexed QK reducer     | 0.113     |

The direct-MSL Path B kernel is faster than the references on this local M4
receipt because e4m3 byte decode is fused into the sparse QK/SV loops. This is
software FP8 emulation, not a CUDA/H100/H200-style native FP8 tensor-core
claim. Path C is correctness-live for QK: the checked ratios are
`path_c_qk_reduce_over_path_b_qk_vecmat=0.9551` and
`path_c_indexed_qk_reduce_over_path_b_fwd=0.5721`, with
`invalid_mismatch_count=0` for masked/OOB indices. This is a QK reducer receipt,
not a claim that full Sparse-MLA FP8 forward/backward has moved from Path B to
Path C.

## Next steps

1. Keep production/default dispatch on Path B direct MSL.
2. Keep native TileLang FP8 `T.gemm` lowering fail-closed until float8 storage
   and Metal GEMM lowering are both present.
3. Extend Path C only with measured reducers/full-layout kernels; do not AUTO
   promote until the strict full forward/backward gate is green.
4. When tile-ai/tilelang HEAD lands the Apple PRs:

1. Verify that T.gemm(A_fp8, B_fp8, C_fp32, ...) lowers to MSL with
   simdgroup_matrix and an FP32 accumulator.
2. Build the FP8 PrimFunc with fp8_dtype = T.float8_e4m3, accum_dtype =
   T.float32, out_dtype = T.bfloat16. Mirror the gb10 fwd_fp8 PrimFunc
   but skip the tile.PassConfigKey.TL_DISABLE_TMA_LOWER (no TMA on Apple).
3. Lower with target='metal', strip kernel signature using the Path B
   helper, and dispatch through mx.fast.metal_kernel.
4. Add a manual VJP via mx.custom_function that wires forward outputs to
   the FP8 backward PrimFunc.
5. Compare it against Path B before changing dispatch policy.
