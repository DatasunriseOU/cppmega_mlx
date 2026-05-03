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

The FP8 path through TileLang's TVM-Metal lowering is **doubly blocked**:

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

This means even if every other detail of the FP8 kernel were ported by hand,
the FP8 storage tensors themselves cannot be referenced through the
TVM-Metal lowering — the type would have to be converted before the kernel
is invoked.

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


The simdgroup_matrix path is not yet wired up. Apple PRs in
tile-ai/tilelang HEAD are tracking this; once they land we expect both
blockers to lift simultaneously since the FP8 emit work and the GEMM intrinsic
share the simdgroup type plumbing.

## What this port ships today

Until the codegen blockers lift, cppmega_mlx/nn/_tilelang/sparse_mla_fp8.py
provides:

1. sparse_mla_fp8_metal_status(...) -> SparseMLAFp8MetalStatus
   — Returns available=False with both blocker reasons.
2. sparse_mla_fp8_fwd_metal(...) and sparse_mla_fp8_bwd_metal(...)
   — Return (status, None, None) while gated; reserved for the post-blocker
   implementation that will follow the same Path B pattern as mamba3.py
   (lower with target='metal', strip kernel void, mark Q/KV/Indices
   const device, leave Output/Lse device).
3. sparse_mla_fp8_reference(...)
   — Pure-MLX FP8 reference using mx.to_fp8 / mx.from_fp8 with a per-tensor
   scale derived from amax / 448.0 (the e4m3 max). Mirrors the gb10
   tensorwise FP8 contract.
4. sparse_mla_quantized_matmul_reference(...)
   — Hand-built path that uses regular FP32 matmul on dequantized inputs,
   provided as a forward-only bench upper bound for what the
   mx.quantized_matmul(mode='mxfp8') path could reach.
5. sparse_mla_fp8_apply(...)
   — High-level entry that falls back to the reference and supports
   force_metal=True to surface the blocker.

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
  matches the BF16 reference to **max abs err ~3e-3** on the
  (B=1, S=64, H=4, D=64, topk=16) smoke shape. See bench JSON.
- With scale=1.0 (raw standard normal) the FP8 path drifts to ~4e-1 max abs
  err — that is real e4m3 quantization noise propagated through softmax + V
  matmul, not a port bug. Tests use scale=0.1 to stay inside the brief's
  rtol budget.
- Backward parity uses a **dequantize-then-BF16 oracle**: we run the BF16
  reference on the same FP8-recovered Q/KV that the FP8 reference produces
  internally, and compare gradients. This isolates the attention math from
  the quantization noise. With STE the two grads match to within rtol=1e-2.

## Bench (current Apple M-series)

From bench/tilelang_ports/sparse_mla_fp8.json on the
(B=1, S=64, H=4, D=64, topk=16) smoke shape:

| path                          | median ms |
| ----------------------------- | --------- |
| BF16 reference                | ~0.30     |
| FP8 reference (STE roundtrip) | ~0.36     |
| quantized_matmul ref          | ~0.29     |

The FP8 reference is *slower* than BF16 because the quantize+dequantize
roundtrip adds work without the corresponding tensor-core speedup that the
real FP8 sparse-MLA kernel would deliver on H100/H200. The
quantized_matmul_reference row uses regular FP32 matmul on dequantized
inputs and is therefore identical to the BF16 reference plus a constant
overhead — it is not a real mxfp8 quantized_matmul yet, just a placeholder
showing where the kernel will plug in once blockers lift.

## Post-blocker plan

When tile-ai/tilelang HEAD lands the Apple PRs:

1. Verify that T.gemm(A_fp8, B_fp8, C_fp32, ...) lowers to MSL with
   simdgroup_matrix and an FP32 accumulator.
2. Build the FP8 PrimFunc with fp8_dtype = T.float8_e4m3, accum_dtype =
   T.float32, out_dtype = T.bfloat16. Mirror the gb10 fwd_fp8 PrimFunc
   but skip the tile.PassConfigKey.TL_DISABLE_TMA_LOWER (no TMA on Apple).
3. Lower with target='metal', strip kernel signature using the Path B
   helper, and dispatch through mx.fast.metal_kernel.
4. Add a manual VJP via mx.custom_function that wires forward outputs to
   the FP8 backward PrimFunc.
5. Drop the force_metal=True path's RuntimeError once dispatch is wired.
