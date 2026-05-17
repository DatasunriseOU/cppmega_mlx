# TileLang block-scaled (MXFP8) sparse-MLA port

This note documents the Apple Metal port of cppmega's MXFP8 block-scaled
sparse-MLA forward and backward kernels. The work lives at:

- Module: cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py
- Tests: tests/test_tilelang_sparse_mla_blockscaled.py
- Bench: scripts/bench_tilelang_sparse_mla_fp8.py -> bench/tilelang_ports/sparse_mla_blockscaled.json

## Source kernels (gb10)

- cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_fused.py
  - sparse_mla_blockscaled_mxfp8_fwd (forward)
  - sparse_mla_blockscaled_mxfp8_bwd_kernel (backward)
- cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_qk.py
  (separate experimental QK-only block-scaled scoring helper)

## ABI

The block-scaled kernel consumes pre-quantized MXFP8 payloads with the
following layout (from tilelang_sparse_mla_blockscaled_fused.py):


q_data:   [B, S,  H, D_total]       torch.float8_e4m3fn
kv_data:  [B, SK, G, D_total]       torch.float8_e4m3fn
q_scale:  [B, S,  H, D_total/32]    torch.float32
kv_scale: [B, SK, G, D_total/32]    torch.float32
indices:  [B, S,  G, topk]          torch.int32, -1 sentinel


MXFP8_BLOCK_SIZE = 32: one FP32 scale per 32-element block along the head
dim. The kernel walks the head dim in 32-element chunks, runs
T.gemm(q_block, kv_block, partial), then accumulates
partial * QScale[h, kb] * KVScale[j, kb] into acc_s before the
online-softmax + S@V flow.

## Status on Apple Silicon (tilelang 0.1.9, MLX 0.31)

Same native TileLang DSL blockers as the tensorwise FP8 path (see
docs/tilelang_ports/sparse_mla_fp8.md), and the former direct-MSL Path B runtime
has now been retired. Current local code keeps the MXFP8 helper/reference
surface and the prepared-buffer Path C E8M0 QK reducer. The high-level
float-carrier wrapper does not proxy through Path C because that would hide
quantization/unpacking staging behind a wrapper boundary.
`sparse_mla_blockscaled_metal_status(...)` therefore reports
`available=False` with the retired-Path-B reason. The blockers below apply to
future native TileLang full fwd/bwd lowering work.

1. **FP8 dtype emission to MSL is unsupported.** Failure at
   3rdparty/tvm/src/target/source/codegen_metal.cc:271:
   Cannot convert type float8_e4m3 to Metal type.
2. **T.gemm is not registered for the metal target.**
   Unsupported target for gemm: metal -keys=metal,gpu ....

Native TileLang block-scaled lowering additionally needs the per-block dequant
pattern `partial * QScale[h, kb] * KVScale[j, kb]` to lower correctly. Path B
bypasses this by decoding the E8M0 scales directly in MSL and multiplying them
inside the fp32 QK loop.

## What this port ships today

cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py provides:

1. sparse_mla_blockscaled_metal_status(...) -> SparseMLABlockScaledMetalStatus
   — Returns available=False with the retired direct-MSL Path B reason. Reports
   the block_size=32 constant in the status dataclass.
2. sparse_mla_blockscaled_fwd_metal(...) and
   sparse_mla_blockscaled_bwd_metal(...)
   — Preserve shape validation for old import sites, then return `None`. They
   no longer quantize, unpack, construct raw MSL, or dispatch kernels.
3. sparse_mla_blockscaled_reference(...)
   — Pure-MLX MXFP8 reference using mx.quantize(mode='mxfp8') /
   mx.dequantize(mode='mxfp8') with group_size=32. The MLX kernel returns
   packed uint32 data and uint8 scale shorthand: each scale slot is a
   power-of-two block exponent, and 4 fp8 e4m3 values pack into one uint32.
   Round-trip happens via the STE wrapper (see below).
4. sparse_mla_blockscaled_apply(...)
   — High-level entry that uses the reference path for float-carrier inputs.
   `force_metal=True` preserves the old Path B meaning and now raises with the
   retired-Path-B reason.
5. Path C E8M0 prepared-buffer surfaces in sparse_mla_blockscaled_path_c.py
   — TileLang-DSL QK reducer plus a fused prepared-buffer forward entry point.
   The public Path C ABI consumes existing raw e4m3 bytes and E8M0 scale
   tensors; it does not quantize float carriers or unpack packed MXFP8 words
   inside the wrapper. The owner-output route
   `sparse_mla_blockscaled_path_c_apply(..., out=..., lse=...)` compiles the
   TileLang apply kernel with `execution_backend="tvm_ffi"` and returns the
   same caller-owned `out`/`lse` MLX arrays instead of building an
   `mx.fast.metal_kernel` wrapper. Reducer lowering failures now fail closed
   instead of building a serial `mx.fast.metal_kernel` fallback; only the
   non-dispatch legacy `T.fp8_scaled_matmul` probe can emit diagnostic MSL.
   The checked-in receipt covers reducer timing only, so the full blockscaled
   Path C route is not production AUTO.

## Scale tensor handling

The gb10 kernel and Apple's mx.quantize(mode='mxfp8') *agree* on the layout
(one scale per 32-element block), but the dequant timing differs:

- **gb10**: stores FP8 data and FP32 scales separately, runs T.gemm on raw
  FP8 with FP32 accumulator, then post-multiplies by QScale * KVScale per
  block.
- **Apple MXFP8**: stores packed uint32 data and uint8 block exponents,
  applies the scale during dequantization. Subsequent matmul is on
  fully-recovered FP32 (or BF16) tensors.

Numerically the two should agree to within FP8 round-trip noise, but Apple's
path effectively does the dequant-then-matmul ordering whereas gb10 does
matmul-then-dequant. For the parity-oracle window this distinction does not
matter — both routes produce the same final attention output to within FP8
mantissa precision — but it will become relevant when the actual TileLang
metal kernel lands and we want to choose between in-kernel vs post-GEMM
dequant.

## FP8 dtype lowering surprises (block-scaled specific)

1. **mx.dequantize(mode='mxfp8') has no VJP in MLX 0.31.** It shares the
   FromFP8 primitive with mx.from_fp8 and raises the same
   ValueError: [Primitive::vjp] Not implemented for FromFP8. We work around
   this with a mx.custom_function STE wrapper (_mxfp8_roundtrip_ste),
   exactly mirroring the tensorwise FP8 module.
2. **mx.quantize(mode='mxfp8') only accepts last_dim % 32 == 0.** Calling
   it with a misaligned head dim raises a NumPy-style ValueError. Our
   reference returns the BF16 reference unchanged for non-aligned shapes —
   this matches the gb10 prototype which asserts (dim + tail_dim) % 32 == 0.
3. **Packed shape contract.** mx.quantize(mode='mxfp8') of
   [..., D] returns [..., D/4] (uint32) data and [..., D/32] (uint8)
   scales. Block size cannot be configured — the underlying MSL kernel hard
   codes group_size 32. Tests at
   tests/test_tilelang_sparse_mla_blockscaled.py::test_quantize_mxfp8_shape_contract
   pin this contract.

## Numerical tolerance

Per the task brief, forward parity vs the BF16 reference uses rtol=5e-3
and backward uses rtol=1e-2. Block-scaled MXFP8 has a higher noise floor
than tensorwise FP8 with per-token scales because the per-32-block scale is
amax-based, not amax-divided-by-448. On (B=1, S=64, H=4, D=64, topk=16)
with scale=0.1:

| path                                 | max_abs_err |
| ------------------------------------ | ----------- |
| FP8 (tensorwise) reference vs BF16   | ~3e-3       |
| Block-scaled MXFP8 reference vs BF16 | ~1.1e-2     |
| Hand-built quantized_matmul vs BF16  | ~0.0        |

atol=2e-2 is the budget the test uses for the block-scaled forward parity
case to clear the noise floor while still catching real correctness drifts.
Backward parity uses a dequantize-then-BF16 oracle the same way the
tensorwise FP8 path does — see docs/tilelang_ports/sparse_mla_fp8.md.

## Bench (current Apple M-series)

From bench/tilelang_ports/sparse_mla_blockscaled.json on the current
(B=1, S=64, H=4, D=64, topk=16) receipt shape:

| path                         | median ms |
| ---------------------------- | --------- |
| BF16 reference               | 0.351     |
| Block-scaled MXFP8 reference | 0.334     |
| quantized_matmul reference   | 0.332     |
| Path B direct-MSL blockscaled fwd | unavailable |
| Path C E8M0 QK reducer       | 0.170     |

The Path C E8M0 reducer remains a partial QK-only route; on the current smoke it
is faster than the full blockscaled reference
(`path_c_e8m0_qk_reduce_over_blockscaled_reference=0.5093`). The checked-in
receipt intentionally has
`qk_reducer_strict.passed=true` and `strict.passed=false`: reducer support is
available, while full Path C blockscaled AUTO promotion still needs a passing
full-dispatch receipt.

## Next steps

1. Keep float-carrier production/default dispatch on the MXFP8 reference until
   full prepared-buffer Path C has a passing fwd/bwd receipt.
2. Keep native TileLang FP8/E8M0 `T.gemm` lowering fail-closed until float8
   storage, E8M0 scale plumbing, and Metal GEMM lowering are present together.
3. Keep full blockscaled Path C on the prepared-buffer API until paired
   correctness and timing receipts clear the strict full-dispatch gate. Do not
   add hidden quantization, casts, CPU staging, or serial fallback kernels in
   the wrapper; prepared FP8 bytes and E8M0 scales stay as existing GPU buffers.
   Data movement rule: the owner-output Path C forward consumes existing
   `q_fp8`, `q_scale`, `kv_fp8`, `kv_scale`, and `indices` GPU buffers, writes
   into caller-owned `out`/`lse`, and creates only the scalar `sm_scale`
   control buffer. It does not allocate large outputs, stage CPU data, cast FP8
   carriers, cast E8M0 scales, or quantize inside the wrapper.
4. When tile-ai/tilelang HEAD lands FP8 + simdgroup support, verify the native
   path against the MXFP8 reference and retired Path B receipt before changing
   dispatch policy:

1. Verify T.gemm(A_fp8, B_fp8, C_fp32, ...) lowers correctly in
   block-scaled context (FP32 accumulator + per-block FP32 scale tensor).
2. Build the block-scaled PrimFunc with QScale/KVScale FP32 fragments,
   one per 32-element block along the head dim. Dequant happens post-GEMM,
   matching tilelang_sparse_mla_blockscaled_fused.py exactly.
3. Lower with target='metal', strip kernel signature, dispatch through
   mx.fast.metal_kernel.
4. Wire the manual VJP via mx.custom_function. The block-scaled backward
   in gb10 returns BF16 grads for *dequantized* Q/KV (not FP8 bytes); the
   Apple port should preserve that contract.
