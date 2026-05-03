# TileLang block-scaled (MXFP8) sparse-MLA port (Path B status)

This note documents the Apple Metal port of cppmega's MXFP8 block-scaled
sparse-MLA forward and backward kernels. The work lives at:

- Module: `cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py`
- Tests: `tests/test_tilelang_sparse_mla_blockscaled.py`
- Bench: `scripts/bench_tilelang_sparse_mla_fp8.py` -> `bench/tilelang_ports/sparse_mla_blockscaled.json`

## Source kernels (gb10)

- `cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_fused.py`
  - `sparse_mla_blockscaled_mxfp8_fwd` (forward)
  - `sparse_mla_blockscaled_mxfp8_bwd_kernel` (backward)
- `cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_blockscaled_qk.py`
  (separate experimental QK-only block-scaled scoring helper)

## ABI

The block-scaled kernel consumes pre-quantized MXFP8 payloads with the
following layout (from `tilelang_sparse_mla_blockscaled_fused.py`):

```
q_data:   [B, S,  H, D_total]       torch.float8_e4m3fn
kv_data:  [B, SK, G, D_total]       torch.float8_e4m3fn
q_scale:  [B, S,  H, D_total/32]    torch.float32
kv_scale: [B, SK, G, D_total/32]    torch.float32
indices:  [B, S,  G, topk]          torch.int32, -1 sentinel
```

`MXFP8_BLOCK_SIZE = 32`: one FP32 scale per 32-element block along the head
dim. The kernel walks the head dim in 32-element chunks, runs
`T.gemm(q_block, kv_block, partial)`, then accumulates
`partial * QScale[h, kb] * KVScale[j, kb]` into `acc_s` before the
online-softmax + S@V flow.

## Status on Apple Silicon (tilelang 0.1.9, MLX 0.31)

Same two blockers as the tensorwise FP8 path (see
`docs/tilelang_ports/sparse_mla_fp8.md`):

1. **FP8 dtype emission to MSL is unsupported.** Failure at
   `3rdparty/tvm/src/target/source/codegen_metal.cc:271`:
   `Cannot convert type float8_e4m3 to Metal type`.
2. **T.gemm is not registered for the metal target.**
   `Unsupported target for gemm: metal -keys=metal,gpu ...`.

Block-scaled additionally needs the per-block dequant pattern
`partial * QScale[h, kb] * KVScale[j, kb]` to lower correctly — but since
both blockers above are upstream of that, this is moot until they lift.

## What this port ships today

`cppmega_mlx/nn/_tilelang/sparse_mla_blockscaled.py` provides:

1. `sparse_mla_blockscaled_metal_status(...) -> SparseMLABlockScaledMetalStatus`
   — Returns `available=False` with both blocker reasons. Reports the
   `block_size=32` constant in the status dataclass.
2. `sparse_mla_blockscaled_fwd_metal(...)` and
   `sparse_mla_blockscaled_bwd_metal(...)`
   — Stubs returning `(status, None, None)` while gated.
3. `sparse_mla_blockscaled_reference(...)`
   — Pure-MLX MXFP8 reference using `mx.quantize(mode='mxfp8')` /
   `mx.dequantize(mode='mxfp8')` with `group_size=32`. The MLX kernel returns
   packed `uint32` data and `uint8` scale shorthand: each scale slot is a
   power-of-two block exponent, and 4 fp8 e4m3 values pack into one uint32.
   Round-trip happens via the STE wrapper (see below).
4. `sparse_mla_blockscaled_apply(...)`
   — High-level entry that falls back to the reference and supports
   `force_metal=True` to surface the blocker.

## Scale tensor handling

The gb10 kernel and Apple's `mx.quantize(mode='mxfp8')` *agree* on the layout
(one scale per 32-element block), but the dequant timing differs:

- **gb10**: stores FP8 data and FP32 scales separately, runs `T.gemm` on raw
  FP8 with FP32 accumulator, then post-multiplies by `QScale * KVScale` per
  block.
- **Apple MXFP8**: stores packed `uint32` data and `uint8` block exponents,
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

1. **`mx.dequantize(mode='mxfp8')` has no VJP in MLX 0.31.** It shares the
   `FromFP8` primitive with `mx.from_fp8` and raises the same
   `ValueError: [Primitive::vjp] Not implemented for FromFP8.` We work around
   this with a `mx.custom_function` STE wrapper (`_mxfp8_roundtrip_ste`),
   exactly mirroring the tensorwise FP8 module.
2. **`mx.quantize(mode='mxfp8')` only accepts `last_dim % 32 == 0`.** Calling
   it with a misaligned head dim raises a NumPy-style `ValueError`. Our
   reference returns the BF16 reference unchanged for non-aligned shapes —
   this matches the gb10 prototype which asserts `(dim + tail_dim) % 32 == 0`.
3. **Packed shape contract.** `mx.quantize(mode='mxfp8')` of
   `[..., D]` returns `[..., D/4]` (uint32) data and `[..., D/32]` (uint8)
   scales. Block size cannot be configured — the underlying MSL kernel hard
   codes group_size 32. Tests at
   `tests/test_tilelang_sparse_mla_blockscaled.py::test_quantize_mxfp8_shape_contract`
   pin this contract.

## Numerical tolerance

Per the task brief, forward parity vs the BF16 reference uses `rtol=5e-3`
and backward uses `rtol=1e-2`. Block-scaled MXFP8 has a higher noise floor
than tensorwise FP8 with per-token scales because the per-32-block scale is
amax-based, not amax-divided-by-448. On `(B=1, S=64, H=4, D=64, topk=16)`
with `scale=0.1`:

| path                                  | max_abs_err |
| ------------------------------------- | ----------- |
| FP8 (tensorwise) reference vs BF16    | ~3e-3       |
| Block-scaled MXFP8 reference vs BF16  | ~1.1e-2     |
| Hand-built quantized_matmul vs BF16   | ~0.0        |

`atol=2e-2` is the budget the test uses for the block-scaled forward parity
case to clear the noise floor while still catching real correctness drifts.
Backward parity uses a dequantize-then-BF16 oracle the same way the
tensorwise FP8 path does — see `docs/tilelang_ports/sparse_mla_fp8.md`.

## Bench (current Apple M-series)

From `bench/tilelang_ports/sparse_mla_blockscaled.json` on the same shape:

| path                          | median ms |
| ----------------------------- | --------- |
| BF16 reference                | ~0.30     |
| Block-scaled MXFP8 reference  | ~0.30     |
| quantized_matmul reference    | ~0.29     |

Block-scaled is roughly the same wall-time as BF16 because Apple's mxfp8
quantize/dequantize is a fused MSL kernel that costs the same as a single
matmul tile pass. Once tilelang's metal target supports FP8 storage we
expect this path to drop to ~0.5x BF16 (matching the H100 ratio).

## Post-blocker plan

When tile-ai/tilelang HEAD lands FP8 + simdgroup support:

1. Verify `T.gemm(A_fp8, B_fp8, C_fp32, ...)` lowers correctly in
   block-scaled context (FP32 accumulator + per-block FP32 scale tensor).
2. Build the block-scaled PrimFunc with `QScale`/`KVScale` FP32 fragments,
   one per 32-element block along the head dim. Dequant happens post-GEMM,
   matching `tilelang_sparse_mla_blockscaled_fused.py` exactly.
3. Lower with `target='metal'`, strip kernel signature, dispatch through
   `mx.fast.metal_kernel`.
4. Wire the manual VJP via `mx.custom_function`. The block-scaled backward
   in gb10 returns BF16 grads for *dequantized* Q/KV (not FP8 bytes); the
   Apple port should preserve that contract.
