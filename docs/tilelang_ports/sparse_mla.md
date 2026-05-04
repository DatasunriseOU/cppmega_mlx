# Sparse Multi-Latent Attention (sparse_mla) Path B/C Metal Port

Status: pure-MLX reference complete; Path B direct-MSL Metal forward/backward is
the fallback production path on Apple Silicon; Path C TileLang-DSL-lowered Metal
forward/backward is available for forced runs via `CPPMEGA_KERNEL_PATH=path_c`
and for per-shape AUTO promotion. AUTO promotes only checked-in receipt rows
whose forward `no_worse_than_path_b` flags are true; unreceipted or failing rows
stay on Path B.

Date: 2026-05-04

## Source Attribution

Forward kernel on gb10:
cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_fwd.py (391 LOC).

Backward kernels on gb10:
cppmega/megatron/sparse_mla_ops/tilelang_sparse_mla_bwd.py (531 LOC; three
prim_funcs: preprocess, bwd, postprocess).

Autograd glue on gb10:
cppmega/megatron/sparse_mla_ops/sparse_mla.py (class SparseMLA).

Both the forward and backward originate from
NVIDIA/Megatron-LM PR #3674 (HollowMan6/Megatron-LM:dsa_cp_thd), which in turn
ports tile-ai/tilelang/examples/deepseek_v32/sparse_mla_{fwd,bwd}.py.

## Algorithm

Sparse-MLA is multi-latent attention (DeepSeek-V3) with per-token index gating.
For each query position the kernel only attends to the top-k KV positions
selected by an external indexer (e.g. the DSA top-k selector). The packed KV
tensor stores both the value channels (the leading d_v dims) and a tail
(rest of qk_dim) used only for QK^T. Concretely:

- q: [B, S, H, qk_dim] where qk_dim = d_v + tail_dim and
  H = num_q_heads (grouped across kv_group KV heads, head_kv = H / kv_group).
- kv: [B, S_kv, kv_group, qk_dim]. The first d_v channels feed V, the
  full qk_dim channels feed K.
- indices: [B, S, kv_group, topk], int32, sentinel -1 for masked entries.

Per token the operator computes (for each kv group g, each q head h in g):


gathered = kv[b, indices[b, s, g, :], g, :]            # [topk, qk_dim]
scores   = (q[b, s, h, :] @ gathered.T) * sm_scale     # [topk]
scores   = where(indices != -1, scores, -inf)
probs    = softmax(scores)                              # masked entries -> 0
out[b, s, h, :] = probs @ gathered[:, :d_v]             # [d_v]


The pure-MLX reference implements the same log-sum-exp (LSE) softmax contract
and is the parity oracle for both Metal paths. The Mac production Path B does
not lower the CUDA-oriented TileLang `T.gemm` source. It uses hand-written MSL
threadgroup reductions through `mx.fast.metal_kernel`, with fp16 carrier I/O and
fp32 score/LSE accumulators. Path C keeps the algorithm expressed in TileLang
DSL, then canonicalizes the lowered MSL back to the same lane-loop shape as Path
B and reuses Path B's host-side dKV scatter/reduction contract.

## Path B Status

Path B is live. On a Metal-capable host:

- `sparse_mla_attention(...)` with the default `CPPMEGA_KERNEL_PATH=auto`
  records `kernel_used="metal_kernel_fwd_v1"`.
- `CPPMEGA_KERNEL_PATH=path_b` forces the same direct-MSL kernel and fails closed
  if Metal is unavailable.
- `CPPMEGA_KERNEL_PATH=ref` forces `sparse_mla_attention_reference(...)`.
- The custom VJP calls the direct-MSL backward kernel and checks gradients
  against the pure-MLX VJP oracle.

The historical TileLang 0.1.9 probe lowering the upstream `T.gemm` primfunc to
`target="metal"` raised:

InternalError: Check failed: (0) is false: Unsupported target for gemm:
metal -keys=metal,gpu -max_function_args=31 -max_num_threads=256
-max_shared_memory_per_block=32768 -max_threads_per_block=256
-thread_warp_size=16

That is no longer the in-tree Path B status. It explains why Path B bypasses the
upstream CUDA-style `T.gemm` pipeline and ships as direct MSL instead.

## Path C Status

Path C is live for sparse-MLA BF16 with a per-shape fail-closed AUTO gate:

- `sparse_mla_path_c_status()` returns available on hosts with MLX Metal and the
  TileLang import stack.
- `CPPMEGA_KERNEL_PATH=path_c` routes sparse-MLA through
  `tilelang_path_c_fwd_bwd_v1`.
- Default AUTO routes receipt-covered forward-green shapes through
  `tilelang_path_c_fwd_bwd_v1`; today that includes `B2_S128_H8_D64`,
  `B4_S512_H8_D64`, and `B4_S1024_H8_D64`.
- Path C mirrors Path B's fp16 carrier and partial-dKV contract, including
  topk16/32/64 parity coverage.
- Default AUTO stays on Path B for unreceipted rows or rows with any false
  forward `no_worse_than_path_b` flag. The current checked-in receipt is full
  forward/backward strict and passes with Path C paired C/B <= 1.0 on every
  benchmark row.

## fp16 vs bf16 Carrier Note

GB10 source uses bfloat16 for the carrier dtype across Q/KV/Output. Apple
Metal in tilelang 0.1.9 has documented bf16 simdgroup MSL bugs (cubecl#1202),
so the Path B contract for cppmega.mlx is to **force fp16 carrier** at the
Metal boundary (see _promote_to_fp16_carrier in
cppmega_mlx/nn/_tilelang/sparse_mla.py). Inputs in bf16 round-trip through
fp32 to avoid mantissa loss on a direct bf16 -> fp16 cast. The pure-MLX
reference accepts any of fp16/bf16/fp32 because reductions promote to fp32
internally.

The fp16 carrier costs about 1 ULP versus bf16 at the same exponent range; for
sparse-MLA where topk is small (16-128) the softmax accumulator stays well
inside fp16 dynamic range with the fp32 LSE buffer. Once the bf16 simdgroup
bug is fixed upstream we can revisit lifting the fp16 forced-cast.

## Files

- cppmega_mlx/nn/sparse_mla.py - pure-MLX reference parity oracle.
- cppmega_mlx/nn/_tilelang/sparse_mla.py - Path B direct-MSL forward/backward
  kernels, status surface, custom VJP wrapper.
- cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py - Path C TileLang DSL
  forward/backward, MSL canonicalization, and forced-path wrapper.
- tests/test_sparse_mla_dispatch.py - production dispatcher, env-policy, and
  dispatch-log coverage for AUTO/ref/path_b/path_c.
- tests/test_tilelang_sparse_mla.py - parity oracle tests, Path B/C status
  surfaces, Path C topk16/32/64 coverage, and lowered-MSL shape guards.
- scripts/bench_tilelang_sparse_mla.py - bench harness (writes
  bench/tilelang_ports/sparse_mla.json).

## Reference Parity Tolerances

- Path B forward parity against pure-MLX reference: atol=2e-3, rtol=1e-3.
- Path B backward parity against pure-MLX VJP: atol=5e-3, rtol=5e-3.
- Path C forward parity against Path B: atol=2e-3, rtol=1e-3.
- Path C forward parity against pure-MLX reference: atol=8e-3, rtol=5e-3.
- Path C LSE parity against Path B: atol=3e-3, rtol=2e-3.
- Path C backward parity against pure-MLX VJP: atol=5e-3, rtol=5e-3.

Tests cover four shape configurations, including:

- (B=2, S=128, H=8, D=64) smoke
- (B=4, S=512, H=8, D=64) larger smoke (matches porting-plan spec)
- (B=1, S=64, H=8, D=64, G=2) kv_group=2 / GQA-style grouping
- tail_dim=16 with d_v=32, qk_dim=48 to exercise the MLA channel split
- Path C topk32/topk64 direct parity against Path B.

## Bench Numbers (M4 Max, MLX 0.31.1)

Use `scripts/bench_tilelang_sparse_mla.py` for current local receipts. The
checked-in routing summary records:

| Shape family | Forward C/B | Paired forward C/B | Backward C/B | Paired backward C/B | Gate |
| ------------ | ----------- | ------------------ | ------------ | ------------------- | ---- |
| B2_S128_H8_D64 | 0.973 | 0.993 | 1.067 | 0.913 | AUTO promotes to Path C |
| B4_S512_H8_D64 | 1.048 | 0.993 | 1.044 | 0.975 | AUTO promotes to Path C |
| B4_S1024_H8_D64 | 1.017 | 0.994 | 0.998 | 0.997 | AUTO promotes to Path C |

Strict receipt: `--strict --max-ratio 1.0 --warmup 5 --iters 20` passed for
all rows with `strict.phase="all"` and no failures. The unpaired ratios remain
diagnostic; the strict gate uses paired C/B medians because Path B and Path C
are measured in an alternating run to avoid Apple-GPU cache/power-state skew.

These are M4 Max smoke/bench receipts only; they are **not** CUDA/H200 or GB10
acceptance claims.

## Next Steps

1. Keep the AUTO policy per-shape and fail-closed: add a dispatch test whenever
   a receipt row changes from failing to passing or passing to failing.
2. Use `CPPMEGA_KERNEL_PATH=path_c` for forced Path C parity/benchmark work and
   keep it fail-closed when TileLang or Metal is unavailable.
3. Regenerate `bench/tilelang_ports/sparse_mla.json` and update
   docs/production_kernel_routing.md, this page, and dispatch tests together
   before widening AUTO promotion to any additional BF16 shapes.
4. Keep FP8 and e8m0 sparse-MLA Path C separate from BF16 Path C; current gaps
   there are scheduler/full-layout coverage, not this BF16 port.
