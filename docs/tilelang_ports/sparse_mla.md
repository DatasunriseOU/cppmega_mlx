# Sparse Multi-Latent Attention (sparse_mla) Path B Port

Status: pure-MLX reference complete; Path B Metal kernel **blocked on tilelang
0.1.9 metal-target T.gemm support**. Reference is already wired into the autograd
path so the gap is correctness-neutral while the blocker stands.

Date: 2026-05-02

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


The TileLang kernel implements this with the standard flash-attention
log-sum-exp (Lse) trick across topk-blocks of size block_I (default 64).
The backward reuses the saved Lse, computes Delta = sum(O * dO) per row, then
runs a second pass to accumulate dQ and dKV (atomic-add into dKV with
split_store=2).

## Path B Status

Probe (see /tmp/probe_sparse_mla_tl.py) lowering a stripped-down variant of
the forward primfunc on tilelang 0.1.9 with target='metal' raises:


InternalError: Check failed: (0) is false: Unsupported target for gemm:
metal -keys=metal,gpu -max_function_args=31 -max_num_threads=256
-max_shared_memory_per_block=32768 -max_threads_per_block=256
-thread_warp_size=16


This is the documented T.gemm blocker. The forward primfunc has 3 T.gemm
calls per pipelined step (Q*K, Q_tail*K_tail, S*V) and the backward has 5
(Q*K, dO*K, dP*K, dP*Q, P*dO). Manual GEMM rewrites of all eight tile mms
(after-the-fact, with TileLang fragments and policies wired in) is roughly a
flash-attention kernel rewrite; outside the scope of a Path B *port*.

A parallel agent is rebuilding tilelang from HEAD with the Apple simdgroup
PRs that wire GemmInst for the metal target. When that lands, the status
helper will flip available=True and the kernel skeleton in
cppmega_mlx/nn/_tilelang/sparse_mla.py becomes the place to implement the
actual TileLang -> MSL -> mx.fast.metal_kernel pipeline.

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
- cppmega_mlx/nn/_tilelang/sparse_mla.py - Path B port scaffold + status
  surface + fallback dispatcher.
- tests/test_tilelang_sparse_mla.py - parity oracle tests + Path B status
  surface tests.
- scripts/bench_tilelang_sparse_mla.py - bench harness (writes
  bench/tilelang_ports/sparse_mla.json).

## Reference Parity Tolerances

- Forward parity (fp16 carrier vs hand-rolled NumPy fp32 oracle):
  atol=1e-3, rtol=1e-3.
- Backward smoke (autograd through reference vs central finite-difference of
  scalar loss): atol=5e-3, rtol=5e-3.

Tests cover four shape configurations, including:

- (B=2, S=128, H=8, D=64) smoke
- (B=4, S=512, H=8, D=64) larger smoke (matches porting-plan spec)
- (B=1, S=64, H=8, D=64, G=2) kv_group=2 / GQA-style grouping
- tail_dim=16 with d_v=32, qk_dim=48 to exercise the MLA channel split

## Bench Numbers (M4 Max, MLX 0.31.1)

Recorded by scripts/bench_tilelang_sparse_mla.py --warmup 3 --iters 8:

| Shape          | reference fwd | apply (fallback) fwd | reference bwd |
| -------------- | ------------- | -------------------- | ------------- |
| B2_S128_H8_D64 | 0.38 ms       | 0.33 ms              | 0.59 ms       |
| B4_S512_H8_D64 | 1.12 ms       | 0.98 ms              | 1.92 ms       |

These are M4 Max smoke timings only; they are **not** GB10 parity claims and
they do not include a Path B kernel row (still gated by the GEMM blocker).
The apply row currently calls the pure-MLX reference path and is reported so
that we can compare the same numbers once the kernel goes live.

## Next Steps Once Blocker Lifts

1. Re-run the probe at /tmp/probe_sparse_mla_tl.py against the tilelang HEAD
   wheel and confirm the metal target builds without Unsupported target for
   gemm.
2. Replace the NotImplementedError bodies in
   cppmega_mlx/nn/_tilelang/sparse_mla.py::sparse_mla_fwd_metal /
   ..._bwd_metal with the TileLang -> MSL -> mx.fast.metal_kernel flow.
   Reuse the transform pattern from /tmp/path_b_msl_mlx/bench_msl_path_b.py
   (paren-balanced signature parser, alphabetic buffer reordering, const
   tagging for inputs only).
3. Wrap the Metal forward in mx.custom_function whose VJP invokes the Metal
   backward.
4. Promote test_path_b_forward_parity and test_path_b_backward_parity from
   skip to passing parity tests against the pure-MLX reference at the same
   tolerances above.
5. Update bench/tilelang_ports/sparse_mla.json to add the path_b kernel row,
   and update this doc with the resulting numbers.
