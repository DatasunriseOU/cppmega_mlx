# Production-shape throughput sweep on local_gb10_quarter at T=4096 (~1.985B)

Date: 2026-05-03
Hardware: Mac Studio M4 Max, 128 GB unified, applegpu_g16s
MLX: 0.31.x
Receipt: bench/baselines/local_gb10_quarter_throughput_m4.json
Bench script: scripts/bench_local_gb10_quarter_throughput.py

This receipt is the M0 production-shape throughput evidence for the local
GB10 quarter profile at sequence length 4096, which is the cppmega parquet
corpus shape (clang_semantic_4k_v10 / clang_commits_4k_v1). Synthetic random
tokens are used at vocab=65536; this is a throughput receipt only and does
not by itself satisfy the M0.4 acceptance gate (real-parquet 100-step
loss-decrease).

## Workload

- Model: cppmega_mlx.recipes.model_factory.local_gb10_quarter
  - depth=13, hidden_size=3584, ffn_hidden_size=18944
  - num_attention_heads=28, head_dim=128, vocab_size=65536
  - pattern=AEMEAEMEAEMR, MTP depth=2, ~1.985B params
  - bf16 weights with grad-checkpoint enabled (mx.checkpoint per block)
  - Loss: next-token cut-cross-entropy (chunked logits, chunk_rows=512) so
    the lm_head does not materialize the full [B*T, V] logits tensor.
- Sequence length: 4096 (parquet-shape constraint; not bench T=512/T=1024).
- Optimizer state dtype: fp32 master moments. bf16 parameters.
- Path B kernels enabled by default (CPPMEGA_KERNEL_PATH=auto).

Benchmark protocol per (B, optimizer):
1. mx.reset_peak_memory() and mx.clear_cache() before allocation.
2. Synthetic random tokens, vocab=65536, fixed seed=4096.
3. 100 steps; first 50 discarded as warm-up.
4. Tokens-per-second median/p10/p90 over the post-warm-up window.
5. Stop the sweep when peak_memory_gb exceeds 88.0 GB (~70% of 128 GB).

Both groups in the Muon+AdamW MultiOptimizer share the parity LR
(`make_muon(cppmega_cuda_parity=True)` -> Muon `lr=1e-4`, AdamW `lr=1e-4`,
`betas=(0.9, 0.999)`, Nesterov off, NS steps=5, fp32 NS carrier).

## A. Lion sweep at T=4096

`make_lion(learning_rate=3e-3, betas=(0.9, 0.99), weight_decay=0.1)` —
the empirically-optimal LR from the Stream E TinyLM smoke (matches AdamW at
~3x LR per Chen et al. 2302.06675).

<!-- LION_TABLE -->

## B. Muon+AdamW sweep at T=4096

`make_muon(cppmega_cuda_parity=True)` — mirrors the cppmega CUDA wiring:
2-D `nn.Linear` weights routed to Muon, embeddings/lm_head/RMSNorm/Mamba
scalars routed to AdamW (Megatron `_is_nonlinear_or_embedding` predicate
inverted).

<!-- MUON_TABLE -->

## C. Path A vs Path B at the winning Muon+AdamW B

Path B is the production-target kernel set
(`cppmega_mlx.nn.mamba3.metal_kernel_fwd_v1` and
`cppmega_mlx.nn.m2rnn.metal_kernel_fwd_v1`). Path A is the pure-MLX reference
that the dispatcher falls back to when `CPPMEGA_KERNEL_PATH=ref`.

`sparse_mla_attention` is **not** wired into the live `local_gb10_quarter`
forward in this codebase: the DSA route in `CausalSelfAttention` is a dense
placeholder (cf. `cppmega_mlx/nn/attention.py` `AttentionRouteInfo`). Path B
verification therefore covers the two ops that DO dispatch in the live model
forward — `mamba3_mimo` and `m2rnn`. When sparse_mla is wired into the model
forward this comparison should grow to a third op.

<!-- PATH_TABLE -->

## D. M4 Max vs GB10 honest comparison

Honest comparison is Muon+AdamW only (that's what GB10 actually runs). Lion
is a memory-tight option for the 128+48 GB Stream F smoke, not the GB10
parity reference.

<!-- M4_VS_GB10_TABLE -->

Caveats:

- The GB10 ~4000 tok/s reference is reported on the cppmega CUDA stack
  with depth=13, hidden=3584, vocab=65536, MTP=2 — but the **GB10 batch
  size at T=4096 is not recorded in the public reference**. We compare
  M4 Max max-sustainable B at T=4096 against the GB10 reference; the
  ratio is hardware-only when the workload shape matches and is otherwise
  a B-mismatch comparison.
- Single-host receipts: `gb10_parity_claim=false` on the receipt. This is
  not a Megatron-distributed parity claim and should not be quoted as one.
- The M4 Max throughput is steady-state median over steps 50-99; the
  first 50 steps include MLX compile + Metal codegen warm-up.

## Conclusions

<!-- CONCLUSIONS -->

## Reproducibility

```
.venv/bin/python scripts/bench_local_gb10_quarter_throughput.py \
  --batch-sizes 1,2,3,4,6,8,10,12 \
  --seq-len 4096 \
  --optimizers lion,muon_adamw \
  --steps 100 \
  --warmup 50 \
  --memory-cap-gb 88 \
  --out bench/baselines/local_gb10_quarter_throughput_m4.json
```

The receipt contains per-row kernel dispatch snapshots and per-row
optimizer state size in GiB. Use `scripts/compare_bench_rows.py` for
cross-row comparisons; do not quote a single row as a hardware-class claim.
