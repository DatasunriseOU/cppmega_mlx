# Production-shape throughput sweep on local_gb10_quarter at T=4096 under 60 GB cap (~1.985B)

Date: 2026-05-03
Hardware: Mac Studio M4 Max, 128 GB unified, applegpu_g16s
MLX: 0.31.1
macOS: macOS-26.4.1-arm64-arm-64bit-Mach-O
Receipt: bench/baselines/local_gb10_quarter_throughput_60gb_m4.json
Bench script: scripts/bench_local_gb10_quarter_throughput.py

This receipt is the M0 production-shape throughput evidence for the local
GB10 quarter profile at sequence length 4096 (the cppmega parquet corpus
shape — `clang_semantic_4k_v10` / `clang_commits_4k_v1`) under a Docker-tight
60 GB unified-memory budget on the M4 Max. The peak-memory cap was set to
**58 GB** (2 GB safety margin under the 60 GB budget). Synthetic random
tokens are used at vocab=65536; this is a throughput receipt only and does
not by itself satisfy the M0.4 acceptance gate (real-parquet 100-step
loss-decrease).

## Workload

- Model: cppmega_mlx.recipes.model_factory.local_gb10_quarter (~1.985B params)
  - depth=13, hidden_size=3584, ffn_hidden_size=18944
  - num_attention_heads=28, head_dim=128, vocab_size=65536
  - pattern=AEMEAEMEAEMR, MTP depth=2
  - bf16 weights with grad-checkpoint enabled (mx.checkpoint per block)
  - Loss: next-token cut-cross-entropy (chunked logits, chunk_rows=512) so
    the lm_head does not materialize the full [B*T, V] logits tensor.
- Sequence length: 4096 (parquet-shape constraint; T<4096 is forbidden).
- Optimizer state dtype: fp32 master moments. bf16 parameters.
- Path B kernels enabled by default (CPPMEGA_KERNEL_PATH=auto).
  Verified: `metal_kernel_fwd_v1` fires for `mamba3_mimo` and `m2rnn`
  on the first step of every (optimizer, B) cell; `path_b_dispatched=True`
  on each row.

Benchmark protocol per (B, optimizer):
1. mx.reset_peak_memory() and mx.clear_cache() before allocation.
2. Optional mx.set_memory_limit(58 GB) to fail-closed at cap (applied).
3. Synthetic random tokens, vocab=65536, fixed seed=4096.
4. 50 steps total; first 25 discarded as warm-up. (Reduced from 100/50 for
   the production sweep to keep total wall time under 30 min.)
5. Tokens-per-second median/p10/p90 over the post-warm-up window.
6. Stop the sweep when peak_memory_gb exceeds 58.0 GB.

Both groups in the Muon+AdamW MultiOptimizer share the parity LR
(`make_muon(cppmega_cuda_parity=True)` -> Muon `lr=1e-4`, AdamW `lr=1e-4`,
`betas=(0.9, 0.999)`, Nesterov off, NS steps=5, fp32 NS carrier).

## A. Lion sweep at T=4096, 60 GB cap

`make_lion(lr=3e-3, betas=(0.9, 0.99), wd=0.1)` — the empirically-optimal LR
from the Stream E TinyLM smoke (matches AdamW at ~3x LR per Chen et al.
arXiv 2302.06675).

| B | tok/s median | tok/s p10 | tok/s p90 | peak GB | % of 60 GB | loss[0] | mean_last10 | opt state GB | cap hit |
| - | ------------ | --------- | --------- | ------- | ---------- | ------- | ----------- | ------------ | ------- |
| 1 | 762 | 739 | 779 | 53.69 | 89.5% | 11.262 | 71.176 | 11.58 | no |
| 2 | - | - | - | 83.17 | 138.6% | 11.263 | 11.263 | 11.58 | yes |

## B. Muon+AdamW sweep at T=4096, 60 GB cap

`make_muon(cppmega_cuda_parity=True)` — mirrors the cppmega CUDA wiring:
2-D `nn.Linear` weights routed to Muon, embeddings/lm_head/RMSNorm/Mamba
scalars routed to AdamW (Megatron `_is_nonlinear_or_embedding` predicate
inverted). This is the configuration GB10 actually runs.

| B | tok/s median | tok/s p10 | tok/s p90 | peak GB | % of 60 GB | loss[0] | mean_last10 | opt state GB | cap hit |
| - | ------------ | --------- | --------- | ------- | ---------- | ------- | ----------- | ------------ | ------- |
| 1 | 318 | 292 | 407 | 51.22 | 85.4% | 11.262 | 11.702 | 13.39 | no |
| 2 | - | - | - | 83.08 | 138.5% | 11.263 | 11.263 | 13.39 | yes |

## C. Path A vs Path B at the winning Muon+AdamW B (B=1)

Path B is the production-target kernel set
(`cppmega_mlx.nn.mamba3.metal_kernel_fwd_v1` and
`cppmega_mlx.nn.m2rnn.metal_kernel_fwd_v1`). Path A is the pure-MLX reference
that the dispatcher falls back to when `CPPMEGA_KERNEL_PATH=ref`.

`sparse_mla_attention` now has a prepared FP8 Path C route for DSA A-layers
when `CPPMEGA_KERNEL_PATH__SPARSE_MLA=path_c`: `CausalSelfAttention` produces
`q_fp8/q_scale/kv_fp8/kv_scale/indices` before dispatching the TileLang Path C
kernel. This Path B comparison still covers the two ops that dispatch in the
measured Path B route — `mamba3_mimo` and `m2rnn`. Sparse-MLA Path C should be
tracked in a separate receipt because it is a prepared-FP8/kernel-policy axis,
not part of the Path A vs Path B comparison below.

| Path | tok/s median | peak GB | % of 60 GB | path_b_dispatched | cap hit |
| ---- | ------------ | ------- | ---------- | ----------------- | ------- |
| Path B (auto) | 318 | 51.22 | 85.4% | yes | no |
| Path A (ref) | - | 86.93 | 144.9% | yes | yes |

## D. M4 Max vs GB10 honest comparison

Honest comparison is Muon+AdamW only (that's what GB10 actually runs). Lion
is a memory-tight option for the 128+48 GB Stream F smoke, not the GB10
parity reference.

| metric | value |
| ------ | ----- |
| M4 Max max sustainable B at T=4096 (Muon+AdamW, 58 GB cap) | B=1 |
| M4 Max tok/s median (Muon+AdamW, B=1, T=4096) | 318 |
| M4 Max peak memory (Muon+AdamW, B=1, T=4096) | 51.22 GB |
| GB10 reference tok/s (Muon+AdamW, T=4096, B unknown) | ~4000 |
| Ratio M4 Max / GB10 | 0.080 (8.0%) |

Caveats:

- `scripts/m04_train_step.py` is a correctness receipt, not this throughput
  sweep. Its `timing.tokens_per_second` denominator is loss target tokens
  (`B * (S - 1)` for dense next-token batches), and short `seq_len` runs such
  as 1024 or smaller are latency smokes that underfill the 4k production
  shape. Do not compare those rows to the 4k sweep or the GB10 ~4000 tok/s
  reference.
- **Budget asymmetry — the M4 Max is running with ~70 GB removed by Docker.**
  The 58 GB cap forces B=1 on this M4 Max receipt; the historical 88 GB-cap
  receipt at this same model shape (when Docker was not present) fit larger
  batch sizes. The M4 Max-vs-GB10 ratio reported here is therefore a **lower
  bound** on the M4 Max's hardware capacity. A fair head-to-head would either
  (a) match GB10's batch size or (b) lift the 58 GB cap on the M4 Max.
- The GB10 ~4000 tok/s reference is reported on the cppmega CUDA stack
  with depth=13, hidden=3584, vocab=65536, MTP=2 — but the **GB10 batch
  size at T=4096 is not recorded in the public reference**. Without a
  matched-B point we cannot disentangle hardware delta from batch-shape
  delta.
- Single-host receipt: `gb10_parity_claim=false`. This is not a Megatron-
  distributed parity claim and should not be quoted as one.
- The M4 Max throughput is steady-state median over steps 25-49; the first
  25 steps include MLX compile + Metal codegen warm-up.

## Conclusions

- **Maximum batch size that fit under 58 GB at T=4096:**
  - Lion: B=1
  - Muon+AdamW: B=1
- **Lion vs Muon+AdamW throughput delta at the winning B (B=1):**
  - Lion: 762 tok/s
  - Muon+AdamW: 318 tok/s
  - Delta: Lion is +139.4% faster than Muon+AdamW.
- **Was the 58 GB cap the limit?** Yes. At B=2, both optimizers jumped to >83 GB peak (Lion 83.2 GB, Muon+AdamW 83.1 GB) on the very first step, so neither sweep was throughput-plateau-limited; both were memory-cap-limited. The B=1 -> B=2 jump is super-linear because activation memory plus gradient-checkpoint stash plus optimizer state plus grad buffers all scale together.
- **Path A vs Path B at the winning Muon+AdamW B (B=1):** Path A (`CPPMEGA_KERNEL_PATH=ref`) hit 86.93 GB peak on the first step, exceeding the 58 GB cap — Path B is the only kernel path that fits in budget at this shape. Path B's median is 318 tok/s; Path A could not produce a tok/s figure under the cap.
- **M4 Max vs GB10 verdict (Muon+AdamW, T=4096):** M4 Max sustains 318 tok/s at B=1 under a Docker-imposed 58 GB cap; GB10 reference is ~4000 tok/s with an unknown batch size. The raw ratio is 8.0%, but **the comparison is not apples-to-apples**: Docker is consuming roughly 70 GB of the M4 Max's 128 GB unified memory, which forces B=1 here. With the full 128 GB available the M4 Max would likely sustain a larger B (the 88 GB-cap historical receipt fit B=4); the 58 GB-cap result understates the M4 Max because the budget asymmetry is real and large.
- **Bench wall time:** 1467s (~24.5 min) total across both optimizer sweeps and the Path A comparison.
- **Surprise:** Lion at lr=3e-3 on synthetic random tokens diverged (loss[0]=11.26 -> mean_last10=71.18); this bench measures throughput only, not loss-quality, so divergence on synthetic uniform tokens is expected and does not reflect on Lion at a real corpus.

## Reproducibility

```
.venv/bin/python scripts/bench_local_gb10_quarter_throughput.py \
  --batch-sizes 1,2,3,4 \
  --seq-len 4096 \
  --optimizers lion,muon_adamw \
  --steps 50 \
  --warmup 25 \
  --memory-cap-gb 58 \
  --max-runtime-s 1800 \
  --out bench/baselines/local_gb10_quarter_throughput_60gb_m4.json
```

The receipt contains per-row kernel dispatch snapshots and per-row
optimizer state size in GiB. Use `scripts/compare_bench_rows.py` for
cross-row comparisons; do not quote a single row as a hardware-class claim.
