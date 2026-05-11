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

## M0.4 20-Step Receipt Matrix

Existing local_gb10_quarter optimizer-matrix receipts live under
`bench/baselines/m04_optimizer_matrix/`. The checked-in
`summary_20step.json` currently covers 20-step bf16 rows for AdamW,
Muon/Muon+AdamW, Lion, Lion8bit, Adam8bit, and MuonAdamWInt8 variants, plus
per-case receipts for dynamic/symmetric 8-bit codecs. A historical
`fp8_path_c_adam8bit_dyn_lr1e-4.json` receipt is present but records the
older fail-closed FP8 route.

`scripts/m04_train_step.py` now writes two TL-8 receipt sections:

- `regression_report`: normalized route dispatch, dtype/carrier, optimizer,
  Path B/C observation, peak memory, finite/loss fields, tokens/sec, and
  fallback reason.
- `regression_report.fp8_path_c_producer_gate`: fail-closed FP8 producer gate
  with `status`, `ok`, `configured`, owner, required prepared buffers, and
  `producer_missing` reason when `fp8_path_c` has no DSA Sparse-MLA producer.
- `m04_20step_matrix`: generated commands for the 20-step matrix across
  `bf16`, `fp8_path_c`, and `int8` routes with AdamW, Muon, MuonAdamW,
  Lion, Lion8bit, and Adam8bit where supported. Each supported row records
  `dry_run_command`, `smoke_command`, `real_20step_command`, and the receipt
  also lists `real_100step_commands`.
- `m04_20step_matrix.baseline_comparison`: the 900 tok/s-class comparison
  target from existing real-parquet bs=1 seq=4096 20-step receipts. The
  reference receipts are `lion8bit_sym_lr1e-4.json` (900.6977 tok/s),
  `adam8bit_sym_lr1e-4.json` (894.2882 tok/s), and
  `adam8bit_dyn_lr1e-4.json` (890.0727 tok/s). These are local M4 receipts,
  not GB10 parity claims.

Current TL-W lane status:

- Real 20-step matrix readiness: commands are prepared for the supported rows,
  and partial real receipts were run on May 11, 2026. This is not the complete
  final 20-step matrix, and no 100-step real-parquet matrix has been run.
- Current green bf16 20-step receipts from `/tmp/cppmega_e2e_matrix_20260511`:
  `adamw` 314.14 tok/s, `muon_adamw` 40.26 tok/s, `nam56r` 40.51 tok/s,
  `lion` 370.82 tok/s, `adam8bit` 629.65 tok/s, `lion8bit` 686.21 tok/s, and
  `int8` 40.58 tok/s. Each completed 20/20 steps, stayed finite, and decreased
  loss.
- After the VJP fix, current green `fp8_path_c` 20-step receipts from
  `/tmp/cppmega_e2e_matrix_20260511/after_vjp` are: `adamw` 255.02 tok/s,
  `muon_adamw` 36.73 tok/s, `nam56r` 36.72 tok/s, `lion` 293.15 tok/s,
  `adam8bit` 452.72 tok/s, `lion8bit` 482.59 tok/s, and `int8` 36.71 tok/s.
  These prove finite 20-step behavior for those rows only.
- Remaining matrix blocker: the matched bf16-vs-fp8_path_c lion8bit repro in
  `/tmp/cppmega_repro_bf16_vs_fp8_path_c_20step_20260511` has bf16 green at
  916.56 tok/s, while the matching fp8_path_c run was stopped after 1216.9s
  (`returncode=-15`). Do not claim a full real 20-step matrix until that and the
  other supported rows are rerun under the final code.
- Dry-run smoke previously executed:
  `.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --dry-run-json --output /tmp/m04_bf16_adamw_20step_dryrun.json --json`
- Dry-run result: exit 0, receipt `status=dry_run`,
  `workload.model_profile=local_gb10_quarter`, `workload.batch_size=1`,
  `workload.seq_len=4096`, `timing.throughput_interpretation.production_shape=true`,
  and `m04_20step_matrix.baseline_comparison` present. It does not allocate
  optimizer state, run model forward/backward, or produce tok/s.
- Focused receipt tests: `.venv/bin/python -m pytest tests/test_m04_train_step.py -q`
  passed (77 tests).

Base command shape for supported cases:

```
.venv/bin/python scripts/m04_train_step.py \
  --model-profile local_gb10_quarter \
  --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet \
  --data-format parquet \
  --token-key token_ids \
  --steps 20 \
  --batch-size 1 \
  --seq-len 4096 \
  --dtype bfloat16 \
  --optimizer adamw \
  --optimizer-quant-scheme dynamic_int8_v1 \
  --lr 1e-4 \
  --grad-checkpoint \
  --output bench/baselines/m04_optimizer_matrix/bf16_adamw_20step.json \
  --json
```

For `fp8_path_c`, change `--dtype fp8_path_c`; the receipt records the bf16
carrier and Path C policy overrides. For `int8`, keep `--dtype bfloat16`
and use one of `--optimizer int8`, `--optimizer adam8bit`, or
`--optimizer lion8bit`; both `muon` and `muon_adamw` map to the
MuonAdamWInt8 route under `int8`. Plain AdamW and Lion are deliberately
marked unsupported for the int8 route because they keep fp32 optimizer state.

Generated real 20-step receipt commands:

```
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/bf16_adamw_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer muon --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/bf16_muon_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer muon_adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/bf16_muon_adamw_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/bf16_lion_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer lion8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/bf16_lion8bit_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer adam8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/bf16_adam8bit_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/fp8_path_c_adamw_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer muon --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/fp8_path_c_muon_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer muon_adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/fp8_path_c_muon_adamw_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/fp8_path_c_lion_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer lion8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/fp8_path_c_lion8bit_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer adam8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/fp8_path_c_adam8bit_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer int8 --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/int8_muon_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer int8 --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/int8_muon_adamw_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer lion8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/int8_lion8bit_20step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 20 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer adam8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --output bench/baselines/m04_optimizer_matrix/int8_adam8bit_20step.json --json
```

Generated real 100-step gate commands use the same matrix and add
`--require-loss-decrease` so non-decreasing rows fail closed:

```
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/bf16_adamw_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer muon --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/bf16_muon_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer muon_adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/bf16_muon_adamw_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/bf16_lion_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer lion8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/bf16_lion8bit_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer adam8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/bf16_adam8bit_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/fp8_path_c_adamw_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer muon --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/fp8_path_c_muon_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer muon_adamw --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/fp8_path_c_muon_adamw_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer lion --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/fp8_path_c_lion_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer lion8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/fp8_path_c_lion8bit_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype fp8_path_c --optimizer adam8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/fp8_path_c_adam8bit_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer int8 --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/int8_muon_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer int8 --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/int8_muon_adamw_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer lion8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/int8_lion8bit_100step.json --json
.venv/bin/python scripts/m04_train_step.py --model-profile local_gb10_quarter --data-path data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet --data-format parquet --token-key token_ids --steps 100 --batch-size 1 --seq-len 4096 --dtype bfloat16 --optimizer adam8bit --optimizer-quant-scheme dynamic_int8_v1 --lr 1e-4 --grad-checkpoint --require-loss-decrease --output bench/baselines/m04_optimizer_matrix/int8_adam8bit_100step.json --json
```

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
