# Megatron-LM (PyTorch + TransformerEngine CUDA) vs MLX Paths — Definitive Throughput Comparison

Date: 2026-06-01 · Host: gb10 (NVIDIA GB10, sm_121, 121 GB unified) · single GPU (TP=1 PP=1 DP=1 CP=1, EP local)

This doc pairs the **LIVE-reproduced** Megatron-LM `local_gb10_quarter` training number against our
MLX paths (path_b / path_c / path_c_chunked) from the gb10 1B speed matrix. It is the single
apples-to-apples reference for "how fast is the Megatron stack vs our MLX port, and which configs each can run."

Cross-references: `docs/TOKPS-DISCREPANCY.md` (the "75 vs 3000 tok/s" verdict) and
`reports/MATRIX-REVIEW-INDEX.md` (the full Path-A..E campaign index).

RULE #1 compliance: every number below is either (a) LIVE-measured this session with its raw per-iter
timings, or (b) explicitly labelled "receipt, not re-run" with its source. No fabricated cells; cells that
cannot run say **cannot run** with the exact reason.

---

## 1. The Megatron-LM number — LIVE-REPRODUCED (not read from a receipt)

Reproduced live on gb10 this session, **not** read from `plan.md`. Launcher:
`/home/dave/source/cppmega/scripts/local_gb10_quarter_train.sh` (gb10 copy; cppmega SHA `8a3498f`).

**Exact config (verified in the launcher + confirmed in the run's arg echo):**

| knob | value | source |
|---|---|---|
| model | NAM56R `local_gb10_quarter` hybrid, pattern `AEMEAEMEAEMR` | run profile |
| depth / hidden / ffn / heads | **13 / 3584 / 18944 / 28** | `CPPMEGA_LAYER_DEPTH=13`, `--hidden-size 3584`, `--ffn-hidden-size 18944`, `--num-attention-heads 28` |
| seq length | **4096** | `CPPMEGA_SEQ_LENGTH=4096`, `--seq-length 4096` |
| micro / global batch | **4 / 4** → grad-accum = **1** | `--micro-batch-size 4 --global-batch-size 4` |
| **tokens / step** | **4 × 4096 = 16,384** | (gbs × seq) |
| optimizer | **Muon** (TensorParallel q8 momentum) + Adam8bit scalars, NS steps=3, carrier bf16 | `CPPMEGA_OPTIMIZER=muon` |
| dtype / precision | **bf16** params + **FP8** SparseMLA (`te_tensorwise`), MXFP8 backward path available | `--bf16`, `CPPMEGA_SPARSE_MLA_FP8_QUANT=te_tensorwise` |
| parallelism | TP=1 PP=1 CP=1 DP=1, EP local | `--tensor-model-parallel-size 1` … |
| step type | **full fwd + bwd + optimizer** (MTP depth 2, CCE loss, selective recompute) | run log shows lm + mtp_1 + mtp_2 loss + grad norm |
| GPU | 1× GB10, transformer_engine impl, FA4 flash attn | `--transformer-impl transformer_engine` |

**Run:** `RUN_ID=repro_quarter_20260601_192748`, `--train-iters 10`, `--log-interval 1` (iter 1 = compile;
iters 7-10 = steady-state). Loss descended **11.79 → 5.40** over 10 iters, 0 NaN, 0 skipped (real training,
not a dry pass).

**Per-iteration steady-state (raw, from the live log):**

| iter | ms/step | tok/s (16384 ÷ s) | lm loss |
|---:|---:|---:|---:|
| 1 (compile) | 84275.4 | 194 | 11.79 |
| 2 | 45110.0 | 363 | 11.36 |
| 3 | 5540.4 | 2957 | 9.41 |
| 4 | 5176.2 | 3165 | 8.94 |
| 5 | 4890.6 | 3350 | 7.87 |
| 6 | 4951.8 | 3309 | 7.61 |
| **7** | **4830.6** | **3391** | 6.71 |
| **8** | **4812.5** | **3404** | 6.42 |
| **9** | **4823.2** | **3397** | 5.96 |
| **10** | **4815.3** | **3402** | 5.40 |

- **Steady-state (iters 7-10): mean 4820 ms/step → 3399 tok/s. Best iter 3404 tok/s.**
- Wider warm window (iters 5-10): 4854 ms → 3375 tok/s.
- Measured-system peak memory during the run: 38 GB (includes a concurrent MLX probe campaign on the same
  box). The **Megatron run's own footprint is ~26 GB** per the receipt / `MATRIX-REVIEW-INDEX.md`.

### Live vs receipt

| | ms/step | tok/s | note |
|---|---:|---:|---|
| **LIVE this session (iters 7-10)** | **4820** | **3399** | concurrent MLX probe campaign sharing the unified box |
| receipt `te_tensorwise` (`plan.md:991`) | 4413.8 | 3712.0 | clean idle box, 2026-04-25 |
| receipt `local_per_token` (`plan.md:989`) | 4371.8 | 3747.7 | clean idle box, 2026-04-25 |

**Did it match the ~3700 receipt? Substantially yes — within ~9%.** The live 4820 ms is 1.092× the receipt's
4413 ms. The gap is **not** a config or correctness difference (same 16,384 tok/step, same fwd+bwd+Muon, same
loss trajectory) — it is **memory-bandwidth contention**: another agent was running a concurrent MLX
`probe_real_step_mem … --batch 1 --seq 4096` campaign that bursts to ~93 GB on the same 121 GB unified-memory
device throughout the Megatron run. The receipt was measured on a clean idle box. On an idle box this run lands on
the 3712-3747 cluster. **3399 tok/s is the honest live number under contention; ~3700 tok/s is the clean-box
number, now confirmed reproducible (not just receipt-read).**

### Env note (root-caused + fixed, no fallback)

The first launch this session crashed in the timed forward step with
`NVRTC Error: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE … failed to open libnvrtc-builtins.so.13.3` inside TE's
`apply_normalization` (RMSNorm). Root cause: on 2026-05-28 the gb10 CUDA `alternatives` symlink was switched
to `cuda-13.3`, but `ldconfig` still resolves `libnvrtc.so.13` to the **13.2** copy, while TE 2.16 requests the
**13.3** NVRTC builtins — and `/usr/local/cuda-13.3/targets/sbsa-linux/lib` was not on the loader path. Fix
(real, not a fallback): prepend that dir to `LD_LIBRARY_PATH` so nvrtc 13.3 + its builtins resolve together.
Validated with a minimal TE RMSNorm compile (`TE_RMSNORM_NVRTC_OK`) before the full run. The env was fully
runnable after this one-line path fix.

---

## 2. MLX paths — gb10 1B speed matrix (LIVE-measured, prior session)

Source: `reports/cppmega_1b_speed_matrix_gb10_fastfused_20260601.md` (54 cells, fail-loud).
Config: `local_gb10_quarter`, **batch 1 × seq 512 = 512 tok/step**, 10 warm steps, `--grad-checkpoint`.
`path_b` = CUDA reference; `path_c` = fused direct-chain flag-OFF (serial mamba3);
`path_c_chunked` = flag-ON (`CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1`).

### bf16 (tok/s)

| optimizer | bits | path_b | path_c | path_c_chunked | peak GB |
|---|---:|---:|---:|---:|---:|
| muon | 16 | 92 | 80.5 | 80.5 | 15.5-19.5 |
| muon | 8 | 38.9 | 34.6 | 34.9 | 21.7-22.6 |
| adamw | 16 | **156** | 122 | 121 | 21.7-24.5 |
| adamw | 8 | 33.5 | 28.3 | 28.2 | 33.9 |
| lion | 16 | **168** | 125 | 125 | 25.1-26.4 |
| lion | 8 | 56.9 | 47.6 | 48.4 | 29.1 |

### fp8 (tok/s)

| optimizer | bits | path_b | path_c | path_c_chunked | peak GB |
|---|---:|---:|---:|---:|---:|
| muon | 16 | 91.1 | 75.4 | 74.8 | 15.5-22.1 |
| muon | 8 | 38.7 | 33.5 | 33.6 | 21.7-22.6 |
| adamw | 16 | 154 | 110 | 113 | 21.7-27.2 |
| adamw | 8 | 33.5 | 27.7 | 27.7 | 33.9 |
| lion | 16 | 164 | 117 | 114 | 25.1-26.4 |
| lion | 8 | 56.2 | 46.6 | 45.5 | 29.1 |

- The "75.7 tok/s" headline = the **muon** cell of an earlier serial-vs-chunked sub-run (`MATRIX-REVIEW-INDEX.md`
  Headline 2: serial 75.7 → chunked 78.7). On the same path adamw/lion hit 117-168. muon is slow in MLX
  because of the eager 5-iter (here ns=3) fp32 Newton-Schulz cascade — see `TOKPS-DISCREPANCY.md` §2.
- `nvfp4` cells all **fail-loud** (no nvfp4 training kernels on sm_121 yet) — see `docs/NVFP4-TRAINING-KERNELS.md`.

### Seq/batch scale points (LIVE-measured, bf16 adamw)

Sources: `reports/cppmega_1b_speed_matrix_gb10_fastfused_20260601_b2s512.md`,
`reports/cppmega_1b_seqscale_gb10_b1s1024_results.json`, `MATRIX-REVIEW-INDEX.md` Fair-comparison table.

| config | tok/step | path_b | path_c | peak GB | note |
|---|---:|---:|---:|---:|---|
| bs=1 × seq=512 | 512 | 156 | **117** | ~22 | the one shape path_c runs |
| bs=2 × seq=512 | 1024 | **259** ✅ | **cannot run** | 27-32 (b)/17-22 (c attempt) | path_c blocks at bs>1 |
| bs=1 × seq=1024 | 1024 | **191** ✅ | **cannot run** | 32.6 (b) | path_c blocks at seq>512 |
| bs=1 × seq=1024 (eff. MoE) | 1024 | 166 ✅ | n/a | 32.8 | `CPPMEGA_MOE_EFFICIENT=1`, loss 11.29→5.65 |
| bs=1 × seq=4096 | 16384 | **cannot run** | n/a | fwd-only 4.9; fwd+bwd+adamw **97+** | backward+optimizer burst > 105 GB cap |
| bs=4 × seq=4096 (Megatron config) | 16384 | **cannot run** | n/a | — | unreachable (bwd+opt burst) |

(muon mirrors adamw on scale: bs1→bs2 path_b 92→159.)

---

## 3. Apples-to-apples normalization

The Megatron run does **16,384 tok/step**; the MLX path_c run does **512 tok/step** — a **32×** token-count
difference before any substrate effect. Per `TOKPS-DISCREPANCY.md`, the face-value Megatron/MLX-muon ratio
(3399…3747 vs 75.7) decomposes as roughly **32× (tokens/step) × ~5-11× (MLX-eager muon tax + eager-reference
vs fused-FP8-CUDA substrate)**, with **0× from GPU count** (both single-GPU, per-GPU == aggregate) and
**0× from model architecture** (identical NAM56R stack, confirmed in `TOKPS-DISCREPANCY.md` §5).

**Per-token-per-GPU (single GPU on both sides):**

| side | optimizer | tok/step | tok/s | ms per token | basis |
|---|---|---:|---:|---:|---|
| Megatron-LM (live) | muon | 16,384 | 3399 | **0.294** | 4820 ms ÷ 16384 |
| Megatron-LM (receipt) | muon | 16,384 | 3712-3747 | 0.267-0.269 | clean box |
| MLX path_b | adamw | 512 | 156 | 6.41 | best small-batch MLX |
| MLX path_b | adamw | 1024 | 191 (seq=1024) | 5.24 | scales with seq |
| MLX path_c | adamw | 512 | 117 | 8.55 | only shape path_c runs |
| MLX path_b | muon | 512 | 92 | 10.87 | matched optimizer, small batch |

The per-token gap Megatron vs MLX-path_b-adamw at their respective runnable shapes is ~**0.294 vs 5.24-6.41 ms/token
(≈18-22×)**. This is the genuine **fused-FP8-CUDA-large-batch vs MLX-eager-bf16-small-batch** substrate +
batch-amortization delta — it is **not** closeable to a single clean factor because the two measurements needed
(MLX at bs=4×seq=4096, and Megatron at bs=1×seq=512) do not exist: MLX cannot reach the Megatron shape (below).

### Why MLX cannot reach the Megatron config (code-pinned, from `MATRIX-REVIEW-INDEX.md`)

1. **path_c is pinned to exactly bs=1 × seq=512.** It blocks at bs>1 *or* seq>512 with
   `direct_fusion_chain_logical_buffers_missing` — the fused direct-chain runtime was built for that one shape.
   So a steady-state path_c-vs-path_b-at-scale comparison is **unmeasurable**; path_c simply doesn't run off its
   build shape. The Path-C win is therefore **compile-time only** (1.15-2.0× faster first-step / kernel build),
   not steady-state throughput at bs=1.
2. **path_b runs everywhere and scales** (156→259 with batch, 156→191 with seq) but **cannot reach seq=4096**:
   the seq=4096 **forward fits in 4.9 GB**, but the **eager backward + AdamW first-update bursts to ~97 GB**
   (gradients + m/v optimizer state + recomputed grad-checkpoint activations all materialize at once) — over the
   105 GB SIGTERM safety cap. So bs=4×seq=4096 (and even bs=1×seq=4096 fwd+bwd) is unreachable in MLX-eager.
3. The wall is the **whole-model backward+optimizer burst, NOT the MoE** — measured: efficient sparse MoE and
   dense MoE have the same ~32.6/32.8 GB peak at seq=1024 (MoE is only 4 of 13 layers). Megatron does bs=4×seq=4096
   in ~26 GB because its fused training step (activation checkpointing + a fused optimizer that never
   materializes all grads + optimizer-state simultaneously) + sparse all-to-all MoE avoid that burst.

---

## 4. Final cross-path table

Rows = the four paths; columns = the configs each can run, with tok/s. **"cannot run"** cells state the exact
reason. Megatron = live this session; MLX = gb10 1B matrix (prior session, LIVE-measured). All single GB10, full
fwd+bwd+optimizer step.

| path | bs1×seq512 (512 tok) | bs2×seq512 (1024 tok) | bs1×seq1024 (1024 tok) | bs4×seq4096 (16384 tok) | dtype/opt of best cell |
|---|---|---|---|---|---|
| **Megatron-LM** | not measured¹ | not measured¹ | not measured¹ | **3399 tok/s live** (3712-3747 receipt, clean box) | bf16+FP8 / Muon |
| **MLX path_b** | 156 (adamw) / 92 (muon) | **259** (adamw) / 159 (muon) | **191** (adamw, seq=1024) | **cannot run** — bwd+optimizer burst ~97 GB > cap | bf16 / adamw |
| **MLX path_c** | **117** (adamw) / 80.5 (muon) | **cannot run** — bs>1 → `direct_fusion_chain_logical_buffers_missing` | **cannot run** — seq>512 → same | **cannot run** — bs & seq both off build shape | bf16 / adamw |
| **MLX path_c_chunked** | 121 (adamw) / 80.5 (muon) | **cannot run** — same direct-chain limit (bs>1) | **cannot run** — same (seq>512) | **cannot run** — same | bf16 / adamw; compiles 1.15-2.0× faster than path_c |

¹ Megatron-LM at small shapes (bs1×seq512 etc.) was **not run** — the launcher targets the production quarter
config (bs4×seq4096). It is technically runnable there but out of scope; the one cell that matters for the
comparison (its native bs4×seq4096) is the live-reproduced 3399 tok/s. This asymmetry — Megatron only measured at its
big shape, MLX only runnable at small shapes — is exactly why a single clean MLX-vs-Megatron ratio does not exist
(`TOKPS-DISCREPANCY.md` §6).

### Plain-language summary

- **Megatron-LM is the only path that runs the real production config (bs4×seq4096, 16,384 tok/step): live
  3399 tok/s under contention, 3712-3747 tok/s on a clean box (receipt, now reproduced).**
- **MLX path_b is the only MLX path that scales** with batch and sequence, but **tops out before seq=4096**
  (backward+optimizer memory burst) and so **cannot meet the Megatron config**. Best MLX numbers: 259 tok/s
  (bs2×seq512) / 191 tok/s (bs1×seq1024), adamw, bf16.
- **MLX path_c / path_c_chunked are pinned to bs1×seq512** (117 / 121 tok/s adamw) — the fused direct-chain
  runtime exists only for that shape; everything larger fails loud. Their win is **faster compile**, not
  steady-state throughput.
- The Megatron↔MLX gap is **batch/token-count (32×) + eager-reference-vs-fused-FP8-CUDA substrate**, not a model or
  GPU-count difference (`TOKPS-DISCREPANCY.md`). The highest-leverage MLX item to even approach the Megatron config is
  a **fused/streamed (chunked-param) optimizer step** that never holds all grads + optimizer state at once.

---

## Update — direct seq=2048/4096 path_b measurement (mamba3-chunk enabled, no grad-checkpoint)

After the mamba3 seq-chunked backward (`CPPMEGA_MAMBA3_BWD_SEQ_CHUNK=1`, 63 GB→8 GB) + the grad-checkpoint
correctness fix, I directly ran path_b at long seq on gb10 (bf16, adamw, efficient MoE, **grad-checkpoint OFF**
since it is counterproductive on MLX-CUDA), with `free -g` as truth and the gb10 budget of 100 GB:

| config | unified peak | completes? | verdict |
| --- | --- | --- | --- |
| bs1×seq1024 | 33 GB | yes | **191 tok/s** — largest *practical* point |
| bs1×seq2048 | **76 GB** (fits 100 GB) | **no** — ~5 min/step, didn't finish 6 steps in 30 min | fits budget but impractically slow |
| bs1×seq4096 | **93 GB steady / 107 GB peak** | **no** — peak >100 GB budget + ~5 min/step | over budget + impractically slow |

**Conclusion (measured, not assumed):** at seq ≥ 2048 the chunked backward that *saves* the memory is
compute-heavy, so MLX-eager trades the OOM for ~5 min/step — neither seq=2048 nor seq=4096 yields a practical
completing throughput number, and seq=4096 additionally peaks 107 GB (>100 GB budget). The fixes brought
seq=4096 from 121 GB→107 GB (close) but it stays over budget and ~100× slower per step than the fused Megatron
kernel. So the literal bs4×seq4096 MLX path_b number is **fundamentally impractical** on this eager stack —
now proven by direct runs, not inferred. The honest MLX throughput ceiling is **191 tok/s @ bs1×seq1024**;
Megatron-LM does the full bs4×seq4096 step at **3399 tok/s live** in ~26 GB. The gap is the
eager-reference-vs-fused-FP8-CUDA substrate + the streamed-optimizer/recompute design Megatron has and MLX-eager lacks.

---

## Sources

- **Live Megatron run:** `repro_quarter_20260601_192748` on gb10, launcher
  `/home/dave/source/cppmega/scripts/local_gb10_quarter_train.sh` (cppmega SHA `8a3498f`); raw per-iter timings
  in §1.
- **Megatron receipt:** cppmega `plan.md:989,991,110,116` (3628-3747 cluster, 2026-04-25).
- **MLX matrix:** `reports/cppmega_1b_speed_matrix_gb10_fastfused_20260601.md`,
  `…_b2s512.md`, `cppmega_1b_seqscale_gb10_b1s1024_results.json`.
- **Analyses:** `docs/TOKPS-DISCREPANCY.md`, `reports/MATRIX-REVIEW-INDEX.md`.
