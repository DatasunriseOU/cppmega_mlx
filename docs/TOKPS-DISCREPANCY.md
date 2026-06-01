# The "75 vs 3000 tok/s" Discrepancy — Engineering Verdict

Status: synthesized from three field reports (A: MLX/Path-C 75.7 tok/s; B: cppmega C++/Megatron ~3000 tok/s; C: model/op equivalence). Numbers below are taken verbatim from those reports with their file:line citations; nothing is invented beyond them.

---

## 1. TL;DR Verdict

- **The headline "75 vs 3000" is NOT a like-for-like comparison and NOT primarily a single perf bug — it is a stack of measurement/config differences on top of a real kernel-substrate gap.** The two numbers measure different optimizers, different batch sizes, different precision, and different kernel implementations. Once normalized, the ~40x collapses to a much smaller residual that is real but expected.
- **Lead cause #1 — the 75.7 is specifically the MUON optimizer cell.** On the *same* MLX Path-C, `adamw`/`lion` hit **450–833 tok/s**; muon drops to ~75 because of **5 fp32 Newton-Schulz iterations (~15 matmuls each) per 2D weight matrix, run eagerly in MLX inside the timed `mx.eval`** every step (`optimizers.py:292-295,363-368,616`). So ~6–11x of the "MLX is slow" story is *muon-vs-adamw on the same MLX path*, not MLX-vs-C++.
- **Lead cause #2 — batch/sequence mismatch is ~32x by itself.** MLX 75.7 was measured at **batch=1 × seq=512 = 512 tokens/step** (`bench...matrix.py:211`, run seq=512; ntokens `m04_train_step.py:2119`). The C++ ~3700 was **batch=4 × seq=4096 = 16,384 tokens/step** (`local_gb10_quarter_train.sh:874-875,44`). That is 32x more tokens/step, and tok/s rises with batch on these hybrid kernels.
- **The "3000" is a real GB10 single-GPU number, but a literal "3000" is UNLOCATED.** The closest located figures are **3628–3747 tok/s** on **1× GB10 (sm_121), single process, MBS=4 GBS=4, grad-accum=1, DP=1**, full fwd+bwd+Muon step (`plan.md:110,116,989,991`). It is **already per-GPU = aggregate** (one GPU), NOT an N-GPU aggregate. The H200 production cluster is a *separate, larger* regime (~74k aggregate / ~9,250 tok/s/GPU, `plan.md:696`, `README.md:381`).
- **Model/architecture is NOT the cause.** Report C confirms both sides are the identical NAM56R `local_gb10_quarter` hybrid (pattern `AEMEAEMEAEMR`, depth 13, hidden 3584, ffn 18944, heads 28, MoE 16/top-4 + shared 1024, DSA ranks 1,2,3, MTP 2), with mathematically equivalent mamba3 recurrence and safe-softmax (sm_scale = 1/√d). The residual gap after normalization is **execution substrate** (MLX bf16 + pure-MLX/Metal reference kernels vs C++ tensorwise-FP8 + fused CUDA/Triton/TileLang), which is exactly what you would expect.

**Bottom line:** the 40x is dominated by (a) muon-only penalty on MLX (~6–11x), (b) 32x token-count difference per step, and (c) a genuine but unsurprising eager-MLX-reference vs fused-FP8-CUDA kernel gap. None of these is a hidden bug; they are apples-to-oranges measurement plus an expected substrate delta.

---

## 2. The 75.7 Number, Decomposed (Report A)

**What it is:** the muon-optimizer cell of the MLX 1B training matrix, a single fresh subprocess, eager, single-device, no data-parallel.

**Config:**
- **batch_size = 1** (`bench_1b_training_matrix.py:211`), **seq_len = 512** (run value; default block-size is 2048 at L212), **steps = 10** (run; default 10 at L213).
- **global_batch_tokens = batch × seq = 1 × 512 = 512 tokens/step.** `ntokens = mask.sum()` (`m04_train_step.py:2119`), = 512 for a fully-valid batch.
- **No grad accumulation:** `grad_accum_steps` defaults to **1** (`compiled.py:1686`) and m04 never overrides it (`m04_train_step.py:12881-12890`) → `do_update` true every step. Effective global batch == micro-batch.
- **Optimizer = muon.** CLI `muon` and `muon_adamw` both collapse to key `"muon_adamw"` (`m04_train_step.py:762-767`) → `MuonAdamWMulti` composite with `cppmega_cuda_parity=True` (`m04_train_step.py:13196-13202`). The int8 `path_c` row uses the same NS core (`m04_train_step.py:13218-13229`).

**The tok/s formula (no recomputation in the harness):**
- The matrix copies tok/s straight from the per-cell m04 JSON receipt: `tok_sec = float(timing["tokens_per_second"])` (`bench_1b_training_matrix.py:1015-1019`); markdown column is that same value (`:1325`).
- m04's `tokens_per_second` is the **arithmetic mean of per-step tok/s over ALL steps including step 0/compile** (`statistics.fmean(tps_values)`, `m04_train_step.py:13100`; values from `m04_train_step.py:12915`). There is no warmup discard at the tok/s level (the bench's `step_sec`/median drops step 0, but the tok/s mean does not).
- Each per-step value: `tokens_per_second = tokens / elapsed` (`compiled.py:1775`), `tokens = ntokens.item()` (`:1765`), `elapsed = perf_counter() - start` (`:1763`).

**What is inside the timed region** (`compiled.py:1740-1763`): full forward + backward via `value_and_grad` + `optimizer.update` (including Newton-Schulz) + **`mx.eval(model.state, optimizer.state, mx.random.state, loss, ntokens, grad_accum)`**. So optimizer-state materialization (Newton-Schulz) is timed and synced every step. It is steady-state-ish but **not** compile-excluded.

**Why muon ≈ 10x slower than adamw/lion on the same path:**
- `make_muon` default `ns_steps=5`, `ns_carrier="fp32"` (`optimizers.py:616,621`); not overridden → **5 fp32 Newton-Schulz iterations**.
- Per iteration: `gram = x @ x.T` + two `mx.addmm` (≈5 matmul-heavy ops × 5 iters ≈ **15 matmuls per 2D matrix per step**) plus `mx.linalg.norm` (`optimizers.py:288,292-295,363-368`), all eager MLX, materialized by the in-timer `mx.eval`.
- Applied to 2D linear weights only (`is_muon_compatible`, `optimizers.py:182-202`); everything else goes to AdamW. `MuonAdamWMulti.apply_gradients` runs **both** groups every step (`optimizers.py:558-572`), so muon cost = AdamW cost + NS overhead → strictly slower. With useful work tiny (batch=1, seq=512), the NS cascade dominates → ~75 tok/s vs 450–833 for adamw/lion.

---

## 3. The 3000 Number, Decomposed (Report B)

**Plainly: a literal "3000" is UNLOCATED.** The closest located real numbers are a tight cluster of GB10 single-GPU runs:

| tok/s | ms/step | config | dtype | source |
|---|---|---|---|---|
| **3747.7** | 4371.8 | quarter, MBS=4, `local_per_token` SparseMLA-FP8 | BF16 + FP8 MLA | `plan.md:989` |
| **3712.0** | 4413.8 | quarter, MBS=4, `te_tensorwise` | BF16 + FP8 MLA | `plan.md:991` |
| **3692.5** | 4437.158 | quarter, MBS=4, seq=4096, `tensorwise` | BF16 | `plan.md:110` |
| **3628.2** | 4515.766 | quarter, MBS=4, seq=4096, MXFP8 TN adapter | MXFP8 | `plan.md:116` |
| 4303.8 | 951.7 | **half** (26-layer) GB10 baseline, real data | BF16 | `blackwell_feature_sweep_2026_04_12.md:52,59` |

**Hardware / parallelism:** 1× GB10 (sm_121), single process, **TP=1 PP=1 DP=1 EP local**, produced by `scripts/local_gb10_quarter_train.sh`. Config: `CPPMEGA_LAYER_DEPTH=13` (line 31), `CPPMEGA_SEQ_LENGTH=4096` (line 44), `--micro-batch-size 4 --global-batch-size 4` (lines 874-875) → **grad-accum = 1**. Optimizer = TensorParallelMuon (q8 momentum) + Adam8bit; profiler shows `TensorParallelMuon.step CUDA total 2.597s` and CCE backward ~796ms → **full fwd+bwd+optimizer step**, not fwd-only.

**Per-GPU-per-step normalization arithmetic:**
- tokens/step = GBS × seq = **4 × 4096 = 16,384 tokens/step**.
- 16,384 / 4.3718 s = **3,747 tok/s**, matching `plan.md:989`.
- This is **1 GPU**, so per-GPU == aggregate. NOT an N-GPU sum. **3700 tok/s = 1 GPU, batch=4, seq=4096, full train step (fwd+bwd+Muon).**

**The H200 numbers are a different scale — do not conflate:** ~56,280 tok/s aggregate on 8× H200 (`nam56r_mimo7_nsys_profile_2026_04_11.md:3,99`), ~74,000 tok/s baseline 8× H200 production (`plan.md:696`, ~289 TFLOP/s, ~31% MFU), ~9,250 tok/s/GPU (BF16) / ~8,100 (FP8) per `README.md:381`. Production config = PP=1 TP=1 EP=8 DP=1, MBS=10 GBS=80, seq=4096, MTP=2, 8× H200 141GB (`production_status.md:15`). These belong to the full 4.73B 52-layer model, not the quarter.

---

## 4. Apples-to-Apples Normalization

Goal: bring both to **tokens/step/GPU at equal batch** and account for every multiplier in the ~40x (3747 / 75.7 ≈ **49.5x** at face value; ~40x as colloquially stated).

### 4a. As-measured (face value)
| side | optimizer | batch×seq | tokens/step | GPUs | tok/s (located) |
|---|---|---|---|---|---|
| MLX Path-C | muon | 1 × 512 | 512 | 1 | **75.7** |
| C++ GB10 quarter | muon (TP-Muon q8) | 4 × 4096 | 16,384 | 1 | **3,747** |

Face-value ratio ≈ **49.5x**.

### 4b. Attributing the multipliers

Both are single-GPU, so **(N-GPU x) = 1.0** — there is no data-parallel aggregation hiding here on either side. The remaining multipliers:

| multiplier | factor | basis |
|---|---|---|
| **(global-batch / token-count x)** | **~32x** | tokens/step 16,384 vs 512 = exactly 32x. Larger batch/seq amortizes per-step fixed cost on these kernels; tok/s scales up with it. This is the single biggest contributor and is pure measurement-config, not a bug. |
| **(N-GPU x)** | **1.0x** | both single-GPU; per-GPU == aggregate on both sides (`plan.md:989`; Report A "single-process, eager, no data-parallel"). |
| **(muon-vs-adamw x) — MLX side only** | **~6–11x** | On the same MLX path, adamw/lion = 450–833 tok/s vs muon 75.7. The 5-iter fp32 Newton-Schulz cascade in eager MLX (`optimizers.py:292-295`) is MLX-specific overhead; the C++ side runs TensorParallelMuon as a tuned CUDA kernel and does NOT pay this penalty. So this multiplier is "MLX-eager-muon tax," distinct from the generic substrate gap. |
| **(MLX-eager-vs-CUDA-fused x)** | **residual** | whatever remains after removing the above; this is the genuine kernel/precision substrate delta (see 4c). |
| **(model-difference x)** | **1.0x** | Report C: identical architecture/dims/op-set. No contribution. |

### 4c. Does the arithmetic close?

Normalize the MLX number to remove the two artifacts that are not substrate:

1. Remove the **muon tax**: use MLX adamw/lion on the same path → **450–833 tok/s** at batch=1×seq=512 (located range, Report A).
2. Remove the **batch/token mismatch**: this is harder — tok/s does not scale perfectly linearly with batch, so we cannot just multiply by 32. But the C++ side at batch=1×seq=512 was **not measured**, so we cannot complete a clean numeric closure. What we *can* say:
   - MLX adamw at the *small* batch already reaches **450–833 tok/s**.
   - The C++ number is at **32x the tokens/step**, where tok/s is higher partly because fixed per-step overhead is amortized.

**Honest statement: the arithmetic does NOT fully close to a single clean number, because the two missing measurements (MLX at batch=4×seq=4096, and C++ at batch=1×seq=512, both with a *matched* optimizer) do not exist in the reports.** What the evidence *does* establish is that the 49.5x decomposes as approximately **32x (token-count) × ~5–11x (muon-tax + eager-reference-kernel/precision substrate)**, with **0x from N-GPU and 0x from model architecture**. The "~40x" is therefore mostly explained by measurement config (batch + optimizer choice) layered on top of a real-but-expected substrate gap; it is not a single mystery 40x perf bug.

---

## 5. Model Equivalence (Report C)

**Same hybrid stack — confirmed, no architectural divergence.** Both repos are the NAM56R `local_gb10_quarter` model and the MLX side explicitly declares the C++ repo as its provenance (`MODEL_FACTORY_UPSTREAM_RECIPE_MODULE = "cppmega.recipes.run_profiles"`, `model_factory.py:28`).

Every profile constant matches: pattern `AEMEAEMEAEMR`, depth **13**, hidden **3584**, ffn **18944**, heads **28**, head_dim **128**, vocab **65536**, seq **4096**, DSA A-ranks **(1,2,3)**, MoE **16 experts / top-4 + shared 1024**, DSA indexer topk **256**, MTP **2**. Block stack maps 1:1: A→DSA/MLA attention, E→MoE, M→Mamba3, R→custom M2RNN. MHC/Engram/Concept/ngram-hash are OFF on both sides.

Math/op spot-checks are **equivalent**: mamba3 selective scan `h_t = exp(A·dt)·h_{t-1} + B·x_t` with matching softplus+dt_bias and B/C RMS-norm; attention safe-softmax with **sm_scale = 1/√head_dim** (rope factor = 1.0 for the standard rope this profile uses); RoPE applied identically.

**Contribution of model difference to the tok/s gap: 0x (none).** The only differences are *implementation substrate*, not architecture:
- **Precision:** C++ tensorwise-FP8 GEMMs + bf16, MXFP8 backward (`run_profiles.py:124-125`); MLX bf16 weights + fp32 softmax/scan reductions, no FP8.
- **Kernels:** C++ FA4 SM120 dense attn, fused `mamba_chunk_scan_combined` SSD (Triton chunk=256), Megatron alltoall MoE, TileLang fused Mamba3; MLX `mlx.fast.sdpa` + gather-based O(n²)-memory reference sparse-MLA, Python-chunked scan capped at 32 (MLX-0.31 correctness cap, `mamba3.py:393`), dense `ReferenceMoE`.

This substrate delta is precisely the "(MLX-eager-vs-CUDA-fused x)" residual in section 4 — real, expected, and the right place to attribute the leftover gap after batch and optimizer are normalized.

---

## 6. Honest Unknowns (RULE #1 — no papering over)

- **A literal "3000 tok/s" is UNLOCATED.** No report cites exactly 3000. The closest real numbers are 3628–3747 (GB10 quarter, single GPU). "3000" is a rounded-down colloquialism for that cluster.
- **The arithmetic does NOT close to a single clean factor.** The two measurements needed for a clean closure do not exist: (a) MLX at batch=4×seq=4096 with adamw or muon, and (b) C++ GB10 quarter at batch=1×seq=512 with a matched optimizer. Without them, the 32x token-count multiplier cannot be cleanly separated from the substrate multiplier numerically.
- **Whether step 0 / compile inflates or deflates the muon mean specifically is UNVERIFIED** (Report A): the m04 tok/s mean includes step 0 (`m04_train_step.py:13100`), but the directional effect on muon was not isolated.
- **No data-parallel in the MLX path is UNVERIFIED only in the negative sense** (Report A): no DP code is referenced in the m04 `local_gb10_quarter` route, but absence-of-reference is not a positive proof.
- **The `nsys` 56,280 vs 74,000 H200 discrepancy is noted but not fully reconciled** in Report B (different effective GBS in the profiled config vs the GBS=80 production run). Not load-bearing for the GB10-vs-MLX question, but flagged.
- **MLX muon at batch=1×seq=512 = 75.7 is a single located cell**; the adamw/lion 450–833 range is located but the exact per-optimizer values at this exact (batch,seq,steps) were given as a range, not pinned per row.

---

## 7. What Would Be a Fair Comparison — and What to Run Next

**A fair comparison must match, at minimum: optimizer, micro-batch × seq (tokens/step), grad-accum, GPU count, and ideally precision.** The current 75.7-vs-3747 matches none of optimizer, batch, or precision.

Concretely, to actually quantify MLX-vs-C++ substrate cost, run these on the *same* `local_gb10_quarter` model:

1. **MLX at the C++ batch:** rerun the MLX Path-C matrix at **batch=4, seq=4096, grad-accum=1**, for **muon** and for **adamw**. This isolates the 32x token-count multiplier and gives a matched-batch MLX number.
2. **Match the optimizer both ways:** report MLX **adamw** vs C++ **Adam8bit/AdamW**, and MLX **muon** vs C++ **TensorParallelMuon**, separately. Do not compare MLX-muon to C++-muon-as-tuned-CUDA without also showing the adamw pair — the muon tax is MLX-eager-specific.
3. **Exclude compile/step-0 consistently:** make the MLX tok/s a tokens/total-time figure (or drop step 0) to match how the C++ ms/step is reported, removing the warmup-inclusion asymmetry (`m04_train_step.py:13100`).
4. **Hold precision constant where possible**, or explicitly label the bf16(MLX) vs FP8/MXFP8(C++) delta as part of the substrate multiplier — do not fold it silently into "MLX is slow."
5. **Optional muon-tax fix on MLX:** the 5-iter fp32 eager Newton-Schulz (`optimizers.py:292-295,616`) is the obvious MLX hotspot; measuring `ns_steps` sensitivity (or a fused/Metal NS) would tell you how much of the muon penalty is recoverable vs intrinsic.

After (1)–(4), the residual MLX-eager-reference vs C++-fused-FP8 ratio at **equal optimizer, equal batch, equal GPU** is the only number that honestly answers "how much slower is the MLX substrate."

---

**Sources:** Report A (`bench_1b_training_matrix.py`, `m04_train_step.py`, `compiled.py`, `optimizers.py`), Report B (`plan.md`, `local_gb10_quarter_train.sh`, `blackwell_feature_sweep_2026_04_12.md`, `nam56r_mimo7_nsys_profile_2026_04_11.md`, `README.md`, `production_status.md`), Report C (`model_factory.py`, `pattern.py`, `nam56r.py`, `hybrid_lm.py`, `mamba3.py`, `attention.py`, `sparse_mla.py` and the C++ `run_profiles.py`, `megatron_args.py`, `nam56r_megatron.py`, `mamba3_mixer.py`, `dsa_sparse_attention.py`).
