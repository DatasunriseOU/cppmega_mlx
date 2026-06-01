# Speed-Matrix Campaign — Review Index (2026-06-01)

One place to review the full Path-A..E / Path-B vs Path-C speed campaign across **local Metal (Apple M4 Max)** and **gb10 (NVIDIA GB10, CUDA sm_121, MLX-CUDA backend)**. Every matrix is fail-loud (RULE #1): a cell that cannot run the *real* selected path reports the real error, never a silent fallback.

## Artifacts

| matrix | platform | cells | MD | HTML |
| --- | --- | --- | --- | --- |
| 1B training (full m04 step) | local Metal | 28 | `local_matrix_latest.md` | `local_matrix_latest.html` |
| 1B training **fast-fused** (path_b / path_c / path_c_chunked) | gb10 CUDA | 54 | `cppmega_1b_speed_matrix_gb10_fastfused_20260601.md` | `cppmega_1b_speed_matrix_gb10_fastfused_20260601.html` |
| v4 op-level GDN/KDA (paths a–e) | local Metal | 40 | `cppmega_v4_speed_matrix_metal_20260531.md` | `cppmega_v4_speed_matrix_metal_20260531.html` |
| v4 op-level GDN/KDA (paths a–e) | gb10 CUDA | 40 | `cppmega_v4_speed_matrix_gb10_20260531.md` | `cppmega_v4_speed_matrix_gb10_20260531.html` |

Companion analyses: `../docs/TOKPS-DISCREPANCY.md` (the "75 vs 3000 tok/s" verdict) and `../docs/NVFP4-TRAINING-KERNELS.md` (nvfp4 fwd wired / bwd fail-loud, gb10 evidence).

---

## Headline 1 — The real Path-C win is on CUDA: fast-fused chunked mamba3

gb10 1B, flag-OFF (serial `path_c`) → flag-ON (`path_c_chunked`, `CPPMEGA_PATH_C_MAMBA3_CHUNKED_SCAN=1`), batch=1 seq=512 steps=10:

| dtype | opt | bits | serial tok/s | chunked tok/s | step/s speedup | compile speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | muon | 16 | 75.7 | **78.7** | 1.02× | **1.38×** |
| bf16 | adamw | 16 | 117 | **122** | 1.02× | **1.59×** |
| bf16 | lion | 16 | 121 | 121 | 0.98× | **1.60×** |
| fp8 | muon | 16 | 71.7 | **74.7** | 1.01× | **1.70×** |
| fp8 | adamw | 16 | 108 | **114** | 1.02× | **2.00×** |
| fp8 | lion | 16 | 114 | **119** | 1.01× | **2.00×** |

**Verdict:** at batch=1 single-step the steady-state is ~parity (≈1.0×, the per-call overhead dominates the tiny workload), but the fast-fused chunked path **compiles 1.15–2.0× faster** (6 small kernels vs one monolithic mega-kernel) and is the only Path-C variant that compiles at all for the long chains (avoids the MTLCompilerService/oversized-kernel crash). The steady-state advantage is expected to grow with batch — see `TOKPS-DISCREPANCY.md` §7 for the fair batch=4 comparison still to run.

## Headline 2 — "75.7 tok/s" identified

`75.7 tok/s = gb10 1B, path_c, muon, bf16, 16-bit, batch=1, seq=512`. It is the **muon** cell; on the same path adamw=117 and lion=121. The Megatron ~3700 tok/s is batch=4 × seq=4096 (=16,384 tok/step, 32× more) on tuned CUDA. Not a bug — full decomposition in `docs/TOKPS-DISCREPANCY.md`.

## Headline 3 — Platform reachability (fail-loud, honest)

- **gb10 1B:** `path_b` is **blocked** (MLX-CUDA cannot run the path_b reference); `path_c` / `path_c_chunked` run and pass the loss check (11.3 → 6.27, finite + decreasing). `nvfp4` cells all **fail-loud** (no nvfp4 training kernels yet; backward GEMM miscompiled on sm_121 — see `NVFP4-TRAINING-KERNELS.md`).
- **gb10 v4 (a–e):** only `path_a` (pure-MLX reference) and `GDN path_e` run on CUDA. `path_b` ("No Metal back-end"), `path_c` (`DLPackDeviceError` — TileLang compiled for `target=metal`, can't take CUDA arrays), `path_d` (Triton disabled), `KDA path_e` (Metal-only kernel) all **fail-loud**. → the v4 op-level Path-C/B/E dispatcher is still **Metal-target-only**; CUDA-target wiring for the v4 GDN/KDA ops is the open item (distinct from the m04-training Path-C, which already works on CUDA per Headline 1).
- **local Metal v4 (a–e):** `path_a/b/c/e` all run; `path_d` fails loud (Triton frontend disabled). Fastest = `path_e` (vendored mlx-lm gated_delta Metal kernel, ~1 ms, 225–287 Melem/s).
- **local Metal 1B:** at batch=1 `path_b` leads `path_c_warm` (e.g. adamw 457 vs 259) — same per-call-overhead-dominates-tiny-workload effect; not representative of the fused advantage at scale.

## Fair comparison (goal #1) — RESOLVED with measured evidence

The like-for-like **batch=4×seq=4096** comparison the goal asked for is **not achievable in MLX-eager on this model** — measured, memory-guarded ramp on gb10 (121 GB unified), bf16 adamw, real tok/s:

| config | path_b | path_c | peak mem (MLX) | note |
| --- | --- | --- | --- | --- |
| bs=1 × seq=512 | 156 tok/s | **117 tok/s** ✅ | ~22 GB | the one shape path_c runs |
| bs=2 × seq=512 | **259** ✅ | blocked | — | path_b scales w/ batch |
| bs=1 × seq=1024 | **191** ✅ | blocked | 33 GB (path_b) | path_b scales w/ seq |
| bs=1 × seq=1024 (efficient MoE) | **166** ✅ | n/a | **32.8 GB** | `CPPMEGA_MOE_EFFICIENT=1`, loss 11.29→5.65 |
| bs=1 × seq=4096 (efficient MoE) | **no-fit** | n/a | fwd-only **4.9 GB**; fwd+bwd+adamw **97 GB+** | backward burst > 105 GB safety cap (SIGTERM, not true OOM) |
| bs=4 × seq=4096 (literal goal) | **no-fit** | n/a | — | unreachable (bwd burst) |

(muon mirrors adamw: bs1→bs2 path_b 92→159.) Three independent, code-pinpointed facts:
- **path_c (CUDA direct-chain) is pinned to exactly bs=1×seq=512** — it blocks at bs>1 *or* seq>512 with `direct_fusion_chain_logical_buffers_missing` (the fused direct-chain runtime was built for that one shape). So *steady-state path_c-vs-path_b at scale is unmeasurable* — path_c doesn't run off its build shape. **path_b runs everywhere and scales** with both batch (156→259) and seq (156→191).
- **The seq=4096 forward FITS** (MLX peak **4.9 GB** efficient / 5.2 GB dense, fwd-only, bs=1) — the OOM wall is **entirely in the backward + AdamW optimizer step**, which bursts from ~21 GB steady to **83 GB (seq=2048)** / **97 GB+ and climbing (seq=4096)** in the few seconds the gradients + m/v optimizer states + recomputed grad-checkpoint activations all land at once.
- **The seq-scaling wall is NOT MoE-dominated at full-model scale.** Measured on gb10: at seq=1024 the dense MoE (32.6 GB) and the efficient sparse-gather MoE (32.8 GB) have *the same* peak — the MoE is only 4 of 13 layers (`AEMEAEMEAEMR`) with a small `expert_hidden=896`, so its activation share is minor relative to the 3584-wide attention/mamba layers + 1B-param AdamW state. The efficient MoE **does** cut the routed-MoE activation ~4× (single-layer fwd+bwd, d=3584, 16 experts, top_k=4, bf16: seq=4096 **3.84 GB → 2.20 GB, 1.75× less, 2.7× faster**), but that win is a small fraction of the full-model backward burst, so it alone does not bring seq=4096 under the safe budget.

**MLX-vs-C++ finding, corrected & measured:** C++/Megatron does bs=4×seq=4096 in **~26 GB** (3700 tok/s) with sparse all-to-all MoE + a fused training step (activation checkpointing + fused optimizer that never materializes all grads/optimizer-state at once). MLX-eager's wall is the **whole-model backward + optimizer burst** (not the MoE specifically): the forward fits in 5 GB, but the eager backward + AdamW first-update peaks ~97 GB at seq=4096 bs=1 — and the MoE, attention, and optimizer all contribute. The efficient MoE removes one O(num_experts) term but the dominant terms (per-layer backward recompute + full grad/optimizer-state materialization) remain.

**Two sub-questions:**
- *steady-state path_c-vs-path_b at scale*: not measurable on gb10 (path_c is bs=1-only there; path_b scales). At bs=1, path_b ≥ path_c (CUDA-eager reference is leaner than fused for small work). The fused Path-C win remains **compile-time** (1.15–2.0×, §Headline 1).
- *MLX-vs-C++ like-for-like at seq=4096*: still not reachable in MLX-eager bs=1 — but the wall is now precisely located: **the backward+optimizer burst (~97 GB), not the MoE**. The honest throughput comparison stays bounded to seq≤1024 (166 tok/s efficient / 191 dense at seq=1024 bs=1), where the per-token gap vs C++ 3700 tok/s @ bs4×seq4096 decomposes as token-count (16384 vs 1024 tok/step) × eager-reference-substrate × fused-vs-eager-optimizer (see `TOKPS-DISCREPANCY.md`).

**What it would take to reach seq=4096 fwd+bwd under 70 GB** (the remaining wall, evidence above): (a) a fused/streamed optimizer step that updates params in chunks instead of materializing all grads + AdamW m/v at once; (b) more aggressive activation checkpointing across the attention/mamba backward (the bulk of the burst); (c) the efficient MoE (now landed, env-gated) is a necessary-but-not-sufficient piece. Item (a) is the highest-leverage next step.

## Resolved since first draft
- **Memory-efficient MoE** ✅ landed, **env-gated** `CPPMEGA_MOE_EFFICIENT=1` (default OFF → existing paths byte-identical). `ReferenceMoE._routed_combine_sparse` gathers only the tokens routed to each expert (top_k), runs the FFN on that subset, scatter-adds back — **same math** as the dense loop. Numeric parity pinned in `tests/test_moe_efficient.py`: **forward bitwise-identical (max-diff 0.0 on BOTH Metal and CUDA)**; gradients within the fp32 reduction-order envelope (7.6e-5 Metal / 3.7e-3 CUDA — pure cuBLAS non-associativity, *not* an algorithm delta — proven by a bitwise-exact gather/scatter round-trip test). Single-layer fwd+bwd: seq=4096 **3.84→2.20 GB (1.75×), 2.7× faster**. Full-model gb10 seq=1024 bs=1: **166 tok/s, 32.8 GB, loss 11.29→5.65** (matches dense 191/32.6). *Conclusion: necessary but not sufficient for seq=4096 — the wall is the whole-model backward+optimizer burst, see Fair-comparison section.* (commits `72d8827`, `6c49768`, `60477f5`, `c19d4a8`).
- **nvfp4 forward** ✅ works on gb10 (rel_err 0.147). **nvfp4 backward** ✅ now REAL on sm_121 via the reduced RtN recipe `NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)` — dgrad 0.1465 / wgrad 0.1343 vs bf16, enabled behind a numeric gate; the DEFAULT RHT+SR recipe stays fail-loud (genuinely datacenter-only). See `NVFP4-TRAINING-KERNELS.md`, `NVFP4-NANOCHAT-RECIPE.md`, `NVFP4-GB10-MIXED-APPROACH.md` (commit `78b7b31`).
- **v4 op-level Path-C on CUDA** ✅ wired (`target=cuda` device-aware, commit `fe32cb1`); gb10 v4 matrix shows path_c `ok`.
- **path_b on CUDA** ✅ wired (commit `a319599`); full gb10 1B matrix shows path_b `ok` and scaling.

## Remaining growth items (honest, not blockers)
- **seq=4096 fwd+bwd memory**: the wall is now pinpointed to the **whole-model backward + AdamW optimizer burst** (forward fits in 4.9 GB; the eager backward + full grad/m/v materialization peaks ~97 GB at seq=4096 bs=1). Highest-leverage fix = a **fused/streamed (chunked-param) optimizer step** that never holds all grads + optimizer state at once, plus deeper attention/mamba backward checkpointing. The efficient MoE (above) is one landed piece; it is not the dominant term.
- Path-C direct-chain training runtime for **bs>1 on CUDA** (currently bs=1-only).
- nvfp4 backward **full convergence-parity** vs bf16 over a long run (only "runs + loss-descends + grads within 0.147" is proven so far).
