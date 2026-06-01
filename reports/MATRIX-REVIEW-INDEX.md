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

The like-for-like **batch=4×seq=4096** comparison the goal asked for is **provably not achievable in MLX-eager on this model** — it OOMs. Measured on gb10 (121 GB unified), memory-guarded ramp:

| config | result (gb10, bf16) |
| --- | --- |
| bs=1 × seq=512 (baseline) | path_b/path_c/path_c_chunked all run (full 54-cell matrix) |
| **bs=2 × seq=512** | path_b **scales** (muon 92→159, adamw 156→259 tok/s); **path_c BLOCKED** — `direct_fusion_chain_training_runtime_missing` (Path-C direct-chain runtime is **bs=1-only** on CUDA) |
| **bs=1 × seq=4096** (seq-matched to C++) | **OOM** — memory spiked 22→**121 GB** instantly, guard SIGTERM'd. MLX-eager cannot reach seq=4096 **even at batch=1** |
| bs=4 × seq=4096 (the literal goal config) | **OOM** (≥100 GB), never completes |

**Why MLX OOMs where C++ doesn't:** C++/Megatron does bs=4×seq=4096 in **~26 GB** (3700 tok/s) using **sparse all-to-all MoE**; the MLX path uses a **dense `ReferenceMoE`** that materializes all 16 experts (O(seq×experts)) plus gather-based O(n²)-memory reference sparse-MLA. So MLX-eager's memory blows up with seq, capping reachable seq well below 4096. **This is the concrete MLX-vs-C++ finding** — not a tuning gap but a different (reference vs production) MoE/attention implementation.

**What this means for the two sub-questions:**
- *steady-state path_c-vs-path_b at scale*: not measurable on gb10 (path_c is bs=1-only there; path_b scales). At bs=1, path_b ≥ path_c (CUDA-eager reference is leaner than fused for small work). The fused Path-C win remains **compile-time** (1.15–2.0×, §Headline 1), not steady-state-throughput at bs=1.
- *MLX-vs-C++ like-for-like*: impossible at C++'s config because MLX-eager OOMs at seq=4096. The honest comparison is bounded to seq=512, where the ~40× headline decomposes as muon-tax × token-count × eager-reference-substrate (see `TOKPS-DISCREPANCY.md`).

**Open growth items** (not blockers, documented honestly): (a) Path-C direct-chain runtime for bs>1 on CUDA; (b) replace dense `ReferenceMoE`/reference sparse-MLA with memory-efficient kernels so MLX can reach seq=4096.

## Resolved since first draft
- **nvfp4 forward** ✅ works on gb10 (rel_err 0.147). **nvfp4 backward** ✅ now REAL on sm_121 via the reduced RtN recipe `NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)` — dgrad 0.1465 / wgrad 0.1343 vs bf16, enabled behind a numeric gate; the DEFAULT RHT+SR recipe stays fail-loud (genuinely datacenter-only). See `NVFP4-TRAINING-KERNELS.md`, `NVFP4-NANOCHAT-RECIPE.md`, `NVFP4-GB10-MIXED-APPROACH.md` (commit `78b7b31`).
- **v4 op-level Path-C on CUDA** ✅ wired (`target=cuda` device-aware, commit `fe32cb1`); gb10 v4 matrix shows path_c `ok`.
- **path_b on CUDA** ✅ wired (commit `a319599`); full gb10 1B matrix shows path_b `ok` and scaling.

## Remaining growth items (honest, not blockers)
- Path-C direct-chain training runtime for **bs>1 on CUDA** (currently bs=1-only).
- Memory-efficient **MoE/attention** kernels (replace dense `ReferenceMoE` + reference sparse-MLA) so MLX-eager can reach seq=4096 without OOM.
- nvfp4 backward **full convergence-parity** vs bf16 over a long run (only "runs + loss-descends + grads within 0.147" is proven so far).
