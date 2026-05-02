# MLX Buy-vs-Build Matrix and Anti-Port Playbook

**Date:** 2026-05-02
**Scope:** Decision matrix for cppmega.mlx port. Distills an 8-agent research pass (3 code-level audits of nanochat / cppmega-local / gb10-remote, 5 ecosystem audits of mlx-lm / kernels / spec-decode / SOTA architectures / data+training stack).

This document is a planning artifact for the cppmega.mlx port, not a capability claim. Where it cites mlx-lm / vllm-mlx / ZMLX behavior, those are external receipts at the date above; verify ABI compatibility against the pinned MLX version before vendoring.

The headline finding is that we should **vendor first, fork second, write Metal kernels last**. Most rows in the master plan that look like "port from torch" already have a high-quality MLX-native implementation we can pin and reuse.

---

## 0. Headline shifts vs the original master plan

1. **mlx-lm already ships `mlx_lm/models/nanochat.py`** (verified locally at `mlx_lm/models/nanochat.py`). It implements ReLU² MLP, QK-RMSNorm, functional RMSNorm pre/post embedding, RoPE with negated frequencies, tanh logit softcap (cap=15) — the upstream-Karpathy nanochat surface. **Vendor it as the model spine** for the port and layer cppmega-specific extensions on top.
2. **MLA, DSA, MoE switch-routing, attention sinks + sliding window, RoPE variants, RMSNorm/LayerNorm fast paths, q4 affine quant, KV-cache q4 quant, prompt cache, sampling, AdamW, Muon, Lion, LoRA/QLoRA/full FT, vanilla speculative decoding** — **all already exist in mlx-lm or mlx core.** Vendor pattern, do not rewrite.
3. **scasella/nanochat-mlx** (MIT, last commit 2026-04-29) ships exactly the two pieces mlx-lm lacks for our port: a Muon+AdamW `MultiOptimizer` matching the GB10 setup (~289 LOC) and a BOS-aligned best-fit packer (~119 LOC). Vendor both files directly.
4. **vllm-mlx `paged_cache.py` + `prefix_cache.py`** are pure-Python Apache-2.0 and standalone-vendorable for paged KV inference (do **not** put paged paths in training graphs).
5. **ZMLX** (`swiglu_mlp`, `rmsnorm`, `gather_qmm_swiglu`, `moe_mlp`) provides Python-MSL kernels for fused decode paths, +6–13 % decode on MoE 4-bit. Vendor `swiglu_mlp` and `rmsnorm`. **`gather_qmm_swiglu` requires a patched MLX build — keep it behind a feature flag.**
6. **mlx-mfa** (MIT) is a vendorable FlashAttention 2 implementation with sliding-window/GQA, 1.75–2.15× over stock SDPA on M1 Max. Vendor for inference; profile on M4 Max before promoting in training.
7. **arozanov/turboquant-mlx** (Apache-2.0) ships fused Hadamard + Lloyd-Max KV-quant Metal kernels — exactly the planned `Stream G` KV quant work. Vendor.
8. **vllm-metal v0.2.0 paged Metal kernel** has a known per-layer sync issue (one `mx.eval+synchronize` per cache write — issue #188). Wrap behind a flag, do not silently rely on it.
9. **GB10 server is 31 commits ahead** of the local cppmega checkout (FA3/FA4 build infra + Mamba3 wave28–32). The active uncommitted work on gb10 is the **wave44A MXFP8 acceptance harness**. None of these block the MLX port — confirm with the team whether MXFP8 acceptance numbers should drive any MLX parity targets.
10. **Speculative decoding is greenfield on GB10** (CUDA accepts the `is_spec_decode` parameter and forwards to upstream Megatron with no cppmega code path). The MLX port is therefore not chasing CUDA parity — it is the first place to ship speculative for this stack.
11. **CUDA-side fp8 / mxfp8 / nvfp4 work is Hopper/SM120-specific.** Skip on M4 Max. Apple has no tensor-core-equivalent matmul accelerator on M4; M5 Neural Accelerators give 4× TTFT vs M4 but stay closed-source.
12. **No `fim`, `ifim`, `stp`, `eagle`, `medusa`, `mtp_draft`, `speculative` files exist anywhere in cppmega or gb10.** Those are all greenfield in MLX, with nanochat fork as the only torch reference.

---

## 1. Master Buy-vs-Build Matrix

Verdict legend: **VENDOR** = drop in unchanged (pin SHA), **FORK** = vendor + patch, **PORT-FUSED** = write fresh MLX with `mx.fast` + `mx.compile` rather than mirroring torch tensor-by-tensor, **WRITE-METAL** = custom Metal kernel after profiling justifies it, **SKIP** = not in scope.

| # | Feature | MLX-native source | Decision | Effort |
|---|---|---|---|---|
| A | engram (per-block n-gram + gate) | nanochat torch ref only | **PORT-FUSED** (1× `mx.compile` per block) | M |
| B | mhc (Sinkhorn manifold mixer) | nanochat torch ref + `tokenbender/mHC...` PyTorch | **PORT-FUSED** (compile 20-iter Sinkhorn loop) | M |
| C | ngram_hash (additive) | already in `cppmega_mlx/nn/ngram_hash.py` | **DONE** | — |
| D | MTP K=2 head (FastMTP) | nanochat `mtp.py` + cppmega `fastmtp_layer.py` (β=0.6 / λ=0.3) | **PORT-FUSED** with shared-weight aliasing | L |
| E | FIM PSM/SPM transform | nanochat `fim.py` (CPU only) | **PORT-FUSED** (cpu) | S |
| F | iFIM instruction-aware | nanochat `experiments/sota_impl/ifim/ifim.py` (CPU only) | **PORT-FUSED** (cpu) | S |
| G | STP JEPA geodesic loss | nanochat `stp.py` (~100 LOC) | **PORT-FUSED** | S |
| H | MLA | **mlx-lm `models/deepseek_v3.py` + `mla.py`** | **VENDOR** | M |
| I | DSA (sparse top-k indexer) | **mlx-lm `models/deepseek_v32.py` + `glm_moe_dsa.py`** (~80 % cov) | **VENDOR** then **WRITE-METAL** if profile justifies | L |
| J | Mamba3 block | mlx-lm `models/mamba2.py` + `ssm.py` (Metal kernel for single-token decode) — Mamba3 trapezoidal/complex/MIMO is the patch | **PORT-FUSED** on top of mamba2 base | L |
| K | M2RNN mixer | already partial in `cppmega_mlx/nn/m2rnn.py`; mlx-xLSTM as cross-reference | **PORT-FUSED** | M |
| L | MoE with EP (single-host) | **mlx-lm `models/switch_layers.py`** (`SwitchGLU`, `QuantizedSwitchLinear`); ZMLX `gather_qmm_swiglu` for quantized | **VENDOR** + optional **FORK** ZMLX | M |
| M | Structure embeddings (5-channel) | already in `cppmega_mlx/nn/structure_embedding.py` | **DONE** | — |
| N | RoPE NTK / YaRN / Llama3 piecewise | **mlx-lm `models/rope_utils.py`** (`initialize_rope`) | **VENDOR** | S |
| O | Sliding-window + sinks | **mlx-lm `models/gpt_oss.py`** (uses `mx.fast.scaled_dot_product_attention(sinks=...)`) | **VENDOR** | S |
| P | Paged KV cache | **vllm-mlx `vllm_mlx/paged_cache.py` + `prefix_cache.py`** (Apache-2.0, vendorable) | **VENDOR** (inference only) | S–M |
| Q | KV-cache quantization | **mlx-lm `QuantizedKVCache`** + **arozanov/turboquant-mlx** Metal kernels | **VENDOR** | S |
| R | Sequence packing (cumulative doc-id mask) | **scasella/nanochat-mlx `dataloader.py`** + write doc-id mask ourselves | **VENDOR** packer + **PORT-FUSED** mask | S–M |
| S | Megatron .bin/.idx ingress | already in `cppmega_mlx/data/megatron_indexed.py` | **DONE** (verify multi-shard) | — |
| T | AdamW with fp32 master moments | `mlx.optimizers.AdamW` (built-in) | **VENDOR** | S |
| U | Lion optimizer | `mlx.optimizers.Lion` (built-in) + scasella `optim.py` Muon+AdamW MultiOptimizer | **VENDOR** | S |
| V | LoRA / DoRA / QLoRA / full FT | **mlx-lm `tuner/`** complete | **VENDOR** | S |
| W | q4 affine inference quantization | **mlx-lm `mlx_lm/quant/{awq,dwq,gptq,dynamic_quant}.py`** with Metal kernels | **VENDOR** (DWQ is the production path per Awni) | S |
| X | Tied / sharded embeddings | mlx core + `Embedding.as_linear()` (single-host); `mlx.nn.distributed` for sharded TP | **VENDOR** single-host; **PATTERN-ONLY** sharded | S |
| Y | Distributed (DP / TP / ZeRO-1) | mlx core `mx.distributed` + JACCL/ring; per-model `shard()` patterns in mlx-lm | **VENDOR** mlx core; **PATTERN-ONLY** ZeRO-1 | M–L |
| Z | Speculative decoding | **mlx-lm `speculative_generate_step`** (vanilla, greedy-match, NOT Leviathan) | **FORK** mlx-lm to add Leviathan rejection step + **PORT-FUSED** MTP self-spec via D | S / M / L |
| AA | BPE tokenizer with FIM/iFIM tokens | nanochat `tokenizer.json` + `cpp_tokenizer.py`, vocab=65536, IDs 2/3/4/5/6/7 | **VENDOR** (data-side) | S |
| BB | Linear + CE fusion (Liger-equivalent) | **mlx-lm `tuner/losses.py`** has chunked CE + Metal kernel + VJP | **VENDOR** pattern, extend for ignore_index | M |
| CC | SwiGLU fused | mlx core + `mx.compile`; **ZMLX `swiglu_mlp`** Metal kernel for decode | **PORT-FUSED** + optional **VENDOR** ZMLX | S |
| DD | Residual + RMSNorm | `mx.fast.rms_norm` + `mx.compile` | **VENDOR** (`mx.fast.rms_norm`) | S |
| EE | Sampling (top-k / top-p / temp / min-p / XTC) | **mlx-lm `sample_utils.py`** | **VENDOR** | S |
| FF | Prompt cache | **mlx-lm `LRUPromptCache` + `PromptTrie`** | **VENDOR** | S |
| GG | Async checkpoint save | none in MLX; pattern: `mx.eval(parameters)` + `ThreadPoolExecutor` worker | **PORT-FUSED** (small wrapper) | S |
| HH | Resumable training (RNG + cursor) | already in `cppmega_mlx/training/checkpoint.py` (saves cursor + grad-accum + RNG) | **DONE** (verify) | — |
| II | Speculative spec sampling correctness | mlx-lm uses greedy-match; mlx-community/speculative-decoding (Swift) is the only Leviathan-correct MLX impl | **FORK** mlx-lm to add Leviathan acceptance | M |

---

## 2. CUDA fusions — DO PORT vs SKIP

From the cppmega CUDA audit, 15 fusions exist. Here is the verdict for each on M4 Max:

**PORT (significant M4 delta, clear MLX path):**
- F1 — Linear+CE swap → **VENDOR** mlx-lm `tuner/losses.py` chunked CE pattern (Metal kernel + VJP).
- F4 — DSA fused indexer BMM → pure-MLX per-head loop (this is loop-reordering, not a CUDA-specific fusion).
- F9 — Mamba3 `torch.compile` regions (5.93× regional speedup on data-dependent A) → **PORT-FUSED** via `mx.compile`.
- F12 — FastMTP shared-block layer → **PORT-FUSED** with shared-weight aliasing (covered by row D above).

**PORT only after profiling justifies (large effort, M4-MSL needed):**
- F5 — MLA TileLang fused (block-scaled MXFP8 Q-K topk) → **WRITE-METAL** later. Strip the MXFP8 part; keep block-scaled-bf16 indexer + topk + sparse SDPA.
- F10 — CuTe DSL Mamba3 MIMO (Hopper WGMMA) → **WRITE-METAL** later. The MIMO recurrence and trapezoidal scan are the actual algorithm; the WGMMA is unportable.

**SKIP (Hopper/SM120-specific or zero M4 delta):**
- F3 — MTP native Hopper CE (PR #3345 transpose-NaN bug) — Hopper-only.
- F6 — CUTLASS MXFP8 GEMM (SM120/121 native MMA) — no M4 hardware.
- F7 — Grouped MXFP8 GEMM (SM120/121) — no M4 hardware.
- F8 — Quantized Muon momentum — <2 % delta; use `mx.quantize` if ever needed.
- F11 — Lemyx FA + KL warmup kernel — 1000-step warmup-only; pure-MLX is fine.
- F13 — Selective FP8 MoE patch — Megatron context manager; replicate as conditional matmul.
- F14 — FP8 activation checkpointing — unified memory makes activation memory less critical.
- F15 — GB10 SMEM preflight — sm_121 hardware constraint, no Metal equivalent.

---

## 3. Top-5 ranked: vendor immediately

Listed in execution order so the team can pick them up sequentially:

1. **mlx-lm `mlx_lm/models/nanochat.py` + `models/cache.py` + `tuner/`** (MIT). Foundation. The model file already implements the five critical nanochat features. Cache + tuner cover LoRA/DoRA/full FT and 12 cache classes including continuous batching.
2. **scasella/nanochat-mlx `nanochat_mlx/dataloader.py` + `optim.py`** (MIT). BOS-aligned best-fit packing dataloader (~119 LOC) and Muon+AdamW MultiOptimizer (~289 LOC) — precisely the pieces mlx-lm lacks. Direct copy.
3. **mlx-lm `mlx_lm/quant/{awq,dwq,gptq,dynamic_quant}.py` + `QuantizedKVCache`** (MIT). DWQ is the production-quality q4 path with Metal kernels.
4. **mlx core `mlx.nn.layers.distributed.{shard_linear, shard_inplace, sum_gradients}` + `mlx.optimizers.Muon` + per-model `shard()` patterns from mlx-lm `models/llama.py` + `models/deepseek_v3.py`** (MIT). Tensor parallelism for 7B+ already wired up.
5. **vllm-mlx `vllm_mlx/paged_cache.py` + `prefix_cache.py`** (Apache-2.0). 2.2 KLOC of standalone block-paged KV with COW, LRU, prefix-chain hashing, ref-counting. Pure Python, no upstream MLX-engine coupling. Drop in if/when contiguous KV runs out.

**Skip / pattern-only:** mlx-omni-server (stagnant), Apple `ml-recurrent-drafter` (Python repo claims MLX but ships no MLX path), `alxndrTL/mamba.py` (stale, mlx-lm supersedes), `stockeh/mlx-optimizers` (Muon now in mlx core), mlx-vlm (vision-coupled).

**Build ourselves (no good MLX impl exists):**
- engram per-block branch with gating + grouped causal SiLU conv
- mhc Sinkhorn manifold mixer
- MTP K=2 recursive shared-block head with shared-weight aliasing
- FIM / iFIM / STP transforms (the Sun-2025 iFIM, the Bavarian-2022 FIM, the Huang/LeCun-2026 STP)
- Mamba3 trapezoidal/complex/MIMO scan on top of mlx-lm Mamba2 base
- Cumulative-doc-id attention mask generator for packed sequences
- 8-bit / paged Adam (only if memory becomes the blocker)
- Linear+CE fusion that handles `ignore_index` and 65k–131k vocab without materializing logits
- Metal-fused paged-attention with sinks + sliding window combined (the OSS pieces don't combine all three)

---

## 4. Anti-port playbook — do NOT carry torch idioms into MLX

15 specific don'ts when porting torch code to MLX. Most are confirmed by the agent reports; a few come from the existing `cppmega_mlx/runtime/` lint rules.

1. **`torch.repeat_interleave(K, n_groups, dim=1)` for GQA → don't materialize.** `mx.fast.scaled_dot_product_attention` accepts `n_q_heads % n_kv_heads == 0` directly; it does the repeat internally. Materializing doubles KV memory.
2. **`F.cross_entropy(reduction='none', ignore_index=-100)` → MLX has no ignore_index.** Implement via boolean mask: `(loss * mask).sum() / mx.maximum(mask.sum(), 1)`.
3. **`torch.nn.Linear(bias=False)` → `mlx.nn.Linear(in, out, bias=False)`.** Both default to `bias=True`. Be explicit.
4. **Don't manual-softmax in fp32 around `mx.fast.scaled_dot_product_attention`.** It does the fp32 softmax internally regardless of input precision; wrapping is a no-op cost.
5. **Don't bypass `mx.fast.rms_norm` with manual `mean(x*x)`.** The fast op is fused, fp32-internal, and ~3-5× faster on Metal. Manual versions break mixed-precision.
6. **Don't `mx.array(scalar)` in a hot path.** Allocates and promotes to fp32 by default; use Python floats inside compiled functions — they are captured as constants. The `tools/lint_mlx.py` check enforces this.
7. **Don't compile around `.eval()` calls.** `mx.eval()` inside a `mx.compile` boundary kills the graph. Move all `mx.eval` outside the compiled step (one call at iteration boundary is the mlx-lm pattern).
8. **Don't recreate `mx.compile`d functions inside the loop.** `@partial(mx.compile, inputs=state, outputs=state)` once at module init. A `lambda` inside `for step in steps:` recompiles every iteration.
9. **Don't depend on `torch.nn.functional.scaled_dot_product_attention(is_causal=True, attn_mask=mask)`.** MLX's `mx.fast.SDPA` has `mask=` (string `"causal"` / array / None) — the two cannot be combined; merge into one mask yourself.
10. **Don't forget `mlx.nn.Conv*d` is NHWC, not NCHW.** Mamba3 depthwise conv channel-axis differs by dim convention.
11. **Don't use `mx.argsort` for top-k.** `mx.argpartition(k)` is asymptotically cheaper. DSA top-2048 over 100 k tokens is the canonical case.
12. **Don't `tree_map` outside `mx.compile`-captured state.** When accumulating gradients, the prev-grad pytree must be in `inputs=state` of the compile decorator; if you treat it as a closure over Python state, MLX silently reverts to eager.
13. **Don't share `nn.Linear` weights via deep copy.** Use direct attribute aliasing (`model.lm_head.weight = model.tok_embed.weight`). MLX module tree walks recognize the shared parameter and AdamW state stays single. This is critical for FastMTP shared-weight K-loop.
14. **Don't `mx.eval(grad)` between forward and backward.** Forces a graph cut, removes fusion opportunity, doubles activation memory. Let `nn.value_and_grad` produce the grad pytree and feed it directly to `optimizer.update`.
15. **Don't promote bool masks to fp32.** `mx.fast.SDPA` accepts bool mask directly (treats `False` as `-inf`). If you `mask.astype(mx.float32) * -1e9` you waste bandwidth and lose precision.

Bonus rules from the agent surveys:

16. **Don't backprop through top-k argmax indices.** Use straight-through estimator with `mx.stop_gradient`. Affects DSA / MoBA / Token Recycling.
17. **Don't materialize a `[B, T_q, T_kv]` mask for sparse attention.** Use sparse gather + indexed SDPA call.
18. **Don't quantize KV during training.** Group sizes >64 cause long-context regressions; group_size=64 is the correct default for inference q4-KV.
19. **Don't clamp epsilon to 0 in RMSNorm.** Even though `mx.fast.rms_norm` is fp32-internal, eps=0 hits inf for masked-zero rows.
20. **Don't put paged KV in a compiled training step.** Dynamic shapes → recompile churn. Paged path is strictly inference.

---

## 5. Speculative decoding — three rows in one

Per the spec-decode agent survey:

- **Vanilla draft-model spec** is in mlx-lm but uses **greedy-match accept**, not Leviathan rejection sampling. For temperature > 0, output distribution diverges from the target. **FORK mlx-lm `speculative_generate_step` to add the Leviathan accept step** (`u ~ Uniform(0,1); accept iff u < min(1, p_target/p_draft)`; on reject sample from `(p_target − p_draft)_+`). ~50 LOC on top of existing mlx-lm code. Phase 1.
- **Token Recycling** (arXiv 2408.08696) — train-free n-gram tree, reuses our tree-attention infrastructure. ~2× reported. **PORT-FUSED.** Phase 2 — strong fit because no draft model means no memory contention on unified memory.
- **MTP self-speculation** (FastMTP, arXiv 2509.18362) — once row D lands, the K=2 head doubles as a draft generator with verify in the same compile. Phase 3.

EAGLE-2 / Medusa / Hydra remain **PATTERN-ONLY** until we are sure phases 1-3 underwhelm; each adds 1–5 KLOC and a separate training pipeline.

---

## 6. Hardware constraints from the precision research

Restating, for completeness when this doc is read in isolation:

- **bf16** on M4 Max is supported but slower than fp16; train in bf16 for CUDA parity, accept the throughput cost.
- **fp16** mixed precision is faster on M4 but breaks CUDA parity and adds loss-scaling complexity. Skip for default training; reconsider for inference-only acceleration.
- **fp8 / mxfp4 / mxfp8 / nvfp4** — no M4 hardware support, no MLX dtype, no roadmap before M5+. **Skip everything fp8-family.**
- **q4 affine g64** for inference is 2.5× faster decode than bf16 on M4 Max (mlx-lm bench: Qwen3-4B q5 at 110 tok/s vs bf16 at 52 tok/s).
- **KV q4 with `--quantized-kv-start 256`** is the recommended inference default on long context.
- **JACCL** RDMA over Thunderbolt 5 needs macOS ≥ 26.2 + M3 Ultra/M4 Pro/Max on **both** ends; ring backend (TB4 or mismatched chips) is the fallback.

---

## 7. GB10 status snapshot (2026-05-02)

- Local cppmega checkout is 31 commits behind gb10. Themes: FA3/FA4 build-wheel infra + Mamba3 wave28-32 stage2 work.
- Active uncommitted work on gb10: **wave44A MXFP8 acceptance harness** (`mxfp8_sidecar_refs.py`, harness script, ~17 file edits). MXFP8 is Hopper/SM120-only; it does not affect MLX scope.
- Production gold: europe BF16 **289 TFLOP/s, 29.2 % MFU**; bench3 FP8 tensorwise **268 TFLOP/s, 27.1 %**. These are **CUDA receipts**, not parity targets for MLX.
- No `fim`, `ifim`, `stp`, `eagle`, `medusa`, `mtp_draft`, `speculative` files anywhere in cppmega or gb10. Spec decoding on MLX is greenfield.
- No MLX/macOS/Apple-specific code paths in the gb10 tree.

---

## 8. Read-the-source pointers

The following files are referenced repeatedly and are worth reading once before starting Stream H or Stream I:

- `mlx_lm/models/nanochat.py` — upstream-Karpathy nanochat surface in MLX. Vendor.
- `mlx_lm/models/deepseek_v3.py` + `mla.py` — MLA reference. Vendor.
- `mlx_lm/models/deepseek_v32.py` + `glm_moe_dsa.py` — DSA reference. Vendor.
- `mlx_lm/models/gpt_oss.py` — sinks + sliding window pattern. Vendor.
- `mlx_lm/models/switch_layers.py` — single-host MoE dispatch. Vendor.
- `mlx_lm/models/cache.py` — 12 cache classes including `QuantizedKVCache`. Vendor selectively.
- `mlx_lm/quant/{awq,dwq,gptq,dynamic_quant}.py` — DWQ is the production q4 path. Vendor.
- `mlx_lm/tuner/{trainer,losses,datasets,callbacks}.py` — training loop with state-capture pattern `state = [model.state, optimizer.state, mx.random.state]`. Vendor pattern.
- `mlx_lm/sample_utils.py` — sampler factory with top-k/top-p/min-p/XTC. Vendor.
- `nanochat_mlx/dataloader.py` (scasella fork) — BOS-aligned best-fit packer. Vendor.
- `nanochat_mlx/optim.py` (scasella fork) — Muon+AdamW MultiOptimizer matching nanochat's setup_optimizer. Vendor.
- `vllm_mlx/paged_cache.py` + `prefix_cache.py` — Apache-2.0 paged KV. Vendor (inference only).
- `nanochat/{engram,mhc,mtp,fim,stp}.py` — torch references for the six greenfield features.
- `nanochat/experiments/sota_impl/ifim/ifim.py` — iFIM reference.
- `cppmega/megatron/fastmtp_layer.py` — K=2 default, β=0.6 / λ=0.3 GB10 baseline.

---

## 9. References

Same set as the master plan, plus the following surfaced by this round of research:

- mlx-lm releases (latest 0.31.3, 2026-04-22): https://github.com/ml-explore/mlx-lm/releases
- mlx-lm `mlx_lm/models/nanochat.py`: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nanochat.py
- scasella/nanochat-mlx: https://github.com/scasella/nanochat-mlx
- vllm-mlx: https://github.com/waybarrios/vllm-mlx
- ZMLX: https://github.com/Hmbown/ZMLX
- mlx-mfa: https://pypi.org/project/mlx-mfa/
- arozanov/turboquant-mlx: https://github.com/arozanov/turboquant-mlx
- vllm-metal: https://github.com/vllm-project/vllm-metal
- humanrouter/ddtree-mlx: https://github.com/humanrouter/ddtree-mlx
- bstnxbt/dflash-mlx: https://github.com/bstnxbt/dflash-mlx
- alexziskind1/mlx-jaccl-cluster: https://github.com/alexziskind1/mlx-jaccl-cluster
- Token Recycling (arXiv 2408.08696): https://arxiv.org/abs/2408.08696
- Cut Cross-Entropy (arXiv 2411.09009): https://arxiv.org/abs/2411.09009
- DeepSeek-V3.2 / DSA (arXiv 2512.02556): https://arxiv.org/pdf/2512.02556
- Mamba-3 (arXiv 2603.15569): https://arxiv.org/abs/2603.15569
- M2RNN (arXiv 2603.14360): https://arxiv.org/abs/2603.14360
- Hyper-Connections / mHC (arXiv 2512.24880): https://arxiv.org/abs/2512.24880
- Engram (arXiv 2601.07372): https://arxiv.org/abs/2601.07372
- FastMTP (arXiv 2509.18362): https://arxiv.org/abs/2509.18362
- iFIM (arXiv 2509.24637): https://arxiv.org/abs/2509.24637
- STP / Semantic Tube Prediction (arXiv 2602.22617): https://arxiv.org/abs/2602.22617
