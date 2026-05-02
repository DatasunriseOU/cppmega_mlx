# cppmega.mlx — Master Port Plan & Architecture Review

**Date:** 2026-05-01
**Scope:** Conservative MLX port planning for Apple Silicon; not a production-parity or M4-vs-GB10 parity claim
**Sibling repos audited:**
- `/Volumes/external/sources/nanochat` (origin / "MegaCpp" fork, commit `4acb8af1`)
- `/Volumes/external/sources/cppmega` (CUDA / Megatron-LM port, GB10 + H200 tuned)
- `/Volumes/external/sources/cppmega.mlx` (this repo)

This document is a planning synthesis from parallel research lanes (local-code audits plus external research streams). It contains the executive review, architectural decisions, risk register, and a 200-step plan organized into 10 work-streams that can largely run in parallel. It is not a completion receipt.

**Wave14 guardrails:** no M4-vs-GB10 parity claim without matched rows; no trainable Metal-kernel adoption without pure-MLX fallback, parity, profiling, and VJP/JVP coverage; no distributed Megatron parity claim; no full NAM56R readiness claim; no fp8 training support claim on M4 unless current source and tests prove it.

---

## 0. Executive Review

### Where we actually are (cppmega.mlx today)

**Current local implementations and guarded scaffolds**
- Package layout (`cppmega_mlx/` with `config / data / kernels / models / nn / recipes / training`), public API boundaries enforced.
- NAM56R route-pattern metadata (`AEMEAEMEAEMR`, depth 52, DSA ranks `(1,2,3,5,6,7,9,10,11)`, MLA ranks `(0,16,32,48)`), not full NAM56R readiness.
- Causal self-attention via `mx.fast.scaled_dot_product_attention` with `nn.RoPE`, GQA-ready.
- Mamba3 reference block (depthwise conv + SSM + cache state).
- M2RNN mixer (chunked / full RNN scans, RoPE projection in state).
- MoE with top-k routing and selectable activation (`gelu / relu2 / swiglu`).
- **NgramHashEmbedding** (multi-head hash n-gram, unified table, seeded primes, MLX `int64` ops) is implemented locally; source offload, PP/MTP checkpoint semantics, and full feature parity are not claimed.
- Structure side-channels (5-component embedding with selective masking).
- Standalone fail-closed Megatron-indexed `.idx/.bin` reader seam + Parquet + NPZ datasets.
- Training loop (eager + `mx.compile`-based `CompiledPretrainingStep`).
- Single-file safetensors checkpointing with metadata; distributed/sharded checkpoint metadata is not implemented.
- Profiling harness (`profile_step` context, `HotspotEvidence`, JSON metrics).
- 573 collected tests across 29 tracked test files, including parity anchors, kernel gates, training smoke, package-export contracts.

**Scaffolded / placeholder**
- DSA mode is dense causal (no sparse top-k indexer yet).
- Metal kernel seam (`kernels/metal_ops.py`) holds optional research kernels with pure-MLX fallbacks; `TrainingKernelStatus.differentiable=False` by design.
- Megatron `.idx` parser rejects ngram sidecar metadata (n-gram is derived in-model instead).

**Missing / partial / not integrated (the real porting backlog)**
- **Features**: FIM/iFIM have local fail-closed CPU token transforms only; `engram` and `mhc` have standalone MLX modules only; MTP has local M0.5 training-side coverage only; STP has a local opt-in deterministic helper plus TinyLM smoke coverage. These slices are not wired into NAM56R integration, do not prove CUDA/Megatron parity, and do not close Stream H.
- **Distributed**: `mx.distributed` (ring / JACCL), multi-Mac ZeRO-style sharding.
- **Sharded checkpoints** (`model.safetensors.index.json`).
- **Sequence packing** (cumulative-doc-id attention mask).
- **Tokenizer**: M0.1 is closed. The vendored GB10 BPE artifact and special-token contract exist for vocab=65536 plus deployed id7=`<CODE_START>` and id45=`<FIM_INSTRUCTION>`, and MLX vendors the nanochat heuristic decoder to match the CUDA reference decode path.
- **Inference / generation**: no KV cache class, no sampler, no MTP-aware or FIM-aware decoding.
- **Quantization**: bf16 only; no `mx.quantize` integration, no q4 inference path.
- **Structural parity anchors, not CUDA tensor parity**: existing `tests/test_cppmega_parity_anchors.py` checks NAM56R route/layer constants, DSA/MLA layer derivation, vocab/MoE anchors, fail-closed parity wording, and optional sibling cppmega source-anchor presence when that checkout exists. The source-file existence check is only one guard; the test is broader than file-presence coverage, but it still does not compare CUDA golden tensors or prove numerical agreement.

### What ../cppmega (CUDA) gives us as a reference target, not MLX status

A heavily-engineered Megatron-LM port:
- 50+ unconditional / conditional Megatron patches (Linear+CE swap, MTP Liger fused CE, DSA fused indexer BMM, `torch.compile` on Mamba3 data-dependent-A, GB10 sm_121 SMEM preflight).
- Custom CUDA: `cutlass_mxfp8_gemm`, `grouped_mxfp8_gemm`, `quantized_muon_momentum`. CUTLASS Block-Scaled MMA with E8M0 scales for GB10.
- TileLang sparse-MLA forward+backward (FP8), CuTe DSL Mamba3 MIMO.
- CUDA/H200 reference receipts include BF16 289 TFLOP/s, 29.2% MFU on H200 PP=1 EP=4 MBS=8, and FP8 268 TFLOP/s on bench3 with Liger reduction-mean workaround. These are sibling-repo CUDA receipts, not cppmega.mlx throughput, FP8, GB10, or production-readiness evidence.
- Features: **ngram_hash** done; **engram** exists as a standalone MLX per-block branch module but is not integrated into NAM56R; **mHC** config-only; **MTP** fused (Hopper CE + Liger); **FIM/iFIM** flag-only; **STP** exists locally only as an opt-in deterministic MLX helper, with nanochat still serving as the source spec; structure embedding done.

### What ../nanochat (origin) gives us as the spec

Full-stack research-grade LLM (40-45K LOC):
- GQA + RMSNorm + RoPE (standard / llama3 piecewise / YaRN), SwiGLU/ReLU², sliding-window, attention sinks, gated attention, optional QK-norm/clip, optional MLA (DeepSeek-V3 style), optional Mamba3 / M2RNN / MoBA / DeepEP / MoDification / GateSkip.
- All seven features implemented in clean Torch:
  - `engram.py` (DeepSeek arXiv 2601.07372, Jan 2026): per-block branch with avg-pool n-gram + optional sigmoid gating + grouped causal SiLU conv.
  - `mhc.py` (Manifold-Branch Mixer, Sinkhorn-Knopp routing ported from MaxText).
  - `ngram_hash.py` (multi-head hash n-gram, BLT 2412.09871 + DeepSeek 2601.07372).
  - `mtp.py` (FastMTP arXiv 2509.18362-style recursive shared-block, β-decayed loss).
  - `fim.py` (Bavarian 2207.14255 PSM/SPM, plus AST-FIM 2506.00204).
  - `ifim.py` (Sun et al. arXiv 2509.24637, Sep 2025: instruction-aware FIM; paper/source proposal only, not the deployed GB10 token-id contract).
  - `stp.py` (Huang/LeCun arXiv 2602.22617, Feb 2026: JEPA geodesic regularization, ~100 LOC).
- Speculative decoding (acceptance-rejection), EAGLE-2 draft head.
- Tokenizer/FIM intent is a source reference only: live local/GB10 artifacts use the deployed M0.1 contract documented below (`id7=<CODE_START>`, id45=`<FIM_INSTRUCTION>`), and M0.1 decode parity is covered by the vendored nanochat heuristic decoder rather than HF reversible round-trip decode.
- Training: pretrain + midtrain + SFT + RL via Muon/AdamW, FA3 (CUDA) / Pallas (TPU) / SDPA fallback.
- Inference: contiguous KV (`engine.py`) + paged KV scheduler (`serving.py`).

### Critical research findings for MLX on M4 Max

**Precision (current policy):**
- bf16 weights + bf16 activations + **fp32 AdamW master moments**. No loss scaling needed (bf16 has fp32 exponent range). Mirrors the CUDA port and avoids parity drift.
- fp16 mixed-precision is a benchmark question, not a default: it can break parity with the CUDA reference, so use it only if repo-local matched runs prove a worthwhile speedup.
- **No fp8 / mxfp* / nvfp* training support is claimed on M4 Max.** Current repo source and tests do not prove those training paths. Keep fp8-family work deferred or inference/quantization-only until local source, tests, and hardware evidence justify a narrower claim.
- **Inference**: q4 affine, group_size=64, embed/lm_head at q8, and KV q4 remain inference-planning candidates. External mlx-lm rows are pattern evidence only until reproduced with repo-local matched shapes.

**Kernels & fusion:**
- Standardize on `mx.fast.scaled_dot_product_attention` for supported MHA/GQA/MQA, causal masks, additive masks, and attention sinks; pin exact MLX behavior to the installed version before relying on version-specific optimizations.
- `mx.fast.rope` with precomputed `freqs` array for NTK / YaRN / llama3 piecewise.
- `mx.fast.rms_norm` / `layer_norm` — fused, mixed-precision-aware. Don't cast around them.
- Use `mx.compile(inputs=..., outputs=...)` around training step + decode step where state capture is required. Prefer Python scalars over `mx.array(scalar)` (silent fp32 promotion). Avoid `shapeless=True` unless shape-dependent code is audited; bucket sequence lengths to powers of 2.
- **Custom Metal candidates only after profiling** (`mx.fast.metal_kernel`): fused SwiGLU (gate+up+silu), fused residual+RMSNorm, paged-attention with sinks+sliding-window, KV-cache quantization (TurboQuant Hadamard + codebook). Treat ZMLX, mlx-mfa, and TurboQuant as pattern references; do not adopt a training-path kernel without pure-MLX fallback, parity coverage, hotspot evidence, and VJP/JVP support.
- Build `metal_kernel` objects once at import; never JIT in the hot path.
- `mx.async_eval` can pipeline graph construction with execution; measure any training-loop gain locally before treating it as a performance assumption.
- Profile via `mx.metal.start_capture` → Xcode `.gputrace` debugger.

**Distributed (future pattern source, not current capability):**
- `mx.distributed` backends include ring, JACCL, MPI, and CUDA-only NCCL, but this repo has not implemented distributed MLX training or Megatron-parity distributed semantics.
- JACCL/TB5 is a future candidate backend and must be measured locally before any capability or performance claim.
- External examples such as mlxgpt.com training nanochat-d20 on 2× Mac Mini M4 Pro over TB5 are reference context, not cppmega.mlx receipts.
- Candidate MLX distributed primitives include `nn.AllToShardedLinear` / `nn.ShardedToAllLinear` and `nn.average_gradients`; do not treat them as cppmega.mlx TP/DP parity until wired and tested locally.
- **No native ZeRO/FSDP** — must hand-roll ZeRO-1-equivalent (shard optimizer state across ranks, all-gather params for fwd/bwd, reduce-scatter grads).

**Data & training stack:**
- Megatron `.bin/.idx` ingress is already wired locally through the standalone fail-closed `cppmega_mlx.data.megatron_indexed` seam. Remaining work is multi-shard/source-converter side-channel preservation and scale validation, not a claim of distributed Megatron runtime parity.
- Sequence packing (concat-with-EOS + cumulative-doc-id mask) must be reimplemented; nanochat-mlx fork has a reference.
- `mlx-data` is C++ thread-pool, not multi-process; GIL-bound for Python transforms. Practical pattern: PyTorch DataLoader → NumPy → `mx.array` on the unified-memory device (zero-copy).
- AdamW is fine for ≤3B; Lion for 7B+ on 64 GB Macs (1× state vs Adam's 2×). 8-bit Adam doesn't exist for MLX.
- Optimizer state is at fp32 by default. For 7B bf16 weights: weights 14 GB + grads 14 GB + Adam state 56 GB ≈ 84 GB → 128 GB Mac is tight; use `--grad-checkpoint` and Lion.
- Future checkpoint target: sharded `.safetensors` with an index manifest, plus `optimizer.safetensors` and `train_state.json` (step, epoch, rng, dataloader cursor, `mx.random.state` b64). Current repo status remains single-file local safetensors, not distributed/sharded Megatron restore.
- Memory knobs every training entrypoint must set: `mx.set_wired_limit(0.7 * total)`, `mx.metal.set_memory_limit(0.85 * total)`, `mx.clear_cache()` every N steps, `mx.async_eval` for pipelining. Wiring >75% has caused kernel panics (mlx-lm #883).
- Logging: W&B remains a future local integration; mlx-lm's `--report-to wandb` is a reference pattern, not a cppmega.mlx receipt.
- Parity tolerances: bf16 single matmul `rtol=1e-3, atol=1e-1`; chained `rtol=1e-2, atol=1e-1`; attention/RMSNorm `atol=5e-2`; full-step grad `atol=1e-1`. Match PyTorch's documented bf16 thresholds.

### Planning decisions (fail-closed, not production commitments)

| Decision | Choice | Why |
|---|---|---|
| Training precision | bf16 + fp32 AdamW master | parity with CUDA reference, no loss-scaling complexity |
| Inference precision (candidate) | q4 affine g64; embed/lm_head q8; KV q4 | external pattern only until repo-local matched-shape rows exist |
| FP8 / mxfp4 / mxfp8 / nvfp4 training | deferred | no current M4 training support claim in this repo; revisit only with source/test/hardware proof |
| Optimizer (default) | AdamW (≤3B) / Lion (7B+) | bnb 8-bit Adam absent; Lion halves state cost |
| Distributed backend | future JACCL/ring evaluation | pattern source only until repo-local distributed training is implemented and tested |
| Sharding strategy | future ZeRO-1-style optimizer-state sharding | no native FSDP/ZeRO in MLX; no distributed Megatron parity claim |
| Data format | use existing standalone Megatron `.bin/.idx` seam | preserves token ingress locally; side-channel preservation and multi-shard scale remain open |
| Sequence packing | concat-with-EOS + cumulative doc-id, attn mask via `mx.cumsum` | nanochat-mlx reference exists |
| Kernels — defaults | `mx.fast.SDPA` / `rope` / `rms_norm` everywhere | fused, mixed-precision aware, supported |
| Kernels — custom Metal | candidates only after profiling, pure-MLX fallback, parity, hotspot evidence, and VJP/JVP if training-side | maintenance cost is real; ZMLX/mlx-mfa/TurboQuant are pattern references |
| Checkpoints | sharded `.safetensors` + index manifest + `optimizer.safetensors` + `train_state.json` | future target; fills RNG/cursor gap without claiming current sharded restore |
| Tokenizer | M0.1 closed on deployed GB10 BPE artifact + nanochat heuristic decoder parity | assert vocab/id mapping at model load; future tokenizer work is FIM/iFIM/AST-FIM integration, not an M0.1 blocker |
| Parity anchors | upgrade `tests/test_cppmega_parity_anchors.py` to numerical | current anchors are structural/doc/source-presence checks, not tensor parity |
| Bench/CI gate | future regression threshold after baselines exist | do not block or claim throughput until local baselines are committed |

---

## 0.4 Buy-vs-build addendum

`docs/mlx_buy_vs_build.md` (added 2026-05-02) is the per-feature decision matrix derived from an 8-agent research pass. **Read it before starting any Stream H/I/G work.** Key findings:

- **mlx-lm already ships `mlx_lm/models/nanochat.py`** plus DeepSeek-V3 (MLA), DeepSeek-V3.2 (DSA), GPT-OSS (sinks + sliding window), Mamba2 + `ssm.py` Metal kernel, switch-routing MoE, AWQ/DWQ/GPTQ, QuantizedKVCache, full LoRA/DoRA/QLoRA tuner. **Vendor first.**
- **scasella/nanochat-mlx** ships exactly the two pieces mlx-lm lacks: Muon+AdamW MultiOptimizer (~289 LOC) + BOS-aligned best-fit packer (~119 LOC). Vendor.
- **vllm-mlx `paged_cache.py`** (Apache-2.0) is standalone-vendorable for paged KV inference.
- **ZMLX** (`swiglu_mlp`, `rmsnorm`) provides Python-MSL kernels for fused decode paths; vendor.
- **arozanov/turboquant-mlx** (Apache-2.0) ships the Hadamard + Lloyd-Max KV-quant Metal kernels we planned for Stream G.
- **Greenfield work that survives**: engram per-block branch, mhc Sinkhorn mixer, MTP K=2 head with shared-weight aliasing, FIM/iFIM/STP transforms, Mamba3 trapezoidal/MIMO patch on top of mlx-lm Mamba2, doc-id mask generator, 8-bit Adam (only if memory blocks).
- **Skip Hopper-only fusions**: CUTLASS MXFP8, Grouped MXFP8, MTP native Hopper CE, FP8 activations, GB10 SMEM preflight.
- **Speculative decoding is greenfield on GB10** (CUDA accepts `is_spec_decode` and forwards to upstream Megatron unused). MLX-side work: fork mlx-lm's `speculative_generate_step` to add Leviathan rejection (mlx-lm uses greedy-match, not Leviathan), then Token Recycling (Phase 2), then MTP self-spec via the K=2 head (Phase 3).

The buy-vs-build addendum also contains a 20-rule anti-port playbook — the things to **not** do when carrying torch into MLX (don't materialize GQA repeats, don't manual-softmax around `mx.fast.SDPA`, don't `mx.array(scalar)` in hot paths, don't deep-copy shared weights, etc.). Lint rules in `tools/lint_mlx.py` enforce a subset.

---

## 0.5 First milestone (M0): `local_gb10_quarter` end-to-end on a single Mac

Before fanning out across all 10 streams, prove the smallest viable path on the resolved mini target. This is a single, sequenced, ~3–5-week milestone that gates the rest of the plan.

**Goal**: load `local_gb10_quarter` config (depth=13, hidden=3584, FFN=18944, 28 heads, head_dim=128, vocab=65536, MTP=2, AEMEAEMEAEMR pattern); run a single bf16 training step end-to-end on a single Mac; and produce a numerical parity row against a one-shot CUDA forward at the same seed within the documented bf16 tolerances.

**Gate set** (must all be green before scaling effort):
- M0.1 — **CLOSED**. Tokenizer: vendor the deployed GB10 tokenizer artifact with vocab=65536 and the reserved ID contract (id 2=BOS, 3=EOT/EOS, 4=FIM_PREFIX, 5=FIM_MIDDLE, 6=FIM_SUFFIX, 7=CODE_START, 45=FIM_INSTRUCTION, 46=SPACE, 47=NL). The wrapper normalizes whitespace runs at encode (`[\r\n]+`->`<NL>`, `[ \t]+`->`<SPACE>`) and decode is plain token concat with sentinel substitution; this fixes the BPE-split decode bug (e.g., `sum`->`s`,`u`,`m`->`s u m`) and gives byte-exact round-trip for inputs without multi-char whitespace runs. The vendored nanochat heuristic decoder has been retired in favor of the explicit-token approach, which matches the CUDA-side `nanochat/cpp_tokenizer.py` decode behavior. Do not reopen this gate over the deployed HF `decoder=null` artifact's non-reversible `decode(encode(text))` behavior.
- M0.2 — Model factory entry for `local_gb10_quarter` in `cppmega_mlx/recipes/model_factory.py`. With M0.1 closed, the default profile is unblocked; acceptance remains the profile contract plus build → forward closure on shape `(B=1, T=512)` returning finite logits with config schema validation rejecting invalid combos.
- M0.3 — **Random-init seed-matched forward parity** (no warm-start, no CUDA weight import for M0): construct MLX model and CUDA reference model with the same `local_gb10_quarter` config; seed both deterministically; compare logits within `rtol=1e-2, atol=1e-1` on the fixed `B=1,T=512,seed=3003` input batch (`tokens_sha256=c645ca4053e5206dcbe58c13aa26f4a9e56c5aa2aee90a4d4778bbc9d9c33549`). Current gate state is fail-closed: no real CUDA logits artifact exists at `bench/parity/cuda/m03_local_gb10_quarter_seed3003_logits.json`, so `scripts/m03_forward_parity_manifest.py` must refuse the default missing artifact and keep `m0_3_closed=false`. A metadata-valid CUDA artifact only reaches `artifact_preflight_status=valid_not_evaluated`; it is not acceptance until a separate numerical harness compares the full logits tensor (`shape=[1,512,65536]`, `numel=33554432`) against MLX and records pass/fail parity.
- M0.4 — One training step in bf16 (loss + backward + optimizer.update). No NaNs. Loss decrease over 100 steps on the target local parquet sample. **Data source**: local ignored `data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet` (~492 MB total of GB10 validation shards when present locally; not committed to git). Current receipts may cover tiny-model/full-parquet plumbing only; full M0.4 remains blocked until the `local_gb10_quarter` path runs this gate. No GB10 scp or full-corpus prep needed for M0.
- M0.5 — MTP K=2 head wired with β=0.6 / λ=0.3; per-depth losses tracked. MTP-disabled inference path also returns sane logits. Current local coverage is bounded to MLX training-side loss semantics and fails closed on Hopper/Liger fused-CE parity until a hardware receipt exists.
- M0.6 — Memory math validated on the actual dev box (Mac Studio M4 Max 128 GB): peak unified-memory < 75% of installed RAM with `--grad-checkpoint` and AdamW.
- M0.7 — Resumable training: save mid-run, kill process, reload, identical loss continues for ≥100 steps with RNG state and dataloader cursor preserved. Current receipts are TinyLM/HybridTinyLM-scoped only; full M0.7 remains fail-closed until `local_gb10_quarter` resumes on the target parquet lane after the M0.4 full-model training gate is proven.

**M0 data note**: full-corpus pretraining is a post-M0 concern. When that lands, options are (a) `scp` materialized `.bin/.idx` shards from GB10's `/home/dave/cppmega-root/data/megatron/clang_semantic_4k_v10_train`, or (b) port `cppmega/scripts/data/prepare_*` to run on Mac. Resolve at that time.

**2nd Mac connection trigger (revised)**: ~1–2 weeks after M0 work starts, when M0.2/M0.3 are green and there's a real working baseline to extend. Earlier is debug surface without payoff; later (waiting for step ~100) leaves Stream F starting cold.

**Mapping to plan steps** (M0 is the first ~35 steps, executed sequentially): A1, A5, A7, A8, A9, A11, A13, A15 → B21, B22, B23, B32, B33, B34 → C41, C50, C58 → D61, D62, D63, D78 → E81, E83, E84, E85, E86, E87, E89, E91 → H151, H152, H153, H154 → J187. Everything else (Streams F, I, full G, full H, the rest of A/B/C/D/E) waits behind M0 acceptance.

**M0 estimate**: 3–5 weeks for one engineer focused. Add a second engineer only after M0 ships — at that point, each stream can fan out independently against a known-good base.

---

## 1. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MLX bf16 path silently slower than fp16 → CUDA-parity-vs-throughput tradeoff | High | Medium | Benchmark both in stream G; if fp16 wins ≥20% on M4, add as opt-in |
| `mx.compile` re-compiles on shape change | High | Medium | Bucket seq_len to powers of 2; document anti-patterns |
| Wired-memory kernel panic on KV overrun | Medium | High | Always set `mx.metal.set_memory_limit` + `set_wired_limit` ≤ 0.7 |
| Sinkhorn (mHC) numerical instability in bf16 | Medium | Medium | Force fp32 inside Sinkhorn iters (mirror nanochat) |
| MTP recursive shared-block breaks `mx.compile` | Medium | Medium | Use static-shape roll-and-mask trick; static K |
| FIM/iFIM token-id collision on tokenizer reload | Medium | High | Assert at model load; reserve range [3..7] in tokenizer training |
| Distributed parity drift across ranks | Medium | High | Determinism harness with seeded `mx.random.state` per rank |
| MacBook thermal throttling masquerading as code regression | High | Low | Run benches on Mac Studio only; thermal monitor in CI |
| `mlx-data` Python-transform GIL bottleneck | Medium | Medium | PyTorch DataLoader bridge fallback; keep transforms in C++ ops |
| MLX version churn (0.30 → 0.31 → 0.32) breaking custom Metal | High | Medium | Pin MLX in pyproject; add MLX-version smoke test in CI |
| FP8 / mxfp4 expectations from CUDA team | High | Low | Document explicit refusal; phase plan with E gated on M5+ |
| Megatron `.idx` format drift (Megatron-Core changes) | Low | Medium | Version-pin spec; golden-file test |
| Apple Silicon ANE temptation for training | Low | Low | Document explicitly: ANE is CoreML-inference only, not exposed to MLX |

---

## 2. The 200-Step Plan

10 streams of 20 steps each. Streams A → J, each step is sized at roughly ½–1 day of focused work for one engineer. Most steps within a stream depend on prior steps in the same stream, but cross-stream dependencies are minimized so multiple engineers can fork off.

### Dependency overview
- A (foundation) feeds all
- B (kernels) feeds C, E, I
- C (architecture) feeds E, F, H
- D (data) feeds E
- E (training) feeds F, G
- H (features) is mostly orthogonal — fork anywhere after C/E
- I (inference) needs B, C, G
- F (distributed) needs E
- J (validation) gates merges; runs continuously

### Stream A — Foundation & Infrastructure (1–20)

1. Pin MLX version in `pyproject.toml`; add `__mlx_version__` runtime check; document required MLX APIs for the supported bf16/q4 paths. Do not imply M4 fp8-family training support.
2. Add pytest markers: `parity`, `kernel`, `training`, `bench`, `distributed`. Document conventions in `pyproject.toml`.
3. Set up benchmark database schema (`bench/baselines/*.json`, indexed by hardware + commit hash + dtype + batch + seq).
4. Set up parity database (per-tensor diff vs CUDA reference, JSON manifest).
5. Add `mx.metal.set_memory_limit(0.85 * total)` and `mx.set_wired_limit(0.7 * total)` to all training/inference entrypoints. Add `cppmega_mlx/runtime/memory.py` with helpers.
6. Add `mx.clear_cache()` cadence to `CompiledPretrainingStep` (every N steps, configurable).
7. Create `cppmega_mlx/runtime/env.py`: detects M4 vs M5, GPU core count, RAM tier, macOS version.
8. Add macOS file-descriptor `ulimit -n 65536` enforcement on startup; abort with clear message if too low.
9. Enforce `multiprocessing.set_start_method('spawn')` before any MLX import in launcher scripts.
10. Add thermal-throttling detector parsing `powermetrics`; auto-pause training if package power-throttled.
11. Document JACCL prerequisites in `docs/system_requirements.md` only after local verification of the MLX version, macOS version, Thunderbolt/RDMA setup, and fallback behavior.
12. Document hardware tiers (M4 Max 64/128GB, Mac Studio M3 Ultra 192GB) in same doc with peak-memory math.
13. Add deterministic-seed harness: capture `mx.random.state`, `np.random`, Python `random`. Reproducibility test.
14. Add CI lint for `mx.array(<python_scalar>)` antipattern (forces fp32 promotion); flag in `tools/lint_mlx.py`.
15. Add `tests/conftest.py` shared fixtures: tiny model (depth=2, width=128), tiny tokenizer, golden batch.
16. Convert `tests/test_cppmega_parity_anchors.py` from structural/doc/source-presence checks to actual numerical anchor assertions (load CUDA-side golden tensors and compare with rtol/atol matrix).
17. Set up nightly bench cron (cron user + GitHub Action) running `scripts/bench_matrix.py` against committed baselines.
18. Add `scripts/profile_capture.py` wrapping `mx.metal.start_capture` / `stop_capture` for one-shot `.gputrace` capture.
19. Add `scripts/check_environment.py` (reports MLX, macOS, RAM, GPU cores, thermal, JACCL availability, file descriptors).
20. Document the contributor workflow in `CLAUDE.md`: use beads, run `pytest -m parity` before merge, capture `.gputrace` for any kernel-touching PR.

### Stream B — Core Ops & Kernels (21–40)

21. Audit all attention sites for `mx.fast.scaled_dot_product_attention` with proper `mask='causal'` / additive mask.
22. Add `mx.fast.rope` with `freqs=` parameter; precompute RoPE freq table once per model in `__init__`.
23. Promote `nn.RMSNorm` usages to verify they bottom out on `mx.fast.rms_norm` (it does, but verify no shadowing elsewhere).
24. Add SDPA sliding-window via additive-mask construction; benchmark vs unfused.
25. Add SDPA attention-sinks via `sinks=` parameter (1-D per-head scalar); test for GPT-OSS-style models.
26. Benchmark stock `mx.fast.SDPA` vs `mlx-mfa` for sliding-window window=256 on M4 Max (mlx-mfa shows 20× on M1; verify on M4); pick winner per-config.
27. If profiling identifies a stable hotspot, prototype fused SwiGLU Metal behind a pure-MLX fallback and parity tests before adoption.
28. If profiling identifies a stable hotspot, prototype fused residual + RMSNorm Metal behind a pure-MLX fallback and parity tests before adoption.
29. Keep custom Metal kernels out of the training path unless `mx.custom_function` plus `.vjp`/`.jvp` coverage, pure-MLX fallback, gradient parity, and hotspot evidence are all present.
30. Build `cppmega_mlx/metal/` directory with `.metal` MSL source files, headers, build manifest.
31. Pre-load all `metal_kernel` objects at import time of `cppmega_mlx.kernels`; never JIT in hot path.
32. Benchmark `mx.compile(shapeless=True)` vs default for fixed-shape training; document recompilation triggers.
33. Bucket dynamic shapes (powers of 2 for seq_len, fixed batch sizes) in dataloader to reduce recompile cost.
34. Add `mx.async_eval` wrapping in `CompiledPretrainingStep` (build step N+1 graph while N runs); measure throughput delta.
35. Implement attention with paged KV cache first as an inference-only pure-MLX/reference path; consider custom Metal only after profiling and fallback coverage.
36. Implement KV-cache quantization as an inference candidate; treat TurboQuant-style Hadamard/codebook designs as references until repo-local rows exist.
37. Benchmark affine q4 inference candidates on M4 Max; keep `mxfp4` / `nvfp4` as future hardware-gated research, not M4 training support.
38. Add NTK-aware RoPE scaling via precomputed `freqs`.
39. Add YaRN RoPE variant (`rope.py` from nanochat).
40. Add Llama3 piecewise scaling RoPE.

### Stream C — Model Architecture & Blocks (41–60)

41. Refactor `attention.py` to support GQA explicitly: `kv_repeat = num_q_heads // num_kv_heads` in QKV projection; assert configs.
42. Add MQA path (`num_kv_heads=1`).
43. Add full MHA path (`num_kv_heads=num_q_heads`).
44. Promote `mode='dsa'` from dense placeholder to actual sparse top-k indexing; mirror cppmega CUDA `dsa_indexer_fused_patch.py`.
45. Implement DSA indexer with per-head BMM accumulation (no `[sq,b,h,sk]` intermediate), per cppmega Patch 3.
46. Verify Mamba3 reference matches CUDA `mamba3_te_mixer.py` numerics within tolerance on golden batch.
47. Verify M2RNN reference matches Megatron M2RNN numerics on golden batch.
48. Add MoE expert-parallel sharding hooks (no compute yet — placeholders for Stream F).
49. Validate NAM56R AEMEAEMEAEMR pattern at depth 52 vs cppmega CUDA gold runs (numerical match on tiny config).
50. Add HybridLM end-to-end forward parity test vs CUDA (load fixed seed, compare logits with rtol=1e-2 atol=1e-1).
51. Add gradient checkpointing wrapper using `mx.checkpoint`; measure activation-memory savings.
52. Implement selective recompute (attention vs MLP separately; mirror CUDA `recompute_moe_experts`).
53. Add weight tying option (tied embed/lm_head) for upstream-nanochat compatibility.
54. Add embedding scale option (`embed_scale=True`) per Karpathy nanochat standard.
55. Add QK-norm option (post Q/K projections).
56. Add gated attention option (sigmoid gate per head/channel).
57. Add attention softcap and output softcap options.
58. Build `cppmega_mlx/recipes/model_factory.py` with `local_gb10_quarter` (M0 target: depth=13, hidden=3584, FFN=18944, 28 heads, head_dim=128, MTP=2) as the first profile, plus `nam56r_full` (depth=52) and the d20/d26/d34/NAM52 references mirrored from `speedrun.sh`/`run1000.sh`.
59. Validate vocab=65536 (local + cppmega tokenizer match) golden anchors. Drop the vocab=131072 alternate if not in the GB10 baseline; document non-claim if dropped.
60. Add config schema validation: refuse invalid combos (e.g., DSA mode without indexer, MTP without `mtp_depth`).

### Stream D — Tokenizer & Data Pipeline (61–80)

61. **Closed for M0.1**: vendor the GB10 HuggingFace JSON tokenizer and helper
    into `cppmega_mlx/tokenizer/` after it satisfies the deployed M0.1 artifact
    contract. Live local revalidation on 2026-05-02
    found that `nanochat/tokenizer.json` is 32K and that both
    `nanochat/tokenizer_v3.json` and
    `nanochat/config/tokenizer_v3_fixed_tokens.json` reserve id 7 for
    `<CODE_START>`. A read-only GB10 check of
    `/home/dave/cppmega-root/cpp_tokenizer_hf/tokenizer.json` and
    `/home/dave/cppmega-root/data/tokenizer/tokenizer.json` found vocab=65536
    with the same id-7 `<CODE_START>` mapping; the option-B extension renames
    the reserved id-45 slot to `<FIM_INSTRUCTION>` without changing vocab size
    or existing embedding rows. Do not retrain or rename these
    artifacts. Required local artifact contract is vocab=65536 with IDs
    `2=BOS, 3=EOT/EOS, 4=FIM_PREFIX, 5=FIM_MIDDLE, 6=FIM_SUFFIX,
    7=CODE_START, 45=FIM_INSTRUCTION, 46=SPACE, 47=NL`. Decode parity is
    handled by the explicit `<SPACE>`/`<NL>` token redesign (encode collapses
    `[\r\n]+`->`<NL>` and `[ \t]+`->`<SPACE>`, decode concatenates tokens and
    substitutes sentinels back); this replaces the previously-vendored
    nanochat heuristic `CppTokenizer.decode` path and matches the CUDA-side
    `nanochat/cpp_tokenizer.py` updated decode. The M0.1 acceptance target
    is Mac-vs-GB10 decode parity, with byte-exact round-trip for inputs
    without multi-char whitespace runs.
62. Add tokenizer vocab assertion at model load (refuse mismatched ID→token mapping; assert vocab_size == 65536; assert presence of all reserved special tokens by id and string form).
63. Extend the existing standalone Megatron `IndexedDataset` `.bin/.idx` reader seam with additional golden fixtures and scale checks.
64. Add IndexedDataset multi-shard support and side-channel schema preservation; do not import Megatron runtime into the Mac path.
65. Implement sequence packing: concat-with-EOS, cumulative doc-id; attention mask via `mx.cumsum` comparison.
66. Implement BOS-aligned best-fit packing (mirror nanochat-mlx loader).
67. Add structure-embedding side-channels in batch: `structure_ids, dep_levels, ast_depth_ids, sibling_index, node_type` (already in `LMTokenBatch`; verify dataset emits them).
68. Implement FIM data transform: PSM 50% / SPM 50%, `fim_rate=0.5`, random span sampling.
69. Implement iFIM data transform: instruction-aware marker id=45 (`<FIM_INSTRUCTION>`), instruction extracted from docstrings/comments via tree-sitter.
70. Implement AST-FIM transform via tree-sitter (optional dep behind extra), masking complete syntactic blocks per arXiv 2506.00204.
71. Add data-loader workers via PyTorch DataLoader → NumPy → `mx.array` bridge (`num_workers, persistent_workers=True`). Current local status: a minimal optional bridge exists for already-local `LMTokenBatch` rows and lazy-imports `torch` only when explicitly requested; this is not a full Stream D closure.
72. Add spawn-only multiprocessing enforcement (no fork after Metal init). Current local status: the optional DataLoader bridge defaults worker contexts to `spawn` and rejects explicit non-spawn contexts without requiring global `multiprocessing.set_start_method(...)`.
73. Implement deterministic shuffle with seed checkpoint (record cursor + RNG state).
74. Verify Parquet loader (existing `cppmega_mlx/data/parquet`) handles structure side-channels.
75. Verify NPZ loader (existing `cppmega_mlx/data/npz`) handles structure side-channels.
76. Validate tokenizer round-trip on golden corpus (10K samples, exact byte match).
77. Add tokenizer parity test vs CUDA cppmega tokenizer (encode-decode equivalence).
78. Add `scripts/data_smoke.py` for end-to-end ingress check (`.bin/.idx` → packed → batched → forward).
79. Document data format in `docs/data_pipeline.md` (file layout, special tokens, packing algorithm, FIM rates).
80. Add stress test: 100M-token shard, measure throughput and peak memory.

### Stream E — Training Loop & Checkpointing (81–100)

81. Standardize on AdamW for single-Mac M0 (≤3B mini on 128 GB) and the future 128+128 peer; switch to Lion for the 128+48 heterogeneous Stream F smoke (only path that fits the 48 GB peer with ZeRO-1). Document the switch threshold and rationale.
82. Add Muon optimizer port (mirror nanochat fork) — optional path.
83. Wire fp32 master moments for AdamW (bf16 weights + fp32 m,v); confirm via `optimizer.state` dtype assertion.
84. Add LR scheduler: cosine annealing + linear warmup; pluggable schedules.
85. Add gradient clipping (norm-based, default 1.0).
86. Add gradient accumulation (configurable `accum_steps`).
87. Add MTP loss head (training-side) with `mtp_depth=2` default (matches GB10 `local_gb10_quarter`), β decay = 0.6 (`CPPMEGA_FASTMTP_DECAY`), λ = 0.3 (`CPPMEGA_FASTMTP_LAMBDA`). K=3 added as a benchmark variant once K=2 is at parity.
88. Add STP auxiliary loss (geodesic regularization, ~100 LOC, XLA-safe static shapes).
89. Compose total loss: `L = L_NTP + λ_MTP · L_MTP + λ_STP · L_STP`.
90. Add per-loss metrics tracking (next-token CE, MTP-k CE per depth, STP cosine).
91. Validate compiled training step end-to-end (forward + backward + update) on tiny model; loss decrease.
92. Save sharded `.safetensors` checkpoints: per-rank shard, `model.safetensors.index.json` manifest.
93. Save `optimizer.safetensors` (flattened state tree via `tree_flatten`).
94. Save `train_state.json`: `{step, epoch, rng_seed, dataloader_cursor, mx.random.state b64, package_versions}`.
95. Async checkpoint saver: spawn background thread post-`mx.eval`; non-blocking.
96. Resumable training: rebuild dataloader cursor + RNG state on load; loss-curve continuity test.
97. Add checkpoint-conversion utility: HuggingFace `.safetensors` → cppmega_mlx (handle Conv2d OIHW→OHWI if any).
98. Add weight surgery: load partial weights, freeze layer ranges (for staged training).
99. Add LoRA / QLoRA fine-tuning only after an installed-API audit; treat `mlx_lm.lora` as a reference pattern until cppmega_mlx checkpoint compatibility is proven locally.
100. Wire W&B logging (`--report-to wandb`); also TensorBoard fallback.

### Stream F — Distributed & Multi-Mac (101–120)

101. Add `mx.distributed.init(backend='auto')` to launcher.
102. Add `scripts/mlx_launch.py` wrapper around `mlx.launch -n N`.
103. Add JACCL/ring setup helper as an experimental MLX distributed lane, clearly separate from Megatron distributed parity.
104. Implement data-parallel via `nn.average_gradients` across ranks.
105. Implement tensor-parallel via `nn.AllToShardedLinear` / `nn.ShardedToAllLinear` for attention/MLP.
106. Prototype ZeRO-1-style optimizer-state sharding only after a repo-local distributed smoke exists; do not label it Megatron/FSDP parity. **Required for the 128+48 heterogeneous smoke** — the 48 GB peer cannot hold a full Lion or AdamW state without ZeRO-1.
107. Add per-rank checkpoint shard naming: `model_rank{R}_of{N}.safetensors`.
108. Add multi-Mac launch test on the 128+48 heterogeneous pair over TB5: end-to-end Lion + ZeRO-1 training step at depth-13 mini. Goal is to validate the distributed code path works, not to chase throughput parity with single-Mac AdamW.
109. Profile JACCL `all_sum` throughput on a locally verified TB5/RDMA setup before setting any performance target.
110. Profile ring `all_sum` throughput on TB4 (baseline).
111. Add fail-soft: if JACCL not available, drop to ring with warning.
112. Document distributed setup in `docs/distributed.md` (host file format, ring/JACCL config).
113. Add expert-parallel for MoE layers (mirror cppmega EP).
114. Add pipeline-parallel scaffolding (helpers + schedule); defer full impl (high complexity, low priority on single-machine).
115. Validate gradient determinism across distributed runs (same seed → same loss to bit-identity at full precision).
116. Add per-rank thermal monitoring; auto-pause + checkpoint if any node thermal-throttles.
117. Add NCCL backend gate (CUDA hosts only — N/A for Mac, raise NotImplementedError).
118. Add `tests/test_distributed_smoke.py`: 2-rank toy run on single machine via TCP loopback.
119. Run mlx-pretrain-style 2-Mac toy run at scale (~100M tokens, ~30 min); record throughput.
120. Expand `docs/multimac_training.md` (initial role-stub already lands at the start of Stream F): full playbook with hardware list, network topology, run scripts, role transitions (48 GB scout↔peer when 2nd 128 GB arrives), and JACCL vs ring fallback decision tree.

### Stream G — Quantization & Precision (121–140)

121. Standardize default dtype = bf16 for training; assert at model init.
122. Add dtype assertion in optimizer: refuse fp32 weights for training (catch accidental upcasts).
123. Implement `mx.quantize` integration for inference: `mode='affine'`, `bits=4`, `group_size=64`.
124. Add per-layer quantization predicate (skip `embed_tokens` / `lm_head` → q8 instead of q4).
125. Add `scripts/quantize_checkpoint.py` wrapping `mlx_lm.convert` for cppmega_mlx checkpoints.
126. Add post-quantization perplexity validation: q4-converted model PPL within +0.05 of bf16 on held-out slice.
127. Add KV-cache quantization (kv_bits=4, kv_group_size=64, quantized_kv_start=256).
128. Test q4 vs bf16 inference latency on M4 Max; record local deltas without a fixed speedup target.
129. Add q8 path (more conservative: 2× compression, near-zero quality loss).
130. Benchmark affine q4 inference first; keep `mxfp4`/`nvfp4` comparisons hardware-gated and outside M4 training claims.
131. Document quantization in `docs/quantization.md` with phase plan and DO/DON'T list.
132. Add quantization parity test: bf16 vs q4 on tiny model; per-token KLD < 0.05.
133. Implement DWQ fine-tuning pipeline (mlx-lm `LEARNED_QUANTS.md` pattern).
134. Add AWQ fine-tuning pipeline.
135. Add GPTQ fine-tuning pipeline.
136. Document phased adoption: A bf16 training → B q4 inference → C KV q4 inference → D int8 research → E fp8-family research only when hardware/source/tests support it.
137. Add explicit refusal for FP8-family training paths on M4 (`raise NotImplementedError` with message pointing at phase plan).
138. Verify any future fp8-family API only in a hardware-gated research lane; do not make it a current M4 support gate.
139. Add GGUF export gate (for llama.cpp comparison runs); not a primary path but useful for benchmarking.
140. Add comparison harness: cppmega_mlx q4 vs llama.cpp q4 throughput on same model.

### Stream H — Features (engram / mhc / mtp / fim / ifim / stp) (141–160)

Cross-stream note: ngram_hash is **already done** (cppmega_mlx/nn/ngram_hash.py). Skip. Current dirty-tree feature slices are partial: `cppmega_mlx/nn/engram.py` is standalone, `cppmega_mlx/nn/mhc.py` is standalone, `cppmega_mlx/data/fim.py` is a CPU FIM/iFIM transform slice, and `cppmega_mlx/training/mtp.py` is a local training-side MTP helper. They are not NAM56R-integrated, not CUDA/Megatron parity receipts, and do not close this stream.

141. Port `engram.py` from `/Volumes/external/sources/nanochat/nanochat/engram.py` to `cppmega_mlx/nn/engram.py`. **Partial local slice exists**: the current module is standalone MLX, not NAM56R-wired and not a parity receipt. Wrap forward in one `mx.compile` per block when integrating; depthwise conv stays as `mx.conv_general` until profile justifies a Metal kernel. See `docs/mlx_buy_vs_build.md` row A.
142. Implement EngramBranch base mode (avg_pool1d n-gram features → bottleneck → out-projection). Current standalone module covers this locally.
143. Add gated mode: `α = σ(RMSNorm(h)·RMSNorm(k)/√d)`. Current standalone module covers this locally.
144. Add grouped causal SiLU conv path (`conv_kernel=4`, optional). Current standalone module covers this locally.
145. Wire `engram_layers` config (per-layer insertion list). This remains open; standalone module presence is not integration.
146. Validate engram parity vs nanochat fork Torch reference (forward pass, rtol=1e-2 atol=1e-2).
147. Port `mhc.py` ManifoldBranchMixer with Sinkhorn-Knopp normalization (5 iters, fp32 cast inside). **Partial local slice exists**: the current module is standalone pure MLX, not wired to an Engram/NAM56R branch combo and not a source parity receipt. Compile the fixed-iter Sinkhorn loop in one `mx.compile` closure when integrating; do not unroll in Python. See `docs/mlx_buy_vs_build.md` row B.
148. Implement N=4 streams default with `blend_alpha` interpolation to uniform.
149. Validate Sinkhorn fp32 path vs MaxText reference (column/row stochasticity within 1e-6).
150. Wire engram + mHC combo path (mHC mixes main_residual + engram_branch + skip_branch).
151. Port MTP head + recursive shared-block trick from `nanochat/mtp.py` and `cppmega/megatron/fastmtp_layer.py`. Default K=2 (GB10 baseline); single shared transformer block recurred K times per FastMTP arXiv 2509.18362. **Critical**: share weights via direct attribute aliasing (`mtp_head.weight = main.lm_head.weight`), not deep copy — MLX module tree walks recognize the shared parameter and AdamW state stays single. See `docs/mlx_buy_vs_build.md` row D + anti-port rule 13.
152. Implement `roll-and-mask` static-shape FIM-safe MTP loss (mask wrapped positions as `ignore_index=-1`); preserve XLA/`mx.compile`-safe static shapes.
153. Add per-depth weighted loss `α_k = β^k / Σ β^j` with β=0.6 default decay (matches `CPPMEGA_FASTMTP_DECAY`); apply at `λ=0.3` (matches `CPPMEGA_FASTMTP_LAMBDA`). Use `reduction='mean'` with broadcast (Liger #968 workaround mirror).
154. Validate MTP parity vs cppmega CUDA `cppmega/megatron/fastmtp_layer.py` and `mtp_native_hopper_ce.py` (loss values + grad norm at fixed seed; document non-claim where Hopper-fused CE is not reproducible on M4). Current M0.5 receipt is fail-closed: no checked-in or locally observed fixed-seed CUDA/GB10 loss+grad-norm artifact exists, so the local MLX K=2 contract tests do not close parity.
155. Port FIM data transform (PSM/SPM, fim_rate, random span sampling) to `cppmega_mlx/data/fim.py`. Current local coverage is CPU token permutation only and remains blocked on the tokenizer artifact contract for full use.
156. Port iFIM data transform (instruction extraction, AST-aware, special token id=45) to `cppmega_mlx/data/ifim.py`. Current local coverage is dependency-free instruction extraction plus token formatting in `cppmega_mlx/data/fim.py`; tree-sitter/AST-aware extraction and dataset/training integration remain open.
157. Port STP loss (~100 LOC, trivial port) to `cppmega_mlx/training/stp_loss.py`. **Local helper done**: deterministic static triples compute `1 - cosine(h[r]-h[s], h[t]-h[r])`, `T < 3` returns zero, tuple/list layer inputs average scalar losses, and defaults remain opt-in (`λ_STP=0`). Receipt: `./.venv/bin/python -m pytest tests/test_stp_loss.py tests/test_package_exports.py -q --tb=short` → `21 passed`; `./.venv/bin/pyright cppmega_mlx/training/stp_loss.py cppmega_mlx/training/loss.py cppmega_mlx/training/__init__.py tests/test_stp_loss.py` → `0 errors`.
158. Add STP variants A (single triple, last layer), B (N triples last layer), C (multi-layer averaged) toggles. The local helper covers these deterministic loss surfaces via `n_spans` and tuple/list hidden-state input, but Stream H remains open until integration/parity receipts land.
159. Wire end-to-end training loss: `L_total = L_NTP + λ_MTP · L_MTP + λ_STP · L_STP`. Current local composition is opt-in helper coverage only; NAM56R/full training-loop integration remains open.
160. Add per-feature parity tests under `tests/features/` (one file per feature).

### Stream I — Inference & Generation (161–180)

161. Implement contiguous KV-cache class `cppmega_mlx/inference/engine.py` (mirror `nanochat/engine.py`).
162. Implement paged KV-cache scheduler `cppmega_mlx/inference/serving.py` (mirror `nanochat/serving.py`).
163. Add temperature/top-k/top-p sampling.
164. Add greedy decode path (temp=0).
165. Add MTP-aware decoding helper (verify MTP heads disabled at inference; standard next-token decode).
166. Add FIM-aware infilling (PSM/SPM token routing for prompt construction).
167. Implement vanilla speculative decoding (acceptance-rejection sampling, target-only verifier) referencing `nanochat/speculative_decode.py`. Note: cppmega CUDA does not implement speculative decoding — this is greenfield work on MLX, not a parity gap.
168. Add EAGLE-2-style draft head (separate small draft model, GQA-aware) referencing `nanochat/experiments/sota_impl/eagle2/`. Gated; defer if MTP self-speculation already meets the throughput target.
169. Add self-speculative decoding via MTP (FastMTP-aligned: same model emits draft tokens via its MTP heads, target verifies in one extra forward) referencing `nanochat/mtp_draft.py`. Cheapest of the three since MTP head already lands in Stream H; benchmark first.
170. Wire mlx-lm `stream_generate` compatibility; export model via mlx-lm registry.
171. Add prompt-cache for repeated prefixes.
172. Validate prompt-cache safety with sliding-window/SSM against the installed mlx-lm prompt-cache APIs; do not assume a generic KV cache is safe for every route.
173. Add OpenAI-compatible serving endpoint (vllm-mlx pattern); `cppmega_mlx/inference/serve.py`.
174. Add throughput benchmark: prefill / decode tok/s on Qwen3-4B-class and NAM56R-class models.
175. Add quality benchmark: ARC / MMLU / HumanEval on q4-quantized model.
176. Add long-context benchmark (NIAH, RULER) on KV-q4 path.
177. Document inference modes in `docs/inference.md`.
178. Add inference-only `quantize_for_inference.py` helper script.
179. Add JSON-mode constrained decoding (logits processor).
180. Add tool-use template support (chat-template special tokens already reserved).

### Stream J — Benchmarking, Profiling & Validation (181–200)

181. Build `bench_matrix.py` matrix sweep (batch × seq × dtype × feature on/off); output to `bench/results/`.
182. Wire `scripts/compare_bench_rows.py` regression detection (compare current run vs `bench/baselines/`).
183. Add CI gate: throughput regression > 5% blocks merge.
184. Set up Xcode `.gputrace` capture pipeline (`scripts/profile_capture.py`); document workflow.
185. Per-kernel timing dashboard: parse Metal frame capture → `bench/per_kernel/<commit>.json`.
186. Add `mactop`-style GPU-saturation check before claiming GPU-bound; add to bench harness.
187. Upgrade `tests/test_cppmega_parity_anchors.py` to numerical assertions (golden-tensor compare vs CUDA reference).
188. Add `tests/test_mtp_parity.py` (vs nanochat fork golden tensors).
189. Add `tests/test_engram_parity.py`.
190. Add `tests/test_mhc_sinkhorn.py` (doubly-stochastic property test).
191. Add `tests/test_stp_loss.py`.
192. Add `tests/test_fim_transform.py` (round-trip PSM and SPM).
193. Add `tests/test_ifim_transform.py` (instruction extraction).
194. Add `tests/test_kv_quant.py` (q4-KV vs bf16-KV PPL drift).
195. Add `tests/test_distributed_smoke.py` (2-rank loopback or real multi-Mac).
196. Add `tests/test_resumable_training.py` (save mid-training, reload, verify identical loss curve).
197. Add `tests/test_grad_checkpointing.py` (loss equivalence with/without checkpointing).
198. Lock nightly throughput baseline JSON in `bench/baselines/m4max_nam56r_d20.json` and `m4max_qwen3_4b.json`.
199. Run a nanochat-d20-style benchmark on M4 Max as a local reference row only; compare external mlxgpt.com or GB10 numbers as unmatched context, not as a pass/fail parity target.
200. Document scoped cppmega.mlx support status in `docs/production_status.md` (single source of truth: throughput, peak memory, supported features, deprecated paths, and explicit non-claims).

---

## 3. Suggested parallelization

A reasonable 4-engineer split:
- **Engineer 1 (kernels-first)**: A1–20, B21–40, J181–186. Owns kernel discipline and bench harness.
- **Engineer 2 (architecture)**: C41–60, F101–120. Owns model and distributed.
- **Engineer 3 (data+training)**: D61–80, E81–100, G121–140. Owns ingress, training loop, quantization.
- **Engineer 4 (features+inference)**: H141–160, I161–180, J187–200. Owns feature ports, inference, validation.

A 2-engineer split would have one engineer take A+B+C+E+G (the platform) and the other take D+F+H+I+J (the application + features). Either split works because most cross-stream dependencies are at the test/benchmark level rather than the code level.

Critical path: A → B → C → E → H (features) is the longest single chain, ~70 steps. With 4 engineers fanning out after A, end-to-end calendar time is roughly 50–60 working days, assuming ½-day per step on average. Streams D, F, G, I, J can largely run in parallel after A is complete.

---

## 4. What NOT to do (anti-goals from research)

- Do not attempt fp8 / mxfp4 / mxfp8 / nvfp4 *training* on M4 Max without current source, hardware, and gradient-path proof; keep it deferred or inference/quantization-only until proven locally.
- Do not claim M4 Max vs GB10 parity without matched rows.
- Do not claim distributed Megatron parity from `mx.distributed` references, JACCL docs, or local single-process tests.
- Do not claim full NAM56R readiness from route/config metadata or tiny/hybrid receipts.
- Do not move forward-only custom Metal kernels into the training path.
- Do not adopt fp16 mixed-precision unless a measured ≥20% speedup justifies the parity break with CUDA.
- Do not write custom Metal for ops that `mx.fast` covers (SDPA, RoPE, RMSNorm, LayerNorm) — maintenance cost without speedup.
- Do not use `mx.array(python_scalar)` in hot paths — silent fp32 promotion kills bf16 perf.
- Do not call `.eval()` inside `mx.compile`'d functions — crashes; capture state via `inputs=`/`outputs=`.
- Do not use `shapeless=True` if your graph branches on shape — silent wrong results.
- Do not target the Apple Neural Engine (ANE) for training — CoreML inference only, not exposed to MLX.
- Do not rely on `mx.random` parity across Python random / NumPy — capture and restore explicitly.
- Do not use `prefetch(num_threads>1)` in mlx-data without `Buffer.ordered_prefetch` if you need determinism.
- Do not fork after Metal init — always `set_start_method('spawn')`.
- Do not wire >75% of unified memory — kernel panics observed in mlx-lm #883.
- Do not `git push` benchmark results without thermal-throttle check (laptop benches are unreliable).
- Do not run `bd push --no-verify` to skip hooks — investigate root cause.

---

## 5. Resolved decisions and remaining open questions

### Resolved 2026-05-01 (from team discussion)

1. **Model size target — RESOLVED**: First milestone is `local_gb10_quarter` (the GB10 quarter-depth profile), defined in `cppmega/recipes/run_profiles.py:387–422`:
   - depth=13 (1/4 of NAM56R's 52), hidden=3584, FFN=18944, num_heads=28, head_dim=128, vocab=65536, MoE pattern AEMEAEMEAEMR
   - mtp_depths=2, exponential decay β=0.6 (`CPPMEGA_FASTMTP_DECAY`), MTP loss weight λ=0.3 (`CPPMEGA_FASTMTP_LAMBDA`)
   - Approximately 3.79B total params (1/4 of full NAM56R reference at depth=52, hidden=4096, heads=32)
   - Lite-stack builder: `cppmega/megatron/nam56r_lite_spec.py` (delegates to full-stack builder)
   - Memory math at bf16 weights + bf16 grads + fp32 AdamW m,v: ≈ 7.6 + 7.6 + 30.4 = 45.6 GB before activations and runtime overhead. With grad checkpointing + Lion (1× state instead of 2×), drops to ≈ 22 GB — fits 64 GB Mac comfortably. AdamW + grad-ckpt path needs 96 GB+ headroom.

2. **Multi-Mac topology — RESOLVED (heterogeneous pair planned)**: One Mac currently active (Mac Studio, Apple M4 Max, 128 GB, macOS 26.4.1, 4× TB5 ports rated up to 120 Gb/s). A 48 GB second Mac will be added at some stage (no purchase — already on hand) and gets **both roles** in `docs/multimac_training.md`:
   - `role: inference_scout` — q4 inference / draft model server / eval & CI runner / parity-anchor checker. 3.79B at q4 ≈ 2 GB weights + KV; trivially fits 48 GB with headroom for batch and prompt cache.
   - `role: training_peer` — Stream F smoke target. Run distributed code paths (DP, ZeRO-1, TP=2) end-to-end on the heterogeneous pair to **prove the plumbing works**, not for production throughput. Memory fit is feasible with **Lion + ZeRO-1**: 7.6 W + 7.6 G + 7.6 Lion-m-half (sharded across 2 ranks) ≈ 22.8 GB params/grads/opt + ~3–5 GB activations w/ grad-ckpt + ~10 GB MLX/macOS = ~35–40 GB peak per rank. Headless mode on the 48 GB peer recommended.
   - **Connection trigger (revised)**: ~1–2 weeks after M0 work starts, when M0.2/M0.3 are green and there's a real working baseline to extend.
   - **Future homogeneous pair**: if/when a second 128 GB Mac becomes available, demote 48 GB to `role: inference_scout` only; the 128+128 pair becomes the production training peer for AdamW + larger configs.
   - **Lion vs AdamW policy**:
     - 128 GB single-Mac M0 → AdamW (default, matches GB10 baseline).
     - 128 GB + 48 GB Stream F smoke → Lion + ZeRO-1 (only path that fits 48 GB peer).
     - 128 GB + 128 GB future production → AdamW preferred (fits comfortably with ZeRO-1; matches GB10 numerics).
   - **Optional half-day smoke after M0**: one-shot JACCL (TB5) + ring fallback measurement on the 128+48 pair, emit a baseline row in `bench/baselines/m4max_heterogeneous_2node.json`, validate the rig works.

3. **Tokenizer — M0.1 CLOSED**: The earlier source statement was
   stale. Local `nanochat/tokenizer.json` is a legacy 32K artifact; local
   `nanochat/tokenizer_v3.json` is 65K and maps id 7 to `<CODE_START>`.
   The fixed-token config has the same id-7
   `<CODE_START>` contract. A read-only GB10 check of
   `/home/dave/cppmega-root/cpp_tokenizer_hf/tokenizer.json` and
   `/home/dave/cppmega-root/data/tokenizer/tokenizer.json` found vocab=65536,
   id7=`<CODE_START>`, id45=`<FIM_INSTRUCTION>`, and the same deployed
   contract. cppmega CUDA loads a HuggingFace tokenizer directory, and the
   local artifact contract now follows that deployed id 7=`<CODE_START>`
   mapping. MLX vendors nanochat's heuristic `CppTokenizer.decode` path and
   closes M0.1 on Mac-vs-GB10 decode parity receipts rather than impossible HF
   reversible decode with `decoder=null`.

4. **Parity tolerance — RESOLVED (default carried forward)**: bf16 single matmul `rtol=1e-3, atol=1e-1`; chained `rtol=1e-2, atol=1e-1`; attention/RMSNorm `atol=5e-2`; full-step grad `atol=1e-1`. Matches PyTorch's documented bf16 thresholds.

5. **MTP depth default — RESOLVED**: K=2 (matches GB10 `mtp_depths=2`). K=3 added as a benchmark configuration to test once K=2 is at parity. β=0.6, λ=0.3 carried over from cppmega CUDA defaults. Architecture: single shared transformer block recurred K times (FastMTP arXiv 2509.18362-style); cppmega CUDA's `fastmtp_layer.py` is the reference.

6. **Speculative decoding — RESOLVED**: GB10/cppmega CUDA does **not** implement speculative decoding (only accepts the `is_spec_decode` parameter and forwards to upstream Megatron without adding its own draft head). This makes MLX greenfield work, not a parity gap. Plan tests all three paths and picks based on M4 measurement:
   - **MTP self-speculation** (FastMTP-aligned, target verifies its own MTP-head drafts) — references `nanochat/mtp_draft.py`. Cheapest to add since MTP head already lands in Stream H.
   - **EAGLE-2 draft head** — references `nanochat/experiments/sota_impl/eagle2/` (separate small draft model, GQA-aware).
   - **Vanilla acceptance-rejection** — references `nanochat/speculative_decode.py` (target-only acceptance sampling with arbitrary draft).
   - Each gets a separate gated path; benchmark and pick the winner per workload.

### Confirmed dev-box hardware (verified via `system_profiler`)

- Mac Studio, Apple M4 Max, **128 GB unified memory**, macOS 26.4.1, 4× TB5 ports (Up to 120 Gb/s, 3 free).
- Closes the AdamW-vs-Lion question for the mini: AdamW default; Lion is a benchmark variant only.
- Closes the JACCL prerequisite question on the local end: macOS ≥ 26.2 ✓, TB5 ✓, M4 Max ✓.
- Second identical Mac Studio M4 Max 128 GB available offline; same JACCL prerequisites assumed met on it.

### Resolved 2026-05-02

7. **Production deployment target — RESOLVED**: 2× Mac Studio M4 Max 128 GB each, local research topology. Stream I scopes to local serving only; mlx-lm `stream_generate` compat stays in scope, OpenAI-style API and cloud-fleet hardening are out of scope until separately reopened.
8. **Sharded checkpoint format — RESOLVED**: local-first manifest with explicit non-claims about Megatron restore. HuggingFace-compatible shard-index is a future converter, not a primary path; defer until a real export need surfaces.
9. **CUDA-side regression policy — RESOLVED**: out of scope. cppmega CUDA evolves on its own track; MLX uses CUDA only as a one-time golden reference where useful. No reciprocal-validation cadence committed.
10. **MLX version pin — RESOLVED (bleeding-edge allowed)**: prefer latest stable, but **latest nightly and even unmerged PRs are in scope** when they contain something we need (a fast-path API, a kernel fix, a `mx.distributed` improvement). Process:
    - Pin per-commit via `pyproject.toml` (e.g. git URL + commit SHA) so builds stay reproducible.
    - When pulling nightly or a PR branch, open a dedicated PR here that re-runs parity + bench gates and explicitly notes which upstream change we depend on and why.
    - Keep a documented backout to the last good stable in `docs/mlx_version_journal.md` so we can roll back per-commit if upstream regresses.
    - Vendor unmerged PR patches under `vendor/mlx_patches/` as `.patch` files when full PR-branch tracking is too noisy.

---

## 6. References (one-line each)

- DeepSeek Engram: arXiv 2601.07372 (Jan 2026).
- DeepSeek-V3 (MTP): arXiv 2412.19437 (Dec 2024).
- Meta MTP: arXiv 2404.19737 (Apr 2024).
- FastMTP: arXiv 2509.18362 (Sep 2025).
- Bavarian FIM: arXiv 2207.14255 (Jul 2022).
- AST-FIM: arXiv 2506.00204 (May 2025).
- Sun iFIM: arXiv 2509.24637 (Sep 2025).
- Huang/LeCun STP: arXiv 2602.22617 (Feb 2026).
- BLT (n-gram hash): arXiv 2412.09871 (Dec 2024).
- Infini-gram: arXiv 2401.17377.
- Sinkformers / Mixture of Attention Heads: arXiv 2210.05144.
- Selective Sinkhorn Routing: arXiv 2511.08972 (Nov 2025).
- MLX docs: https://ml-explore.github.io/mlx/
- MLX distributed (JACCL/ring): https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- mlx-lm BENCHMARKS: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md
- mlx-lm LEARNED_QUANTS: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md
- MaxText (Sinkhorn ref): https://github.com/AI-Hypercomputer/maxText
- nanochat-mlx (community ref): https://github.com/scasella/nanochat-mlx
- mlx-pretrain (community ref): https://github.com/N8python/mlx-pretrain
- mlxgpt (multi-Mac training): https://mlxgpt.com/
- ZMLX (custom Metal patterns): https://github.com/Hmbown/ZMLX
- mlx-mfa (FlashAttention 2 for MLX): https://pypi.org/project/mlx-mfa/
- TurboQuant (KV quant Metal kernels): https://medium.com/@antonrozanov/turboquant-on-mlx-4-6x-kv-cache-compression-with-custom-metal-kernels-9cdee3f7d2a2
- Apple ML Research (M5 NA): https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- MLX issue #1670 (fp8 missing): https://github.com/ml-explore/mlx/issues/1670
- Megatron IndexedDataset: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/datasets.html
