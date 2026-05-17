# cppmega (CUDA) — DeepSeek V4 / GatedDeltaNet Integration Plan for GB10, H200, B200

**Status:** Plan v1 — May 2026
**Target:** `/Volumes/external/sources/cppmega/` (PyTorch + Megatron-LM, NVIDIA stack)
**Hardware:** GB10 (Grace-Blackwell SuperChip, single-socket), H200 (Hopper SM90), B200 (Blackwell SM100)
**Scope:** Upstream-grade integration of DeepSeek V4-era kernels into cppmega's Megatron stack

---

## Hardware capability matrix

| Feature | H200 (SM90) | B200 (SM100) | GB10 (SM100 + Grace) |
|---|---|---|---|
| TMA (Tensor Memory Accelerator) | ✅ | ✅ | ✅ |
| WGMMA | ✅ FP16/BF16/FP8 | ✅ + FP4/MXFP | ✅ + FP4/MXFP |
| tcgen05 (TMEM) | ❌ | ✅ | ✅ |
| 4th-gen tensor cores | ✅ | — | — |
| 5th-gen tensor cores | ❌ | ✅ | ✅ |
| HBM3e | ✅ 141 GB | ✅ 192 GB | ✅ 384 GB (split per Grace+B200) |
| NVLink 5 | ❌ (NVLink4) | ✅ 1.8 TB/s | ✅ 1.8 TB/s (NVL72 fabric optional) |
| FP8 GEMM via DeepGEMM | ✅ E4M3 + FP32 scales | ✅ UE8M0 packed scales | ✅ UE8M0 packed scales |
| FP4 / MXFP | ❌ | ✅ | ✅ |
| Unified memory | — | — | ✅ Grace-Blackwell coherent (900 GB/s C2C) |
| Per-socket HBM bandwidth | 4.8 TB/s | 8.0 TB/s | 8.0 TB/s + Grace DDR5X 512 GB/s |

**Implications:**
- **B200/GB10 unlocks** tcgen05 (TMEM resident accumulators), 5th-gen tensor cores, FP4 fast-paths
- **GB10 unique**: Grace-Blackwell C2C lets us page-fault model weights between GPU HBM and CPU LPDDR5X transparently — natural fit for Engram's HBM↔DRAM offload trick
- **H200 stays on** WGMMA-only FP8, FP32 scales, no TMEM

---

## ROI ordering for cppmega CUDA stack

Different priority than MLX side — here we're chasing wall-clock perf, not portability.

| # | ROI | H200 | B200 | GB10-specific |
|---|---|---|---|---|
| 1 | TileKernels integration into Megatron specs | Drop-in via TileLang | + TMEM/FP4 | + Grace offload |
| 2 | FlashMLA upgrade (full sparse-FP8 path) | Existing path | + 5th-gen TC | Same as B200 |
| 3 | DeepGEMM upgrade (UE8M0 scales, FP4, Mega-MoE) | SM90 path | UE8M0 + FP4 | Same as B200 |
| 4 | Lightning Indexer (DSA) full integration | TileLang ref | TMEM-resident scores | Same |
| 5 | mHC (Manifold HyperConnection) full kernel | TileLang Hopper fallback | Native TileKernels SM100 | + Grace-coherent state |
| 6 | Engram fused gate+RMSNorm + hash | TileLang Hopper | TileKernels SM100 | **Grace LPDDR offload primary path** |
| 7 | MTP depth-D + tied-embedding optimization | Existing | TMEM accum | Same |
| 8 | NSA (Native Sparse Attention) | Triton + TileLang | TileLang + TMEM | Same |
| 9 | DualPipe + EPLB tuning for V4 size | Tested | New for V4 | New for V4 |
| 10 | FP4 / MXFP routes (B200/GB10 only) | — | New | New |

---

# ROI 1 — TileKernels integration into Megatron-LM specs

## Why

`deepseek-ai/TileKernels` is the V4 kernel library. Currently `~/sources/cppmega/cppmega/megatron/mamba_local_spec.py` imports `GatedDeltaNet` from upstream Megatron. We want to **wire the TileKernels-backed Engram, mHC, MoE, Quant kernels** into the same `ModuleSpec` machinery.

## Where in cppmega

- `cppmega/cppmega/megatron/cppmega_v4_te_stack_spec.py` — **new** stack spec for V4 architecture
- `cppmega/cppmega/megatron/cppmega_mhc_layer.py` — **new** layer wrapping TileKernels mHC pipeline
- `cppmega/cppmega/megatron/cppmega_engram_layer.py` — **new** wrapping TileKernels engram
- Existing: `cppmega/cppmega/megatron/cppmega_mamba3_tp_mixer.py` style — replicate the pattern

## Step-by-step

1. **Install** `pip install tile-kernels` (or build from source at `~/sources/TileKernels` for editability).
2. **Define `MhcLayerSubmodules`** mirroring `MambaLayerSubmodules` — fields: `pre_norm`, `mix_split`, `pre_apply_mix`, `head_compute`, `post`, `expand`, `multilayer_recompute`.
3. **Wire `tile_kernels.modeling.mhc_pipeline`** as the autograd Function inside `CppMegaMhcLayer.__call__`.
4. **`EngramLayer`** similar — wire `tile_kernels.engram` fused gate+norm.
5. **Stack spec** `build_cppmega_v4_te_stack_spec(config)` mirroring `mamba_local_spec.build_local_stack_spec`:
   - Per-layer routing: A → SelfAttention, M → Mamba3TE, R → M2RNN, E → DeepSeekMoE, L → GatedDeltaNet, N → CppMegaEngramLayer, C → CBlock, H → CppMegaMhcLayer (wraps the n-stream residual)
6. **Megatron `TransformerConfig` extensions:**
   - `mhc_n_streams: int = 4`
   - `mhc_sinkhorn_iters: int = 20`
   - `engram_use_hash: bool = False`
   - `moe_scoring: str = "sqrtsoftplus"`
   - `moe_aux_loss_free: bool = True`

## Hardware-specific paths

- **H200**: TileKernels `sm90` build — uses TMA + WGMMA-FP8 with FP32 scales
- **B200/GB10**: TileKernels `sm100` build — adds TMEM accumulators, UE8M0 scales, FP4 weight option

## Tests

`cppmega/tests/test_cppmega_v4_stack_spec.py`:
1. Stack spec builds without errors on H200 + B200
2. Forward pass on synthetic input matches reference Python impl from `tile_kernels.torch.*` within tol
3. Backward pass numerical parity (autograd vs TileKernels VJP)
4. NVTX trace shows expected kernel dispatch (TileLang kernel names visible)
5. Megatron checkpoint save/load round-trip preserves all V4 params
6. FP8 mode produces non-NaN gradients across 100 training steps

## Verification

- `pytest cppmega/tests/test_cppmega_v4_stack_spec.py -v`
- `python -m cppmega.benchmarks.mhc_stack --hw sm90` and `--hw sm100`
- Ncu profile: confirm TileKernels-generated SASS in hot kernels

---

# ROI 2 — FlashMLA upgrade (full sparse-FP8 decode + prefill)

## Why

cppmega already has FlashMLA wired for prefill. V3.2-Exp's update (Sep 2025) adds **FP8 sparse decoding** with TileLang research-reference kernels. We want both paths active.

## Where in cppmega

- `cppmega/cppmega/megatron/dsa_sparse_attention.py` — extend with FP8 decode path
- `cppmega/cppmega/megatron/dsa_path_config.py` — dispatch table per HW

## Step-by-step

1. **Update FlashMLA** in `cppmega/3rdparty/` or as a pinned commit: pull current `deepseek-ai/FlashMLA` main (Sep 2025 release).
2. **Add B200/GB10 path:** FlashMLA's SM100 build (if available, else TileLang reference). Use `torch.cuda.get_device_capability()` to dispatch.
3. **Wire absorb trick:** `FlashMLA.absorb_decode(q, c_kv, w_uk, w_o)` API — applies `W_UK^T` to Q and `W_O^T` to V cache during projection; the kernel sees the absorbed form.
4. **FP8 KV cache:** integrate with cppmega's existing FP8 KV cache infrastructure.

## Hardware-specific paths

- **H200**: WGMMA FP8 E4M3, FP32 per-block scales, 410 TFLOPS reported by DeepSeek
- **B200/GB10**: 5th-gen tensor cores with UE8M0 packed scales, 350 TFlops reported (lower numerically but at much lower energy)

## Tests

1. Decode parity: dense vs absorbed produce identical logits (tol 1e-3 fp16)
2. Prefill parity: existing FlashMLA prefill stays byte-identical
3. FP8 KV decode numerical safety: cosine sim with FP16 baseline > 0.999 across 100 sequences
4. Throughput: tokens/sec increases vs old path (assert on benchmark)

---

# ROI 3 — DeepGEMM upgrade (UE8M0, FP4, Mega-MoE)

## Why

DeepGEMM is the FP8 GEMM backbone. New features matter for V4:
- **UE8M0 packed scales** on Blackwell (smaller scale tensors, faster)
- **FP4 mode** for expert weights
- **Mega-MoE** — FP8 inputs × FP4 weights fused with all-to-all comm overlap
- **HyperConnection kernels** — fused mHC primitives at the GEMM layer

## Where in cppmega

- `cppmega/cppmega/megatron/fp8_activations.py` — existing FP8 cast
- New: `cppmega/cppmega/megatron/cppmega_deepgemm_dispatch.py` — HW-aware DeepGEMM dispatch

## Step-by-step

1. **Pin DeepGEMM** to latest `deepseek-ai/DeepGEMM` (`~/sources/rent_kernels/DeepGEMM/`).
2. **Update FP8 cast** path to emit UE8M0-packed scales on SM100, FP32 scales on SM90.
3. **Wire Mega-MoE** for E layers — replace grouped GEMM in `CppMegaTEGroupedMLP` with DeepGEMM's Mega-MoE path on B200/GB10.
4. **FP4 expert weights:** add config flag `moe_expert_dtype: Literal["fp16","bf16","fp8","fp4"]`. FP4 only enabled on SM100+.

## Hardware paths

- H200: FP8 GEMM only, no FP4 path
- B200/GB10: FP8 input × FP4 weight Mega-MoE — **largest speedup of any ROI** on V4

## Tests

1. UE8M0 scales pack/unpack round-trip integer-exact
2. FP4 weight quant produces logits within 1% cosine of BF16 baseline
3. Mega-MoE produces same output as separate fp8_gemm + all_to_all on tiny synthetic
4. Loss curve over 1k steps on tiny GPT — FP4 mode converges (no catastrophic divergence)

---

# ROI 4 — Lightning Indexer (DSA full integration)

## Why

V3.2's Lightning Indexer is a tiny FP8 attention (head_dim=32 or so) that scores all KV positions cheaply, then top-k feeds into sparse MLA. Currently cppmega DSA uses heuristic indices. Learned indexer is more accurate.

## Where in cppmega

- `cppmega/cppmega/megatron/dsa_sparse_attention.py` — add `LightningIndexer` module
- Wired into existing DSA flow

## Step-by-step

1. **Implement `LightningIndexer(nn.Module)`** per V3.2-Exp reference:
   - Small Q/K projections (head_dim 32-64)
   - Non-interleaved RoPE (**gotcha:** MLA RoPE is interleaved; these are different)
   - FP8 GEMM via DeepGEMM for the score matmul
   - Top-k via `torch.topk` (or custom kernel for speed)
2. **Two-stage forward:**
   - Stage 1: indexer → top-k indices `(B, S, kv_group, topk)`
   - Stage 2: FlashMLA `sparse_decode(q, c_kv, topk_indices, w_uk, w_o)`
3. **Training:** indexer trained via straight-through estimator on top-k attention sums (per V3.2 paper)

## Hardware paths

- All HW: same indexer logic, different GEMM backend (DeepGEMM FP8/FP4)
- B200/GB10: indexer scores can sit in TMEM, avoiding HBM round-trip

## Tests

1. Non-interleaved RoPE: indexer Q · K^T pattern matches direct PyTorch RoPE (no MLA-style interleaving)
2. Top-k validity: all indices ≤ current causal position
3. Indexer-vs-heuristic perplexity gap shrinks over training (sanity)
4. End-to-end DSA forward matches reference within tol

---

# ROI 5 — mHC kernel for cppmega (TileKernels-backed)

## Why

mHC is the V4 residual structure (4-way streams + Sinkhorn-projected mix). TileKernels has the official kernels. cppmega has no mHC today.

## Where in cppmega

- `cppmega/cppmega/megatron/cppmega_mhc_layer.py` — new (covered above in ROI 1)
- Per-block insertion: between every TransformerLayer's residual add and the next layer's input

## Step-by-step

1. **Wrap** `tile_kernels.modeling.mhc_pipeline` (autograd Function from TileKernels).
2. **Stream tensor shape**: `(B, S, n_streams, D)` flowing through the stack. First block expands `(B, S, D) → (B, S, n=4, D)`; last block contracts.
3. **20-iter Sinkhorn** (per V4-Pro config) — `tile_kernels.mhc.sinkhorn_kernel` handles fused fwd+bwd
4. **Backward** uses **corrected** Sinkhorn VJP (TileKernels Issue #2) — two matvecs, never materializes `R^T R`

## Hardware paths

- H200: TileLang lowers to Triton-compatible PTX (no TMEM, fallback to register accumulators)
- B200/GB10: TMEM-resident `R` and `R^T diag(r)^-1 R · x` accumulator

## Tests

1. Sinkhorn fwd: 20 iters → row/col sums ≈ 1 (tol 1e-4)
2. Sinkhorn bwd: corrected formula matches `torch.autograd.gradcheck` (tol 1e-4)
3. mHC fwd shape contract
4. Identity init: `A=B=C=I`, `F=0` → output equals input
5. End-to-end: 1B-param model with mHC n=4 vs baseline — train 1k steps, loss curve converges and is comparable

---

# ROI 6 — Engram with Grace LPDDR offload (GB10-unique)

## Why

V4 Engram offloads **static knowledge** (entities, formulas) from HBM→DRAM via gated lookup. Cuts HBM 60%, 2-3× inference speedup. **GB10's Grace-Blackwell coherent C2C makes this offload natural** — the DRAM is the Grace side, accessible at 900 GB/s. On H200/B200 without Grace, fall back to `cudaMemAdvise(cudaMemAdviseSetPreferredLocation, cpuId)` for managed memory pages.

## Where in cppmega

- `cppmega/cppmega/megatron/cppmega_engram_layer.py` — new
- `cppmega/cppmega/megatron/cppmega_engram_offload.py` — HW-aware offload policy

## Step-by-step

1. **Engram pipeline** (per TileKernels `engram/`):
   - Hash routing: `hash(input) → memory slot index`
   - Fused gate + RMSNorm
   - Gated lookup from memory bank
   - Weight grad reduction over hash buckets
2. **Memory bank placement:**
   - **GB10**: allocate on Grace side via `torch.utils.cpp_extension`-loaded `cudaMallocManaged` with C2C-preferred location
   - **H200/B200**: HBM resident if fits, else CPU-pinned with prefetch on access
3. **Read path:** TileKernels `engram_gate_kernel` + `engram_hash_kernel`
4. **Write path:** `engram_fused_weight_kernel` + `engram_grad_w_reduce_kernel`

## Hardware paths

- **GB10**: Grace LPDDR5X is the natural Engram backing store; full speed
- **B200**: HBM resident or CPU spill via PCIe
- **H200**: same as B200 but PCIe5 instead of PCIe5+CXL

## Tests

1. Hash routing collision handling (deterministic eviction)
2. Memory bank persistent across training restart
3. Backward gradient through hash lookup (Straight-Through Estimator)
4. GB10-specific: verify pages live on Grace side via `cudaMemRangeGetAttribute(cudaMemRangeAttributePreferredLocation)`
5. Throughput: HBM use drops 50%+, end-to-end tokens/sec stable or higher

---

# ROI 7 — MTP depth-D + tied-embedding fast path

## Why

cppmega already has MTP. V3 update: **D sequential blocks** (not one shared). Plus tied embedding optimization — share `token_embedding.weight` with `lm_head.weight` (stop-gradient through one side).

## Where in cppmega

- `cppmega/cppmega/megatron/multi_token_prediction.py` (likely path) — extend

## Step-by-step

1. **Add `SequentialMTPBlock`** — D copies of small transformer block, each with own RMSNorm
2. **Tied embedding** flag: `tie_word_embeddings: bool = True` — when set, lm_head reuses `token_embedding.weight.T`
3. **TMEM accumulation** on B200/GB10 for the per-depth logits → CE loss reduction

## Tests

1. SequentialMTPBlock has D distinct modules
2. Tied embedding: `lm_head.weight is token_embedding.weight` (or transposed view)
3. Depth-D forward + backward gradient parity
4. Megatron checkpoint round-trip preserves D blocks

---

# ROI 8 — NSA (Native Sparse Attention)

## Why

For long contexts (>128k), V4 uses NSA alongside DSA. Three branches: Compress, Select, Sliding.

## Where in cppmega

- `cppmega/cppmega/megatron/nsa_attention.py` — new

## Step-by-step

1. **Reference**: arxiv 2502.11089 + Flash Sparse Attention (arxiv 2508.18224)
2. **Compress branch**: pooled coarse blocks, projected to KV
3. **Select branch**: top-k via Lightning Indexer (reuse ROI 4)
4. **Sliding branch**: standard local window
5. **Mix**: gated combination of three branches via learned scalar gates

## Hardware paths

- All HW: compress is trivially TMA-friendly. Select reuses Lightning Indexer kernel. Sliding is standard FlashAttention.

## Tests

Per-branch parity vs reference, then mix.

---

# ROI 9 — DualPipe + EPLB tuning for V4 sizes

## Why

V4-Pro is 1.6T params, 49B active, 384 experts. V4-Flash is 284B/13B. Both need DualPipe pipeline schedule + EPLB expert balancing tuned for these specific shapes.

## Where in cppmega

- `cppmega/cppmega/megatron/dualpipe_v4_spec.py` — new
- Wire into existing `cppmega/cppmega/megatron/cppmega_eplb.py`

## Step-by-step

1. **DualPipe shape calibration** — bidirectional 1F1B schedule with V4's actual layer counts
2. **EPLB** for 384 experts: choose per-node groups, replication factor
3. **Profile** with `~/sources/rent_kernels/profile-data` traces from DeepSeek (real V3 traces, V4 likely similar shape)

## Hardware paths

- H200: 8 GPUs per node, NVLink4
- B200/GB10: NVL72 fabric → 72 GPUs in coherent domain → very different EPLB tuning

## Tests

Trace analysis: kernel overlap, all-to-all duration, pipeline bubble percentage.

---

# ROI 10 — FP4 / MXFP routes (B200/GB10 only)

## Why

Blackwell adds FP4 (E2M1) and MXFP (Microsoft's mixed precision format) support at the tensor-core level. DeepGEMM uses these for expert weights in V4 Mega-MoE. Up to 2× throughput vs FP8 in compute-bound layers.

## Where in cppmega

- `cppmega/cppmega/megatron/fp4_quant.py` — new
- Integration into `fp8_activations.py` (rename or extend)

## Step-by-step

1. **FP4 weight quant**: block-128 stochastic rounding (per DeepSeek-V3.2 prelim work)
2. **MXFP scales**: per-32-element block, e8m0 scale dtype
3. **Dispatch**: `weight_dtype="fp4"` flag wired through `TransformerConfig`
4. **TileKernels `quant/`** modules provide the FP8/FP4 conversion paths — use directly

## Hardware

- B200/GB10 only (H200 has no FP4 tensor cores)

## Tests

1. FP4 round-trip preserves 4-bit info
2. End-to-end PPL gap to BF16 < 1% on tiny eval
3. Throughput micro-bench: ≥ 1.7× FP8 baseline on Mega-MoE

---

# Repo references for cppmega CUDA stack

Cloned in `/Volumes/external/sources/rent_kernels/`:

| Repo | Purpose | Used by ROI |
|---|---|---|
| `FlashMLA/` | MLA prefill+decode kernels (CUDA + WGMMA) | 2 |
| `DeepGEMM/` | FP8/FP4 GEMM, Mega-MoE, HyperConnection kernels | 3, 10 |
| `DeepSeek-V3/` | Reference PyTorch impl of V3 MoE + MTP + routing | 1, 4, 7 |
| `DeepSeek-V3.2-Exp/` | V3.2 sparse attention + Lightning Indexer reference | 2, 4 |
| `tilelang/` (symlink → `../tilelang/`) | TileLang DSL, deepseek_v32 examples | 1, 2, 5 |
| `~/sources/TileKernels/` | V4 official kernel library (mHC, engram, MoE, quant) | 1, 3, 5, 6 |

External (not cloned, install via pip):
- `tile-kernels` (PyPI when published; else built from `~/sources/TileKernels`)
- Megatron-LM upstream (cppmega already pins this)

---

# Cross-cutting concerns

## Megatron API discipline

Every new layer must:
- Provide `ModuleSpec` with `Submodules` dataclass
- Inherit from `MegatronModule` or `GraphableMegatronModule` (for TP/CP/PP)
- Implement `sharded_state_dict` correctly
- Support `te_attention.cppmega_use_te_norm=True` to fuse RMSNorm into adjacent linear

## Distributed considerations

- **TP (Tensor Parallel)**: TileKernels' mHC requires the `(B, S, n_streams, D)` tensor to be TP-sharded on the D axis only — n_streams stays replicated across TP rank. Sinkhorn matrix is small enough to replicate.
- **EP (Expert Parallel)**: V4 has 384 experts. With EPv2 (DeepEP) the all-to-all gets the FP8/FP4 inputs natively.
- **PP (Pipeline Parallel)**: DualPipe scheduling — bidirectional 1F1B. Update pipeline boundary handling for mHC streams (must pass all 4 streams across PP boundary).
- **CP (Context Parallel)**: long-ctx training uses NSA; CP shards on sequence dim. NSA compress branch shares across CP ranks via all-gather on the coarse representation.

## FP8 / FP4 invariants

- FP8 E4M3 scales: per-block (128, 128) FP32 on H200, UE8M0 packed on B200/GB10
- FP4 E2M1: weight-only, activations stay FP8
- **No silent casts** between dtypes — every cast logs to NVTX

## Beads tracking (cppmega side)

```
bd create --title="CUDA ROI 1: TileKernels Megatron spec integration" --type=feature --priority=2
bd create --title="CUDA ROI 2: FlashMLA SM100 sparse FP8 decode" --type=feature --priority=2
bd create --title="CUDA ROI 3: DeepGEMM UE8M0 + FP4 + Mega-MoE" --type=feature --priority=2
bd create --title="CUDA ROI 4: Lightning Indexer full DSA" --type=feature --priority=2
bd create --title="CUDA ROI 5: mHC kernel layer" --type=feature --priority=2
bd create --title="CUDA ROI 6: Engram with GB10 Grace offload" --type=feature --priority=3
bd create --title="CUDA ROI 7: MTP depth-D Sequential" --type=feature --priority=3
bd create --title="CUDA ROI 8: NSA" --type=feature --priority=4
bd create --title="CUDA ROI 9: DualPipe + EPLB for V4 sizes" --type=feature --priority=3
bd create --title="CUDA ROI 10: FP4 / MXFP routes" --type=feature --priority=3
```

---

# Sequencing recommendation

1. **Wave 1 (foundation):** ROI 1 (TileKernels specs) + ROI 3 (DeepGEMM upgrade) — these unlock everything downstream
2. **Wave 2 (perf):** ROI 2 (FlashMLA full path) + ROI 4 (Lightning Indexer) + ROI 7 (MTP depth-D) — perf wins
3. **Wave 3 (V4 novel):** ROI 5 (mHC) + ROI 6 (Engram, GB10-specialized) — V4-specific architecture
4. **Wave 4 (research):** ROI 8 (NSA) + ROI 9 (DualPipe tuning) + ROI 10 (FP4)

Wall-clock with a single dev across waves: ~3 months. With 2 devs in parallel on waves 2-4: ~6 weeks.

---

## Definition of done (CUDA side)

A ROI is **done** when:
1. Merged to cppmega main with attribution
2. Tests pass on H200 + B200 (GB10 if available)
3. NCU profile shows expected kernel dispatch (TileLang or DeepGEMM SASS)
4. Megatron checkpoint round-trip works
5. End-to-end training step does not regress vs prior baseline
6. Beads issue closed
