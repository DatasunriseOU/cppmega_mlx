# Kernel Research Findings — DeepSeek V4 & GatedDeltaNet Landscape

**Compiled:** May 2026
**Method:** 12 parallel research agents (Perplexity Pro/Gemini 3.1 Pro, Perplexity Deep Research, Brave, Tavily, Exa neural, HuggingFace hub+papers — 2 waves: GDN + DeepSeek)
**Result:** 10 repos cloned to `/Volumes/external/sources/rent_kernels/` + 1 symlink + 1 already at `~/sources/TileKernels`

---

## 1. The big picture

Two parallel research threads merged into one architectural vision:

1. **GatedDeltaNet (GDN)** — modern linear attention with delta-rule recurrence (Yang et al. 2024, arxiv 2412.06464). Used in Qwen3-Next, Kimi-Linear, m-a-p models. Standard for hybrid attention+SSM stacks.
2. **DeepSeek V4** — released April 22, 2026, with a new TileLang-based kernel layer (`deepseek-ai/TileKernels`) covering MoE, Quant, Engram, Manifold HyperConnection (mHC).

Both threads point at the same conclusion: **TileLang has become the de facto kernel DSL** for next-gen architectures, and our existing TileLang dispatch layer is the natural integration surface.

---

## 2. DeepSeek V4 — confirmed and characterized

### 2.1 Models on HuggingFace (verified)

| Model | Params (total / active) | Released | Downloads | License |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Pro` | 1.6T / 49B | Apr 22, 2026 | 3.1M | MIT |
| `deepseek-ai/DeepSeek-V4-Pro-Base` | 1.6T / 49B | Apr 22, 2026 | — | MIT |
| `deepseek-ai/DeepSeek-V4-Flash` | 284B / 13B | Apr 22, 2026 | — | MIT |
| `deepseek-ai/DeepSeek-V4-Flash-Base` | 284B / 13B | Apr 22, 2026 | — | MIT |

Arch tag: `deepseek_v4`. Context: 1M. Modes: Non-Think / High / Max.

### 2.2 V4 vs V3.2 architectural delta

| Aspect | V3.2 | V4 |
|---|---|---|
| Attention | DSA (sparse-MLA + FP8 indexer) | **CSA + HCA + MLA** hybrid |
| KV cache @ 1M tokens | baseline | **10%** of V3.2 |
| FLOPs/token @ 1M | baseline | **27%** of V3.2 |
| Residual structure | standard `x + F(x)` | **mHC** (4-way parallel streams, Sinkhorn-projected mix) |
| MoE routing score | sigmoid | **`sqrt(softplus(x))`** |
| First-layer FFNs | dense | **Hash-routing MoE** (3 layers) |
| Optimizer | AdamW | **Muon (Newton-Schulz)** hybrid |
| Memory module | none | **Engram** (HBM↔DRAM static knowledge) |
| Long-context attention | DSA only | **+NSA** (Native Sparse Attention) |
| Experts | 256 | **384 routed + 1 shared, top-6** |
| Post-training | R1-style RL | **On-Policy Distillation (OPD)** |

### 2.3 V4-Pro concrete config (from feimatrix dump)

- 384 routed experts, top-6, 1 shared
- `topk_method=noaux_tc`, `scoring_func=sqrtsoftplus`
- `q_lora_rank=1536`, `qk_rope_head_dim=64`
- `num_hash_layers=3`, `hc_sinkhorn_iters=20`
- FP8 e4m3, block scale `[128, 128]`, `activation_scheme=dynamic`

---

## 3. TileKernels — the V4 kernel layer

**`deepseek-ai/TileKernels`** — clone at `~/sources/TileKernels/` (separate from rent_kernels).

| Property | Value |
|---|---|
| Released | April 22, 2026 (same day as V4 weights) |
| Stars | 1,135 |
| License | MIT |
| Language | Python + TileLang |
| Target | SM90 (Hopper) + SM100 (Blackwell) |
| Dependencies | TileLang ≥0.1.9, PyTorch ≥2.10, CUDA 13.1+ |
| Authors | Xiangwen Wang, Chenhao Xu, Huanqi Cao, Rui Tian, Weilin Zhao, Kuai Yu, Chenggang Zhao |

**Authorial overlap:** Cao, Yu, Zhao are also on the mHC paper (arxiv 2512.24880) — direct DeepSeek ↔ TileLang collaboration.

### 3.1 Module layout

```
tile_kernels/
├── moe/             # Top-k gating, fused expand/reduce, normalize, aux-loss-free
├── quant/           # FP8/FP4/E5M6 per-token/per-block/per-channel
├── transpose/       # Batched transpose, memory-aware layouts
├── engram/          # Fused RMSNorm + gate + grad-w reduce + hash routing
├── mhc/             # Sinkhorn norm + mix split/apply + multilayer recompute
├── modeling/        # torch.autograd.Function wrappers (engram_gate, mhc_pipeline)
├── torch/           # Pure-PyTorch reference impls (golden)
└── testing/         # Pytest harness, --run-benchmark, TK_FULL_TEST=1
```

### 3.2 Engram kernel files (4)
- `engram_fused_weight_kernel.py`
- `engram_gate_kernel.py`
- `engram_grad_w_reduce_kernel.py`
- `engram_hash_kernel.py`

### 3.3 mHC kernel files (9)
- `sinkhorn_kernel.py` (Sinkhorn-Knopp normalization)
- `expand_kernel.py`
- `head_compute_mix_kernel.py`
- `multilayer_recompute_kernel.py`
- `norm_fn_kernel.py`
- `post_kernel.py`
- `pre_apply_mix_kernel.py`
- `pre_big_fuse_kernel.py`
- `pre_split_mixes_kernel.py`

### 3.4 MoE kernel files (12)
- `aux_fi_kernel.py` (auxiliary-loss-free routing)
- `topk_gate_kernel.py`, `top2_sum_gate_kernel.py`, `topk_sum_and_topk_group_idx_kernel.py`
- `expand_to_fused_kernel.py`, `reduce_fused_kernel.py`, `get_fused_mapping_kernel.py`
- `inplace_unique_group_indices_kernel.py`, `mask_indices_by_tp_kernel.py`
- `group_count_kernel.py`, `normalize_weight_kernel.py`
- `scoring.py`, `common.py`

### 3.5 mHC forward math

```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)
```

Where:
- `X_l` shape: `(B, S, n=4, D)` — 4 parallel residual streams
- `B_l` is projected onto **Birkhoff polytope** via **Sinkhorn-Knopp** (20 iters in V4-Pro)
- `A_l`, `C_l` are learnable mix-split / mix-apply matrices
- `F_l` is the block's residual function (attention or FFN)

**Sinkhorn backward** (TileKernels Issue #2 — corrected version):
Solves `(diag(c) − Rᵀ·diag(r)⁻¹·R) · β = s_c − Rᵀ·diag(r)⁻¹·s_r` via two matvecs.
**Critical:** never materialize `RᵀR`. The naive `r=c=1` assumption only holds at full convergence.

---

## 4. GatedDeltaNet — implementation landscape

### 4.1 Recurrence

```
S_t = α_t · (I − β_t · k_t · k_tᵀ) · S_{t-1} + β_t · k_t · v_tᵀ
y_t = S_tᵀ · q_t
```

Chunkwise parallel form: 4 phases per chunk — intra-chunk Householder prefix products, inter-chunk state recurrence, intra-chunk output, mirrored backward with `solve_tril`.

### 4.2 Implementation landscape (ranked by port value)

| Rank | Project | Framework | Fused? | License | Port-to-MLX |
|---|---|---|---|---|---|
| 1 | **mlx-lm PR #1217** | MLX + Metal | ✅ fwd+bwd | MIT | **Trivial** (already MLX) |
| 2 | **mlx-swift-lm PR #257** | MLX-Swift + Metal | ✅ single-threadgroup | MIT | **Trivial** (already Metal) |
| 3 | **D-CSIL/mlx-recurrence** | MLX + Metal | ✅ fwd + custom VJP for SSM/GLA | MIT | **Trivial** (extend skeleton) |
| 4 | **QwenLM/FlashQLA** | TileLang | ✅ fwd+bwd (2-3× fwd, 2× bwd vs FLA) | (check) | **Moderate** (TileLang → our dispatch) |
| 5 | **tile-ai/tilelang PR #695** | TileLang | ✅ full op set | MIT | **Moderate** (already in our tilelang!) |
| 6 | **fla-org/flash-linear-attention** | Triton + PyTorch | ✅ chunk + recurrent | MIT | **Moderate** (canonical reference) |
| 7 | **Megatron-Core `torch_chunk_gated_delta_rule`** | Pure PyTorch | — | Apache | **Moderate** (math oracle, no kernel) |
| 8 | **NVlabs/GatedDeltaNet** | PyTorch | — | ❌ NC | Skip (license) |
| 9 | **flashinfer Blackwell** | CUDA + PTX (TMEM, TMA) | ✅ 7-GEMM persistent | BSD-3 | **Hard** (Blackwell-specific) |
| 10 | **Helion `gdn_fwd_h`** | PyTorch Helion DSL | ✅ tile-DSL | BSD/MIT | **Low** (Rosetta stone) |
| 11 | **vLLM PRs #38787, #39563, #40711** | Triton + CUDA | ✅ decode + preprocessing fusion | Apache | **Low** (decode reference) |
| 12 | **SGLang PRs #18271, #20074** | Triton | ✅ fused sigmoid gating + delta update | Apache | **Low** (production reference) |
| 13 | **ExecuTorch PR #18878** | Metal shader | ✅ derived from MLX delegate | BSD-3 | **Trivial** (Metal-native) |
| 14 | **Kaden-Schutt/hipfire** | HIP + Rust | ✅ 6 kernels | (check) | **High** (HIP→Metal) but best **docs** ever |
| 15 | **TensorRT-Edge `gatedDeltaNetPlugin`** | CuTe DSL | ✅ decode + prefill | NVIDIA NC | Skip (license) |
| 16 | **Megatron-LM PR #1989** | Megatron + Triton | ✅ Qwen3-Next production | BSD-3 | Reference for TP/CP/PP semantics |
| 17 | **llama.cpp PR #19504 + #20334** | C/C++ + CUDA + Vulkan | partial | MIT | Reference for CPU + Vulkan |
| 18 | **alxndrTL/mamba.py `mlx/pscan_mlx.py`** | MLX (Blelloch scan) | ✅ parallel scan | (check) | **Trivial** template |

### 4.2.1 metal_gdn.py — scaffolding, not yet end-to-end

`tilelang/tilelang/tileop/metal_gdn.py` provides **5 low-level Metal Simdgroup macros**:

| Macro | Purpose |
|---|---|
| `kkt_score_tile` | One 8×8 KKT score tile (Householder-style K·Kᵀ) |
| `kkt_score_tile_accum` | Accumulate 8-column slice for key_dim > 8 |
| `apply_kkt_gate_triangular_tile` | Gate decay + causal triangular mask |
| `wu_score_tiles` / `wu_score_tiles_strided` | Accumulate A/K/V tile into W and U outputs |
| `wu_linear_element` | Scalar W or U element from solved GDN A |

These are **building blocks**, not a complete GDN-on-Metal kernel. Tested in `testing/python/metal/test_metal_internal_scaffolding.py` (name says it all). Listed in `RFC_upstream_tvm_migration.md` — being migrated to upstream TVM.

**Implication for our port:** `mlx-lm PR #1217` remains the prod path. Building blocks here are useful **only** if we later need to write a custom Metal GDN kernel — they save us writing simdgroup primitives from scratch.

### 4.2.2 KDA (Kimi Delta Attention) — DPLR super-set, fully TileLang-ready

**`tilelang/examples/kda/`** contains a complete TileLang implementation of **Kimi Delta Attention** (Moonshot AI, arxiv 2510.26692) — a strict super-set of GDN with **DPLR (Diagonal Plus Low-Rank)** transport matrix `A = αI + β·u·vᵀ` plus a gated-LA forward branch. Reports +2-3% perplexity vs GDN.

**11 kernel files (~3340 LoC TileLang):**

| File | LoC | Role |
|---|---|---|
| `chunk_delta_h_fwd.py` | 306 | Forward state recurrence |
| `chunk_delta_bwd.py` | 309 | Backward state |
| `chunk_bwd_intra.py` | 492 | Intra-chunk backward |
| `chunk_inter_solve_fused.py` | 566 | Inter-chunk solve (fused, biggest) |
| `chunk_bwd_dqkwg.py` | 274 | Backward dQ, dK, dW, dG |
| `chunk_intra_token_parallel.py` | 273 | Per-token parallel intra |
| `chunk_o.py` | 242 | Output projection |
| `chunk_bwd_dv.py` | 150 | Backward dV |
| `chunk_bwd_gla_dA.py` | 147 | Gated-LA dA branch |
| `wy_fast.py` | 231 | WY-fast prepare |
| `wy_fast_bwd.py` | 350 | WY-fast backward |
| `FLA_KDA/fla_chunk_delta.py` | — | FLA-compatible wrapper |

**Production deployments:** Already in vLLM. Used by Moonshot's Kimi-Linear models.

**For us:** This is a separate block in our pattern — symbol **"K"** distinct from "L" (GDN). Different state shape, different recurrence inner loop. Plan: ROI 3.5, port from these TileLang sources into MLX array ops first (correctness baseline), then optional Metal kernel later.

### 4.3 Validation checkpoints on HuggingFace

For numerical parity testing against trained weights:

| Repo | Params | Downloads | License | Notes |
|---|---|---|---|---|
| `m-a-p/1.3B-100B-GatedDeltaNet-pure` | 1.3B | 1.6K | unspec. | Largest pure-GDN model |
| `linear-moe-hub/Gated-Deltanet-1.3B` | 1.3B | 924 | apache-2.0 | Uses FLA |
| `linear-moe-hub/Gated-Deltanet-340M` | 340M | 633 | apache-2.0 | Smaller, faster validation |
| `Idiap/gated-deltanet-attn-0.4B-10B` | 0.4B | 3 | MIT | Latest (Nov 2025), hybrid sparse+linear |
| `m-a-p/*-GatedDeltaNet-hybrid-{3,6,12,24}-1` | various | small | — | Hybrid ratios = pattern sweep |

All run through `fla.models.gated_deltanet.modeling_gated_deltanet.GatedDeltaNetForCausalLM` which calls `fla.ops.gated_delta_rule.chunk_gated_delta_rule`.

### 4.4 Key papers

| Paper | arXiv | Year | Significance |
|---|---|---|---|
| Gated Delta Networks: Improving Mamba2 with Delta Rule | 2412.06464 | 2024 | Original GDN paper |
| Parallelizing Linear Transformers with the Delta Rule | 2406.06484 | 2024 | Chunkwise algorithm backbone |
| Gated Linear Attention (GLA) | 2312.06635 | 2023 | FLA library origin |
| Tiled Flash Linear Attention (TFLA) | 2503.14376 | 2025 | Large-chunk tiling recipe |
| **Kimi Linear (KDA)** | 2510.26692 | 2025 | DPLR superset of GDN, in vLLM |
| Compiler-First State Space Duality | 2603.09555 | 2026 | XLA-only Mamba-2 |

---

## 5. Repository inventory (`/Volumes/external/sources/rent_kernels/`)

All cloned with full history (`git fetch --unshallow` after initial shallow). Sizes after deepen:

| Path | Source | Size | Commits | License | Role |
|---|---|---|---|---|---|
| `FlashMLA/` | [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | 2.1M | 60 | MIT | MLA decode + sparse FP8 kernels (CUDA, Hopper/Blackwell) |
| `DeepGEMM/` | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 3.2M | 190 | MIT | FP8/FP4 GEMM, Mega-MoE, HyperConnection (CUDA) |
| `DeepSeek-V3/` | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | 2.3M | 73 | MIT | PyTorch reference: MoE + MTP + sigmoid routing + aux-loss-free |
| `DeepSeek-V3.2-Exp/` | [deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | 2.0M | 18 | MIT | V3.2 DSA + Lightning Indexer reference |
| `flash-linear-attention/` | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | 15M | 1927 | MIT | Canonical GDN Triton kernels + layers |
| `FlashQLA/` | [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA) | 572K | 19 | (check) | **TileLang SoTA GDN**, 2-3× fwd / 2× bwd vs FLA |
| `mlx-recurrence/` | [D-CSIL/mlx-recurrence](https://github.com/D-CSIL/mlx-recurrence) | 860K | 8 | MIT | MLX-Metal fused fwd+VJP for SSM and GLA (template for GDN) |
| `hipfire/` | [Kaden-Schutt/hipfire](https://github.com/Kaden-Schutt/hipfire) | 81M | 1121 | (check) | HIP/ROCm + **DELTANET.md** (best algorithm docs anywhere) |
| `mamba.py/` | [alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py) | 21M | 222 | (check) | MLX folder — Blelloch parallel scan template |
| `tilelang/` (symlink) | `/Volumes/external/sources/tilelang/` (your fork) | — | — | MIT | Our TileLang fork with `examples/deepseek_v32/` |
| `~/sources/TileKernels/` | [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels) | — | — | MIT | **V4 kernel layer** — mHC, Engram, MoE, Quant |

---

## 6. Key documents and gotchas

### 6.1 FlashMLA docs (must-read for MLA port)

- `~/sources/rent_kernels/FlashMLA/docs/20250422-new-kernel-deep-dive.md` — MLA absorption trick (`W_UK · W_O` folds into QK GEMM)
- `~/sources/rent_kernels/FlashMLA/docs/20250929-hopper-fp8-sparse-deep-dive.md` — FP8 sparse decoding pipeline

### 6.2 Algorithm references

- **GDN Rosetta stone:** `~/sources/rent_kernels/hipfire/DELTANET.md` — full math derivation, includes critical gotcha `S@k` vs `Sᵀ@k`, alpha-gate fusion, state quantization, per-variant grid/block/LDS table
- **Sinkhorn corrected backward:** TileKernels Issue #2 + arxiv 2512.24880
- **MLA absorb trick:** FlashMLA deep-dive (April 2025 doc)
- **DSA architecture:** V3.2-Exp tech report (arxiv 2512.02556)
- **Lightning Indexer:** non-interleaved RoPE (vs MLA's interleaved) — critical gotcha from V3.2-Exp README

### 6.3 V3.2 indexer kernels (split across repos)

- Sparse attention kernels: `FlashMLA/csrc/`
- Lightning Indexer logit kernels: `DeepGEMM/deep_gemm/include/`
- Readable TileLang reference: `tilelang/examples/deepseek_v32/sparse_mla_fwd_seesaw.py`
- TileLang FP8 indexer: `tilelang/examples/deepseek_v32/fp8_lighting_indexer.py`
- mHC training kernel: merged in tilelang via PR #1758 (March 2026)

---

## 7. Apple Silicon strategy

Apple Silicon **=** SM120 desktop Blackwell (no TMEM/tcgen05). Mirror SGLang PR #24692's strategy:
- Don't try to lift TileLang kernels directly
- Use TileKernels' `torch/` reference as golden
- Write Metal-shader fallbacks with the same fused patterns
- **Engram is "free" on Apple**: unified memory natively covers the HBM↔DRAM split that Engram simulates

## 8. CUDA strategy

For cppmega CUDA stack on GB10/H200/B200:
- Wire TileKernels via Megatron `ModuleSpec` machinery
- B200/GB10 path: TMEM accumulators, UE8M0 packed scales, FP4 weight option
- GB10 unique: Grace LPDDR5X is the natural backing store for Engram via cuda C2C coherent

---

## 9. Top-5 highest-value artifacts overall

1. **`~/sources/TileKernels/`** — V4 kernel layer, MIT, contains MHC + Engram + MoE + Quant in TileLang. Reference for both MLX (use `torch/` golden) and CUDA (use `tile_kernels` direct).
2. **`tilelang/examples/deepseek_v32/`** — DSA sparse MLA fwd/bwd + Lightning Indexer in TileLang. Already in our tilelang fork.
3. **mlx-lm PR #1217** — Metal GDN fwd+bwd, MIT, LoRA-converged. Vendor and wrap as our "L" block.
4. **`~/sources/rent_kernels/DeepSeek-V3/inference/model.py`** — single-file PyTorch reference for MoE bias-correction, sigmoid routing, MTP. Cross-check our impls.
5. **`~/sources/rent_kernels/FlashMLA/docs/`** — algorithmic blueprints for MLA absorb and FP8 sparse decoding.

---

## 10. References to integration plans

- **MLX integration:** `reports/INTEGRATION_PLAN_CPPMEGA_MLX.md`
- **CUDA integration:** `reports/INTEGRATION_PLAN_CPPMEGA_CUDA.md`
- **This report:** `reports/RENT_KERNELS_FINDINGS.md` (md) + `reports/RENT_KERNELS_FINDINGS.html` (interactive HTML with cards)
