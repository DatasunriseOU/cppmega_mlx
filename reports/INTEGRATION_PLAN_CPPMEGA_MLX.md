# cppmega.mlx — DeepSeek V4 / GatedDeltaNet Integration Plan

**Status:** Plan v1 — May 2026
**Target:** MLX-native cppmega port on Apple Silicon (M1/M2/M3/M4)
**Scope:** Land 9 ROI items from the kernel research wave, end-to-end, with tests at every step

---

## Guiding principles

1. **Reference first, kernel second.** Every block lands as pure-MLX correctness reference (Path A, golden), then Path B/C/D/E land alongside; cross-path parity is the gate.
2. **One PR per ROI item.** No mega-PRs. Each sub-ROI is its own beads issue, branch, tests, and merge.
3. **Drop-in over fork.** Where an external impl already exists (mlx-lm PR #1217), vendor a single file under `cppmega_mlx/nn/_external/` with attribution + license, then adapt.
4. **Tests run our 1B model on our real parquet data.** No HuggingFace checkpoints, no synthetic shapes for correctness tests. Real training cell via `scripts/m04_train_step.py` against `data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet`. Cross-path parity = identical loss curves over N steps within fp16 tol.
5. **Atomic commits, never `git add -A`** — pre-existing uncommitted changes in `cppmega_mlx/nn/_tilelang/m2rnn.py` and `cppmega_mlx/nn/m2rnn.py` are not ours.

---

## ROI ordering (by cost × value)

| # | ROI | Cost | Value | Status |
|---|---|---|---|---|
| 1 | MoE aux-loss-free bias balancing + `sqrt(softplus)` scoring | XS (~1 day) | High | **✅ Done** — `cppmega_v4.nn.moe_v4`, 18 tests |
| 2 | MTP depth-D sequential heads (V3-style) | S (~2 days) | Medium-High | **✅ Done** — `cppmega_v4.nn.mtp_v4`, 16 tests |
| 3.A | GDN "L" Path A pure-MLX reference (FLA `naive.py` port) | S (~1 day) | Foundation | **✅ Done** — 19 tests incl. parity vs PyTorch FLA |
| 3.B-F | GDN Paths B/C/D/E + auto-mode dispatch | M (~4-5 days) | Performance | ⏭ Deferred (Metal/TileLang kernel spike) |
| 3.5.A | KDA "K" Path A pure-MLX reference (FLA `kda/naive.py` port) | S (~1 day) | Foundation | **✅ Done** — 19 tests incl. parity vs PyTorch FLA |
| 3.5.B-D | KDA Paths B/C/D | M (~3-4 days) | Performance | ⏭ Deferred (Metal/TileLang kernel spike) |
| 3.7 | GDN+KDA head-to-head benchmark + auto-promotion receipt | S (~1-2 days) | High | ⏭ Deferred (depends on Paths B/C/D/E) |
| 4 | mHC primitives port (Sinkhorn + 6 helpers from TileKernels `torch/mhc.py`) | S (~1 day) | High | **✅ Done** — 12 tests incl. 5 parity vs TileKernels torch |
| 5 | FlashMLA absorb trick for MLA | S (~2 days) | Medium | ⏭ Deferred (no Python ref in FlashMLA; CUDA/CUTLASS only) |
| 6 | Engram primitives port (hash + fused gate from TileKernels `torch/engram.py`) | S (~1 day) | Medium | **✅ Done** — 7 tests incl. 3 parity vs TileKernels torch |
| 7 | DSA Lightning Indexer (V3.2-style) | M (~4 days) | Medium | ⏭ Deferred (heavy deps: act_quant, fp8_index, rotate_activation) |
| 8 | NSA (Native Sparse Attention) for long-ctx | L (~7 days) | Low (research) | ⏭ Deferred (research spike) |
| 9 | CSA + HCA hybrid attention stack (V4) | L (~7 days) | Low (research) | ⏭ Deferred (research spike) |

**Plugin status (May 2026):** 6 algorithm-foundation ROIs landed as pure-MLX Path A ports (all minimal-edit copies of upstream PyTorch references — FLA for GDN/KDA, TileKernels torch for mHC/Engram, math-equivalent for MoE/MTP). Every port carries a numerical parity test against the upstream PyTorch original. The `cppmega_v4/` plugin remains fully isolated: zero modifications under `cppmega_mlx/`. 91/91 v4 tests + 147/147 combined regression green at last commit (`723db80`).

**Deferred items** (Paths B/C/D/E for GDN+KDA, ROI 5/7/8/9) need Metal/TileLang/CUDA kernel work, ~1-3 days per item. Each can be picked off the deferred list in a follow-up session — the Path A reference and parity-oracle infrastructure are already in place.

Wall-clock with a single dev: ~7-8 weeks for ROI 1-7 (multi-path GDN/KDA + head-to-head). ROI 8-9 are research spikes, separate cycle.

**Why multi-path for GDN/KDA:** We have a working A/B/C pattern from mamba3 (Path A pure-MLX reference, Path B hand-written MSL, Path C TileLang-DSL→Metal). Apply the same to GDN/KDA so we can **benchmark all paths head-to-head** and let `auto_mode` pick the winner per shape, mirroring `mamba3_path_c_auto_mode_for_inputs` semantics. mlx-lm vendor is the 5th comparison anchor (GDN only — no mlx-lm KDA exists).

**Note on metal_gdn.py:** `tilelang/tilelang/tileop/metal_gdn.py` provides 5 low-level Metal Simdgroup macros (`kkt_score_tile`, `wu_score_tiles`, `apply_kkt_gate_triangular_tile`, etc.) — these are **scaffolding-level building blocks**, not an end-to-end Metal GDN kernel. Tested in `testing/python/metal/test_metal_internal_scaffolding.py`. A full Metal GDN kernel built on these blocks may land upstream later; until then, `mlx-lm PR #1217` is the prod path.

---

# ROI 1 — MoE aux-loss-free balancing + sqrt(softplus) scoring

## Why

DeepSeek-V3/V3.2/V4 use **expert-bias correction** instead of an auxiliary load-balancing loss. Each expert gets a learnable bias `b_i` that nudges its routing score; the bias is updated per step by `b_i += u * sign(load_i − mean_load)`. Combined with **`sqrt(softplus(x))`** scoring (V4-only), this gives more stable expert utilization without backprop pollution from an aux loss.

**Sources:**
- Paper: arxiv 2408.15664 ("Auxiliary-Loss-Free Load Balancing")
- Reference impl: `~/sources/rent_kernels/DeepSeek-V3/inference/model.py` (`MoE`, `Gate` classes)
- TileKernels: `~/sources/TileKernels/tile_kernels/moe/aux_fi_kernel.py` (kernel semantics)
- V4 config: `feimatrix` dump — `topk_method=noaux_tc`, `scoring_func=sqrtsoftplus`

## Where

- `cppmega_mlx/nn/moe.py` — extend `ReferenceMoE` and `MoEConfig`
- `cppmega_mlx/models/hybrid_lm.py` — propagate config

## Step-by-step

1. **Read** `DeepSeek-V3/inference/model.py` `Gate.forward` and bias update logic.
2. **Add config fields** to `MoEConfig`:
   - `scoring: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "softmax"` (default = backward-compat)
   - `aux_loss_free: bool = False`
   - `bias_update_rate: float = 1e-3`
   - `node_limited_routing: int | None = None` (cap experts per node group)
3. **Add parameter** `expert_bias: mx.array` of shape `(num_experts,)` initialized to zeros, **non-trainable** (manually updated post-step). Use `mx.zeros` and `freeze` mechanism.
4. **Modify scoring** in `ReferenceMoE.__call__`:
   ```python
   scores = self.gate(x)
   if scoring == "sqrtsoftplus":
       scores = mx.sqrt(mx.nn.softplus(scores))
   elif scoring == "sigmoid":
       scores = mx.sigmoid(scores)
   # else softmax fallback (existing)
   if aux_loss_free:
       scores = scores + self.expert_bias[None, None, :]  # broadcast
   ```
5. **Add bias update method** `ReferenceMoE.update_bias_after_step(load_per_expert: mx.array)`:
   ```python
   mean_load = load_per_expert.mean()
   delta = self.bias_update_rate * mx.sign(load_per_expert - mean_load)
   self.expert_bias = self.expert_bias + delta
   ```
   Called from training loop after each step.
6. **Wire HybridTinyConfig** `moe_scoring`, `moe_aux_loss_free`, `moe_bias_update_rate`, `moe_node_limited_routing` config fields → `MoEConfig`.

## Tests (`tests/test_moe_aux_loss_free.py` — new file)

1. **Scoring math:** `sqrtsoftplus(x)` matches `mx.sqrt(mx.nn.softplus(x))` element-wise against a PyTorch oracle.
2. **Bias shape:** `expert_bias` is `(num_experts,)`, broadcasts to scores.
3. **Bias update:** Given synthetic load `[10, 5, 0, 5]`, after one update `expert_bias[0]` decreases by `update_rate`, `expert_bias[2]` increases by `update_rate`.
4. **Backward-compat:** With `aux_loss_free=False`, scoring is identical to the existing softmax path (byte-identical regression test).
5. **HybridTinyConfig YAML roundtrip** with the new fields (extends `test_hybrid_lm_extensions.py`).
6. **Integration:** Tiny model with `pattern="AE"`, train 50 steps on synthetic data — load variance decreases over time.

## Verification

- `pytest tests/test_moe_aux_loss_free.py` — all pass
- `pytest tests/test_hybrid_lm_extensions.py` — no regression
- `pytest tests/test_checkpoint.py::test_checkpoint_resume_restores_hybrid_custom_blocks_and_optimizer` — checkpoint round-trip with `expert_bias` parameter

## Commit message

`feat(moe): add aux-loss-free bias balancing and sqrtsoftplus scoring (DeepSeek-V3/V4)`

---

# ROI 2 — MTP depth-D sequential heads (V3-style)

## Why

Our `MinimalMTPHead` (`cppmega_mlx/training/mtp.py`) reuses one shared block recursively for all depths. DeepSeek-V3 uses **D sequential transformer blocks** (one per depth), each with its own RMSNorm, with shared token embeddings and lm_head. This is the real architecture; ours is a contracted reference.

**Sources:**
- `~/sources/rent_kernels/DeepSeek-V3/inference/model.py` — `MTP` class
- DeepSeek-V3 tech report (arxiv 2412.19437)

## Where

- `cppmega_mlx/training/mtp.py` — extend with `SequentialMTPHead`
- `cppmega_mlx/models/hybrid_lm.py` — config + autoattach

## Step-by-step

1. **Read** `DeepSeek-V3/inference/model.py` `MTP` class — note: each depth has its own `RMSNorm` + a single shared `TransformerBlock` (D copies).
2. **Add class** `SequentialMTPHead(nn.Module)` next to `MinimalMTPHead`:
   - Constructor takes `depth`, `hidden_size`, `vocab_size`, shared `token_embedding`, shared `lm_head`
   - Owns `D` separate `MTPBlock` modules (each = norm + projection + small transformer block)
   - Forward: at each depth `d`, take `h_{d-1}`, apply depth-`d` block, predict next-d-tokens
3. **Add config field** `mtp_architecture: Literal["minimal", "sequential"] = "minimal"` on `HybridTinyConfig`.
4. **Auto-attach branch** in `HybridTinyLM.__init__`:
   ```python
   if mtp_config is not None:
       cls = SequentialMTPHead if cfg.mtp_architecture == "sequential" else MinimalMTPHead
       self.mtp_head = cls(self.token_embedding, self.lm_head, config=mtp_config, depth=cfg.mtp_depth)
   ```
5. **Shared weights invariant:** `mtp_head.token_embedding is model.token_embedding` (no copy).

## Tests (`tests/test_mtp_sequential.py` — new)

1. **Construction:** `SequentialMTPHead(depth=3)` has 3 distinct `MTPBlock` instances (different `id()`).
2. **Shared weights:** `mtp_head.token_embedding is model.token_embedding`.
3. **Forward shape:** returns `D` logits tensors, each `(B, S, vocab_size)`.
4. **Loss math:** per-depth CE losses combine with `compute_weighted_mtp_loss` correctly.
5. **Gradient flow:** backward through one depth doesn't pollute the other depths' gradients (assertion on `mx.grad`).
6. **YAML round-trip** for new config field.
7. **Backward-compat:** existing `MinimalMTPHead` tests still pass.

## Verification

- `pytest tests/test_mtp_sequential.py tests/test_mtp_loss.py tests/test_hybrid_lm_extensions.py` — all pass
- Train 100 steps with `mtp_architecture="sequential"` vs `"minimal"` on synthetic — sequential loss converges faster (sanity check, not a hard assertion)

## Commit message

`feat(mtp): add SequentialMTPHead for DeepSeek-V3 depth-D architecture`

---

# ROI 3 — GDN block "L" (multi-path: A pure-MLX + B hand-MSL + C TileLang-DSL + D Triton-frontend + E vendor mlx-lm)

## Why

GatedDeltaNet (Yang et al. 2024, arxiv 2412.06464) is the modern linear-attention block used by Qwen3-Next, Kimi-Linear, m-a-p, and Idiap models. We need a "L" symbol in our pattern.

**Strategy: mirror our existing mamba3 multi-path A/B/C pattern** and add vendor + Triton-frontend anchors. Land all paths, then benchmark head-to-head; let `auto_mode` pick the winner per (batch, seq_len, head_dim) shape — same machinery as `mamba3_path_c_auto_mode_for_inputs`.

## The 5 paths

| Path | What | Source | Why this anchor |
|---|---|---|---|
| **A** | Pure-MLX reference (slow, golden) | Hand-write from FLA `chunk_gated_delta_rule` math | Correctness oracle; numerical parity anchor for B/C/D/E |
| **B** | Hand-written MSL via `mx.fast.metal_kernel` | Translate FLA/upstream Triton manually to MSL | Fastest path while DSL backends mature (proven mamba3 pattern) |
| **C** | TileLang DSL → Metal via `tilelang.compile(target="metal", execution_backend="tvm_ffi")` | Lift `tilelang/examples/gdn/*.py` `@T.prim_func`s into our Path C skeleton | Long-term primary; lets us reuse upstream TileLang fixes |
| **D** | Triton frontend via `tilelang/poc/triton_frontend/from_triton_kernel()` | FLA `fla/ops/gated_delta_rule/chunk.py` Triton kernel | Auto-port from canonical reference; coverage test for our Triton→TileLang mapper |
| **E** | Vendor mlx-lm PR #1217 Metal op | `ml-explore/mlx-lm` PR #1217 — Metal fwd+bwd | External baseline; if E beats B/C/D long-term, we just consume upstream |

**Sources cloned (algorithm references only — no checkpoint loading):**
- `~/sources/rent_kernels/flash-linear-attention/fla/ops/gated_delta_rule/chunk.py` — FLA canonical Triton (path A math reference, path D source)
- `~/sources/rent_kernels/tilelang/examples/gdn/*.py` — 7 TileLang prims (path C source)
- `~/sources/rent_kernels/hipfire/DELTANET.md` — algorithm Rosetta stone
- `~/sources/rent_kernels/mlx-recurrence/` — D-CSIL structural template
- mlx-lm PR #1217 (not cloned; fetch via `git checkout` of the PR branch when starting path E)

**Validation data:** `data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet` (our real C++ training data). Training cell via `scripts/m04_train_step.py`. No HuggingFace checkpoints involved.

## Where

```
cppmega_mlx/nn/
├── linear_attention.py            # Path A pure-MLX reference + public LinearAttentionBlock module
├── _tilelang/
│   ├── linear_attention.py        # Path B hand-MSL (mx.fast.metal_kernel, mirror mamba3.py)
│   ├── linear_attention_path_c.py # Path C TileLang-DSL → Metal (mirror mamba3_path_c.py)
│   └── linear_attention_triton.py # Path D Triton frontend lift
└── _external/
    └── mlx_lm_gated_delta_update.py # Path E vendored (MIT + provenance header)
```

Plus:
- `cppmega_mlx/recipes/pattern.py` — symbol "L" + role "linear_attention"
- `cppmega_mlx/models/hybrid_lm.py` — `_ROUTE_SYMBOL_BACKENDS`, `HybridTinyBlock`, config

## Step-by-step (mirroring mamba3 wave structure)

**Sub-ROI 3.A — Path A pure-MLX reference** (~1 day)
1. Implement `chunk_gated_delta_rule_reference(q, k, v, g, beta, chunk_size, ...)` in pure MLX (`mx.matmul`, `mx.tril`, `mx.cumsum`) translating FLA math.
2. Wrap in `LinearAttentionBlock` (config, projections, QK L2 norm, short conv, output proj). This is the public surface.
3. Wire `pattern.py` symbol "L", `_ROUTE_SYMBOL_BACKENDS`, `HybridTinyBlock`, `HybridTinyConfig`. Tests pass on Path A.

**Sub-ROI 3.B — Path B hand-MSL** (~1 day)
1. Translate FLA Triton kernel to hand-written MSL string (mirror `_tilelang/mamba3.py` shape — `_FWD_KERNEL_SOURCE`, `_BWD_KERNEL_SOURCE`).
2. Wire via `mx.fast.metal_kernel`, expose `linear_attention_fwd_metal` / `bwd_metal`.
3. Custom VJP gluing fwd+bwd.
4. Path A vs Path B parity test (tol 1e-4 fp16).

**Sub-ROI 3.C — Path C TileLang DSL** (~1.5 days)
1. Copy skeleton from `cppmega_mlx/nn/_tilelang/mamba3_path_c.py` (status object, `tilelang.compile(..., target="metal", execution_backend="tvm_ffi", out_idx=...)` dispatcher, parity gates, Z3 lane mappings if applicable).
2. Lift `tilelang/examples/gdn/example_chunk_delta_h.py` (+ `chunk_delta_bwd`, `chunk_o`, `wy_fast`, `cumsum`) as `@T.prim_func` source.
3. Apply our Metal-specific patterns: no atomics → scatter-add inside kernel, no `*_partial` outputs → final owner buffers, TOPK state in static threadgroup buffers.
4. Status + auto-promotion gates: `gdn_path_c_status()`, `gdn_path_c_auto_mode_for_inputs(...)`.
5. Path A vs Path C parity (tol 1e-4 fp16).

**Sub-ROI 3.D — Path D Triton frontend** (~0.5 days, optional)
1. Use `tilelang/poc/triton_frontend/from_triton_kernel(fla_chunk_gated_delta_rule_kernel)` to lift FLA Triton directly.
2. Flag any unmapped Triton ops — file issue on `triton_frontend` for coverage expansion if needed.
3. If conversion succeeds → run through same `tilelang.compile(target="metal")` → Metal MSL.
4. Path A vs Path D parity.

**Sub-ROI 3.E — Path E vendor mlx-lm** (~1 day)
1. Fetch mlx-lm PR #1217 branch, extract `gated_delta_update` Metal kernel + Python wrapper.
2. Vendor under `cppmega_mlx/nn/_external/mlx_lm_gated_delta_update.py` with MIT license header + provenance (PR URL + commit hash).
3. Thin adapter that exposes the same `(q,k,v,g,beta) → output` contract as A/B/C/D.
4. Path A vs Path E parity.

**Sub-ROI 3.F — Dispatch + auto-mode** (~0.5 days)
1. Add `cppmega_mlx/runtime/kernel_policy.py` entry for `linear_attention` (mirror `m2rnn`, `sparse_mla` route entries).
2. Env `CPPMEGA_KERNEL_PATH__LINEAR_ATTENTION=path_a|path_b|path_c|path_d|path_e|auto`.
3. `linear_attention_auto_mode_for_inputs(B, S, head_dim, ...)` picks the best path per shape (initially per benchmark, eventually receipt-driven).

## Tests (`tests/test_linear_attention.py` + per-path files + 1B harness)

**Unit tests (tiny shapes, fast — all paths):**
1. Pattern parsing: `parse_nam_pattern("AL")`, `expand_nam_pattern` produces `linear_attention` role.
2. Block construction `pattern="L"` builds the block with correct shapes.
3. Forward shape `(B,S,D) → (B,S,D)`.
4. doc_ids masking (state does not cross doc boundary in packed batch).
5. YAML round-trip.
6. Checkpoint round-trip.
7. Chunkwise vs recurrent equivalence on tiny seq (S=8).

**Cross-path parity tests (tiny shapes):**
8. **Path A vs Path B** forward + backward (fp16 tol 1e-4, fp32 tol 1e-6).
9. **Path A vs Path C** forward + backward.
10. **Path A vs Path D** forward + backward (if Triton frontend coverage permits).
11. **Path A vs Path E** forward + backward.

**Auto-mode dispatch:**
12. Env override `CPPMEGA_KERNEL_PATH__LINEAR_ATTENTION=path_{a..e}` forces each path; output equals direct call.

**1B model end-to-end tests** (real model, real data — primary correctness gate):
13. **Train 20 steps** of our 1B-class model on `data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet` with `pattern` containing "L" (e.g., `pattern="ALEM"`), per path. Mirror `scripts/bench_1b_training_matrix.py` shape: same hidden_size, num_heads, seq_len as production 1B config.
14. **Loss curves identical across paths** within fp16 tol — Path A loss is golden, B/C/D/E within `1e-3` relative.
15. **No NaN / inf** in any path's grads across 20 steps.
16. **Memory budget**: peak HBM stays under M-series unified-memory ceiling we already enforce for 1B (~24 GB on M2 Ultra, configurable).
17. **Checkpoint round-trip mid-training**: save at step 10, resume, finish to step 20, final loss matches uninterrupted run within fp16 tol.

## Verification

- All unit + parity + 1B-end-to-end tests pass per path
- No regression in existing suite (≥ 198 prior tests stay green)
- `pytest -q tests/test_linear_attention*.py` ≥ 60 new tests passing
- **1B end-to-end on real parquet:** 20-step training run with `pattern="ALEM"`, all 5 paths produce identical loss curves within fp16 tol — receipt JSON dropped to `reports/raw/cppmega_1b_path_matrix_cells/linear_attention_path_*.json` for downstream HTML matrix
- No HF dependency anywhere — purely our 1B + our parquet

## Commit ordering

One commit per sub-ROI. Each commit lands with its own beads issue.
- `feat(linear_attention): add 'L' block scaffold + Path A pure-MLX reference`
- `feat(linear_attention): add Path B hand-MSL kernel`
- `feat(linear_attention): add Path C TileLang-DSL kernel`
- `feat(linear_attention): add Path D Triton-frontend kernel`
- `feat(linear_attention): add Path E vendor mlx-lm`
- `feat(linear_attention): add auto-mode dispatch and receipt`

---

# ROI 3.5 — KDA block "K" (multi-path: A pure-MLX + B hand-MSL + C TileLang-DSL + D Triton-frontend)

## Why

**KDA = Kimi Delta Attention** (Moonshot AI, arxiv 2510.26692) is a **strict super-set of GDN** that replaces the classical Householder recurrence with a **DPLR (Diagonal Plus Low-Rank) transport matrix**: `A = αI + β·u·vᵀ`. Plus a gated-LA branch in the forward pass. Reports +2-3% perplexity over GDN on Moonshot's eval suite. Already integrated into vLLM, used in production Kimi-Linear checkpoints.

GDN and KDA differ enough that two **separate blocks** in the pattern is cleaner than one parametrized block — different state shapes, different recurrence inner loop, different test surface.

**Path coverage:** A/B/C/D (same A/B/C/D as GDN; **no path E** — there is no mlx-lm KDA op yet). If KDA appears in mlx-lm later, add path E in a follow-up.

| Path | What | Source |
|---|---|---|
| **A** | Pure-MLX reference (golden) | Hand-write from `tilelang/examples/kda/FLA_KDA/fla_chunk_delta.py` |
| **B** | Hand-written MSL | Translate FLA KDA Triton to hand MSL |
| **C** | TileLang DSL → Metal | Lift `tilelang/examples/kda/*.py` (11 prims) via our Path C skeleton |
| **D** | Triton frontend | FLA KDA Triton through `triton_frontend.from_triton_kernel()` |

**Sources (algorithm references only):**
- Paper: arxiv 2510.26692 (Kimi Linear)
- TileLang impl: `~/sources/rent_kernels/tilelang/examples/kda/` (11 files, ~3340 LoC, already TileLang — path C source)
- FLA-compatible math reference: `tilelang/examples/kda/FLA_KDA/fla_chunk_delta.py` (path A math reference)
- vLLM production path: `vllm/model_executor/models/kimi_linear*` (reference for kernel boundaries)

**Validation data:** `data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet` — our real C++ training data. No HF checkpoints.

## Kernel files ready to port

| File | LoC | Role |
|---|---|---|
| `chunk_delta_h_fwd.py` | 306 | Forward state recurrence (chunkwise) |
| `chunk_delta_bwd.py` | 309 | Backward state |
| `chunk_bwd_intra.py` | 492 | Intra-chunk backward |
| `chunk_bwd_dqkwg.py` | 274 | Backward dQ, dK, dW, dG |
| `chunk_bwd_dv.py` | 150 | Backward dV |
| `chunk_bwd_gla_dA.py` | 147 | Gated-LA dA branch |
| `chunk_inter_solve_fused.py` | 566 | Inter-chunk solve (fused, biggest file) |
| `chunk_intra_token_parallel.py` | 273 | Per-token parallel intra |
| `chunk_o.py` | 242 | Output projection |
| `wy_fast.py` | 231 | WY-fast prepare |
| `wy_fast_bwd.py` | 350 | WY-fast backward |

Total: ~3340 LoC TileLang. They're CUDA/Triton-side TileLang; for MLX we either lift the algorithm into MLX array ops (slow but correct first) or wait for a Metal-grade port.

## Where

- `cppmega_mlx/nn/kimi_delta_attention.py` — new `KimiDeltaAttentionBlock`
- `cppmega_mlx/nn/_external/` — optional vendored helpers from FLA_KDA
- `cppmega_mlx/recipes/pattern.py` — add symbol "K" + role "kda"
- `cppmega_mlx/models/hybrid_lm.py` — extend `_ROUTE_SYMBOL_BACKENDS`, etc.

## Step-by-step

1. **Read** the KDA paper (arxiv 2510.26692) — focus on Section 3 (DPLR formulation).
2. **Read** `tilelang/examples/kda/FLA_KDA/fla_chunk_delta.py` — this is the FLA-compatible reference wrapper, easiest entry point.
3. **Implement** `KimiDeltaAttentionConfig` dataclass:
   ```python
   @dataclass(frozen=True)
   class KimiDeltaAttentionConfig:
       hidden_size: int
       num_heads: int = 4
       head_dim_k: int = 128
       head_dim_v: int = 256
       chunk_size: int = 64
       dplr_low_rank: int = 1   # rank of the low-rank perturbation
       use_gla_branch: bool = True   # the gated-LA forward branch
       use_qk_l2norm: bool = True
       conv_kernel: int = 4
   ```
4. **Port chunk_delta_h_fwd** — translate the TileLang fwd to pure MLX (using `mx.matmul`, `mx.triu`/`mx.tril` for the chunk masks). This is the foundation kernel.
5. **Port chunk_delta_bwd** — backward pass for the state recurrence.
6. **Port chunk_o** — output projection.
7. **Port gated-LA branch** (`chunk_bwd_gla_dA.py` + forward equivalent) — the second forward path KDA adds on top of GDN.
8. **Implement custom MLX VJP** wrapping all backward kernels for end-to-end autograd.
9. **Add to pattern.py:** symbol "K" → role "kda", update all helper functions analogously to "L".
10. **Update `_ROUTE_SYMBOL_BACKENDS`**: `"K": "kda"`.
11. **HybridTinyBlock branch** for `symbol == "K"`: `self.block = KimiDeltaAttentionBlock(config.kda_config())`.
12. **HybridTinyConfig** gets `kda_*` fields + `kda_config()` method.
13. **`route_delta`** branch: `delta = cast(KimiDeltaAttentionBlock, self.block)(x, doc_ids=doc_ids)`.
14. **doc_ids threading** (consistent with engram + linear_attention).

## Tests (`tests/test_kda.py` — new)

**Unit tests (tiny shapes):**
1. Pattern parsing: `parse_nam_pattern("LK")` accepts both symbols.
2. Block construction: `pattern="K"` builds `KimiDeltaAttentionBlock`.
3. Forward shape: `(B, S, D)` → `(B, S, D)`.
4. **DPLR collapse-to-GDN sanity:** with `dplr_low_rank=0`, output matches classical GDN reference (within tol — cross-check).
5. Chunkwise vs recurrent equivalence (tiny seq S=8).
6. doc_ids masking (state does not cross document boundary).
7. Mixed-pattern: `pattern="ALKE"` builds — A → L → K → E, each block correctly typed.
8. YAML round-trip with `kda_*` fields.
9. Checkpoint round-trip.

**Cross-path parity (tiny shapes):**
10. Path A vs Path B fwd+bwd.
11. Path A vs Path C fwd+bwd.
12. Path A vs Path D fwd+bwd.

**1B end-to-end on real parquet:**
13. Train 20 steps of 1B model on `data/parquet_samples/gb10/clang_semantic_4k_v10/val_00000.parquet` with `pattern="AKEM"`, per path.
14. Loss curves identical across A/B/C/D within fp16 tol.
15. No NaN/inf in any path's grads.
16. Peak HBM within 1B-class budget.
17. Mid-training checkpoint resume.

## Verification

- All new tests pass; existing suite no regression
- `pytest -q tests/` baseline grows by ~11 tests
- **Convergence smoke:** train 100 steps on tiny data with `pattern="LK"` (mix of GDN + KDA) — loss decreases

## Commit message

`feat(kda): add 'K' block (Kimi Delta Attention) via tilelang/examples/kda port to MLX`

---

# ROI 3.7 — GDN+KDA head-to-head benchmark + auto-promotion receipt

## Why

Once all paths land (A/B/C/D/E for GDN; A/B/C/D for KDA), we need a **systematic comparison** to drive auto-mode dispatch. Mirror the existing `mamba3_path_c_receipt_auto_mode` machinery — produce a JSON receipt per (block, dtype, shape) that auto-mode reads.

## Sources / template

- `cppmega_mlx/nn/_tilelang/mamba3_path_c.py::mamba3_path_c_receipt_auto_mode` — existing pattern
- `cppmega_mlx/nn/_tilelang/mamba3_path_c.py::mamba3_path_c_auto_mode_for_inputs` — auto dispatch
- `scripts/bench_*.py` (existing benchmark harness)
- `reports/raw/cppmega_1b_path_matrix_cells/` — existing receipt JSON format

## Where

- `scripts/bench_linear_attention_path_matrix.py` — new benchmark harness (mirror `bench_1b_training_matrix.py`)
- `cppmega_mlx/runtime/kernel_policy.py` — extend with `linear_attention` and `kda` route entries
- `cppmega_mlx/nn/_tilelang/linear_attention_path_c.py::linear_attention_path_c_receipt_auto_mode` — receipt helper (mirror mamba3)
- Equivalent for KDA

## Step-by-step

1. **Define benchmark cell shapes** — same set as mamba3 receipt: typical small (`B=1,S=2048,D=2048,H=16`), 1B-scale (`B=4,S=2048,D=2048,H=16`), 7B-scale (`B=2,S=4096,D=4096,H=32`).
2. **Run all 5 GDN paths × 4 KDA paths × N shapes × {fp16, fp32, bf16}** through the bench harness. Record wall-clock fwd/bwd/fwd+bwd, peak HBM, output max-abs error vs Path A.
3. **Emit receipt JSON** under `reports/raw/cppmega_1b_path_matrix_cells/linear_attention_path_*.json` and `kda_path_*.json`.
4. **Implement `linear_attention_auto_mode_for_inputs(B, S, D, H, dtype)`** that reads receipts and dispatches to the fastest path within numerical tol.
5. **HTML render** like `reports/cppmega_1b_path_matrix.html` — color-coded grid of fastest-per-cell.

## Tests (`tests/test_linear_attention_receipt.py`, `tests/test_kda_receipt.py`)

1. Receipt JSON schema matches existing `mamba3_path_c` schema
2. `auto_mode_for_inputs` returns correct path per receipt
3. Forced-env (`CPPMEGA_KERNEL_PATH__LINEAR_ATTENTION=path_c`) overrides receipt
4. Receipt rebuild deterministic on fixed seed

## Verification

- All 5 GDN paths and 4 KDA paths run to completion across all cells
- Auto-mode picks the fastest path per shape (verified by spot-check)
- `reports/linear_attention_path_matrix.html` renders
- No regression in existing mamba3 receipt tests

## Commit message

`feat(bench): add GDN/KDA path-matrix benchmark + auto-promotion receipt (mirror mamba3)`

---

# ROI 4 — mHC (Manifold HyperConnection) upgrade

## Why

Our `ManifoldBranchMixer` is a tiny placeholder. DeepSeek V4 uses **formal mHC**: 4-way parallel residual streams with mixing matrices `A_l`, `B_l`, `C_l`, where `B_l` is projected onto the **Birkhoff polytope via Sinkhorn-Knopp iterations** (20 iters in V4-Pro config). Forward:
```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)
```

**Sources:**
- Paper: arxiv 2512.24880 (formal mHC)
- TileKernels: `~/sources/TileKernels/tile_kernels/mhc/` (full pipeline: sinkhorn, pre_split_mixes, pre_apply_mix, head_compute_mix, post, expand, multilayer_recompute, norm_fn)
- Golden reference: `~/sources/TileKernels/tile_kernels/torch/` (PyTorch)
- Backward math: TileKernels Issue #2 — Sinkhorn corrected backward via two matvecs, never materializes `RᵀR`
- HF analysis: `tokenbender/mHC-manifold-constrained-hyper-connections`

## Where

- `cppmega_mlx/nn/mhc.py` — extend `ManifoldBranchMixer` + `ManifoldBranchMixerConfig`
- `cppmega_mlx/models/hybrid_lm.py` — config plumbing

## Step-by-step

1. **Read** `~/sources/TileKernels/tile_kernels/torch/` files for mHC reference (this is the golden source). Specifically the `mhc_pipeline` autograd Function.
2. **Read** `~/sources/TileKernels/tile_kernels/mhc/sinkhorn_kernel.py` to understand the kernel structure (it's TileLang Triton — we mirror semantics, not kernel).
3. **Read** TileKernels Issue #2 — the corrected backward is critical (avoids `r=c=1` assumption that only holds at convergence).
4. **Add config:** extend `ManifoldBranchMixerConfig` with:
   - `n_streams: int = 4` (parallel residual streams)
   - `sinkhorn_iters: int = 20`
   - `expand: bool = True` (project D → n_streams · D at first layer, contract at last)
5. **Implement** `sinkhorn_normalize(M, iters=20)` — pure MLX, row-then-column normalize loop in fp32, then cast back.
6. **Implement** `mhc_forward(X, A, B, C, F_out)`:
   - `X` shape: `(B, S, n_streams, D)`
   - `B_normalized = sinkhorn_normalize(B)` (Birkhoff projection)
   - Compute the recurrence above
7. **Implement custom VJP** for Sinkhorn — use the **corrected** backward from Issue #2: solve `(diag(c) − Rᵀ diag(r)⁻¹ R) β = s_c − Rᵀ diag(r)⁻¹ s_r` via two matvecs.
8. **Extend `HybridTinyBlock.__call__`** to accept the n-stream tensor when `mhc_n_streams > 1`. Wrap entrance and exit (expand at first block, contract at last) — or keep streams across blocks (advanced).
9. **Default behavior:** if `n_streams == 1`, behave like current `ManifoldBranchMixer` (full backward-compat).

## Tests (`tests/test_mhc_v4.py` — new)

1. **Sinkhorn fwd:** after 20 iters on a 4×4 matrix, row sums ≈ 1, col sums ≈ 1 (tol 1e-4)
2. **Sinkhorn bwd parity:** compare custom-VJP gradient to `mx.grad` of unrolled loop (tol 1e-4)
3. **Birkhoff projection idempotent:** `sinkhorn(sinkhorn(M)) ≈ sinkhorn(M)`
4. **mHC forward shape:** `(B, S, n=4, D)` in → `(B, S, n=4, D)` out
5. **Identity at init:** with `A=B=C=I_n`, `F_l=0`, output equals input
6. **PyTorch golden parity:** load `~/sources/TileKernels/tile_kernels/torch/` mhc reference, run same input, assert max abs diff < 1e-3
7. **n_streams=1 backward-compat:** matches existing `ManifoldBranchMixer` output byte-for-byte
8. **YAML round-trip**

## Verification

- Numerical parity with TileKernels torch reference
- `pytest -q tests/test_mhc_v4.py tests/test_hybrid_lm_extensions.py`
- **Convergence smoke:** tiny model with mHC n=4 vs n=1 — loss curves comparable

## Commit message

`feat(mhc): upgrade ManifoldBranchMixer to DeepSeek-V4 mHC with Sinkhorn-Knopp Birkhoff projection`

---

# ROI 5 — FlashMLA absorb trick

## Why

DeepSeek's MLA decode kernel folds the `W_UK · W_O` matmul into the QK GEMM via the **absorption trick**. This **halves decode FLOPs** for MLA without changing accuracy. Apple Silicon won't get FlashMLA's Hopper kernel, but the algebraic trick ports cleanly.

**Sources:**
- `~/sources/rent_kernels/FlashMLA/docs/20250422-new-kernel-deep-dive.md` — explains absorb trick
- `~/sources/rent_kernels/FlashMLA/flash_mla/flash_mla_interface.py` — Python API for shape hints

## Where

- `cppmega_mlx/nn/attention.py` — extend `CausalSelfAttention` for `mode="mla"` decode path

## Step-by-step

1. **Read** `FlashMLA/docs/20250422-new-kernel-deep-dive.md` carefully — the trick is: at decode time, instead of computing `output = softmax(Q K^T) V W_O`, compute `output = softmax(Q' K^T) V'` where `Q' = Q W_UK^T` and `V'` already absorbs `W_O`. The compressed-latent KV is the natural fit.
2. **Add method** `CausalSelfAttention.decode_absorbed(...)` — applies only when `mode in ("mla", "full", "gqa")` and `kv_cache is not None`.
3. **Config flag** `mla_absorb: bool = True` (default on for MLA-mode decode).
4. **Forward branch:** when `kv_cache is not None and mla_absorb and cache_position > 0`, route through `decode_absorbed`; otherwise use existing dense SDPA.
5. **Validate** that the absorbed-projection produces numerically identical (or near-identical) outputs to the dense path on a tiny test.

## Tests (`tests/test_mla_absorb.py` — new)

1. **Numerical parity:** prefill with `mla_absorb=True` vs `False` — outputs identical (tol 1e-5)
2. **Decode parity:** same prompt prefilled, then 5 decode steps — outputs identical
3. **FLOP reduction sanity:** count matmul ops via instrumented forward — `decode_absorbed` does fewer than the dense path
4. **Existing attention tests pass** with `mla_absorb=True` as default
5. **Backward-compat:** with `mla_absorb=False`, byte-identical to pre-PR

## Verification

- `pytest tests/test_mla_absorb.py tests/test_attention*.py tests/test_hybrid_lm*.py`
- Decode throughput micro-bench: tokens/sec increases on a 1B model

## Commit message

`perf(mla): add FlashMLA absorb trick (W_UK·W_O fused into QK) for decode`

---

# ROI 6 — Engram TileKernels-grade upgrade

## Why

Our `EngramBranch` is a clean reference but doesn't have fused RMSNorm + gate + grad-reduce that TileKernels ships. DeepSeek V4's **Engram is a conditional memory module** that offloads static knowledge from HBM→DRAM with gated lookup, cutting HBM 60% and giving 2-3× inference speedup. **On Apple Silicon, unified memory means the HBM↔DRAM split is "free"** — Engram becomes a particularly natural fit.

**Sources:**
- TileKernels modules: `~/sources/TileKernels/tile_kernels/engram/`
  - `engram_fused_weight_kernel.py`
  - `engram_gate_kernel.py`
  - `engram_grad_w_reduce_kernel.py`
  - `engram_hash_kernel.py`
- Golden: `~/sources/TileKernels/tile_kernels/torch/` engram refs

## Where

- `cppmega_mlx/nn/engram.py` — extend `EngramBranch` with fused-style path
- Optionally `cppmega_mlx/nn/_metal/engram_fused.metal` — Metal shader for the fused gate+RMSNorm (advanced)

## Step-by-step

1. **Read** all 4 TileKernels engram files + the torch reference to understand the fused pipeline.
2. **Add config:**
   - `engram_use_hash: bool = False` (hash-routed lookup for memory module)
   - `engram_hash_size: int = 0` (size of hash table; 0 disables)
   - `engram_fused: bool = False` (use fused gate+RMSNorm)
3. **Implement** `engram_fused_forward(x, weight, gate, eps)` — RMSNorm + gate fused in pure MLX first (correctness baseline).
4. **(Optional) Metal shader path** under `cppmega_mlx/nn/_metal/engram_fused.metal` invoked via `mx.fast` custom kernel API — only if MLX exposes that path, otherwise stay in pure-MLX.
5. **Hash-routed branch:** small embedding table + hash function for memory-style lookup (`engram_hash_kernel.py` semantics).
6. **Apple-specific:** mark static-knowledge weights as `mx.array` allocated in `mx.metal.set_memory_limit`-compatible region for explicit page locality (research, may not be needed).

## Tests (`tests/test_engram_v4.py` — new)

1. **Fused path numerical parity:** `engram_fused=True` matches `False` (which uses 2-stage RMSNorm+gate) within tol 1e-5
2. **Hash routing:** with `engram_use_hash=True`, output shape preserved; collisions handled deterministically
3. **Backward gradient parity:** fused vs unfused, max abs diff < 1e-4
4. **doc_ids still threads through** (regression from ROI engram doc_ids work)
5. **YAML round-trip**

## Verification

- All new + existing engram tests pass
- Throughput micro-bench shows fused path ≥ unfused (or at least no regression on Apple Silicon)

## Commit message

`feat(engram): add fused gate+RMSNorm path + optional hash routing (DeepSeek-V4 Engram)`

---

# ROI 7 — DSA Lightning Indexer (V3.2-style)

## Why

DeepSeek V3.2's **DSA** is two-stage: a cheap FP8 **Lightning Indexer** (small head_dim) scores all KV positions → top-k indices → sparse MLA on the selected KV. Our `mode="dsa"` already has sparse MLA with `sparse_topk` — but the *indexer* itself is a separate, cheap kernel that selects the top-k. Right now we use a heuristic (causal nearest-k). Replacing it with a learned indexer is a meaningful upgrade.

**Sources:**
- `~/sources/rent_kernels/DeepSeek-V3.2-Exp/` — custom modeling code with reference DSA
- `~/sources/rent_kernels/DeepGEMM/deep_gemm/include/` — Lightning Indexer logit kernels (CUDA, reference only)
- `~/sources/rent_kernels/tilelang/examples/deepseek_v32/sparse_mla_fwd_seesaw.py` — TileLang reference (PR #1636)
- Critical gotcha (from V3.2 README): **RoPE in indexer is NON-interleaved, MLA RoPE is interleaved**

## Where

- `cppmega_mlx/nn/lightning_indexer.py` — new module
- `cppmega_mlx/nn/attention.py` — wire indexer into `mode="dsa"` path
- `cppmega_mlx/nn/_tilelang/sparse_mla_path_c.py` — already exists, may need indexer integration

## Step-by-step

1. **Read** V3.2-Exp modeling code, esp. the indexer init + forward.
2. **Implement** `LightningIndexer(nn.Module)`:
   - Small head dim (e.g. 32), FP8 quant
   - Q projection from hidden states (NON-interleaved RoPE — careful!)
   - K projection from KV cache (NON-interleaved RoPE)
   - Score = `Q · K^T` (FP8 GEMM, falls back to fp16 on Apple)
   - Top-k indices output
3. **Integrate** into `CausalSelfAttention` when `mode="dsa"`: instead of `causal_sparse_indices` (heuristic), call `LightningIndexer` to get learned top-k indices, then feed those into existing sparse MLA path.
4. **Config flag:** `indexer_type: Literal["causal_heuristic", "lightning"] = "causal_heuristic"` (default = backward-compat).

## Tests (`tests/test_lightning_indexer.py` — new)

1. **RoPE non-interleaved correctness:** indexer Q/K produce expected rotation pattern
2. **Top-k output shape:** `(B, S, kv_group, topk)` int32, all valid indices
3. **Causal validity:** indices are always ≤ current position
4. **End-to-end DSA forward** with `indexer_type="lightning"` vs `"causal_heuristic"` — shapes match, outputs differ (sanity)
5. **YAML round-trip**

## Verification

- All tests pass
- Sparse-MLA Path C tests don't regress
- Optional: train tiny model with learned indexer vs heuristic — learned should match or beat heuristic loss

## Commit message

`feat(attention): add Lightning Indexer for DSA top-k selection (DeepSeek-V3.2)`

---

# ROI 8 — NSA (Native Sparse Attention) — research spike

## Why

DeepSeek's NSA (arxiv 2502.11089) is **hardware-aligned trainable sparse attention** for long-context. Three branches: Compress (coarse), Select (top-k fine), Sliding (window). Used in V4 alongside DSA. Research-grade; not critical until we want >32k contexts.

**Sources:**
- Paper: arxiv 2502.11089
- Alternative: arxiv 2508.18224 (Flash Sparse Attention, GQA-friendly)
- Community: `fla-org/native-sparse-attention` (if exists)

## Plan

Spike to validate it fits our MLX/Metal constraints. Out-of-scope for v1 integration.

---

# ROI 9 — CSA + HCA hybrid attention stack (V4)

## Why

V4 attention is **CSA + HCA + MLA** — Compressed Sparse Attention (m-token KV compression + Lightning Indexer top-k) and Heavily Compressed Attention. Combined with mHC residual streams. This is the full V4-Pro architecture.

**Sources:**
- arxiv 2512.24880 (mHC paper)
- HF blog: `RadicalNotionAI/mhc-ablation-challenges`
- V4-Pro config dump (feimatrix)

## Plan

Spike after ROI 1-7 are in. The full hybrid stack is research-grade and depends on having all the sub-components working first.

---

# Cross-cutting infrastructure

## YAML serialization regression

Every ROI adds config fields. Each PR must:
- Update `HybridTinyConfig.to_yaml()` / `from_yaml()` if a new tuple-typed field is added (coerce in `from_dict`)
- Add the new field to `tests/test_checkpoint.py::test_checkpoint_resume_restores_hybrid_custom_blocks_and_optimizer` list-vs-tuple expectations

## Pattern symbol registry

Each new symbol (L, future) updates:
- `cppmega_mlx/recipes/pattern.py`: `NamSymbol`, `LayerRole`, `SUPPORTED_NAM_SYMBOLS`, `ORDERED_NAM_SYMBOLS`, `_ROLE_BY_SYMBOL`, role-list helpers
- `cppmega_mlx/models/hybrid_lm.py`: `HybridBackend`, `HybridBlockModule`, `_ROUTE_SYMBOL_BACKENDS`, `HybridTinyBlock.__init__`, `route_delta`, `validate_backend.expected_cls`
- `tests/test_nam56r_pattern.py`: `counts` and `role_counts` dict expectations
- `tests/test_cppmega_parity_anchors.py`: same

## Beads tracking

One beads issue per ROI:
```
bd create --title="ROI 1: MoE aux-loss-free balancing + sqrtsoftplus" --type=feature --priority=2
bd create --title="ROI 2: MTP SequentialMTPHead" --type=feature --priority=2
bd create --title="ROI 3.A: GDN 'L' Path A pure-MLX reference + block scaffold" --type=feature --priority=2
bd create --title="ROI 3.B: GDN Path B hand-MSL via mx.fast.metal_kernel" --type=feature --priority=2
bd create --title="ROI 3.C: GDN Path C TileLang-DSL via tilelang.compile target=metal" --type=feature --priority=2
bd create --title="ROI 3.D: GDN Path D Triton frontend lift of FLA kernel" --type=feature --priority=3
bd create --title="ROI 3.E: GDN Path E vendor mlx-lm PR #1217" --type=feature --priority=3
bd create --title="ROI 3.F: GDN auto-mode dispatch + kernel_policy entry" --type=feature --priority=2
bd create --title="ROI 3.5.A: KDA 'K' Path A pure-MLX reference + block scaffold" --type=feature --priority=2
bd create --title="ROI 3.5.B: KDA Path B hand-MSL" --type=feature --priority=2
bd create --title="ROI 3.5.C: KDA Path C TileLang-DSL (lift 11 tilelang/examples/kda prims)" --type=feature --priority=2
bd create --title="ROI 3.5.D: KDA Path D Triton frontend" --type=feature --priority=3
bd create --title="ROI 3.7: GDN+KDA head-to-head benchmark + auto-promotion receipt" --type=feature --priority=2
bd create --title="ROI 4: mHC Sinkhorn Birkhoff" --type=feature --priority=2
bd create --title="ROI 5: FlashMLA absorb trick" --type=feature --priority=2
bd create --title="ROI 6: Engram TileKernels-grade" --type=feature --priority=3
bd create --title="ROI 7: DSA Lightning Indexer" --type=feature --priority=3
bd create --title="ROI 8: NSA research spike" --type=task --priority=4
bd create --title="ROI 9: CSA+HCA hybrid V4 attention" --type=task --priority=4
```

## CI / regression invariants

After every ROI merge:
- `pytest tests/ --ignore=tests/test_hybrid_lm_gradients.py` baseline: ≥ 2005 passed (pre-existing 2 dsa_path_c failures stay as-is, unrelated)
- `pytest tests/test_hybrid_lm_extensions.py` strict: 100% pass
- YAML roundtrip with **all** new fields included
- Checkpoint round-trip end-to-end

## Repo references

Cloned in `/Volumes/external/sources/rent_kernels/`:
- `FlashMLA/`, `DeepGEMM/`, `DeepSeek-V3/`, `DeepSeek-V3.2-Exp/`
- `flash-linear-attention/`, `FlashQLA/`, `mlx-recurrence/`, `hipfire/`, `mamba.py/`, `tilelang/`

Plus `~/sources/TileKernels/` (already cloned).

---

## Definition of done

A ROI is **done** when:
1. Code merged to main with one atomic commit
2. New tests added + passing
3. Existing tests (esp. checkpoint, pattern parity, YAML round-trip) green
4. Beads issue closed
5. Pushed to origin
6. README or this plan updated with status
