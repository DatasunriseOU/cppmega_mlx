# NAM56R full readiness assessment — MLX port

**Status**: assessment (2026-08-01)  
**Scope**: `cppmega_mlx` native construction, inference, and training of the
full NAM56R 52-layer hybrid model described in `cppmega`.

---

## 1. What "full readiness" means

A production-ready MLX NAM56R must be able to:

1. Construct a 52-layer model matching the source Megatron topology
   (`AEMEAEMEAEMR`, depth 52, DSA ranks `1,2,3,5,6,7,9,10,11`).
2. Load weights converted from a Megatron checkpoint without shape/key errors.
3. Run forward and backward passes at NAM56R dimensions on the target hardware
   (GB10 / M4 Max / CUDA) within the memory envelope.
4. Produce numerically acceptable output versus the Megatron reference for the
   same input tokens and graph sidecar.
5. Support the production training recipe: optimiser, data pipeline, loss
   scaling, checkpointing, and eval harness.

This doc scores each block as **landed**, **partial**, or **missing**, and
lists the concrete gaps that block a readiness claim.

---

## 2. Landed blocks

### 2.1 Layer pattern and routing

- **Files**: `cppmega_mlx/recipes/pattern.py`, `cppmega_mlx/recipes/nam56r.py`
- **Tests**: `tests/test_nam56r_pattern.py` (11 passed)
- **Status**: landed.
- **Evidence**:
  - `build_nam56r_pattern()` expands `AEMEAEMEAEMR` to depth 52 and returns
    correct counts: 13 attention, 22 MoE, 13 Mamba3, 4 M2RNN.
  - DSA A-rank routing `1,2,3,5,6,7,9,10,11` maps to layers
    `(5, 9, 13, 21, 25, 29, 37, 41, 45)`; the remaining attention layers are
    MLA `(1, 17, 33, 49)`.
  - `Nam56RParityContract` explicitly refuses to claim fully native Megatron
    parity because Mamba3 and M2RNN are custom seams and upstream symbols
    `D`, `G`, `|` are unsupported.

### 2.2 Configuration translation

- **Files**: `cppmega_mlx/recipes/nam56r.py::build_hybrid_tiny_config_from_nam56r`
- **Tests**: `tests/test_nam56r_pattern.py::test_nam56r_recipe_*`
- **Status**: landed for smoke dimensions.
- **Evidence**:
  - The recipe maps NAM56R config fields (`hidden_size`, `moe`, `mamba3`,
    `m2rnn`, `source_structure_env`, `ngram_hash`) into `HybridTinyConfig`.
  - It preserves the pattern, DSA rank routing, and structure/ngram-hash
    parameters.

### 2.3 Graph-route data structures

- **Files**: `cppmega_mlx/nn/domain_graph_routes.py`,
  `cppmega_mlx/nn/code_graph_routes.py`
- **Status**: landed for data parsing.
- **Evidence**:
  - Domain and code graph route tensors can be built from sidecar dictionaries
    and fed into the model.
  - Numerical parity with the CUDA graph-route attention bias is tracked
    separately (see §3.4).

---

## 3. Partial blocks

### 3.1 Attention (MLA + DSA)

- **Files**: `cppmega_mlx/nn/attention.py`, `cppmega_mlx/nn/sparse_mla.py`
- **Status**: partial.
- **What works**:
  - `DenseCppLM` has a dense attention path for smoke tests.
  - `sparse_mla.py` contains the local sparse-MLA indexer scaffolding.
- **Gaps**:
  - **MLA**: no absorbed-MLA or fused-MLA kernel matching Megatron's
    `FusedMLASelfAttention` / `AbsorbedMLASelfAttention`.  Memory and FLOPs
    are therefore not at NAM56R production levels.
  - **DSA**: the sparse indexer exists but has not been proven at NAM56R
    `topk=256` scale against the CUDA reference on real weights.
  - **Window size / SWA**: not wired through the MLX attention modules;
    `window_size` is a CUDA/TE concept today.
  - **Graph-route attention bias**: the CUDA side has
    `CppMegaFA4ScoreModAttention` + `ChunkNativeGraphBias`; MLX has no
    equivalent fused kernel.

### 3.2 Mamba3

- **Files**: `cppmega_mlx/nn/mamba3.py`,
  `cppmega_mlx/nn/_tilelang/mamba3*.py`
- **Status**: partial (TileLang Phase 4 in progress, P041–P046).
- **What works**:
  - Reference scan logic, chunked precompute core, and Path C TileLang kernels
    exist.
  - `tests/test_mamba3_*` exercise small shapes.
- **Gaps**:
  - **MIMO 3D → 2D shared-memory refactor** (P046) is not closed.
  - **Path B / `_msl_transform.py`** removal is pending strict parity.
  - **Mamba3 CP / TP**: not implemented on MLX.
  - **Numerical parity at NAM56R hidden size**: not demonstrated end-to-end.

### 3.3 M2RNN

- **Files**: `cppmega_mlx/nn/m2rnn.py`, `cppmega_mlx/nn/_tilelang/m2rnn*.py`
- **Status**: partial.
- **What works**:
  - Python reference and Path C TileLang kernels exist.
- **Gaps**:
  - No end-to-end NAM56R-layer parity versus CUDA.
  - Runtime backward chunking and memory reclaim need profiling at full size.

### 3.4 MoE

- **Files**: `cppmega_mlx/nn/mhc.py`, `cppmega_mlx/models/hybrid_lm.py`
- **Status**: partial.
- **What works**:
  - `HybridTinyConfig` accepts `moe_num_experts`, `moe_top_k`, etc.
  - Basic MoE routing scaffolding is present.
- **Gaps**:
  - No GB10-optimised grouped GEMM for the 22 MoE layers at NAM56R scale.
  - EP (expert parallelism) and capacity-factor handling are not implemented.

### 3.5 Checkpoint conversion

- **Files**: `docs/mtr005_megatron_dcp_to_mlx.md`, conversion scripts in
  `scripts/`
- **Status**: partial.
- **What works**:
  - Design doc and one-off converters exist.
- **Gaps**:
  - No continuously validated pipeline from Megatron DCP → MLX checkpoint.
  - Key mapping for custom seams (Mamba3, M2RNN, structure embeds) is
    hand-maintained and brittle.

### 3.6 Training loop

- **Files**: `cppmega_mlx/models/stable_loop_cpp_lm.py`,
  `cppmega_mlx/models/code_loop_world_model.py`
- **Status**: partial.
- **What works**:
  - Stable loop and code-loop world-model trainers run for tiny models.
- **Gaps**:
  - No full NAM56R training recipe with data loading, loss scaling, and
    checkpointing at production dimensions.
  - Optimiser state sharding (FSDP/ZeRO-1) is not implemented.

---

## 4. Missing blocks

| Block | Why it blocks readiness | Tracking |
|---|---|---|
| **FP8 training on Metal** | NAM56R production targets MXFP8 on GB10; MLX/Metal has no native FP8 training path. | `docs/FP8-ACTIVATIONS-PATHC.md`, `MLX-INTERNAL-MONO-FUSION.md` |
| **Context parallelism** | 128k extension phase requires CP; MLX has no CP implementation. | `cppmega/docs/document_isolation_cp128k_design.md` |
| **Distributed training** | No Megatron-style TP/PP/EP on MLX. | `docs/distributed_zero1_smoke_procedure.md` |
| **M4 Max ↔ GB10 parity** | Need matched rows proving numerical equivalence across Metal and CUDA. | P095 |
| **Production eval harness** | Need perplexity / downstream evals on converted NAM56R weights. | `tests/test_train_eval_graph_routes.py` |
| **Memory model** | No proven max_alloc budget for full NAM56R on target hardware. | `docs/research/precision_strategy_decision.md` |

---

## 5. Readiness verdict

**NAM56R is not fully ready in `cppmega_mlx`.**

The *recipe and pattern topology* are correct and tested.  The *model skeleton*
can be built.  However, the *compute kernels* that make NAM56R feasible
(absorbed MLA, DSA sparse attention at scale, grouped MoE GEMM, Mamba3/M2RNN
Path C parity, FP8) are either partial or missing.

### 5.1 What can be claimed today

- "MLX NAM56R pattern and routing match the source Megatron layout."
- "Tiny/smoke NAM56R-shaped models can be constructed and trained locally."
- "Full native Megatron parity is explicitly not claimed."

### 5.2 What cannot be claimed today

- "MLX NAM56R trains at production dimensions."
- "MLX NAM56R matches CUDA numerics on real checkpoints."
- "MLX NAM56R supports 128k context / CP / FP8 / distributed training."

---

## 6. Recommended next steps

1. **Close TileLang Phase 4** (P041–P046) — this unblocks Mamba3/M2RNN parity
   and removes the legacy Path B kernels.
2. **MLA/DSA parity** — port the CUDA FA4 `ChunkNativeGraphBias` concept to MLX
   or prove the existing `sparse_mla.py` path at NAM56R scale.
3. **MoE grouped GEMM** — profile `mhc.py` at NAM56R dimensions; decide whether
   to use MLX primitives, custom Metal, orTileLang.
4. **Checkpoint converter CI** — automate Megatron DCP → MLX and add a
   shape/key sanity test on every relevant upstream commit.
5. **Memory envelope** — run a single forward/backward step at quarter/half
   NAM56R on GB10 and M4 Max to validate the precision-strategy assumptions.
6. **Defer CP/FP8/distributed** until steps 1–5 close; these are hard
   dependencies for 128k and production training but not for local parity
   debugging.

---

## 7. References

- `cppmega_mlx/recipes/nam56r.py`
- `cppmega_mlx/recipes/pattern.py`
- `cppmega_mlx/models/hybrid_lm.py`
- `cppmega_mlx/nn/attention.py`
- `cppmega_mlx/nn/sparse_mla.py`
- `cppmega_mlx/nn/mamba3.py`
- `cppmega_mlx/nn/m2rnn.py`
- `cppmega_mlx/nn/mhc.py`
- `tests/test_nam56r_pattern.py`
- `docs/mlx_port_master_plan.md`
- `docs/research/precision_strategy_decision.md`
- `docs/mtr005_megatron_dcp_to_mlx.md`
