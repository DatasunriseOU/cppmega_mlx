# cppmega Parity Anchors

Date: 2026-04-30

This file records the cppmega CUDA/Megatron contracts that the MLX port must
preserve. It is not a request to port Megatron, Transformer Engine, Triton,
TileLang, CUDA graphs, NCCL, or CUDA kernels to macOS. On Apple Silicon, the
runtime target remains MLX plus optional Metal kernels with pure-MLX fallbacks.

## External Mac Runtime Basis

- MLX is the local array, module, optimizer, compile, and custom-Metal-kernel
  substrate for macOS.
- MLX-LM is a reference for Apple Silicon LLM training patterns, especially
  local fine-tuning and optimizer/update-loop conventions.
- Hugging Face Apple/Metal kernels are references for kernel packaging and
  experiments on M-series hardware. They are not dependencies of the training
  path until parity, backward behavior, dtype behavior, and benchmark evidence
  are proven locally.
- Local GB10 Parquet files under `data/parquet_samples/` are ignored smoke
  inputs for data-contract tests. They are not source-of-truth anchors and must
  not be treated as committed fixtures or benchmark rows.

## NAM56R Constants

These constants are model-layout anchors, not CUDA runtime anchors.

| Anchor | Value |
| --- | --- |
| Source pattern | `AEMEAEMEAEMR` |
| Depth | `52` |
| Expanded counts | `A=13`, `E=22`, `M=13`, `R=4` |
| R layer numbers | `12,24,36,48` |
| A layer numbers | `1,5,9,13,17,21,25,29,33,37,41,45,49` |
| DSA rank tuple preserved from launchers | `1,2,3,5,6,7,9,10,11` |
| cppmega launcher-indexed DSA layer numbers | `5,9,13,21,25,29,37,41,45` |
| cppmega launcher-indexed MLA layer numbers | `1,17,33,49` |
| MoE defaults | `16` experts, `top_k=4`, routed hidden `896`, shared hidden `1024` |
| Vocab contracts | local/profile `65536`, megacpp tokenizer `131072` |

The DSA tuple deserves special handling. The MLX helper treats
`dsa_a_layer_ranks` as zero-based A-layer indices, matching the production
cppmega H200 launchers, which compute absolute layers as `attn_nums[r]`, and
`CppMegaSelectiveAttentionLayer`, which derives `a_rank` with
`attention_layer_numbers.index(layer_idx)`. That indexed contract yields the
`9 DSA + 4 MLA` split above.

## Source Reference Anchors

These files in `../cppmega` are the current source references for parity:

| Source | Anchor contract | Not being ported as-is |
| --- | --- | --- |
| `cppmega/megatron/nam56r_layout.py` | Import-safe NAM56R pattern/depth helpers, R layer loading, A-layer number loading, and DSA rank env parsing. | Megatron symbol probing is only source-side compatibility logic. |
| `cppmega/recipes/nam56r_megatron.py` | Full source parser accepts `A/M/D/E/G/R/|` and reports whether a recipe is fully native Megatron. | Local MLX pattern parsing intentionally accepts only `A/E/M/R` and fails closed on upstream-only symbols. |
| `cppmega/recipes/megatron_args.py` | Emits Megatron CLI arguments for MLA, MTP, MoE, DSA, and bf16-only DSA indexer settings. | Megatron launcher flags, CUDA graph, distributed optimizer, and DSA indexer runtime. |
| `cppmega/recipes/nam56r_launch.py` | NAM symbol to Megatron hybrid pattern mapping and R custom-layer index derivation. | Megatron hybrid-layer CLI construction is not the MLX runtime. |
| `cppmega/recipes/nam56r_nemo_recipe.py` | MoE defaults and local/profile vocab constant. | NeMo/Megatron CLI recipes and distributed launch policy. |
| `cppmega/megatron/nam56r_full_spec.py` | Full NAM56R Megatron recipe/runtime anchors: mixed MLA/DSA A-layers, MTP dense-attention override, Author Mamba3 or TP Mamba3 mixer selection, M2RNN-on-R placement, and TE provider requirement. | Transformer Engine provider, native MLA/DSA/MTP module specs, TP mixer runtime, CUDA execution, and distributed Megatron graph behavior. |
| `cppmega/megatron/nam56r_te_spec.py` | TE-preserving NAM56R spec that only swaps the Mamba mixer to Author Mamba3 or M2RNN by source layer placement. | Transformer Engine attention/MoE/MTP submodules and Megatron runtime execution. |
| `cppmega/megatron/nam56r_noconv_spec.py` | No-conv Mamba3 B/C feature branch with R layers still routed to `CppMegaM2RNNMixer`. | TE fused submodules, vanilla SSD/Triton scan path, and full Author Mamba3 feature parity. |
| `cppmega/megatron/mamba3_te_stack_spec.py` | Upstream TE stack spec replacing only the Mamba mixer with the cppmega Mamba3 mixer. | TE GatedDeltaNet, MLP, MTP, and Megatron ModuleSpec execution. |
| `scripts/remote_production_h200_nam56r_v1.sh` | Production defaults for pattern, depth, R layers, ngram hash, structure, DSA tuple, and the indexed DSA/MLA preflight. | H200 launcher, CUDA graph, NCCL, TE, and distributed shell runtime. |
| `scripts/remote_sweep_h200_dsa_production.sh` | Same DSA 9+4 indexed preflight used in sweep lanes. | Remote H200 sweep orchestration and CUDA-only runtime checks. |
| `cppmega/megatron/m2rnn_spec.py` | M2RNN feature semantics: projection split, recurrence, decay gate, residual path, and training-only limitations. | Megatron `ModuleSpec`, Transformer Engine norm, Triton scan kernel, and CUDA training execution. |
| `cppmega/megatron/mamba3_te_in_proj.py` | Author Mamba3 in-projection slices `[z,x,B,C,dd_dt,dd_A,trap,angles]` and TP partition-size semantics. | `TELayerNormColumnParallelLinear` and Transformer Engine checkpoint resharding runtime. |
| `cppmega/megatron/mamba3_mixer.py` | Mamba3 feature semantics: QK norm on B/C, B/C bias, optional data-dependent A, and split path packed `xBC` causal conv before `[x,B,C]` split. | Megatron `MambaMixer`, `mamba_ssm` Triton kernels, CUDA graph compatibility mechanics. |
| `cppmega/megatron/mamba3_te_mixer.py` | Author Mamba3 kernel path: `[z,x,B,C,dd_dt,dd_A,trap,angles]` projection feeds data-dependent-A, trapezoidal, RoPE/angle, and SISO/MIMO scan kernels. | Transformer Engine, Triton/TileLang scan kernels, CUDA inference step kernels, and distributed TP runtime. |
| `cppmega/features/engram/ngram_hash.py` | Additive n-gram hash enrichment defaults: orders `(2,3)`, heads `8`, table size `500000`, embed dim `16`. | Torch module implementation and CUDA execution. |
| `cppmega/features/structure/embedding.py` | Additive structure embedding components, especially `core = structure, dep_level`. | Torch module implementation and CUDA execution. |
| `cppmega/megatron/custom_embedding.py` | Optional env-gated ngram and structure enrichments added beside Megatron token embeddings. | Megatron `LanguageModelEmbedding`, pipeline-stage sharded-state mechanics, and CUDA dropout/sequence-parallel runtime. |
| `cppmega/megatron/structure_batch.py` | Source-side structure side-channel key set: `structure_ids`, `dep_levels`, `ast_depth_ids`, `sibling_index_ids`, and `node_type_ids`. | Torch batch plumbing and Megatron data-loader integration. |
| `cppmega/megatron/custom_gpt_model.py` | Source-side handoff for setting cppmega structure inputs on the GPT model before the embedding layer consumes them. | Megatron GPT model execution, pipeline parallelism, and CUDA runtime. |
| `cppmega/megatron/fastmtp_layer.py` | Optional `CPPMEGA_FASTMTP=1` Torch/Megatron FastMTP path with checkpointing, cadence, and optional Liger CE. | Native MLX MTP layer, Megatron MTP monkey patching, Liger CE, and production MTP scheduling. |
| `cppmega/megatron/mtp_native_hopper_ce.py` | Hopper/Megatron native linear-CE path for main-plus-MTP loss fusion. | Hopper/GB10 CE kernels, native Megatron CE patching, and fused main+MTP launch behavior. |
| `cppmega/megatron/dsa_local_spec.py` | Source-side helper for validating official Megatron DSA before copying residual behavior. | Local native DSA implementation or sparse-MLA training kernel. |
| `cppmega/megatron/dsa_sparse_attention.py` | CUDA/Torch sparse gather-scatter DSA attention replacement for Megatron's unfused DSA function. | Sparse DSA/MLA Metal kernel and differentiable MLX training implementation. |
| `cppmega/megatron/moe_dispatcher_patch.py` | Runtime monkey patch around Megatron/Transformer Engine MoE dispatcher behavior. | Megatron all-to-all dispatcher parity, grouped-GEMM scheduling, and expert-parallel overlap. |
| `cppmega/megatron/selective_fp8_moe_patch.py` | Selective FP8 gating for MoE layers while attention/Mamba/R layers remain bf16. | FP8/NVFP4 training, Transformer Engine FP8 contexts, and local MLX mixed-precision dispatcher parity. |
| `scripts/data_prep_parquet_to_megatron.py` | Parquet-to-Megatron conversion currently reads a configurable token column whose default is `token_ids`. | Structure side-channel conversion; the converter does not emit sidecar metadata for the MLX reader. |
| `scripts/remote_smoke_h200_dsa_9_4_m.sh` | H200 DSA/MLA 9+4 smoke anchor with ngram/structure env flags and source-side runtime patches. | Local MLX launcher support, CUDA runtime, and distributed Megatron execution. |
| `scripts/remote_smoke_h200_nam56r_k_pp1.sh` | H200 NAM56R smoke anchor for launcher env flags, native MLA/MoE/MTP/DSA args, and 9+4 DSA/MLA preflight. | Local MLX launcher support, CUDA runtime, and distributed Megatron execution. |
| `scripts/remote_train_gb10_nam56r_single.sh` | GB10 NAM56R source-side train script anchor. | GB10 performance parity or local MLX launcher support. |
| `scripts/remote_train_h200_nam56r_full.sh` | H200 full NAM56R train script anchor. | H200 distributed Megatron launcher parity. |
| `scripts/remote_train_h200_nam56r_lite.sh` | H200 lite NAM56R train script anchor. | Local tiny smoke equivalence to H200 lite training. |
| `scripts/remote_train_h200_nam56r_grid.sh` | H200 NAM56R grid train script anchor. | Local distributed sweep orchestration. |
| `scripts/remote_train_h200_nam56r_tp2.sh` | H200 NAM56R TP=2 train script anchor. | MLX tensor-parallel parity. |
| `scripts/remote_train_h200_nam56r_noconv.sh` | H200 no-conv NAM56R train script anchor. | MLX no-conv branch performance or feature parity. |
| `scripts/remote_train_h200_nam56r_europe_sweep.sh` | H200 Europe sweep train script anchor. | Remote fleet orchestration or benchmark parity. |

## Full NAM56R Megatron Recipe/Runtime Anchors

The source NAM56R recipe is broader than the local MLX subset. In
`../cppmega`, full recipe/runtime coverage spans the parser and argument
emitters, the full/TE/no-conv Megatron specs, Mamba3 and M2RNN mixer placement,
native MLA/MTP/DSA module specs, FastMTP, Hopper/GB10 CE patching, DSA sparse
attention, MoE dispatcher patches, selective FP8 MoE, and H200/GB10 train
scripts. Those anchors are evidence for what not to overclaim locally.

Current MLX coverage for NAM56R remains the fail-closed `A/E/M/R` layout,
zero-based A-layer DSA rank mapping, tiny reference attention/MoE/Mamba3/M2RNN
blocks, local side-channel ingress, and local checkpoint/eval/profile plumbing.
It does not include Transformer Engine, CUDA graph capture, NCCL,
Triton/TileLang scan kernels, native MTP, native DSA, sparse MLA, Hopper/GB10
linear-CE kernels, or TP/PP/VPP/EP/SP/distributed optimizer behavior.

## Custom Seam Gap Table

This section tracks cppmega-owned contracts that are not yet fully represented
in the MLX local path. Current MLX coverage is local/tiny/partial: it is enough
for small correctness, import, data-contract, and smoke-training tests, but it
does not prove full NAM56R, H200, GB10, CUDA, or distributed Megatron behavior.
H200 scripts are source/runtime anchors for `../cppmega`, not MLX-supported
launchers.

### Structure Runtime Handoff

- Source anchors: `cppmega/megatron/structure_batch.py`,
  `cppmega/megatron/custom_gpt_model.py`, and
  `cppmega/megatron/custom_embedding.py`.
- Current MLX coverage: `LMTokenBatch` carries `structure_ids`, `dep_levels`,
  `ast_depth_ids`, `sibling_index_ids`, and `node_type_ids`; NPZ, Parquet, and
  Megatron-indexed readers can pass those keys through `model_kwargs()`.
  `HybridTinyLM` consumes those keys through `CppMegaStructureEmbedding`.
- Missing contract: end-to-end training/script handoff is still tiny-local, and
  `TinyLM` still uses simplified structure addition rather than full source
  embedding semantics.

### Megatron Indexed Side-Channel Preservation

- Source anchor: `scripts/data_prep_parquet_to_megatron.py`.
- Current MLX coverage: the indexed reader accepts `MMIDIDX` token shards, raw
  `.bin` handoffs, and canonical token-aligned binary sidecars for
  `attention_mask` plus all five structure arrays via `side_channel_paths` or
  direct top-level path entries. Integer sidecars are range-checked before int32
  MLX materialization, and ngram sidecars fail closed because hashes are derived
  from `input_ids` at the model seam.
- Missing contract: The source converter writes only `token_ids` to `.bin/.idx`;
  structure side channels are not guaranteed to survive Parquet-to-Megatron
  conversion, and no source multi-shard/packed sidecar schema is claimed.

### Full Structure Embedding Semantics

- Source anchors: `cppmega/features/structure/embedding.py` and
  `cppmega/megatron/custom_embedding.py`.
- Current MLX coverage: `cppmega_mlx/nn/structure_embedding.py` mirrors
  component parsing, clamping, bottleneck projection, zero init, component
  scales, and missing-component masking, and `HybridTinyLM` uses it on the
  forward path.
- Missing contract: `TinyLM` still uses a simplified shared `nn.Embedding`
  addition; full Megatron pipeline-stage and launcher behavior is not
  represented.

### Ngram Hash Model And Config Integration

- Source anchors: `cppmega/features/engram/ngram_hash.py`,
  `cppmega/megatron/custom_embedding.py`, and H200 launchers with
  `CPPMEGA_NGRAM_HASH_*`.
- Current MLX coverage: `cppmega_mlx/nn/ngram_hash.py` implements the local
  additive hash module and tests its default orders, heads, table size, and
  projection behavior. `HybridTinyLM` can add it to token embeddings, and
  `build_hybrid_tiny_config_from_nam56r()` maps the central NAM56R config
  defaults into that model path.
- Missing contract: source launcher/env ingestion and
  `CPPMEGA_NGRAM_HASH_OFFLOAD` runtime behavior are documented/configured but
  not implemented as local MLX runtime behavior; `TinyLM` does not consume ngram
  enrichment.

### Mamba3 Pure-MLX Reference

- Source anchors: `cppmega/megatron/mamba3_te_in_proj.py`,
  `cppmega/megatron/mamba3_mixer.py`, and
  `cppmega/megatron/mamba3_te_mixer.py`.
- Current MLX coverage: `Mamba3ReferenceBlock` preserves projection split
  sizes, trainable local recurrence, B/C QK norm and bias, data-dependent
  `dt/A` terms, and the source split-path packed `xBC` causal conv before B/C
  transform.
- Missing contract: projected `angles` are not consumed, and the Author RoPE,
  trapezoidal, SISO/MIMO scan kernel semantics are not yet implemented as pure
  MLX or as trainable Metal kernels with VJP coverage.

### Env, Launcher, And H200 Runtime Anchors

- Source anchors: `cppmega/megatron/custom_embedding.py`,
  `scripts/remote_production_h200_nam56r_v1.sh`,
  `scripts/remote_sweep_h200_dsa_production.sh`,
  `scripts/remote_smoke_h200_dsa_9_4_m.sh`, and
  `scripts/remote_smoke_h200_nam56r_k_pp1.sh`.
- Current MLX coverage: NAM layout and central dataclasses record
  `CPPMEGA_STRUCTURE_*` and `CPPMEGA_NGRAM_HASH_*` defaults, enable flags, and
  the ngram offload non-local runtime caveat.
- Missing contract: local scripts still do not ingest the full source launcher
  env contract, H200 launcher shell behavior is not an MLX feature, and
  distributed Megatron behavior remains outside the MLX scaffold. Distributed
  MLX training is not implemented, and distributed Megatron parity is not claimed.

### Parquet-To-Megatron Schema

- Source anchor: `scripts/data_prep_parquet_to_megatron.py`.
- Current MLX coverage: `TokenParquetDataset` can read side-channel columns
  when they already exist in Parquet, and `MegatronIndexedDataset` can read
  explicit token-aligned sidecars beside `.bin/.idx` shards.
- Missing contract: The source converter writes only `token_ids`; structure
  side channels are not preserved by that converter today.

### Megatron PP/MTP Checkpoint Layout

- Source anchor: `cppmega/megatron/custom_embedding.py` `sharded_state_dict()`.
- Current MLX coverage: MLX checkpoints are simple safetensors directories with
  metadata and no distributed Megatron dependency.
- Missing contract: pipeline-stage and MTP replica-id semantics for custom
  ngram/structure submodules are absent; keep this a documented non-goal unless
  import/export interop becomes required.

### Performance Claim Boundary

M4 Max vs GB10 parity is not proven by this document, by local M4 rows, or by
ignored local GB10 Parquet smoke inputs. Use the matched-row protocol in
`docs/porting_plan.md` and `docs/perf_baseline.md` before making any
cross-machine claim.

## MoE Trainability Audit Note

The local MLX MoE path is a correctness-first reference, not a Megatron
dispatcher clone. Targeted tests now prove gradients, parameter updates, and
AdamW state materialization for the router gate, all routed expert projections,
and the shared expert projections both in a standalone `ReferenceMoE` train step
and through the hybrid LM `E` route.

The remaining MoE parity risks stay source-runtime-only for this lane:
Megatron all-to-all/token dispatcher behavior, grouped GEMM scheduling,
capacity/drop-pad policy, expert-parallel overlap, identity chunk-sort
short-circuiting from `moe_dispatcher_patch.py`, and selective FP8 gating from
`selective_fp8_moe_patch.py` are not emulated by the MLX reference module.

## Porting Rule

Preserve model-facing semantics first: layer layout, route selection, config
constants, tensor shapes, enrichment inputs, and checkpoint-visible metadata.
Replace CUDA/Megatron runtime mechanisms with MLX-native modules, MLX compile
patterns, and optional Metal kernels only after pure-MLX parity tests exist.

Forward-only Metal kernels remain optional diagnostics or inference-style
experiments unless they define MLX custom-function VJP/JVP coverage for every
trainable path. CUDA-only optimizations should be documented as source
references, not silently emulated with weaker local behavior.
