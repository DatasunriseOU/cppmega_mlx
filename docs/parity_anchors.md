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
| `cppmega/recipes/nam56r_launch.py` | NAM symbol to Megatron hybrid pattern mapping and R custom-layer index derivation. | Megatron hybrid-layer CLI construction is not the MLX runtime. |
| `cppmega/recipes/nam56r_nemo_recipe.py` | MoE defaults and local/profile vocab constant. | NeMo/Megatron CLI recipes and distributed launch policy. |
| `scripts/remote_production_h200_nam56r_v1.sh` | Production defaults for pattern, depth, R layers, ngram hash, structure, DSA tuple, and the indexed DSA/MLA preflight. | H200 launcher, CUDA graph, NCCL, TE, and distributed shell runtime. |
| `scripts/remote_sweep_h200_dsa_production.sh` | Same DSA 9+4 indexed preflight used in sweep lanes. | Remote H200 sweep orchestration and CUDA-only runtime checks. |
| `cppmega/megatron/m2rnn_spec.py` | M2RNN feature semantics: projection split, recurrence, decay gate, residual path, and training-only limitations. | Megatron `ModuleSpec`, Transformer Engine norm, Triton scan kernel, and CUDA training execution. |
| `cppmega/megatron/mamba3_te_in_proj.py` | Author Mamba3 in-projection slices `[z,x,B,C,dd_dt,dd_A,trap,angles]` and TP partition-size semantics. | `TELayerNormColumnParallelLinear` and Transformer Engine checkpoint resharding runtime. |
| `cppmega/megatron/mamba3_mixer.py` | Mamba3 feature semantics: QK norm on B/C, B/C bias, optional data-dependent A, and split path packed `xBC` causal conv before `[x,B,C]` split. | Megatron `MambaMixer`, `mamba_ssm` Triton kernels, CUDA graph compatibility mechanics. |
| `cppmega/megatron/mamba3_te_mixer.py` | Author Mamba3 kernel path: `[z,x,B,C,dd_dt,dd_A,trap,angles]` projection feeds data-dependent-A, trapezoidal, RoPE/angle, and SISO/MIMO scan kernels. | Transformer Engine, Triton/TileLang scan kernels, CUDA inference step kernels, and distributed TP runtime. |
| `cppmega/features/engram/ngram_hash.py` | Additive n-gram hash enrichment defaults: orders `(2,3)`, heads `8`, table size `500000`, embed dim `16`. | Torch module implementation and CUDA execution. |
| `cppmega/features/structure/embedding.py` | Additive structure embedding components, especially `core = structure, dep_level`. | Torch module implementation and CUDA execution. |
| `cppmega/megatron/custom_embedding.py` | Optional env-gated ngram and structure enrichments added beside Megatron token embeddings. | Megatron `LanguageModelEmbedding`, pipeline-stage sharded-state mechanics, and CUDA dropout/sequence-parallel runtime. |

## Custom Seam Gap Table

This table tracks cppmega-owned contracts that are not yet fully represented in
the MLX local path. The MLX side may already have partial coverage; a gap means
the source contract is not yet preserved end-to-end.

| Seam | cppmega source anchor | Current MLX coverage | Missing contract | Candidate MLX target |
| --- | --- | --- | --- | --- |
| Structure runtime handoff | `cppmega/megatron/structure_batch.py`, `cppmega/megatron/custom_gpt_model.py`, `cppmega/megatron/custom_embedding.py` | `LMTokenBatch` carries `structure_ids`, `dep_levels`, `ast_depth_ids`, `sibling_index_ids`, and `node_type_ids`; NPZ, Parquet, and Megatron-indexed readers can pass those keys through `model_kwargs()`. NPZ ingress fails closed for ambiguous side-channel payloads, ngram sidecars, and non-integer or out-of-int32 token/structure IDs instead of silently truncating or dropping them. `HybridTinyLM` consumes those keys through `CppMegaStructureEmbedding`. | End-to-end training/script handoff is still tiny-local, and `TinyLM` still uses simplified structure addition rather than full source embedding semantics. | `cppmega_mlx/data/batch.py`, `cppmega_mlx/data/token_dataset.py`, `cppmega_mlx/data/parquet_dataset.py`, `cppmega_mlx/data/megatron_indexed.py`, `cppmega_mlx/models/tiny_lm.py`, `cppmega_mlx/models/hybrid_lm.py` |
| Megatron indexed side-channel preservation | `scripts/data_prep_parquet_to_megatron.py`, `scripts/remote_smoke_h200_structure_ingress.sh`, `scripts/remote_smoke_h200_structure_poly.sh` | The indexed reader accepts `MMIDIDX` token shards, raw `.bin` handoffs, and canonical token-aligned binary sidecars for `attention_mask` plus all five structure arrays via `side_channel_paths` or direct top-level path entries. Integer sidecars are range-checked before int32 MLX materialization, and ngram sidecars fail closed because hashes are derived from `input_ids` at the model seam. | Source converter still writes only token `.bin/.idx`; no multi-shard/packed sidecar schema exists yet; ngram sidecars remain a non-goal because ngram hash derives from `input_ids`. | `cppmega_mlx/data/megatron_indexed.py`, future source-side converter docs/scripts |
| Full structure embedding semantics in model path | `cppmega/features/structure/embedding.py`, `cppmega/megatron/custom_embedding.py` | `cppmega_mlx/nn/structure_embedding.py` mirrors component parsing, clamping, bottleneck projection, zero init, component scales, and missing-component masking, and `HybridTinyLM` uses it on the forward path. | `TinyLM` still uses a simplified shared `nn.Embedding` addition; full Megatron pipeline-stage and launcher behavior is not represented. | `cppmega_mlx/models/tiny_lm.py`, `cppmega_mlx/models/hybrid_lm.py`, `cppmega_mlx/nn/structure_embedding.py` |
| Ngram hash model/config integration | `cppmega/features/engram/ngram_hash.py`, `cppmega/megatron/custom_embedding.py`, H200 launchers with `CPPMEGA_NGRAM_HASH_*` | `cppmega_mlx/nn/ngram_hash.py` implements the local additive hash module and tests its default orders, heads, table size, and projection behavior. `HybridTinyLM` can add it to token embeddings, and `build_hybrid_tiny_config_from_nam56r()` maps the central NAM56R config defaults into that model path. | Source launcher/env ingestion and `CPPMEGA_NGRAM_HASH_OFFLOAD` runtime behavior are documented/configured but not implemented as local MLX runtime behavior; `TinyLM` does not consume ngram enrichment. | `cppmega_mlx/config/model.py`, `cppmega_mlx/models/tiny_lm.py`, `cppmega_mlx/models/hybrid_lm.py`, `cppmega_mlx/nn/ngram_hash.py` |
| Mamba3 pure-MLX reference | `cppmega/megatron/mamba3_te_in_proj.py`, `cppmega/megatron/mamba3_mixer.py`, `cppmega/megatron/mamba3_te_mixer.py` | `Mamba3ReferenceBlock` preserves projection split sizes, trainable local recurrence, B/C QK norm and bias, data-dependent `dt/A` terms, and the source split-path packed `xBC` causal conv before B/C transform. | Projected `angles` are not consumed, and the Author RoPE, trapezoidal, SISO/MIMO scan kernel semantics are not yet implemented as pure MLX or as trainable Metal kernels with VJP coverage. | `cppmega_mlx/nn/mamba3.py`, future MLX custom-function Metal kernels only after backward parity tests |
| Env and launcher contract mapping | `cppmega/megatron/custom_embedding.py`, `scripts/remote_production_h200_nam56r_v1.sh`, `scripts/remote_sweep_h200_dsa_production.sh` | NAM layout and central dataclasses record `CPPMEGA_STRUCTURE_*` and `CPPMEGA_NGRAM_HASH_*` defaults, enable flags, and the ngram offload non-local runtime caveat. | Local scripts still do not ingest the full source launcher env contract, and distributed Megatron behavior remains outside the MLX scaffold. | `cppmega_mlx/config/model.py`, local scripts, docs |
| Parquet-to-Megatron schema | `scripts/data_prep_parquet_to_megatron.py` | `TokenParquetDataset` can read side-channel columns when they already exist in Parquet. | The source converter writes only `token_ids` to `.bin/.idx`; structure side channels are not guaranteed to survive Parquet-to-Megatron conversion. | `cppmega_mlx/data/parquet_dataset.py`, `cppmega_mlx/data/megatron_indexed.py`, future converter/schema docs |
| Megatron PP/MTP checkpoint layout | `cppmega/megatron/custom_embedding.py` `sharded_state_dict()` | MLX checkpoints are simple safetensors directories with metadata and no distributed Megatron dependency. | Pipeline-stage and MTP replica-id semantics for custom ngram/structure submodules are absent; keep this a documented non-goal unless import/export interop becomes required. | `cppmega_mlx/training/checkpoint.py`, `docs/checkpointing.md`, this parity anchor |

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
