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
- Local GB10 Parquet files under data/parquet_samples/ are ignored smoke
  inputs for data-contract tests. They are not source-of-truth anchors and must
  not be treated as committed fixtures or benchmark rows.

## NAM56R Constants

These constants are model-layout anchors, not CUDA runtime anchors.

| Anchor                                     | Value                                                      |
| ------------------------------------------ | ---------------------------------------------------------- |
| Source pattern                             | AEMEAEMEAEMR                                               |
| Depth                                      | 52                                                         |
| Expanded counts                            | A=13, E=22, M=13, R=4                                      |
| R layer numbers                            | 12,24,36,48                                                |
| A layer numbers                            | 1,5,9,13,17,21,25,29,33,37,41,45,49                        |
| DSA rank tuple preserved from launchers    | 1,2,3,5,6,7,9,10,11                                        |
| cppmega launcher-indexed DSA layer numbers | 5,9,13,21,25,29,37,41,45                                   |
| cppmega launcher-indexed MLA layer numbers | 1,17,33,49                                                 |
| MoE defaults                               | 16 experts, top_k=4, routed hidden 896, shared hidden 1024 |
| Vocab contracts                            | local/profile 65536, megacpp tokenizer 131072              |

The DSA tuple deserves special handling. The MLX helper treats
dsa_a_layer_ranks as zero-based A-layer indices, matching the production
cppmega H200 launchers, which compute absolute layers as attn_nums[r], and
CppMegaSelectiveAttentionLayer, which derives a_rank with
attention_layer_numbers.index(layer_idx). That indexed contract yields the
9 DSA + 4 MLA split above.

## Source Reference Anchors

These files in ../cppmega are the current source references for parity:

| Source                                          | Anchor contract                                                                                                                                   | Not being ported as-is                                                                                                                 |
| ------------------------------------------------| --------------------------------------------------------------------------------------------------------------------------------------------------| ---------------------------------------------------------------------------------------------------------------------------------------|
| cppmega/megatron/nam56r_layout.py               | Import-safe NAM56R pattern/depth helpers, R layer loading, A-layer number loading, and DSA rank env parsing.                                      | Megatron symbol probing is only source-side compatibility logic.                                                                       |
| cppmega/recipes/nam56r_megatron.py              | Full source parser accepts A, M, D, E, G, R, and pipe-delimited patterns, and reports whether a recipe is fully native Megatron.                  | Local MLX pattern parsing intentionally accepts only A/E/M/R and fails closed on upstream-only symbols.                                |
| cppmega/recipes/megatron_args.py                | Emits Megatron CLI arguments for MLA, MTP, MoE, DSA, and bf16-only DSA indexer settings.                                                          | Megatron launcher flags, CUDA graph, distributed optimizer, and DSA indexer runtime.                                                   |
| cppmega/recipes/nam56r_launch.py                | NAM symbol to Megatron hybrid pattern mapping and R custom-layer index derivation.                                                                | Megatron hybrid-layer CLI construction is not the MLX runtime.                                                                         |
| cppmega/recipes/nam56r_nemo_recipe.py           | MoE defaults and local/profile vocab constant.                                                                                                    | NeMo/Megatron CLI recipes and distributed launch policy.                                                                               |
| cppmega/megatron/nam56r_full_spec.py            | Full NAM56R Megatron recipe/runtime anchors: mixed MLA/DSA A-layers, MTP dense-attention override, Author Mamba3 or TP Mamba3 mixer selection,    |                                                                                                                                        |
|                                                 | M2RNN-on-R placement, and TE provider requirement.                                                                                                | Transformer Engine provider, native MLA/DSA/MTP module specs, TP mixer runtime, CUDA execution, and distributed Megatron graph behavior|
| cppmega/megatron/nam56r_te_spec.py              | TE-preserving NAM56R spec that only swaps the Mamba mixer to Author Mamba3 or M2RNN by source layer placement.                                    | Transformer Engine attention/MoE/MTP submodules and Megatron runtime execution.                                                        |
| cppmega/megatron/nam56r_noconv_spec.py          | No-conv Mamba3 B/C feature branch with R layers still routed to CppMegaM2RNNMixer.                                                                | TE fused submodules, vanilla SSD/Triton scan path, and full Author Mamba3 feature parity.                                              |
| cppmega/megatron/mamba3_te_stack_spec.py        | Upstream TE stack spec replacing only the Mamba mixer with the cppmega Mamba3 mixer.                                                              | TE GatedDeltaNet, MLP, MTP, and Megatron ModuleSpec execution.                                                                         |
| scripts/remote_production_h200_nam56r_v1.sh     | Production defaults for pattern, depth, R layers, ngram hash, structure, DSA tuple, and the indexed DSA/MLA preflight.                            | H200 launcher, CUDA graph, NCCL, TE, and distributed shell runtime.                                                                    |
| scripts/remote_sweep_h200_dsa_production.sh     | Same DSA 9+4 indexed preflight used in sweep lanes.                                                                                               | Remote H200 sweep orchestration and CUDA-only runtime checks.                                                                          |
| cppmega/megatron/m2rnn_spec.py                  | M2RNN feature semantics: projection split, recurrence, decay gate, residual path, and training-only limitations.                                  | Megatron ModuleSpec, Transformer Engine norm, Triton scan kernel, and CUDA training execution.                                         |
| cppmega/megatron/mamba3_te_in_proj.py           | Author Mamba3 in-projection slices [z,x,B,C,dd_dt,dd_A,trap,angles] and TP partition-size semantics.                                              | TELayerNormColumnParallelLinear and Transformer Engine checkpoint resharding runtime.                                                  |
| cppmega/megatron/mamba3_mixer.py                | Mamba3 feature semantics: QK norm on B/C, B/C bias, optional data-dependent A, and split path packed xBC causal conv before [x,B,C] split.        | Megatron MambaMixer, mamba_ssm Triton kernels, CUDA graph compatibility mechanics.                                                     |
| cppmega/megatron/mamba3_te_mixer.py             | Author Mamba3 kernel path: [z,x,B,C,dd_dt,dd_A,trap,angles] projection feeds data-dependent-A, trapezoidal, RoPE/angle, and SISO/MIMO scan kernels| Transformer Engine, Triton/TileLang scan kernels, CUDA inference step kernels, and distributed TP runtime.                             |
| cppmega/features/engram/ngram_hash.py           | Additive n-gram hash enrichment defaults: orders (2,3), heads 8, table size 500000, embed dim 16.                                                 | Torch module implementation and CUDA execution.                                                                                        |
| cppmega/features/structure/embedding.py         | Additive structure embedding components, especially core = structure, dep_level.                                                                  | Torch module implementation and CUDA execution.                                                                                        |
| cppmega/megatron/custom_embedding.py            | Optional env-gated ngram and structure enrichments added beside Megatron token embeddings.                                                        | Megatron LanguageModelEmbedding, pipeline-stage sharded-state mechanics, and CUDA dropout/sequence-parallel runtime.                   |
| cppmega/megatron/structure_batch.py             | Source-side structure side-channel key set: structure_ids, dep_levels, ast_depth_ids, sibling_index_ids, and node_type_ids.                       | Torch batch plumbing and Megatron data-loader integration.                                                                             |
| cppmega/megatron/custom_gpt_model.py            | Source-side handoff for setting cppmega structure inputs on the GPT model before the embedding layer consumes them.                               | Megatron GPT model execution, pipeline parallelism, and CUDA runtime.                                                                  |
| cppmega/megatron/fastmtp_layer.py               | Optional CPPMEGA_FASTMTP=1 Torch/Megatron FastMTP path with checkpointing, cadence, and optional Liger CE.                                        | Native MLX MTP layer, Megatron MTP monkey patching, Liger CE, and production MTP scheduling.                                           |
| cppmega/megatron/mtp_native_hopper_ce.py        | Hopper/Megatron native linear-CE path for main-plus-MTP loss fusion.                                                                              | Hopper/GB10 CE kernels, native Megatron CE patching, and fused main+MTP launch behavior.                                               |
| cppmega/megatron/dsa_local_spec.py              | Source-side helper for validating official Megatron DSA before copying residual behavior.                                                         | Local native DSA implementation or sparse-MLA training kernel.                                                                         |
| cppmega/megatron/dsa_sparse_attention.py        | CUDA/Torch sparse gather-scatter DSA attention replacement for Megatron's unfused DSA function.                                                   | Sparse DSA/MLA Metal kernel and differentiable MLX training implementation.                                                            |
| cppmega/megatron/moe_dispatcher_patch.py        | Runtime monkey patch around Megatron/Transformer Engine MoE dispatcher behavior.                                                                  | Megatron all-to-all dispatcher parity, grouped-GEMM scheduling, and expert-parallel overlap.                                           |
| cppmega/megatron/selective_fp8_moe_patch.py     | Selective FP8 gating for MoE layers while attention/Mamba/R layers remain bf16.                                                                   | FP8/NVFP4 training, Transformer Engine FP8 contexts, and local MLX mixed-precision dispatcher parity.                                  |
| scripts/data_prep_parquet_to_megatron.py        | Parquet-to-Megatron conversion currently reads a configurable token column whose default is token_ids.                                            | Structure side-channel conversion; the converter does not emit sidecar metadata for the MLX reader.                                    |
| scripts/remote_smoke_h200_dsa_9_4_m.sh          | H200 DSA/MLA 9+4 smoke anchor with ngram/structure env flags and source-side runtime patches.                                                     | Local MLX launcher support, CUDA runtime, and distributed Megatron execution.                                                          |
| scripts/remote_smoke_h200_nam56r_k_pp1.sh       | H200 NAM56R smoke anchor for launcher env flags, native MLA/MoE/MTP/DSA args, and 9+4 DSA/MLA preflight.                                          | Local MLX launcher support, CUDA runtime, and distributed Megatron execution.                                                          |
| scripts/remote_train_gb10_nam56r_single.sh      | GB10 NAM56R source-side train script anchor.                                                                                                      | GB10 performance parity or local MLX launcher support.                                                                                 |
| scripts/remote_train_h200_nam56r_full.sh        | H200 full NAM56R train script anchor.                                                                                                             | H200 distributed Megatron launcher parity.                                                                                             |
| scripts/remote_train_h200_nam56r_lite.sh        | H200 lite NAM56R train script anchor.                                                                                                             | Local tiny smoke equivalence to H200 lite training.                                                                                    |
| scripts/remote_train_h200_nam56r_grid.sh        | H200 NAM56R grid train script anchor.                                                                                                             | Local distributed sweep orchestration.                                                                                                 |
| scripts/remote_train_h200_nam56r_tp2.sh         | H200 NAM56R TP=2 train script anchor.                                                                                                             | MLX tensor-parallel parity.                                                                                                            |
| scripts/remote_train_h200_nam56r_noconv.sh      | H200 no-conv NAM56R train script anchor.                                                                                                          | MLX no-conv branch performance or feature parity.                                                                                      |
| scripts/remote_train_h200_nam56r_europe_sweep.sh| H200 Europe sweep train script anchor.                                                                                                            | Remote fleet orchestration or benchmark parity.                                                                                        |

## Full NAM56R Megatron Recipe/Runtime Anchors

The source NAM56R recipe is broader than the local MLX subset. In
../cppmega, full recipe/runtime coverage spans the parser and argument
emitters, the full/TE/no-conv Megatron specs, Mamba3 and M2RNN mixer placement,
native MLA/MTP/DSA module specs, FastMTP, Hopper/GB10 CE patching, DSA sparse
attention, MoE dispatcher patches, selective FP8 MoE, and H200/GB10 train
scripts. Those anchors are evidence for what not to overclaim locally.

Current MLX coverage for NAM56R remains the fail-closed A/E/M/R layout,
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
H200 scripts are source/runtime anchors for ../cppmega, not MLX-supported
launchers.

### M0.1 Tokenizer Contract

- Acceptance anchor: `cppmega-mlx-t8f.1` / M0.1 requires a HuggingFace JSON BPE
  tokenizer with vocab=65536 and special IDs `2=<BOS>`, `3=<EOS>`/EOT,
  `4=<FIM_PREFIX>`, `5=<FIM_MIDDLE>`, `6=<FIM_SUFFIX>`, and
  deployed id `7=<CODE_START>`. The iFIM extension uses literal
  `<FIM_INSTRUCTION>` at id 45; do not alias `FIM_INSTRUCTION_ID` to id 7.
- Historical local receipts: `nanochat/tokenizer.json` is vocab=32768, while
  `nanochat/tokenizer_v3.json` and
  `nanochat/config/tokenizer_v3_fixed_tokens.json` are 65K-family artifacts
  that map id 7 to `<CODE_START>`.
- Artifact receipt: read-only inspection of
  `/home/dave/cppmega-root/cpp_tokenizer_hf/tokenizer.json` and
  `/home/dave/cppmega-root/data/tokenizer/tokenizer.json` found vocab=65536,
  id7=`<CODE_START>`, id45=`<FIM_INSTRUCTION>`, id46=`<SPACE>`,
  id47=`<NL>`, and SHA-256
  `d3c4711161a452ee36d64222b6977845ddd58b1e723a7de54158c64c50d2a888`,
  matching the vendored local artifact contract.
- Closure receipt: M0.1 is closed after vendoring the deployed GB10 tokenizer
  plus the explicit whitespace-sentinel wrapper. The deployed JSON remains
  `decoder=null`; encode collapses newline runs to `<NL>` and space/tab runs to
  `<SPACE>`, while decode concatenates token strings and substitutes those
  sentinels. The acceptance target is byte-exact parity with the CUDA reference
  wrapper, not HF reversible `decode(encode(text))` for arbitrary whitespace.
  Recorded receipts are 11/11 curated samples and 1000/1000 source-line samples
  byte-exact against GB10.

### M0.3 Random-Init Forward Parity Manifest

- Acceptance anchor: `cppmega-mlx-t8f.3` / M0.3 requires seed-matched
  `local_gb10_quarter` MLX and CUDA reference forwards, a fixed input batch,
  and logits within `rtol=1e-2, atol=1e-1`.
- Fixed-input scaffold metadata: `B=1`, `T=512`, `seed=3003`,
  `vocab=65536`, and closure-required token SHA-256
  `c645ca4053e5206dcbe58c13aa26f4a9e56c5aa2aee90a4d4778bbc9d9c33549`.
- Required CUDA logits artifact: the real GB10/CUDA golden must be supplied at
  `bench/parity/cuda/m03_local_gb10_quarter_seed3003_logits.json`. Current
  local state has no artifact at that path; the manifest must therefore remain
  `status=refused` with `artifact_preflight_status=missing` when the default
  path is probed.
- Artifact JSON contract: `format=cppmega_cuda_m03_forward_logits_v1`,
  `profile=local_gb10_quarter`, `tensor_name=local_gb10_quarter.logits`,
  `seed=3003`, `batch_size=1`, `seq_len=512`, `vocab_size=65536`,
  `shape=[1,512,65536]`, `dtype=bf16`, `logits_dtype` in `bf16`/`bfloat16`/
  `torch.bfloat16`, matching `input_tokens_sha256`, a `logits_sha256`,
  source/hardware/CUDA runtime metadata, and a finite numeric `logits_summary`
  with `numel=33554432`, `min`, `max`, `mean`, `std`, `l2_norm`, and
  `max_abs`.
- Artifact preflight statuses are intentionally not pass/fail parity:
  `not_supplied` means no artifact path was supplied, `missing` means the
  required/provided path was absent, `invalid` means metadata/checksum/summary
  contract validation failed, and `valid_not_evaluated` means metadata was
  valid but logits were still not numerically compared. `valid_not_evaluated`
  is not M0.3 acceptance and must still leave `m0_3_closed=false`.
- Current MLX coverage: `cppmega_mlx/training/parity.py` provides a
  fail-closed manifest builder for `bench/parity/m03_random_init.json`, and
  `scripts/m03_forward_parity_manifest.py` records either a blocked scaffold or
  a supplied CUDA artifact path as refused/not-evaluated. The script does not
  evaluate CUDA logits, full-profile MLX logits, receipt tolerances, or
  pass/fail parity.
- Local readiness scope: the script records full `local_gb10_quarter` profile
  metadata, but it deliberately sets `full_profile_allocation_executed=false`
  and `full_profile_forward_executed=false`. Its optional MLX execution is only
  a tiny smoke forward, recorded with `local_mlx_forward_scope=tiny_smoke_only`.
- Missing contract: the scaffold does not run CUDA, import CUDA weights, or
  warm-start the MLX model. It never emits a pass/closure manifest; future M0.3
  closure still requires an external CUDA logits artifact and a separate
  numerical harness for the same config, seed, fixed input batch, and tolerance
  policy.
- Non-claims: the manifest is forward-only architecture evidence. It is not a
  GB10 performance claim, M4-vs-GB10 parity claim, distributed Megatron parity
  claim, CUDA weight-conversion receipt, or M0.7 resumable-training receipt.

### Structure Runtime Handoff

- Source anchors: cppmega/megatron/structure_batch.py,
  cppmega/megatron/custom_gpt_model.py, and
  cppmega/megatron/custom_embedding.py.
- Current MLX coverage: LMTokenBatch carries structure_ids, dep_levels,
  ast_depth_ids, sibling_index_ids, and node_type_ids; NPZ, Parquet, and
  Megatron-indexed readers can pass those keys through model_kwargs().
  HybridTinyLM consumes those keys through CppMegaStructureEmbedding.
- Missing contract: end-to-end training/script handoff is still tiny-local, and
  TinyLM still uses simplified structure addition rather than full source
  embedding semantics.

### MTP / FastMTP Local MLX Coverage

- Source anchors: `cppmega/megatron/fastmtp_layer.py` and
  `cppmega/megatron/mtp_native_hopper_ce.py`.
- Current MLX coverage: `cppmega_mlx/training/mtp.py` implements the
  training-side K=2 default, static-shape roll-and-mask labels/teacher IDs,
  one shared recurrent MTP block, direct module aliasing for the model
  embedding/lm head surfaces, beta-decayed normalized per-depth loss weights
  with beta=0.6, and lambda=0.3 composition through
  `cppmega_mlx/training/loss.py`.
- Test anchor: `tests/test_mtp_loss.py` covers K=2 defaults, shared-head
  aliasing, static roll-and-mask masking, per-depth weighting, lambda
  integration, MTP-disabled inference sanity, decoder-hidden-state routing,
  and CUDA-parity detach semantics for main hidden states and output head
  weight.
- Non-claims: local MLX MTP does not implement Liger CE, Hopper native linear
  CE, fused main-plus-MTP CE launch behavior, Megatron monkey patching,
  production cadence scheduling, sequence-parallel gather, or CUDA/Hopper/GB10
  kernel parity. Treat Hopper fused CE as source-only evidence until a separate
  hardware-gated receipt exists.
- Fail-closed M0.5 receipt: no checked-in or locally observed fixed-seed
  CUDA/GB10 artifact currently records the required FastMTP loss values plus
  grad norm for `cppmega/megatron/fastmtp_layer.py` and
  `cppmega/megatron/mtp_native_hopper_ce.py`. M0.5 remains open until that
  artifact exists and a numerical MLX-vs-CUDA comparison is evaluated; the
  local tests above are contract coverage, not closure evidence.
- Hardening receipt: the M0.5 manifest builder/validator now rejects nested
  CUDA/MLX overclaim fields such as numerical-harness, parity-passed, and full
  acceptance flags. This is scaffold hardening only; it does not replace the
  required CUDA artifact or numerical harness.

### Engram Standalone Local Slice

- Source anchors: `nanochat/nanochat/engram.py` and the source-side ngram
  enrichment family under `cppmega/features/engram/`.
- Current MLX coverage: `cppmega_mlx/nn/engram.py` provides a standalone
  `EngramBranch` with parsed n-gram orders, causal local averaging, optional
  document-boundary masking, optional sigmoid gating, optional grouped causal
  SiLU convolution, and zero-initialized output/value projection behavior.
- Test anchor: `tests/test_engram.py` covers standalone local behavior, not full
  model integration.
- Missing contract: no `engram_layers` route wiring into NAM56R/HybridTinyLM is
  claimed here, no Engram+mHC branch combo is wired, and no nanochat Torch or
  cppmega CUDA parity row exists. This does not close Stream H.

### mHC Standalone Local Slice

- Source anchors: `nanochat/nanochat/mhc.py` and the MaxText-style
  Sinkhorn branch-mixing contract referenced by Stream H.
- Current MLX coverage: `cppmega_mlx/nn/mhc.py` provides a standalone
  `ManifoldBranchMixer` with fp32 Sinkhorn-style normalization, branch-shape and
  dtype validation, branch routing weights, uniform fallback for invalid weights,
  and `blend_alpha` interpolation.
- Test anchor: `tests/test_mhc.py` covers standalone local routing/mixing
  invariants, not full model integration.
- Missing contract: no Engram/main/skip residual combo is wired into the model,
  no source MaxText/nanochat parity row exists, and no compile/performance
  receipt exists for the integrated branch path. This does not close Stream H.

### FIM / iFIM CPU Transform Slice

- Source anchors: `nanochat/nanochat/fim.py`, `nanochat/nanochat/ifim.py`, and
  the M0.1 tokenizer special-token acceptance contract above.
- Current MLX coverage: `cppmega_mlx/data/fim.py` provides dependency-free
  token-level PSM/SPM FIM permutations, sampled FIM transforms, instruction-aware
  FIM formatting, lightweight comment/signature instruction extraction, and a
  fail-closed `FIMSpecialTokenIds` wrapper over the required special-token IDs.
- Test anchor: `tests/test_fim_transform.py` covers local transform behavior and
  fail-closed special-token validation.
- Missing contract: no tree-sitter/AST-aware iFIM or AST-FIM extraction is
  wired, and no dataset/training/inference integration is claimed. M0.1 is
  already closed by the vendored GB10 tokenizer plus explicit `<SPACE>`/`<NL>`
  sentinel decode parity receipt; the remaining gap is Stream H feature
  integration.

### Megatron Indexed Side-Channel Preservation

- Source anchor: scripts/data_prep_parquet_to_megatron.py.
- Current MLX coverage: the indexed reader accepts MMIDIDX token shards, raw
  .bin handoffs, and canonical token-aligned binary sidecars for
  attention_mask plus all five structure arrays via side_channel_paths or
  direct top-level path entries. Integer sidecars are range-checked before int32
  MLX materialization, and ngram sidecars fail closed because hashes are derived
  from input_ids at the model seam.
- Missing contract: The source converter writes only token_ids to .bin/.idx;
  structure side channels are not guaranteed to survive Parquet-to-Megatron
  conversion, and no source multi-shard/packed sidecar schema is claimed.

### Full Structure Embedding Semantics

- Source anchors: cppmega/features/structure/embedding.py and
  cppmega/megatron/custom_embedding.py.
- Current MLX coverage: cppmega_mlx/nn/structure_embedding.py mirrors
  component parsing, clamping, bottleneck projection, zero init, component
  scales, and missing-component masking, and HybridTinyLM uses it on the
  forward path.
- Missing contract: TinyLM still uses a simplified shared nn.Embedding
  addition; full Megatron pipeline-stage and launcher behavior is not
  represented.

### Ngram Hash Model And Config Integration

- Source anchors: cppmega/features/engram/ngram_hash.py,
  cppmega/megatron/custom_embedding.py, and H200 launchers with
  CPPMEGA_NGRAM_HASH_*.
- Current MLX coverage: cppmega_mlx/nn/ngram_hash.py implements the local
  additive hash module and tests its default orders, heads, table size, and
  projection behavior. HybridTinyLM can add it to token embeddings, and
  build_hybrid_tiny_config_from_nam56r() maps the central NAM56R config
  defaults into that model path.
- Missing contract: source launcher/env ingestion and
  CPPMEGA_NGRAM_HASH_OFFLOAD runtime behavior are documented/configured but
  not implemented as local MLX runtime behavior; TinyLM does not consume ngram
  enrichment.

### Mamba3 Pure-MLX Reference

- Source anchors: cppmega/megatron/mamba3_te_in_proj.py,
  cppmega/megatron/mamba3_mixer.py, and
  cppmega/megatron/mamba3_te_mixer.py.
- Current MLX coverage: Mamba3ReferenceBlock preserves the Author packed
  projection split [z,x,B,C,dd_dt,dd_A,trap,angles], source-shaped
  (angle_dt, ssm, k, v) cache, trainable local recurrence, B/C QK norm and
  bias, data-dependent dt/A terms, local trapezoidal input scaling from trap,
  cumulative projected-angle Author RoPE over B/C, and the source split-path
  packed xBC causal conv before the [x,B,C] split.
- Missing contract: exact Author TE/Triton/TileLang/CUDA SISO/MIMO kernels,
  tensor-parallel/distributed runtime, and source-kernel numerical parity are
  not implemented as local MLX behavior. Trainable Metal kernels still require
  explicit VJP/JVP coverage before they can replace the pure-MLX reference.

### Env, Launcher, And H200 Runtime Anchors

- Source anchors: cppmega/megatron/custom_embedding.py,
  scripts/remote_production_h200_nam56r_v1.sh,
  scripts/remote_sweep_h200_dsa_production.sh,
  scripts/remote_smoke_h200_dsa_9_4_m.sh, and
  scripts/remote_smoke_h200_nam56r_k_pp1.sh.
- Current MLX coverage: NAM layout and central dataclasses record
  CPPMEGA_STRUCTURE_* and CPPMEGA_NGRAM_HASH_* defaults, enable flags, and
  the ngram offload non-local runtime caveat.
- Missing contract: local scripts still do not ingest the full source launcher
  env contract, H200 launcher shell behavior is not an MLX feature, and
  distributed Megatron behavior remains outside the MLX scaffold. Distributed
  MLX training is not implemented, and distributed Megatron parity is not claimed.

### Parquet-To-Megatron Schema

- Source anchor: scripts/data_prep_parquet_to_megatron.py.
- Current MLX coverage: TokenParquetDataset can read side-channel columns
  when they already exist in Parquet, and MegatronIndexedDataset can read
  explicit token-aligned sidecars beside .bin/.idx shards.
- Missing contract: The source converter writes only token_ids; structure
  side channels are not preserved by that converter today.

### Megatron PP/MTP Checkpoint Layout

- Source anchor: cppmega/megatron/custom_embedding.py sharded_state_dict().
- Current MLX coverage: MLX checkpoints are simple safetensors directories with
  metadata and no distributed Megatron dependency.
- Missing contract: pipeline-stage and MTP replica-id semantics for custom
  ngram/structure submodules are absent; keep this a documented non-goal unless
  import/export interop becomes required.

### Performance Claim Boundary

M4 Max vs GB10 parity is not proven by this document, by local M4 rows, or by
ignored local GB10 Parquet smoke inputs. Use the matched-row protocol in
docs/porting_plan.md and docs/perf_baseline.md before making any
cross-machine claim.

## MoE Trainability Audit Note

The local MLX MoE path is a correctness-first reference, not a Megatron
dispatcher clone. Targeted tests now prove gradients, parameter updates, and
AdamW state materialization for the router gate, all routed expert projections,
and the shared expert projections both in a standalone ReferenceMoE train step
and through the hybrid LM E route.

The remaining MoE parity risks stay source-runtime-only for this lane:
Megatron all-to-all/token dispatcher behavior, grouped GEMM scheduling,
capacity/drop-pad policy, expert-parallel overlap, identity chunk-sort
short-circuiting from moe_dispatcher_patch.py, and selective FP8 gating from
selective_fp8_moe_patch.py are not emulated by the MLX reference module.

## Porting Rule

Preserve model-facing semantics first: layer layout, route selection, config
constants, tensor shapes, enrichment inputs, and checkpoint-visible metadata.
Replace CUDA/Megatron runtime mechanisms with MLX-native modules, MLX compile
patterns, and optional Metal kernels only after pure-MLX parity tests exist.

Forward-only Metal kernels remain optional diagnostics or inference-style
experiments unless they define MLX custom-function VJP/JVP coverage for every
trainable path. CUDA-only optimizations should be documented as source
references, not silently emulated with weaker local behavior.
