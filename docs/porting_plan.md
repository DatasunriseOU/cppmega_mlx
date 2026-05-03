# cppmega.mlx Integration Roadmap

Date: 2026-04-30

This plan tracks the MLX-native Apple Silicon port. It is not a CUDA/Megatron
runtime clone and it is not a performance claim. The current repo proves local
correctness and smoke-training plumbing for small MLX models; it does not prove
full NAM56R readiness or M4 Max parity with GB10.

## Current Contract

- Preserve cppmega model semantics where they are already specified: NAM A/M/E/R
  routing intent, structure side channels, ngram enrichment, MoE routing,
  Mamba3/M2RNN reference behavior, checkpoint/resume metadata, and fixed-shape
  token batches.
- Use MLX and MLX-LM patterns as the training substrate: mlx.nn.Module,
  nn.value_and_grad, mx.compile, mx.eval, safetensors checkpoints, and
  MLX memory telemetry.
- Prefer pure MLX reference implementations first. Custom Metal kernels remain
  optional and must keep pure-MLX fallbacks.
- Treat Hugging Face Apple M4 kernels as external references or test fixtures
  until parity, dtype behavior, backward behavior, and hotspot evidence are
  proven locally.

## Implemented

### Local Model And Feature Surface

- cppmega_mlx/config/model.py has pure-Python NAM56R-oriented config
  dataclasses, vocab constants, MoE/Mamba/M2RNN/MLA/DSA config validation, and
  the default NAM56R pattern/depth metadata.
- cppmega_mlx/recipes/pattern.py and cppmega_mlx/recipes/nam56r.py expand
  NAM A/M/E/R route patterns and DSA/MLA A-layer markers without importing
  Megatron or CUDA. DSA routing preserves the source cppmega zero-based A-layer
  index contract, so the default tuple maps to DSA layers
  5,9,13,21,25,29,37,41,45 and MLA layers 1,17,33,49.
- cppmega_mlx/nn/attention.py, cppmega_mlx/nn/mamba3.py,
  cppmega_mlx/nn/m2rnn.py, and cppmega_mlx/nn/moe.py provide small MLX
  reference blocks for attention, Mamba3, M2RNN, and MoE smoke coverage.
  Mamba3 also exposes a source-shaped local cache-state carrier for continuation
  tests; this is not a Megatron inference-cache integration.
- cppmega_mlx/nn/ngram_hash.py and
  cppmega_mlx/nn/structure_embedding.py provide local enrichment modules.
  HybridTinyLM uses the source-equivalent structure module and optional
  model-derived ngram hash enrichment from config; TinyLM still keeps only the
  earlier simplified structure side-channel path.
- cppmega_mlx/models/tiny_lm.py provides a tiny decoder-only LM with optional
  structure side channels.
- cppmega_mlx/models/hybrid_lm.py wires tiny A/M/E/R route blocks into one
  smoke model. It keeps NAM56R route intent visible, but it is explicitly not a
  full NAM56R implementation and does not claim production kernel behavior.

### Data, Training, And Checkpointing

- cppmega_mlx/data/batch.py defines LMTokenBatch, model kwargs, masks, and
  synthetic fixed-shape token batches.
- cppmega_mlx/data/token_dataset.py implements an NPZ-backed fixed-shape token
  dataset with optional side-channel arrays and deterministic resume cursors.
- cppmega_mlx/data/parquet_dataset.py adds an optional Parquet handoff seam
  for token, token-list, or text columns. It imports pyarrow/pandas only
  when used and keeps those packages out of the base dependency set.
- Local GB10 Parquet samples live only under ignored data/parquet_samples/.
  They are smoke-test inputs for schema/side-channel coverage, not checked-in
  fixtures and not performance evidence.
- cppmega_mlx/data/megatron_indexed.py adds a standalone fail-closed
  Megatron .bin/.idx reader seam for safe MMIDIDX token shards and explicit
  raw .bin handoffs. It does not import Megatron, Torch, or CUDA code and is
  wired into open_token_dataset(..., format="megatron") plus .bin/.idx
  suffix inference.
- cppmega_mlx/training/loss.py implements next-token cross entropy over the
  local batch contract.
- cppmega_mlx/training/loop.py remains a small eager one-step helper.
- cppmega_mlx/training/compiled.py implements the current MLX-LM-style
  CompiledPretrainingStep: fixed-key batch normalization, eager/compiled
  paths, gradient accumulation, and compile captures for
  [model.state, optimizer.state, mx.random.state].
- cppmega_mlx/training/checkpoint.py saves and loads MLX safetensors
  directory checkpoints with model weights, optional optimizer state, metadata,
  package versions, tokenizer/vocab contract fields, and compiled-step resume
  metadata including pending gradient accumulation.
- cppmega_mlx/training/eval.py evaluates average next-token loss over local
  batch iterables.
- cppmega_mlx/training/profile.py provides synchronized timing and MLX
  active/peak/cache memory snapshots for measured train/eval scopes.
- Package-root exports are convenience surfaces for local MLX readers,
  reference blocks, checkpoint/eval/profile helpers, and guarded MLX-LM adapter
  probes, not full MLX-LM trainer, Megatron distributed runtime, CUDA/TE, or
  trainable Metal-kernel claims.

### Scripts And Docs

- scripts/bench_tiny.py runs a synthetic tiny training benchmark with JSON and
  stable --compare-line output, compile timing, steady-state tokens/sec, MLX
  memory telemetry, structure-side-channel option, and wired-limit options.
- scripts/train_tiny_npz.py runs a tiny NPZ-backed local train smoke through
  TokenNpzDataset, TinyLM, and CompiledPretrainingStep.
- scripts/train_hybrid_tiny.py runs a tiny hybrid A/M/E/R train smoke with
  checkpoint/resume and optional validation batches.
- scripts/bench_matrix.py runs comparable bench_tiny.py rows across small
  batch/sequence/dtype/profile/route matrices.
- docs/perf_mamba_m2rnn.md records the current M4 Max route smoke baseline:
  tiny 6662.73 tokens/sec, Mamba3 hybrid-m 2785.08 tokens/sec, and
  M2RNN hybrid-r 4120.29 tokens/sec for the tiny eager smoke shape. It also
  records eager/compiled Mamba3 and M2RNN checkpoint/resume script receipts.
- scripts/compare_bench_rows.py compares benchmark JSON rows and preserves the
  "matched shape first" rule for M4/GB10 reporting.
- docs/perf_baseline.md defines the detailed M4 Max vs GB10 measurement
  methodology and the benchmark output contract.
- docs/checkpointing.md documents the checkpoint/resume layout.
- docs/metal_kernel_policy.md documents the fallback, fail-closed, and
  differentiability gate for custom Metal kernels.
- docs/research/mlx_core_and_metal.md,
  docs/research/mlx_lm_training_patterns.md, and
  docs/research/apple_kernel_survey.md record the MLX/MLX-LM/HF kernel
  research basis.
- README.md, this roadmap, tests/test_package_exports.py, and
  tests/test_external_research_contract.py keep the public export, research,
  and documentation contracts aligned with the fail-closed runtime scope.

## External Framework Decisions

Current decision after the 2026-04-30 MLX/Metal research refresh:

- Checked references: MLX core, MLX-LM, the Hugging Face kernel catalog filtered
  to Apple M4/trending, mlx-examples, mlx-tune, mlx-forge, ForgeLLM,
  MLX-GRPO, and PMetal. These references are evidence for local choices below,
  not imported dependencies.
- Use MLX core as the P0 runtime, not as a compatibility shim over Megatron.
  The repo should keep building on mlx.core, mlx.nn, mlx.optimizers,
  nn.value_and_grad, mx.compile, mx.eval, MLX memory telemetry, and MLX
  custom-operation APIs where they are needed and locally proven.
- Use MLX-LM as a pattern source for Apple-Silicon LLM training mechanics:
  trainer shape, safetensors/checkpoint conventions, LoRA/full fine-tuning
  examples, quantized-model handling, wired-memory notes, and mx.distributed
  references. Do not make MLX-LM the cppmega trainer base until its assumptions
  match fixed-shape cppmega batches, NAM A/M/E/R routing, Mamba3/M2RNN blocks,
  side-channel arrays, and the local checkpoint metadata contract.
- The local mlx_lm_adapter boundary is intentionally fail-closed for full
  trainer use. It supports only dense int32 token plus (offset, length)
  loss-argument conversion for API probes; LMTokenBatch side channels, route
  metadata, masks, and full-pretraining checkpoint state remain repo-local.
- Use mlx-examples and community MLX training projects only as reading
  material for small mechanics such as transformer LM loops, LoRA scripts, and
  data adapters. They are not current dependencies and should not replace the
  repo-local training loop or tests.
- Treat Hugging Face Apple M4 kernels as source references and possible
  parity/performance test fixtures. The current filtered list includes MLX
  RMSNorm, activation/ReLU, paged attention, Metal flash SDPA, GPT-OSS Metal
  kernels, and bitsandbytes-MPS-style entries. Those names are reference
  labels, not adoption decisions. Do not borrow a kernel into the training path
  until the corresponding pure-MLX op is correct, forward/backward parity is
  covered across dtype/shape cases, and profiling shows a stable cppmega
  hotspot.
- Keep ../nanochat as a Torch reference only. It can help clarify source
  semantics, tests, and expected model behavior, but it is too slow for this
  local Mac objective and is not a Metal-native training substrate.
- Do not claim M4 Max is "not worse than GB10" from local rows alone. The only
  acceptable comparison is the matched-row protocol below with identical model
  source, shape, dtype, optimizer/update policy, data contract, warmup, measured
  steps, and metric definitions.

### Test Surface

The current collected test files are:

- tests/test_archive_bench_baseline_script.py
- tests/test_attention.py
- tests/test_bench_baselines.py
- tests/test_bench_matrix.py
- tests/test_bench_script.py
- tests/test_check_environment_script.py
- tests/test_checkpoint.py
- tests/test_checkpoint_subprocess_resume.py
- tests/test_compare_bench_rows.py
- tests/test_compiled_train.py
- tests/test_config.py
- tests/test_cppmega_parity_anchors.py
- tests/test_data_pipeline_doc.py
- tests/test_data_smoke_script.py
- tests/test_dataloader_bridge.py
- tests/test_engram.py
- tests/test_env_runtime.py
- tests/test_eval.py
- tests/test_external_research_contract.py
- tests/test_fim_transform.py
- tests/test_hybrid_lm.py
- tests/test_hybrid_lm_gradients.py
- tests/test_lint_mlx.py
- tests/test_m03_forward_parity_manifest_script.py
- tests/test_m04_train_step.py
- tests/test_m2rnn.py
- tests/test_mamba3.py
- tests/test_megatron_indexed.py
- tests/test_memory_audit.py
- tests/test_memory_runtime.py
- tests/test_metal_ops.py
- tests/test_mhc.py
- tests/test_mlx_lm_adapter.py
- tests/test_model_factory.py
- tests/test_moe.py
- tests/test_mtp_loss.py
- tests/test_muon_group_splitter.py
- tests/test_nam56r_pattern.py
- tests/test_ngram_hash.py
- tests/test_package_exports.py
- tests/test_parity_manifest.py
- tests/test_parquet_dataset.py
- tests/test_plasticity.py
- tests/test_profile.py
- tests/test_profile_capture_script.py
- tests/test_pytest_markers.py
- tests/test_real_parquet_samples.py
- tests/test_runtime_exports.py
- tests/test_seed_runtime.py
- tests/test_sequence_packing.py
- tests/test_stp_loss.py
- tests/test_structure_embedding.py
- tests/test_system_requirements_doc.py
- tests/test_tiny_train.py
- tests/test_token_dataset.py
- tests/test_tokenizer_contract.py
- tests/test_tokenizer_loader.py
- tests/test_train_hybrid_tiny_script.py
- tests/test_train_tiny_npz_script.py
- tests/test_training_exports.py

## Wave-Next Work

1. Keep the local tiny and hybrid tiny lanes green while increasing coverage.
   Mamba3 and M2RNN checkpoint save/resume, validation-loop, and structure
   side-channel variants now have local script-path receipts; keep them as
   regression lanes rather than next-work placeholders.
2. Keep the hybrid tiny A/M/E/R route-semantics regression lanes green before
   increasing dimensions. Current coverage includes single-route loss,
   route-specific gradients, route-specific optimizer updates, and mixed-route
   eager/compiled train-step checks; future growth should extend those
   contracts rather than replacing them with shape-only smoke tests.
3. Extend the data handoff path from fixed-shape NPZ and optional Parquet token
   shards to the standalone MegatronIndexedDataset seam. Tiny CLI/training
   ingress is wired through scripts/train_tiny_npz.py --dataset-format
   megatron; remaining work is source-converter side-channel preservation and a
   multi-shard sidecar/schema, not pulling Megatron into the Mac runtime.
4. Promote the current scripts/bench_matrix.py M4 Max smoke result into an
   archived JSON regression baseline, then grow it across a small matrix of
   batch/sequence/dtype/profile/route shapes. Track compile time separately from
   steady-state throughput.
5. Create a matched GB10 tiny benchmark runner only after the M4 Max JSON
   contract is stable. The comparison must use identical model source, shape,
   dtype, optimizer/update policy, data contract, warmup, measured step count,
   and metric definitions.
6. Profile before kernel work. Only consider a custom Metal kernel when a pure
   MLX implementation is correct, tests cover the op, and benchmark evidence
   identifies a stable hotspot.
7. If a training-path custom kernel is needed, implement it behind a pure MLX
   fallback and an mx.custom_function VJP/JVP. A forward-only Metal kernel is
   acceptable only for preprocessing, diagnostics, or optional non-training
   paths.
8. Add checkpoint sharding and a standalone RNG tensor payload only when
   single-file safetensors or stochastic-layer reproducibility become real
   blockers. Until then, keep the checkpoint format simple and fully tested.

## Blocked Or Not Proven

- Full NAM56R readiness is not proven. The repo has config/pattern helpers and a
  tiny hybrid smoke model, not full NAM56R capacity, parallelism, cache
  behavior, data scale, or production performance.
- Full NAM56R Megatron parity is fail-closed. The local MLX subset does not
  implement Transformer Engine, CUDA graph capture, NCCL, Triton, TileLang,
  native MTP, native DSA, sparse MLA, Hopper/GB10 linear-CE kernels, Megatron
  launcher parity, distributed optimizer, or TP/PP/VPP/EP/SP process-group
  behavior.
- Local Mamba3/M2RNN placement is layout-visible only. M and R routes are
  tiny MLX reference blocks at the source layer positions; they are not the
  source nam56r_full_spec.py, nam56r_te_spec.py, or nam56r_noconv_spec.py
  runtime with TE, Triton scans, TP mixer behavior, or H200/GB10 launch scripts.
- Stream H feature work is only partially sliced locally. `cppmega_mlx/nn/engram.py`
  and `cppmega_mlx/nn/mhc.py` are standalone modules, `cppmega_mlx/data/fim.py`
  is a fail-closed CPU FIM/iFIM transform slice, `cppmega_mlx/training/mtp.py`
  is local training-side MTP coverage, and `cppmega_mlx/training/stp_loss.py`
  is opt-in deterministic STP helper coverage. None of these are
  NAM56R-integrated, CUDA/Megatron parity receipts, full Stream H closure, or
  Hopper/Liger fused-CE evidence.
- MoE support is reference-local. The shared/routed expert defaults are
  regression anchors, but Megatron all-to-all dispatch, grouped GEMM,
  expert-parallel overlap, selective FP8 MoE, capacity/drop-pad policy, and
  Transformer Engine dispatcher monkey patches are not implemented.
- M4 Max vs GB10 parity is not proven. A local M4 benchmark is a local
  regression baseline only until a matched GB10 run exists with the protocol
  below.
- Megatron .bin/.idx support is wired into open_token_dataset; the tiny
  training script accepts --dataset-format megatron, but indexed batches remain
  token-only until the source converter preserves side channels and a multi-shard
  sidecar/schema is defined.
- Source-equivalent structure embedding is on the HybridTinyLM forward path
  through CppMegaStructureEmbedding; TinyLM still uses simplified
  summed/modded structure additions, and full Megatron launcher/training parity
  remains outside the tiny-local scaffold.
- Ngram hash enrichment is wired into HybridTinyLM and the NAM56R-to-hybrid
  config mapping as a model-derived additive embedding. Script-level env
  ingestion, source offload behavior, and Megatron PP/MTP checkpoint semantics
  remain documented non-local gaps.
- Parquet side-channel columns can be read locally, but the source
  Parquet-to-Megatron converter writes only token IDs, so structure metadata is
  not preserved across that handoff.
- Real Parquet samples are local-only under ignored data/parquet_samples/.
  Their presence proves that optional smoke tests can run on this machine; it
  does not create a portable repo fixture or a GB10 comparison row.
- Megatron PP/MTP checkpoint replica-id semantics for custom ngram/structure
  submodules are not represented in the MLX safetensors checkpoint format; this
  remains a non-goal unless Megatron checkpoint interop/export is required.
- HF Apple M4 kernels are not adopted into the training path.
- Custom Metal kernels are not training-critical. The current prototype
  cppmega_mlx/kernels/metal_ops.py is a squared_relu forward-only optional
  seam with pure MLX fallback and fail-closed explicit Metal mode.
- Distributed MLX training is not implemented in this repo. mx.distributed
  remains a future pattern source, not a current capability claim.
- Apple training readiness at large route shapes remains a verification gap:
  before any NAM56R-scale local training or training-path Metal kernel claim,
  collect a route/shape profile with MLX active/peak/cache memory, compile-cache
  churn, optimizer-state size, checkpoint/write behavior, and any macOS wired
  memory setting used. Kernel adoption still requires the pure-MLX fallback,
  dtype/shape parity, and custom VJP/JVP coverage described above.

## M4 Max vs GB10 Comparison Protocol

Strict rule: an M4 Max run is never a GB10 parity claim by itself. It is only a
local regression baseline until a matched GB10 run exists.

For every comparison, record:

- Exact repo snapshot or source archive used on both machines.
- Exact command line and JSON output from scripts/bench_tiny.py.
- Hardware label, device name, OS, MLX/Metal stack on M4 Max, and CUDA/driver or
  GB10 runtime stack on GB10.
- Model source, parameter count, dtype, batch size, sequence length, vocab size,
  hidden size, head count, layer count, MLP size, optimizer, learning rate, and
  gradient accumulation policy.
- Data contract: synthetic data vs the same token batch file. Do not compare
  synthetic M4 numbers to real-data GB10 numbers.
- Compile time, warmup count, measured step count, mean/median steady-state step
  time, tokens/sec, and peak memory.
- Memory semantics: MLX active/peak/cache bytes on M4 Max must not be compared
  as if they were whole-device GB10 telemetry.
- Wired-limit settings and whether they were applied.
- Thermal/power context and whether other GPU/accelerator work was running.

Protocol:

1. Run the M4 Max command and save JSON.
2. Run the GB10 command with the same shape, dtype, model source, warmup, and
   measured step count.
3. Compare the largest common batch first. Any larger single-host batch is a
   capacity study, not parity.
4. Keep compile time separate from steady-state throughput.
5. If one run changes a runtime setting such as wired limit, cache behavior, or
   precision, repeat the other side or mark the result as unmatched.

## Verification Commands

Collect tests:

bash
./.venv/bin/python -m pytest --collect-only -q


Run the lightweight suite:

bash
./.venv/bin/python -m pytest


Run the local M4 benchmark baseline:

bash
./.venv/bin/python scripts/bench_tiny.py \
  --batch-size 2 \
  --seq-len 64 \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --hardware-label "M4 Max" \
  --json


Create a tiny NPZ and run the train smoke:

bash
TMP_DIR="$(mktemp -d)"
TMP_NPZ="$TMP_DIR/tiny_tokens.npz"
./.venv/bin/python - "$TMP_NPZ" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
tokens = (np.arange(2 * 128, dtype=np.int32) % 64).reshape(2, 128)
np.savez(path, tokens=tokens, vocab_size=np.array(64, dtype=np.int32))
PY

./.venv/bin/python scripts/train_tiny_npz.py "$TMP_NPZ" \
  --batch-size 2 \
  --seq-len 64 \
  --steps 2 \
  --dtype bfloat16 \
  --json

rm -rf "$TMP_DIR"


## Research Sources Checked

- MLX GitHub/release:
  https://github.com/ml-explore/mlx
- MLX GitHub API, verified 2026-04-30:
  https://api.github.com/repos/ml-explore/mlx
- MLX latest release API, verified 2026-04-30:
  https://api.github.com/repos/ml-explore/mlx/releases/latest
- MLX compile docs:
  https://ml-explore.github.io/mlx/build/html/usage/compile.html
- MLX custom Metal kernel docs:
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX-LM GitHub/README:
  https://github.com/ml-explore/mlx-lm
- MLX-LM GitHub API, verified 2026-04-30:
  https://api.github.com/repos/ml-explore/mlx-lm
- MLX-LM latest release API, verified 2026-04-30:
  https://api.github.com/repos/ml-explore/mlx-lm/releases/latest
- MLX examples GitHub/README:
  https://github.com/ml-explore/mlx-examples
- MLX examples GitHub API, verified 2026-04-30:
  https://api.github.com/repos/ml-explore/mlx-examples
- MLX examples contents API, verified 2026-04-30:
  https://api.github.com/repos/ml-explore/mlx-examples/contents
- Hugging Face Apple M4 kernels listing:
  https://huggingface.co/kernels?hardware=apple-m4&sort=trending
- Hugging Face kernels API attempt, verified 2026-04-30:
  https://huggingface.co/api/kernels?hardware=apple-m4&sort=trending
- HF kernel cards for mlx-rmsnorm, mlx-quantization-metal-kernels,
  metal-flash-sdpa, paged-attention, gpt-oss-metal-kernels,
  bitsandbytes-mps, and activation.

Current source snapshot, verified 2026-04-30:

- MLX repo API returned HTTP 200; repo default branch is main, license is MIT,
  updated at 2026-04-30T14:38:51Z, pushed at 2026-04-28T16:39:57Z. Latest
  upstream release is v0.31.2, published 2026-04-22T01:40:04Z, at
  https://github.com/ml-explore/mlx/releases/tag/v0.31.2.
- MLX-LM repo API returned HTTP 200; repo default branch is main, license is
  MIT, updated at 2026-04-30T12:34:03Z, pushed at
  2026-04-23T13:54:02Z. Latest upstream release is v0.31.3, published
  2026-04-22T07:43:57Z, at
  https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3.
- Local execution package versions remain mlx==0.31.1, mlx-lm==0.31.2,
  mlx-metal==0.31.1, transformers==5.5.4, and safetensors==0.7.0; upstream
  release notes are drift context, not a local runtime claim.
- MLX examples repo API returned HTTP 200; repo default branch is main,
  license is MIT, updated at 2026-04-30T10:20:18Z, pushed at
  2026-04-06T18:56:05Z. The contents API showed transformer_lm, llms,
  lora, bert, and t5 directories as useful mechanics references. Treat
  these as candidates for reading only, not as an adopted dependency or trainer
  base.
- Hugging Face Apple M4 kernels listing returned HTTP 200 HTML with 10 embedded
  entries: kernels-community/mlx-rmsnorm, kernels-community/relu,
  kernels-community/paged-attention,
  kernels-community/mlx-quantization-metal-kernels,
  kernels-community/metal-flash-sdpa, kernels-community/gpt-oss-metal-kernels,
  kernels-community/bitsandbytes-mps, kernels-community/activation,
  drbh/test-repo, and drbh/first-kernel.
- The guessed Hugging Face kernels API endpoint returned HTTP 404 with
  Sorry, we can't find the page you are looking for.
- Later unauthenticated refresh attempts on 2026-04-30 hit GitHub REST API
  rate limit HTTP 403, so the GitHub API fields above remain a point-in-time
  snapshot. The Hugging Face Apple M4 HTML listing still returned HTTP 200 with
  the same 10 embedded entries.

Current source refresh for lane cppmega-mlx-refill-01, verified 2026-05-01:

- MLX custom Metal kernel docs returned HTTP 200 at
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html and
  show custom kernels can be wrapped with mlx.core.custom_function plus a VJP
  for differentiated training use.
- MLX custom_function docs returned HTTP 200 at
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html
  and define the vjp, jvp, and vmap override surface used by the kernel gate.
- MLX-LM README returned HTTP 200 at
  https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md and
  documents Apple-silicon fine-tuning, mx.distributed, and macOS 15
  iogpu.wired_limit_mb guidance for large models.
- Hugging Face Apple M4 kernel listing returned HTTP 200 at
  https://huggingface.co/kernels?hardware=apple-m4&sort=trending with 10
  embedded Metal-capable entries. This remains reference evidence only, not a
  cppmega adoption or performance claim.
