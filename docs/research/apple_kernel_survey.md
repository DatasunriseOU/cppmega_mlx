# Apple / HF Kernel Survey For cppmega.mlx

Date: 2026-04-30

Scope: current Apple Silicon kernel and MLX/MLX-LM integration guidance for the
cppmega.mlx port. This survey has been updated after rereading the current repo:
the compiled tiny training path, NPZ smoke, checkpointing, benchmark harness,
and prototype Metal seam are now implemented, so the plan below is no longer a
first-milestone-only sketch.

## Evidence Used

Local repo:

- README.md
- docs/porting_plan.md
- docs/perf_baseline.md
- docs/perf_mamba_m2rnn.md
- docs/checkpointing.md
- docs/metal_kernel_policy.md
- docs/research/mlx_core_and_metal.md
- docs/research/mlx_lm_training_patterns.md
- cppmega_mlx/training/compiled.py
- cppmega_mlx/data/token_dataset.py
- cppmega_mlx/models/tiny_lm.py
- cppmega_mlx/models/hybrid_lm.py
- cppmega_mlx/kernels/metal_ops.py
- scripts/bench_tiny.py
- scripts/bench_matrix.py
- scripts/train_tiny_npz.py
- scripts/train_hybrid_tiny.py
- current tests/test_*.py list

External/direct sources:

- https://github.com/ml-explore/mlx
- https://api.github.com/repos/ml-explore/mlx
- https://api.github.com/repos/ml-explore/mlx/releases/latest
- https://ml-explore.github.io/mlx/build/html/usage/compile.html
- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- https://github.com/ml-explore/mlx-lm
- https://api.github.com/repos/ml-explore/mlx-lm
- https://api.github.com/repos/ml-explore/mlx-lm/releases/latest
- https://github.com/ml-explore/mlx-examples
- https://api.github.com/repos/ml-explore/mlx-examples
- https://api.github.com/repos/ml-explore/mlx-examples/contents
- https://huggingface.co/kernels?hardware=apple-m4&sort=trending
- https://huggingface.co/api/kernels?hardware=apple-m4&sort=trending
- HF cards for kernels-community/mlx-rmsnorm,
   kernels-community/metal-flash-sdpa,
   kernels-community/paged-attention,
   kernels-community/mlx-quantization-metal-kernels,
   kernels-community/gpt-oss-metal-kernels,
   kernels-community/bitsandbytes-mps, and
   kernels-community/activation

Current external source snapshot, verified 2026-04-30 and W3.5 refreshed
2026-05-01:

- Prior same-day GitHub REST observations, retained as drift context only,
  recorded MLX latest release v0.31.2 published 2026-04-22T01:40:04Z and
  MLX-LM latest release v0.31.3 published 2026-04-22T07:43:57Z. Do not make
  tests depend on mutable star, fork, updated_at, or catalog counters.
- The W3.5 2026-05-01 GitHub refresh for the MLX and MLX-LM repo/latest-release
  endpoints returned HTTP 200.
- MLX README, MLX-LM README, MLX-LM loss source, and MLX custom Metal kernel
  docs direct fetches returned HTTP 200.
- MLX examples are useful reference directories such as transformer_lm,
  llms, lora, bert, and t5; they remain reading material, not trainer
  dependencies.
- Hugging Face Apple M4 kernels listing returned HTTP 200 HTML and, after
  HTML-unescaping the embedded KernelList payload, still exposed 10 kernel
  entries. The guessed API endpoint returned HTTP 404 with Sorry, we can't find
  the page you are looking for.
- Direct git ls-remote checks against the listed Hugging Face kernel repos
  returned live HEADs for 9 of the 10 repos. Several HEADs differ from the
  listing sha fields, so the HTML sha values are catalog metadata, not a
  pin for source adoption.
- Brave MCP web-search attempts returned HTTP 429 rate-limit errors, so the
  recorded external evidence above comes from direct primary fetches.

## Primary Receipts Refresh

Direct primary-source refresh, verified 2026-04-30 and rechecked by W3.5 on
2026-05-01 where noted:

- MLX README direct fetch returned HTTP 200 from
  https://raw.githubusercontent.com/ml-explore/mlx/main/README.md; the
  relevant current support surface is Apple Silicon arrays plus mlx.nn,
  mlx.optimizers, automatic differentiation, graph optimization, lazy
  computation, dynamic graphs, GPU execution, and unified memory.
- MLX-LM README direct fetch returned HTTP 200 from
  https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md; it is a
  generation/fine-tuning package on Apple Silicon with MLX, HF Hub integration,
  quantization/upload support, LoRA/full fine-tuning, and mx.distributed.
- MLX-LM loss source direct fetch returned HTTP 200 from
  https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/tuner/losses.py;
  it still uses can_run_metal(), mx.fast.metal_kernel,
  @mx.custom_function, and .vjp for differentiable KL/JS Metal loss
  kernels, while keeping non-Metal fallback paths.
  This is a pattern receipt, not permission to place a forward-only cppmega
  kernel on the training path without local VJP/JVP coverage.
- Hugging Face Apple M4 kernel listing direct fetch returned HTTP 200 from
  https://huggingface.co/kernels?hardware=apple-m4&sort=trending; the page
  embedded 10 Apple M4 entries: mlx-rmsnorm, relu, paged-attention,
  mlx-quantization-metal-kernels, metal-flash-sdpa,
  gpt-oss-metal-kernels, bitsandbytes-mps, activation, drbh/test-repo,
  and drbh/first-kernel. A same-turn direct git check could not read
  drbh/first-kernel without credentials; use the listing metadata for that
  demo repo only.

Repo decision from these receipts: first-party MLX ops and compiled execution
remain the local training substrate; MLX-LM remains a pattern source; HF Apple
M4 kernels remain source references and parity-test fixtures only. No HF kernel
is on the training path, and no M4 Max vs GB10 parity claim follows from these
receipts.
External kernel repositories must not be remote-loaded into cppmega training.
Any useful HF, MLX example, or MLX-LM kernel pattern must be pinned, licensed,
reimplemented or vendored in-tree, and covered by local fallback/parity and
VJP/JVP gates before it can move from reference material to training code.

## Current Repo Status

The current implementation already has the MLX-first substrate that earlier
research recommended:

- CompiledPretrainingStep normalizes batches to fixed keys, supports eager and
  compiled execution, accumulates gradients, and captures
  [model.state, optimizer.state, mx.random.state] in the compiled path.
- TokenNpzDataset reads fixed-shape NPZ token shards with optional side-channel
  arrays and deterministic cursor helpers. Parquet and Megatron indexed data are
  future seams.
- TinyLM and HybridTinyLM exercise local training and tiny A/M/E/R route
  semantics. The hybrid model is smoke-sized and explicitly not full NAM56R.
- scripts/bench_tiny.py records compile time separately from steady-state
  throughput and emits MLX memory telemetry.
- scripts/bench_matrix.py runs comparable benchmark rows across small
  batch/sequence/dtype/route/profile matrices using the same report contract.
- scripts/train_tiny_npz.py provides a local train smoke using the NPZ dataset
  and compiled/eager pretraining step.
- scripts/train_hybrid_tiny.py exercises the tiny A/M/E/R hybrid route model
  through the same compiled/eager step, checkpoint, and optional eval path.
- docs/perf_mamba_m2rnn.md records the current M4 Max smoke matrix for
  tiny, hybrid-m, and hybrid-r, plus eager/compiled Mamba3 and M2RNN
  train/checkpoint/resume receipts. These are local regression receipts only,
  not GB10 comparison evidence.
- cppmega_mlx/kernels/metal_ops.py contains only an optional prototype
  squared_relu Metal seam with pure MLX fallback and fail-closed explicit
  Metal mode. It is not training-critical.

## Decision

Use MLX first-party ops and compiled graph execution as the default kernel
substrate. Keep Hugging Face Apple M4 kernels as references and parity-test
fixtures only until there is a measured cppmega hotspot plus proven backward and
dtype behavior.

Adoption priority:

1. Keep the current compiled tiny train path green and extend it into a better
   smoke/pretraining harness.
2. Use first-party MLX/mlx.nn/mx.fast operations before custom kernels.
3. Use MLX-LM as a pattern source for compiled step structure, safetensors
   conventions, memory wiring, and future distributed shape, not as the
   cppmega pretraining trainer.
4. Use HF Apple M4 kernels for source review and op-level parity tests, not as
   remote-loaded production dependencies.
5. Introduce custom mx.fast.metal_kernel only behind a pure MLX fallback; if
   the op is differentiated, require mx.custom_function with VJP/JVP before it
   enters training.

## Source Matrix

| Source                       | Current use                                                                                                             | Decision                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| MLX core                     | Arrays, modules, optimizers, nn.value_and_grad, mx.compile, mx.eval, memory telemetry, fast ops, custom Metal API       | P0 runtime substrate                                                            |
| MLX-LM                       | Compiled-step pattern, wired-limit/memory conventions, safetensors and Hub conventions, future mx.distributed reference | P0/P1 pattern source, not trainer dependency                                    |
| MLX examples                 | transformer_lm, llms, lora, bert, and t5 mechanics for small model/training patterns                                    | Reference only; do not make it a trainer base or dependency                     |
| MLX community repos          | Reference implementations for simple model mechanics and fine-tuning/RL/multimodal ideas                                | Reference only                                                                  |
| HF Apple M4 kernels          | RMSNorm, activation, attention, quantization, GPT-style inference kernel references                                     | P2 references/test fixtures                                                     |
| Custom cppmega Metal kernels | Optional prototype seam only today                                                                                      | Blocked from training until fallback, parity, profiling, and VJP/JVP gates pass |

## HF Apple M4 Kernel Snapshot

The Apple M4 filtered HF kernels page showed 10 entries on 2026-04-30. Counts
below are point-in-time listing metadata, not adoption or quality proof. The
catalog is mutable even within the day; direct repo HEADs checked after the HTML
fetch are recorded separately below.

| Kernel                                           | Drivers                     | Downloads | Last modified            | SHA                                      |
| ------------------------------------------------ | --------------------------- | --------: | ------------------------ | ---------------------------------------- |
| kernels-community/mlx-rmsnorm                    | Metal                       |         5 | 2026-04-30T18:39:26.000Z | ba4229bb80ec474f8196e3b2feffb661bfba30be |
| kernels-community/relu                           | CUDA, ROCm, Metal, XPU, CPU |    38,972 | 2026-04-30T21:14:58.000Z | e417a11b50085675a3fbab75a75e5a1c137469e1 |
| kernels-community/paged-attention                | CUDA, ROCm, Metal           |        30 | 2026-04-30T21:30:35.000Z | e4cf9c63c76f5bbcb2142f69dbf9d3d7bb149fb9 |
| kernels-community/mlx-quantization-metal-kernels | Metal                       |        25 | 2026-04-30T18:43:20.000Z | 35b71b84e62f6ea5516f1834dbaa2d17df7fd169 |
| kernels-community/metal-flash-sdpa               | Metal                       |        41 | 2026-04-30T18:44:27.000Z | 76f2476def1cfad6bad9133d5c6cd5c05f5418a7 |
| kernels-community/gpt-oss-metal-kernels          | Metal                       |        44 | 2026-04-30T18:43:56.000Z | 7e271cf432005a22aca3d85b7fc6c82ce22e80b4 |
| kernels-community/bitsandbytes-mps               | Metal                       |         3 | 2026-04-30T18:43:03.000Z | bbf141fc155dd09af1b015c8d89e76393aa67408 |
| kernels-community/activation                     | CUDA, Metal                 |    34,769 | 2026-04-30T18:43:12.000Z | b3bfcb2c5da69cbf744c7f35bbde3c148c904872 |
| drbh/test-repo                                   | Metal                       |         0 | 2026-04-30T23:37:10.000Z | 50290b1041b82b3836ca449fb0688773740ec5eb |
| drbh/first-kernel                                | Metal                       |         1 | 2026-03-20T16:21:35.000Z | 798f87eaf694ebbc2e687bd7f8586b4d84842ed0 |

Direct repository HEAD refresh after the listing fetch:

| Kernel repo                                      | Direct HEAD                                                         |
| ------------------------------------------------ | ------------------------------------------------------------------- |
| kernels-community/mlx-rmsnorm                    | e2126471619665e1ceb7b7e60f008c90f36c27ac                            |
| kernels-community/relu                           | 48de06fe65377e49236206bc6f17d3a4aad66d1e                            |
| kernels-community/paged-attention                | 2d1ad74c4a91f197035a750e27b2d5b07b9ff511                            |
| kernels-community/mlx-quantization-metal-kernels | ddbe8b1ec02f0ca5f79449f0376cc85df5163836                            |
| kernels-community/metal-flash-sdpa               | 97f03dc042320ef985672e1468181a490780b023                            |
| kernels-community/gpt-oss-metal-kernels          | 3b158cb990be6100023423c0b479f594a8cc9767                            |
| kernels-community/bitsandbytes-mps               | ceb930bc347d17df1e8f0d160e89f17340267e98                            |
| kernels-community/activation                     | f07ca652c79e45849086ffd3ea15a3329362f2c1                            |
| drbh/test-repo                                   | 0278434780825e03ac50f946f04cb9a8966983ca                            |
| drbh/first-kernel                                | unavailable through unauthenticated git ls-remote; listing SHA only |

Kernel-specific decisions:

| Kernel                            | Card signal                                                                        | cppmega.mlx decision                                                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| mlx-rmsnorm                       | MIT card, rmsnorm_forward, rmsnorm_backward, no benchmark on card                  | Compare only after MLX RMSNorm baseline. Do not replace first-party MLX without parity and hotspot evidence.                    |
| relu                              | General multi-platform activation kernel                                           | Low value; built-in MLX activations cover the current need.                                                                     |
| paged-attention                   | Paged cache functions and a benchmark command                                      | Decode/KV-cache oriented. Defer for pretraining unless backward and train-shape relevance are proven.                           |
| mlx-quantization-metal-kernels    | Metal-only quantization listing                                                    | Quantization/inference reference only. Do not put on the pretraining path without a measured local hotspot and fallback parity. |
| metal-flash-sdpa                  | flash_attention_varlen functions, no benchmark on card                             | Useful attention source material. Keep MLX SDPA/reference attention first.                                                      |
| gpt-oss-metal-kernels             | Matmul, embedding, RMSNorm, RoPE, SDPA, top-k, routing metadata, scatter functions | Good reading material for future matmul/routing/scatter lanes; not a training dependency.                                       |
| bitsandbytes-mps                  | 4-bit quantize/dequantize/GEMV/GEMM/linear functions                               | Inference/quantization reference. Defer for pretraining.                                                                        |
| activation                        | Fused activation and activation-and-mul functions plus benchmark command           | Candidate only if profiling identifies a fused activation bottleneck.                                                           |
| drbh/test-repo, drbh/first-kernel | Demo/test repos                                                                    | Do not adopt.                                                                                                                   |

HF kernel adoption gate:

- Pin exact revision and license before any direct dependency.
- Add dtype/shape parity tests against the pure MLX implementation.
- Prove backward behavior for differentiated training inputs.
- Benchmark the local cppmega shape that motivated adoption.
- Keep a pure MLX fallback and an explicit unsupported-path failure mode.

## Integration Roadmap Impact

### Implemented

- Compiled tiny training step with stable batch keys and gradient accumulation.
- NPZ fixed-shape data reader with side-channel arrays.
- Optional Parquet handoff and standalone Megatron .bin/.idx token reader
  seams.
- Tiny LM and hybrid tiny A/M/E/R smoke model.
- JSON/compare-line benchmark harness with compile timing and memory telemetry,
  plus a small matrix runner.
- Tiny NPZ and hybrid tiny train smokes.
- Local M4 Max smoke matrix for tiny, Mamba3 route hybrid-m, and M2RNN
  route hybrid-r: 6662.73, 2785.08, and 4120.29 tokens/sec
  respectively for the intentionally tiny eager smoke shape.
- Mamba3 and M2RNN eager/compiled one-step smokes plus checkpoint/resume
  continuity through scripts/train_hybrid_tiny.py, including step cursor
  advance from 1 to 2 and final trained tokens 6 in each M/R mode.
- Safetensors checkpoint/resume helper.
- Optional forward-only prototype Metal kernel seam.

### Wave-Next

- Expand the current M/R route smoke receipts into a repeatable pretraining
  harness with validation batches and broader route/side-channel coverage.
- Turn the current M4 Max route smoke matrix into a stable archived JSON
  regression baseline before any GB10 comparison.
- Add a matched GB10 run only after the M4 JSON contract and route metadata are
  stable.
- Lock route-specific hybrid tiny gradients, masks, and structure side-channel
  behavior before increasing dimensions.
- Add op-level profiling before considering any HF-inspired or cppmega-owned
  Metal kernel.
- Add parquet and Megatron .bin/.idx readers after the NPZ contract stays
  green.

### Blocked / Not Proven

- No full NAM56R readiness claim.
- No M4 Max vs GB10 parity claim without matched GB10 data.
- No HF kernel is on the training path.
- No custom Metal kernel is accepted into differentiated training.
- No distributed MLX training path is implemented.

## M4 Max vs GB10 Protocol

A local M4 Max benchmark is a local regression baseline only. It is not a GB10
parity claim until the same benchmark contract has been run on GB10.

Minimum matched fields:

- same repo snapshot and script
- same model source and parameter count
- same batch size, sequence length, dtype, vocab/model dimensions, optimizer,
  learning rate, and gradient accumulation policy
- same data contract, either synthetic on both sides or the same token shard on
  both sides
- same warmup and measured step counts
- compile time reported separately from steady-state throughput
- memory telemetry reported with framework-specific semantics clearly labeled
- wired-limit or allocator residency settings recorded as part of the stack

If one host can run a larger batch, report that separately as a capacity result.
The parity table must start with the largest common shape.

## Commands To Keep In Reports

Collect tests:

bash
./.venv/bin/python -m pytest --collect-only -q


Run tests:

bash
./.venv/bin/python -m pytest


Run the local benchmark:

bash
./.venv/bin/python scripts/bench_tiny.py \
  --batch-size 2 \
  --seq-len 64 \
  --dtype bfloat16 \
  --warmup-steps 5 \
  --steps 20 \
  --hardware-label "M4 Max" \
  --json


Run a tiny NPZ smoke:

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

