# MLX Core And Metal Kernel Research

Date: 2026-04-30

Scope: lane 1 research for porting cppmega local training from the Megatron/CUDA production path into an MLX-native Apple Silicon path on the local M4 Max. This is not a source-code change plan for this lane; it is evidence and recommendations for the implementation lanes.

## Evidence Used

Local runnable evidence:

- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi
- .venv/lib/python3.13/site-packages/mlx/core/fast.pyi
- .venv/lib/python3.13/site-packages/mlx/core/metal.pyi
- .venv/lib/python3.13/site-packages/mlx/nn/utils.py
- .venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py
- .venv/lib/python3.13/site-packages/mlx_lm/tuner/losses.py
- cppmega_mlx/training/loop.py
- cppmega_mlx/training/compiled.py
- cppmega_mlx/training/checkpoint.py
- cppmega_mlx/training/eval.py
- cppmega_mlx/training/profile.py
- scripts/bench_tiny.py
- scripts/bench_matrix.py
- scripts/train_tiny_npz.py
- scripts/train_hybrid_tiny.py
- docs/perf_mamba_m2rnn.md
- docs/porting_plan.md

Local upstream clones inspected:

- /tmp/cppmega_mlx_research_mlx, ml-explore/mlx at e8ebdeb
- /tmp/cppmega_mlx_research_mlx_lm, ml-explore/mlx-lm at ed1fca4

External sources checked:

- https://github.com/ml-explore/mlx
- https://api.github.com/repos/ml-explore/mlx
- https://api.github.com/repos/ml-explore/mlx/releases/latest
- https://ml-explore.github.io/mlx/build/html/usage/compile.html
- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
- https://github.com/ml-explore/mlx-lm
- https://api.github.com/repos/ml-explore/mlx-lm
- https://api.github.com/repos/ml-explore/mlx-lm/releases/latest
- https://github.com/ml-explore/mlx-examples
- https://api.github.com/repos/ml-explore/mlx-examples
- https://api.github.com/repos/ml-explore/mlx-examples/contents
- https://huggingface.co/kernels?hardware=apple-m4&sort=trending
- https://huggingface.co/api/kernels?hardware=apple-m4&sort=trending

Current external source snapshot, verified 2026-04-30 and W3.5 refreshed
2026-05-01:

- Prior same-day GitHub REST observations, retained as drift context only,
  recorded ml-explore/mlx on default branch main, MIT license, pushed
  2026-04-28T16:39:57Z, latest release v0.31.2 published
  2026-04-22T01:40:04Z; and ml-explore/mlx-lm on default branch main,
  MIT license, pushed 2026-04-23T13:54:02Z, latest release v0.31.3
  published 2026-04-22T07:43:57Z. Do not make tests depend on mutable star,
  fork, updated_at, or catalog counters.
- The W3.5 2026-05-01 GitHub refresh for the MLX and MLX-LM repo/latest-release
  endpoints returned HTTP 200. Treat repo counters as mutable external context,
  not a current contract.
- MLX README, MLX-LM README, MLX-LM loss source, and MLX custom Metal kernel
  docs direct fetches returned HTTP 200.
- The Hugging Face Apple M4 kernels listing returned HTTP 200 HTML and, after
  HTML-unescaping the embedded KernelList payload, still exposed 10 kernel
  entries. The guessed JSON endpoint
  https://huggingface.co/api/kernels?hardware=apple-m4&sort=trending
  returned HTTP 404 with Sorry, we can't find the page you are looking for.
- Brave MCP web-search attempts returned HTTP 429 rate-limit errors, so the
  recorded external evidence above comes from direct primary fetches.

## Primary Receipts Refresh

Direct primary-source refresh, verified 2026-04-30 and rechecked by W3.5 on
2026-05-01 where noted:

- MLX README direct fetch returned HTTP 200 from
  https://raw.githubusercontent.com/ml-explore/mlx/main/README.md. It
  describes MLX as an Apple Silicon array framework with mlx.nn,
  mlx.optimizers, automatic differentiation, graph optimization, lazy
  computation, dynamic graphs, CPU/GPU execution, and unified memory.
- MLX-LM README direct fetch returned HTTP 200 from
  https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md. It
  describes generation and fine-tuning on Apple Silicon with MLX, HF Hub
  integration, quantization/upload support, low-rank and full fine-tuning, and
  mx.distributed inference/fine-tuning.
- Hugging Face Apple M4 kernel listing direct fetch returned HTTP 200 from
  https://huggingface.co/kernels?hardware=apple-m4&sort=trending and exposed
  10 Apple M4 kernel entries: kernels-community/mlx-rmsnorm,
  kernels-community/relu, kernels-community/paged-attention,
  kernels-community/mlx-quantization-metal-kernels,
  kernels-community/metal-flash-sdpa,
  kernels-community/gpt-oss-metal-kernels,
  kernels-community/bitsandbytes-mps, kernels-community/activation,
  drbh/test-repo, and drbh/first-kernel.
- MLX-LM loss source direct fetch returned HTTP 200 from
  https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/tuner/losses.py;
  it still uses can_run_metal(), mx.fast.metal_kernel,
  @mx.custom_function, and .vjp for differentiable KL/JS Metal loss
  kernels, while keeping non-Metal fallback paths.
  Treat this as a reference pattern only; local cppmega training use still
  requires fallback parity plus explicit VJP/JVP gates.

Repo decision from these receipts: MLX is the local runtime substrate, MLX-LM is
a pattern source for training mechanics, and HF Apple M4 kernels are
reference-only receipts for source review and parity tests. They are not
training dependencies and do not create any M4 Max vs GB10 parity claim.
External kernels must not be remote-loaded into the trainer; any useful pattern
must be pinned, licensed, reimplemented or vendored in-tree, and gated by the
same fallback/parity/VJP checks before training use.

The local .venv stubs remain the highest-trust source for what this checkout
can run immediately. Upstream latest releases are newer than the installed local
execution stack, so any v0.31.2 / v0.31.3 notes below are external drift
context, not a claim that the local package has those APIs.

MLX examples are candidate reading material only. Use them to cross-check
minimal transformer_lm, llms, lora, bert, and t5 mechanics, but keep
cppmega's trainer, data contract, checkpoints, and kernels owned in this repo.

## Local Stack And Device

Installed packages:

- mlx==0.31.1
- mlx-lm==0.31.2
- mlx-metal==0.31.1

Upstream comparison, verified 2026-04-30:

- latest upstream MLX release: v0.31.2
- latest upstream MLX-LM release: v0.31.3
- local execution source of truth: installed mlx==0.31.1,
  mlx-lm==0.31.2, and mlx-metal==0.31.1

Default MLX device:

- Device(gpu, 0)

Local Apple GPU device info from mx.device_info():

- device_name: Apple M4 Max
- architecture: applegpu_g16s
- memory_size: 137438953472 bytes, approximately 128 GiB unified memory
- max_recommended_working_set_size: 115448725504 bytes, approximately 107.5 GiB
- max_buffer_length: 86586540032 bytes, approximately 80.6 GiB
- resource_limit: 499000

Implication: cppmega.mlx can target large local experiments if weights, optimizer state, activations, and compile cache fit under the wired/working-set limits. One single MLX buffer cannot exceed the max buffer length, so very large tensor layouts need sharding/chunking even when total unified memory is sufficient.

## Current cppmega.mlx Baseline

The repo already follows the right direction:

- README.md says this is an MLX-native local training port, not a direct Megatron/CUDA runtime port.
- docs/porting_plan.md says the Mac lane is correctness-first, preserving cppmega model semantics while replacing the runtime with MLX, MLX-LM patterns, and later Metal kernels.
- cppmega_mlx/training/loop.py has a minimal eager training step using nn.value_and_grad(model, loss_fn), optimizer.update(model, grads), and mx.eval(model.parameters(), optimizer.state, loss, ntokens).
- cppmega_mlx/training/compiled.py now owns the MLX-LM-style compiled/eager
  pretraining step: fixed-key batch normalization, gradient accumulation,
  state capture for [model.state, optimizer.state, mx.random.state], and a
  shape/dtype signature guard for compiled batches.
- cppmega_mlx/training/checkpoint.py now saves full model parameters,
  optional optimizer state, package/version metadata, tokenizer/vocab contract
  fields, and CompiledPretrainingStep resume metadata including pending
  gradient accumulation.
- cppmega_mlx/training/eval.py and cppmega_mlx/training/profile.py provide
  small eval and timing/memory telemetry helpers.
- scripts/bench_tiny.py, scripts/bench_matrix.py, scripts/train_tiny_npz.py,
  and scripts/train_hybrid_tiny.py are the current local evidence paths.
- docs/perf_mamba_m2rnn.md now records a local Apple M4 Max route smoke
  baseline: tiny at 6662.73 tokens/sec, Mamba3 hybrid-m at 2785.08
  tokens/sec, and M2RNN hybrid-r at 4120.29 tokens/sec for the intentionally
  tiny eager smoke shape. It also records eager and compiled M/R train
  checkpoint/resume continuity through scripts/train_hybrid_tiny.py.

The compiled-step milestone is therefore implemented. The next training-loop
work is not a new compile wrapper; it is hardening route/side-channel coverage,
validation, larger-but-still-stable benchmark matrix collection, and profiling
before any custom Metal adoption.

## Training Primitives

Relevant local stubs:

- .venv/lib/python3.13/site-packages/mlx/nn/utils.py:12-38 defines nn.value_and_grad(model, fn) over model.trainable_parameters().
- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:4204-4254 defines mx.value_and_grad.
- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:4255-4275 defines mx.grad.
- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:4298-4326 defines mx.compile.
- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:4340-4354 defines mx.checkpoint.
- .venv/lib/python3.13/site-packages/mlx/nn/utils.py:41-71 defines an nn.utils.checkpoint(module, fn=None) helper, but mlx.nn.checkpoint is not exported as a top-level nn.checkpoint in the installed package.

Training recommendation:

1. Keep the model as an mlx.nn.Module.
2. Use nn.value_and_grad(model, loss_fn) for normal module training.
3. Capture model, optimizer, and RNG state in one compiled step: state = [model.state, optimizer.state, mx.random.state].
4. Use @partial(mx.compile, inputs=state, outputs=state) around the outer step so forward, backward, and update live in one graph.
5. Use mx.eval(state, metrics, grad_accum) at each iteration boundary to force completion and collect timings.
6. Add gradient accumulation before trying bigger batch/sequence sizes.
7. Add checkpointing per expensive layer after correctness and baseline timings are stable.

The MLX-LM trainer is the closest local pattern:

- .venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py:228-229 sets mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"]) when Metal is available.
- .venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py:237-240 enables gradient checkpointing and builds nn.value_and_grad(model, loss).
- .venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py:246-262 captures [model.state, optimizer.state, mx.random.state], compiles step, accumulates grads, averages distributed grads, scales by accumulation steps, and updates the optimizer.
- .venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py:319-329 calls the compiled step, then mx.eval(state, losses, n_tokens, grad_accum).

## Compile, value_and_grad, And Checkpoint

What matters for cppmega:

- mx.compile fuses operations and reduces graph overhead, but the first call compiles and can be slow. Compile once and reuse.
- mx.compile recompiles when input shape, rank, dtype, or input count changes. Bucket sequence lengths and batch shapes to keep compile cache reuse high.
- Avoid creating compiled lambdas inside training loops.
- Compiled functions are intended to be pure. Do not inspect arrays, append arrays to Python state, or depend on side effects inside a compiled step.
- shapeless=True exists, but shape-dependent model code can break under shapeless compilation; use it only after explicit tests.
- mx.checkpoint recomputes intermediate states during backward to reduce activation memory. Apply it at transformer/Mamba/M2RNN block granularity, not inside tiny ops.

Recommended compiled step shape:

python
from functools import partial

state = [model.state, optimizer.state, mx.random.state]
loss_and_grad = nn.value_and_grad(model, loss_fn)

@partial(mx.compile, inputs=state, outputs=state)
def step(batch, prev_grad, do_update):
    (loss, ntokens), grad = loss_and_grad(model, batch)
    if prev_grad is not None:
        grad = tree_map(lambda x, y: x + y, grad, prev_grad)
    if do_update:
        grad = tree_map(lambda x: x / grad_accum_steps, grad)
        optimizer.update(model, grad)
        grad = None
    return loss, ntokens, grad


For this repo, the pattern above is already represented by
cppmega_mlx/training/compiled.py. Keep cppmega_mlx/training/loop.py as a
minimal eager smoke/debug helper; future changes should extend the script and
test harness around CompiledPretrainingStep instead of replacing the eager
helper.

## Fast Ops And SDPA

Local installed mlx.core.fast APIs:

- .venv/lib/python3.13/site-packages/mlx/core/fast.pyi:12-27: mx.fast.rms_norm
- .venv/lib/python3.13/site-packages/mlx/core/fast.pyi:29-47: mx.fast.layer_norm
- .venv/lib/python3.13/site-packages/mlx/core/fast.pyi:49-76: mx.fast.rope
- .venv/lib/python3.13/site-packages/mlx/core/fast.pyi:78-138: mx.fast.scaled_dot_product_attention
- .venv/lib/python3.13/site-packages/mlx/core/fast.pyi:140-196: mx.fast.metal_kernel

mx.fast.scaled_dot_product_attention should be the default attention target before any custom attention kernel. It supports MHA, GQA, and MQA with:

- q: [B, N_q, T_q, D]
- k: [B, N_kv, T_kv, D]
- v: [B, N_kv, T_kv, D]
- mask: None, "causal", or boolean/additive array mask broadcast-compatible with [B, N, T_q, T_kv]
- sinks: optional attention sinks
- fp32 softmax regardless of input precision

Important limitations from upstream source:

- /tmp/cppmega_mlx_research_mlx/mlx/fast.cpp:622-710 enforces rank-4 q/k/v, matching batch dimensions, matching q/k head dim, matching k/v KV-head count, and n_q_heads % n_kv_heads == 0.
- /tmp/cppmega_mlx_research_mlx/mlx/fast.cpp:898-904 rejects VJP with respect to mask or attention sinks. This is fine for normal training where masks are not learnable, but it matters for any learned/synthetic masking path.

Recommendation:

- Port dense/GQA attention to MLX SDPA first.
- Keep sparse/MLA/custom masking out of the first performance target unless the CUDA model semantics require it for correctness.
- If a custom attention kernel becomes necessary, start from an SDPA parity test and retain the MLX SDPA fallback.

## Metal Kernel API

Local installed API:

- .venv/lib/python3.13/site-packages/mlx/core/fast.pyi:140-196 defines mx.fast.metal_kernel(name, input_names, output_names, source, header="", ensure_row_contiguous=True, atomic_outputs=False).
- .venv/lib/python3.13/site-packages/mlx/core/metal.pyi:6-37 exposes mx.metal.is_available, memory telemetry, start_capture(path), stop_capture(), and device_info().
- Local capability refresh on 2026-04-30 showed mx.default_device() as
  Device(gpu, 0), mx.metal.is_available() as True,
  mx.fast.metal_kernel present, and mx.custom_function present on
  Apple M4 Max.
- Local cppmega_mlx.kernels.metal_ops.metal_kernel_status() returned
  MetalKernelStatus(available=True, reason='Metal kernel path is available')
  with the installed stack (mlx==0.31.1, mlx-lm==0.31.2,
  mlx-metal==0.31.1).
- A local squared_relu(..., backend="metal") smoke path is therefore eligible
  on this host, but only for supported non-empty floating tensors and only
  behind the pure-MLX fallback policy.

Upstream docs:

- /tmp/cppmega_mlx_research_mlx/docs/src/dev/custom_metal_kernels.rst:45-47 says each kernel construction creates a Metal library and can trigger JIT compilation, so build each kernel object once and reuse it.
- /tmp/cppmega_mlx_research_mlx/docs/src/dev/custom_metal_kernels.rst:101-132 documents ensure_row_contiguous; the default can copy inputs to row-contiguous layout, and disabling it requires stride-aware indexing.
- /tmp/cppmega_mlx_research_mlx/docs/src/dev/custom_metal_kernels.rst:201-203 shows custom_function plus metal_kernel for forward and backward kernels.
- /tmp/cppmega_mlx_research_mlx/docs/src/dev/custom_metal_kernels.rst:301-313 documents VJP with init_value=0 and atomic_outputs=True for scatter-style backward accumulation.

Rules for cppmega kernels:

- can_run_metal() is only the device/backend probe: default MLX GPU,
  mx.metal present, and mx.metal.is_available().
- Per-kernel readiness must be reported through metal_kernel_status(...),
  which also checks for mx.fast.metal_kernel, supported dtype, non-empty
  input, and a constructed kernel object.
- backend="auto" may fall back to pure MLX for any unsupported Metal status;
  backend="metal" must fail closed with the status reason.
- Unknown backend labels, including "cuda", are invalid for the local
  MLX/Metal path and must raise rather than become compatibility aliases.
- TrainingKernelStatus(training_safe=True, ...) must be impossible unless
  in-tree ownership, source pinning, license coverage, pure-MLX fallback
  coverage, local parity coverage, profiled hotspot evidence, differentiability,
  VJP/backward coverage, and an MLX fallback backend are all true. Forward-mode
  JVP coverage can be tracked separately, but callers that depend on mx.jvp
  must require it before using a custom Metal training path.
- Plain mx.fast.metal_kernel is not enough for a training op if the op has custom gradients. Wrap it in @mx.custom_function and define .vjp or .jvp.
- For forward-only preprocessing or non-trainable features, a plain Metal kernel can be acceptable if it is outside the differentiated loss path.
- Remote HF kernels, MLX examples, and MLX-LM kernels are references only. They
  do not become supported training dependencies unless the source is pinned,
  licensed, owned in-tree, backed by profiled hotspot evidence, and passes the
  same pure-MLX fallback, explicit Metal-failure, dtype parity, and VJP/JVP
  gates.
- Build kernel objects at module import or factory initialization time, not inside the training loop.
- Use ensure_row_contiguous=True only where the hidden copy is acceptable. For hot gather/scatter kernels, prefer explicit stride-aware kernels and tests that detect accidental copies.
- For backward scatter/gather, expect atomics and zero initialization. Test deterministic tolerances carefully.

Local lint guardrails:

- tools/lint_mlx.py enforces MLX001 for mx.array(<python scalar literal>)
  without an explicit dtype.
- MLX002 blocks ad hoc mx.fast.metal_kernel construction outside
  cppmega_mlx/kernels/metal_ops.py so custom kernels stay behind the owned
  fallback/parity/VJP/JVP/profile-evidence policy seam.
- MLX003 blocks differentiated custom Metal use when a file combines
  metal_kernel with mx/nn autodiff calls but lacks @mx.custom_function plus an
  explicit .vjp or .jvp marker.
- MLX004 keeps benchmark throughput honest by flagging tokens_per_second
  derived from first-call/compile timing and compile_time_s derived from warmup
  or steady-state step timings.

## custom_function, VJP, And JVP

Local installed API:

- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:3976-4104 defines mx.custom_function with optional .vjp, .jvp, and .vmap.
- Captured variables are treated as constants; gradients are computed only for explicit inputs.
- The VJP receives primals, cotangents, and outputs, and must return a pytree matching the primals.
- The JVP receives primals and tangents, and must return a pytree matching the outputs.

MLX-LM has real differentiable Metal examples:

- .venv/lib/python3.13/site-packages/mlx_lm/tuner/losses.py:7-8 checks mx.default_device() == mx.gpu and mx.metal.is_available().
- .venv/lib/python3.13/site-packages/mlx_lm/tuner/losses.py:344-374 wraps KL divergence Metal forward/backward kernels with @mx.custom_function and .vjp.
- .venv/lib/python3.13/site-packages/mlx_lm/tuner/losses.py:750-782 does the same for JS divergence.

Recommended cppmega differentiable kernel targets, in order:

1. Fused next-token cross entropy or auxiliary loss reductions once the vocabulary/sequence shapes are stable.
2. MoE/top-k routing and expert packing/scatter if pure MLX indexing becomes the bottleneck.
3. Engram/ngram and structure embedding enrichment if profiling shows they are memory-bound and frequent enough.
4. Mamba/M2RNN scan/chunk kernels after the reference MLX block is correct.
5. Custom sparse/MLA attention only after SDPA-backed attention is correct and measured.

## Memory, Wired Limit, And Profiling

Relevant local stubs:

- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:755-780: get_active_memory, get_peak_memory, reset_peak_memory, get_cache_memory.
- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:782-817: set_memory_limit, set_cache_limit.
- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:819-847: set_wired_limit.
- .venv/lib/python3.13/site-packages/mlx/core/__init__.pyi:849-854: clear_cache.
- /tmp/cppmega_mlx_research_mlx/mlx/backend/metal/allocator.cpp:243-250 rejects wired limits above max_recommended_working_set_size.

MLX-LM large-model note:

- /tmp/cppmega_mlx_research_mlx_lm/README.md:260-283 says large models can be slow when large relative to RAM, MLX-LM attempts to wire model/cache memory on macOS 15+, and the system wired limit can be increased with sudo sysctl iogpu.wired_limit_mb=N where N is larger than model size but smaller than machine memory.

Recommended startup behavior:

python
if mx.metal.is_available():
    info = mx.device_info()
    mx.set_wired_limit(info["max_recommended_working_set_size"])


Recommended telemetry per training report:

- active memory: mx.get_active_memory()
- peak memory: mx.get_peak_memory()
- cache memory: mx.get_cache_memory()
- reset peak after warmup: mx.reset_peak_memory()
- optional cache pressure relief: mx.clear_cache()
- Metal trace around short profiler windows: mx.metal.start_capture("step.gputrace") and mx.metal.stop_capture()

Memory strategy:

- Always warm up before timing because compile/JIT costs distort first-step throughput.
- Keep static shape buckets to avoid compile churn and allocator churn.
- Use activation checkpointing before custom kernels when the issue is activation memory.
- Use gradient accumulation when the issue is batch memory.
- Use chunking when a single tensor risks the max_buffer_length ceiling.

## Hugging Face Apple M4 Kernels

The Apple M4 filtered Hugging Face kernels page returned HTTP 200 and 10
trending kernels on 2026-04-30. The unsupported API attempt noted above returned
HTTP 404, so the snapshot below comes from the HTML-embedded KernelList
metadata.

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

These should be treated as reference material, not production dependencies. The immediately relevant items for cppmega are metal-flash-sdpa, paged-attention, mlx-rmsnorm, mlx-quantization-metal-kernels, activation, and gpt-oss-metal-kernels.

Recommendation:

- Do not add a dependency on HF kernels in the first local trainer.
- Use them to compare Metal coding patterns, launch geometry, and benchmark methodology.
- Keep our first production kernels in-tree behind MLX fallbacks and parity tests.

## Limitations And Risk

MLX vs Megatron/CUDA:

- Do not emulate Megatron runtime constructs directly. Rebuild the local trainer around MLX Module, Optimizer, value_and_grad, compile, and MLX-native checkpointing.
- There is no local equivalent to CUDA Transformer Engine, CUTLASS, Triton, NCCL-heavy tensor/pipeline/expert parallelism, or Hopper/Blackwell FP8/MXFP8 tensor paths.
- MLX distributed exists, and MLX-LM initializes mx.distributed, but a single M4 Max lane should first target single-node correctness and single-GPU throughput.

Compile and kernel risks:

- Shape churn will cause recompilation; bucket data.
- Python side effects inside compiled functions are unsafe.
- Hidden row-contiguous copies can make a kernel look correct but slow.
- Custom VJP/JVP bugs can silently poison training; every custom kernel needs finite-difference or MLX-reference gradient tests on small shapes.
- Atomics in backward can introduce nondeterministic accumulation order; use tolerances and compare against reference gradients.
- Metal kernel JIT/build overhead must be excluded from steady-state timing.

Throughput risk:

- GB10 throughput can only be evaluated with matched GB10 rows for the same
  reduced MLX-native workload shape, data contract, dtype, and timing protocol.
- M4-only rows cannot support workloads whose CUDA speed comes from
  TE/CUTLASS/Triton kernels, Blackwell FP8/MXFP8 tensor paths, or multi-GPU
  Megatron parallelism without algorithmic rewrites.
- The benchmark target should therefore be stable local iteration throughput for
  the MLX-native reduced lane plus an honest matched-row comparison protocol,
  not blanket parity with production Megatron on GB10 or H200.

## Recommended Port Order

1. Keep CompiledPretrainingStep green in both eager and compiled modes while
   extending the current Mamba3/M2RNN checkpoint/resume receipts into broader
   route/side-channel and validation coverage.
2. Use the existing bench_tiny.py and bench_matrix.py reports as the
   benchmark contract: tokens/sec, compile time separated from steady-state
   time, active/peak/cache memory, batch/sequence shape, dtype, route, and model
   recipe.
3. Use built-in MLX fast ops first: SDPA, RMSNorm, RoPE, layer norm, matmul, and
   compiled fused elementwise operations.
4. Add mx.checkpoint around large blocks if memory blocks the target
   batch/sequence length.
5. Profile after correctness. Only then create custom Metal kernels for
   measured bottlenecks.
6. Wrap differentiable kernels with mx.custom_function and .vjp or .jvp,
   with pure-MLX forward/backward reference tests.
7. Use HF Apple M4 kernels and MLX-LM loss kernels as implementation references,
   but keep cppmega kernels in-tree with MLX fallbacks.
8. Keep GB10 comparison honest: compare a stable MLX-native local lane against
   an equivalent local/GB10 lane, not against full CUDA Megatron features that
   the Mac lane intentionally does not implement.

## Concrete Acceptance Criteria For The Next Implementation Lane

- Compiled step continues to run the tiny and hybrid tiny smoke paths without
  changing loss semantics.
- Checkpoint/resume continues to restore model, optimizer, step cursor, and
  mid-gradient accumulation state where present, including the existing
  eager/compiled Mamba3 and M2RNN script receipts.
- A fixed-shape warmup plus measured loop reports stable tokens/sec through
  bench_tiny.py and bench_matrix.py.
- Memory report includes active, peak, cache, and wired-limit values.
- Benchmarks keep compile/JIT time separate from steady-state step time.
- No custom Metal kernels are added until the benchmark identifies a hotspot.
- Any custom kernel has a pure MLX fallback plus forward and gradient parity tests.
