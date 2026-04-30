# MLX-LM Training And Checkpoint Patterns

Date: 2026-04-30

Scope: lane 2 research for reusing MLX-LM training, checkpoint, tokenizer, and
data patterns in a custom cppmega MLX pretraining loop on the local M4 Max. This
is a documentation-only research lane.

## Evidence Used

Local runnable stack:

- `mlx==0.31.1`
- `mlx-lm==0.31.2`
- `transformers==5.5.4`
- `safetensors==0.7.0`
- default device: `Device(gpu, 0)`
- Apple GPU: `Apple M4 Max`, `applegpu_g16s`, 128 GiB unified memory

Local files inspected:

- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/trainer.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/datasets.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/utils.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tuner/lora.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/lora.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/utils.py`
- `.venv/lib/python3.13/site-packages/mlx_lm/tokenizer_utils.py`
- `.venv/lib/python3.13/site-packages/mlx/core/__init__.pyi`
- `cppmega_mlx/training/loop.py`
- `cppmega_mlx/training/compiled.py`
- `cppmega_mlx/training/checkpoint.py`
- `cppmega_mlx/training/eval.py`
- `cppmega_mlx/training/profile.py`
- `cppmega_mlx/data/batch.py`
- `docs/perf_mamba_m2rnn.md`

External/direct upstream checks:

- `https://github.com/ml-explore/mlx`
- `https://api.github.com/repos/ml-explore/mlx`
- `https://api.github.com/repos/ml-explore/mlx/releases/latest`
- `https://github.com/ml-explore/mlx-lm`
- `https://api.github.com/repos/ml-explore/mlx-lm`
- `https://api.github.com/repos/ml-explore/mlx-lm/releases/latest`
- `https://github.com/ml-explore/mlx-examples`
- `https://api.github.com/repos/ml-explore/mlx-examples`
- `https://api.github.com/repos/ml-explore/mlx-examples/contents`
- `/tmp/cppmega_mlx_research/mlx-lm`, `ml-explore/mlx-lm` at `ed1fca4`
- `https://huggingface.co/kernels?hardware=apple-m4&sort=trending`
- `https://huggingface.co/api/kernels?hardware=apple-m4&sort=trending`

The installed `.venv` is the execution source of truth for this checkout. The
fresh upstream clone is a drift check; `trainer.py` and `datasets.py` matched the
installed package, while `utils.py` and `tokenizer_utils.py` had small upstream
drift.

Current external source snapshot, verified 2026-04-30:

- MLX repo API returned HTTP 200 for `ml-explore/mlx`: default branch `main`,
  MIT license, updated `2026-04-30T18:24:14Z`, pushed
  `2026-04-28T16:39:57Z`, 25,877 stars, 1,732 forks; latest release is
  `v0.31.2`, published `2026-04-22T01:40:04Z`.
- MLX-LM repo API returned HTTP 200 for `ml-explore/mlx-lm`: default branch
  `main`, MIT license, updated `2026-04-30T20:14:26Z`, pushed
  `2026-04-23T13:54:02Z`, 5,096 stars, 634 forks; latest release is
  `v0.31.3`, published `2026-04-22T07:43:57Z`.
- MLX-LM `v0.31.3` highlights are bug fixes plus a thread-local generation
  stream to accompany MLX `v0.31.2`. That is external drift context only; this
  document keeps the installed `mlx-lm==0.31.2` trainer and stubs as the local
  execution contract.
- MLX examples repo API returned HTTP 200 for `ml-explore/mlx-examples`:
  default branch `main`, MIT license, updated `2026-04-30T20:14:21Z`, pushed
  `2026-04-06T18:56:05Z`. The contents API showed `transformer_lm`, `llms`,
  `lora`, `bert`, and `t5` reference directories. These are mechanics
  references only, not a cppmega trainer dependency.
- The Hugging Face Apple M4 kernels listing returned HTTP 200 HTML and embedded
  10 `KernelList` entries. The guessed API endpoint returned HTTP 404 with
  `Sorry, we can't find the page you are looking for.`
- A later direct refresh on 2026-04-30 still returned HTTP 200 for the GitHub
  REST repo and latest-release endpoints above. The Hugging Face Apple M4 HTML
  listing still returned HTTP 200 with the same 10 embedded entries, and the
  guessed API endpoint still returned HTTP 404.

## Primary Receipts Refresh

Direct primary-source refresh, verified 2026-04-30:

- MLX README direct fetch returned HTTP 200 from
  `https://raw.githubusercontent.com/ml-explore/mlx/main/README.md` and
  confirmed the Apple Silicon runtime features cppmega.mlx should build on:
  `mlx.nn`, `mlx.optimizers`, automatic differentiation, graph optimization,
  lazy computation, dynamic graphs, GPU execution, and unified memory.
- MLX-LM README direct fetch returned HTTP 200 from
  `https://raw.githubusercontent.com/ml-explore/mlx-lm/main/README.md` and
  confirmed it is an Apple Silicon generation/fine-tuning package with HF Hub
  integration, quantization/upload support, low-rank/full fine-tuning, and
  `mx.distributed`.
- Hugging Face Apple M4 kernel listing direct fetch returned HTTP 200 from
  `https://huggingface.co/kernels?hardware=apple-m4&sort=trending` and showed 10
  entries, including `mlx-rmsnorm`, `mlx-quantization-metal-kernels`,
  `metal-flash-sdpa`, `paged-attention`, `gpt-oss-metal-kernels`,
  `bitsandbytes-mps`, and `activation`.

Repo decision from these receipts: MLX-LM should remain a source of training
patterns, not the cppmega trainer base. HF Apple M4 kernels are reference-only
material and test-fixture candidates; they are not pretraining dependencies and
do not prove M4 Max parity with GB10.

## Bottom Line

Use MLX-LM as a pattern library, not as the cppmega pretraining framework.
Use `mlx-examples` the same way: a reference for small examples and mechanics,
not an adopted dependency or trainer base.

Reuse:

- compiled train-step structure from `mlx_lm.tuner.trainer.train`
- `nn.value_and_grad(model, loss_fn)` for module-centered gradients
- `@partial(mx.compile, inputs=state, outputs=state)` with `state = [model.state, optimizer.state, mx.random.state]`
- `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` on Metal
- `mx.eval(...)` after each step/report boundary to force GPU completion
- `average_gradients(grad)` if `mx.distributed` is used later
- MLX-LM sharded `model*.safetensors` plus `model.safetensors.index.json` layout
- Hugging Face tokenizer loading/saving conventions when compatible

Do not reuse directly:

- `mlx_lm.lora.train_model` as a pretraining loop
- `mlx_lm.tuner.trainer.train` without replacing its adapter-save path
- `mlx_lm.tuner.trainer.default_loss` for cppmega side-channel batches
- `mlx_lm.tuner.trainer.iterate_batches` for high-throughput pretraining
- MLX-LM LoRA adapter save/load as a full-training checkpoint
- `mlx_lm.utils.load_model` unless cppmega exposes a compatible `Model` and `ModelArgs`

The current `cppmega_mlx/training/loop.py` remains an eager one-step smoke
helper, and `cppmega_mlx/training/compiled.py` now provides the cppmega-owned
compiled pretraining step that borrows MLX-LM's execution pattern while owning
data, loss, checkpoint, and resume semantics. The current
`docs/perf_mamba_m2rnn.md` receipt confirms eager and compiled Mamba3/M2RNN
one-step train smokes plus checkpoint/resume cursor continuity through
`scripts/train_hybrid_tiny.py`. The next implementation work is to harden that
lane across route gradients, side-channel batches, validation, and larger
benchmark matrix collection.

## Trainer API Patterns To Reuse

Installed MLX-LM trainer surfaces:

- `TrainingArgs(batch_size, iters, val_batches, steps_per_report, steps_per_eval, steps_per_save, max_seq_length, adapter_file, grad_checkpoint, grad_accumulation_steps, clear_cache_threshold)`
- `default_loss(model, batch, lengths) -> (loss, ntoks)`
- `iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None)`
- `evaluate(model, dataset, batch_size, num_batches, max_seq_length=2048, loss=default_loss, iterate_batches=iterate_batches, clear_cache_threshold=0)`
- `train(model, optimizer, train_dataset, val_dataset=None, args=TrainingArgs(), loss=default_loss, iterate_batches=iterate_batches, training_callback=None)`

The important train-step pattern is:

```python
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map

if mx.metal.is_available():
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

loss_value_and_grad = nn.value_and_grad(model, loss_fn)
state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(batch, prev_grad, do_update):
    (loss, ntoks), grad = loss_value_and_grad(model, batch)

    if prev_grad is not None:
        grad = tree_map(lambda x, y: x + y, grad, prev_grad)

    if do_update:
        grad = average_gradients(grad)
        grad = tree_map(lambda x: x / grad_accum_steps, grad)
        optimizer.update(model, grad)
        grad = None

    return loss, ntoks, grad
```

cppmega should keep this shape but make the batch a simple stable pytree of MLX
arrays. Avoid feeding custom Python objects into the compiled function until
explicitly tested. Normalize `LMTokenBatch` to arrays before `step(...)`.

Recommended cppmega training-loop skeleton:

```python
def train_pretrain(model, optimizer, batches, *, grad_accum_steps, save_every):
    mx.random.seed(seed)
    model.train()

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    state = [model.state, optimizer.state, mx.random.state]
    loss_and_grad = nn.value_and_grad(model, cppmega_loss)
    grad_accum = None

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (loss, ntoks), grad = loss_and_grad(model, batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)
        if do_update:
            grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None
        return loss, ntoks, grad

    for step_idx, batch in enumerate(batches, start=1):
        loss, ntoks, grad_accum = step(
            batch,
            grad_accum,
            step_idx % grad_accum_steps == 0,
        )
        mx.eval(state, loss, ntoks, grad_accum)

        if step_idx % save_every == 0:
            save_full_pretrain_checkpoint(model, optimizer, step_idx)
```

Key constraints:

- Compile once and reuse; do not create compiled functions inside the loop.
- Keep batch keys, ranks, dtypes, and shapes stable to avoid compile churn.
- Use length buckets or fixed sequence lengths for pretraining throughput.
- Put validation, logging, checkpoint serialization, and Python metrics outside
  the compiled function.
- Use `mx.eval(state, metrics, grad_accum)` before timing or saving.
- Add activation checkpointing at block granularity only after baseline parity is
  stable.

## Loss And Batch Contract

MLX-LM `default_loss` assumes:

- `batch` is a dense int token matrix shaped `(B, S)`
- `inputs = batch[:, :-1]`
- `targets = batch[:, 1:]`
- `lengths` is shaped `(B, 2)` and stores `(offset, length)`
- the mask is built with target positions `1..S-1`
- logits come from `model(inputs)`
- token loss is `nn.losses.cross_entropy(logits, targets)`

This is fine for basic next-token SFT/pretraining but too narrow for cppmega:

- cppmega batches already include `attention_mask` and structure side channels in
  `LMTokenBatch`
- future batches may include ngram, syntax, document, packed-boundary, or
  recurrence side channels
- the MLX-LM mask is prompt-offset oriented, not a general packed-document mask
- MLX-LM loss has no place for cppmega auxiliary losses

cppmega should keep a custom loss with this interface:

```python
def cppmega_loss(model, batch):
    tokens = batch["tokens"]
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    mask = batch.get("attention_mask")
    target_mask = None if mask is None else mask[:, 1:].astype(mx.float32)

    logits = model(
        inputs,
        structure_ids=batch.get("structure_ids", None),
        dep_levels=batch.get("dep_levels", None),
        ast_depth_ids=batch.get("ast_depth_ids", None),
        sibling_index_ids=batch.get("sibling_index_ids", None),
        node_type_ids=batch.get("node_type_ids", None),
    )
    return next_token_loss(logits, targets, target_mask)
```

For compilation, prefer `batch` as a fixed-key dict or tuple where absent optional
features are represented by zero-size or sentinel arrays rather than changing the
input structure from step to step.

## Data And Batching Patterns

MLX-LM data classes:

- `TextDataset`: reads `{"text": ...}`, tokenizes, appends EOS if missing
- `ChatDataset`: reads `{"messages": [...]}` and uses `tokenizer.apply_chat_template(...)`
- `CompletionsDataset`: wraps prompt/completion into chat messages
- `ConcatenatedDataset`: multiplexes several datasets
- `CacheDataset`: lazily caches processed samples
- `load_local_dataset`: reads `train.jsonl`, `valid.jsonl`, `test.jsonl`
- `load_hf_dataset` and `load_custom_hf_dataset`: use `datasets.load_dataset`

`iterate_batches` behavior:

- sorts indices by a length function
- requires `len(dataset) >= batch_size`
- in distributed mode, requires `batch_size % world_size == 0`
- creates per-rank slices from the global batch
- shuffles batch order with NumPy
- truncates samples longer than `max_seq_length`
- pads to `1 + nearest_multiple_of_32(max_length)` capped by `max_seq_length`
- materializes an `np.int32` zero-padded array and converts it with `mx.array`
- yields `(batch, lengths)` where `batch` is `(local_batch, padded_seq)`

Use only the simple ideas:

- local JSONL/HF dataset wrappers are useful for finetuning smoke tests
- `CacheDataset` is useful for small tokenization-once datasets
- pad-to-32 bucketing is useful for compile/cache reuse
- per-rank slicing is a simple distributed pattern if needed later

Do not use `iterate_batches` as the cppmega pretraining data path:

- it tokenizes sample records, not pre-tokenized corpus shards
- it pads individual samples rather than packing token streams
- it has no document-boundary, structure, or recurrence side-channel contract
- it does not preserve Megatron-style `.bin/.idx` semantics
- its SFT prompt-offset path is not enough for packed pretraining masks

## Adapter Boundary

`cppmega_mlx/training/mlx_lm_adapter.py` is intentionally a probe and
conversion boundary, not a trainer bridge. It may convert `LMTokenBatch` or a
plain `(B, S)` token matrix into the installed MLX-LM `default_loss` argument
shape: dense `int32` tokens plus `int32` `lengths` rows shaped `(offset,
length)`. The adapter drops nothing silently into MLX-LM training because full
trainer integration is fail-closed via `require_supported_mlx_lm_trainer_integration`.

The unsupported boundary is deliberate for the current installed trainer:
`mlx_lm.tuner.trainer.default_loss` accepts only `(model, batch, lengths)`,
`iterate_batches` yields token arrays from tokenizer-style datasets, and
`train` saves adapter weights rather than cppmega full-pretraining checkpoints.
cppmega side channels (`attention_mask`, structure fields, route metadata, and
future packed/recurrence channels) must stay on the repo-local trainer until a
new integration surface proves those fields are preserved end to end.

Recommended cppmega data ladder:

1. Keep current synthetic `LMTokenBatch` smoke tests.
2. Add a pre-tokenized parquet/npz/safetensors token-batch reader with fixed
   shapes.
3. Add optional structure side-channel arrays with the same `(B, S)` shape.
4. Add document-boundary masks before packed streams.
5. Add Megatron `.bin/.idx` compatibility only after the MLX token-batch path is
   stable.

## Tokenizer Handling

MLX-LM loading path:

- `mlx_lm.utils.load(path_or_hf_repo, tokenizer_config=None, model_config=None, adapter_path=None, lazy=False, return_config=False, revision=None)`
- `_download(...)` fetches `*.json`, `model*.safetensors`, `*.py`, tokenizer model files, `*.txt`, `*.jsonl`, and `*.jinja`
- `load_model(...)` instantiates an MLX-LM model class from `config.json`
- `load_tokenizer(...)` downloads tokenizer artifacts and calls `tokenizer_utils.load(...)`
- `tokenizer_utils.load(...)` calls `AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra)`
- the returned `TokenizerWrapper` proxies the HF tokenizer and adds streaming
  detokenizer helpers plus `eos_token_ids`
- `mlx_lm.utils.save(...)` calls `tokenizer.save_pretrained(dst_path)`

cppmega recommendations:

- Keep tokenization outside compiled training steps.
- For pretraining, consume token IDs; do not call `apply_chat_template`.
- Save tokenizer artifacts beside full-model checkpoints if the run owns a
  tokenizer.
- Pass explicit tokenizer config. Use `trust_remote_code=True` only for trusted
  local/model repos, not as a hidden default.
- Prefer local tokenizer paths when streaming detokenization speed matters;
  MLX-LM detects tokenizer decoder type from local `tokenizer.json`.
- Preserve `eos_token_id` from `generation_config.json` if importing an HF/MLX-LM
  model, because MLX-LM threads that into `TokenizerWrapper.eos_token_ids`.
- Validate `vocab_size`, BOS/EOS/PAD IDs, and added vocab before comparing
  cppmega CUDA/Megatron data with MLX batches.

For local pretraining from existing token IDs, the tokenizer should be metadata,
not a runtime dependency of every batch.

## Full-Model Safetensors Save/Load

MLX core APIs:

- `mx.save_safetensors(file, arrays, metadata=None)`
- `mx.load(file, return_metadata=False)`
- `model.load_weights(path_or_items, strict=True)`
- `tree_flatten(model.parameters())`

MLX-LM full-model save path:

- `mlx_lm.utils.MAX_FILE_SIZE_GB = 5`
- `make_shards(weights, max_file_size_gb=5)` splits a weight dict by `nbytes`
- `save_model(save_path, model, donate_model=False)` writes:
  - `model.safetensors` for one shard, or
  - `model-00001-of-000NN.safetensors` for many shards
  - `model.safetensors.index.json`
- each shard is written with `mx.save_safetensors(..., metadata={"format": "mlx"})`
- the index stores `metadata.total_size`, `metadata.total_parameters`, and sorted
  `weight_map`
- `load_model(model_path, lazy=False, strict=True, model_config=None, get_model_classes=...)`
  loads all `model*.safetensors`, instantiates a model from `config.json`, calls
  optional `sanitize`, applies quantization transforms if configured, then
  `model.load_weights(list(weights.items()), strict=strict)`

Current cppmega baseline:

- `cppmega_mlx/training/checkpoint.py` writes one `model.safetensors`
- it writes `metadata.json` sidecar with `format=cppmega_mlx_checkpoint_v1`
- it can write and reload optional `optimizer.safetensors`
- it can serialize `CompiledPretrainingStep` state, including step cursor,
  trained tokens, pending microbatch count, and a gradient accumulator sidecar
  for exact mid-accumulation resume
- it records package versions, tokenizer/vocab contract fields, and model config
- it does not shard large checkpoints
- it does not save RNG state as an independent tensor payload yet

Required cppmega pretraining checkpoint format for larger runs:

- full `model.parameters()`, not `model.trainable_parameters()`
- sharded safetensors for large runs
- `model.safetensors.index.json` or compatible weight map
- `config.json` containing cppmega model config
- tokenizer artifacts if a tokenizer is part of the run
- optimizer state safetensors
- RNG state
- trainer state JSON with iteration, consumed tokens, scheduler state, gradient
  accumulation state, data cursor/shard position, package versions, and device
  info

Recommended layout:

```text
checkpoint-000100/
  config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors.index.json
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  optimizer.safetensors
  rng.safetensors
  trainer_state.json
```

Minimal full-model save helper:

```python
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

def save_full_model(path, model, *, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(
        str(path / "model.safetensors"),
        weights,
        metadata={"format": "mlx"},
    )
    (path / "trainer_state.json").write_text(json.dumps(metadata, indent=2) + "\n")
```

For serious runs, replace the single-file write with MLX-LM-style sharding before
large checkpoints exceed practical file or buffer sizes.

## Adapter And LoRA Pitfalls

MLX-LM LoRA flow:

- `mlx_lm.lora.train_model` calls `model.freeze()`
- `fine_tune_type == "full"` unfreezes only the last `num_layers` layers
- `fine_tune_type in ["lora", "dora"]` mutates selected modules into LoRA/DoRA modules
- `linear_to_lora_layers(model, num_layers, config, use_dora=False)` assumes a
  `model.layers` stack and module names/keys
- supported conversion targets include `nn.Linear`, `nn.QuantizedLinear`,
  `SwitchLinear`, `QuantizedSwitchLinear`, `nn.Embedding`, and
  `nn.QuantizedEmbedding`
- `load_adapters(model, adapter_path)` expects `adapter_config.json` plus
  `adapters.safetensors`, reconstructs LoRA layers, then loads weights with
  `strict=False`
- trainer checkpointing saves `dict(tree_flatten(model.trainable_parameters()))`
  to `adapters.safetensors` and numbered `*_adapters.safetensors`

Pitfalls for cppmega:

- adapter checkpoints are not full pretraining checkpoints
- `fine_tune_type="full"` in MLX-LM still follows the adapter-file path and may
  save only trainable parameters
- `model.freeze()` is wrong for scratch pretraining unless every parameter is
  explicitly unfrozen
- custom cppmega modules such as M2RNN, Mamba variants, ngram/structure
  embeddings, and future MoE routing may not have LoRA-compatible module names
  or `to_lora` methods
- `strict=False` can hide missing or stale adapter keys
- LoRA dropout and train/eval mode matter for reproducibility

Recommendation: do not mix LoRA/adapters into the first cppmega local pretraining
loop. Add adapter support later as a separate finetuning mode with explicit tests
that prove which modules are frozen, trainable, saved, loaded, and fused.

## What To Reuse In cppmega.mlx Now

Implemented reuse:

1. `CompiledPretrainingStep` uses the MLX-LM
   `state = [model.state, optimizer.state, mx.random.state]` compile pattern.
2. Eager `one_step_train` remains available for simple tests and debugging.
3. Checkpoint helpers save/load model weights, optimizer state, tokenizer/model
   metadata, and compiled-step resume metadata.
4. Fixed-shape token-batch iteration exists for NPZ, optional Parquet handoffs,
   and standalone Megatron `.bin/.idx` reads.
5. Tests compare eager and compiled tiny-model behavior, cover checkpoint
   resume, and reject compiled batch signature churn.
6. `scripts/train_hybrid_tiny.py` has local M4 Max receipts for eager/compiled
   Mamba3 and M2RNN train checkpoint/resume continuity, with resumed runs
   advancing from step `1` to step `2` and final trained tokens `6`.
7. `scripts/bench_matrix.py` has a local smoke baseline for `tiny`,
   `hybrid-m`, and `hybrid-r`; use it as regression evidence only, not as a
   GB10 comparison.

Immediate implementation targets:

1. Extend the existing Mamba3/M2RNN checkpoint/resume script receipts to
   validation loops and route/side-channel variants.
2. Add a dedicated RNG-state payload if deterministic stochastic layers enter
   the local training path.
3. Add route/side-channel regression tests for A/M/E/R blocks before increasing
   dimensions.
4. Archive the current `bench_matrix.py` M4 Max route baseline as JSON and grow
   it only after route metadata, compile settings, and shape keys stay stable.
5. Keep sharded safetensors deferred until single-file checkpoint size becomes
   a measured blocker.

Keep out of the first port:

- chat/completion dataset formats
- LoRA/DoRA conversion
- HF Hub upload
- `mlx_lm.utils.sharded_load` distributed inference path
- custom Metal kernels
- large-model quantization conversion

## Optional Adapter Probe

The local compatibility seam is `cppmega_mlx.training.mlx_lm_adapter`. It is a
probe and batch-shape adapter, not a training-loop dependency:

- `describe_mlx_lm_trainer_apis()` lazily imports `mlx_lm.tuner.trainer` and
  reports the installed signatures for `TrainingArgs`, `default_loss`,
  `iterate_batches`, `evaluate`, and `train`.
- if MLX-LM is absent or the trainer module moves, the probe returns
  `available=False`, `missing_apis`, and an error string instead of raising at
  import time.
- `as_mlx_lm_token_mapping(batch, offset=0)` converts `LMTokenBatch`, a batch
  mapping, or a dense `mx.array` into `{"tokens": tokens, "lengths": lengths}`.
- `as_mlx_lm_loss_args(batch, offset=0)` returns `(tokens, lengths)` for local
  experiments against MLX-LM's `default_loss`.

The exported mapping intentionally drops cppmega side channels such as
`attention_mask`, `structure_ids`, and `dep_levels`. That keeps interop probes
honest: a plain token-only MLX-LM path can be checked without implying that
MLX-LM owns cppmega's packed-mask, structure, recurrence, or auxiliary-loss
contract.

## Verification Checklist For The Implementation Lane

Before claiming the MLX pretraining loop is usable:

- eager and compiled tiny-model losses match within tolerance
- one optimizer update changes the same parameters in eager and compiled modes
- full checkpoint reload restores model parameters exactly
- optimizer state reload resumes the same next update
- RNG reload preserves deterministic dropout/sample behavior where applicable
- token/sec is measured after warmup, not on first compile
- peak memory is reported with `mx.get_peak_memory()`
- data batches keep stable keys, shapes, and dtypes
- full checkpoint uses `model.parameters()`, not `trainable_parameters()`
- adapter files are not used for pretraining resume

This keeps MLX-LM's proven Apple Silicon execution patterns while avoiding its
SFT/adapter-specific assumptions in cppmega's local pretraining path.
