# Checkpointing

Local MLX pretraining checkpoints use MLX safetensors files with MLX-LM-compatible
weight naming where possible.

## Layout

Directory checkpoints contain:

- `model.safetensors`: full `model.parameters()` flattened with MLX names and
  saved with `mx.save_safetensors(..., metadata={"format": "mlx"})`.
- `optimizer.safetensors`: optional flattened optimizer state when
  `save_checkpoint(..., optimizer=...)` is used.
- `gradient_accumulator.safetensors`: optional flattened gradient accumulator
  when checkpointing mid gradient-accumulation window.
- `metadata.json`: checkpoint manifest and training metadata.

Passing a direct `.safetensors` path is still supported for the older simple
model-only save/load path. Optimizer state is only saved for directory
checkpoints because it needs a second safetensors file.

The current implementation is deliberately single-file for model weights:
`model.safetensors` must be present. MLX-LM's export format can split weights
into `model-00001-of-000NN.safetensors` files with
`model.safetensors.index.json`; cppmega-mlx refuses to load that layout until a
real sharded pretraining resume path exists.

## Manifest

`metadata.json` keeps the existing `cppmega_mlx_checkpoint_v1` format marker and
records:

- `step` from caller metadata when available.
- `model_config` when the model exposes `config.to_dict()`.
- model tensor count and relative model weight filename.
- optimizer state presence, relative filename, tensor count, and tensor names.
- installed package versions for `cppmega-mlx`, `mlx`, `mlx-lm`, `safetensors`,
  and `numpy`.
- tokenizer/vocab contract fields when available, including model-derived
  `vocab_size`, `max_seq_length`, and `structure_vocab_size`.
- `training_state` when a `CompiledPretrainingStep` is passed, including the
  Python-side step cursor, trained-token cursor, compile flag, gradient
  accumulation width, pending microbatch count, and optional gradient
  accumulator file metadata.
- `rng`, currently either `{"mode": "not_saved"}` or seed provenance
  `{"mode": "seed", "seed": N, ...}`. Standalone serialized RNG state is not
  supported.
- `sharding`, currently `{"mode": "single_file", "num_shards": 1,
  "weights": ["model.safetensors"], "index": null}`. Multi-shard manifests and
  index payloads are not supported.

Caller metadata is merged into the manifest after the default contract is built,
so existing fields such as `train_loss` remain compatible.

## Resume Contract

Use `load_checkpoint(model, path, optimizer=optimizer)` for full continuation.
The loader restores model weights first, then reconstructs the optimizer state
from `optimizer.safetensors`. A missing optimizer file raises
`FileNotFoundError` when an optimizer is requested.

Metadata validation runs before model weights are loaded whenever a
`metadata.json` file is present. Unsupported RNG payloads or sharding requests
therefore fail closed before mutating the target model. The only accepted RNG
metadata is seed provenance; exact standalone MLX/NumPy/Python random-state
roundtrip remains intentionally unsupported until local training has a concrete
resume blocker that requires it.

Unsupported sharding requests also fail closed. The manifest may only describe
the single local weight file. Fields such as `weight_map`, `shards`,
`index_file`, or `max_file_size_gb` are rejected rather than being silently
ignored, because accepting them would imply MLX-LM-style sharded resume support
that this helper does not yet implement.

Pass `training_step=stepper` to `save_checkpoint(...)` and
`load_checkpoint(...)` when exact trainer continuation is required. The helper
then saves/restores the `CompiledPretrainingStep.state_dict()` payload. At an
optimizer-update boundary this is JSON-only metadata. In the middle of a
gradient-accumulation window it also writes `gradient_accumulator.safetensors`
and refuses to resume unless that accumulator can be restored.

## Tiny NPZ CLI

`scripts/train_hybrid_tiny.py` exposes the checkpoint helper through the local
HybridTinyLM smoke trainer:

- `--checkpoint-save-interval N --checkpoint-dir DIR` writes interval
  checkpoints as `DIR/checkpoint-000001/`, `DIR/checkpoint-000002/`, and so on,
  using the global resumed step.
- `--checkpoint-path PATH` writes a final full-training checkpoint after the
  requested local steps finish.
- `--resume-from PATH` restores model weights and optimizer state, restores the
  `CompiledPretrainingStep` cursor from `training_state` when present, falls
  back to legacy top-level `step` / `trained_tokens` metadata otherwise, and
  reconstructs the NPZ dataset cursor from `batch_cursor.global_batch_offset`.

The manifest records `training_config`, `dataset`, `trained_tokens`, and
`batch_cursor` alongside the lower-level model/optimizer fields. This mirrors
the MLX-LM trainer's `steps_per_save` cadence while saving full pretraining
state instead of LoRA adapter-only weights.

When the dataset path points at ignored local samples such as
`data/parquet_samples/gb10/...`, the checkpoint manifest records that local path
as provenance only. The checkpoint is still a local smoke artifact unless the
same dataset file and metadata are staged on the target host; do not treat it as
a portable GB10/M4 parity receipt by itself.

Regression coverage exercises both Mamba3-only (`--pattern M`) and M2RNN-only
(`--pattern R`) HybridTinyLM routes in eager and compiled modes. It also covers
a mixed `AEMR` HybridTinyLM checkpoint with custom DSA-routed attention, MoE,
Mamba3, and M2RNN block parameters plus AdamW state, then verifies the next
eager train step matches an uninterrupted run. The resume path restores the
same Python-side `CompiledPretrainingStep` cursor for both compiled modes, and
the next checkpoint records the continued global batch cursor rather than
restarting the dataset iterator.

The current implementation intentionally keeps single-shard model weights. Large
runs should add MLX-LM-style sharding (`model-00001-of-000NN.safetensors` plus
`model.safetensors.index.json`) before checkpoints approach practical file-size
limits. Until that lands, an index file next to a missing `model.safetensors`
is treated as an unsupported checkpoint layout, not as a partial resume target.
