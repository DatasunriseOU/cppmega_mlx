# Checkpointing

Local MLX pretraining checkpoints use MLX safetensors files with MLX-LM-compatible
weight naming where possible. The on-disk weight files are written through
MLX's mx.save_safetensors(..., metadata={"format": "mlx"}) API, which accepts
an optional string metadata map for .safetensors output.

## Layout

Directory checkpoints contain:

- model.safetensors: full model.parameters() flattened with MLX names and
  saved with mx.save_safetensors(..., metadata={"format": "mlx"}).
- optimizer.safetensors: optional flattened optimizer state when
  save_checkpoint(..., optimizer=...) is used.
- gradient_accumulator.safetensors: optional flattened gradient accumulator
  when checkpointing mid gradient-accumulation window.
- metadata.json: checkpoint manifest and training metadata.

Passing a direct .safetensors path is still supported for the older simple
model-only save/load path. Optimizer state is only saved for directory
checkpoints because it needs a second safetensors file.

The current implementation is deliberately single-file for model weights:
model.safetensors must be present. MLX-LM's export format can split weights
into model-00001-of-000NN.safetensors files with
model.safetensors.index.json; cppmega-mlx refuses to load that layout until a
real sharded pretraining resume path exists. Current MLX-LM save_weights
writes the same single model.safetensors name for one shard, switches to
model-00001-of-000NN.safetensors names when multiple shards are needed, and
always emits model.safetensors.index.json with a weight_map; cppmega-mlx
intentionally does not consume that index yet.

## Manifest

metadata.json keeps the existing cppmega_mlx_checkpoint_v1 format marker and
records:

- step from caller metadata when available.
- model_config when the model exposes config.to_dict().
- model tensor count and relative model weight filename.
- optimizer state presence, relative filename, tensor count, and tensor names.
- installed package versions for cppmega-mlx, mlx, mlx-lm, safetensors,
  and numpy.
- tokenizer/vocab contract fields when available, including model-derived
  vocab_size, max_seq_length, and structure_vocab_size, plus caller-provided
  tokenizer_path, tokenizer_name, bos_token_id, eos_token_id, and pad_token_id.
- training_state when a CompiledPretrainingStep is passed, including the
  Python-side step cursor, trained-token cursor, compile flag, gradient
  accumulation width, pending microbatch count, and optional gradient
  accumulator file metadata.
- evaluation when supplied by the trainer, as validation receipt/provenance
  metadata. The helper validates the current local receipt fields
  (requested_batches, planned_batches, evaluated_batches, and
  metrics.loss / metrics.ntokens / metrics.batches /
  metrics.tokens_per_second) plus MLX-LM callback-style iteration,
  val_loss, and val_time fields, but does not use them for resume.
- rng, currently {"mode": "not_saved"}, seed provenance
  {"mode": "seed", "seed": N, ...}, or a single-process snapshot
  {"mode": "snapshot", "snapshot": ...}. Directory checkpoints that save
  optimizer/training state default to snapshot mode unless caller metadata
  explicitly opts out with {"mode": "not_saved"}. Standalone serialized RNG
  state outside the rng object is not supported.
- sharding, currently {"mode": "single_file", "num_shards": 1,
  "weights": ["model.safetensors"], "index": null}. Multi-shard manifests and
  index payloads are not supported.

Caller metadata is merged into the manifest after the default contract is built,
so existing fields such as train_loss remain compatible.

## Resume Contract

Use load_checkpoint(model, path, optimizer=optimizer) for full continuation.
The loader restores model weights first, then reconstructs the optimizer state
from optimizer.safetensors. A missing optimizer file raises
FileNotFoundError when an optimizer is requested.

Metadata validation runs before model weights are loaded whenever a
metadata.json file is present. Unsupported RNG payloads or sharding requests
therefore fail closed before mutating the target model. Accepted RNG metadata is
limited to explicit opt-out, seed provenance, or a single-process snapshot
captured by cppmega_mlx.runtime.seed.capture_rng_state(). On load, snapshot
mode restores Python's global random state, NumPy's legacy global RNG state,
and MLX's mutable mx.random.state when the installed MLX build exposes it. If
a checkpoint snapshot says MLX RNG state is available but restore cannot replay
it, loading fails closed rather than silently continuing with a divergent random
stream.

Snapshot RNG support is a local single-process contract. It does not cover
distributed per-rank RNGs, independently created NumPy Generator instances,
DataLoader worker state, or Megatron tensor/pipeline/expert parallel RNG
streams.

M0.7 acceptance is currently scoped to local TinyLM/HybridTinyLM checkpoint
mechanics and the subprocess HybridTinyLM NPZ resume regression. That receipt
proves a 37-step interrupted tiny run can reload and match the uninterrupted
100-step continuation suffix with optimizer state, training_state, RNG snapshot,
and batch cursor preserved. It is not full local_gb10_quarter acceptance:
M0.7 remains fail-closed until the resolved tokenizer/model-factory path, the
target GB10 parquet training lane, and the exact 100-step continuation check all
run on the M0 target.

Unsupported sharding requests also fail closed. The manifest may only describe
the single local weight file. Fields such as weight_map, shards,
index_file, or max_file_size_gb are rejected rather than being silently
ignored, because accepting them would imply MLX-LM-style sharded resume support
that this helper does not yet implement.

Megatron pipeline-parallel, tensor-parallel, expert-parallel, distributed
optimizer, and MTP replica semantics are explicit non-goals for this local
single-process checkpoint format. In the source Megatron codebase, cppmega
custom embedding tensors have distributed-checkpoint ownership rules such as
replica_id[0] == 0 for the PP first-stage main copy and replica_id[0] == 1
for the MTP-stage copy. This MLX helper does not encode or replay those
pipeline/MTP replica assignments; metadata keys such as parallel_state,
megatron_parallel_state, sharded_state_dict, replica_id, pre_process,
post_process, and mtp_process are rejected instead of being interpreted.

Pass training_step=stepper to save_checkpoint(...) and
load_checkpoint(...) when exact trainer continuation is required. The helper
then saves/restores the CompiledPretrainingStep.state_dict() payload. At an
optimizer-update boundary this is JSON-only metadata. In the middle of a
gradient-accumulation window it also writes gradient_accumulator.safetensors
and refuses to resume unless that accumulator can be restored.

## Tiny NPZ CLI

scripts/train_hybrid_tiny.py exposes the checkpoint helper through the local
HybridTinyLM smoke trainer:

- --checkpoint-save-interval N --checkpoint-dir DIR writes interval
  checkpoints as DIR/checkpoint-000001/, DIR/checkpoint-000002/, and so on,
  using the global resumed step.
- --checkpoint-path PATH writes a final full-training checkpoint after the
  requested local steps finish. When --eval-batches is enabled, the final
  checkpoint manifest includes the same evaluation receipt emitted by the
  script JSON payload.
- --resume-from PATH restores model weights and optimizer state, restores the
  CompiledPretrainingStep cursor from training_state when present, falls
  back to legacy top-level step / trained_tokens metadata otherwise, and
  reconstructs the NPZ dataset cursor from batch_cursor.global_batch_offset.

The manifest records training_config, dataset, trained_tokens,
batch_cursor, and a normalized resume_cursor alongside the lower-level
model/optimizer fields. resume_cursor mirrors the script JSON receipt with
the restored step, trained-token count, and batch cursor needed to resume from
the next fixed-shape batch. This mirrors the MLX-LM trainer's steps_per_save
cadence while saving full pretraining state instead of LoRA adapter-only
weights.

When the dataset path points at ignored local samples such as
data/parquet_samples/gb10/..., the checkpoint manifest records that local path
as provenance only. The checkpoint is still a local smoke artifact unless the
same dataset file and metadata are staged on the target host; do not treat it as
a portable GB10/M4 parity receipt by itself.

Regression coverage exercises both Mamba3-only (--pattern M) and M2RNN-only
(--pattern R) HybridTinyLM routes in eager and compiled modes. The script
smokes save/resume/eval for both M and R with full structure side channels
(structure_ids, dep_levels, ast_depth_ids, sibling_index_ids, and
node_type_ids) and asserts the JSON payload plus final manifest expose
resume_cursor and validation metadata. It also covers a mixed AEMR
HybridTinyLM checkpoint with custom DSA-routed attention, MoE, Mamba3, and
M2RNN block parameters plus AdamW state, then verifies the next eager train step
matches an uninterrupted run. The resume path restores the same Python-side
CompiledPretrainingStep cursor for both compiled modes, and the next
checkpoint records the continued global batch cursor rather than restarting the
dataset iterator.

The synthetic dry-run path writes the same full structure side-channel set plus
attention_mask, so --dry-run-json exercises the model-threaded side-channel
contract even when no local NPZ or parquet sample is supplied.

The current implementation intentionally keeps single-shard model weights. Large
runs should add MLX-LM-style sharding (model-00001-of-000NN.safetensors plus
model.safetensors.index.json) before checkpoints approach practical file-size
limits. Until that lands, an index file next to a missing model.safetensors
is treated as an unsupported checkpoint layout, not as a partial resume target.

External checkpoint-format sources checked for this boundary:

- MLX mlx.core.save_safetensors docs:
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.save_safetensors.html
- MLX-LM mlx_lm/utils.py save/load conventions:
  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py
