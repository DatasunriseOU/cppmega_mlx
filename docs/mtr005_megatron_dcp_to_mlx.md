# MTR-005: Megatron torch_dist to MLX

`scripts/convert_megatron_dense500m_torchdist_to_mlx.py` converts the H200
`h200_cpp_world_mini` Megatron distributed checkpoint into evaluator-compatible
MLX safetensors and `model.json`.

## Supported source contract

The converter reads `.metadata`, `metadata.json`, and `common.pt` directly. It
fails unless the checkpoint uses `torch_dist` v1 and the dense AF route pattern
(`*-` repeated 24 times), tied embeddings, RMSNorm, SwiGLU, RoPE, and grouped
QKV with 20 query heads, 4 KV heads, and head dimension 64. MLA, DSA, MoE, MTP,
untied output weights, biases, unknown model tensors, shape mismatches, and
unsupported dtypes are blocking errors.

`optimizer.*` tensors are explicitly excluded. They are counted in the
manifest but are never loaded into the inference checkpoint.

## Conversion and parity gate

```bash
python3 scripts/convert_megatron_dense500m_torchdist_to_mlx.py \
  --checkpoint /path/to/stage_or_iter \
  --output /path/to/model.safetensors \
  --seq-len 1024
```

The output is staged and published only after a fresh MLX model reloads the
emitted safetensors and matches an independent NumPy forward over the raw DCP
tensor names. The reference implements Megatron's alternating attention/MLP
layers, grouped QKV packing, split-half RoPE, graph-route bias, n-gram and core
structure embeddings, RMSNorm, SwiGLU, and the tied output projection.

The full 24-layer gate records max, mean, p99, and RMS logit error and requires
`atol=0.004`, `rtol=0.001`. It uses one document because the source Megatron
n-gram module has no document-boundary input; packed-document n-gram parity is
therefore explicitly not claimed.

`model.json` includes SHA-256 and byte size for every source artifact, the
validated source architecture, the complete source-to-target tensor map,
optimizer exclusion count, and an empty `unsupported_tensors` list on success.

## Atomic publication

Conversion writes to a staging directory next to the final output:

1. `weights.tmp.safetensors` is materialized and fsynced.
2. `model.json.tmp` is written with `publish.weights_sha256` and
   `publish.completion_marker`, then fsynced.
3. `os.replace` publishes the weights first, then `model.json` last.

The presence of `model.json` therefore implies the complete, matching pair.
`published_checkpoint_status(output)` checks the marker and recomputes the
weights SHA-256; it returns `complete: False` with a reason if either is wrong.

## RoPE-only position encoding

The source checkpoint uses RoPE, not a learned absolute position table. The
converter emits `rope_only: true` in `model.json` and the target
`DenseCppLM` is instantiated with `rope_only=True`, so no
`position_embedding.weight` is created or loaded. This keeps converted weights
strictly minimal and blocks accidental continued-training steps that would
reintroduce a learned position table.

## Graph-route parity

The converter maps n-gram hash and core structure embedding tensors, so the
target model can consume real graph sidecars (`block_bias`,
`edge_kind_bias`). The parity gate (`verify_emitted_dcp_logit_parity`) runs
an independent NumPy Megatron reference with the same graph-route biases and
compares logits against a freshly reloaded MLX model.

Note: packed-document n-gram parity is still not claimed because the source
Megatron n-gram module has no document-boundary input. Single-document graph
parity is the supported gate.

## Side-channel preservation

`model.json["runtime_requirements"]["side_channels"]` lists the n-gram and
structure side-channel tensors required by the emitted checkpoint. The
converter records exactly which source tensors were mapped and which were
excluded (optimizer tensors only, with a reason).

## Beyond dense GQA

`docs/status/mamba3_dsa_mlx_conversion_decision.md` records the decision to
extend conversion as a staged partial port in the order
mamba3 → DSA → MoE → MTP. Each stage will reuse the same parity procedure
(ATOL/RTOL gate, SHA-256 receipts, atomic publish) before the converter is
widened. The dense500m GQA path described here remains the canonical baseline.
