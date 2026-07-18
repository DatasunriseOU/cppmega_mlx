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
