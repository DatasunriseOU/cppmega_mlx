# Graph-Supervised DSA Indexer

This is the code-routing contract for cppmega's C++ graph-aware sparse
attention path.  The key rule is that compiler/indexer structure is used before
top-k selection, not as an after-the-fact annotation.

## Data Path

1. Parquet / Megatron sidecars carry token and graph structure:
   `token_chunk_*`, `token_call_edges`, `token_type_edges`, and future
   `def_use`/diagnostic routes.
2. `MegatronIndexedDataset.graph_route_packet_for_sample()` reconstructs a
   `GraphPacket` for one training window.
3. `code_graph_routes.build_attention_bias()` turns graph edges into a fixed
   block prior:

   ```text
   S_graph[t, s] = sum_r alpha_r * A_r[t, s]
   ```

   where `A_r` is the block adjacency for relation `r` (`call`, `type`, ...).
4. The DSA lightning indexer uses the graph prior before sparse top-k:

   ```text
   I_final[b, t, s] = I_neural[b, t, s] + beta * S_graph[b, t, s]
   selected = topk(I_final)
   ```

5. Sparse MLA attends only to the selected KV blocks plus forced local/sink
   blocks.

## Training Objective

The indexer is trained with both neural and compiler-grade teachers:

```text
L_indexer =
    KL(dense_attention_blocks || softmax(I_final))
  + lambda_bce * BCE(I_final, graph_edge_targets)
  + lambda_cov * coverage_hinge(I_final, graph_edge_targets, topk)
  + lambda_ctr * block_contrastive(...)
```

`graph_edge_targets` is built from the same graph candidates that produce
`S_graph`.  This keeps the training target aligned with the inference-time
selection rule.

## Implemented MLX Surfaces

- `cppmega_mlx.nn.code_graph_routes`
  - `GraphRouteConfig`
  - `CodeGraphRouter`
  - `build_attention_bias`
  - `build_block_candidates`
- `cppmega_mlx.nn.sparse_mla`
  - `lightning_indexer_scores(..., block_bias, beta)`
  - `indexer_topk_indices`
  - `graph_indexed_attention_reference`
- `cppmega_mlx.training.indexer_losses`
  - `apply_graph_indexer_bias`
  - `select_graph_biased_topk`
  - `total_indexer_loss`
  - `recall_at_k`
  - `edge_targets_from_candidates`

## Inference Contract

Inference uses the same score as training:

```text
I_final = I_neural + beta * S_graph
```

The graph prior can bias valid candidates but cannot unmask forbidden entries.
Masked scores stay at `-inf`.

## Megatron Contract

The `.bin/.idx` token stream alone is not enough for this model.  The training
prefix must be accompanied by sidecar metadata for:

- token-aligned structure/AST/semantic channels;
- document/window ids;
- graph route sidecars for chunk metadata and edge pairs.

The Megatron/CUDA path must consume those sidecars and apply the same
pre-top-k bias to the DSA indexer logits before fused sparse attention.  A
token-only `.bin/.idx` prefix is acceptable only for a plain LM baseline, not
for the graph-supervised C++ world model.

## Current Limitations

- MLX uses dense reference tensors for the graph prior and top-k tests; this is
  correctness-first, not the final fast kernel.
- Def-use and diagnostic routes should join the same `GraphPacket`/bias path
  when their sidecars are available.
- Megatron fused kernels still need the same bias hook at the CUDA/TE path.
