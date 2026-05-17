# Verbatim port of deepseek-ai/TileKernels/tile_kernels/torch/engram.py with
# the smallest possible edits to run on MLX instead of PyTorch.
#
# Upstream source:
#   /Users/dave/sources/TileKernels/tile_kernels/torch/engram.py
# Upstream license: MIT (deepseek-ai/TileKernels)
#
# Edits made (kept minimal so a diff against the upstream file is short):
#   - import torch                                -> import mlx.core as mx
#   - torch.Tensor annotations                    -> mx.array
#   - x.view(-1)                                  -> x.reshape(-1)
#   - x.unsqueeze(0|1|-1)                         -> x[None] / x[..., None]
#   - .to(torch.int32) / int64                    -> .astype(mx.int32) / int64
#   - .clone()                                    -> dropped (functional)
#   - hashes.bitwise_xor_(other)                  -> hashes = mx.bitwise_xor(hashes, other)
#     (MLX is functional, no in-place ops)
#   - x.cumsum(0, dtype=torch.int32)              -> mx.cumsum(x, axis=0).astype(mx.int32)
#   - torch.cat([...], dim=...)                   -> mx.concatenate([...], axis=...)
#   - torch.stack([...], dim=...)                 -> mx.stack([...], axis=...)
#   - torch.zeros(1, dtype=..., device=...)       -> mx.zeros((1,), dtype=...)
#   - x.float() / x.bfloat16()                    -> x.astype(mx.float32|bfloat16)
#   - torch.rsqrt / x.pow(2) / x.sigmoid          -> mx.rsqrt / mx.square / mx.sigmoid
#   - torch.einsum('...d,...d->...', a, b)        -> mx.einsum('...d,...d->...', a, b)
#   - dot.abs().clamp_min(v).sqrt() * dot.sign()  -> mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), v))
# Algorithm, signatures, dtypes, return shapes unchanged.

import mlx.core as mx


def make_offsets(vocab_sizes: mx.array) -> mx.array:
    """Compute exclusive prefix-sum offsets from vocab_sizes.

    Args:
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.

    Returns:
        Offsets of shape (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = vocab_sizes.shape[0]
    offsets_list = []
    for layer_idx in range(num_ngram_layers):
        flat = vocab_sizes[layer_idx].reshape(-1)
        prefix = mx.concatenate(
            [
                mx.zeros((1,), dtype=mx.int32),
                mx.cumsum(flat[:-1], axis=0).astype(mx.int32),
            ]
        )
        offsets_list.append(prefix)
    return mx.stack(offsets_list, axis=0)


def engram_hash_ref(
    ngram_token_ids: mx.array,
    multipliers: mx.array,
    vocab_sizes: mx.array,
    offsets: mx.array,
) -> mx.array:
    """Pure PyTorch reference implementation of engram hash.

    Args:
        ngram_token_ids: N-gram token IDs of shape (num_tokens, max_ngram_size), int32.
        multipliers: Per-layer hash multipliers of shape (num_ngram_layers, max_ngram_size), int64.
        vocab_sizes: Per-layer per-ngram embedding table sizes of shape
            (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
        offsets: Per-layer embedding table offsets of shape
            (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.

    Returns:
        Embedding indices of shape (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_ngram_layers = multipliers.shape[0]
    max_ngram_size = multipliers.shape[1]

    prod = ngram_token_ids.astype(mx.int64)[None] * multipliers[:, None]

    ans = [[] for _ in range(num_ngram_layers)]
    hashes = prod[:, :, 0]
    for i in range(1, max_ngram_size):
        hashes = mx.bitwise_xor(hashes, prod[:, :, i])
        for layer_idx in range(num_ngram_layers):
            ans[layer_idx].append(
                (
                    hashes[layer_idx][..., None]
                    % vocab_sizes[layer_idx, i - 1].astype(mx.int64)[None]
                ).astype(mx.int32)
            )

    for layer_idx in range(num_ngram_layers):
        ans[layer_idx] = mx.concatenate(ans[layer_idx], axis=-1)

    output = mx.stack(ans, axis=0)
    return output + offsets[:, None]


def engram_gate_ref(
    hidden_states: mx.array,
    k: mx.array,
    v: mx.array,
    weight_hidden: mx.array,
    weight_embed: mx.array,
    clamp_value: float,
    eps: float,
    save_for_backward: bool = False,
):
    """Pure PyTorch reference implementation of engram gate (vectorized, supports autograd).

    Computes: output = x + sigmoid(signed_sqrt(dot(RMSNorm(x, wh), RMSNorm(k, we)) * scalar)) * v

    Args:
        hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Key embeddings of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Value embeddings of shape (num_tokens, hidden_size), bfloat16.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        eps: Epsilon for RMSNorm numerical stability.
        save_for_backward: If True, also return (dot, gate_score, rstd_x, rstd_k).

    Returns:
        If save_for_backward is False: output tensor of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        If save_for_backward is True: tuple of (output, dot, gate_score, rstd_x, rstd_k).
    """
    hidden_size = hidden_states.shape[-1]
    scalar = hidden_size**-0.5

    x = hidden_states.astype(mx.float32)
    k_f = k.astype(mx.float32)
    wh = weight_hidden.astype(mx.float32)[None]
    we = weight_embed.astype(mx.float32)[None]

    # RMSNorm
    rstd_x = mx.rsqrt(mx.mean(mx.square(x), axis=-1) + eps)
    rstd_k = mx.rsqrt(mx.mean(mx.square(k_f), axis=-1) + eps)

    # Dot -> sqrt-gate -> sigmoid
    # raw_dot is the unnormalized sum(x * wh * k * we), matching the kernel's dot_out
    raw_dot = mx.einsum('...d,...d->...', x * wh, k_f * we)
    dot = raw_dot * rstd_x * rstd_k * scalar
    signed_sqrt = mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), clamp_value))
    gate_score = mx.sigmoid(signed_sqrt)

    output = x + gate_score[..., None] * v[..., None, :]
    output = output.astype(mx.bfloat16)

    if save_for_backward:
        return output, raw_dot, gate_score, rstd_x, rstd_k
    return output
