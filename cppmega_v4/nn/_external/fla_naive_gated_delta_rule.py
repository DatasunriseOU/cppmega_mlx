# Verbatim port of fla-org/flash-linear-attention/fla/ops/gated_delta_rule/naive.py
# with the smallest possible edits to run on MLX instead of PyTorch.
#
# Upstream source:
#   /Volumes/external/sources/rent_kernels/flash-linear-attention/
#   fla/ops/gated_delta_rule/naive.py
# Upstream license: MIT (Songlin Yang, Yu Zhang, Zhiyuan Li — 2023-2026)
# Upstream copyright header retained below.
#
# Edits made (kept minimal so a diff against the upstream file is short):
#   - import torch                     -> import mlx.core as mx
#   - import torch.nn.functional as F  -> removed (only used in chunk variant)
#   - from einops import rearrange     -> removed (replaced by inline reshape)
#   - torch.Tensor type annotations    -> mx.array
#   - .transpose(1, 2).contiguous()    -> mx.transpose(x, (0, 2, 1, *rest))
#   - .to(torch.float32)               -> .astype(mx.float32)
#   - .to(v)                           -> .astype(v.dtype)
#   - .clone()                         -> dropped (MLX is functional, no aliasing)
#   - .unsqueeze(-1)                   -> [..., None]
#   - torch.zeros(...).to(v)           -> mx.zeros(shape, dtype=v.dtype)
#   - torch.einsum('bhd,bhdm->bhm',
#                  b_q, h)             -> mx.sum(b_q[..., :, None] * h, axis=-2)
#   - o[:, :, i] = ...                 -> append to list, mx.stack at end
#     (MLX is functional and cannot assign by index)
# Algorithm, variable names, loop body, and final-state semantics unchanged.
#
# Chunk variant (naive_chunk_gated_delta_rule) is NOT ported here yet —
# its in-place mutation of `attn[..., i, :i]` requires a deeper rewrite that
# is out of scope for this minimal-edit port.

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import mlx.core as mx


def naive_recurrent_gated_delta_rule(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    beta: mx.array,
    g: mx.array,
    scale: float = None,
    initial_state: mx.array = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    def _t12(x):  # ".transpose(1, 2).contiguous()" — supports rank 3 and 4.
        perm = (0, 2, 1) + tuple(range(3, x.ndim))
        return mx.transpose(x, perm).astype(mx.float32)
    q, k, v, beta, g = map(_t12, [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o_list = []
    h = mx.zeros((B, H, K, V), dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.astype(mx.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i]
        h = h * mx.exp(g[:, :, i])[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h + b_k[..., None] * b_v[..., None, :]
        # einsum('bhd,bhdm->bhm', b_q, h): contract d -> output (B,H,V)
        o_list.append(mx.sum(b_q[..., :, None] * h, axis=-2))

    if not output_final_state:
        h = None
    # stack list along the time axis (originally o[:, :, i] = ...) then
    # transpose back to [B, T, H, V] to match upstream return shape.
    o = mx.stack(o_list, axis=2)
    o = mx.transpose(o, (0, 2, 1, 3))
    return o, h
