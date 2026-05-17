# Verbatim port of fla-org/flash-linear-attention/fla/ops/kda/naive.py with
# the smallest possible edits to run on MLX instead of PyTorch.
#
# Upstream source:
#   /Volumes/external/sources/rent_kernels/flash-linear-attention/
#   fla/ops/kda/naive.py
# Upstream license: MIT (Songlin Yang, Yu Zhang, Zhiyuan Li — 2023-2026)
#
# Edits made (kept minimal so a diff against the upstream file is short):
#   - import torch                     -> import mlx.core as mx
#   - from einops import rearrange     -> removed (unused in recurrent fn)
#   - torch.Tensor annotations         -> mx.array
#   - .to(torch.float)                 -> .astype(mx.float32)
#   - .to(dtype) / .to(q) (move-to)    -> .astype(dtype)
#   - x.repeat_interleave(G, dim=2)    -> mx.repeat(x, G, axis=2)
#   - k.new_zeros((B, HV, K, V)).to(q) -> mx.zeros((B, HV, K, V), dtype=q.dtype)
#   - torch.zeros_like(v)              -> mx.zeros_like(v)
#   - o[:, i] = ...                    -> append to list, mx.stack at end
#   - torch.einsum('b h k, b h v -> b h k v', a, b)
#                                      -> a[..., :, None] * b[..., None, :]
#   - torch.einsum('b h k, b h k v -> b h v', q_i, S)
#                                      -> mx.sum(q_i[..., :, None] * S, axis=-2)
# Algorithm, variable names, loop body, final-state semantics unchanged.
#
# Chunk variant (naive_chunk_kda) is NOT ported here yet — it relies on the
# same in-place attn[..., i, :i] mutation pattern as the GDN chunk variant,
# which needs a deeper rewrite that is out of scope for this minimal-edit port.

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import mlx.core as mx


def naive_recurrent_kda(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    scale: float | None = None,
    initial_state: mx.array | None = None,
    output_final_state: bool = False,
):
    r"""
    Args:
        q (mx.array):
            Queries of shape ``[B, T, H, K]``.
        k (mx.array):
            Keys of shape ``[B, T, H, K]``.
        v (mx.array):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (mx.array):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (mx.array):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[mx.array]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.astype(mx.float32), [q, k, v, g, beta])
    q = mx.repeat(q, G, axis=2) * scale   # [B, T, HV, K]
    k = mx.repeat(k, G, axis=2)           # [B, T, HV, K]

    S = mx.zeros((B, HV, K, V), dtype=q.dtype)
    if initial_state is not None:
        S = S + initial_state
    o_list = []
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * mx.exp(g_i)[..., None]
        # einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - ...)
        inner = v_i - (k_i[..., None] * S).sum(-2)
        beta_k = b_i[..., None] * k_i           # [B, HV, K]
        S = S + beta_k[..., :, None] * inner[..., None, :]
        # einsum('b h k, b h k v -> b h v', q_i, S)
        o_list.append(mx.sum(q_i[..., :, None] * S, axis=-2))
    if not output_final_state:
        S = None
    o = mx.stack(o_list, axis=1)
    return o.astype(dtype), S
