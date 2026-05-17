# Verbatim port of deepseek-ai/TileKernels/tile_kernels/torch/mhc.py with the
# smallest possible edits to run on MLX instead of PyTorch.
#
# Upstream source:
#   /Users/dave/sources/TileKernels/tile_kernels/torch/mhc.py
# Upstream license: MIT (deepseek-ai/TileKernels)
#
# Edits made (kept minimal so a diff against the upstream file is short):
#   - import torch                     -> import mlx.core as mx
#   - torch.Tensor type annotations    -> mx.array
#   - x.softmax(-1)                    -> mx.softmax(x, axis=-1)
#   - x.sum(-2, keepdim=True)          -> mx.sum(x, axis=-2, keepdims=True)
#   - x.unsqueeze(-2)                  -> x[..., None, :] / mx.expand_dims(x, -2)
#   - x.expand(*shape)                 -> mx.broadcast_to(x, shape)
#   - .contiguous()                    -> dropped (MLX is functional)
#   - torch.sigmoid                    -> mx.sigmoid
#   - torch.cat([...])                 -> mx.concatenate([...])
#   - x.view(...)                      -> x.reshape(...)
#   - .bfloat16() / .float()           -> .astype(mx.bfloat16) / .astype(mx.float32)
#   - .square() / .rsqrt()             -> mx.square(x) / mx.rsqrt(x)
#   - torch.einsum(...)                -> mx.einsum(...)  (MLX 0.31+ has einsum)
# Algorithm, function names, signatures, and dtypes unchanged.

import mlx.core as mx


def expand_to_mhc_ref(hidden: mx.array, mhc_mult: int) -> mx.array:
    target_shape = (*hidden.shape[:-1], mhc_mult, hidden.shape[-1])
    return mx.broadcast_to(hidden[..., None, :], target_shape)


def sinkhorn_normalize_ref(x: mx.array, repeat: int = 10, eps: float = 1e-6) -> mx.array:
    x = mx.softmax(x, axis=-1) + eps
    x = x / (mx.sum(x, axis=-2, keepdims=True) + eps)
    for _ in range(repeat - 1):
        x = x / (mx.sum(x, axis=-1, keepdims=True) + eps)
        x = x / (mx.sum(x, axis=-2, keepdims=True) + eps)
    return x


def mhc_head_compute_mix_ref(
    input_mix: mx.array,
    mhc_scale: mx.array,
    mhc_base: mx.array,
    mhc_pre_eps: float,
) -> mx.array:
    mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
    return mx.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


def mhc_pre_split_mixes_ref(
    input_mixes: mx.array,
    mhc_scale: mx.array,
    mhc_base: mx.array,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    a, b = input_mixes.shape[:2]
    mhc_scale = mx.concatenate(
        [
            mx.broadcast_to(mhc_scale[0:1], (mhc_mult,)),
            mx.broadcast_to(mhc_scale[1:2], (mhc_mult,)),
            mx.broadcast_to(mhc_scale[2:3], (mhc_mult * mhc_mult,)),
        ],
    )
    input_mixes = input_mixes * mhc_scale + mhc_base

    pre_layer_mix = mx.sigmoid(input_mixes[:, :, :mhc_mult])[..., None] + mhc_pre_eps
    post_layer_mix = (
        mx.sigmoid(input_mixes[:, :, mhc_mult : 2 * mhc_mult]) * mhc_post_mult_value
    )[..., None]
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult :].reshape(a, b, mhc_mult, mhc_mult)

    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix_ref(x: mx.array, mix: mx.array) -> mx.array:
    return mx.sum(x * mix, axis=-2).astype(mx.bfloat16)


def mhc_post_ref(
    x: mx.array,
    residual: mx.array,
    post_layer_mix: mx.array,
    comb_res_mix: mx.array,
) -> mx.array:
    term2 = mx.einsum('abmn,abmc->abnc', comb_res_mix, residual.astype(mx.float32))
    return (x.astype(mx.float32)[..., None, :] * post_layer_mix + term2).astype(mx.bfloat16)


def mhc_pre_norm_fn_ref(
    residual: mx.array,
    mhc_fn: mx.array,
    mhc_norm_weight: mx.array | None,
    mhc_norm_eps: float,
) -> mx.array:
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    # residual.flatten(2, 3) -> merge dims 2 and 3
    s = residual.shape
    residual = residual.reshape(s[0], s[1], s[2] * s[3]).astype(mx.float32)
    assert mhc_fn.dtype == residual.dtype == mx.float32
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = mx.einsum(
        'mbk,nbk->mbn',
        residual.reshape(-1, 1, rms_group_size),
        mhc_fn.reshape(mhc_mult, 1, rms_group_size),
    )
    sqrsum = mx.sum(mx.square(residual.reshape(-1, 1, rms_group_size)), axis=-1)
    mixes = mx.sum(
        mixes * mx.rsqrt(sqrsum[..., None] / rms_group_size + mhc_norm_eps),
        axis=-2,
    )
    return mixes.reshape(*residual.shape[:2], -1)
