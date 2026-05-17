# Vendored from ml-explore/mlx-lm PR #1224 (mlx_lm/models/qwen3_5_moe.py
# `Model.sanitize` FP8 dequant block), generalized to a standalone utility.
# Upstream license: MIT, © 2025-2026 Apple Inc.
#
# Mirrors the inline FP8 dequant pattern that lives in models/{deepseek_v3,
# deepseek_v32, minimax, mimo_v2_flash, ministral3, qwen3_5_moe}.py — pulled
# out so cppmega_v4 (MoE + Lightning Indexer ROI 7) can dequant FP8
# checkpoints without re-copying the snippet in every loader.

"""FP8 block-scaled dequant utility (block size 128, paired weight_scale_inv).

Triggered by HF checkpoints with quant_method=fp8 (e.g. Qwen/Qwen3.6-27B-FP8,
DeepSeek-V3-FP8). Pairs each weight tensor with its `weight_scale_inv` and
multiplies them block-wise to recover bfloat16 weights.
"""

from typing import Mapping

import mlx.core as mx

_BLOCK_SIZE = 128


def dequant_block_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Block-dequantize a single FP8 weight tensor.

    Args:
        weight: [m, n] in fp8.
        scale_inv: [ceil(m/128), ceil(n/128)] in fp32; broadcast across
            the [128, 128] block.

    Returns:
        [m, n] in bfloat16.
    """
    weight = mx.from_fp8(weight, dtype=mx.bfloat16)
    bs = _BLOCK_SIZE
    m, n = weight.shape
    pad_bottom = (-m) % bs
    pad_side = (-n) % bs
    weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
    weight = weight.reshape(
        ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
    )
    weight = (weight * scale_inv[:, None, :, None]).reshape(
        m + pad_bottom, n + pad_side
    )
    return weight[:m, :n].astype(mx.bfloat16)


def sanitize_fp8_weights(weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
    """Dequantize all FP8-quantized tensors in a state-dict.

    Iterates over keys and for every ``<prefix>.weight_scale_inv`` pair with a
    matching ``<prefix>.weight`` in fp8, replaces the fp8 weight with its
    bfloat16 dequant. ``activation_scale`` keys are dropped (PTQ artefact).
    Returns the rewritten state-dict (or the input unchanged if no fp8 keys).
    """
    if not any("weight_scale_inv" in k for k in weights):
        return dict(weights)

    new_weights: dict[str, mx.array] = {}
    for k, v in weights.items():
        if "weight_scale_inv" in k:
            wk = k.replace("_scale_inv", "")
            new_weights[wk] = dequant_block_fp8(weights[wk], v)
        elif "activation_scale" in k:
            continue
        elif k not in new_weights:
            new_weights[k] = v
    return new_weights


__all__ = ["dequant_block_fp8", "sanitize_fp8_weights"]
