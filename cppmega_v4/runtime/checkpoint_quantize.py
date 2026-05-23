"""V7-C06: per-tensor symmetric int8 quantisation + fp16 cast helpers.

quantize_int8(t) → (q_int8, scale_fp32)
dequantize_int8(q, scale) → fp32 tensor

cast_fp16(t) → fp16; uncast_to_fp32 just calls .astype(mx.float32).
"""

from __future__ import annotations

import mlx.core as mx


def quantize_int8(t: mx.array) -> tuple[mx.array, float]:
    """Symmetric int8: scale = max(abs(t)) / 127, q = round(t / scale)."""
    if t.size == 0:
        return mx.zeros(t.shape, dtype=mx.int8), 1.0
    absmax = float(mx.max(mx.abs(t.astype(mx.float32))).item())
    scale = max(absmax / 127.0, 1e-9)
    q = mx.round(t.astype(mx.float32) / scale)
    q = mx.clip(q, -127, 127).astype(mx.int8)
    return q, float(scale)


def dequantize_int8(q: mx.array, scale: float) -> mx.array:
    return q.astype(mx.float32) * scale


def cast_fp16(t: mx.array) -> mx.array:
    return t.astype(mx.float16)


def uncast_to_fp32(t: mx.array) -> mx.array:
    return t.astype(mx.float32)


def quantize_state_int8(
    state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, float]]:
    """Per-tensor symmetric int8 quantisation across a whole state dict.

    Returns (quant_state, scales) where ``quant_state[k]`` is the int8
    tensor and ``scales[k]`` is its per-tensor scale factor. Recover
    the original via :func:`dequantize_state_int8`."""
    quant: dict[str, mx.array] = {}
    scales: dict[str, float] = {}
    for k, t in state.items():
        q, s = quantize_int8(t)
        quant[k] = q
        scales[k] = float(s)
    return quant, scales


def dequantize_state_int8(
    quant: dict[str, mx.array],
    scales: dict[str, float],
) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for k, q in quant.items():
        out[k] = dequantize_int8(q, scales[k])
    return out


def cast_state_fp16(state: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k: cast_fp16(t) for k, t in state.items()}


def uncast_state_fp32(state: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k: uncast_to_fp32(t) for k, t in state.items()}


# ---------------------------------------------------------------------------
# V7-C06: save_checkpoint integration — opt-in compress mode.
# ---------------------------------------------------------------------------

# Allowed values for opts.compress in stage_train.
_VALID_COMPRESS_MODES = ("none", "weights-int8", "opt-fp16", "both")

_QUANT_SCALES_META_KEY = "_v7_c06_int8_scales_json"
_QUANT_FORMAT_META_KEY = "_v7_c06_compress_mode"


def save_state_compressed(
    state: dict[str, mx.array],
    path: str,
    *,
    compress: str = "none",
    metadata: dict[str, str] | None = None,
    role: str = "weights",
) -> dict[str, str]:
    """Save a state dict honouring the V7-C06 ``compress`` knob.

    role:
        ``\"weights\"`` — int8 path runs when compress in
          {weights-int8, both}.
        ``\"opt\"``   — fp16 path runs when compress in
          {opt-fp16, both}.

    Returns the merged metadata dict actually written (so callers can
    record what was applied). Quantisation scales are JSON-encoded into
    the safetensors ``__metadata__`` block under
    ``_v7_c06_int8_scales_json`` so :func:`load_state_compressed` can
    invert the transform without an out-of-band sidecar.
    """
    import json as _json
    import safetensors.mlx as st_mlx
    if compress not in _VALID_COMPRESS_MODES:
        raise ValueError(
            f"compress must be one of {_VALID_COMPRESS_MODES}, "
            f"got {compress!r}")
    meta = dict(metadata or {})
    out_state = state
    if role == "weights" and compress in ("weights-int8", "both"):
        quant, scales = quantize_state_int8(state)
        out_state = quant
        meta[_QUANT_SCALES_META_KEY] = _json.dumps(scales, sort_keys=True)
        meta[_QUANT_FORMAT_META_KEY] = "weights-int8"
    elif role == "opt" and compress in ("opt-fp16", "both"):
        out_state = cast_state_fp16(state)
        meta[_QUANT_FORMAT_META_KEY] = "opt-fp16"
    else:
        meta.setdefault(_QUANT_FORMAT_META_KEY, "none")
    st_mlx.save_file(out_state, path, metadata=meta)
    return meta


def load_state_compressed(path: str) -> dict[str, mx.array]:
    """Inverse of :func:`save_state_compressed`: reads metadata, applies
    the right dequantisation/cast, returns the reconstructed state in
    its original dtype (fp32)."""
    import json as _json
    from safetensors import safe_open
    state: dict[str, mx.array] = {}
    with safe_open(path, framework="mlx") as f:
        meta = f.metadata() or {}
        mode = meta.get(_QUANT_FORMAT_META_KEY, "none")
        for k in f.keys():
            state[k] = f.get_tensor(k)
    if mode == "weights-int8":
        scales = _json.loads(meta[_QUANT_SCALES_META_KEY])
        state = dequantize_state_int8(state, scales)
    elif mode == "opt-fp16":
        state = uncast_state_fp32(state)
    return state


__all__ = [
    "quantize_int8", "dequantize_int8",
    "cast_fp16", "uncast_to_fp32",
    "quantize_state_int8", "dequantize_state_int8",
    "cast_state_fp16", "uncast_state_fp32",
    "save_state_compressed", "load_state_compressed",
]
