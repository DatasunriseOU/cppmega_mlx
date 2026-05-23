"""V7-C06: int8 + fp16 checkpoint compression helpers."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.runtime.checkpoint_quantize import (
    cast_fp16, dequantize_int8, quantize_int8, uncast_to_fp32,
)


def test_v7_c06_int8_round_trip_bounded_error():
    t = mx.random.normal(shape=(32, 32), key=mx.random.key(0))
    q, scale = quantize_int8(t)
    assert q.dtype == mx.int8
    deq = dequantize_int8(q, scale)
    # Quantisation error bounded by scale (one int8 step ≈ scale/2).
    err = float(mx.max(mx.abs(t - deq)).item())
    assert err <= scale * 1.0 + 1e-6


def test_v7_c06_int8_size_savings_4x():
    """fp32 → int8 should be 4x smaller in byte count."""
    t = mx.zeros((128,), dtype=mx.float32)
    q, _ = quantize_int8(t)
    # fp32 = 4 bytes, int8 = 1 byte; per-element 4x.
    assert q.dtype.size * 4 == 4  # int8 size * 4 == fp32 size


def test_v7_c06_fp16_round_trip_close_to_fp32():
    t = mx.random.normal(shape=(64,), key=mx.random.key(1))
    h = cast_fp16(t)
    back = uncast_to_fp32(h)
    # fp16 to fp32 round trip within fp16 precision.
    err = float(mx.max(mx.abs(t - back)).item())
    assert err < 1e-2


def test_v7_c06_int8_handles_zero_tensor():
    t = mx.zeros((4, 4), dtype=mx.float32)
    q, scale = quantize_int8(t)
    deq = dequantize_int8(q, scale)
    assert mx.allclose(deq, t, atol=1e-9)


def test_v7_c06_int8_handles_large_dynamic_range():
    t = mx.array([[-1000.0, 0.0, 1000.0], [-0.001, 0.0, 0.001]])
    q, scale = quantize_int8(t)
    deq = dequantize_int8(q, scale)
    # Large dynamic range: error bounded by scale (8/127 ≈ ).
    assert float(mx.max(mx.abs(t - deq)).item()) < scale * 1.1


# ---------------------------------------------------------------------------
# V7-C06 integration tests: save_state_compressed end-to-end through
# safetensors __metadata__ scales storage + round-trip.
# ---------------------------------------------------------------------------


from cppmega_v4.runtime.checkpoint_quantize import (
    save_state_compressed, load_state_compressed,
    quantize_state_int8, dequantize_state_int8,
)


def _state(seed: int = 0) -> dict:
    return {
        "layer0.w": mx.random.normal(shape=(8, 16),
                                       key=mx.random.key(seed)),
        "layer0.b": mx.random.normal(shape=(16,),
                                       key=mx.random.key(seed + 1)),
        "head.w":   mx.random.normal(shape=(8, 4),
                                       key=mx.random.key(seed + 2)),
    }


def test_v7_c06_compress_none_is_default_and_roundtrips(tmp_path):
    """AC#5: default no-compress preserves prior behaviour."""
    state = _state()
    path = str(tmp_path / "w.safetensors")
    meta = save_state_compressed(state, path, compress="none")
    assert meta["_v7_c06_compress_mode"] == "none"
    loaded = load_state_compressed(path)
    for k in state:
        assert mx.allclose(loaded[k], state[k], atol=0.0)


def test_v7_c06_weights_int8_records_scales_in_metadata(tmp_path):
    """AC#3: quant scales stored in safetensors __metadata__ and
    loaded back exactly."""
    import json
    from safetensors import safe_open
    state = _state()
    path = str(tmp_path / "w.safetensors")
    save_state_compressed(state, path, compress="weights-int8")
    with safe_open(path, framework="mlx") as f:
        meta = f.metadata() or {}
    assert meta["_v7_c06_compress_mode"] == "weights-int8"
    scales = json.loads(meta["_v7_c06_int8_scales_json"])
    # Every weight key has a scale; scales are positive floats.
    for k in state:
        assert k in scales
        assert scales[k] > 0


def test_v7_c06_weights_int8_load_dequant_bounded_logits_error(tmp_path):
    """AC#4 (weights side): max abs logits diff < 1e-2 for int8 quant
    on a tiny linear model.

    Tiny model: y = x @ W + b; W is a 16x32 random matrix. Compute
    logits with fp32 weights and with dequantised int8 weights;
    assert error stays under 1e-2."""
    W = mx.random.normal(shape=(16, 32), key=mx.random.key(7)) * 0.1
    b = mx.random.normal(shape=(32,), key=mx.random.key(8)) * 0.01
    x = mx.random.normal(shape=(4, 16), key=mx.random.key(9))
    logits_fp32 = x @ W + b

    # Save → load through int8 path.
    path = str(tmp_path / "lin.safetensors")
    save_state_compressed({"W": W, "b": b}, path,
                          compress="weights-int8")
    loaded = load_state_compressed(path)
    logits_q = x @ loaded["W"] + loaded["b"]

    max_err = float(mx.max(mx.abs(logits_fp32 - logits_q)).item())
    assert max_err < 1e-2, (
        f"int8 weights logits diff {max_err} exceeded 1e-2 bound")


def test_v7_c06_opt_fp16_roundtrip_exact_on_representable_values(tmp_path):
    """AC#4 (opt side): fp16 round-trip is exact for representable
    values (small integers / round binary fractions)."""
    state = {
        "moments.m": mx.array([0.5, 1.0, -2.0, 0.25, 8.0]),
        "moments.v": mx.array([[1.0, 0.5], [0.125, 4.0]]),
    }
    path = str(tmp_path / "opt.safetensors")
    save_state_compressed(state, path, compress="opt-fp16",
                          role="opt")
    loaded = load_state_compressed(path)
    for k in state:
        assert mx.array_equal(loaded[k], state[k]), (
            f"fp16 round-trip changed {k}")


def test_v7_c06_invalid_compress_mode_raises(tmp_path):
    state = {"t": mx.zeros((2,))}
    path = str(tmp_path / "x.safetensors")
    with pytest.raises(ValueError) as exc:
        save_state_compressed(state, path, compress="garbage")
    assert "compress must be one of" in str(exc.value)


def test_v7_c06_state_int8_helpers_roundtrip():
    state = _state()
    q, scales = quantize_state_int8(state)
    deq = dequantize_state_int8(q, scales)
    for k in state:
        # Bounded by per-tensor scale.
        err = float(mx.max(mx.abs(state[k] - deq[k])).item())
        assert err <= scales[k] + 1e-6, (k, err, scales[k])


def test_v7_c06_compress_both_records_int8_metadata(tmp_path):
    """compress='both' on weights role still picks the int8 path —
    opt-fp16 only fires when role='opt'."""
    state = _state()
    path = str(tmp_path / "w.safetensors")
    meta = save_state_compressed(state, path, compress="both",
                                  role="weights")
    assert meta["_v7_c06_compress_mode"] == "weights-int8"
    assert "_v7_c06_int8_scales_json" in meta
