from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    InferenceQuantizationConfig,
    make_quantized_kv_cache,
    quantize_kv_cache,
    quantize_linear_for_inference,
    should_start_kv_quantization,
    validate_kv_head_dim,
)
from cppmega_mlx.inference.quantization import quantize_module_for_inference


def _as_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.array(array)


def test_inference_quantization_config_defaults_to_q4_and_kv_q4() -> None:
    config = InferenceQuantizationConfig()

    assert config.bits == 4
    assert config.group_size == 64
    assert config.mode == "affine"
    assert config.kv_bits == 4
    assert config.kv_group_size == 64
    assert config.quantized_kv_start == 256


def test_quantize_linear_for_inference_returns_quantized_linear() -> None:
    linear = nn.Linear(64, 8, bias=True)
    quantized = quantize_linear_for_inference(linear, group_size=32, bits=4)

    assert isinstance(quantized, nn.QuantizedLinear)
    output = quantized(mx.ones((2, 64), dtype=mx.float32))

    assert output.shape == (2, 8)
    assert output.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(output)))


def test_quantized_linear_output_is_close_to_source_linear_for_tiny_layer() -> None:
    linear = nn.Linear(64, 4, bias=False)
    inputs = mx.arange(2 * 64, dtype=mx.float32).reshape(2, 64) / 64.0

    source = linear(inputs)
    quantized = quantize_linear_for_inference(linear, group_size=32, bits=4)(inputs)

    assert source.shape == quantized.shape
    np.testing.assert_allclose(_as_numpy(quantized), _as_numpy(source), atol=0.25, rtol=0.25)


def test_quantize_linear_for_inference_rejects_non_linear_module() -> None:
    with pytest.raises(TypeError, match="mlx.nn.Linear"):
        quantize_linear_for_inference(nn.ReLU())  # type: ignore[arg-type]


def test_quantize_linear_for_inference_rejects_non_divisible_input_dim() -> None:
    linear = nn.Linear(48, 8)

    with pytest.raises(ValueError, match="divisible by group_size"):
        quantize_linear_for_inference(linear, group_size=32)


def test_quantize_module_for_inference_converts_nested_linear_layers() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(64, 16)
            self.extra = [nn.Linear(64, 16)]

        def __call__(self, inputs: mx.array) -> mx.array:
            return self.proj(inputs) + self.extra[0](inputs)

    class TinyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = Block()
            self.proj = nn.Linear(64, 16)

        def __call__(self, inputs: mx.array) -> mx.array:
            return self.block(inputs) + self.proj(inputs)

    model = TinyModule()

    returned = quantize_module_for_inference(model, group_size=32, bits=4)

    assert returned is model
    assert isinstance(model.block.proj, nn.QuantizedLinear)
    assert isinstance(model.block.extra[0], nn.QuantizedLinear)
    assert isinstance(model.proj, nn.QuantizedLinear)
    output = model(mx.ones((2, 64), dtype=mx.float32))

    assert output.shape == (2, 16)
    assert bool(mx.all(mx.isfinite(output)))


def test_quantize_module_for_inference_skips_default_output_head_name() -> None:
    class TinyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(64, 16)
            self.lm_head = nn.Linear(64, 16)

    model = TinyModule()

    quantize_module_for_inference(model, group_size=32, bits=4)

    assert isinstance(model.proj, nn.QuantizedLinear)
    assert isinstance(model.lm_head, nn.Linear)


def test_quantize_module_for_inference_uses_skip_predicate() -> None:
    class TinyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.keep_float = nn.Linear(64, 16)
            self.quantized = nn.Linear(64, 16)

    model = TinyModule()

    quantize_module_for_inference(
        model,
        group_size=32,
        bits=4,
        skip_predicate=lambda name, _: name == "keep_float",
    )

    assert isinstance(model.keep_float, nn.Linear)
    assert isinstance(model.quantized, nn.QuantizedLinear)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bits": 3}, "bits"),
        ({"group_size": 16}, "group_size"),
        ({"mode": "symmetric"}, "mode"),
        ({"kv_bits": 3}, "bits"),
        ({"kv_group_size": 16}, "group_size"),
        ({"quantized_kv_start": -1}, "quantized_kv_start"),
    ],
)
def test_inference_quantization_config_fails_closed(
    kwargs: dict[str, int | str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        InferenceQuantizationConfig(**kwargs)  # type: ignore[arg-type]


def test_make_quantized_kv_cache_uses_q4_defaults() -> None:
    cache = make_quantized_kv_cache()

    assert isinstance(cache, QuantizedKVCache)
    assert cache.bits == 4
    assert cache.group_size == 64
    assert cache.offset == 0


def test_quantized_kv_cache_update_smoke_with_valid_group_size() -> None:
    cache = make_quantized_kv_cache(group_size=32, bits=4)
    keys = mx.arange(1 * 2 * 3 * 32, dtype=mx.float32).reshape(1, 2, 3, 32)
    values = keys + 100.0

    packed_keys, packed_values = cache.update_and_fetch(keys, values)
    mx.eval(*(tuple(packed_keys) + tuple(packed_values)))

    assert cache.offset == 3
    assert [part.shape for part in packed_keys] == [(1, 2, 3, 4), (1, 2, 3, 1), (1, 2, 3, 1)]
    assert [part.dtype for part in packed_keys] == [mx.uint32, mx.float32, mx.float32]
    assert [part.shape for part in packed_values] == [
        (1, 2, 3, 4),
        (1, 2, 3, 1),
        (1, 2, 3, 1),
    ]


def test_quantize_kv_cache_converts_existing_kv_cache() -> None:
    source = KVCache()
    keys = mx.ones((1, 2, 1, 32), dtype=mx.float32)
    values = mx.ones((1, 2, 1, 32), dtype=mx.float32)
    source.update_and_fetch(keys, values)

    quantized = quantize_kv_cache(source, group_size=32, bits=4)

    assert isinstance(quantized, QuantizedKVCache)
    assert quantized.offset == 1
    assert quantized.bits == 4
    assert quantized.group_size == 32


def test_quantize_kv_cache_returns_matching_quantized_cache() -> None:
    cache = make_quantized_kv_cache(group_size=32, bits=4)

    assert quantize_kv_cache(cache, group_size=32, bits=4) is cache


def test_quantize_kv_cache_rejects_incompatible_quantized_cache() -> None:
    cache = make_quantized_kv_cache(group_size=32, bits=4)

    with pytest.raises(ValueError, match="incompatible"):
        quantize_kv_cache(cache, group_size=64, bits=4)


def test_quantize_kv_cache_rejects_non_kv_cache() -> None:
    with pytest.raises(TypeError, match="KVCache"):
        quantize_kv_cache(object())  # type: ignore[arg-type]


def test_validate_kv_head_dim_fails_closed() -> None:
    validate_kv_head_dim(64, group_size=32)

    with pytest.raises(ValueError, match="positive"):
        validate_kv_head_dim(0, group_size=32)
    with pytest.raises(ValueError, match="divisible"):
        validate_kv_head_dim(48, group_size=32)
    with pytest.raises(ValueError, match="group_size"):
        validate_kv_head_dim(64, group_size=16)


def test_should_start_kv_quantization_threshold() -> None:
    assert not should_start_kv_quantization(255)
    assert should_start_kv_quantization(256)
    assert should_start_kv_quantization(10, quantized_kv_start=0)

    with pytest.raises(ValueError, match="position"):
        should_start_kv_quantization(-1)
    with pytest.raises(ValueError, match="quantized_kv_start"):
        should_start_kv_quantization(0, quantized_kv_start=-1)


def test_inference_root_exports_quantization_helpers() -> None:
    assert inference.InferenceQuantizationConfig is InferenceQuantizationConfig
    assert inference.make_quantized_kv_cache is make_quantized_kv_cache
    assert inference.quantize_kv_cache is quantize_kv_cache
    assert inference.quantize_linear_for_inference is quantize_linear_for_inference
    assert inference.should_start_kv_quantization is should_start_kv_quantization
    assert inference.validate_kv_head_dim is validate_kv_head_dim
    assert {
        "InferenceQuantizationConfig",
        "make_quantized_kv_cache",
        "quantize_kv_cache",
        "quantize_linear_for_inference",
        "should_start_kv_quantization",
        "validate_kv_head_dim",
    } <= set(inference.__all__)
