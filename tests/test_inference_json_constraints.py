from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    generate_tokens,
    generate_tokens_with_kv_cache,
    generate_tokens_with_prompt_cache,
    sample_next_token,
    stream_generate_tokens,
)


def _as_numpy(tokens: mx.array) -> np.ndarray:
    mx.eval(tokens)
    return np.array(tokens)


def _json_api() -> tuple[type[Any], type[Any]]:
    names = ("JsonTokenIds", "JsonConstrainedLogitsProcessor")
    missing = [name for name in names if not hasattr(inference, name)]
    assert not missing, f"missing JSON constraint API exports: {missing}"
    return (
        cast(type[Any], getattr(inference, "JsonTokenIds")),
        cast(type[Any], getattr(inference, "JsonConstrainedLogitsProcessor")),
    )


def _json_ids() -> Any:
    JsonTokenIds, _ = _json_api()
    return JsonTokenIds(
        object_start=1,
        object_end=2,
        array_start=3,
        array_end=4,
        colon=5,
        comma=6,
        string=7,
        number=8,
        true_literal=9,
        false_literal=10,
        null_literal=11,
        whitespace=(0,),
        eos_token_id=12,
    )


class _PreferenceLogitsModel:
    def __init__(self, preferences_by_call: list[list[int]], *, vocab_size: int = 13) -> None:
        self.preferences_by_call = preferences_by_call
        self.vocab_size = vocab_size
        self.calls = 0
        self.seen_shapes: list[tuple[int, int]] = []

    def __call__(
        self,
        tokens: mx.array,
        *,
        kv_cache: inference.ContiguousKVCache | None = None,
        **kwargs: object,
    ) -> mx.array:
        del kv_cache, kwargs
        batch_size, sequence_length = tokens.shape
        self.seen_shapes.append((int(batch_size), int(sequence_length)))
        call_idx = min(self.calls, len(self.preferences_by_call) - 1)
        self.calls += 1
        logits = np.full(
            (batch_size, sequence_length, self.vocab_size),
            -1000.0,
            dtype=np.float32,
        )
        for rank, token_id in enumerate(self.preferences_by_call[call_idx]):
            logits[:, -1, token_id] = 100.0 - rank
        return mx.array(logits)


class _PreferenceKVLogitsModel(_PreferenceLogitsModel):
    def __call__(
        self,
        tokens: mx.array,
        *,
        kv_cache: inference.ContiguousKVCache | None = None,
        **kwargs: object,
    ) -> mx.array:
        if kv_cache is None:
            raise AssertionError("expected KV cache")
        del kwargs
        return super().__call__(tokens, kv_cache=kv_cache)


def _step_logits(preferences: list[int], *, vocab_size: int = 13) -> mx.array:
    logits = np.full((1, vocab_size), -1000.0, dtype=np.float32)
    for rank, token_id in enumerate(preferences):
        logits[:, token_id] = 100.0 - rank
    return mx.array(logits)


def _make_cache() -> inference.ContiguousKVCache:
    return inference.ContiguousKVCache(
        inference.ContiguousKVCacheConfig(
            num_layers=1,
            batch_size=1,
            num_kv_heads=1,
            head_dim=32,
        )
    )


def _make_prefilled_cache() -> inference.ContiguousKVCache:
    cache = _make_cache()
    keys = mx.zeros((1, 1, 1, 32), dtype=mx.float32)
    cache.update_and_fetch(0, keys, keys)
    return cache


def _processor(*, start_position: int = 0) -> Any:
    _, JsonConstrainedLogitsProcessor = _json_api()
    return JsonConstrainedLogitsProcessor(_json_ids(), start_position=start_position)


def test_json_processor_allows_only_valid_next_tokens_for_object_prefixes() -> None:
    ids = _json_ids()
    processor = _processor()

    assert set(processor.allowed_token_ids(mx.array([[ids.object_start]], dtype=mx.int32))) == {
        ids.whitespace[0],
        ids.object_end,
        ids.string,
    }
    assert set(
        processor.allowed_token_ids(
            mx.array([[ids.object_start, ids.string]], dtype=mx.int32)
        )
    ) == {ids.whitespace[0], ids.colon}
    assert set(
        processor.allowed_token_ids(
            mx.array([[ids.object_start, ids.string, ids.colon]], dtype=mx.int32)
        )
    ) == {
        ids.whitespace[0],
        ids.object_start,
        ids.array_start,
        ids.string,
        ids.number,
        ids.true_literal,
        ids.false_literal,
        ids.null_literal,
    }


def test_json_processor_allows_eos_only_after_complete_root_value() -> None:
    ids = _json_ids()
    processor = _processor()

    allowed = set(
        processor.allowed_token_ids(
            mx.array([[ids.object_start, ids.object_end]], dtype=mx.int32)
        )
    )

    assert allowed == {ids.whitespace[0], ids.eos_token_id}


def test_json_processor_fails_closed_for_invalid_json_prefix() -> None:
    ids = _json_ids()
    processor = _processor()

    with pytest.raises(ValueError, match="invalid JSON prefix"):
        processor.allowed_token_ids(
            mx.array([[ids.object_start, ids.colon]], dtype=mx.int32)
        )


def test_sample_next_token_applies_json_processor_before_greedy_decode() -> None:
    ids = _json_ids()
    logits = mx.full((1, 13), -1000.0, dtype=mx.float32)
    logits[:, ids.array_start] = 100.0
    logits[:, ids.string] = 10.0

    token = sample_next_token(
        logits,
        temperature=0.0,
        tokens=mx.array([[ids.object_start]], dtype=mx.int32),
        logits_processors=[_processor()],
    )

    np.testing.assert_array_equal(_as_numpy(token), np.array([[ids.string]], dtype=np.int32))


def test_sample_next_token_requires_prefix_tokens_for_logits_processors() -> None:
    with pytest.raises(ValueError, match="tokens"):
        sample_next_token(
            mx.zeros((1, 13), dtype=mx.float32),
            temperature=0.0,
            logits_processors=[_processor()],
        )


def test_generation_helpers_thread_json_processor_through_current_prefix() -> None:
    ids = _json_ids()
    prompt = mx.array([[99]], dtype=mx.int32)
    processor = _processor(start_position=1)
    preferences = [
        [ids.colon, ids.object_start],
        [ids.colon, ids.object_end],
        [ids.comma, ids.eos_token_id],
    ]

    eager_model = _PreferenceLogitsModel(preferences)
    eager = generate_tokens(
        eager_model,
        prompt,
        max_new_tokens=3,
        temperature=0.0,
        eos_token_id=ids.eos_token_id,
        logits_processors=[processor],
    )
    np.testing.assert_array_equal(
        _as_numpy(eager),
        np.array([[99, ids.object_start, ids.object_end, ids.eos_token_id]], dtype=np.int32),
    )
    assert eager_model.seen_shapes == [(1, 1), (1, 2), (1, 3)]

    kv_model = _PreferenceKVLogitsModel(preferences)
    kv = generate_tokens_with_kv_cache(
        kv_model,
        prompt,
        max_new_tokens=3,
        cache=_make_cache(),
        temperature=0.0,
        eos_token_id=ids.eos_token_id,
        logits_processors=[processor],
    )
    np.testing.assert_array_equal(_as_numpy(kv), _as_numpy(eager))
    assert kv_model.seen_shapes == [(1, 1), (1, 1), (1, 1)]

    prompt_cache_model = _PreferenceKVLogitsModel(preferences[1:])
    prompt_cache = inference.PromptCacheEntry(
        prompt_ids=prompt,
        cache=_make_prefilled_cache(),
        next_logits=_step_logits(preferences[0]),
    )
    cached = generate_tokens_with_prompt_cache(
        prompt_cache_model,
        prompt,
        prompt_cache=prompt_cache,
        max_new_tokens=3,
        temperature=0.0,
        eos_token_id=ids.eos_token_id,
        logits_processors=[processor],
    )
    np.testing.assert_array_equal(_as_numpy(cached), _as_numpy(eager))
    assert prompt_cache_model.seen_shapes == [(1, 1), (1, 1)]

    stream_model = _PreferenceLogitsModel(preferences)
    chunks = list(
        stream_generate_tokens(
            stream_model,
            prompt,
            max_new_tokens=3,
            temperature=0.0,
            eos_token_id=ids.eos_token_id,
            logits_processors=[processor],
        )
    )
    np.testing.assert_array_equal(_as_numpy(chunks[-1].tokens), _as_numpy(eager))
    assert [chunk.finish_reason for chunk in chunks] == [None, None, "eos"]
