from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import mlx.core as mx
import numpy as np
import pytest

import cppmega_mlx.inference as inference
from cppmega_mlx.inference import (
    ContiguousKVCache,
    generate_tokens,
    generate_tokens_with_kv_cache,
    next_token_logits,
)
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM


def _as_numpy(tokens: mx.array) -> np.ndarray:
    mx.eval(tokens)
    return np.array(tokens)


class _ScriptedLogitsModel:
    def __init__(
        self,
        next_ids_by_step: Sequence[Sequence[int]],
        *,
        vocab_size: int = 16,
        max_seq_length: int | None = None,
    ) -> None:
        self.next_ids_by_step = next_ids_by_step
        self.vocab_size = vocab_size
        self.calls = 0
        self.seen_shapes: list[tuple[int, int]] = []
        if max_seq_length is not None:
            self.config = SimpleNamespace(max_seq_length=max_seq_length)

    def __call__(self, tokens: mx.array) -> mx.array:
        batch_size, sequence_length = tokens.shape
        self.seen_shapes.append((batch_size, sequence_length))
        step = min(self.calls, len(self.next_ids_by_step) - 1)
        self.calls += 1

        next_ids = self.next_ids_by_step[step]
        assert len(next_ids) == batch_size
        logits = np.full(
            (batch_size, sequence_length, self.vocab_size),
            -1000.0,
            dtype=np.float32,
        )
        for row, token_id in enumerate(next_ids):
            logits[row, -1, token_id] = 1000.0
        return mx.array(logits)


class _KVScriptedLogitsModel(_ScriptedLogitsModel):
    def __init__(
        self,
        next_ids_by_step: Sequence[Sequence[int]],
        *,
        vocab_size: int = 16,
        max_seq_length: int | None = None,
    ) -> None:
        super().__init__(
            next_ids_by_step,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
        )
        self.seen_cache_ids: list[int] = []

    def __call__(
        self,
        tokens: mx.array,
        *,
        kv_cache: ContiguousKVCache | None = None,
    ) -> mx.array:
        assert kv_cache is not None
        self.seen_cache_ids.append(id(kv_cache))
        return super().__call__(tokens)


def _make_cache(
    *,
    batch_size: int = 1,
    head_dim: int = 32,
    max_seq_len: int | None = None,
) -> ContiguousKVCache:
    return ContiguousKVCache(
        inference.ContiguousKVCacheConfig(
            num_layers=1,
            batch_size=batch_size,
            num_kv_heads=1,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
        )
    )


def _tiny_attention_lm() -> HybridTinyLM:
    return HybridTinyLM(
        HybridTinyConfig(
            vocab_size=17,
            hidden_size=8,
            pattern="A",
            depth=1,
            dsa_a_layer_ranks=(),
            num_attention_heads=1,
            max_seq_length=8,
            structure_vocab_size=8,
            structure_bottleneck_dim=8,
            structure_num_categories=4,
            structure_max_dep_level=4,
            structure_max_ast_depth=4,
            structure_max_sibling_index=4,
            structure_num_node_types=8,
        )
    )


def test_generate_tokens_greedy_appends_full_prefix_steps() -> None:
    model = _ScriptedLogitsModel([[4], [5], [6]])
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

    tokens = generate_tokens(
        model,
        prompt,
        max_new_tokens=3,
        temperature=0.0,
    )

    np.testing.assert_array_equal(
        _as_numpy(tokens),
        np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
    )
    assert model.seen_shapes == [(1, 3), (1, 4), (1, 5)]


def test_generate_tokens_preserves_prompt_prefix_exactly() -> None:
    model = _ScriptedLogitsModel([[7, 8], [9, 10]])
    prompt = mx.array([[1, 2], [3, 4]], dtype=mx.int32)

    tokens = generate_tokens(
        model,
        prompt,
        max_new_tokens=2,
        temperature=0.0,
    )
    generated = _as_numpy(tokens)

    np.testing.assert_array_equal(generated[:, :2], _as_numpy(prompt))
    np.testing.assert_array_equal(
        generated,
        np.array([[1, 2, 7, 9], [3, 4, 8, 10]], dtype=np.int32),
    )


def test_generate_tokens_stops_on_eos_only_when_all_rows_emit_eos() -> None:
    eos = 2
    model = _ScriptedLogitsModel([[eos, 1], [eos, eos], [5, 5]])
    prompt = mx.array([[10], [11]], dtype=mx.int32)

    tokens = generate_tokens(
        model,
        prompt,
        max_new_tokens=3,
        eos_token_id=eos,
        temperature=0.0,
    )

    np.testing.assert_array_equal(
        _as_numpy(tokens),
        np.array([[10, eos, eos], [11, 1, eos]], dtype=np.int32),
    )
    assert model.calls == 2


def test_generate_tokens_returns_prompt_for_zero_new_tokens() -> None:
    model = _ScriptedLogitsModel([[1]])
    prompt = mx.array([[1, 2]], dtype=mx.int32)

    tokens = generate_tokens(model, prompt, max_new_tokens=0)

    assert tokens is prompt
    assert model.calls == 0


def test_generate_tokens_rejects_invalid_prompt_rank() -> None:
    model = _ScriptedLogitsModel([[1]])

    with pytest.raises(ValueError, match="prompt_ids"):
        generate_tokens(model, mx.array([1, 2], dtype=mx.int32), max_new_tokens=1)


def test_generate_tokens_rejects_negative_max_new_tokens() -> None:
    model = _ScriptedLogitsModel([[1]])

    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_tokens(
            model,
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=-1,
        )


def test_generate_tokens_rejects_non_3d_logits() -> None:
    class BadRankModel:
        def __call__(self, tokens: mx.array) -> mx.array:
            return mx.zeros((tokens.shape[0], 4), dtype=mx.float32)

    with pytest.raises(ValueError, match="shape"):
        generate_tokens(
            BadRankModel(),
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
        )


def test_next_token_logits_returns_last_standard_position() -> None:
    tokens = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
    logits = mx.array(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ],
        dtype=mx.float32,
    )

    step_logits = next_token_logits(logits, tokens)

    np.testing.assert_array_equal(
        _as_numpy(step_logits),
        np.array([[3.0, 4.0, 5.0], [9.0, 10.0, 11.0]], dtype=np.float32),
    )


def test_generate_tokens_rejects_mtp_tuple_output() -> None:
    class MTPModel:
        def __call__(self, tokens: mx.array) -> tuple[mx.array, mx.array]:
            main_logits = mx.zeros(
                (tokens.shape[0], tokens.shape[1], 4),
                dtype=mx.float32,
            )
            mtp_logits = mx.zeros(
                (tokens.shape[0], 2, tokens.shape[1], 4),
                dtype=mx.float32,
            )
            return main_logits, mtp_logits

    with pytest.raises(ValueError, match="MTP/draft tuple"):
        generate_tokens(
            MTPModel(),
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
        )


def test_generate_tokens_rejects_structured_mtp_output() -> None:
    class StructuredMTPModel:
        def __call__(self, tokens: mx.array) -> dict[str, mx.array]:
            return {
                "logits": mx.zeros(
                    (tokens.shape[0], tokens.shape[1], 4),
                    dtype=mx.float32,
                ),
                "mtp_logits": mx.zeros(
                    (tokens.shape[0], 2, tokens.shape[1], 4),
                    dtype=mx.float32,
                ),
            }

    with pytest.raises(ValueError, match="structured model outputs"):
        generate_tokens(
            StructuredMTPModel(),
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
        )


def test_generate_tokens_rejects_4d_mtp_logits() -> None:
    class FourDMTPModel:
        def __call__(self, tokens: mx.array) -> mx.array:
            return mx.zeros(
                (tokens.shape[0], 2, tokens.shape[1], 4),
                dtype=mx.float32,
            )

    with pytest.raises(ValueError, match="MTP/draft logits"):
        generate_tokens(
            FourDMTPModel(),
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
        )


def test_generate_tokens_rejects_non_array_model_output() -> None:
    class BadOutputModel:
        def __call__(self, tokens: mx.array) -> object:
            return object()

    with pytest.raises(TypeError, match="mlx.core.array"):
        generate_tokens(
            BadOutputModel(),
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
        )


def test_generate_tokens_rejects_logits_batch_mismatch() -> None:
    class BadBatchModel:
        def __call__(self, tokens: mx.array) -> mx.array:
            return mx.zeros((tokens.shape[0] + 1, tokens.shape[1], 4), dtype=mx.float32)

    with pytest.raises(ValueError, match="batch size"):
        generate_tokens(
            BadBatchModel(),
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
        )


def test_generate_tokens_rejects_logits_sequence_mismatch() -> None:
    class BadSequenceModel:
        def __call__(self, tokens: mx.array) -> mx.array:
            return mx.zeros((tokens.shape[0], tokens.shape[1] + 1, 4), dtype=mx.float32)

    with pytest.raises(ValueError, match="sequence length"):
        generate_tokens(
            BadSequenceModel(),
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
        )


def test_generate_tokens_rejects_max_seq_length_overflow_before_forward() -> None:
    model = _ScriptedLogitsModel([[3], [4]], max_seq_length=3)

    with pytest.raises(ValueError, match="max_seq_length"):
        generate_tokens(
            model,
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=2,
            temperature=0.0,
        )
    assert model.calls == 1


def test_generate_tokens_seeded_sampling_is_deterministic() -> None:
    class UniformModel:
        def __call__(self, tokens: mx.array) -> mx.array:
            batch_size, sequence_length = tokens.shape
            return mx.zeros((batch_size, sequence_length, 5), dtype=mx.float32)

    prompt = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
    left = generate_tokens(
        UniformModel(),
        prompt,
        max_new_tokens=4,
        rng_key=mx.random.key(123),
    )
    right = generate_tokens(
        UniformModel(),
        prompt,
        max_new_tokens=4,
        rng_key=mx.random.key(123),
    )

    np.testing.assert_array_equal(_as_numpy(left), _as_numpy(right))


def test_generate_tokens_with_kv_cache_prefills_then_decodes_one_token_steps() -> None:
    model = _KVScriptedLogitsModel([[4], [5], [6]])
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    cache = _make_cache()

    tokens = generate_tokens_with_kv_cache(
        model,
        prompt,
        max_new_tokens=3,
        cache=cache,
        temperature=0.0,
    )

    np.testing.assert_array_equal(
        _as_numpy(tokens),
        np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
    )
    assert model.seen_shapes == [(1, 3), (1, 1), (1, 1)]
    assert model.seen_cache_ids == [id(cache), id(cache), id(cache)]


def test_generate_tokens_with_kv_cache_handles_batched_rows() -> None:
    model = _KVScriptedLogitsModel([[4, 5], [6, 7]])
    prompt = mx.array([[1, 2], [3, 4]], dtype=mx.int32)

    tokens = generate_tokens_with_kv_cache(
        model,
        prompt,
        max_new_tokens=2,
        cache=_make_cache(batch_size=2),
        temperature=0.0,
    )

    np.testing.assert_array_equal(
        _as_numpy(tokens),
        np.array([[1, 2, 4, 6], [3, 4, 5, 7]], dtype=np.int32),
    )
    assert model.seen_shapes == [(2, 2), (2, 1)]


def test_generate_tokens_with_kv_cache_stops_on_eos_without_extra_decode() -> None:
    eos = 2
    model = _KVScriptedLogitsModel([[eos, eos], [9, 9]])
    prompt = mx.array([[10], [11]], dtype=mx.int32)
    cache = _make_cache(batch_size=2)

    tokens = generate_tokens_with_kv_cache(
        model,
        prompt,
        max_new_tokens=3,
        cache=cache,
        eos_token_id=eos,
        temperature=0.0,
    )

    np.testing.assert_array_equal(
        _as_numpy(tokens),
        np.array([[10, eos], [11, eos]], dtype=np.int32),
    )
    assert model.calls == 1
    assert model.seen_shapes == [(2, 1)]


def test_generate_tokens_with_kv_cache_returns_prompt_for_zero_new_tokens() -> None:
    model = _KVScriptedLogitsModel([[1]])
    prompt = mx.array([[1, 2]], dtype=mx.int32)

    tokens = generate_tokens_with_kv_cache(
        model,
        prompt,
        max_new_tokens=0,
        cache=_make_cache(),
    )

    assert tokens is prompt
    assert model.calls == 0


def test_generate_tokens_with_kv_cache_rejects_max_seq_overflow_before_decode() -> None:
    model = _KVScriptedLogitsModel([[3], [4]], max_seq_length=3)

    with pytest.raises(ValueError, match="max_seq_length"):
        generate_tokens_with_kv_cache(
            model,
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=2,
            cache=_make_cache(),
            temperature=0.0,
        )
    assert model.calls == 1
    assert model.seen_shapes == [(1, 2)]


def test_generate_tokens_with_kv_cache_builds_cache_from_shape_kwargs() -> None:
    model = _KVScriptedLogitsModel([[3], [4]])

    tokens = generate_tokens_with_kv_cache(
        model,
        mx.array([[1, 2]], dtype=mx.int32),
        max_new_tokens=2,
        num_layers=1,
        num_kv_heads=1,
        head_dim=32,
        temperature=0.0,
    )

    np.testing.assert_array_equal(
        _as_numpy(tokens),
        np.array([[1, 2, 3, 4]], dtype=np.int32),
    )
    assert len(set(model.seen_cache_ids)) == 1


def test_generate_tokens_with_kv_cache_rejects_mixed_cache_configuration() -> None:
    model = _KVScriptedLogitsModel([[1]])

    with pytest.raises(ValueError, match="cache"):
        generate_tokens_with_kv_cache(
            model,
            mx.array([[1, 2]], dtype=mx.int32),
            max_new_tokens=1,
            cache=_make_cache(),
            num_layers=1,
            num_kv_heads=1,
            head_dim=32,
        )


def test_generate_tokens_with_kv_cache_rejects_batch_mismatch() -> None:
    model = _KVScriptedLogitsModel([[1]])

    with pytest.raises(ValueError, match="batch_size"):
        generate_tokens_with_kv_cache(
            model,
            mx.array([[1, 2], [3, 4]], dtype=mx.int32),
            max_new_tokens=1,
            cache=_make_cache(batch_size=1),
        )
    assert model.calls == 0


def test_generate_tokens_with_kv_cache_seeded_sampling_is_deterministic() -> None:
    class UniformKVModel:
        def __init__(self) -> None:
            self.seen_cache_ids: list[int] = []

        def __call__(
            self,
            tokens: mx.array,
            *,
            kv_cache: ContiguousKVCache | None = None,
        ) -> mx.array:
            assert kv_cache is not None
            self.seen_cache_ids.append(id(kv_cache))
            batch_size, sequence_length = tokens.shape
            return mx.zeros((batch_size, sequence_length, 5), dtype=mx.float32)

    prompt = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
    left_model = UniformKVModel()
    right_model = UniformKVModel()

    left = generate_tokens_with_kv_cache(
        left_model,
        prompt,
        max_new_tokens=4,
        cache=_make_cache(batch_size=2),
        rng_key=mx.random.key(123),
    )
    right = generate_tokens_with_kv_cache(
        right_model,
        prompt,
        max_new_tokens=4,
        cache=_make_cache(batch_size=2),
        rng_key=mx.random.key(123),
    )

    np.testing.assert_array_equal(_as_numpy(left), _as_numpy(right))
    assert len(set(left_model.seen_cache_ids)) == 1
    assert len(set(right_model.seen_cache_ids)) == 1


def test_real_hybrid_tiny_lm_greedy_kv_cache_matches_full_prefix_generation() -> None:
    model = _tiny_attention_lm()
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

    full_prefix = generate_tokens(
        model,
        prompt,
        max_new_tokens=2,
        temperature=0.0,
    )
    cached = generate_tokens_with_kv_cache(
        model,
        prompt,
        max_new_tokens=2,
        cache=_make_cache(
            head_dim=8,
            max_seq_len=8,
        ),
        temperature=0.0,
    )

    np.testing.assert_array_equal(_as_numpy(cached), _as_numpy(full_prefix))


def test_inference_root_exports_generate_tokens() -> None:
    assert inference.generate_tokens is generate_tokens
    assert "generate_tokens" in inference.__all__


def test_inference_root_exports_generate_tokens_with_kv_cache() -> None:
    assert inference.generate_tokens_with_kv_cache is generate_tokens_with_kv_cache
    assert "generate_tokens_with_kv_cache" in inference.__all__


def test_inference_root_exports_next_token_logits() -> None:
    assert inference.next_token_logits is next_token_logits
    assert "next_token_logits" in inference.__all__
