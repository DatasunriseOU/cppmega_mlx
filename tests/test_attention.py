from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from cppmega_mlx.inference.engine import (
    ContiguousKVCacheConfig,
    make_contiguous_kv_cache,
)
from cppmega_mlx.nn.attention import (
    AttentionConfig,
    CausalSelfAttention,
    SPARSE_MLA_FP8_PREPARED_BUFFER_NAMES,
    SPARSE_MLA_FP8_PRODUCER_OWNER,
    SPARSE_MLA_FP8_PRODUCER_STAGE,
    SparseMLAFp8Prepared,
    causal_sparse_indices,
    causal_sdpa_mask,
    sparse_indices_from_attention_mask,
)
from cppmega_mlx.runtime.kernel_policy import clear_dispatch_log, get_dispatch_log


def _rand(shape: tuple[int, ...], seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


def _tree_arrays(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _tree_arrays(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tree_arrays(value)
    else:
        yield tree


def test_attention_config_preserves_mla_and_dsa_modes() -> None:
    mla = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    dsa = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="dsa")

    assert mla.mode == "mla"
    assert dsa.mode == "dsa"
    assert mla.is_gqa
    assert dsa.q_head_dim == 4


def test_attention_config_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="mode"):
        AttentionConfig(d_model=16, num_q_heads=4, mode="dense")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="divisible"):
        AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=3)
    with pytest.raises(ValueError, match="sliding_window"):
        AttentionConfig(d_model=16, num_q_heads=4, sliding_window=0)
    with pytest.raises(ValueError, match="sparse_topk"):
        AttentionConfig(d_model=16, num_q_heads=4, sparse_topk=0)


def test_causal_sdpa_mask_is_boolean_and_sliding_windowed() -> None:
    mask = causal_sdpa_mask(5, sliding_window=2, expand_heads=True)
    mx.eval(mask)

    assert mask.shape == (1, 1, 5, 5)
    assert mask.dtype == mx.bool_
    np.testing.assert_array_equal(
        np.array(mask[0, 0]),
        np.array(
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [False, True, True, False, False],
                [False, False, True, True, False],
                [False, False, False, True, True],
            ],
            dtype=np.bool_,
        ),
    )


def test_causal_sdpa_mask_rejects_invalid_lengths() -> None:
    with pytest.raises(ValueError, match="seq_length"):
        causal_sdpa_mask(0)
    with pytest.raises(ValueError, match="key_length"):
        causal_sdpa_mask(4, key_length=0)
    with pytest.raises(ValueError, match="query_offset"):
        causal_sdpa_mask(4, query_offset=-1)
    with pytest.raises(ValueError, match="sliding_window"):
        causal_sdpa_mask(4, sliding_window=-1)


def test_causal_sdpa_mask_supports_cached_decode_offsets() -> None:
    mask = causal_sdpa_mask(
        2,
        query_offset=3,
        key_length=5,
        sliding_window=3,
        expand_heads=True,
    )
    mx.eval(mask)

    assert mask.shape == (1, 1, 2, 5)
    np.testing.assert_array_equal(
        np.array(mask[0, 0]),
        np.array(
            [
                [False, True, True, True, False],
                [False, False, True, True, True],
            ],
            dtype=np.bool_,
        ),
    )


def test_causal_sparse_indices_match_causal_topk_window() -> None:
    indices = causal_sparse_indices(2, 5, 2, 3)
    mx.eval(indices)

    assert indices.shape == (2, 5, 2, 3)
    assert indices.dtype == mx.int32
    np.testing.assert_array_equal(
        np.array(indices[0, :, 0, :]),
        np.array(
            [
                [0, -1, -1],
                [1, 0, -1],
                [2, 1, 0],
                [3, 2, 1],
                [4, 3, 2],
            ],
            dtype=np.int32,
        ),
    )


def test_sparse_indices_from_attention_mask_selects_newest_valid_keys() -> None:
    mask = mx.array(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, True, True, False],
            ]
        ],
        dtype=mx.bool_,
    )

    indices = sparse_indices_from_attention_mask(
        mask,
        batch_size=1,
        seq_length=3,
        kv_group=2,
        topk=2,
        key_length=4,
    )
    mx.eval(indices)

    assert indices.shape == (1, 3, 2, 2)
    np.testing.assert_array_equal(
        np.sort(np.array(indices[0, :, 0, :]), axis=-1),
        np.array([[-1, 0], [0, 1], [1, 2]], dtype=np.int32),
    )


def test_dsa_prepare_sparse_mla_fp8_emits_first_class_buffers() -> None:
    cfg = AttentionConfig(
        d_model=16,
        num_q_heads=4,
        num_kv_heads=2,
        mode="dsa",
        sparse_topk=3,
    )
    attn = CausalSelfAttention(cfg)
    x = _rand((2, 5, 16), seed=31)

    prepared = attn.prepare_sparse_mla_fp8(x)
    mx.eval(
        prepared.q_fp8,
        prepared.q_scale,
        prepared.kv_fp8,
        prepared.kv_scale,
        prepared.indices,
    )

    assert isinstance(prepared, SparseMLAFp8Prepared)
    assert prepared.q_fp8.shape == (2, 5, 4, 4)
    assert prepared.kv_fp8.shape == (2, 5, 2, 4)
    assert prepared.q_scale.shape == (2, 5, 4)
    assert prepared.kv_scale.shape == (2, 5, 2)
    assert prepared.indices.shape == (2, 5, 2, 3)
    assert prepared.q_fp8.dtype == mx.uint8
    assert prepared.kv_fp8.dtype == mx.uint8
    assert prepared.q_scale.dtype == mx.float32
    assert prepared.kv_scale.dtype == mx.float32
    assert prepared.indices.dtype == mx.int32
    assert prepared.d_v == cfg.q_head_dim
    assert prepared.sm_scale == pytest.approx(cfg.q_head_dim**-0.5)
    assert prepared.causal is True
    assert prepared.producer_owner == SPARSE_MLA_FP8_PRODUCER_OWNER
    assert prepared.producer_stage == SPARSE_MLA_FP8_PRODUCER_STAGE
    assert prepared.prepared_buffer_names == SPARSE_MLA_FP8_PREPARED_BUFFER_NAMES
    assert prepared.hidden_wrapper_quantization_allowed is False


def test_dsa_path_c_consumes_existing_prepared_buffers_without_wrapper_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as fp8_path_c

    cfg = AttentionConfig(
        d_model=16,
        num_q_heads=4,
        num_kv_heads=2,
        mode="dsa",
        sparse_topk=2,
    )
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 4, 16), seed=132)
    prepared = attn.prepare_sparse_mla_fp8(x)
    calls: list[dict[str, object]] = []

    def fail_quantization(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("kernel-boundary quantization must not run")

    def fake_apply(
        q_fp8: mx.array,
        q_scale: mx.array,
        kv_fp8: mx.array,
        kv_scale: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
    ) -> mx.array:
        del sm_scale, d_v, sinks, return_lse
        calls.append(
            {
                "q_fp8": q_fp8,
                "q_scale": q_scale,
                "kv_fp8": kv_fp8,
                "kv_scale": kv_scale,
                "indices": indices,
                "force_path_c": force_path_c,
            }
        )
        return mx.zeros((1, 4, 4, cfg.q_head_dim), dtype=mx.float16)

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_c")
    monkeypatch.setattr(
        fp8_path_c, "_to_fp8_with_per_token_scale", fail_quantization
    )
    monkeypatch.setattr(fp8_path_c, "sparse_mla_fp8_path_c_apply", fake_apply)
    monkeypatch.setattr(
        attn,
        "prepare_sparse_mla_fp8",
        lambda *args, **kwargs: prepared,
    )

    out = attn(x, mask="causal")
    mx.eval(out)

    assert out.shape == x.shape
    assert len(calls) == 1
    call = calls[0]
    assert call["q_fp8"] is prepared.q_fp8
    assert call["q_scale"] is prepared.q_scale
    assert call["kv_fp8"] is prepared.kv_fp8
    assert call["kv_scale"] is prepared.kv_scale
    assert call["indices"] is prepared.indices
    assert call["force_path_c"] is True


def test_dsa_path_c_routes_through_sparse_mla_fp8_prepared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as fp8_path_c

    cfg = AttentionConfig(
        d_model=16,
        num_q_heads=4,
        num_kv_heads=2,
        mode="dsa",
        sparse_topk=2,
    )
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 4, 16), seed=32)
    calls: list[dict[str, object]] = []

    def fake_apply(
        q_fp8: mx.array,
        q_scale: mx.array,
        kv_fp8: mx.array,
        kv_scale: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
    ) -> mx.array:
        assert sinks is None
        del return_lse
        calls.append(
            {
                "q_fp8": q_fp8,
                "q_scale": q_scale,
                "kv_fp8": kv_fp8,
                "kv_scale": kv_scale,
                "indices": indices,
                "sm_scale": sm_scale,
                "force_path_c": force_path_c,
            }
        )
        assert d_v == cfg.q_head_dim
        return mx.zeros((1, 4, 4, cfg.q_head_dim), dtype=mx.float16)

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_c")
    monkeypatch.setattr(fp8_path_c, "sparse_mla_fp8_path_c_apply", fake_apply)
    clear_dispatch_log()

    out = attn(x, mask="causal")
    mx.eval(out)

    assert out.shape == x.shape
    assert len(calls) == 1
    call = calls[0]
    assert call["force_path_c"] is True
    assert call["q_fp8"].dtype == mx.uint8
    assert call["kv_fp8"].dtype == mx.uint8
    assert call["q_scale"].dtype == mx.float32
    assert call["kv_scale"].dtype == mx.float32
    assert call["indices"].dtype == mx.int32
    assert get_dispatch_log()[-1] == {
        "op_name": "sparse_mla",
        "path": "path_c",
        "kernel_used": "tilelang_fp8_prepared_path_c_fwd",
    }


def test_dsa_path_c_routes_explicit_masks_as_sparse_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as fp8_path_c

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_c")
    attn = CausalSelfAttention(
        AttentionConfig(
            d_model=16, num_q_heads=4, num_kv_heads=2, mode="dsa", sparse_topk=2
        )
    )
    x = _rand((1, 4, 16), seed=33)
    explicit_mask = mx.array(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
            ]
        ],
        dtype=mx.bool_,
    )
    seen_indices: list[mx.array] = []

    def fake_apply(
        q_fp8: mx.array,
        q_scale: mx.array,
        kv_fp8: mx.array,
        kv_scale: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
    ) -> mx.array:
        del q_scale, kv_fp8, kv_scale, sm_scale, sinks, return_lse
        assert force_path_c is True
        seen_indices.append(indices)
        return mx.zeros(
            (q_fp8.shape[0], q_fp8.shape[1], q_fp8.shape[2], d_v or q_fp8.shape[-1]),
            dtype=mx.float16,
        )

    monkeypatch.setattr(fp8_path_c, "sparse_mla_fp8_path_c_apply", fake_apply)

    out = attn(x, mask=explicit_mask)
    mx.eval(out, seen_indices[0])

    prepared = attn.prepare_sparse_mla_fp8(x, mask=explicit_mask)
    assert out.shape == x.shape
    assert prepared.causal is True
    np.testing.assert_array_equal(
        np.sort(np.array(seen_indices[0][0, :, 0, :]), axis=-1),
        np.array([[-1, 0], [0, 1], [1, 2], [2, 3]], dtype=np.int32),
    )


def test_dsa_path_c_routes_sinks_to_sparse_mla_fp8_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as fp8_path_c

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_c")
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="dsa")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 3, 16), seed=34)
    sinks = mx.array([0.0, 0.1, -0.2, 0.3], dtype=mx.float32)
    sink_calls: list[mx.array | None] = []

    def fake_apply(
        q_fp8: mx.array,
        q_scale: mx.array,
        kv_fp8: mx.array,
        kv_scale: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
    ) -> mx.array:
        del q_scale, kv_fp8, kv_scale, indices, sm_scale, return_lse
        assert force_path_c is True
        sink_calls.append(sinks)
        return mx.zeros(
            (q_fp8.shape[0], q_fp8.shape[1], q_fp8.shape[2], d_v or q_fp8.shape[-1]),
            dtype=mx.float16,
        )

    monkeypatch.setattr(fp8_path_c, "sparse_mla_fp8_path_c_apply", fake_apply)

    out = attn(x, sinks=sinks)
    mx.eval(out)

    assert out.shape == x.shape
    assert sink_calls == [sinks]


@pytest.mark.parametrize("mode", ["mla", "dsa"])
def test_causal_attention_output_shape_and_route_marker(mode: str) -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode=mode)  # type: ignore[arg-type]
    attn = CausalSelfAttention(cfg)
    x = _rand((2, 5, 16), seed=5)

    out = attn(x)
    mx.eval(out)

    assert out.shape == x.shape
    assert attn.route_info.mode == mode
    assert attn.route_info.backend == "mlx.fast.sdpa"
    assert not attn.route_info.sparse_reference


def test_causal_prefix_invariance_with_gqa() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    prefix = _rand((1, 4, 16), seed=1)
    suffix_a = _rand((1, 3, 16), seed=2)
    suffix_b = _rand((1, 3, 16), seed=3)

    out_a = attn(mx.concatenate([prefix, suffix_a], axis=1))
    out_b = attn(mx.concatenate([prefix, suffix_b], axis=1))
    mx.eval(out_a, out_b)

    np.testing.assert_allclose(
        np.array(out_a[:, : prefix.shape[1], :]),
        np.array(out_b[:, : prefix.shape[1], :]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_mask_sentinel_matches_default_mask() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 5, 16), seed=12)

    default_out = attn(x)
    sentinel_out = attn(x, mask="causal")
    mx.eval(default_out, sentinel_out)

    np.testing.assert_allclose(
        np.array(sentinel_out),
        np.array(default_out),
        atol=1e-6,
        rtol=1e-6,
    )


def test_explicit_boolean_mask_is_forwarded_without_string_comparison() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 5, 16), seed=13)
    explicit_mask = causal_sdpa_mask(5)

    explicit_out = attn(x, mask=explicit_mask)
    default_out = attn(x)
    mx.eval(explicit_out, default_out)

    np.testing.assert_allclose(
        np.array(explicit_out),
        np.array(default_out),
        atol=1e-6,
        rtol=1e-6,
    )


def test_sliding_window_attention_does_not_see_old_prefix_tokens() -> None:
    cfg = AttentionConfig(
        d_model=16,
        num_q_heads=4,
        num_kv_heads=2,
        mode="mla",
        sliding_window=3,
    )
    attn = CausalSelfAttention(cfg)
    prefix_a = _rand((1, 2, 16), seed=7)
    prefix_b = _rand((1, 2, 16), seed=8)
    local_tail = _rand((1, 3, 16), seed=9)

    out_a = attn(mx.concatenate([prefix_a, local_tail], axis=1))
    out_b = attn(mx.concatenate([prefix_b, local_tail], axis=1))
    mx.eval(out_a, out_b)

    np.testing.assert_allclose(
        np.array(out_a[:, -1, :]),
        np.array(out_b[:, -1, :]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_attention_contiguous_kv_cache_matches_full_prefix_last_token() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    prefix = _rand((1, 4, 16), seed=21)
    next_token = _rand((1, 1, 16), seed=22)
    full = attn(mx.concatenate([prefix, next_token], axis=1))

    cache = make_contiguous_kv_cache(
        ContiguousKVCacheConfig(
            num_layers=1,
            batch_size=1,
            num_kv_heads=2,
            head_dim=4,
            max_seq_len=8,
        )
    )
    _ = attn(prefix, kv_cache=cache, layer_idx=0)
    cached_next = attn(next_token, kv_cache=cache, layer_idx=0)
    mx.eval(full, cached_next)

    assert cache.position() == 5
    np.testing.assert_allclose(
        np.array(cached_next[:, -1, :]),
        np.array(full[:, -1, :]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_causal_attention_kv_cache_requires_layer_idx() -> None:
    attn = CausalSelfAttention(
        AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2)
    )
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=4,
    )

    with pytest.raises(ValueError, match="layer_idx"):
        attn(_rand((1, 2, 16), seed=23), kv_cache=cache)


def test_dsa_path_c_kv_cache_keeps_fp8_buffers_in_mlx_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cppmega_mlx.nn._tilelang.sparse_mla_fp8_path_c as fp8_path_c

    monkeypatch.setenv("CPPMEGA_KERNEL_PATH__SPARSE_MLA", "path_c")
    cfg = AttentionConfig(
        d_model=16,
        num_q_heads=4,
        num_kv_heads=2,
        mode="dsa",
        sparse_topk=2,
    )
    attn = CausalSelfAttention(cfg)
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
    )
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def fake_apply(
        q_fp8: mx.array,
        q_scale: mx.array,
        kv_fp8: mx.array,
        kv_scale: mx.array,
        indices: mx.array,
        *,
        sm_scale: float,
        d_v: int | None = None,
        sinks: mx.array | None = None,
        return_lse: bool = False,
        force_path_c: bool = False,
    ) -> mx.array:
        del q_scale, kv_scale, sm_scale, sinks, return_lse
        assert force_path_c is True
        calls.append((tuple(q_fp8.shape), tuple(kv_fp8.shape), tuple(indices.shape)))
        return mx.zeros(
            (q_fp8.shape[0], q_fp8.shape[1], q_fp8.shape[2], d_v or q_fp8.shape[-1]),
            dtype=mx.float16,
        )

    monkeypatch.setattr(fp8_path_c, "sparse_mla_fp8_path_c_apply", fake_apply)

    prefix = _rand((1, 3, 16), seed=35)
    next_token = _rand((1, 1, 16), seed=36)
    first = attn(prefix, mask="causal", kv_cache=cache, layer_idx=0)
    second = attn(next_token, mask="causal", kv_cache=cache, layer_idx=0)
    mx.eval(first, second)

    assert cache.position() == 4
    assert cache.layers[0].empty()
    assert cache.sparse_fp8_layers[0].offset == 4
    assert calls == [
        ((1, 3, 4, 4), (1, 3, 2, 4), (1, 3, 2, 2)),
        ((1, 1, 4, 4), (1, 4, 2, 4), (1, 1, 2, 2)),
    ]


@pytest.mark.parametrize("bits", [4, 8])
def test_causal_attention_uses_mlx_lm_quantized_kv_cache(bits: int) -> None:
    attn = CausalSelfAttention(
        AttentionConfig(d_model=256, num_q_heads=4, num_kv_heads=2)
    )
    cache = make_contiguous_kv_cache(
        num_layers=1,
        batch_size=1,
        num_kv_heads=2,
        head_dim=64,
        quantized=True,
        kv_group_size=64,
        kv_bits=bits,
    )
    x = _rand((1, 2, 256), seed=24 + bits)

    out = attn(x, kv_cache=cache, layer_idx=0)
    mx.eval(out)

    assert out.shape == x.shape
    assert cache.position() == 2
    assert np.isfinite(np.array(out)).all()


def test_attention_sinks_are_forwarded_to_fast_sdpa() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 4, 16), seed=10)
    sinks = mx.array([0.0, 0.25, -0.125, 0.5], dtype=mx.float32)

    out = attn(x, sinks=sinks)
    mx.eval(out)

    assert out.shape == x.shape
    assert np.isfinite(np.array(out)).all()


def test_attention_sink_shape_validation_fails_closed() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="mla")
    attn = CausalSelfAttention(cfg)
    x = _rand((1, 4, 16), seed=11)

    with pytest.raises(ValueError, match="one value per query head"):
        attn(x, sinks=mx.zeros((2,), dtype=mx.float32))
    with pytest.raises(ValueError, match="1D"):
        attn(x, sinks=mx.zeros((1, 4), dtype=mx.float32))


def test_attention_train_step_has_finite_loss_and_gradients() -> None:
    cfg = AttentionConfig(d_model=16, num_q_heads=4, num_kv_heads=2, mode="dsa")
    attn = CausalSelfAttention(cfg)
    opt = optim.Adam(learning_rate=1e-3)
    x = _rand((2, 6, 16), seed=6)

    def loss_fn(model: CausalSelfAttention, hidden: mx.array) -> mx.array:
        return mx.mean(model(hidden) ** 2)

    loss, grads = nn.value_and_grad(attn, loss_fn)(attn, x)
    opt.update(attn, grads)
    mx.eval(loss, grads, attn.parameters())

    assert np.isfinite(np.array(loss)).all()
    grad_arrays = list(_tree_arrays(grads))
    assert grad_arrays
    for grad in grad_arrays:
        assert np.isfinite(np.array(grad)).all()
