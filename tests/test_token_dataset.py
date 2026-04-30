from __future__ import annotations

import numpy as np
import pytest
from tests.test_megatron_indexed import _write_mmididx

from cppmega_mlx.config.model import (
    LOCAL_PROFILE_VOCAB_SIZE,
    MEGACPP_TOKENIZER_VOCAB_SIZE,
)
from cppmega_mlx.data.megatron_indexed import MegatronIndexedDataset
from cppmega_mlx.data.parquet_dataset import TokenParquetDataset
from cppmega_mlx.data.token_dataset import TokenNpzDataset, open_token_dataset


def _write_npz(path, **arrays) -> None:
    np.savez(path, **arrays)


def test_npz_flat_tokens_yield_fixed_lm_batches(tmp_path) -> None:
    path = tmp_path / "tokens.npz"
    _write_npz(
        path,
        tokens=np.arange(24, dtype=np.int32),
        attention_mask=np.ones(24, dtype=np.float32),
        vocab_size=np.array(MEGACPP_TOKENIZER_VOCAB_SIZE, dtype=np.int64),
        tokenizer_contract=np.array("megacpp"),
    )

    dataset = TokenNpzDataset(path, seq_len=6, batch_size=2)
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 4
    assert dataset.num_batches == 2
    assert dataset.dropped_samples == 0
    assert batch.tokens.shape == (2, 6)
    np.testing.assert_array_equal(np.array(batch.tokens[0]), np.arange(6))
    np.testing.assert_array_equal(np.array(batch.inputs[0]), np.arange(5))
    np.testing.assert_array_equal(np.array(batch.targets[0]), np.arange(1, 6))
    assert dataset.metadata.vocab_size == MEGACPP_TOKENIZER_VOCAB_SIZE
    assert dataset.metadata.tokenizer_contract == "megacpp"
    assert dataset.metadata.local_profile_vocab_size == LOCAL_PROFILE_VOCAB_SIZE


def test_npz_2d_tokens_and_structure_channels_are_sliced_together(tmp_path) -> None:
    path = tmp_path / "structured.npz"
    tokens = np.arange(30, dtype=np.int32).reshape(2, 15)
    structure_ids = (tokens % 7).astype(np.int32)
    dep_levels = (tokens % 3).astype(np.int32)
    _write_npz(
        path,
        tokens=tokens,
        structure_ids=structure_ids,
        dep_levels=dep_levels,
        vocab_size=np.array(64, dtype=np.int64),
        tokenizer_contract=np.array("local_profile"),
    )

    dataset = TokenNpzDataset(path, seq_len=5, batch_size=3)
    batch = next(dataset.iter_batches())

    assert dataset.num_samples == 6
    assert batch.tokens.shape == (3, 5)
    assert batch.structure_ids is not None
    assert batch.dep_levels is not None
    np.testing.assert_array_equal(np.array(batch.structure_ids), np.array(batch.tokens) % 7)
    np.testing.assert_array_equal(np.array(batch.model_kwargs()["structure_ids"]), np.array(batch.tokens[:, :-1]) % 7)
    assert dataset.metadata.vocab_size == 64
    assert dataset.metadata.tokenizer_contract == "local_profile"


def test_shuffle_is_deterministic_and_resume_skips_consumed_batches(tmp_path) -> None:
    path = tmp_path / "shuffle.npz"
    _write_npz(path, tokens=np.arange(60, dtype=np.int32))

    left = TokenNpzDataset(path, seq_len=5, batch_size=2, shuffle=True, seed=123)
    right = TokenNpzDataset(path, seq_len=5, batch_size=2, shuffle=True, seed=123)
    other_seed = TokenNpzDataset(path, seq_len=5, batch_size=2, shuffle=True, seed=124)

    left_batches = list(left.iter_batches())
    right_batches = list(right.iter_batches())
    resumed = next(left.iter_batches(resume_batch=2))
    cursor = left.cursor_after(2)

    assert [np.array(b.tokens).tolist() for b in left_batches] == [
        np.array(b.tokens).tolist() for b in right_batches
    ]
    assert [np.array(b.tokens).tolist() for b in left_batches] != [
        np.array(b.tokens).tolist() for b in other_seed.iter_batches()
    ]
    np.testing.assert_array_equal(np.array(resumed.tokens), np.array(left_batches[2].tokens))
    assert cursor.epoch == 0
    assert cursor.batch_offset == 2
    assert cursor.global_batch_offset == 2


def test_cursor_after_includes_dataset_resume_batch_across_epoch_rollover(tmp_path) -> None:
    path = tmp_path / "resume_cursor.npz"
    _write_npz(path, tokens=np.arange(40, dtype=np.int32))

    dataset = TokenNpzDataset(
        path,
        seq_len=5,
        batch_size=2,
        shuffle=True,
        seed=7,
        loop=True,
        resume_batch=3,
    )
    stream = dataset.iter_batches()
    next(stream)
    next(stream)
    expected_next = next(stream)

    cursor = dataset.cursor_after(2)
    restored = TokenNpzDataset(
        path,
        seq_len=5,
        batch_size=2,
        shuffle=True,
        seed=7,
        loop=True,
    )
    actual_next = next(
        restored.iter_batches(
            resume_batch=cursor.batch_offset,
            epoch=cursor.epoch,
        )
    )

    assert cursor.epoch == 1
    assert cursor.batch_offset == 1
    assert cursor.global_batch_offset == 5
    np.testing.assert_array_equal(
        np.array(actual_next.tokens),
        np.array(expected_next.tokens),
    )


def test_loop_rolls_to_next_epoch_with_new_shuffle_order(tmp_path) -> None:
    path = tmp_path / "loop.npz"
    _write_npz(path, tokens=np.arange(40, dtype=np.int32))

    dataset = TokenNpzDataset(
        path, seq_len=5, batch_size=2, shuffle=True, seed=5, loop=True
    )
    batches = dataset.iter_batches()
    epoch0_first = next(batches)
    for _ in range(dataset.num_batches - 1):
        next(batches)
    epoch1_first = next(batches)

    np.testing.assert_array_equal(
        np.array(epoch0_first.tokens),
        dataset._tokens[dataset.sample_order(epoch=0)[:2]],
    )
    np.testing.assert_array_equal(
        np.array(epoch1_first.tokens),
        dataset._tokens[dataset.sample_order(epoch=1)[:2]],
    )


def test_open_token_dataset_preserves_existing_format_routing(
    tmp_path, monkeypatch
) -> None:
    npz_path = tmp_path / "tokens.npz"
    _write_npz(npz_path, tokens=np.arange(12, dtype=np.int32))
    parquet_path = tmp_path / "tokens.parquet"
    bin_path = tmp_path / "tokens.bin"
    np.arange(12, dtype=np.uint16).tofile(bin_path)

    assert isinstance(
        open_token_dataset(npz_path, seq_len=4, batch_size=1), TokenNpzDataset
    )
    monkeypatch.setattr(
        TokenParquetDataset,
        "__init__",
        lambda self, path, **kwargs: None,
    )
    assert isinstance(
        open_token_dataset(parquet_path, seq_len=4, batch_size=1),
        TokenParquetDataset,
    )
    assert isinstance(
        open_token_dataset(bin_path, seq_len=4, batch_size=1, dtype="uint16"),
        MegatronIndexedDataset,
    )


def test_open_token_dataset_routes_suffixless_megatron_prefix(tmp_path) -> None:
    prefix = tmp_path / "clang_semantic_4k_v10_train"
    _write_mmididx(prefix, [np.arange(12, dtype=np.int32)], dtype=np.int32)

    dataset = open_token_dataset(prefix, seq_len=4, batch_size=1)

    assert isinstance(dataset, MegatronIndexedDataset)
    assert dataset.path == prefix
    assert dataset.index_metadata.dtype == "int32"


def test_rejects_bad_shapes_and_incomplete_sample_sets(tmp_path) -> None:
    missing = tmp_path / "missing.npz"
    _write_npz(missing, ids=np.arange(8, dtype=np.int32))
    with pytest.raises(ValueError, match="tokens"):
        TokenNpzDataset(missing, seq_len=4, batch_size=1)

    too_short = tmp_path / "too_short.npz"
    _write_npz(too_short, tokens=np.arange(3, dtype=np.int32))
    with pytest.raises(ValueError, match="full fixed-shape"):
        TokenNpzDataset(too_short, seq_len=4, batch_size=1)

    bad_side_channel = tmp_path / "bad_side_channel.npz"
    _write_npz(
        bad_side_channel,
        tokens=np.arange(12, dtype=np.int32),
        structure_ids=np.arange(8, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="structure_ids"):
        TokenNpzDataset(bad_side_channel, seq_len=4, batch_size=1)


def test_npz_rejects_token_ids_outside_int32_range(tmp_path) -> None:
    too_large = tmp_path / "too_large_tokens.npz"
    _write_npz(
        too_large,
        tokens=np.array([0, np.iinfo(np.int32).max + 1, 2, 3], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="token IDs exceed int32 range"):
        TokenNpzDataset(too_large, seq_len=4, batch_size=1)

    negative = tmp_path / "negative_tokens.npz"
    _write_npz(negative, tokens=np.array([0, -1, 2, 3], dtype=np.int64))

    with pytest.raises(ValueError, match="token IDs must be non-negative"):
        TokenNpzDataset(negative, seq_len=4, batch_size=1)


def test_npz_rejects_non_integer_token_ids(tmp_path) -> None:
    path = tmp_path / "float_tokens.npz"
    _write_npz(path, tokens=np.arange(4, dtype=np.float32))

    with pytest.raises(ValueError, match="token IDs must use an integer dtype"):
        TokenNpzDataset(path, seq_len=4, batch_size=1)


def test_npz_rejects_structure_side_channels_outside_int32_range(tmp_path) -> None:
    path = tmp_path / "oversized_structure.npz"
    _write_npz(
        path,
        tokens=np.arange(4, dtype=np.int32),
        structure_ids=np.array(
            [0, np.iinfo(np.int32).max + 1, 2, 3],
            dtype=np.uint64,
        ),
    )

    with pytest.raises(ValueError, match="structure_ids side-channel IDs exceed int32 range"):
        TokenNpzDataset(path, seq_len=4, batch_size=1)


def test_npz_rejects_negative_structure_side_channels(tmp_path) -> None:
    path = tmp_path / "negative_structure.npz"
    _write_npz(
        path,
        tokens=np.arange(4, dtype=np.int32),
        structure_ids=np.array([0, 1, -1, 3], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="structure_ids side-channel IDs must be non-negative"):
        TokenNpzDataset(path, seq_len=4, batch_size=1)


def test_npz_rejects_non_integer_structure_side_channels(tmp_path) -> None:
    path = tmp_path / "float_structure.npz"
    _write_npz(
        path,
        tokens=np.arange(4, dtype=np.int32),
        structure_ids=np.arange(4, dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="structure_ids side-channel IDs must use an integer dtype",
    ):
        TokenNpzDataset(path, seq_len=4, batch_size=1)


def test_npz_ambiguous_side_channels_fail_closed(tmp_path) -> None:
    path = tmp_path / "ambiguous_side_channels.npz"
    _write_npz(
        path,
        tokens=np.arange(4, dtype=np.int32),
        side_channels=np.array(["structure_ids"]),
    )

    with pytest.raises(NotImplementedError, match="ambiguous keys: side_channels"):
        TokenNpzDataset(path, seq_len=4, batch_size=1)


@pytest.mark.parametrize(
    "key",
    ["ngram_ids", "ngram_hash", "ngram_hash_ids", "ngram_sidecar", "ngrams"],
)
def test_npz_ngram_sidecars_fail_closed(tmp_path, key: str) -> None:
    path = tmp_path / f"{key}.npz"
    _write_npz(
        path,
        tokens=np.arange(4, dtype=np.int32),
        **{key: np.arange(4, dtype=np.int32)},
    )

    with pytest.raises(NotImplementedError, match="ngram sidecars are not supported"):
        TokenNpzDataset(path, seq_len=4, batch_size=1)
