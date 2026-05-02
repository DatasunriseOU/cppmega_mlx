from __future__ import annotations

import importlib
import sys

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data import (
    LMTokenBatch,
    LocalTokenBatchDataset,
    TorchDataLoaderBridgeConfig,
    build_spawn_dataloader,
    iter_mlx_batches,
    synthetic_token_batch,
)
from cppmega_mlx.data import dataloader_bridge


def test_bridge_module_does_not_import_torch_on_package_import() -> None:
    sys.modules.pop("torch", None)
    sys.modules.pop("torch.utils.data", None)

    importlib.reload(dataloader_bridge)

    assert "torch" not in sys.modules
    assert "torch.utils.data" not in sys.modules


def test_local_token_batch_dataset_materializes_numpy_batches() -> None:
    batch = synthetic_token_batch(batch_size=2, seq_length=4, include_structure=True)
    dataset = LocalTokenBatchDataset([batch])

    sample = dataset[0]

    assert len(dataset) == 1
    assert set(sample) == {
        "tokens",
        "attention_mask",
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
    }
    assert sample["tokens"].dtype == np.int32
    assert sample["attention_mask"].dtype == np.float32
    np.testing.assert_array_equal(sample["tokens"], np.array(batch.tokens))


def test_bridge_accepts_mapping_and_mlx_array_batches() -> None:
    dataset = LocalTokenBatchDataset(
        [
            {"tokens": np.arange(8, dtype=np.int64).reshape(2, 4)},
            mx.array(np.arange(8, 16, dtype=np.int32).reshape(2, 4)),
        ]
    )

    assert dataset[0]["tokens"].dtype == np.int32
    np.testing.assert_array_equal(
        dataset[1]["tokens"],
        np.arange(8, 16, dtype=np.int32).reshape(2, 4),
    )


def test_bridge_fails_closed_on_bad_batch_schema() -> None:
    with pytest.raises(ValueError, match="must include 'tokens'"):
        LocalTokenBatchDataset([{"attention_mask": np.ones((2, 4), dtype=np.float32)}])

    with pytest.raises(ValueError, match="unsupported DataLoader bridge batch keys"):
        LocalTokenBatchDataset(
            [
                {
                    "tokens": np.arange(8, dtype=np.int32).reshape(2, 4),
                    "document_ids": np.zeros((2, 4), dtype=np.int32),
                }
            ]
        )

    with pytest.raises(ValueError, match="structure_ids must match tokens shape"):
        LocalTokenBatchDataset(
            [
                {
                    "tokens": np.arange(8, dtype=np.int32).reshape(2, 4),
                    "structure_ids": np.arange(4, dtype=np.int32).reshape(1, 4),
                }
            ]
        )

    with pytest.raises(ValueError, match="tokens IDs must use an integer dtype"):
        LocalTokenBatchDataset([{"tokens": np.arange(8, dtype=np.float32).reshape(2, 4)}])


def test_spawn_only_multiprocessing_validation_happens_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataloader_bridge, "_load_torch_dataloader", _torch_must_not_load)
    batch = synthetic_token_batch(batch_size=1, seq_length=4)

    with pytest.raises(ValueError, match="multiprocessing_context must be 'spawn'"):
        build_spawn_dataloader(
            [batch],
            num_workers=1,
            multiprocessing_context="fork",
        )


def test_bridge_rejects_incompatible_worker_options_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataloader_bridge, "_load_torch_dataloader", _torch_must_not_load)
    batch = synthetic_token_batch(batch_size=1, seq_length=4)

    with pytest.raises(ValueError, match="persistent_workers requires num_workers > 0"):
        build_spawn_dataloader([batch], persistent_workers=True)

    with pytest.raises(ValueError, match="prefetch_factor requires num_workers > 0"):
        build_spawn_dataloader([batch], prefetch_factor=2)

    with pytest.raises(ValueError, match="num_workers must be non-negative"):
        build_spawn_dataloader([batch], num_workers=-1)


def test_bridge_fails_closed_when_torch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataloader_bridge, "is_torch_dataloader_available", lambda: False)

    with pytest.raises(
        dataloader_bridge.TorchDataLoaderBridgeError,
        match="requires optional dependency 'torch'",
    ):
        build_spawn_dataloader([synthetic_token_batch(batch_size=1, seq_length=4)])


def test_torch_dataloader_bridge_round_trips_to_lm_token_batch() -> None:
    pytest.importorskip("torch")
    source = LMTokenBatch(
        tokens=mx.array(np.arange(8, dtype=np.int32).reshape(2, 4)),
        attention_mask=mx.ones((2, 4), dtype=mx.float32),
    )
    loader = build_spawn_dataloader([source], config=TorchDataLoaderBridgeConfig())

    [batch] = list(iter_mlx_batches(loader))

    np.testing.assert_array_equal(np.array(batch.tokens), np.array(source.tokens))
    assert batch.attention_mask is not None
    np.testing.assert_array_equal(
        np.array(batch.attention_mask),
        np.ones((2, 4), dtype=np.float32),
    )


def test_data_loader_bridge_exports_are_public() -> None:
    import cppmega_mlx.data as data

    expected_exports = {
        "LocalTokenBatchDataset",
        "TorchDataLoaderBridgeConfig",
        "TorchDataLoaderBridgeError",
        "build_spawn_dataloader",
        "is_torch_dataloader_available",
        "iter_mlx_batches",
    }

    assert expected_exports <= set(data.__all__)
    assert data.LocalTokenBatchDataset is LocalTokenBatchDataset
    assert callable(data.build_spawn_dataloader)
    assert callable(data.iter_mlx_batches)


def _torch_must_not_load() -> object:
    raise AssertionError("torch should not be imported for fail-closed validation")
