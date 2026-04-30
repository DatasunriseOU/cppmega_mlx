from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.training import mlx_lm_adapter
from cppmega_mlx.training.mlx_lm_adapter import (
    MLX_LM_DENSE_BATCH_KEYS,
    MLXLMTrainerIntegrationUnsupported,
    REQUIRED_TRAINER_PARAMETERS,
    TRAINER_API_NAMES,
    as_mlx_lm_loss_args,
    as_mlx_lm_token_mapping,
    describe_mlx_lm_batch_route_metadata,
    describe_mlx_lm_trainer_apis,
    require_supported_mlx_lm_trainer_integration,
)


def test_describe_mlx_lm_trainer_apis_reports_installed_surface() -> None:
    info = describe_mlx_lm_trainer_apis()

    assert info.module == "mlx_lm.tuner.trainer"
    assert info.error is None
    assert info.module_file is not None
    assert info.module_file.endswith("mlx_lm/tuner/trainer.py")
    assert info.api_signatures is not None
    assert not info.missing_apis
    assert not info.incompatible_apis
    assert not info.compatibility_errors
    assert info.available is True
    assert info.package_versions is not None
    assert info.package_versions["mlx-lm"].startswith("0.31.")
    assert set(info.api_signatures) == set(TRAINER_API_NAMES)
    for api_name, required_parameters in REQUIRED_TRAINER_PARAMETERS.items():
        for parameter in required_parameters:
            assert parameter in info.api_signatures[api_name]


def test_describe_mlx_lm_trainer_apis_fails_gracefully_when_import_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_import_module(name: str):
        if name == "mlx_lm.tuner.trainer":
            raise ModuleNotFoundError("simulated missing trainer")
        return __import__(name)

    monkeypatch.setattr(mlx_lm_adapter.importlib, "import_module", blocked_import_module)

    info = describe_mlx_lm_trainer_apis()

    assert info.available is False
    assert info.api_signatures == {}
    assert info.missing_apis == TRAINER_API_NAMES
    assert info.incompatible_apis == ()
    assert info.package_versions is not None
    assert info.error is not None
    assert "simulated missing trainer" in info.error


def test_describe_mlx_lm_trainer_apis_fails_closed_when_signatures_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = ModuleType("mlx_lm.tuner.trainer")

    @dataclass
    class TrainingArgs:
        batch_size: int = 4
        max_seq_length: int = 2048

    def default_loss(model, tokens):
        return model, tokens

    def iterate_batches(dataset, batch_size):
        return iter(())

    def evaluate(model, dataset, batch_size, num_batches):
        return 0.0

    def train(model, optimizer, dataset):
        return None

    setattr(trainer, "TrainingArgs", TrainingArgs)
    setattr(trainer, "default_loss", default_loss)
    setattr(trainer, "iterate_batches", iterate_batches)
    setattr(trainer, "evaluate", evaluate)
    setattr(trainer, "train", train)
    setattr(trainer, "__file__", "/fake/mlx_lm/tuner/trainer.py")

    def fake_import_module(name: str) -> ModuleType:
        if name == "mlx_lm.tuner.trainer":
            return trainer
        raise AssertionError(name)

    monkeypatch.setattr(mlx_lm_adapter.importlib, "import_module", fake_import_module)

    info = describe_mlx_lm_trainer_apis()

    assert info.available is False
    assert info.missing_apis == ()
    assert set(info.incompatible_apis) == {
        "TrainingArgs",
        "default_loss",
        "iterate_batches",
        "evaluate",
        "train",
    }
    assert any(
        "default_loss missing required parameter(s): batch, lengths" in error
        for error in info.compatibility_errors
    )
    assert any(
        "train missing required parameter(s): train_dataset, args, loss, iterate_batches" in error
        for error in info.compatibility_errors
    )


def test_as_mlx_lm_token_mapping_keeps_only_dense_token_contract() -> None:
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=32,
        seed=11,
        include_structure=True,
    )

    mapping = as_mlx_lm_token_mapping(batch, offset=1)
    tokens, lengths = as_mlx_lm_loss_args(batch.as_dict(), offset=1)
    mx.eval(mapping["tokens"], mapping["lengths"], tokens, lengths)

    assert tuple(mapping) == MLX_LM_DENSE_BATCH_KEYS
    np.testing.assert_array_equal(np.array(mapping["tokens"]), np.array(batch.tokens))
    np.testing.assert_array_equal(np.array(tokens), np.array(batch.tokens))
    np.testing.assert_array_equal(
        np.array(mapping["lengths"]),
        np.array([[1, 6], [1, 6]], dtype=np.int32),
    )
    np.testing.assert_array_equal(np.array(lengths), np.array(mapping["lengths"]))
    assert "structure_ids" not in mapping
    assert "attention_mask" not in mapping


def test_as_mlx_lm_loss_args_match_installed_default_loss_contract() -> None:
    batch = synthetic_token_batch(batch_size=2, seq_length=5, seed=17)

    tokens, lengths = as_mlx_lm_loss_args(batch, offset=[1, 3])
    mx.eval(tokens, lengths)

    assert tokens.shape == (2, 5)
    assert lengths.shape == (2, 2)
    assert tokens.dtype == mx.int32
    assert lengths.dtype == mx.int32
    np.testing.assert_array_equal(
        np.array(lengths),
        np.array([[1, 5], [3, 5]], dtype=np.int32),
    )


def test_as_mlx_lm_token_mapping_accepts_plain_arrays_and_per_row_offsets() -> None:
    tokens = mx.array(np.arange(12, dtype=np.int32).reshape(2, 6))
    mapping = as_mlx_lm_token_mapping(tokens, offset=[0, 2])
    mx.eval(mapping["tokens"], mapping["lengths"])

    np.testing.assert_array_equal(np.array(mapping["tokens"]), np.arange(12).reshape(2, 6))
    np.testing.assert_array_equal(
        np.array(mapping["lengths"]),
        np.array([[0, 6], [2, 6]], dtype=np.int32),
    )


@pytest.mark.parametrize("offset", [-1, [-1, 0], [0]])
def test_as_mlx_lm_token_mapping_rejects_bad_offsets(offset: object) -> None:
    tokens = mx.array(np.arange(12, dtype=np.int32).reshape(2, 6))
    with pytest.raises(ValueError, match="offset"):
        as_mlx_lm_token_mapping(tokens, offset=offset)  # type: ignore[arg-type]


def test_as_mlx_lm_token_mapping_requires_int32_tokens() -> None:
    tokens = mx.array(np.arange(12, dtype=np.int64).reshape(2, 6))

    with pytest.raises(TypeError, match="requires int32 tokens"):
        as_mlx_lm_token_mapping(tokens)


def test_describe_mlx_lm_batch_route_metadata_reports_dropped_side_channels() -> None:
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=6,
        vocab_size=32,
        seed=13,
        include_structure=True,
    )

    metadata = describe_mlx_lm_batch_route_metadata(
        batch.as_dict(),
        route_symbols="AEMR",
        route_roles=("attention", "expert", "mamba", "recurrent"),
    )

    assert metadata.token_shape == (2, 6)
    assert metadata.token_dtype == "int32"
    assert metadata.route_symbols == ("A", "E", "M", "R")
    assert metadata.route_roles == ("attention", "expert", "mamba", "recurrent")
    assert metadata.has_attention_mask is True
    assert metadata.structure_fields == (
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
    )
    assert metadata.dropped_fields == ("attention_mask", *metadata.structure_fields)


def test_describe_mlx_lm_batch_route_metadata_can_read_model_route_attributes() -> None:
    class RouteModel:
        route_symbols = ("A", "E")
        route_roles = ("attention", "expert")

    batch = synthetic_token_batch(batch_size=1, seq_length=4, include_structure=False)

    metadata = describe_mlx_lm_batch_route_metadata(batch, model=RouteModel())

    assert metadata.route_symbols == ("A", "E")
    assert metadata.route_roles == ("attention", "expert")
    assert metadata.dropped_fields == ("attention_mask",)
    assert metadata.structure_fields == ()


def test_describe_mlx_lm_batch_route_metadata_preserves_single_role_string() -> None:
    batch = synthetic_token_batch(batch_size=1, seq_length=4, include_structure=False)

    metadata = describe_mlx_lm_batch_route_metadata(
        batch,
        route_symbols="A",
        route_roles="attention",
    )

    assert metadata.route_symbols == ("A",)
    assert metadata.route_roles == ("attention",)


def test_require_supported_mlx_lm_trainer_integration_fails_closed() -> None:
    batch = synthetic_token_batch(
        batch_size=1,
        seq_length=4,
        include_structure=True,
    )

    with pytest.raises(
        MLXLMTrainerIntegrationUnsupported,
        match="mlx-lm trainer integration is unsupported",
    ) as exc_info:
        require_supported_mlx_lm_trainer_integration(
            batch,
            model=type("RouteModel", (), {"route_symbols": "MR"})(),
        )

    message = str(exc_info.value)
    assert "dropped_fields=attention_mask, structure_ids" in message
    assert "route_symbols=MR" in message
