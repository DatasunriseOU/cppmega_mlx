from __future__ import annotations

import math

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig
from cppmega_mlx.training.checkpoint import load_checkpoint, save_checkpoint
from cppmega_mlx.training.loop import one_step_train
from cppmega_mlx.training.loss import next_token_cross_entropy


def _tiny_config() -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=48,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        max_seq_length=16,
        structure_vocab_size=16,
    )


def _flat_params(model: TinyLM) -> dict[str, np.ndarray]:
    mx.eval(model.parameters())
    return {name: np.array(value) for name, value in tree_flatten(model.parameters())}


def test_next_token_loss_accepts_structure_batch() -> None:
    model = TinyLM(_tiny_config())
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=7,
        vocab_size=model.config.vocab_size,
        seed=123,
        include_structure=True,
    )

    logits = model(batch.inputs, **batch.model_kwargs())
    loss, ntokens = next_token_cross_entropy(model, batch)
    mx.eval(logits, loss, ntokens)

    assert logits.shape == (2, 6, model.config.vocab_size)
    assert math.isfinite(float(loss.item()))
    assert int(ntokens.item()) == 12


def test_one_step_train_updates_parameters_on_gpu() -> None:
    model = TinyLM(_tiny_config())
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=model.config.vocab_size,
        seed=7,
        include_structure=True,
    )
    before = _flat_params(model)

    result = one_step_train(model, optimizer, batch)
    after = _flat_params(model)

    assert result.ntokens == 14
    assert math.isfinite(result.loss)
    assert result.loss > 0
    assert result.tokens_per_second > 0
    assert str(mx.default_device().type).endswith("gpu")
    assert any(np.max(np.abs(after[name] - before[name])) > 0 for name in before)


def test_checkpoint_roundtrip_restores_full_model_weights(tmp_path) -> None:
    config = _tiny_config()
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=9,
        include_structure=True,
    )
    step = one_step_train(model, optimizer, batch)

    metadata = save_checkpoint(
        model,
        tmp_path / "ckpt",
        metadata={"step": 1, "train_loss": step.loss},
    )
    restored = TinyLM(config)
    loaded_metadata = load_checkpoint(restored, tmp_path / "ckpt")

    original_params = _flat_params(model)
    restored_params = _flat_params(restored)
    for name, expected in original_params.items():
        np.testing.assert_allclose(restored_params[name], expected, rtol=0, atol=0)

    original_logits = model(batch.inputs, **batch.model_kwargs())
    restored_logits = restored(batch.inputs, **batch.model_kwargs())
    mx.eval(original_logits, restored_logits)
    np.testing.assert_allclose(
        np.array(restored_logits), np.array(original_logits), rtol=0, atol=0
    )
    assert metadata["format"] == "cppmega_mlx_checkpoint_v1"
    assert loaded_metadata["model_config"] == config.to_dict()
    assert loaded_metadata["step"] == 1
