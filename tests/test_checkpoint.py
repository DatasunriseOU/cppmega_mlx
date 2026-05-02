from __future__ import annotations

import json

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.models.tiny_lm import TinyLM, TinyLMConfig
from cppmega_mlx.runtime.seed import capture_rng_state
from cppmega_mlx.training.checkpoint import (
    FORMAT_NAME,
    FORMAT_VERSION,
    GRAD_ACCUM_NAME,
    METADATA_NAME,
    OPTIMIZER_NAME,
    RNG_MODE_NOT_SAVED,
    RNG_MODE_SEED,
    SHARD_INDEX_NAME,
    SHARDING_MODE_SINGLE_FILE,
    WEIGHTS_NAME,
    load_checkpoint,
    save_checkpoint,
)
from cppmega_mlx.training.compiled import CompiledPretrainingStep
from cppmega_mlx.training.loop import one_step_train


def _tiny_config() -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        max_seq_length=16,
        structure_vocab_size=16,
    )


def _hybrid_config() -> HybridTinyConfig:
    return HybridTinyConfig(
        vocab_size=32,
        hidden_size=16,
        pattern="AEMR",
        depth=4,
        dsa_a_layer_ranks=(0,),
        num_attention_heads=4,
        max_seq_length=8,
        structure_vocab_size=16,
        structure_components="all",
        structure_bottleneck_dim=16,
        structure_num_categories=11,
        structure_max_dep_level=8,
        structure_max_ast_depth=8,
        structure_max_sibling_index=8,
        structure_num_node_types=32,
        moe_num_experts=3,
        moe_top_k=2,
        moe_expert_hidden_size=24,
        moe_shared_expert_hidden_size=8,
        mamba_expand=2,
        mamba_head_dim=4,
        mamba_state_dim=4,
        mamba_groups=2,
        mamba_mimo_rank=2,
        mamba_is_mimo=True,
        mamba_conv_kernel=2,
        mamba_chunk_size=4,
        mamba_rope_fraction=0.5,
        m2rnn_k_head_dim=4,
        m2rnn_v_head_dim=4,
        m2rnn_num_q_heads=1,
        m2rnn_num_k_heads=1,
        m2rnn_num_v_heads=2,
        m2rnn_num_f_heads=2,
        m2rnn_num_weight_heads=1,
        m2rnn_chunk_size=4,
    )


def _flat_arrays(tree) -> dict[str, np.ndarray]:
    mx.eval(tree)
    return {name: np.array(value) for name, value in tree_flatten(tree)}


def _assert_tree_allclose(actual, expected, *, atol: float = 0.0) -> None:
    actual_flat = _flat_arrays(actual)
    expected_flat = _flat_arrays(expected)
    assert actual_flat.keys() == expected_flat.keys()
    for name, expected_value in expected_flat.items():
        np.testing.assert_allclose(
            actual_flat[name],
            expected_value,
            rtol=0,
            atol=atol,
        )


def _rewrite_manifest(checkpoint_path, update) -> dict:
    metadata_path = checkpoint_path / METADATA_NAME
    manifest = json.loads(metadata_path.read_text())
    update(manifest)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _add_manifest_only_tensor(summary: dict) -> None:
    tensors = list(summary["tensors"])
    tensors.append("not.in.file")
    summary.update({"num_tensors": len(tensors), "tensors": tensors})


def _replace_manifest_tensor(summary: dict) -> None:
    tensors = list(summary["tensors"])
    assert tensors
    tensors[0] = "not.in.file"
    summary.update({"num_tensors": len(tensors), "tensors": tensors})


def test_checkpoint_manifest_records_resume_contract(tmp_path) -> None:
    config = _tiny_config()
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=11,
        include_structure=True,
    )
    one_step_train(model, optimizer, batch)

    payload = save_checkpoint(
        model,
        tmp_path / "ckpt",
        optimizer=optimizer,
        metadata={
            "step": 1,
            "tokenizer_path": "tokenizers/synthetic.json",
            "tokenizer_name": "synthetic-token-ids",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
    )
    manifest = json.loads((tmp_path / "ckpt" / METADATA_NAME).read_text())

    assert payload == manifest
    assert manifest["format"] == FORMAT_NAME
    assert manifest["version"] == FORMAT_VERSION
    assert manifest["step"] == 1
    assert manifest["weights"] == WEIGHTS_NAME
    assert manifest["rng"] == {"mode": RNG_MODE_NOT_SAVED}
    assert manifest["sharding"] == {
        "mode": SHARDING_MODE_SINGLE_FILE,
        "num_shards": 1,
        "weights": [WEIGHTS_NAME],
        "index": None,
    }
    assert manifest["model_config"] == config.to_dict()
    assert manifest["optimizer"]["present"] is True
    assert manifest["optimizer"]["file"] == OPTIMIZER_NAME
    assert manifest["optimizer"]["num_tensors"] > 2
    assert "token_embedding.weight.m" in manifest["optimizer"]["tensors"]
    assert manifest["package_versions"]["mlx"] is not None
    assert manifest["package_versions"]["mlx-lm"] is not None
    assert manifest["package_versions"]["safetensors"] is not None
    assert manifest["tokenizer_contract"]["vocab_size"] == config.vocab_size
    assert manifest["tokenizer_contract"]["max_seq_length"] == config.max_seq_length
    assert manifest["tokenizer_contract"]["structure_vocab_size"] == config.structure_vocab_size
    assert manifest["tokenizer_contract"]["tokenizer_path"] == "tokenizers/synthetic.json"
    assert manifest["tokenizer_contract"]["tokenizer_name"] == "synthetic-token-ids"
    assert manifest["tokenizer_contract"]["bos_token_id"] == 1
    assert manifest["tokenizer_contract"]["eos_token_id"] == 2
    assert manifest["tokenizer_contract"]["pad_token_id"] == 0
    assert (tmp_path / "ckpt" / WEIGHTS_NAME).exists()
    assert (tmp_path / "ckpt" / OPTIMIZER_NAME).exists()


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"format": "unknown"}, "unsupported format"),
        ({"version": FORMAT_VERSION + 1}, "unsupported version"),
        ({"step": "1"}, "step"),
        ({"trained_tokens": True}, "trained_tokens"),
        ({"tokenizer_contract": "local_profile"}, "tokenizer_contract must be an object"),
        ({"tokenizer_contract": {"vocab_size": -1}}, "tokenizer_contract.vocab_size"),
        ({"batch_cursor": "cursor"}, "batch_cursor must be an object"),
        (
            {"batch_cursor": {"epoch": 0, "batch_offset": 0}},
            "batch_cursor.global_batch_offset is required",
        ),
        (
            {
                "batch_cursor": {
                    "epoch": 0,
                    "batch_offset": -1,
                    "global_batch_offset": 0,
                }
            },
            "batch_cursor.batch_offset",
        ),
    ],
)
def test_checkpoint_load_validates_resume_metadata(tmp_path, override, error) -> None:
    config = _tiny_config()
    model = TinyLM(config)
    checkpoint_path = tmp_path / "ckpt"
    save_checkpoint(
        model,
        checkpoint_path,
        metadata={
            "step": 1,
            "batch_cursor": {
                "epoch": 0,
                "batch_offset": 1,
                "global_batch_offset": 1,
            },
        },
    )
    metadata_path = checkpoint_path / METADATA_NAME
    manifest = json.loads(metadata_path.read_text())
    manifest.update(override)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=error):
        load_checkpoint(TinyLM(config), checkpoint_path)


@pytest.mark.parametrize(
    ("training_state_override", "error"),
    [
        ({"compiled": "false"}, "training_state.compiled"),
        (
            {"gradient_accumulator": {"present": "false"}},
            "training_state.gradient_accumulator.present",
        ),
    ],
)
def test_checkpoint_load_rejects_coerced_training_state_booleans(
    tmp_path,
    training_state_override,
    error,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "bad-training-state"
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=False)
    save_checkpoint(
        model,
        checkpoint_path,
        optimizer=optimizer,
        training_step=step,
    )
    metadata_path = checkpoint_path / METADATA_NAME
    manifest = json.loads(metadata_path.read_text())
    manifest["training_state"].update(training_state_override)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=error):
        load_checkpoint(
            TinyLM(config),
            checkpoint_path,
            optimizer=optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
            training_step=CompiledPretrainingStep(
                TinyLM(config),
                optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
                compile=False,
            ),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"tensor_model_parallel_size": 2},
        {"sharded_state_dict": {"embedding.weight": {"replica_id": [0, 1, 0]}}},
        {"parallelism": {"tensor_model_parallel_size": 2}},
        {"training_state": {"megatron_parallel_state": {"tp_rank": 0}}},
        {"training_state": {"state": {"step": 0}, "parallel_state": {"tp_rank": 0}}},
        {"training_state": {"state": {"step": 0}, "mtp_process": True}},
    ],
)
def test_checkpoint_save_rejects_explicit_distributed_or_sharded_metadata(
    tmp_path,
    metadata,
) -> None:
    checkpoint_path = tmp_path / "unsupported-megatron-state"

    with pytest.raises(ValueError, match="unsupported distributed/sharded"):
        save_checkpoint(TinyLM(_tiny_config()), checkpoint_path, metadata=metadata)

    assert not (checkpoint_path / WEIGHTS_NAME).exists()


@pytest.mark.parametrize(
    "override",
    [
        {"tensor_model_parallel_size": 2},
        {"parallelism": {"tensor_model_parallel_size": 2}},
        {"sharded_state_dict": {"embedding.weight": {"replica_id": [0, 1, 0]}}},
        {"training_state": {"megatron_parallel_state": {"tp_rank": 0}}},
        {"training_state": {"state": {"step": 0}, "parallel_state": {"tp_rank": 0}}},
        {"training_state": {"state": {"step": 0}, "pre_process": True, "mtp_process": True}},
    ],
)
def test_checkpoint_load_rejects_explicit_distributed_or_sharded_metadata(
    tmp_path,
    override,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "load-unsupported-megatron-state"
    save_checkpoint(TinyLM(config), checkpoint_path)
    _rewrite_manifest(checkpoint_path, lambda manifest: manifest.update(override))

    with pytest.raises(ValueError, match="unsupported distributed/sharded"):
        load_checkpoint(TinyLM(config), checkpoint_path)


def test_checkpoint_allows_architecture_metadata_that_resembles_parallel_config(
    tmp_path,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "architecture-metadata"
    architecture_metadata = {
        "model_config": {
            "moe": {"expert_model_parallel_size": 1},
            "mamba3": {"partition_sizes": [4, 4, 2, 2]},
        },
        "dataset": {
            "provenance": {
                "distributed": "source corpus was built by a distributed job",
            },
        },
    }

    manifest = save_checkpoint(
        TinyLM(config),
        checkpoint_path,
        metadata=architecture_metadata,
    )
    loaded = load_checkpoint(TinyLM(config), checkpoint_path)

    assert loaded == manifest
    assert loaded["model_config"] == config.to_dict()
    assert loaded["dataset"] == architecture_metadata["dataset"]


def test_checkpoint_load_rejects_unknown_training_state_fields(tmp_path) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "unknown-training-state"
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=False)
    save_checkpoint(
        model,
        checkpoint_path,
        optimizer=optimizer,
        training_step=step,
    )
    _rewrite_manifest(
        checkpoint_path,
        lambda manifest: manifest["training_state"].update({"trainer_epoch": 1}),
    )

    with pytest.raises(ValueError, match="unsupported training_state fields"):
        load_checkpoint(
            TinyLM(config),
            checkpoint_path,
            optimizer=optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
            training_step=CompiledPretrainingStep(
                TinyLM(config),
                optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
                compile=False,
            ),
        )


def test_checkpoint_reserved_metadata_contract_roundtrips(tmp_path) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "ckpt"
    saved = save_checkpoint(
        TinyLM(config),
        checkpoint_path,
        metadata={
            "format": "caller-format",
            "version": 99,
            "tokenizer_contract": {
                "vocab_size": 128,
                "max_seq_length": 8,
                "tokenizer_name": "caller-tokenizer",
            },
            "batch_cursor": {
                "epoch": 1,
                "batch_offset": 2,
                "global_batch_offset": 6,
            },
        },
    )

    loaded = load_checkpoint(TinyLM(config), checkpoint_path)

    assert saved == loaded
    assert loaded["format"] == FORMAT_NAME
    assert loaded["version"] == FORMAT_VERSION
    assert loaded["tokenizer_contract"]["vocab_size"] == 128
    assert loaded["tokenizer_contract"]["max_seq_length"] == 8
    assert loaded["tokenizer_contract"]["tokenizer_name"] == "caller-tokenizer"
    assert loaded["batch_cursor"]["global_batch_offset"] == 6


def test_checkpoint_evaluation_metadata_contract_roundtrips(tmp_path) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "eval-contract"
    evaluation = {
        "dataset": {"path": "validation.npz", "num_batches": 2},
        "requested_batches": 2,
        "planned_batches": 1,
        "evaluated_batches": 1,
        "metrics": {
            "loss": 1.25,
            "ntokens": 7,
            "batches": 1,
            "seconds": 0.125,
            "tokens_per_second": 56.0,
        },
        "iteration": 3,
        "val_loss": 1.25,
        "val_time": 0.125,
    }

    saved = save_checkpoint(
        TinyLM(config),
        checkpoint_path,
        metadata={"evaluation": evaluation},
    )
    loaded = load_checkpoint(TinyLM(config), checkpoint_path)

    assert loaded == saved
    assert loaded["evaluation"] == evaluation


@pytest.mark.parametrize(
    ("evaluation", "error"),
    [
        ({"requested_batches": "1"}, "evaluation.requested_batches"),
        ({"evaluated_batches": True}, "evaluation.evaluated_batches"),
        ({"val_loss": float("nan")}, "evaluation.val_loss"),
        ({"metrics": {"loss": -1.0}}, "evaluation.metrics.loss"),
        ({"metrics": {"ntokens": "7"}}, "evaluation.metrics.ntokens"),
    ],
)
def test_checkpoint_save_rejects_invalid_evaluation_metadata(
    tmp_path,
    evaluation,
    error,
) -> None:
    checkpoint_path = tmp_path / "bad-eval"

    with pytest.raises(ValueError, match=error):
        save_checkpoint(
            TinyLM(_tiny_config()),
            checkpoint_path,
            metadata={"evaluation": evaluation},
        )

    assert not (checkpoint_path / WEIGHTS_NAME).exists()


@pytest.mark.parametrize(
    ("evaluation", "error"),
    [
        ("done", "evaluation must be an object"),
        ({"planned_batches": -1}, "evaluation.planned_batches"),
        ({"val_time": "0.1"}, "evaluation.val_time"),
        ({"metrics": "loss=1.0"}, "evaluation.metrics must be an object"),
        ({"metrics": {"tokens_per_second": -1.0}}, "evaluation.metrics.tokens_per_second"),
    ],
)
def test_checkpoint_load_rejects_invalid_evaluation_metadata(
    tmp_path,
    evaluation,
    error,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "load-bad-eval"
    save_checkpoint(TinyLM(config), checkpoint_path)
    _rewrite_manifest(
        checkpoint_path,
        lambda manifest: manifest.update({"evaluation": evaluation}),
    )

    with pytest.raises(ValueError, match=error):
        load_checkpoint(TinyLM(config), checkpoint_path)


def test_checkpoint_rng_seed_and_single_file_sharding_metadata_roundtrip(tmp_path) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "rng-seed"
    saved = save_checkpoint(
        TinyLM(config),
        checkpoint_path,
        metadata={
            "rng": {
                "mode": RNG_MODE_SEED,
                "seed": 1234,
                "source": "cli --seed",
            },
            "sharding": {
                "mode": SHARDING_MODE_SINGLE_FILE,
                "num_shards": 1,
                "weights": [WEIGHTS_NAME],
                "index": None,
                "source": "cppmega_mlx.training.checkpoint",
            },
        },
    )

    loaded = load_checkpoint(TinyLM(config), checkpoint_path)

    assert loaded == saved
    assert loaded["rng"] == {
        "mode": RNG_MODE_SEED,
        "seed": 1234,
        "source": "cli --seed",
    }
    assert loaded["sharding"] == {
        "mode": SHARDING_MODE_SINGLE_FILE,
        "num_shards": 1,
        "weights": [WEIGHTS_NAME],
        "index": None,
        "source": "cppmega_mlx.training.checkpoint",
    }


def test_checkpoint_fails_closed_for_local_rng_snapshot_metadata(tmp_path) -> None:
    config = _tiny_config()
    rng_snapshot = capture_rng_state()
    save_path = tmp_path / "save-local-rng-snapshot"

    with pytest.raises(ValueError, match="unsupported rng fields"):
        save_checkpoint(TinyLM(config), save_path, metadata={"rng": rng_snapshot})

    assert not (save_path / WEIGHTS_NAME).exists()

    load_path = tmp_path / "load-local-rng-snapshot"
    save_checkpoint(TinyLM(config), load_path)
    _rewrite_manifest(load_path, lambda manifest: manifest.update({"rng": rng_snapshot}))

    with pytest.raises(ValueError, match="unsupported rng fields"):
        load_checkpoint(TinyLM(config), load_path)


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        ({"rng_state": {"state": [1, 2, 3]}}, "standalone RNG payloads"),
        ({"rng": {"mode": "state", "state": [1, 2, 3]}}, "standalone RNG payloads"),
        ({"rng": {"mode": RNG_MODE_SEED}}, "rng.seed is required"),
        ({"rng": {"mode": RNG_MODE_SEED, "seed": -1}}, "rng.seed"),
    ],
)
def test_checkpoint_save_fails_closed_for_unsupported_rng_payloads(
    tmp_path,
    metadata,
    error,
) -> None:
    checkpoint_path = tmp_path / "bad-rng"

    with pytest.raises(ValueError, match=error):
        save_checkpoint(TinyLM(_tiny_config()), checkpoint_path, metadata=metadata)

    assert not (checkpoint_path / WEIGHTS_NAME).exists()


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        ({"sharding": "single"}, "sharding must be an object"),
        ({"sharding": {"mode": "mlx_lm_index"}}, "unsupported sharding.mode"),
        ({"sharding": {"mode": SHARDING_MODE_SINGLE_FILE, "num_shards": 2}}, "num_shards"),
        (
            {"sharding": {"mode": SHARDING_MODE_SINGLE_FILE, "weights": ["model-00001.safetensors"]}},
            "sharding.weights",
        ),
        (
            {"sharding": {"mode": SHARDING_MODE_SINGLE_FILE, "index": SHARD_INDEX_NAME}},
            "sharding.index",
        ),
        (
            {"sharding": {"mode": SHARDING_MODE_SINGLE_FILE, "weight_map": {}}},
            "checkpoint sharding payloads",
        ),
    ],
)
def test_checkpoint_save_fails_closed_for_unsupported_sharding_requests(
    tmp_path,
    metadata,
    error,
) -> None:
    checkpoint_path = tmp_path / "bad-shard"

    with pytest.raises(ValueError, match=error):
        save_checkpoint(TinyLM(_tiny_config()), checkpoint_path, metadata=metadata)

    assert not (checkpoint_path / WEIGHTS_NAME).exists()


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"rng": {"mode": "state", "payload": [1, 2, 3]}}, "standalone RNG payloads"),
        ({"rng": {"mode": RNG_MODE_SEED, "seed": True}}, "rng.seed"),
        ({"sharding": {"mode": "mlx_lm_index"}}, "unsupported sharding.mode"),
        ({"sharding": {"mode": SHARDING_MODE_SINGLE_FILE, "num_shards": 2}}, "num_shards"),
        (
            {"sharding": {"mode": SHARDING_MODE_SINGLE_FILE, "weight_map": {}}},
            "checkpoint sharding payloads",
        ),
    ],
)
def test_checkpoint_load_fails_closed_for_unsupported_rng_and_sharding_metadata(
    tmp_path,
    override,
    error,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "load-bad-policy"
    save_checkpoint(TinyLM(config), checkpoint_path)
    metadata_path = checkpoint_path / METADATA_NAME
    manifest = json.loads(metadata_path.read_text())
    manifest.update(override)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=error):
        load_checkpoint(TinyLM(config), checkpoint_path)


def test_checkpoint_load_rejects_mlx_lm_shard_index_without_single_weight_file(
    tmp_path,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "mlx-lm-sharded"
    save_checkpoint(TinyLM(config), checkpoint_path)
    (checkpoint_path / WEIGHTS_NAME).unlink()
    (checkpoint_path / SHARD_INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"token_embedding.weight": "model-00001-of-00002.safetensors"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="sharded checkpoint layout"):
        load_checkpoint(TinyLM(config), checkpoint_path)


def test_checkpoint_load_fails_closed_without_model_weights(tmp_path) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "missing-weights"
    save_checkpoint(TinyLM(config), checkpoint_path)
    (checkpoint_path / WEIGHTS_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="No model weights found"):
        load_checkpoint(TinyLM(config), checkpoint_path)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"optimizer": "adamw"}, "optimizer must be an object"),
        ({"optimizer": {"present": "yes"}}, "optimizer.present"),
        (
            {"optimizer": {"present": True, "file": None}},
            "optimizer.file must be a string",
        ),
        (
            {"optimizer": {"present": False, "file": OPTIMIZER_NAME}},
            "optimizer.file requires",
        ),
        ({"optimizer": {"present": False, "num_tensors": -1}}, "optimizer.num_tensors"),
        (
            {"optimizer": {"present": False, "tensors": [1]}},
            "optimizer.tensors",
        ),
    ],
)
def test_checkpoint_load_validates_optimizer_metadata(tmp_path, override, error) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "bad-optimizer-metadata"
    save_checkpoint(TinyLM(config), checkpoint_path)
    metadata_path = checkpoint_path / METADATA_NAME
    manifest = json.loads(metadata_path.read_text())
    manifest.update(override)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=error):
        load_checkpoint(TinyLM(config), checkpoint_path)


def test_checkpoint_load_fails_closed_when_optimizer_requested_but_metadata_absent(
    tmp_path,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "no-optimizer"
    save_checkpoint(TinyLM(config), checkpoint_path)

    with pytest.raises(FileNotFoundError, match="No optimizer state recorded"):
        load_checkpoint(
            TinyLM(config),
            checkpoint_path,
            optimizer=optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
        )


@pytest.mark.parametrize(
    ("mutate_manifest", "error"),
    [
        (lambda manifest: _add_manifest_only_tensor(manifest), "model tensor count mismatch"),
        (lambda manifest: _replace_manifest_tensor(manifest), "model tensor names mismatch"),
    ],
)
def test_checkpoint_load_rejects_model_tensor_metadata_mismatch(
    tmp_path,
    mutate_manifest,
    error,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "bad-model-summary"
    save_checkpoint(TinyLM(config), checkpoint_path)
    _rewrite_manifest(checkpoint_path, mutate_manifest)

    with pytest.raises(ValueError, match=error):
        load_checkpoint(TinyLM(config), checkpoint_path)


@pytest.mark.parametrize(
    ("mutate_manifest", "error"),
    [
        (
            lambda manifest: _add_manifest_only_tensor(manifest["optimizer"]),
            "optimizer tensor count mismatch",
        ),
        (
            lambda manifest: _replace_manifest_tensor(manifest["optimizer"]),
            "optimizer tensor names mismatch",
        ),
    ],
)
def test_checkpoint_load_rejects_optimizer_tensor_metadata_mismatch(
    tmp_path,
    mutate_manifest,
    error,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "bad-optimizer-summary"
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=25,
        include_structure=True,
    )
    one_step_train(model, optimizer, batch)
    save_checkpoint(model, checkpoint_path, optimizer=optimizer)
    _rewrite_manifest(checkpoint_path, mutate_manifest)

    with pytest.raises(ValueError, match=error):
        load_checkpoint(
            TinyLM(config),
            checkpoint_path,
            optimizer=optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
        )


def test_checkpoint_resume_restores_model_and_optimizer_state(tmp_path) -> None:
    config = _tiny_config()
    uninterrupted = TinyLM(config)
    uninterrupted_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    first_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=21,
        include_structure=True,
    )
    second_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=22,
        include_structure=True,
    )
    one_step_train(uninterrupted, uninterrupted_optimizer, first_batch)

    save_checkpoint(
        uninterrupted,
        tmp_path / "resume",
        optimizer=uninterrupted_optimizer,
        metadata={"step": 1},
    )
    resumed = TinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    metadata = load_checkpoint(
        resumed,
        tmp_path / "resume",
        optimizer=resumed_optimizer,
    )

    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=1e-7)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=1e-7,
    )
    assert metadata["step"] == 1

    expected_step = one_step_train(uninterrupted, uninterrupted_optimizer, second_batch)
    resumed_step = one_step_train(resumed, resumed_optimizer, second_batch)

    np.testing.assert_allclose(resumed_step.loss, expected_step.loss, rtol=0, atol=0)
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=1e-7)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=1e-7,
    )


def test_checkpoint_resume_next_step_matches_uninterrupted_with_cursor_and_rng_metadata(
    tmp_path,
) -> None:
    config = _tiny_config()
    rng_seed = 1307
    batches = [
        synthetic_token_batch(
            batch_size=2,
            seq_length=8,
            vocab_size=config.vocab_size,
            seed=121,
            include_structure=True,
        ),
        synthetic_token_batch(
            batch_size=2,
            seq_length=8,
            vocab_size=config.vocab_size,
            seed=122,
            include_structure=True,
        ),
    ]

    def make_stepper() -> tuple[TinyLM, optim.Optimizer, CompiledPretrainingStep]:
        mx.random.seed(rng_seed)
        model = TinyLM(config)
        optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
        mx.eval(model.state, optimizer.state)
        return model, optimizer, CompiledPretrainingStep(
            model,
            optimizer,
            compile=False,
        )

    uninterrupted, uninterrupted_optimizer, uninterrupted_step = make_stepper()
    interrupted, interrupted_optimizer, interrupted_step = make_stepper()

    uninterrupted_step(batches[0])
    interrupted_step(batches[0])

    cursor = {
        "epoch": 0,
        "batch_offset": 1,
        "global_batch_offset": 1,
    }
    rng_metadata = {
        "mode": RNG_MODE_SEED,
        "seed": rng_seed,
        "source": "test deterministic mx.random.seed",
    }
    manifest = save_checkpoint(
        interrupted,
        tmp_path / "cursor-rng-resume",
        optimizer=interrupted_optimizer,
        training_step=interrupted_step,
        metadata={
            "step": interrupted_step.state.step,
            "batch_cursor": cursor,
            "rng": rng_metadata,
        },
    )

    assert manifest["step"] == 1
    assert manifest["optimizer"]["present"] is True
    assert manifest["rng"] == rng_metadata
    assert manifest["batch_cursor"] == cursor
    assert manifest["training_state"]["state"] == {"step": 1, "trained_tokens": 14}
    assert manifest["training_state"]["grad_accum_steps"] == 1
    assert manifest["training_state"]["pending_microbatches"] == 0
    assert manifest["training_state"]["gradient_accumulator_present"] is False

    mx.random.seed(999)
    resumed = TinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    resumed_step = CompiledPretrainingStep(
        resumed,
        resumed_optimizer,
        compile=False,
    )
    loaded = load_checkpoint(
        resumed,
        tmp_path / "cursor-rng-resume",
        optimizer=resumed_optimizer,
        training_step=resumed_step,
    )

    assert loaded == manifest
    assert resumed_step.state.to_dict() == {"step": 1, "trained_tokens": 14}
    _assert_tree_allclose(resumed.parameters(), interrupted.parameters(), atol=0)
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=1e-7)
    _assert_tree_allclose(
        resumed_optimizer.state,
        interrupted_optimizer.state,
        atol=0,
    )
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=1e-7,
    )

    next_index = loaded["batch_cursor"]["global_batch_offset"]
    expected_metrics = uninterrupted_step(batches[next_index])
    resumed_metrics = resumed_step(batches[next_index])

    assert expected_metrics.updated is True
    assert resumed_metrics.updated is True
    assert resumed_metrics.step == expected_metrics.step == 2
    assert resumed_metrics.trained_tokens == expected_metrics.trained_tokens == 28
    np.testing.assert_allclose(resumed_metrics.loss, expected_metrics.loss, rtol=0, atol=0)
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=1e-7)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=1e-7,
    )


def test_checkpoint_resume_restores_hybrid_custom_blocks_and_optimizer(tmp_path) -> None:
    config = _hybrid_config()
    first_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=71,
        include_structure=True,
    )
    second_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=72,
        include_structure=True,
    )

    mx.random.seed(701)
    uninterrupted = HybridTinyLM(config)
    uninterrupted_optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    assert uninterrupted.route_roles == ("attention", "moe", "mamba3", "m2rnn")
    one_step_train(uninterrupted, uninterrupted_optimizer, first_batch)

    manifest = save_checkpoint(
        uninterrupted,
        tmp_path / "hybrid-resume",
        optimizer=uninterrupted_optimizer,
        metadata={"step": 1, "tokenizer_name": "hybrid-synthetic-token-ids"},
    )

    mx.random.seed(999)
    resumed = HybridTinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.0)
    metadata = load_checkpoint(
        resumed,
        tmp_path / "hybrid-resume",
        optimizer=resumed_optimizer,
    )

    assert metadata == manifest
    assert metadata["model_config"] == {
        **config.to_dict(),
        "dsa_a_layer_ranks": [0],
        "ngram_hash_orders": [2, 3],
    }
    assert metadata["tokenizer_contract"]["structure_vocab_size"] == config.structure_vocab_size
    assert "layers.1.block.router.gate.weight.m" in metadata["optimizer"]["tensors"]
    assert "layers.2.block.in_proj.weight.m" in metadata["optimizer"]["tensors"]
    assert "layers.3.block.state_weight.m" in metadata["optimizer"]["tensors"]
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=0)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=0,
    )

    expected_step = one_step_train(uninterrupted, uninterrupted_optimizer, second_batch)
    resumed_step = one_step_train(resumed, resumed_optimizer, second_batch)

    np.testing.assert_allclose(resumed_step.loss, expected_step.loss, rtol=0, atol=0)
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=1e-7)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=1e-7,
    )


def test_checkpoint_resume_restores_compiled_step_at_update_boundary(tmp_path) -> None:
    config = _tiny_config()
    first_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=31,
        include_structure=True,
    )
    second_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=32,
        include_structure=True,
    )

    def make_stepper() -> tuple[TinyLM, optim.Optimizer, CompiledPretrainingStep]:
        mx.random.seed(503)
        model = TinyLM(config)
        optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
        mx.eval(model.state, optimizer.state)
        return model, optimizer, CompiledPretrainingStep(
            model,
            optimizer,
            compile=True,
            grad_accum_steps=2,
        )

    uninterrupted, uninterrupted_optimizer, uninterrupted_step = make_stepper()
    interrupted, interrupted_optimizer, interrupted_step = make_stepper()

    uninterrupted_step(first_batch)
    uninterrupted_step(first_batch)
    interrupted_step(first_batch)
    interrupted_step(first_batch)

    manifest = save_checkpoint(
        interrupted,
        tmp_path / "compiled-resume",
        optimizer=interrupted_optimizer,
        training_step=interrupted_step,
    )
    assert manifest["training_state"]["state"] == {"step": 2, "trained_tokens": 28}
    assert manifest["training_state"]["pending_microbatches"] == 0
    assert manifest["training_state"]["gradient_accumulator_present"] is False
    assert manifest["training_state"]["gradient_accumulator"]["present"] is False
    assert not (tmp_path / "compiled-resume" / GRAD_ACCUM_NAME).exists()

    resumed = TinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    resumed_step = CompiledPretrainingStep(
        resumed,
        resumed_optimizer,
        compile=True,
        grad_accum_steps=2,
    )
    loaded = load_checkpoint(
        resumed,
        tmp_path / "compiled-resume",
        optimizer=resumed_optimizer,
        training_step=resumed_step,
    )

    assert loaded["training_state"] == manifest["training_state"]
    assert resumed_step.state.to_dict() == {"step": 2, "trained_tokens": 28}
    _assert_tree_allclose(resumed.parameters(), interrupted.parameters(), atol=0)
    _assert_tree_allclose(
        resumed_optimizer.state,
        interrupted_optimizer.state,
        atol=0,
    )
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=5e-6)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=5e-6,
    )

    expected_metrics = uninterrupted_step(second_batch)
    resumed_metrics = resumed_step(second_batch)

    assert expected_metrics.updated is False
    assert resumed_metrics.updated is False
    assert resumed_metrics.step == expected_metrics.step == 3
    assert resumed_metrics.trained_tokens == expected_metrics.trained_tokens == 42
    np.testing.assert_allclose(resumed_metrics.loss, expected_metrics.loss, rtol=0, atol=5e-6)
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=5e-6)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=5e-6,
    )


def test_checkpoint_resume_restores_compiled_step_mid_accumulation(tmp_path) -> None:
    config = _tiny_config()
    first_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=41,
        include_structure=True,
    )
    second_batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=42,
        include_structure=True,
    )

    def make_stepper() -> tuple[TinyLM, optim.Optimizer, CompiledPretrainingStep]:
        mx.random.seed(607)
        model = TinyLM(config)
        optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
        mx.eval(model.state, optimizer.state)
        return model, optimizer, CompiledPretrainingStep(
            model,
            optimizer,
            compile=True,
            grad_accum_steps=2,
        )

    uninterrupted, uninterrupted_optimizer, uninterrupted_step = make_stepper()
    interrupted, interrupted_optimizer, interrupted_step = make_stepper()

    uninterrupted_step(first_batch)
    interrupted_step(first_batch)

    manifest = save_checkpoint(
        interrupted,
        tmp_path / "compiled-mid-accum",
        optimizer=interrupted_optimizer,
        training_step=interrupted_step,
    )
    assert manifest["training_state"]["state"] == {"step": 1, "trained_tokens": 14}
    assert manifest["training_state"]["pending_microbatches"] == 1
    assert manifest["training_state"]["gradient_accumulator_present"] is True
    assert manifest["training_state"]["gradient_accumulator"]["present"] is True
    assert manifest["training_state"]["gradient_accumulator"]["file"] == GRAD_ACCUM_NAME
    assert (tmp_path / "compiled-mid-accum" / GRAD_ACCUM_NAME).exists()

    resumed = TinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    resumed_step = CompiledPretrainingStep(
        resumed,
        resumed_optimizer,
        compile=True,
        grad_accum_steps=2,
    )
    load_checkpoint(
        resumed,
        tmp_path / "compiled-mid-accum",
        optimizer=resumed_optimizer,
        training_step=resumed_step,
    )

    expected_metrics = uninterrupted_step(second_batch)
    resumed_metrics = resumed_step(second_batch)

    assert expected_metrics.updated is True
    assert resumed_metrics.updated is True
    assert resumed_metrics.step == expected_metrics.step == 2
    assert resumed_metrics.trained_tokens == expected_metrics.trained_tokens == 28
    np.testing.assert_allclose(resumed_metrics.loss, expected_metrics.loss, rtol=0, atol=0)
    _assert_tree_allclose(resumed.parameters(), uninterrupted.parameters(), atol=1e-7)
    _assert_tree_allclose(
        resumed_optimizer.state,
        uninterrupted_optimizer.state,
        atol=1e-7,
    )


def test_checkpoint_resume_rejects_compiled_mode_mismatch(tmp_path) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "compiled-mode-mismatch"
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=True)
    save_checkpoint(
        model,
        checkpoint_path,
        optimizer=optimizer,
        training_step=step,
    )

    resumed = TinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    with pytest.raises(ValueError, match="training_state.compiled"):
        load_checkpoint(
            resumed,
            checkpoint_path,
            optimizer=resumed_optimizer,
            training_step=CompiledPretrainingStep(
                resumed,
                resumed_optimizer,
                compile=False,
            ),
        )


def test_checkpoint_resume_rejects_grad_accum_step_mismatch(tmp_path) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "grad-accum-mismatch"
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(model, optimizer, compile=False, grad_accum_steps=2)
    save_checkpoint(
        model,
        checkpoint_path,
        optimizer=optimizer,
        training_step=step,
    )

    resumed = TinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    with pytest.raises(ValueError, match="training_state.grad_accum_steps"):
        load_checkpoint(
            resumed,
            checkpoint_path,
            optimizer=resumed_optimizer,
            training_step=CompiledPretrainingStep(
                resumed,
                resumed_optimizer,
                compile=False,
                grad_accum_steps=1,
            ),
        )


@pytest.mark.parametrize(
    ("mutate_manifest", "error"),
    [
        (
            lambda manifest: _add_manifest_only_tensor(
                manifest["training_state"]["gradient_accumulator"]
            ),
            "training_state.gradient_accumulator tensor count mismatch",
        ),
        (
            lambda manifest: _replace_manifest_tensor(
                manifest["training_state"]["gradient_accumulator"]
            ),
            "training_state.gradient_accumulator tensor names mismatch",
        ),
    ],
)
def test_checkpoint_load_rejects_gradient_accumulator_metadata_mismatch(
    tmp_path,
    mutate_manifest,
    error,
) -> None:
    config = _tiny_config()
    checkpoint_path = tmp_path / "bad-grad-accum-summary"
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(
        model,
        optimizer,
        compile=False,
        grad_accum_steps=2,
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=45,
        include_structure=True,
    )
    step(batch)
    save_checkpoint(
        model,
        checkpoint_path,
        optimizer=optimizer,
        training_step=step,
    )
    _rewrite_manifest(checkpoint_path, mutate_manifest)

    resumed = TinyLM(config)
    resumed_optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    with pytest.raises(ValueError, match=error):
        load_checkpoint(
            resumed,
            checkpoint_path,
            optimizer=resumed_optimizer,
            training_step=CompiledPretrainingStep(
                resumed,
                resumed_optimizer,
                compile=False,
                grad_accum_steps=2,
            ),
        )


def test_checkpoint_mid_accumulation_resume_fails_closed_without_accumulator(
    tmp_path,
) -> None:
    config = _tiny_config()
    model = TinyLM(config)
    optimizer = optim.AdamW(learning_rate=1e-2, weight_decay=0.0)
    step = CompiledPretrainingStep(
        model,
        optimizer,
        compile=True,
        grad_accum_steps=2,
    )
    batch = synthetic_token_batch(
        batch_size=2,
        seq_length=8,
        vocab_size=config.vocab_size,
        seed=51,
        include_structure=True,
    )
    step(batch)

    save_checkpoint(
        model,
        tmp_path / "missing-accum",
        optimizer=optimizer,
        training_step=step,
    )
    (tmp_path / "missing-accum" / GRAD_ACCUM_NAME).unlink()

    with pytest.raises(FileNotFoundError, match="gradient accumulator"):
        load_checkpoint(
            TinyLM(config),
            tmp_path / "missing-accum",
            optimizer=optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
            training_step=CompiledPretrainingStep(
                TinyLM(config),
                optim.AdamW(learning_rate=1e-2, weight_decay=0.0),
                compile=True,
                grad_accum_steps=2,
            ),
        )
