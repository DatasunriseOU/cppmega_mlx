from __future__ import annotations

from pathlib import Path
from types import ModuleType

import cppmega_mlx.data as data
import cppmega_mlx.config as config
import cppmega_mlx.kernels as kernels
import cppmega_mlx.models as models
import cppmega_mlx.nn as nn
import cppmega_mlx.nn.m2rnn as m2rnn
import cppmega_mlx as package
import cppmega_mlx.recipes as recipes
import cppmega_mlx.training as training

PACKAGE_ROOTS = (package, config, data, kernels, models, nn, recipes, training)
FOREIGN_RUNTIME_ALIASES = {
    "CudaGraphTrainer",
    "DifferentiableMetalKernel",
    "DistributedDataParallel",
    "MegatronTrainer",
    "MetalTrainingKernel",
    "TensorParallelTrainer",
    "TransformerEngine",
    "TrainingArgs",
    "default_loss",
    "evaluate",
    "iterate_batches",
    "metal_train",
    "squared_relu_train_metal",
    "train",
    "train_with_metal",
}


def _assert_public_exports(module: ModuleType) -> set[str]:
    exports = module.__all__

    assert isinstance(exports, list)
    assert all(isinstance(name, str) for name in exports)
    assert len(exports) == len(set(exports))
    for name in exports:
        assert hasattr(module, name), f"{module.__name__}.{name} is listed but missing"
    return set(exports)


def test_package_exports_resolve_and_stay_unique() -> None:
    for module in PACKAGE_ROOTS:
        _assert_public_exports(module)


def test_package_exports_do_not_leak_foreign_runtime_aliases() -> None:
    for module in PACKAGE_ROOTS:
        exports = _assert_public_exports(module)
        leaked = exports & FOREIGN_RUNTIME_ALIASES
        assert leaked == set(), f"{module.__name__} leaked foreign runtime aliases: {sorted(leaked)}"
        for name in FOREIGN_RUNTIME_ALIASES:
            assert not hasattr(module, name), f"{module.__name__}.{name} must stay private/absent"


def test_package_root_exposes_subpackages_without_overclaiming_runtime() -> None:
    assert package.__version__ == "0.1.0"
    assert {
        "config",
        "data",
        "kernels",
        "models",
        "nn",
        "recipes",
        "training",
    } <= _assert_public_exports(package)
    assert package.config is config
    assert package.data is data
    assert package.kernels is kernels
    assert package.models is models
    assert package.nn is nn
    assert package.recipes is recipes
    assert package.training is training

    exports = _assert_public_exports(package)
    assert FOREIGN_RUNTIME_ALIASES.isdisjoint(exports)
    for name in FOREIGN_RUNTIME_ALIASES:
        assert not hasattr(package, name), name


def test_package_roots_expose_local_mlx_contracts() -> None:
    assert {
        "LMTokenBatch",
        "TokenNpzDataset",
        "TokenParquetDataset",
        "MegatronIndexedDataset",
        "open_megatron_indexed_dataset",
        "megatron_indexed_side_channel_schema",
    } <= _assert_public_exports(data)
    assert {
        "Mamba3CacheState",
        "Mamba3ReferenceBlock",
        "M2RNNMixer",
        "M2RNNMixerState",
        "ReferenceMoE",
        "CppMegaNgramHashEmbedding",
        "CppMegaStructureEmbedding",
    } <= _assert_public_exports(nn)
    assert {
        "CompiledPretrainingStep",
        "EvalMetrics",
        "TrainStepResult",
        "save_checkpoint",
        "load_checkpoint",
        "next_token_cross_entropy",
        "describe_mlx_lm_trainer_apis",
        "require_supported_mlx_lm_trainer_integration",
        "STRUCTURE_FIELD_NAMES",
        "HotspotEvidence",
        "require_kernel_hotspot_evidence",
    } <= _assert_public_exports(training)
    assert {
        "MetalKernelStatus",
        "MetalKernelUnsupported",
        "TrainingKernelStatus",
        "squared_relu",
        "squared_relu_reference",
        "squared_relu_training_status",
    } <= _assert_public_exports(kernels)


def test_nn_root_reexports_public_m2rnn_api() -> None:
    m2rnn_exports = _assert_public_exports(m2rnn)
    nn_exports = _assert_public_exports(nn)

    assert m2rnn_exports <= nn_exports


def test_training_root_does_not_export_foreign_trainers_or_generic_evaluate() -> None:
    exports = _assert_public_exports(training)
    assert FOREIGN_RUNTIME_ALIASES.isdisjoint(exports)
    for name in FOREIGN_RUNTIME_ALIASES:
        assert not hasattr(training, name), name


def test_adapter_metadata_exports_are_boundary_constants_not_trainer_aliases() -> None:
    from cppmega_mlx.training import (
        MLX_LM_DENSE_BATCH_KEYS,
        STRUCTURE_FIELD_NAMES,
        TRAINER_API_NAMES,
        require_supported_mlx_lm_trainer_integration,
    )
    from cppmega_mlx.training.mlx_lm_adapter import (
        STRUCTURE_FIELD_NAMES as ADAPTER_STRUCTURE_FIELD_NAMES,
    )

    assert STRUCTURE_FIELD_NAMES == ADAPTER_STRUCTURE_FIELD_NAMES
    assert set(STRUCTURE_FIELD_NAMES) == {
        "structure_ids",
        "dep_levels",
        "ast_depth_ids",
        "sibling_index_ids",
        "node_type_ids",
    }
    assert MLX_LM_DENSE_BATCH_KEYS == ("tokens", "lengths")
    assert {"TrainingArgs", "train", "iterate_batches", "evaluate"} <= set(TRAINER_API_NAMES)
    assert callable(require_supported_mlx_lm_trainer_integration)


def test_kernel_root_exports_status_gate_not_trainable_metal_claims() -> None:
    exports = _assert_public_exports(kernels)
    assert FOREIGN_RUNTIME_ALIASES.isdisjoint(exports)
    assert kernels.squared_relu_training_status().differentiable is False


def test_porting_plan_documents_package_export_boundary() -> None:
    assert data.__file__ is not None
    doc = Path(data.__file__).resolve().parents[2] / "docs" / "porting_plan.md"
    assert doc is not None
    text = doc.read_text()
    normalized = " ".join(text.split())

    assert "Package-root exports are convenience surfaces" in text
    assert (
        "not full MLX-LM trainer, Megatron distributed runtime, CUDA/TE, "
        "or trainable Metal-kernel claims"
    ) in normalized
