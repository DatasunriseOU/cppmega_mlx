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
from cppmega_mlx.data import fim as fim_module

PACKAGE_ROOTS = (package, config, data, kernels, models, nn, recipes, training)
FIM_DATA_EXPORTS = {
    "EOT_ID",
    "FIMMode",
    "FIM_MIDDLE_ID",
    "FIM_PREFIX_ID",
    "FIM_SUFFIX_ID",
    "REQUIRED_SPECIAL_TOKEN_IDS",
    "SpecialTokenMapping",
    "apply_fim_permutation",
    "apply_fim_transform",
    "sample_middle_span",
    "validate_required_special_token_ids",
}
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
    } | FIM_DATA_EXPORTS <= _assert_public_exports(data)
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
        "next_token_cross_entropy_with_stp",
        "describe_mlx_lm_trainer_apis",
        "require_supported_mlx_lm_trainer_integration",
        "STRUCTURE_FIELD_NAMES",
        "HotspotEvidence",
        "require_kernel_hotspot_evidence",
        "compute_stp_loss",
        "next_token_and_stp_loss",
        "STPLossConfig",
        "STPLossMetrics",
        "M05_MTP_PARITY_POLICY",
        "M05_MTP_PARITY_PROFILE",
        "build_m05_mtp_parity_manifest",
        "m05_loss_values_sha256",
        "validate_m05_mtp_parity_manifest_dict",
        "write_m05_mtp_parity_manifest_json",
    } <= _assert_public_exports(training)
    assert {
        "MetalKernelStatus",
        "MetalKernelUnsupported",
        "TrainingKernelStatus",
        "squared_relu",
        "squared_relu_reference",
        "squared_relu_training_status",
    } <= _assert_public_exports(kernels)


def test_data_root_reexports_fim_transform_and_tokenizer_contract() -> None:
    exports = _assert_public_exports(data)

    assert FIM_DATA_EXPORTS <= exports
    assert data.apply_fim_permutation is fim_module.apply_fim_permutation
    assert data.apply_fim_transform is fim_module.apply_fim_transform
    assert data.sample_middle_span is fim_module.sample_middle_span
    assert data.FIM_PREFIX_ID == 4
    assert data.FIM_MIDDLE_ID == 5
    assert data.FIM_SUFFIX_ID == 6
    assert data.REQUIRED_SPECIAL_TOKEN_IDS == {
        "BOS": 2,
        "EOT": 3,
        "FIM_PREFIX": 4,
        "FIM_MIDDLE": 5,
        "FIM_SUFFIX": 6,
        "CODE_START": 7,
        "FIM_INSTRUCTION": 45,
        "SPACE": 46,
        "NL": 47,
    }


def test_data_wildcard_import_includes_fim_reference_exports() -> None:
    namespace: dict[str, object] = {}

    exec("from cppmega_mlx.data import *", {}, namespace)

    for name in FIM_DATA_EXPORTS:
        assert name in namespace
    assert namespace["apply_fim_transform"] is data.apply_fim_transform
    assert namespace["FIM_PREFIX_ID"] == 4
    assert namespace["FIM_MIDDLE_ID"] == 5
    assert namespace["FIM_SUFFIX_ID"] == 6
    assert namespace["EOT_ID"] == 3
    assert namespace["REQUIRED_SPECIAL_TOKEN_IDS"] is data.REQUIRED_SPECIAL_TOKEN_IDS


def test_nn_root_reexports_public_m2rnn_api() -> None:
    m2rnn_exports = _assert_public_exports(m2rnn)
    nn_exports = _assert_public_exports(nn)

    assert m2rnn_exports <= nn_exports


def test_training_root_does_not_export_foreign_trainers_or_generic_evaluate() -> None:
    exports = _assert_public_exports(training)
    assert FOREIGN_RUNTIME_ALIASES.isdisjoint(exports)
    for name in FOREIGN_RUNTIME_ALIASES:
        assert not hasattr(training, name), name


def test_fim_reference_transform_stays_data_only_not_training_integration() -> None:
    training_exports = _assert_public_exports(training)

    assert FIM_DATA_EXPORTS.isdisjoint(training_exports)
    for name in FIM_DATA_EXPORTS:
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
