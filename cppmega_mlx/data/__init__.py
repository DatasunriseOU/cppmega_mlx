"""Data readers and batch collation helpers.

The package also contains portable corpus/indexer modules used on Linux hosts
where Apple's MLX runtime is unavailable. Public MLX-backed exports are loaded
on first attribute access so importing a portable submodule does not eagerly
import the entire training runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "LMTokenBatch": "cppmega_mlx.data.batch",
    "ensure_lm_batch": "cppmega_mlx.data.batch",
    "synthetic_token_batch": "cppmega_mlx.data.batch",
    "LocalTokenBatchDataset": "cppmega_mlx.data.dataloader_bridge",
    "TorchDataLoaderBridgeConfig": "cppmega_mlx.data.dataloader_bridge",
    "TorchDataLoaderBridgeError": "cppmega_mlx.data.dataloader_bridge",
    "build_spawn_dataloader": "cppmega_mlx.data.dataloader_bridge",
    "is_torch_dataloader_available": "cppmega_mlx.data.dataloader_bridge",
    "iter_mlx_batches": "cppmega_mlx.data.dataloader_bridge",
    "EOT_ID": "cppmega_mlx.data.fim",
    "FIMSpecialTokenIds": "cppmega_mlx.data.fim",
    "FIM_INSTRUCTION_ID": "cppmega_mlx.data.fim",
    "FIMMode": "cppmega_mlx.data.fim",
    "FIM_MIDDLE_ID": "cppmega_mlx.data.fim",
    "FIM_PREFIX_ID": "cppmega_mlx.data.fim",
    "FIM_SPECIAL_TOKEN_IDS": "cppmega_mlx.data.fim",
    "FIM_SUFFIX_ID": "cppmega_mlx.data.fim",
    "apply_fim_permutation": "cppmega_mlx.data.fim",
    "apply_fim_transform": "cppmega_mlx.data.fim",
    "apply_ifim_permutation": "cppmega_mlx.data.fim",
    "apply_ifim_transform": "cppmega_mlx.data.fim",
    "extract_ifim_instruction_text": "cppmega_mlx.data.fim",
    "sample_middle_span": "cppmega_mlx.data.fim",
    "MegatronIndexedDataset": "cppmega_mlx.data.megatron_indexed",
    "MegatronIndexedMetadata": "cppmega_mlx.data.megatron_indexed",
    "MegatronIndexedMultiShardDataset": "cppmega_mlx.data.megatron_indexed",
    "MegatronIndexedMultiShardMetadata": "cppmega_mlx.data.megatron_indexed",
    "megatron_indexed_side_channel_schema": "cppmega_mlx.data.megatron_indexed",
    "open_megatron_indexed_dataset": "cppmega_mlx.data.megatron_indexed",
    "ProductionMegatronDatasetMetadata": "cppmega_mlx.data.production_bundle",
    "open_production_megatron_bundle": "cppmega_mlx.data.production_bundle",
    "OversizedSamplePolicy": "cppmega_mlx.data.packing",
    "PackedSequences": "cppmega_mlx.data.packing",
    "PackingStrategy": "cppmega_mlx.data.packing",
    "cumulative_doc_ids_from_eos": "cppmega_mlx.data.packing",
    "document_boundary_mask": "cppmega_mlx.data.packing",
    "mlx_cumulative_doc_ids_from_eos": "cppmega_mlx.data.packing",
    "mlx_document_boundary_mask": "cppmega_mlx.data.packing",
    "mlx_sequence_packing_attention_mask": "cppmega_mlx.data.packing",
    "pack_bos_aligned_best_fit": "cppmega_mlx.data.packing",
    "pack_documents_with_eos": "cppmega_mlx.data.packing",
    "MultiShardTokenParquetDataset": "cppmega_mlx.data.parquet_dataset",
    "ParquetColumns": "cppmega_mlx.data.parquet_dataset",
    "TokenParquetDataset": "cppmega_mlx.data.parquet_dataset",
    "MAX_PLATFORM_IDS": "cppmega_mlx.data.platform_context",
    "PLATFORM_VOCAB": "cppmega_mlx.data.platform_context",
    "PLATFORM_VOCAB_SIZE": "cppmega_mlx.data.platform_context",
    "PlatformContext": "cppmega_mlx.data.platform_context",
    "encode_platform_context": "cppmega_mlx.data.platform_context",
    "parse_platform_context": "cppmega_mlx.data.platform_context",
    "platform_ids_array": "cppmega_mlx.data.platform_context",
    "render_platform_context": "cppmega_mlx.data.platform_context",
    "REQUIRED_SPECIAL_TOKEN_IDS": "cppmega_mlx.data.tokenizer_contract",
    "SpecialTokenMapping": "cppmega_mlx.data.tokenizer_contract",
    "TOOL_USE_SPECIAL_TOKEN_IDS": "cppmega_mlx.data.tokenizer_contract",
    "validate_required_special_token_ids": "cppmega_mlx.data.tokenizer_contract",
    "BatchCursor": "cppmega_mlx.data.token_dataset",
    "TokenDatasetMetadata": "cppmega_mlx.data.dataset_metadata",
    "TokenNpzDataset": "cppmega_mlx.data.token_dataset",
    "iterate_token_batches": "cppmega_mlx.data.token_dataset",
    "open_token_dataset": "cppmega_mlx.data.token_dataset",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
