"""Data readers and batch collation helpers."""

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch, synthetic_token_batch
from cppmega_mlx.data.fim import (
    EOT_ID,
    FIMMode,
    FIM_MIDDLE_ID,
    FIM_PREFIX_ID,
    FIM_SUFFIX_ID,
    apply_fim_permutation,
    apply_fim_transform,
    sample_middle_span,
)
from cppmega_mlx.data.megatron_indexed import (
    MegatronIndexedDataset,
    MegatronIndexedMetadata,
    megatron_indexed_side_channel_schema,
    open_megatron_indexed_dataset,
)
from cppmega_mlx.data.packing import (
    OversizedSamplePolicy,
    PackedSequences,
    cumulative_doc_ids_from_eos,
    document_boundary_mask,
    pack_documents_with_eos,
)
from cppmega_mlx.data.parquet_dataset import ParquetColumns, TokenParquetDataset
from cppmega_mlx.data.tokenizer_contract import (
    REQUIRED_SPECIAL_TOKEN_IDS,
    SpecialTokenMapping,
    validate_required_special_token_ids,
)
from cppmega_mlx.data.token_dataset import (
    BatchCursor,
    TokenDatasetMetadata,
    TokenNpzDataset,
    iterate_token_batches,
    open_token_dataset,
)

__all__ = [
    "BatchCursor",
    "EOT_ID",
    "FIMMode",
    "FIM_MIDDLE_ID",
    "FIM_PREFIX_ID",
    "FIM_SUFFIX_ID",
    "LMTokenBatch",
    "MegatronIndexedDataset",
    "MegatronIndexedMetadata",
    "OversizedSamplePolicy",
    "ParquetColumns",
    "PackedSequences",
    "REQUIRED_SPECIAL_TOKEN_IDS",
    "SpecialTokenMapping",
    "TokenDatasetMetadata",
    "TokenNpzDataset",
    "TokenParquetDataset",
    "apply_fim_permutation",
    "apply_fim_transform",
    "cumulative_doc_ids_from_eos",
    "document_boundary_mask",
    "ensure_lm_batch",
    "iterate_token_batches",
    "megatron_indexed_side_channel_schema",
    "open_megatron_indexed_dataset",
    "open_token_dataset",
    "pack_documents_with_eos",
    "sample_middle_span",
    "synthetic_token_batch",
    "validate_required_special_token_ids",
]
