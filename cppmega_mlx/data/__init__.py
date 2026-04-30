"""Data readers and batch collation helpers."""

from cppmega_mlx.data.batch import LMTokenBatch, ensure_lm_batch, synthetic_token_batch
from cppmega_mlx.data.megatron_indexed import (
    MegatronIndexedDataset,
    MegatronIndexedMetadata,
)
from cppmega_mlx.data.parquet_dataset import ParquetColumns, TokenParquetDataset
from cppmega_mlx.data.token_dataset import (
    BatchCursor,
    TokenDatasetMetadata,
    TokenNpzDataset,
    iterate_token_batches,
    open_token_dataset,
)

__all__ = [
    "BatchCursor",
    "LMTokenBatch",
    "MegatronIndexedDataset",
    "MegatronIndexedMetadata",
    "ParquetColumns",
    "TokenDatasetMetadata",
    "TokenNpzDataset",
    "TokenParquetDataset",
    "ensure_lm_batch",
    "iterate_token_batches",
    "open_token_dataset",
    "synthetic_token_batch",
]
