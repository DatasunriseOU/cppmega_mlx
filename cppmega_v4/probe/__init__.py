"""Contract Probe — dry-run capability check.

See ``ContractProbe.md`` (repo root) for the design.

Stage A surface (this commit):
  - capabilities: TokenizerCapabilities / ParquetCapabilities / ColumnSpec
    + introspect_tokenizer / introspect_parquet
"""

from __future__ import annotations

from cppmega_v4.probe.capabilities import (
    ColumnSpec,
    ParquetCapabilities,
    TokenizerCapabilities,
    introspect_parquet,
    introspect_tokenizer,
)

__all__ = [
    "ColumnSpec",
    "ParquetCapabilities",
    "TokenizerCapabilities",
    "introspect_parquet",
    "introspect_tokenizer",
]
