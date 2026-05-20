"""Contract Probe — dry-run capability check.

See ``ContractProbe.md`` (repo root) for the design.

Stage A surface (this commit):
  - capabilities: TokenizerCapabilities / ParquetCapabilities / ColumnSpec
    + introspect_tokenizer / introspect_parquet
"""

from __future__ import annotations

from cppmega_v4.probe.alternatives import Alternative, generate_alternatives
from cppmega_v4.probe.capabilities import (
    ColumnSpec,
    ParquetCapabilities,
    TokenizerCapabilities,
    introspect_parquet,
    introspect_tokenizer,
)
from cppmega_v4.probe.dry_forward import DryForwardResult, dry_forward
from cppmega_v4.probe.probe import (
    ContractProbeReport,
    ProbeFinding,
    contract_probe,
)
from cppmega_v4.probe.requirements import (
    BRICK_REQUIREMENTS,
    LOSS_REQUIREMENTS,
    DataRequirement,
)

__all__ = [
    "Alternative",
    "BRICK_REQUIREMENTS",
    "ColumnSpec",
    "ContractProbeReport",
    "DataRequirement",
    "DryForwardResult",
    "LOSS_REQUIREMENTS",
    "ParquetCapabilities",
    "ProbeFinding",
    "TokenizerCapabilities",
    "contract_probe",
    "dry_forward",
    "generate_alternatives",
    "introspect_parquet",
    "introspect_tokenizer",
]
