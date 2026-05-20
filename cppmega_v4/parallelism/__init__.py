"""ParallelismSpec — planning/sizing layer for FSDP/TP/EP/PP.

See ``ParallelismSpec.md`` (repo root) for the design + lessons imported
from ../cppmega (Megatron production) and ../nanochat (multi-stack:
FSDP2 + Megatron TP + SPMD).

Stage A surface (this commit):
  - topology: DeviceKind, DeviceSpec, DeviceTopology + builtin factories
  - sharding_spec: ParallelismKind, AxisAssignment, ShardingSpec +
    builtin factories (single_device, fsdp2_only, megatron_ep_only,
    fsdp2_plus_tp)
"""

from __future__ import annotations

from cppmega_v4.parallelism.distributed_memory import (
    DistributedMemoryReport,
    PerRankMemory,
    estimate_distributed_memory,
)
from cppmega_v4.parallelism.sharding_spec import (
    AxisAssignment,
    ParallelismKind,
    ShardingSpec,
    fsdp2_only,
    fsdp2_plus_tp,
    megatron_ep_only,
    single_device,
)
from cppmega_v4.parallelism.topology import (
    TOPOLOGY_BUILTINS,
    DeviceKind,
    DeviceSpec,
    DeviceTopology,
    a100_8x,
    b100_8x,
    gb10_quarter,
    h100_8x,
    h200_8x,
    m3_ultra_solo,
    tpu_v5p_4,
    tpu_v6e_8,
)

__all__ = [
    "AxisAssignment",
    "DeviceKind",
    "DeviceSpec",
    "DeviceTopology",
    "DistributedMemoryReport",
    "ParallelismKind",
    "PerRankMemory",
    "ShardingSpec",
    "TOPOLOGY_BUILTINS",
    "a100_8x",
    "b100_8x",
    "estimate_distributed_memory",
    "fsdp2_only",
    "fsdp2_plus_tp",
    "gb10_quarter",
    "h100_8x",
    "h200_8x",
    "m3_ultra_solo",
    "megatron_ep_only",
    "single_device",
    "tpu_v5p_4",
    "tpu_v6e_8",
]
