"""ShardingSpec — declarative parallelism strategy over a DeviceTopology.

A :class:`ShardingSpec` carries:
  - the topology (which devices, what mesh)
  - a tuple of :class:`AxisAssignment` (which mesh axis runs which
    ParallelismKind at what degree)
  - global knobs that affect memory accounting and known footguns:
    master_weights_fp32, grad_reduce_dtype, compile_mode, fp8_enabled,
    activation_checkpointing

Pure data layer. The memory estimator (Stage B) and gotcha checker
(Stage C) read these without modifying them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from cppmega_v4.parallelism.topology import DeviceTopology


class ParallelismKind(str, Enum):
    """Recognised parallelism strategies (3D + ZeRO variants)."""

    DP        = "dp"           # replicate weights, shard data
    FSDP1     = "fsdp1"         # legacy fully_sharded_data_parallel
    FSDP2     = "fsdp2"         # modern fully_shard (ZeRO-3)
    ZERO1     = "zero1"         # optimizer state only
    ZERO2     = "zero2"         # optim state + gradients
    TP        = "tp"           # tensor parallel intra-layer
    SP        = "sp"           # sequence parallel (with TP)
    EP        = "ep"           # expert parallel (MoE)
    PP        = "pp"           # pipeline parallel
    PP_VPP    = "pp_vpp"       # virtual pipeline parallel


_VALID_GRAD_DTYPES: Final[frozenset[str]] = frozenset({"bf16", "fp32"})
_VALID_COMPILE_MODES: Final[frozenset[str]] = frozenset({
    "off", "regional", "whole_model",
})
_VALID_ACTIVATION_CHECKPOINT: Final[frozenset[str]] = frozenset({
    "off", "full", "selective",
})


@dataclass(frozen=True)
class AxisAssignment:
    """One row in the sharding table: this mesh axis runs this kind.

    Fields:
      axis_name: name of the mesh axis (must exist in topology.mesh_axes
        or be ``"none"`` for plain replication).
      kind: which parallelism family runs on this axis.
      degree: degree of parallelism. Must equal the mesh-axis degree
        when ``axis_name`` is a real mesh axis.
    """

    axis_name: str
    kind: ParallelismKind
    degree: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ParallelismKind):
            raise TypeError(
                f"AxisAssignment.kind must be ParallelismKind, got "
                f"{type(self.kind).__name__}"
            )
        if not isinstance(self.axis_name, str) or not self.axis_name.strip():
            raise ValueError(
                "AxisAssignment.axis_name must be non-empty str"
            )
        if not isinstance(self.degree, int) or self.degree < 1:
            raise ValueError(
                f"AxisAssignment.degree must be int ≥ 1, got {self.degree!r}"
            )


@dataclass(frozen=True)
class ShardingSpec:
    """Declarative parallelism strategy.

    Fields:
      topology: the underlying device mesh.
      axis_assignments: which mesh axis runs which parallelism kind.
        Every entry's ``axis_name`` must exist in topology.mesh_axes
        and its ``degree`` must match the mesh axis degree.
      brick_axes: optional per-brick override — when present, only these
        bricks participate in the named axes. Empty dict = global.
      master_weights_fp32: keep an fp32 master-weight copy alongside
        bf16/fp8 compute weights. Doubles param memory; required for
        certain optimisers / loss scaling regimes (see nanochat
        ``megatron_optimizer.py`` for the duplication pain story).
      grad_reduce_dtype: ``"bf16"`` (default; faster) | ``"fp32"`` (more
        stable for very deep stacks; doubles grad-buffer cost).
      compile_mode: ``"off"`` | ``"regional"`` (per-block; required to
        avoid FSDP2/Megatron compile footguns — see GotchaChecker
        Stage C) | ``"whole_model"`` (KNOWN BROKEN with FSDP2 and Megatron;
        the spec accepts it so we can flag it loudly).
      fp8_enabled: forward in FP8 via Transformer Engine / torchao; this
        does NOT make grads or optimiser FP8 — known duplication pain.
      activation_checkpointing: ``"off"`` (peak activations) | ``"full"``
        (only block boundaries kept) | ``"selective"`` (per-layer
        cherry-pick).
    """

    topology: DeviceTopology
    axis_assignments: tuple[AxisAssignment, ...]
    brick_axes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    master_weights_fp32: bool = False
    grad_reduce_dtype: str = "bf16"
    compile_mode: str = "regional"
    fp8_enabled: bool = False
    activation_checkpointing: str = "full"

    def __post_init__(self) -> None:
        if not isinstance(self.topology, DeviceTopology):
            raise TypeError(
                f"ShardingSpec.topology must be DeviceTopology, got "
                f"{type(self.topology).__name__}"
            )
        if not self.axis_assignments:
            raise ValueError(
                "ShardingSpec.axis_assignments must declare at least one entry"
            )
        seen_axes: set[str] = set()
        for a in self.axis_assignments:
            if not isinstance(a, AxisAssignment):
                raise TypeError(
                    f"ShardingSpec.axis_assignments entries must be "
                    f"AxisAssignment, got {type(a).__name__}"
                )
            if a.axis_name in seen_axes:
                raise ValueError(
                    f"ShardingSpec axis {a.axis_name!r} appears more than once"
                )
            seen_axes.add(a.axis_name)
            if a.axis_name == "none":
                continue
            if a.axis_name not in self.topology.mesh_axes:
                raise ValueError(
                    f"ShardingSpec axis {a.axis_name!r} not in topology mesh "
                    f"{sorted(self.topology.mesh_axes)}"
                )
            if a.degree != self.topology.mesh_axes[a.axis_name]:
                raise ValueError(
                    f"ShardingSpec axis {a.axis_name!r} degree ({a.degree}) "
                    f"must match topology mesh degree "
                    f"({self.topology.mesh_axes[a.axis_name]})"
                )
        if self.grad_reduce_dtype not in _VALID_GRAD_DTYPES:
            raise ValueError(
                f"ShardingSpec.grad_reduce_dtype={self.grad_reduce_dtype!r} "
                f"not in {sorted(_VALID_GRAD_DTYPES)}"
            )
        if self.compile_mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"ShardingSpec.compile_mode={self.compile_mode!r} not in "
                f"{sorted(_VALID_COMPILE_MODES)}"
            )
        if self.activation_checkpointing not in _VALID_ACTIVATION_CHECKPOINT:
            raise ValueError(
                f"ShardingSpec.activation_checkpointing="
                f"{self.activation_checkpointing!r} not in "
                f"{sorted(_VALID_ACTIVATION_CHECKPOINT)}"
            )

    def axis_kinds(self) -> frozenset[ParallelismKind]:
        return frozenset(a.kind for a in self.axis_assignments)

    def degree_of(self, kind: ParallelismKind) -> int:
        """Return aggregate degree across all axes carrying ``kind``."""
        d = 1
        for a in self.axis_assignments:
            if a.kind is kind:
                d *= a.degree
        return d

    @property
    def num_ranks(self) -> int:
        """Total number of ranks across the mesh."""
        return self.topology.num_devices


# ---------------------------------------------------------------------------
# Built-in factories (the patterns from cppmega + nanochat production)
# ---------------------------------------------------------------------------


def single_device(topology: DeviceTopology) -> ShardingSpec:
    """Replicate everything on every device (DP-only)."""
    return ShardingSpec(
        topology=topology,
        axis_assignments=(
            AxisAssignment(
                axis_name=next(iter(topology.mesh_axes)),
                kind=ParallelismKind.DP,
                degree=next(iter(topology.mesh_axes.values())),
            ),
        ),
        compile_mode="regional",
    )


def fsdp2_only(
    topology: DeviceTopology,
    *,
    fp8_enabled: bool = False,
    activation_checkpointing: str = "full",
) -> ShardingSpec:
    """FSDP2 across the dp axis only (Zaero-3, no TP/EP)."""
    dp_axis = "dp" if "dp" in topology.mesh_axes else next(iter(topology.mesh_axes))
    return ShardingSpec(
        topology=topology,
        axis_assignments=(
            AxisAssignment(
                axis_name=dp_axis, kind=ParallelismKind.FSDP2,
                degree=topology.mesh_axes[dp_axis],
            ),
        ),
        fp8_enabled=fp8_enabled,
        activation_checkpointing=activation_checkpointing,
        compile_mode="regional",
    )


def megatron_ep_only(
    topology: DeviceTopology,
    *,
    ep_axis: str = "ep",
    fp8_enabled: bool = False,
) -> ShardingSpec:
    """Megatron-style EP=N production pattern (cppmega EP=4/8)."""
    if ep_axis not in topology.mesh_axes:
        raise ValueError(
            f"megatron_ep_only: topology missing required mesh axis "
            f"{ep_axis!r}; have {sorted(topology.mesh_axes)}"
        )
    return ShardingSpec(
        topology=topology,
        axis_assignments=(
            AxisAssignment(
                axis_name=ep_axis, kind=ParallelismKind.EP,
                degree=topology.mesh_axes[ep_axis],
            ),
        ),
        fp8_enabled=fp8_enabled,
        compile_mode="regional",
    )


def fsdp2_plus_tp(
    topology: DeviceTopology,
    *,
    dp_axis: str = "dp",
    tp_axis: str = "tp",
) -> ShardingSpec:
    """FSDP2 across dp + Megatron TP across tp (3D parallel base)."""
    for ax in (dp_axis, tp_axis):
        if ax not in topology.mesh_axes:
            raise ValueError(
                f"fsdp2_plus_tp: topology missing required mesh axis "
                f"{ax!r}; have {sorted(topology.mesh_axes)}"
            )
    return ShardingSpec(
        topology=topology,
        axis_assignments=(
            AxisAssignment(
                axis_name=dp_axis, kind=ParallelismKind.FSDP2,
                degree=topology.mesh_axes[dp_axis],
            ),
            AxisAssignment(
                axis_name=tp_axis, kind=ParallelismKind.TP,
                degree=topology.mesh_axes[tp_axis],
            ),
        ),
        compile_mode="regional",   # mandatory — see GotchaChecker
    )


__all__ = [
    "AxisAssignment",
    "ParallelismKind",
    "ShardingSpec",
    "fsdp2_only",
    "fsdp2_plus_tp",
    "megatron_ep_only",
    "single_device",
]
