"""Device topology — declarative description of physical training hardware.

A :class:`DeviceTopology` carries:
  - the per-device hardware spec (kind, HBM, interconnect, bandwidth)
  - a logical mesh decomposition (axis name → axis degree) whose product
    equals the number of devices

Built-in factories cover the topologies our team actually deploys:
``h100_8x``, ``h200_8x``, ``a100_8x``, ``b100_8x``, ``gb10_quarter``
(Mac quarter / GB10 dev box), ``tpu_v6e_8``, ``tpu_v5p_4``,
``m3_ultra_solo``.

The mesh axes follow the 3D-parallelism convention from
``../nanochat/parallelism_3d.py``: ``dp`` (data parallel),
``tp`` (tensor parallel), ``ep`` (expert parallel), ``pp`` (pipeline).
``fsdp`` reuses the ``dp`` axis when sharding the optimiser state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final


class DeviceKind(str, Enum):
    """Recognised accelerator hardware. Strings so JSON-friendly."""

    H100_80GB = "h100_80gb"
    H200_141GB = "h200_141gb"
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    B100_80GB = "b100_80gb"
    GB10 = "gb10"                  # Apple/NVIDIA hybrid in the mac quarter
    TPU_V5P = "tpu_v5p"
    TPU_V6E = "tpu_v6e"
    M3_ULTRA = "m3_ultra"          # Apple Silicon Mac Studio / unified mem


_HBM_BYTES_BY_KIND: Final[dict[DeviceKind, int]] = {
    DeviceKind.H100_80GB:   80 * 1024 ** 3,
    DeviceKind.H200_141GB:  141 * 1024 ** 3,
    DeviceKind.A100_40GB:   40 * 1024 ** 3,
    DeviceKind.A100_80GB:   80 * 1024 ** 3,
    DeviceKind.B100_80GB:   80 * 1024 ** 3,
    DeviceKind.GB10:        128 * 1024 ** 3,    # GB10 unified ~128 GB usable
    DeviceKind.TPU_V5P:     95 * 1024 ** 3,     # v5p chip HBM ≈ 95 GB
    DeviceKind.TPU_V6E:     32 * 1024 ** 3,
    DeviceKind.M3_ULTRA:    512 * 1024 ** 3,    # full unified memory budget
}


_VALID_INTERCONNECTS: Final[frozenset[str]] = frozenset({
    "nvlink", "nvlink_4th_gen", "nvlink_5th_gen",
    "infiniband", "ethernet",
    "uci",                        # Apple GB10 inter-chip
    "tpu_ici",                    # Inter-Chip Interconnect (TPU pod)
    "unified_memory",             # M3 Ultra single-chip
})


@dataclass(frozen=True)
class DeviceSpec:
    """One physical device.

    Fields:
      kind: hardware family from :class:`DeviceKind`.
      hbm_bytes: usable HBM (driver overhead already deducted).
      interconnect: one of :data:`_VALID_INTERCONNECTS`.
      bandwidth_gbps: rough device-to-device bandwidth (used by gotcha
        checks that compare collective bandwidth vs activation size).
    """

    kind: DeviceKind
    hbm_bytes: int
    interconnect: str
    bandwidth_gbps: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DeviceKind):
            raise TypeError(
                f"DeviceSpec.kind must be DeviceKind, got {type(self.kind).__name__}"
            )
        if self.hbm_bytes < 1:
            raise ValueError(
                f"DeviceSpec.hbm_bytes must be ≥ 1, got {self.hbm_bytes}"
            )
        if self.interconnect not in _VALID_INTERCONNECTS:
            raise ValueError(
                f"DeviceSpec.interconnect={self.interconnect!r} not in "
                f"{sorted(_VALID_INTERCONNECTS)}"
            )
        if self.bandwidth_gbps <= 0:
            raise ValueError(
                f"DeviceSpec.bandwidth_gbps must be > 0, got {self.bandwidth_gbps}"
            )


@dataclass(frozen=True)
class DeviceTopology:
    """A logical mesh of devices.

    The product of ``mesh_axes.values()`` must equal ``len(devices)``.
    Axis names are conventional (dp / tp / ep / pp / sp) but
    user-defined keys are allowed.
    """

    devices: tuple[DeviceSpec, ...]
    mesh_axes: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("DeviceTopology.devices must not be empty")
        for d in self.devices:
            if not isinstance(d, DeviceSpec):
                raise TypeError(
                    f"DeviceTopology.devices entries must be DeviceSpec, "
                    f"got {type(d).__name__}"
                )
        if not self.mesh_axes:
            raise ValueError(
                "DeviceTopology.mesh_axes must declare at least one axis"
            )
        # All-positive degrees + product == device count.
        product = 1
        for name, degree in self.mesh_axes.items():
            if not isinstance(degree, int) or degree < 1:
                raise ValueError(
                    f"mesh axis {name!r} degree must be int ≥ 1, got {degree!r}"
                )
            product *= degree
        if product != len(self.devices):
            raise ValueError(
                f"mesh axes product ({product}) must equal device count "
                f"({len(self.devices)}); axes={self.mesh_axes!r}"
            )

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def total_hbm_bytes(self) -> int:
        return sum(d.hbm_bytes for d in self.devices)

    def axis_degree(self, name: str) -> int:
        """Return mesh degree along ``name`` (default 1 when unknown)."""
        return self.mesh_axes.get(name, 1)


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def _devices_of(kind: DeviceKind, count: int, *, interconnect: str,
                bandwidth_gbps: float) -> tuple[DeviceSpec, ...]:
    hbm = _HBM_BYTES_BY_KIND[kind]
    return tuple(
        DeviceSpec(
            kind=kind, hbm_bytes=hbm,
            interconnect=interconnect, bandwidth_gbps=bandwidth_gbps,
        )
        for _ in range(count)
    )


def h100_8x(*, dp: int = 8, tp: int = 1, ep: int = 1, pp: int = 1) -> DeviceTopology:
    """8x H100:80GB single-node (NVLink 4th gen, ~900 GB/s)."""
    return DeviceTopology(
        devices=_devices_of(DeviceKind.H100_80GB, 8,
                            interconnect="nvlink_4th_gen",
                            bandwidth_gbps=900.0),
        mesh_axes={"dp": dp, "tp": tp, "ep": ep, "pp": pp},
    )


def h200_8x(*, dp: int = 8, tp: int = 1, ep: int = 1, pp: int = 1) -> DeviceTopology:
    """8x H200:141GB single-node (NVLink 4th gen, ~900 GB/s)."""
    return DeviceTopology(
        devices=_devices_of(DeviceKind.H200_141GB, 8,
                            interconnect="nvlink_4th_gen",
                            bandwidth_gbps=900.0),
        mesh_axes={"dp": dp, "tp": tp, "ep": ep, "pp": pp},
    )


def a100_8x(*, hbm: str = "80gb",
            dp: int = 8, tp: int = 1, ep: int = 1, pp: int = 1) -> DeviceTopology:
    """8x A100 single-node. ``hbm`` selects "40gb" or "80gb"."""
    kind = DeviceKind.A100_80GB if hbm == "80gb" else DeviceKind.A100_40GB
    return DeviceTopology(
        devices=_devices_of(kind, 8,
                            interconnect="nvlink",
                            bandwidth_gbps=600.0),
        mesh_axes={"dp": dp, "tp": tp, "ep": ep, "pp": pp},
    )


def b100_8x(*, dp: int = 8, tp: int = 1, ep: int = 1, pp: int = 1) -> DeviceTopology:
    """8x B100 single-node (NVLink 5th gen)."""
    return DeviceTopology(
        devices=_devices_of(DeviceKind.B100_80GB, 8,
                            interconnect="nvlink_5th_gen",
                            bandwidth_gbps=1800.0),
        mesh_axes={"dp": dp, "tp": tp, "ep": ep, "pp": pp},
    )


def gb10_quarter() -> DeviceTopology:
    """1x GB10 (Mac quarter dev box, unified ~128 GB)."""
    return DeviceTopology(
        devices=_devices_of(DeviceKind.GB10, 1,
                            interconnect="unified_memory",
                            bandwidth_gbps=800.0),
        mesh_axes={"dp": 1},
    )


def tpu_v6e_8() -> DeviceTopology:
    """8x TPU v6e (32 GB chip HBM, ICI inter-chip)."""
    return DeviceTopology(
        devices=_devices_of(DeviceKind.TPU_V6E, 8,
                            interconnect="tpu_ici",
                            bandwidth_gbps=1640.0),
        mesh_axes={"dp": 4, "tp": 2},   # 2D mesh: data x model
    )


def tpu_v5p_4() -> DeviceTopology:
    """4x TPU v5p (~95 GB chip HBM, ICI)."""
    return DeviceTopology(
        devices=_devices_of(DeviceKind.TPU_V5P, 4,
                            interconnect="tpu_ici",
                            bandwidth_gbps=2400.0),
        mesh_axes={"dp": 2, "tp": 2},
    )


def m3_ultra_solo() -> DeviceTopology:
    """1x Mac Studio M3 Ultra (unified memory ~512 GB)."""
    return DeviceTopology(
        devices=_devices_of(DeviceKind.M3_ULTRA, 1,
                            interconnect="unified_memory",
                            bandwidth_gbps=800.0),
        mesh_axes={"dp": 1},
    )


# ---------------------------------------------------------------------------
# Registry (Stage E GUI dropdown)
# ---------------------------------------------------------------------------


TOPOLOGY_BUILTINS: dict[str, str] = {
    "h100_8x":       "cppmega_v4.parallelism.topology:h100_8x",
    "h200_8x":       "cppmega_v4.parallelism.topology:h200_8x",
    "a100_8x":       "cppmega_v4.parallelism.topology:a100_8x",
    "b100_8x":       "cppmega_v4.parallelism.topology:b100_8x",
    "gb10_quarter":  "cppmega_v4.parallelism.topology:gb10_quarter",
    "tpu_v6e_8":     "cppmega_v4.parallelism.topology:tpu_v6e_8",
    "tpu_v5p_4":     "cppmega_v4.parallelism.topology:tpu_v5p_4",
    "m3_ultra_solo": "cppmega_v4.parallelism.topology:m3_ultra_solo",
}


__all__ = [
    "DeviceKind",
    "DeviceSpec",
    "DeviceTopology",
    "TOPOLOGY_BUILTINS",
    "a100_8x",
    "b100_8x",
    "gb10_quarter",
    "h100_8x",
    "h200_8x",
    "m3_ultra_solo",
    "tpu_v5p_4",
    "tpu_v6e_8",
]
