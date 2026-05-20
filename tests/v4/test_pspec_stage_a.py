"""PSpec Stage A tests — topology + sharding_spec data-layer."""

from __future__ import annotations

import pytest

from cppmega_v4.parallelism import (
    AxisAssignment,
    DeviceKind,
    DeviceSpec,
    DeviceTopology,
    ParallelismKind,
    ShardingSpec,
    TOPOLOGY_BUILTINS,
    a100_8x,
    b100_8x,
    fsdp2_only,
    fsdp2_plus_tp,
    gb10_quarter,
    h100_8x,
    h200_8x,
    m3_ultra_solo,
    megatron_ep_only,
    single_device,
    tpu_v5p_4,
    tpu_v6e_8,
)


# ---------------------------------------------------------------------------
# DeviceSpec validation
# ---------------------------------------------------------------------------


def test_device_spec_rejects_non_enum_kind():
    with pytest.raises(TypeError, match="DeviceKind"):
        DeviceSpec(
            kind="h100", hbm_bytes=80 * 1024**3,  # type: ignore[arg-type]
            interconnect="nvlink", bandwidth_gbps=900.0,
        )


def test_device_spec_rejects_zero_hbm():
    with pytest.raises(ValueError, match="hbm_bytes"):
        DeviceSpec(
            kind=DeviceKind.H100_80GB, hbm_bytes=0,
            interconnect="nvlink", bandwidth_gbps=900.0,
        )


def test_device_spec_rejects_unknown_interconnect():
    with pytest.raises(ValueError, match="interconnect"):
        DeviceSpec(
            kind=DeviceKind.H100_80GB, hbm_bytes=80 * 1024**3,
            interconnect="quantum_tunneling",
            bandwidth_gbps=900.0,
        )


def test_device_spec_rejects_non_positive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth_gbps"):
        DeviceSpec(
            kind=DeviceKind.H100_80GB, hbm_bytes=80 * 1024**3,
            interconnect="nvlink", bandwidth_gbps=0.0,
        )


# ---------------------------------------------------------------------------
# DeviceTopology validation
# ---------------------------------------------------------------------------


def _one_h100() -> DeviceSpec:
    return DeviceSpec(
        kind=DeviceKind.H100_80GB, hbm_bytes=80 * 1024**3,
        interconnect="nvlink", bandwidth_gbps=900.0,
    )


def test_topology_rejects_empty_devices():
    with pytest.raises(ValueError, match="devices must not be empty"):
        DeviceTopology(devices=(), mesh_axes={"dp": 1})


def test_topology_rejects_non_device_spec():
    with pytest.raises(TypeError, match="DeviceSpec"):
        DeviceTopology(devices=("not a spec",), mesh_axes={"dp": 1})  # type: ignore[arg-type]


def test_topology_rejects_empty_mesh_axes():
    with pytest.raises(ValueError, match="mesh_axes must declare"):
        DeviceTopology(devices=(_one_h100(),), mesh_axes={})


def test_topology_rejects_bad_axis_degree():
    with pytest.raises(ValueError, match="degree must be int"):
        DeviceTopology(devices=(_one_h100(),), mesh_axes={"dp": 0})


def test_topology_rejects_product_mismatch():
    with pytest.raises(ValueError, match="mesh axes product"):
        DeviceTopology(
            devices=(_one_h100(), _one_h100()),
            mesh_axes={"dp": 4},  # 4 ≠ 2 devices
        )


def test_topology_num_devices_and_total_hbm():
    t = h100_8x()
    assert t.num_devices == 8
    assert t.total_hbm_bytes == 8 * 80 * 1024**3


def test_topology_axis_degree_returns_1_for_unknown():
    t = h100_8x()
    assert t.axis_degree("tp") == 1
    assert t.axis_degree("totally_made_up") == 1


# ---------------------------------------------------------------------------
# Built-in topology factories
# ---------------------------------------------------------------------------


def test_h100_8x_default_is_dp_only():
    t = h100_8x()
    assert t.num_devices == 8
    assert t.mesh_axes == {"dp": 8, "tp": 1, "ep": 1, "pp": 1}
    assert all(d.kind is DeviceKind.H100_80GB for d in t.devices)


def test_h100_8x_can_split_mesh():
    t = h100_8x(dp=2, tp=2, ep=2, pp=1)   # product = 8
    assert t.mesh_axes["dp"] == 2
    assert t.mesh_axes["tp"] == 2


def test_h100_8x_rejects_bad_mesh_product():
    with pytest.raises(ValueError, match="mesh axes product"):
        h100_8x(dp=3, tp=2, ep=2)  # product != 8


def test_h200_8x_uses_141gb_hbm():
    t = h200_8x()
    assert all(d.kind is DeviceKind.H200_141GB for d in t.devices)
    assert t.devices[0].hbm_bytes == 141 * 1024**3


def test_a100_8x_default_is_80gb():
    t = a100_8x()
    assert all(d.kind is DeviceKind.A100_80GB for d in t.devices)


def test_a100_8x_can_select_40gb():
    t = a100_8x(hbm="40gb")
    assert all(d.kind is DeviceKind.A100_40GB for d in t.devices)


def test_b100_8x_uses_nvlink5():
    t = b100_8x()
    assert t.devices[0].interconnect == "nvlink_5th_gen"
    assert t.devices[0].bandwidth_gbps == 1800.0


def test_gb10_quarter_is_single_device_unified_memory():
    t = gb10_quarter()
    assert t.num_devices == 1
    assert t.devices[0].kind is DeviceKind.GB10
    assert t.devices[0].interconnect == "unified_memory"


def test_tpu_v6e_8_is_2d_mesh_dp_tp():
    t = tpu_v6e_8()
    assert t.num_devices == 8
    assert t.mesh_axes == {"dp": 4, "tp": 2}


def test_tpu_v5p_4_is_2d_mesh():
    t = tpu_v5p_4()
    assert t.num_devices == 4
    assert t.mesh_axes == {"dp": 2, "tp": 2}


def test_m3_ultra_solo_uses_512gb_unified():
    t = m3_ultra_solo()
    assert t.num_devices == 1
    assert t.devices[0].hbm_bytes == 512 * 1024**3
    assert t.devices[0].interconnect == "unified_memory"


def test_topology_builtins_registry_covers_each_factory():
    assert set(TOPOLOGY_BUILTINS.keys()) == {
        "h100_8x", "h200_8x", "a100_8x", "b100_8x", "gb10_quarter",
        "tpu_v6e_8", "tpu_v5p_4", "m3_ultra_solo",
    }


@pytest.mark.parametrize("name", sorted(TOPOLOGY_BUILTINS.keys()))
def test_every_builtin_topology_well_formed(name):
    factory = {
        "h100_8x": h100_8x, "h200_8x": h200_8x, "a100_8x": a100_8x,
        "b100_8x": b100_8x, "gb10_quarter": gb10_quarter,
        "tpu_v6e_8": tpu_v6e_8, "tpu_v5p_4": tpu_v5p_4,
        "m3_ultra_solo": m3_ultra_solo,
    }[name]
    t = factory()
    assert t.num_devices >= 1
    assert t.total_hbm_bytes > 0


# ---------------------------------------------------------------------------
# AxisAssignment validation
# ---------------------------------------------------------------------------


def test_axis_assignment_rejects_non_enum_kind():
    with pytest.raises(TypeError, match="ParallelismKind"):
        AxisAssignment(
            axis_name="dp", kind="fsdp2", degree=8,  # type: ignore[arg-type]
        )


def test_axis_assignment_rejects_blank_name():
    with pytest.raises(ValueError, match="axis_name"):
        AxisAssignment(axis_name="", kind=ParallelismKind.DP, degree=8)


def test_axis_assignment_rejects_bad_degree():
    with pytest.raises(ValueError, match="degree must be int"):
        AxisAssignment(axis_name="dp", kind=ParallelismKind.DP, degree=0)


# ---------------------------------------------------------------------------
# ShardingSpec validation
# ---------------------------------------------------------------------------


def test_sharding_spec_rejects_non_topology():
    with pytest.raises(TypeError, match="DeviceTopology"):
        ShardingSpec(
            topology="not a topology",  # type: ignore[arg-type]
            axis_assignments=(
                AxisAssignment("dp", ParallelismKind.DP, 8),
            ),
        )


def test_sharding_spec_rejects_empty_assignments():
    with pytest.raises(ValueError, match="must declare at least one"):
        ShardingSpec(topology=h100_8x(), axis_assignments=())


def test_sharding_spec_rejects_duplicate_axis():
    with pytest.raises(ValueError, match="appears more than once"):
        ShardingSpec(
            topology=h100_8x(),
            axis_assignments=(
                AxisAssignment("dp", ParallelismKind.DP, 8),
                AxisAssignment("dp", ParallelismKind.FSDP2, 8),
            ),
        )


def test_sharding_spec_rejects_axis_not_in_mesh():
    with pytest.raises(ValueError, match="not in topology mesh"):
        ShardingSpec(
            topology=h100_8x(),
            axis_assignments=(
                AxisAssignment("totally_made_up_axis",
                               ParallelismKind.DP, 8),
            ),
        )


def test_sharding_spec_rejects_degree_mismatch():
    with pytest.raises(ValueError, match="must match topology mesh degree"):
        ShardingSpec(
            topology=h100_8x(),
            axis_assignments=(
                AxisAssignment("dp", ParallelismKind.DP, 4),  # mesh has 8
            ),
        )


def test_sharding_spec_accepts_none_axis_for_pure_replication():
    """Axis 'none' is a sentinel for non-mesh-bound replication."""
    spec = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(
            AxisAssignment("none", ParallelismKind.DP, 8),
        ),
    )
    assert spec.axis_assignments[0].axis_name == "none"


def test_sharding_spec_rejects_bad_grad_reduce_dtype():
    with pytest.raises(ValueError, match="grad_reduce_dtype"):
        ShardingSpec(
            topology=h100_8x(),
            axis_assignments=(AxisAssignment("dp", ParallelismKind.DP, 8),),
            grad_reduce_dtype="fp16",
        )


def test_sharding_spec_rejects_bad_compile_mode():
    with pytest.raises(ValueError, match="compile_mode"):
        ShardingSpec(
            topology=h100_8x(),
            axis_assignments=(AxisAssignment("dp", ParallelismKind.DP, 8),),
            compile_mode="trash",
        )


def test_sharding_spec_rejects_bad_activation_checkpointing():
    with pytest.raises(ValueError, match="activation_checkpointing"):
        ShardingSpec(
            topology=h100_8x(),
            axis_assignments=(AxisAssignment("dp", ParallelismKind.DP, 8),),
            activation_checkpointing="weird",
        )


def test_sharding_spec_accepts_whole_model_compile_for_gotcha_detection():
    """The spec ACCEPTS 'whole_model' so the Stage C gotcha checker can
    flag it loudly — it's KNOWN BROKEN with FSDP2 / Megatron, but we
    need to surface that as a diagnostic, not refuse construction."""
    spec = ShardingSpec(
        topology=h100_8x(),
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.FSDP2, 8),
        ),
        compile_mode="whole_model",
    )
    assert spec.compile_mode == "whole_model"


def test_sharding_spec_axis_kinds_returns_frozenset():
    spec = ShardingSpec(
        topology=h100_8x(dp=2, tp=2, ep=2, pp=1),
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.FSDP2, 2),
            AxisAssignment("tp", ParallelismKind.TP, 2),
            AxisAssignment("ep", ParallelismKind.EP, 2),
        ),
    )
    assert spec.axis_kinds() == frozenset({
        ParallelismKind.FSDP2, ParallelismKind.TP, ParallelismKind.EP,
    })


def test_sharding_spec_degree_of_aggregates_same_kind_axes():
    """If two axes both run EP (unusual but legal), degree_of multiplies."""
    spec = ShardingSpec(
        topology=h100_8x(dp=2, tp=1, ep=2, pp=2),
        axis_assignments=(
            AxisAssignment("dp", ParallelismKind.EP, 2),
            AxisAssignment("ep", ParallelismKind.EP, 2),
        ),
    )
    assert spec.degree_of(ParallelismKind.EP) == 4


def test_sharding_spec_num_ranks_matches_topology():
    spec = single_device(h100_8x())
    assert spec.num_ranks == 8


# ---------------------------------------------------------------------------
# Built-in sharding-spec factories
# ---------------------------------------------------------------------------


def test_single_device_dp_only():
    t = h100_8x()
    spec = single_device(t)
    assert spec.axis_kinds() == {ParallelismKind.DP}
    assert spec.compile_mode == "regional"


def test_fsdp2_only_uses_dp_axis():
    t = h100_8x()
    spec = fsdp2_only(t)
    assert spec.axis_kinds() == {ParallelismKind.FSDP2}
    assert spec.compile_mode == "regional"   # mandatory


def test_fsdp2_only_propagates_fp8_flag():
    spec = fsdp2_only(h100_8x(), fp8_enabled=True)
    assert spec.fp8_enabled is True


def test_megatron_ep_only_requires_ep_axis():
    """A topology without an 'ep' mesh axis must be rejected."""
    no_ep = gb10_quarter()   # mesh = {"dp": 1} only — no ep
    with pytest.raises(ValueError, match="missing required mesh axis"):
        megatron_ep_only(no_ep)


def test_megatron_ep_only_uses_existing_ep_axis():
    t = h100_8x(dp=2, tp=1, ep=4, pp=1)
    spec = megatron_ep_only(t)
    assert spec.axis_kinds() == {ParallelismKind.EP}
    assert spec.degree_of(ParallelismKind.EP) == 4


def test_fsdp2_plus_tp_3d_pattern():
    t = h100_8x(dp=4, tp=2, ep=1, pp=1)
    spec = fsdp2_plus_tp(t)
    assert spec.axis_kinds() == {ParallelismKind.FSDP2, ParallelismKind.TP}
    assert spec.degree_of(ParallelismKind.FSDP2) == 4
    assert spec.degree_of(ParallelismKind.TP) == 2
    # Regional compile is mandatory — the only safe mode with FSDP2/TP
    assert spec.compile_mode == "regional"


def test_fsdp2_plus_tp_rejects_missing_axes():
    with pytest.raises(ValueError, match="missing required mesh axis"):
        fsdp2_plus_tp(h100_8x(), tp_axis="missing_axis")


# ---------------------------------------------------------------------------
# Frozen invariant
# ---------------------------------------------------------------------------


def test_device_spec_is_frozen():
    d = _one_h100()
    with pytest.raises((AttributeError, TypeError)):
        d.hbm_bytes = 1  # type: ignore[misc]


def test_topology_is_frozen():
    t = h100_8x()
    with pytest.raises((AttributeError, TypeError)):
        t.devices = ()  # type: ignore[misc]


def test_sharding_spec_is_frozen():
    s = single_device(h100_8x())
    with pytest.raises((AttributeError, TypeError)):
        s.compile_mode = "off"  # type: ignore[misc]
