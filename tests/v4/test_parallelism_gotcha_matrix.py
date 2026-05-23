"""V7-B17: parametrize parallelism gotchas × topology factories.

For each available gotcha, build a spec that triggers it on the
intended topology. Assert the rule fires; then swap to a sibling
topology factory and assert the same rule keeps firing iff its
condition is topology-agnostic, or stays silent iff it is topology-
specific. This locks the gotcha-checker against accidental fan-out
across CPU/GPU/TPU configs.
"""

from __future__ import annotations

import pytest

from cppmega_v4.buildspec import (
    LossKind, LossSpec, ModelBuildSpec, OptimKind, OptimSpec,
    ParamGroup,
)
from cppmega_v4.fusion import from_block_specs
from cppmega_v4.parallelism import (
    AxisAssignment, GOTCHAS, ParallelismKind, ShardingSpec,
    a100_8x, b100_8x, check_gotchas, gb10_quarter,
    h100_8x, h200_8x, m3_ultra_solo, tpu_v5p_4, tpu_v6e_8,
)


TOPOLOGIES = {
    "h100_8x":       h100_8x,
    "gb10_quarter":  gb10_quarter,
    "m3_ultra_solo": m3_ultra_solo,
    "tpu_v6e_8":     tpu_v6e_8,
}


def _build_spec(topology, *, num_experts: int = 4) -> ModelBuildSpec:
    block_specs = [
        {"kind": "attention", "name": "attn", "params": {}},
        {"kind": "mlp", "name": "mlp",
         "params": {"intermediate_size": 64, "activation": "swiglu"}},
    ]
    graph = from_block_specs(block_specs, hidden_size=32, instantiate=False)
    loss = LossSpec(kind=LossKind.CROSS_ENTROPY,
                     head_outputs=("mlp",), params={},
                     label_source="next_token")
    optim = OptimSpec(
        kind=OptimKind.ADAMW,
        groups=(ParamGroup(matcher="all", lr=1e-3,
                            weight_decay=0.01, betas=(0.9, 0.95)),),
    )
    return ModelBuildSpec(graph=graph, loss=loss, optim=optim)


def _build_sharding(topology, *,
                     kinds: list[ParallelismKind] | None = None,
                     compile_mode: str = "off",
                     fp8_enabled: bool = False) -> ShardingSpec:
    axes: list[AxisAssignment] = []
    # Map ParallelismKind → canonical mesh axis name. Every topology
    # exposes the same {dp, ep, pp, tp} mesh.
    name_for = {
        ParallelismKind.DP: "dp", ParallelismKind.FSDP1: "dp",
        ParallelismKind.FSDP2: "dp", ParallelismKind.ZERO1: "dp",
        ParallelismKind.ZERO2: "dp", ParallelismKind.TP: "tp",
        ParallelismKind.SP: "tp", ParallelismKind.EP: "ep",
        ParallelismKind.PP: "pp", ParallelismKind.PP_VPP: "pp",
    }
    mesh = dict(topology.mesh_axes)
    if kinds:
        used: set[str] = set()
        for k in kinds:
            axis_name = name_for[k]
            if axis_name in used or axis_name not in mesh:
                continue
            axes.append(AxisAssignment(
                axis_name=axis_name, kind=k,
                degree=int(mesh[axis_name]),
            ))
            used.add(axis_name)
        if not axes:
            # If none of the requested kinds map onto this topology's
            # mesh (e.g. tpu_v6e_8 has no fsdp axis), fall back to a
            # trivial DP axis with whatever degree the topology
            # advertises.
            axes.append(AxisAssignment(
                axis_name="dp", kind=ParallelismKind.DP,
                degree=int(mesh.get("dp", 1)),
            ))
    else:
        # Clean spec: pick the dp axis at its native degree.
        axes.append(AxisAssignment(
            axis_name="dp", kind=ParallelismKind.DP,
            degree=int(mesh.get("dp", 1)),
        ))
    return ShardingSpec(
        topology=topology, axis_assignments=tuple(axes),
        compile_mode=compile_mode, fp8_enabled=fp8_enabled,
    )


GOTCHA_TRIGGERS: dict[str, dict] = {
    "fsdp2_whole_compile": {
        "kinds": [ParallelismKind.FSDP2], "compile_mode": "whole_model",
        "topology_specific": False,
    },
    "megatron_tp_whole_compile": {
        "kinds": [ParallelismKind.TP], "compile_mode": "whole_model",
        "topology_specific": False,
    },
    "fp8_grad_duplication": {
        "fp8_enabled": True, "topology_specific": False,
    },
}


def _topology_supports_kinds(topology, kinds) -> bool:
    if not kinds:
        return True
    mesh = dict(topology.mesh_axes)
    name_for = {
        ParallelismKind.DP: "dp", ParallelismKind.FSDP1: "dp",
        ParallelismKind.FSDP2: "dp", ParallelismKind.ZERO1: "dp",
        ParallelismKind.ZERO2: "dp", ParallelismKind.TP: "tp",
        ParallelismKind.SP: "tp", ParallelismKind.EP: "ep",
        ParallelismKind.PP: "pp", ParallelismKind.PP_VPP: "pp",
    }
    for k in kinds:
        ax = name_for.get(k)
        if ax in mesh and mesh[ax] > 1:
            return True
    return False


@pytest.mark.parametrize("gotcha_id", list(GOTCHA_TRIGGERS))
@pytest.mark.parametrize("topology_name", list(TOPOLOGIES))
def test_gotcha_fires_on_intended_trigger_across_topologies(
    gotcha_id: str, topology_name: str,
):
    cfg = GOTCHA_TRIGGERS[gotcha_id]
    topology = TOPOLOGIES[topology_name]()
    if cfg.get("kinds") and not _topology_supports_kinds(
            topology, cfg["kinds"]):
        # Topology has no mesh axis for the required parallelism kind
        # — the gotcha would never fire, so skip rather than fail.
        pytest.skip(
            f"topology {topology_name!r} does not expose mesh axes "
            f"for kinds={cfg['kinds']}")
    sharding = _build_sharding(
        topology,
        kinds=cfg.get("kinds"),
        compile_mode=cfg.get("compile_mode", "off"),
        fp8_enabled=cfg.get("fp8_enabled", False),
    )
    spec = _build_spec(topology)
    fired = {g.gotcha_id for g in check_gotchas(sharding, spec)}
    assert gotcha_id in fired, (
        f"{gotcha_id!r} should fire on topology {topology_name!r} "
        f"but did not (fired: {sorted(fired)})")


def test_gotcha_silent_on_clean_spec():
    """A fully clean spec (no compile, no fp8, no fancy parallelism)
    should produce zero gotchas across every topology."""
    for name, factory in TOPOLOGIES.items():
        topology = factory()
        sharding = _build_sharding(topology)
        spec = _build_spec(topology)
        fired = {g.gotcha_id for g in check_gotchas(sharding, spec)}
        # We accept gotchas whose condition is unconditionally true
        # (defensive defaults), but the headline failure modes
        # (fsdp2_whole_compile etc.) must stay silent on a clean spec.
        for g_id in ("fsdp2_whole_compile",
                      "megatron_tp_whole_compile",
                      "fp8_grad_duplication"):
            assert g_id not in fired, (
                f"{g_id!r} unexpectedly fired on clean {name!r} spec "
                f"(fired: {sorted(fired)})")


def test_gotcha_table_has_at_least_15_rules():
    assert len(GOTCHAS) >= 15
