"""V3-1: stage_train honors OptimKind from the spec.

Every OptimKind value must produce a real mlx optimizer instance, push
loss/weights through one+ gradient step, and surface its kind in extras.
The previous implementation hardcoded ``optim.AdamW`` — the UI dropdown
was decorative. These tests pin the dispatch so a regression flips a row.
"""

from __future__ import annotations

import pytest

from cppmega_v4.buildspec import OptimKind
from cppmega_v4.buildspec import optim_spec as _opt_factories
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runner.stages import _build_optimizer, _summarize_model


_FACTORY_FOR_KIND = {
    OptimKind.ADAMW: lambda lr: _opt_factories.adamw(lr=lr),
    OptimKind.LION: lambda lr: _opt_factories.lion(lr=lr),
    OptimKind.LION_8BIT: lambda lr: _opt_factories.lion8bit(lr=lr),
    OptimKind.ADAM_8BIT: lambda lr: _opt_factories.adam8bit(lr=lr),
    OptimKind.MUON: lambda lr: _opt_factories.muon(lr=lr),
    OptimKind.MUON_ADAMW_HYBRID: lambda lr:
        _opt_factories.muon_adamw_hybrid(muon_lr=lr, adam_lr=lr),
    OptimKind.SGD: lambda lr: _opt_factories.sgd(lr=lr),
}


def _make_optim(kind: OptimKind, lr: float):
    return _FACTORY_FOR_KIND[kind](lr)


_GROUP_FOR_KIND = {
    "adamw": lambda lr: {"matcher": "all", "lr": lr,
                         "weight_decay": 0.01, "betas": [0.9, 0.95]},
    "lion": lambda lr: {"matcher": "all", "lr": lr,
                        "weight_decay": 0.0, "betas": [0.9, 0.99]},
    "lion8bit": lambda lr: {"matcher": "all", "lr": lr,
                            "weight_decay": 0.0, "betas": [0.9, 0.99]},
    "adam8bit": lambda lr: {"matcher": "all", "lr": lr,
                            "weight_decay": 0.01, "betas": [0.9, 0.999]},
    "muon": lambda lr: {"matcher": "all", "lr": lr,
                        "weight_decay": 0.0, "ns_steps": 5},
    "muon_adamw_hybrid": lambda lr: {"matcher": "all", "lr": lr,
                                     "weight_decay": 0.0, "ns_steps": 5},
    "sgd": lambda lr: {"matcher": "all", "lr": lr, "weight_decay": 0.0},
}


def _verify_params(kind_str: str, lr: float = 1e-3) -> VerifyParams:
    """Minimal valid VerifyParams with attention+mlp bricks and the chosen
    optimizer kind. Mirrors the wire form sent by the UI."""
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64, "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1, "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {
            "kind": kind_str,
            "groups": [_GROUP_FOR_KIND[kind_str](lr)],
        },
    })


def _run_train(kind_str: str, lr: float = 1e-3) -> dict:
    spec = _verify_params(kind_str, lr=lr)
    report = run_pipeline(spec, Pipeline.from_dict({"stages": [
        "parse", "verify_build_spec", "build_model", "train",
    ]}))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", \
        f"stage_train failed for {kind_str}: {train.error}"
    return train.extras


# ---------------------------------------------------------------------------
# Unit tests on _build_optimizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind,expected", [
    (OptimKind.ADAMW, "adamw"),
    (OptimKind.LION, "lion"),
    (OptimKind.LION_8BIT, "lion8bit"),
    (OptimKind.ADAM_8BIT, "adam8bit"),
    (OptimKind.MUON, "muon"),
    (OptimKind.MUON_ADAMW_HYBRID, "muon_adamw_hybrid"),
    (OptimKind.SGD, "sgd"),
])
def test_build_optimizer_dispatches_every_kind(kind, expected):
    """_build_optimizer maps every OptimKind to (optimizer, kind_str)."""
    spec = _make_optim(kind, lr=1e-4)
    opt, label = _build_optimizer(spec, base_lr=1e-4)
    assert label == expected
    assert callable(getattr(opt, "update", None))


def test_build_optimizer_none_falls_back_to_adamw():
    opt, label = _build_optimizer(None, base_lr=1e-3)
    assert label == "adamw"
    assert callable(getattr(opt, "update", None))


def test_build_optimizer_string_kind_works():
    """Wire-form OptimSpecPayload carries kind as str; _build_optimizer
    must accept both Enum and str."""
    from cppmega_v4.jsonrpc.schema import OptimSpecPayload
    payload = OptimSpecPayload.model_validate({
        "kind": "lion",
        "groups": [{"matcher": "all", "lr": 1e-4}],
    })
    opt, label = _build_optimizer(payload, base_lr=1e-4)
    assert label == "lion"


# ---------------------------------------------------------------------------
# End-to-end through Pipeline (UI → wire → stage_train)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind_str,lr", [
    ("adamw", 1e-3),
    ("lion", 1e-4),    # Lion sign-based, low lr per Chen et al.
    ("lion8bit", 1e-4),
    ("adam8bit", 1e-3),
    ("sgd", 1e-1),     # SGD needs larger lr to move weights at 2 steps
])
def test_stage_train_surfaces_optimizer_kind_in_extras(kind_str, lr):
    """The optimizer label in extras must match the wire-form kind."""
    extras = _run_train(kind_str, lr=lr)
    assert extras["optimizer_kind"] == kind_str, \
        f"UI selected {kind_str} but extras reports {extras['optimizer_kind']}"
    assert extras["weight_delta_norm"] > 0.0, "weights must have moved"
    assert all(
        loss_item == loss_item for loss_item in extras["losses"]
    ), "losses must be finite"


def test_stage_train_muon_uses_muon():
    """Muon is sign-of-grad NS-orthogonalised: fundamentally different math
    from AdamW. extras must report 'muon' so UI tests can assert propagation."""
    extras = _run_train("muon", lr=1e-3)
    assert extras["optimizer_kind"] == "muon"
    assert extras["weight_delta_norm"] > 0.0


def test_stage_train_model_summary_includes_optimizer():
    """model_summary.optimizer_kind mirrors extras.optimizer_kind so a
    single DOM read picks up the propagated value."""
    extras = _run_train("lion", lr=1e-4)
    assert extras["model_summary"]["optimizer_kind"] == "lion"


def test_build_optimizer_handles_optim_payload_kind_string():
    """Wire-form OptimSpecPayload sends OptimKind as a plain string.
    _build_optimizer must normalise it back to OptimKind without crashing.
    Pinned because mismatched typing here silently falls into the AdamW
    default path and hides V3-1."""
    spec = _verify_params("lion", lr=1e-4)
    opt, label = _build_optimizer(spec.optim, base_lr=1e-4)
    assert label == "lion"


# ---------------------------------------------------------------------------
# Unit tests on _summarize_model
# ---------------------------------------------------------------------------


def test_summarize_model_extracts_activation_and_norms():
    """_summarize_model snapshots what user clicked in BrickContextPanel."""
    spec = _verify_params("adamw")
    # spec.graph is GraphSpec from schema; _summarize_model uses .nodes + .params
    summary = _summarize_model(spec, "adamw", "constant")
    assert summary["mlp_activation"] == "swiglu"
    assert summary["optimizer_kind"] == "adamw"
    assert summary["schedule_kind"] == "constant"
    assert summary["num_brick_kinds"] == 2  # attention + mlp


def test_summarize_model_handles_missing_bricks():
    """No mlp/attention bricks → None defaults, no crash."""
    spec = VerifyParams.model_validate({
        "graph": {"nodes": [{"id": "x", "kind": "embedding", "params": {}}],
                  "edges": []},
        "dim_env": {"H": 32},
        "loss": {"kind": "cross_entropy", "head_outputs": ["x"]},
        "optim": {"kind": "adamw",
                  "groups": [_GROUP_FOR_KIND["adamw"](1e-3)]},
    })
    summary = _summarize_model(spec, "adamw", "constant")
    assert summary["mlp_activation"] is None
    assert summary["num_brick_kinds"] == 1
