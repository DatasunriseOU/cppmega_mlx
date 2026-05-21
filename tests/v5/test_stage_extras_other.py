"""G21: dry_forward / loss_smoke / optimizer_smoke return rich extras."""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec() -> VerifyParams:
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
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _stages(stage_names: list[str]) -> dict[str, dict]:
    r = run_pipeline(_spec(), Pipeline.from_dict({"stages": stage_names}))
    return {s.name: s.to_dict() for s in r.stages}


def test_dry_forward_rich_extras():
    out = _stages(["parse", "verify_build_spec", "dry_forward"])
    df = out["dry_forward"]
    assert df["batch"] == 1
    assert df["seq_len"] == 8
    assert df["hidden"] == 32
    assert df["num_nodes"] >= 1


def test_loss_smoke_rich_extras():
    out = _stages(["parse", "verify_build_spec", "loss_smoke"])
    ls = out["loss_smoke"]
    assert ls["loss_finite"] is True
    assert ls["seq_len"] == 8
    assert isinstance(ls["loss_value"], float)


def test_optimizer_smoke_rich_extras():
    out = _stages(["parse", "verify_build_spec", "optimizer_smoke"])
    os_ = out["optimizer_smoke"]
    assert os_["optimizer_kind"] == "adamw"
    assert os_["num_groups"] == 1
