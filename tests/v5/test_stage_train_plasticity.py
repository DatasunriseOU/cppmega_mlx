"""V7-E33..E35: FIRE/DASH/ReDo are wired into stage_train."""

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
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1,
                    "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(opts: dict) -> dict:
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_plasticity_disabled_by_default_empty_trace():
    extras = _run({"num_steps": 2})
    assert extras["plasticity"] == {}


def test_fire_fires_at_step_and_modifies_keys():
    extras = _run({"num_steps": 3, "fire_at_step": 1})
    pl = extras["plasticity"]
    assert pl.get("fire_fired_at_step") == 1
    assert isinstance(pl.get("fire_keys_modified"), list)
    # FIRE only touches 2D params; the tiny attn+mlp model has at
    # least one (mlp.up / mlp.down).
    assert len(pl["fire_keys_modified"]) >= 1


def test_dash_runs_periodically_and_records_steps():
    extras = _run({"num_steps": 4, "dash_every": 2})
    pl = extras["plasticity"]
    # dash_every=2 over 4 steps → fires on (step+1)%2==0 → steps 1, 3.
    assert pl.get("dash_steps_applied") == [1, 3]
    assert pl.get("dash_last_keys_count", 0) > 0


def test_redo_does_not_fire_without_layer_map():
    extras = _run({"num_steps": 3, "redo_every": 1})
    pl = extras["plasticity"]
    # Without redo_layer_map the dormant-neuron recycling skips.
    assert pl.get("redo_steps_applied") is None
