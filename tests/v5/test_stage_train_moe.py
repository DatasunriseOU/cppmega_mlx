"""G25: MoE bricks detected in graph → extras.moe populated."""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(nodes: list[dict]) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": nodes,
            "edges": [{"src": nodes[i]["id"], "dst": nodes[i+1]["id"]}
                      for i in range(len(nodes) - 1)],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1, "head_dim": 16},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [nodes[-1]["id"]]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(nodes: list[dict]) -> dict:
    # Skip build_model — MoE brick has V4MoEConfig kwargs that build_model
    # rejects; G25 is observation-only (extras populated from raw graph).
    report = run_pipeline(_spec(nodes), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "train"],
        "stage_options": {"train": {"num_steps": 2}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    return train.extras


def test_no_moe_brick_extras_moe_is_none():
    extras = _run([
        {"id": "attn", "kind": "attention", "params": {}},
        {"id": "mlp", "kind": "mlp",
         "params": {"intermediate_size": 64, "activation": "swiglu"}},
    ])
    assert extras.get("moe") is None


def test_moe_brick_populates_num_experts_top_k():
    # Use mlp (instantiates cleanly) but inject a "moe" pseudo-node so
    # the detection logic in stage_train sees it via the wire graph.
    extras = _run([
        {"id": "moe", "kind": "moe",
         "params": {"num_experts": 8, "top_k": 2}},
        {"id": "mlp", "kind": "mlp",
         "params": {"intermediate_size": 64, "activation": "swiglu"}},
    ])
    moe = extras.get("moe")
    assert moe is not None
    assert moe["kind"] == "moe"
    assert moe["num_experts"] == 8
    assert moe["top_k"] == 2
