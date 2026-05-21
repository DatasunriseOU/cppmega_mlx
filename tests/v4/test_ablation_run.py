"""E7-11 tests: ablation.run RPC."""

from __future__ import annotations

from cppmega_v4.architectures import build_preset_specs
from cppmega_v4.jsonrpc import dispatch
from cppmega_v4.jsonrpc.schema import METHOD_REGISTRY


def _base_spec(preset: str = "llama3_8b") -> dict:
    specs = build_preset_specs(preset, hidden_size=128)
    return {
        "graph": {
            "nodes": [
                {"id": s.get("name"), "kind": s["kind"],
                 "params": s.get("params", {})}
                for s in specs
            ],
            "edges": [
                {"src": specs[i].get("name"),
                 "dst": specs[i + 1].get("name")}
                for i in range(len(specs) - 1)
            ],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128, "nh": 2, "nkv": 1,
                    "head_dim": 64, "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [specs[-1].get("name")]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 3e-4,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    }


def test_method_registered():
    assert "ablation.run" in METHOD_REGISTRY


def test_activation_ablation_runs_three_variants():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "ablation.run",
        "params": {
            "base_spec": _base_spec(),
            "ablation_axis": "activation",
            "variants": ["glu", "swiglu", "gelu"],
            "num_steps": 2,
        },
    })
    assert resp.error is None, resp.error
    res = resp.result
    assert len(res["results"]) == 3
    names = {r["variant"] for r in res["results"]}
    assert names == {"glu", "swiglu", "gelu"}
    # At least one variant should succeed
    assert any(r["status"] == "ok" for r in res["results"])


def test_optimizer_ablation_runs_adamw_and_sgd():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "ablation.run",
        "params": {
            "base_spec": _base_spec(),
            "ablation_axis": "optimizer",
            "variants": ["adamw", "sgd"],
            "num_steps": 2,
        },
    })
    assert resp.error is None, resp.error
    res = resp.result
    assert len(res["results"]) == 2


def test_optimizer_ablation_lion_pulls_recommended_params():
    """The Lion variant must trigger _mutate's catalogue lookup; the
    spec's first group should end up with Lion's recommended lr (1e-4)
    even if the train stage itself uses its own (AdamW-backed) loop."""
    from cppmega_v4.jsonrpc.ablation_method import _mutate
    from cppmega_v4.jsonrpc.schema import VerifyParams
    mutated = _mutate(
        VerifyParams.model_validate(_base_spec()),
        "optimizer", "lion",
    )
    assert mutated.optim.kind == "lion"
    assert mutated.optim.groups[0].lr == 1e-4
    assert mutated.optim.groups[0].weight_decay == 0.01


def test_ranked_by_final_loss_sorted_ascending():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "ablation.run",
        "params": {
            "base_spec": _base_spec(),
            "ablation_axis": "activation",
            "variants": ["glu", "swiglu"],
            "num_steps": 2,
        },
    })
    assert resp.error is None
    ranked = resp.result["ranked_by_final_loss"]
    ok_results = [r for r in resp.result["results"]
                  if r["status"] == "ok" and r["losses"]]
    if len(ok_results) >= 2:
        final_losses = {r["variant"]: r["losses"][-1] for r in ok_results}
        prev = -float("inf")
        for name in ranked:
            cur = final_losses[name]
            assert cur >= prev
            prev = cur


def test_failed_variant_does_not_abort_others():
    """Pass a deliberately bogus optimizer name → that variant fails;
    other variants still complete."""
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "ablation.run",
        "params": {
            "base_spec": _base_spec(),
            "ablation_axis": "optimizer",
            "variants": ["adamw", "totally_made_up_optim"],
            "num_steps": 2,
        },
    })
    assert resp.error is None
    statuses = {r["variant"]: r["status"] for r in resp.result["results"]}
    assert statuses["adamw"] == "ok"
    assert statuses["totally_made_up_optim"] == "fail"


def test_baseline_variant_is_first_in_variants_list():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "ablation.run",
        "params": {
            "base_spec": _base_spec(),
            "ablation_axis": "activation",
            "variants": ["glu", "swiglu"],
            "num_steps": 2,
        },
    })
    assert resp.result["baseline_variant"] == "glu"


def test_elapsed_ms_total_positive():
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "ablation.run",
        "params": {
            "base_spec": _base_spec(),
            "ablation_axis": "activation",
            "variants": ["glu"],
            "num_steps": 2,
        },
    })
    assert resp.error is None
    assert resp.result["elapsed_ms_total"] > 0
