"""E7-2 tests: dim auto-adjust feedback (inference_log)."""

from __future__ import annotations

from cppmega_v4.spec.inference_log import build_inference_log
from cppmega_v4.jsonrpc import dispatch
from cppmega_v4.architectures import build_preset_specs


def _graph(preset: str, hidden: int = 128):
    specs = build_preset_specs(preset, hidden_size=hidden)
    return {
        "nodes": [
            {"id": s.get("name"), "kind": s["kind"],
             "params": s.get("params", {})}
            for s in specs
        ],
        "edges": [
            {"src": specs[i].get("name"), "dst": specs[i + 1].get("name")}
            for i in range(len(specs) - 1)
        ],
    }


def test_attention_auto_derives_num_heads_from_hidden():
    g = {"nodes": [{"id": "attn_0", "kind": "attention", "params": {}}]}
    log = build_inference_log(g, {"H": 128, "head_dim": 64})
    nh = [e for e in log if e.brick == "attn_0" and e.param == "num_heads"]
    assert nh and nh[0].source == "auto"
    assert nh[0].value == 2  # 128 / 64
    assert "H=128/head_dim=64" in nh[0].reason


def test_attention_user_override_num_heads_marked_user():
    g = {"nodes": [{"id": "attn_0", "kind": "attention",
                    "params": {"num_heads": 16}}]}
    log = build_inference_log(g, {"H": 128, "head_dim": 64})
    nh = [e for e in log if e.param == "num_heads"]
    assert nh[0].source == "user"
    assert nh[0].value == 16


def test_mlp_intermediate_size_defaults_to_4_times_H():
    g = {"nodes": [{"id": "mlp_0", "kind": "mlp", "params": {}}]}
    log = build_inference_log(g, {"H": 128})
    inter = [e for e in log if e.param == "intermediate_size"]
    assert inter[0].source == "auto"
    assert inter[0].value == 512


def test_mlp_activation_defaults_to_glu():
    g = {"nodes": [{"id": "mlp_0", "kind": "mlp", "params": {}}]}
    log = build_inference_log(g, {"H": 128})
    act = [e for e in log if e.param == "activation"]
    assert act[0].source == "auto"
    assert act[0].value == "glu"


def test_mlp_activation_override_marked_user():
    g = {"nodes": [{"id": "mlp_0", "kind": "mlp",
                    "params": {"activation": "swiglu"}}]}
    log = build_inference_log(g, {"H": 128})
    act = [e for e in log if e.param == "activation"]
    assert act[0].source == "user"
    assert act[0].value == "swiglu"


def test_moe_picks_up_dim_env_defaults():
    g = {"nodes": [{"id": "moe_0", "kind": "moe", "params": {}}]}
    log = build_inference_log(g, {"H": 128, "num_experts": 8, "top_k": 2})
    ne = [e for e in log if e.param == "num_experts"]
    assert ne[0].source == "auto"
    assert ne[0].value == 8


def test_other_brick_kinds_emit_user_rows_for_provided_params():
    g = {"nodes": [{"id": "mamba_0", "kind": "mamba3",
                    "params": {"d_state": 16}}]}
    log = build_inference_log(g, {"H": 128})
    assert any(e.param == "d_state" and e.source == "user" for e in log)


def test_verify_response_carries_inference_log():
    """End-to-end: verify RPC must include inference_log."""
    g = _graph("llama3_8b")
    resp = dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "verify",
        "params": {
            "graph": g,
            "dim_env": {"B": 1, "S": 8, "H": 128, "nh": 2, "nkv": 1,
                        "head_dim": 64, "num_experts": 4, "top_k": 2},
            "loss": {"kind": "cross_entropy",
                     "head_outputs": [g["nodes"][-1]["id"]]},
            "optim": {"kind": "adamw",
                      "groups": [{"matcher": "all", "lr": 3e-4,
                                  "weight_decay": 0.01,
                                  "betas": [0.9, 0.95]}]},
        },
    })
    assert resp.error is None
    log = resp.result["inference_log"]
    assert len(log) >= 2  # at least num_heads + head_dim for attention
    assert all("brick" in e and "param" in e and "source" in e for e in log)


def test_inference_log_sources_are_user_or_auto():
    g = _graph("llama3_8b")
    log = build_inference_log(
        g, {"H": 128, "head_dim": 64, "num_experts": 4, "top_k": 2})
    for e in log:
        assert e.source in ("user", "auto"), e


def test_reasons_are_non_empty():
    g = _graph("llama3_8b")
    log = build_inference_log(g, {"H": 128, "head_dim": 64})
    for e in log:
        assert e.reason, e
