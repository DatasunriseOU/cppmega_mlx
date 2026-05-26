"""Unit tests for mechanistic activation probing hooks and telemetry in runner/stages.py.

Asserts:
  - stage_verify_build_spec correctly flags research hooks targeting non-existent nodes as WARNINGs.
  - stage_train correctly executes autograd-safe research probes (Monitor, Sparsity, SAE, Causal Patching).
  - Analytical statistics are published to train_event_bus per step in real-time.
"""

from __future__ import annotations

import threading
import time
import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runtime import train_event_bus as bus


@pytest.fixture(autouse=True)
def _reset():
    bus.reset()
    yield
    bus.reset()


def _spec() -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _collect_events(run_id: str, out: list, sentinel_seen: list) -> None:
    q = bus.subscribe(run_id)
    while True:
        try:
            ev = q.get(timeout=3.0)
        except Exception:
            break
        if ev is None:
            sentinel_seen.append(True)
            break
        out.append(ev)


def test_research_hooks_diagnostic_warning():
    """Verify that hooks targeting nonexistent nodes raise warnings during verify stage."""
    spec = _spec()
    stage_options = {
        "verify_build_spec": {
            "research_hooks": {
                "nonexistent_node": {"type": "monitor"}
            }
        }
    }
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec"],
        "stage_options": stage_options,
    }))
    
    verify_stage = next(s for s in report.stages if s.name == "verify_build_spec")
    assert verify_stage.status == "ok", verify_stage.error
    assert verify_stage.warnings >= 1
    
    # Run with existent node -> should have 0 warnings.
    stage_options_clean = {
        "verify_build_spec": {
            "research_hooks": {
                "attn": {"type": "monitor"}
            }
        }
    }
    report_clean = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec"],
        "stage_options": stage_options_clean,
    }))
    verify_stage_clean = next(s for s in report_clean.stages if s.name == "verify_build_spec")
    assert verify_stage_clean.warnings == 0


def test_research_hooks_train_telemetry():
    """Verify that train telemetry publishes analytical statistics for research hooks."""
    events: list[dict] = []
    sentinel: list[bool] = []
    run_id = "rid-research-hooks"
    
    t = threading.Thread(target=_collect_events,
                         args=(run_id, events, sentinel),
                         daemon=True)
    t.start()
    
    # Give subscriber a chance to register
    time.sleep(0.05)
    
    stage_options = {
        "train": {
            "num_steps": 2,
            "run_id": run_id,
            "abort_token": run_id,
            "step_delay": 0.0,
            "research_hooks": {
                "attn": {
                    "type": "sparsity",
                    "threshold": 0.05
                },
                "mlp": {
                    "type": "causal",
                    "causal_factor": 1.2,
                    "noise_level": 0.01
                }
            }
        }
    }
    
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": stage_options,
    }))
    
    t.join(timeout=5.0)
    train_stage = next(s for s in rep.stages if s.name == "train")
    assert train_stage.status == "ok", train_stage.error
    assert sentinel == [True]
    assert len(events) == 2
    
    for ev in events:
        assert "research_hooks" in ev
        hooks_data = ev["research_hooks"]
        assert isinstance(hooks_data, dict)
        
        # Verify stats are computed for both active hooks
        for node in ["attn", "mlp"]:
            assert node in hooks_data
            node_stats = hooks_data[node]
            assert isinstance(node_stats, dict)
            
            # Verify Monitor stats
            for k in ["mean", "std", "l2_norm", "min", "max"]:
                assert k in node_stats
                assert isinstance(node_stats[k], float)
            
            # Verify Sparsity stat
            assert "sparsity" in node_stats
            assert 0.0 <= node_stats["sparsity"] <= 1.0
            
            # Verify SAE stats
            assert "sae_l0_sparsity" in node_stats
            assert 0.0 <= node_stats["sae_l0_sparsity"] <= 1.0
            assert "sae_reconstruction_mse" in node_stats
            assert isinstance(node_stats["sae_reconstruction_mse"], float)
            assert node_stats["sae_reconstruction_mse"] >= 0.0
