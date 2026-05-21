"""G01: stage_train honors LossKind.MTP_WEIGHTED with K extra LM heads.

Previously the kind was only echoed in extras; the loss kernel stayed
hardcoded CE. Now stage_train builds K Linear heads, computes
Σ β_i × CE(head_i, shifted_labels(targets, i)), and reports per-head
losses in extras.mtp = {k, betas, per_head_losses}.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec_with_loss(loss_payload: dict) -> VerifyParams:
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
        "loss": loss_payload,
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
    })


def _run(loss_payload: dict, num_steps: int = 2) -> dict:
    spec = _spec_with_loss(loss_payload)
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_mtp_extras_populated_for_k2():
    """MTP_WEIGHTED k=2 produces extras.mtp.{k, betas, per_head_losses}."""
    extras = _run({
        "kind": "mtp_weighted",
        "head_outputs": ["mlp"],
        "params": {"k": 2, "beta": 0.6},
    })
    mtp = extras.get("mtp")
    assert mtp is not None
    assert mtp["k"] == 2
    assert len(mtp["betas"]) == 2
    assert all(abs(b - 0.6) < 1e-6 for b in mtp["betas"])
    assert len(mtp["per_head_losses"]) == 2
    assert all(loss > 0 and loss < 1e4 for loss in mtp["per_head_losses"])


def test_mtp_k3_three_heads():
    """K=3 produces 3 heads + 3 betas + 3 per_head_losses."""
    extras = _run({
        "kind": "mtp_weighted",
        "head_outputs": ["mlp"],
        "params": {"k": 3, "beta": 0.33},
    })
    assert extras["mtp"]["k"] == 3
    assert len(extras["mtp"]["per_head_losses"]) == 3


def test_mtp_k1_collapses_to_single_head():
    """K=1 is degenerate — runs but produces one head."""
    extras = _run({
        "kind": "mtp_weighted",
        "head_outputs": ["mlp"],
        "params": {"k": 1, "beta": 1.0},
    })
    assert extras["mtp"]["k"] == 1
    assert len(extras["mtp"]["per_head_losses"]) == 1


def test_cross_entropy_does_not_emit_mtp_extras():
    """Plain CE → extras.mtp is None."""
    extras = _run({
        "kind": "cross_entropy",
        "head_outputs": ["mlp"],
        "params": {},
    })
    assert extras.get("mtp") is None


def test_mtp_per_head_losses_differ_across_shifts():
    """Each head sees different shifted labels → per-head losses differ
    by more than float noise. Proves K heads were actually wired with
    distinct supervision, not all targeting the same labels."""
    extras = _run({
        "kind": "mtp_weighted",
        "head_outputs": ["mlp"],
        "params": {"k": 3, "beta": 0.5},
    }, num_steps=4)
    losses = extras["mtp"]["per_head_losses"]
    # At least one pair should differ in absolute terms — same head on
    # different shifted labels MUST give different CE for random init.
    max_diff = max(abs(a - b) for i, a in enumerate(losses)
                   for b in losses[i + 1:])
    assert max_diff > 1e-4, \
        f"per_head_losses {losses} too similar — K heads may share labels"


def test_mtp_explicit_beta_i_overrides_flat_beta():
    """When beta_0..beta_{k-1} are explicitly set they override flat beta."""
    extras = _run({
        "kind": "mtp_weighted",
        "head_outputs": ["mlp"],
        "params": {"k": 2, "beta_0": 0.8, "beta_1": 0.2},
    })
    assert extras["mtp"]["betas"] == [0.8, 0.2]
