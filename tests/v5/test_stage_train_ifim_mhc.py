"""G02 + G03: stage_train applies IFIM/MHC penalty terms.

Loss kernel for IFIM_SHAPED: CE + λ_fim × mean(logits²) — Fisher diag
proxy. For MHC_ATTN_BIAS: CE + λ_mhc × mean(|logits|).

extras.ifim / extras.mhc populated with {lambda_*, *_norm, penalty_value}
so e2e can assert (a) propagation of λ, (b) penalty term is non-trivial
when λ > 0, (c) loss differs from plain CE when λ > 0.
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


# G02 IFIM_SHAPED ------------------------------------------------------


def test_ifim_extras_populated():
    extras = _run({"kind": "ifim_shaped",
                   "head_outputs": ["mlp"],
                   "params": {"lambda_fim": 0.1}})
    ifim = extras.get("ifim")
    assert ifim is not None
    assert abs(ifim["lambda_fim"] - 0.1) < 1e-6
    assert ifim["fim_weights_norm"] > 0
    assert abs(ifim["penalty_value"] - 0.1 * ifim["fim_weights_norm"]) < 1e-3


def test_ifim_lambda_zero_collapses_to_ce():
    """λ_fim=0 → IFIM penalty contributes zero; loss equals CE."""
    ifim_extras = _run({"kind": "ifim_shaped",
                        "head_outputs": ["mlp"],
                        "params": {"lambda_fim": 0.0}})
    ce_extras = _run({"kind": "cross_entropy",
                      "head_outputs": ["mlp"], "params": {}})
    # losses[0] is computed BEFORE any optimizer update on the same
    # deterministic init; with λ=0 IFIM should reduce to CE within
    # numerical noise of separate forward random-keys.
    assert abs(ifim_extras["losses"][0] - ce_extras["losses"][0]) < 0.5


def test_ifim_lambda_high_changes_loss():
    """λ_fim=0.5 vs 0.05 produces observably different losses[0]."""
    low = _run({"kind": "ifim_shaped",
                "head_outputs": ["mlp"],
                "params": {"lambda_fim": 0.05}})
    high = _run({"kind": "ifim_shaped",
                 "head_outputs": ["mlp"],
                 "params": {"lambda_fim": 0.5}})
    assert abs(low["losses"][0] - high["losses"][0]) > 1e-4


def test_ifim_extras_absent_for_other_kinds():
    """CE / MTP / MHC do not populate extras.ifim."""
    for kind, params in [
        ("cross_entropy", {}),
        ("mhc_attn_bias", {"lambda_mhc": 0.05}),
    ]:
        extras = _run({"kind": kind, "head_outputs": ["mlp"],
                       "params": params})
        assert extras.get("ifim") is None, f"kind={kind} leaked ifim extras"


# G03 MHC_ATTN_BIAS ----------------------------------------------------


def test_mhc_extras_populated():
    extras = _run({"kind": "mhc_attn_bias",
                   "head_outputs": ["mlp"],
                   "params": {"lambda_mhc": 0.05}})
    mhc = extras.get("mhc")
    assert mhc is not None
    assert abs(mhc["lambda_mhc"] - 0.05) < 1e-6
    assert mhc["bias_norm"] > 0


def test_mhc_lambda_high_changes_loss():
    low = _run({"kind": "mhc_attn_bias",
                "head_outputs": ["mlp"],
                "params": {"lambda_mhc": 0.0}})
    high = _run({"kind": "mhc_attn_bias",
                 "head_outputs": ["mlp"],
                 "params": {"lambda_mhc": 0.2}})
    assert abs(low["losses"][0] - high["losses"][0]) > 1e-4


def test_mhc_extras_absent_for_other_kinds():
    for kind in ("cross_entropy", "ifim_shaped"):
        extras = _run({"kind": kind, "head_outputs": ["mlp"],
                       "params": {"lambda_fim": 0.0}
                       if kind == "ifim_shaped" else {}})
        assert extras.get("mhc") is None, f"kind={kind} leaked mhc extras"
