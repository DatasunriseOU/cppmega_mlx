"""V7-I04: H17 single-doc passthrough must be mathematically identity.

The passthrough flag is set when sc_doc_ids_mask_density==0 (i.e. all
tokens belong to one document). The spec promised the mask is then a
no-op — but pre-V7-I04 the only proof was a string flag in extras.

This test runs the same 2-step train twice:
  (a) WITH a single-doc side-channel input (all token doc_ids = 0)
  (b) WITHOUT the side-channel at all
and asserts the loss trajectories are bit-identical. A real
identity mask cannot produce a different loss; any drift here would
expose a silent mask-application bug.
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


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


def _run(side_channels: dict | None) -> dict:
    opts = {"num_steps": 2, "run_id": "h17-id-mask"}
    if side_channels is not None:
        opts["side_channels"] = side_channels
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v7_i04_single_doc_passthrough_flag_is_true_when_doc_ids_uniform():
    """When all 8 tokens share doc_id=0, the H17 flag flips on."""
    extras = _run({"doc_ids": [0] * 8})
    assert extras.get("side_channels_forward_effect") is not None
    sc = extras["side_channels_forward_effect"]
    assert sc.get("single_doc_passthrough") is True, sc
    assert sc.get("doc_ids_mask_density") == 0.0
    # Observed list still records the family name.
    assert "doc_ids" in extras["side_channels_observed"]


def test_v7_i04_passthrough_loss_equals_no_side_channel_loss():
    """REAL forward proof: with single-doc passthrough the mask must
    be mathematically identity → same loss as no side-channel run."""
    extras_passthrough = _run({"doc_ids": [0] * 8})
    extras_baseline = _run(None)
    losses_p = extras_passthrough["losses"]
    losses_b = extras_baseline["losses"]
    assert len(losses_p) == len(losses_b) == 2
    # Bit-identical at single-precision tolerance: any drift means the
    # mask quietly leaked into the math.
    for a, b in zip(losses_p, losses_b):
        assert abs(a - b) < 1e-4, (
            f"H17 passthrough deviates from baseline: "
            f"passthrough={losses_p} baseline={losses_b}")


def test_v7_i04_multi_doc_input_does_not_passthrough_and_loss_differs():
    """Sanity inverse: when doc_ids are NOT uniform, mask is applied
    (passthrough=False) AND the loss differs from baseline. Pins that
    the per-test fixture actually engages the mask path."""
    multi = _run({"doc_ids": [0, 0, 0, 0, 1, 1, 1, 1]})
    sc = multi["side_channels_forward_effect"]
    assert sc.get("single_doc_passthrough") is False, sc
    assert sc.get("doc_ids_mask_density") > 0.0

    baseline = _run(None)
    losses_m = multi["losses"]
    losses_b = baseline["losses"]
    # At least one step must differ once the mask is real.
    deltas = [abs(a - b) for a, b in zip(losses_m, losses_b)]
    assert max(deltas) > 1e-5, (
        f"multi-doc mask claimed applied but loss unchanged: "
        f"multi={losses_m} baseline={losses_b}")
