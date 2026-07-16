"""G15: long-run convergence — N=100 with smoothed loss observability."""

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


def _run(num_steps: int) -> dict:
    report = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_losses_smoothed_present_short_run():
    extras = _run(8)
    assert "losses_smoothed" in extras
    assert len(extras["losses_smoothed"]) == 8


def test_long_run_n100_smoothed_shape_correct():
    extras = _run(100)
    assert len(extras["losses"]) == 100
    assert len(extras["losses_smoothed"]) == 100
    # Smoothed[0] equals losses[0] (window of 1 at start)
    assert abs(extras["losses_smoothed"][0] - extras["losses"][0]) < 1e-4


def test_long_run_smoothed_curve_monotone_in_window():
    """Smoothed (window=10) loss over N=100 should be less noisy than
    raw — its step-to-step total variation should be lower than raw's.

    Global variance is not a valid assertion here: the training curve has a
    strong downward trend, and a causal window introduces a bounded lag at
    the endpoints that can make variance increase slightly while still
    removing high-frequency noise.
    """
    extras = _run(100)
    raw = extras["losses"]
    smoothed = extras["losses_smoothed"]
    raw_variation = sum(abs(right - left) for left, right in zip(raw, raw[1:]))
    smoothed_variation = sum(
        abs(right - left) for left, right in zip(smoothed, smoothed[1:])
    )
    assert smoothed_variation < raw_variation, (
        f"smoothed variation {smoothed_variation} should be below "
        f"raw variation {raw_variation}"
    )
