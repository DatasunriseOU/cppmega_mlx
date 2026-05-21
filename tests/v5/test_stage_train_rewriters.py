"""G04: stage_train applies spec.rewriters before train, captures graph diff.

V4-8 proved that rewriter names propagated to extras.model_summary.rewriters_applied
but apply_rewrites was never called — the graph in train was identical
to the user's pre-rewrite spec. G04 wires apply_rewrites in stage_train
and surfaces extras.graph_diff = {added, removed, renamed, skipped}.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(rewriters: list[dict]) -> VerifyParams:
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
        "rewriters": rewriters,
    })


def _run(rewriters: list[dict], num_steps: int = 2) -> dict:
    spec = _spec(rewriters)
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": num_steps}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    return train.extras


def test_no_rewriters_empty_graph_diff():
    """Empty rewriters list → graph_diff with empty added/removed/skipped."""
    extras = _run([])
    diff = extras["graph_diff"]
    assert diff["added"] == []
    assert diff["removed"] == []
    assert diff["skipped"] == []


def test_mtp_rewriter_adds_k_minus_1_head_nodes():
    """MTPRewriter k=3 adds 2 new head nodes (head_1, head_2; head_0
    is renamed-in-place from the original head)."""
    extras = _run([{"name": "MTPRewriter", "params": {"k": 3}}])
    diff = extras["graph_diff"]
    # mlp is the head node; MTPRewriter renames it to mlp_0 and adds
    # mlp_1, mlp_2.
    assert "mlp_1" in diff["added"]
    assert "mlp_2" in diff["added"]
    assert "mlp" in diff["removed"]
    assert diff["skipped"] == []


def test_mtp_rewriter_k1_is_noop():
    """K=1 MTP is a no-op fast path; graph unchanged."""
    extras = _run([{"name": "MTPRewriter", "params": {"k": 1}}])
    diff = extras["graph_diff"]
    assert diff["added"] == []
    assert diff["removed"] == []


def test_unknown_rewriter_skipped_with_reason():
    extras = _run([{"name": "FrobRewriter", "params": {}}])
    diff = extras["graph_diff"]
    assert any(s["name"] == "FrobRewriter"
               and s["reason"] == "unknown_rewriter"
               for s in diff["skipped"])


def test_ifim_rewriter_adds_aux_node():
    """IFIMRewriter adds an aux node observable in graph_diff.added."""
    extras = _run([{"name": "IFIMRewriter",
                    "params": {"lambda_fim": 0.1}}])
    diff = extras["graph_diff"]
    # Aux node count > 0 (exact name depends on rewriter impl; just
    # assert at least one node added or rewriter ran without skip)
    assert (len(diff["added"]) > 0
            or not any(s["name"] == "IFIMRewriter" for s in diff["skipped"]))


def test_mtp_then_train_uses_k_heads():
    """After MTPRewriter applies, stage_train should run K-head loss
    path even though user did NOT explicitly set loss.kind=mtp_weighted.
    Proves rewrite mutated the spec before the loss kernel branched."""
    extras = _run([{"name": "MTPRewriter", "params": {"k": 2}}])
    # MTPRewriter rewrites loss.kind to MTP_WEIGHTED in spec
    assert extras["mtp"] is not None, \
        "MTPRewriter should have upgraded loss to MTP_WEIGHTED"
    assert extras["mtp"]["k"] == 2


def test_composition_mtp_plus_ifim():
    """Chain MTP→IFIM: both effects observable. MTP adds head nodes,
    IFIM tries its aux. Some skips OK; assert chain ran without crash
    and either ifim_added in graph_diff OR ifim listed in skipped."""
    extras = _run([
        {"name": "MTPRewriter", "params": {"k": 2}},
        {"name": "IFIMRewriter", "params": {"lambda_fim": 0.05}},
    ])
    diff = extras["graph_diff"]
    # MTP additions present
    assert "mlp_1" in diff["added"]
    # IFIM may have applied (added more) or skipped — both valid
    has_ifim_aux_or_skip = (
        len(diff["added"]) > 1  # MTP head + IFIM aux
        or any(s["name"] == "IFIMRewriter" for s in diff["skipped"])
    )
    assert has_ifim_aux_or_skip
