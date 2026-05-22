"""V7-C01: opt-state load surfaces arch mismatch honestly.

Currently a checkpoint produced for a different architecture (extra
brick, renamed layer, changed hidden size) loads silently — keys are
dropped by tree_unflatten and shape mismatches raise opaque errors
mid-step. v7 adds a structural fingerprint diff:

  * extras.checkpoint.opt_state_arch_diff dict with
    {missing_keys, extra_keys, shape_mismatch} when the checkpoint's
    opt-state fingerprint doesn't match the current model.
  * opts.opt_state_strict=True → skip the load and surface a
    "cold restart" warning instead of partial overwrite.
  * Default (strict=False) preserves the prior best-effort behavior
    but still reports the diff so the user can see what was dropped.
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(intermediate: int = 256) -> VerifyParams:
    """intermediate_size controls mlp weight shape → easy way to
    produce a checkpoint that won't fit a different architecture."""
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 4, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": intermediate}},
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


def _train(spec, **opts) -> dict:
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_c01_arch_diff_null_on_matching_load(tmp_path):
    """Matched arch → no diff reported."""
    save = str(tmp_path / "opt.safetensors")
    _train(_spec(), num_steps=2, opt_state_save_path=save)
    extras = _train(_spec(), num_steps=1, opt_state_load_path=save)
    assert extras["checkpoint"]["opt_state_loaded_path"] == save
    assert extras["checkpoint"]["opt_state_arch_diff"] is None
    assert extras["checkpoint"]["opt_state_warning"] is None


def test_c01_arch_diff_populated_on_shape_mismatch(tmp_path):
    """Save with intermediate=256 → load into intermediate=128 model.
    Should populate opt_state_arch_diff with shape_mismatch entries
    AND skip the load (would crash mid-step otherwise) — train
    proceeds as cold restart."""
    save = str(tmp_path / "opt.safetensors")
    _train(_spec(intermediate=256), num_steps=2,
           opt_state_save_path=save)
    extras = _train(_spec(intermediate=128), num_steps=1,
                    opt_state_load_path=save)
    diff = extras["checkpoint"]["opt_state_arch_diff"]
    assert diff is not None, (
        "expected opt_state_arch_diff for mismatched intermediate_size"
    )
    assert diff["shape_mismatch_count"] > 0
    # Non-strict but shape_mismatch → load skipped, warning surfaced,
    # training still completes (cold restart).
    assert extras["checkpoint"]["opt_state_loaded_path"] is None
    warning = extras["checkpoint"]["opt_state_warning"]
    assert isinstance(warning, str)
    assert "shape mismatch" in warning.lower()
    assert len(extras["losses"]) == 1


def test_c01_strict_mode_blocks_load_on_mismatch(tmp_path):
    """opt_state_strict=True → load skipped, warning + diff populated,
    opt_state_loaded_path stays None."""
    save = str(tmp_path / "opt.safetensors")
    _train(_spec(intermediate=256), num_steps=2,
           opt_state_save_path=save)
    extras = _train(_spec(intermediate=128), num_steps=1,
                    opt_state_load_path=save,
                    opt_state_strict=True)
    assert extras["checkpoint"]["opt_state_loaded_path"] is None
    diff = extras["checkpoint"]["opt_state_arch_diff"]
    assert diff is not None
    warning = extras["checkpoint"]["opt_state_warning"]
    assert isinstance(warning, str) and "strict mode" in warning.lower()
    assert "cold restart" in warning.lower()
    # Training still produced losses (cold start).
    assert len(extras["losses"]) == 1


def test_c01_strict_mode_passes_through_when_matched(tmp_path):
    """opt_state_strict=True on a matched arch → still loads cleanly."""
    save = str(tmp_path / "opt.safetensors")
    _train(_spec(), num_steps=2, opt_state_save_path=save)
    extras = _train(_spec(), num_steps=1,
                    opt_state_load_path=save,
                    opt_state_strict=True)
    assert extras["checkpoint"]["opt_state_loaded_path"] == save
    assert extras["checkpoint"]["opt_state_arch_diff"] is None
