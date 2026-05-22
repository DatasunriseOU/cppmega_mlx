"""V01: rng_key roundtrip enables strict 1e-5 loss continuation.

Without rng_key save+load, the per-step random target stream restarts
from key(0) on a resumed Train, so resumed.losses[0] does NOT equal
the contiguous run's losses[N]. V01 persists the active key in the
opt-state safetensors side-car under "_rng_key" and restores it on
load. This test pins the strict 1e-5 bound.
"""

from __future__ import annotations

import pytest

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


def _train(spec, **opts) -> dict:
    rep = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": opts},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", f"train failed: {tr.error}"
    return tr.extras


def test_v01_rng_key_roundtrip_strict_loss_parity(tmp_path):
    """Contiguous 5-step Run A losses[4] == (4-save + load + 1-step)
    Run B losses[0] within 1e-5 once weights + opt.state + rng_key
    are all restored."""
    a = _train(_spec(), num_steps=5)
    save_ckpt = str(tmp_path / "ck.safetensors")
    save_opt = str(tmp_path / "opt.safetensors")
    _train(_spec(), num_steps=4,
           checkpoint_save_path=save_ckpt,
           opt_state_save_path=save_opt)
    resumed = _train(_spec(), num_steps=1,
                     checkpoint_load_path=save_ckpt,
                     opt_state_load_path=save_opt)
    assert resumed["checkpoint"]["rng_key_loaded"] is True

    diff = abs(a["losses"][4] - resumed["losses"][0])
    assert diff < 1e-5, (
        f"V01 rng_key roundtrip insufficient: contig l4={a['losses'][4]}, "
        f"resumed l0={resumed['losses'][0]}, diff={diff}"
    )


def test_v01_rng_key_loaded_flag_false_without_opt_state():
    """Without opt_state_load_path, rng_key_loaded stays False."""
    extras = _train(_spec(), num_steps=2)
    assert extras["checkpoint"]["rng_key_loaded"] is False


def test_v01_rng_key_loaded_flag_when_present(tmp_path):
    """opt_state_load_path with a key-bearing file flips the flag."""
    save_opt = str(tmp_path / "opt.safetensors")
    _train(_spec(), num_steps=2, opt_state_save_path=save_opt)
    extras = _train(_spec(), num_steps=1, opt_state_load_path=save_opt)
    assert extras["checkpoint"]["rng_key_loaded"] is True
