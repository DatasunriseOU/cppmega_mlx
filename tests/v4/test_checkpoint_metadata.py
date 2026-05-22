"""V7-C03: self-describing checkpoint metadata round-trip + warnings."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runner.stages import read_ckpt_metadata


def _spec(intermediate: int = 256, optim_kind: str = "adamw") -> VerifyParams:
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
        "optim": {"kind": optim_kind,
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


def test_c03_metadata_round_trip_arch_train_opt(tmp_path):
    """Save → read metadata → assert arch/train/opt + version keys."""
    save = str(tmp_path / "w.safetensors")
    _train(_spec(), num_steps=2, checkpoint_save_path=save)
    meta = read_ckpt_metadata(save)
    assert meta is not None
    assert "cppmega_version" in meta
    assert "arch" in meta and isinstance(meta["arch"], dict)
    assert "config_hash" in meta["arch"]
    assert "config_json" in meta["arch"]
    assert "train" in meta and isinstance(meta["train"], dict)
    assert "global_step" in meta["train"]
    assert meta["train"]["global_step"] == 2
    assert "opt" in meta and isinstance(meta["opt"], dict)
    assert meta["opt"]["kind"] == "adamw"


def test_c03_load_validates_arch_hash_match(tmp_path):
    """Matched arch on load → extras.checkpoint.metadata populated, no
    metadata_warning."""
    save = str(tmp_path / "w.safetensors")
    _train(_spec(), num_steps=2, checkpoint_save_path=save)
    extras = _train(_spec(), num_steps=1, checkpoint_load_path=save)
    md = extras["checkpoint"]["metadata"]
    assert md is not None
    assert md["arch"]["config_hash"]
    assert extras["checkpoint"]["metadata_warning"] is None


def test_c03_load_warns_on_arch_hash_mismatch(tmp_path):
    """Save with intermediate=256, load into intermediate=128 model →
    metadata_warning includes 'arch.config_hash mismatch'."""
    save = str(tmp_path / "w.safetensors")
    _train(_spec(intermediate=256), num_steps=2,
           checkpoint_save_path=save)
    extras = _train(_spec(intermediate=128), num_steps=1,
                    checkpoint_load_path=save)
    warning = extras["checkpoint"]["metadata_warning"]
    assert isinstance(warning, str)
    assert "arch.config_hash mismatch" in warning


def test_c03_ckpt_strict_blocks_weight_load_on_arch_mismatch(tmp_path):
    """opts.ckpt_strict=True + arch mismatch → checkpoint_loaded is
    rolled back to None (fresh weights)."""
    save = str(tmp_path / "w.safetensors")
    _train(_spec(intermediate=256), num_steps=2,
           checkpoint_save_path=save)
    extras = _train(_spec(intermediate=128), num_steps=1,
                    checkpoint_load_path=save,
                    ckpt_strict=True)
    # mlx safetensors load may still succeed key-wise for matching
    # tensors and silently drop mismatched ones; strict mode rolls
    # back the loaded_path field.
    assert extras["checkpoint"]["loaded_path"] is None
    assert "arch.config_hash mismatch" in (
        extras["checkpoint"]["metadata_warning"] or "")


def test_c03_ckpt_inspect_cli_pretty_prints(tmp_path):
    """python -m cppmega_v4.tools.ckpt_inspect FILE → valid JSON."""
    save = str(tmp_path / "w.safetensors")
    _train(_spec(), num_steps=2, checkpoint_save_path=save)
    r = subprocess.run(
        [sys.executable, "-m", "cppmega_v4.tools.ckpt_inspect", save],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    parsed = json.loads(r.stdout)
    assert parsed["cppmega_version"]
    assert parsed["arch"]["config_hash"]
