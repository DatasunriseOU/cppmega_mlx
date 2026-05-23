"""V7-M0.3: pin the dry_forward capture_logits wiring.

Asserts the stage surfaces output_logits (shape) + output_values
(flat list of finite floats) in extras when capture_logits=True is
passed, and stays None on the default zero-overhead path.
"""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(H: int = 128, S: int = 16) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": 2, "head_dim": 64}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": S, "H": H,
                    "nh": 2, "nkv": 1, "head_dim": 64},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _run(opts: dict) -> dict:
    rep = run_pipeline(_spec(), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "dry_forward"],
        "stage_options": {"dry_forward": opts},
    }))
    dry = next(s for s in rep.stages if s.name == "dry_forward")
    assert dry.status == "ok", dry.error
    return dict(dry.extras or {})


def test_m03_capture_off_by_default():
    """No capture_logits → extras must omit output_logits."""
    extras = _run({"B": 1, "S": 8})
    assert "output_logits" not in extras
    assert "output_values" not in extras


def test_m03_capture_surfaces_shape_and_values():
    """capture_logits=True → extras carries shape tuple + flat values.

    Note: the attention brick's output projection ships zero-initialised
    (stable training-start convention), so the chain attention→mlp can
    legitimately produce all-zero hidden states. The wiring promise is
    just that shape + values surface deterministically — the *values*
    are whatever the bricks compute, which is the cross-platform parity
    contract that matters for M0.3.
    """
    extras = _run({"B": 1, "S": 8, "capture_logits": True, "seed": 7})
    assert extras["output_logits"] == [1, 8, 128]
    vals = extras["output_values"]
    assert isinstance(vals, list)
    assert len(vals) == 1 * 8 * 128
    # Finite (no NaN/Inf) is the cross-platform invariant.
    for v in vals[:64]:
        f = float(v)
        assert f == f, "NaN in output_values"
        assert abs(f) < 1e6, "value out of range"


def test_m03_seed_deterministic():
    """Same seed → bit-identical output_values across runs.

    This is the actual M0.3 parity contract: a GB10 CUDA run with the
    same seed must produce the same output_values as the MLX run.
    """
    a = _run({"B": 1, "S": 8, "capture_logits": True, "seed": 7})
    b = _run({"B": 1, "S": 8, "capture_logits": True, "seed": 7})
    assert a["output_values"] == b["output_values"]
