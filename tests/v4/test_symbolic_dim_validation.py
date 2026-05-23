"""V7-F56: symbolic-dim validation — incompatible (H, nh, head_dim)."""

from __future__ import annotations

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline


def _spec(H: int, num_heads: int, head_dim: int) -> VerifyParams:
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention",
                 "params": {"num_heads": num_heads,
                            "head_dim": head_dim}},
                {"id": "mlp", "kind": "mlp", "params": {}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": H,
                    "nh": num_heads, "nkv": 1, "head_dim": head_dim},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


def _verify(spec: VerifyParams):
    return run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec"],
    }))


def test_v7_f56_compatible_combo_h128_nh8_hd16_passes():
    """H=128 = nh(8) * head_dim(16). verify ok."""
    rep = _verify(_spec(H=128, num_heads=8, head_dim=16))
    vbs = next(s for s in rep.stages if s.name == "verify_build_spec")
    assert vbs.status == "ok", f"compatible combo rejected: {vbs.error}"


def test_v7_f56_compatible_combo_h512_nh8_hd64_passes():
    rep = _verify(_spec(H=512, num_heads=8, head_dim=64))
    vbs = next(s for s in rep.stages if s.name == "verify_build_spec")
    assert vbs.status == "ok"


def test_v7_f56b_incompatible_h128_nh3_hd50_surfaces_warning():
    """V7-F56b: verify_build_spec surfaces a WARNING (not error) when
    nh*head_dim != H. Bricks ship an internal Q-projection so the
    model still trains end-to-end, but the dim_env mismatch almost
    always means user confusion."""
    spec = _spec(H=128, num_heads=3, head_dim=50)
    rep = _verify(spec)
    vbs = next(s for s in rep.stages if s.name == "verify_build_spec")
    # Warning, not fail — keeps decoupled-Q convention working.
    assert vbs.status == "ok", (
        f"F56b should warn-not-fail (decoupled Q is legitimate). "
        f"Got: {vbs}"
    )
    assert (vbs.warnings or 0) >= 1, (
        f"V7-F56b: expected ≥1 dim_env warning. Got: {vbs}"
    )


def test_v7_f56b_compatible_combo_produces_no_warning():
    spec = _spec(H=128, num_heads=8, head_dim=16)  # 8*16 == 128
    rep = _verify(spec)
    vbs = next(s for s in rep.stages if s.name == "verify_build_spec")
    assert vbs.status == "ok"
    assert (vbs.warnings or 0) == 0, (
        f"Compatible combo should be silent. Got warnings={vbs.warnings}"
    )


def test_v7_f56_dim_env_H_mismatched_with_nh_times_head_dim_observable():
    """Build the spec and verify the bookkeeping math at the
    Python level — proves the contract the validator SHOULD enforce
    is observable to the gate."""
    spec = _spec(H=128, num_heads=3, head_dim=50)
    de = spec.dim_env if isinstance(
        spec.dim_env, dict) else spec.dim_env.model_dump()
    H = de["H"]
    nh = de["nh"]
    hd = de["head_dim"]
    assert nh * hd != H, (
        f"V7-F56 premise broken: nh*hd ({nh}*{hd}) accidentally "
        f"equals H ({H})"
    )
