"""V7-B16: mx.distributed world=1 fallback + stage_train survives."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline
from cppmega_v4.runtime import distributed as _d


@pytest.fixture(autouse=True)
def _clean():
    _d.reset_for_test()
    yield
    _d.reset_for_test()


def test_mx_distributed_init_strict_false_returns_world_1():
    """mx.distributed.init(strict=False) returns a singleton size 1
    group when no launcher is active. Our wrapper must reflect that."""
    info = _d.init(force_single=True)
    assert info.world_size == 1
    assert info.rank == 0
    assert info.real is False
    # is_distributed() must be False in fallback mode.
    assert _d.is_distributed() is False


def test_stage_train_survives_compile_regional_plus_fake_ranks_1():
    """compile_mode='regional' + fake_ranks=1 + world=1 fallback must
    not raise inside stage_train. This is the canonical 'works on my
    laptop' path."""
    spec_payload = {
        "graph": {
            "nodes": [
                {"id": "attn", "kind": "attention", "params": {}},
                {"id": "mlp", "kind": "mlp",
                 "params": {"intermediate_size": 64,
                             "activation": "swiglu"}},
            ],
            "edges": [{"src": "attn", "dst": "mlp"}],
        },
        "dim_env": {"B": 1, "S": 8, "H": 32, "nh": 2, "nkv": 1,
                    "head_dim": 16},
        "loss": {"kind": "cross_entropy", "head_outputs": ["mlp"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01, "betas": [0.9, 0.95]}]},
        "sharding": {
            "topology": {"factory": "m3_ultra_solo", "kwargs": {}},
            "axis_assignments": [],
            "compile_mode": "regional",
        },
    }
    spec = VerifyParams.model_validate(spec_payload)
    report = run_pipeline(spec, Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model", "train"],
        "stage_options": {"train": {"num_steps": 2, "fake_ranks": 1}},
    }))
    train = next(s for s in report.stages if s.name == "train")
    assert train.status == "ok", f"stage_train failed: {train.error}"
    # distributed.real must be False on the single-process path.
    dist = (train.extras or {}).get("distributed", {})
    assert dist.get("real") is False
    assert dist.get("world_size") == 1
