"""V7-F55: tokenizer × preset compatibility matrix."""

from __future__ import annotations

import json
import pathlib

import pytest

from cppmega_v4.architectures.presets import build_preset_specs
from cppmega_v4.jsonrpc.schema import VerifyParams
from cppmega_v4.runner import Pipeline, run_pipeline

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
MATRIX_PATH = REPO / "tests" / "fixtures" / "MATRIX.json"

PRESETS = ["llama3_8b", "mistral_small_3_1", "qwen3_dense_0_6b"]


def _load_pairs():
    if not MATRIX_PATH.exists():
        return []
    m = json.loads(MATRIX_PATH.read_text())
    pairs = []
    parquet = m["parquets"]["T2_gpt2_small__P1_minimal"]["path"]
    if not pathlib.Path(parquet).exists():
        return []
    for tok in m["tokenizers"].values():
        if pathlib.Path(tok["path"]).exists():
            pairs.append(tok["path"])
    return [(preset, tok, parquet)
            for preset in PRESETS for tok in pairs]


PAIRS = _load_pairs()


def _spec(preset: str) -> VerifyParams:
    specs = build_preset_specs(preset, hidden_size=128)
    return VerifyParams.model_validate({
        "graph": {
            "nodes": [
                {"id": f"n{i}", "kind": s["kind"],
                 "params": s.get("params", {})}
                for i, s in enumerate(specs)
            ],
            "edges": [
                {"src": f"n{i}", "dst": f"n{i + 1}"}
                for i in range(len(specs) - 1)
            ],
        },
        "dim_env": {"B": 1, "S": 8, "H": 128,
                    "nh": 2, "nkv": 1, "head_dim": 64,
                    "num_experts": 4, "top_k": 2},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [f"n{len(specs) - 1}"]},
        "optim": {"kind": "adamw",
                  "groups": [{"matcher": "all", "lr": 1e-3,
                              "weight_decay": 0.01,
                              "betas": [0.9, 0.95]}]},
    })


@pytest.mark.parametrize("preset,tokenizer,parquet", PAIRS)
def test_v7_f55_preset_x_tokenizer_trains(preset, tokenizer, parquet):
    if not PAIRS:
        pytest.skip("MATRIX fixtures missing")
    rep = run_pipeline(_spec(preset), Pipeline.from_dict({
        "stages": ["parse", "verify_build_spec", "build_model",
                   "train"],
        "stage_options": {"train": {
            "num_steps": 2,
            "tokenizer_path": tokenizer,
            "parquet_path": parquet,
        }},
    }))
    tr = next(s for s in rep.stages if s.name == "train")
    assert tr.status == "ok", (
        f"{preset} × {pathlib.Path(tokenizer).name}: {tr.error}"
    )
    for L in tr.extras["losses"]:
        assert -1e10 < L < 1e10
    assert tr.extras["data_source"] in ("parquet", "parquet_tokenized",
                                          "synthetic")
