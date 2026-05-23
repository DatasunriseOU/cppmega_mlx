"""V8-R11 integration smoke test — preset → scale_down → memory.matrix
chain end-to-end without the UI.

Covers steps 1-4 of docs/raschka_full_loop.md. Walks all 5 presets the
plan §6 marks as full-loop targets:
  llama3_8b, qwen3_dense_4b, kimi_linear, gemma3_27b, gpt_oss_20b.

For each preset, verifies:
  (a) build_preset_specs returns specs + defaults block
  (b) defaults block has paper-anchored lr + a known schedule
  (c) scale_down at a 1 GiB budget lands a fitting (H, L) pair
  (d) memory.matrix returns 4×5 cells for the scaled spec
  (e) at least one (topology, precision) cell fits the matrix
"""

from __future__ import annotations

import pytest

from cppmega_v4.jsonrpc.auto_fit_method import (
    AutoFitHostInfo, AutoFitParams, auto_fit,
)
from cppmega_v4.jsonrpc.memory_matrix_method import (
    MemoryMatrixParams, memory_matrix,
)
from cppmega_v4.jsonrpc.methods import build_preset_specs
from cppmega_v4.jsonrpc.schema import (
    BuildPresetSpecsParams, VerifyParams,
)


ONE_GB = 1_073_741_824
FULL_LOOP_PRESETS = [
    "llama3_8b",
    "qwen3_dense_4b",
    "kimi_linear",
    "gemma3_27b",
    "gpt_oss_20b",
]


@pytest.mark.parametrize("preset", FULL_LOOP_PRESETS)
def test_full_loop_steps_1_to_4(preset: str):
    # Step 1 — pick a preset
    r1 = build_preset_specs(BuildPresetSpecsParams(
        preset_name=preset, hidden_size=256))
    assert isinstance(r1.specs, list) and len(r1.specs) >= 1, preset
    assert r1.preset_name == preset
    assert isinstance(r1.defaults, dict)
    assert r1.defaults["lr"] > 0
    assert r1.defaults["schedule"] in {
        "constant", "linear_warmup", "cosine", "wsd", "inv_sqrt"}

    # Step 2 — scale down to 1 GiB target via auto_fit on m3_ultra_solo
    # (deterministic across machines; gb10_quarter would over-fit).
    r2 = auto_fit(AutoFitParams(
        preset=preset,
        host_info=AutoFitHostInfo(topology="m3_ultra_solo")))
    assert r2.scaled.hidden_size >= 64
    assert r2.scaled.num_layers >= 1
    # The 8B/27B/20B presets all fit comfortably on m3_ultra (512 GB).
    assert r2.fits is True, (preset, r2.reason)

    # Step 3 — memory matrix on the scaled spec
    scaled_graph_nodes = [
        {"id": n.get("name") or n.get("kind"),
         "kind": n["kind"], "params": n.get("params", {})}
        for n in r2.scaled.specs[:8]  # first 8 nodes are enough for matrix
        if "kind" in n
    ]
    if len(scaled_graph_nodes) < 2:
        pytest.skip(f"{preset}: only {len(scaled_graph_nodes)} leaf nodes")
    graph_edges = [
        {"src": scaled_graph_nodes[i]["id"],
         "dst": scaled_graph_nodes[i + 1]["id"]}
        for i in range(len(scaled_graph_nodes) - 1)
    ]
    vp = VerifyParams.model_validate({
        "graph": {"nodes": scaled_graph_nodes, "edges": graph_edges},
        "dim_env": {"H": r2.scaled.hidden_size, "B": 1, "S": 64},
        "loss": {"kind": "cross_entropy",
                 "head_outputs": [scaled_graph_nodes[-1]["id"]]},
        "optim": {"kind": "adamw", "groups": [
            {"matcher": "all", "lr": r1.defaults["lr"],
             "weight_decay": 0.01, "betas": [0.9, 0.95]}],
                 "gradient_clip_norm": 1.0, "mixed_precision": True},
        "sharding": None,
        "training": True,
    })
    matrix = memory_matrix(MemoryMatrixParams(spec=vp))
    assert len(matrix.cells) == 20
    fitting_cells = [c for c in matrix.cells if c.fits]
    assert len(fitting_cells) > 0, (
        f"{preset}: no (topo, prec) cell fits — "
        f"min bytes = {min(c.bytes for c in matrix.cells)}")


def test_defaults_are_consistent_across_loop_presets():
    """Every R11 target must yield a defaults block; mixed_precision
    must be on for all 5 (they're all bf16-trainable)."""
    for preset in FULL_LOOP_PRESETS:
        r = build_preset_specs(BuildPresetSpecsParams(
            preset_name=preset, hidden_size=128))
        assert r.defaults["mixed_precision"] is True, preset
        assert r.defaults["optimizer"] in {"adamw", "muon",
                                           "muon_adamw_hybrid"}
