from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_muon_ns_carrier_bench_emits_local_only_receipt(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "muon_ns_carrier.json"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "bench_muon_ns_carrier.py"),
        "--matrix-size",
        "16",
        "--ns-steps",
        "2",
        "--warmup",
        "1",
        "--iters",
        "1",
        "--smoke-steps",
        "3",
        "--smoke-hidden",
        "4",
        "--smoke-batch-size",
        "4",
        "--output",
        str(output),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["kind"] == "cppmega.mlx.local_m4_muon_ns_carrier_receipt"
    assert payload["source_bead"] == "cppmega-mlx-c08.4"
    assert payload["guards"]["local_only"] is True
    assert payload["guards"]["gb10_parity_claim"] is False
    assert payload["guards"]["m4_vs_gb10_throughput_parity_claim"] is False
    assert payload["config"]["muon_ns_carrier_env"] == "CPPMEGA_MUON_NS_CARRIER"
    assert set(payload["timing"]) == {"fp32", "bf16"}
    assert payload["timing"]["fp32"]["iters"] == 1
    assert payload["timing"]["bf16"]["iters"] == 1
    assert set(payload["smoke"]) == {"fp32", "bf16", "allowed_bf16_final_loss"}
    acceptance = payload["acceptance"]
    assert acceptance["orthogonalization_atol"] == 1e-2
    assert "max_abs_orthogonalization_error" in acceptance
    assert "median_ns_loop_speedup_fp32_over_bf16" in acceptance
    assert "bf16_no_loss_regression_vs_fp32" in acceptance
