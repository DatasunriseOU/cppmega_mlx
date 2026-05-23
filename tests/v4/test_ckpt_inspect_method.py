"""V7-C03: ckpt.inspect RPC round-trip.

Writes a tiny safetensors checkpoint with the cppmega metadata schema
produced by stages.write_ckpt_metadata, hits POST /rpc with ckpt.inspect,
and asserts the parsed sub-objects come back wired so the UI can render
arch_hash / opt_kind / version on Load.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from safetensors.numpy import save_file as save_safetensors

from cppmega_v4.jsonrpc import create_app


@pytest.fixture
def client():
    return TestClient(create_app(cache_capacity=2))


def _write_ckpt(path: Path, *, version: str = "v1",
                arch_hash: str = "deadbeef" * 8,
                opt_kind: str = "adamw", lr: float = 3e-4,
                step: int = 17) -> None:
    metadata = {
        "cppmega_version": version,
        "arch": json.dumps(
            {"config_hash": arch_hash, "config_json": {}}, sort_keys=True),
        "train": json.dumps({"global_step": step}, sort_keys=True),
        "opt": json.dumps({"kind": opt_kind, "lr": lr}, sort_keys=True),
    }
    # Smallest possible payload — one scalar weight.
    import numpy as np
    save_safetensors({"w": np.zeros((1,), dtype=np.float32)},
                     str(path), metadata=metadata)


def test_ckpt_inspect_missing_file(client, tmp_path):
    payload = {
        "jsonrpc": "2.0", "id": "m1", "method": "ckpt.inspect",
        "params": {"path": str(tmp_path / "nope.safetensors")},
    }
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["exists"] is False


def test_ckpt_inspect_no_metadata(client, tmp_path):
    import numpy as np
    p = tmp_path / "bare.safetensors"
    save_safetensors({"w": np.zeros((1,), dtype=np.float32)}, str(p))
    payload = {
        "jsonrpc": "2.0", "id": "m2", "method": "ckpt.inspect",
        "params": {"path": str(p)},
    }
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["exists"] is True
    assert body["result"]["has_metadata"] is False


def test_ckpt_inspect_round_trip(client, tmp_path):
    p = tmp_path / "ckpt.safetensors"
    _write_ckpt(p, opt_kind="muon_adamw_hybrid", lr=1e-3, step=42)

    payload = {
        "jsonrpc": "2.0", "id": "m3", "method": "ckpt.inspect",
        "params": {"path": str(p)},
    }
    r = client.post("/rpc", json=payload)
    assert r.status_code == 200
    res = r.json()["result"]
    assert res["exists"] is True
    assert res["has_metadata"] is True
    assert res["opt_kind"] == "muon_adamw_hybrid"
    assert res["opt_lr"] == pytest.approx(1e-3)
    assert res["global_step"] == 42
    # arch_hash must be the 64-char sha256 we wrote.
    assert isinstance(res["arch_hash"], str)
    assert len(res["arch_hash"]) == 64


# Silence the unused mlx warning — keep the import so future tests on
# the same module can use mlx-backed safetensors without re-importing.
_ = mx
