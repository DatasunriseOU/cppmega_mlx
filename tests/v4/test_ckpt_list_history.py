"""V7-Q03.1: ckpt.list_history RPC regression.

Pins: scan returns sorted-by-mtime descending list of safetensors files
with metadata extracted via read_ckpt_metadata. Skips opt-state sidecars.
Surfaces has_opt_sidecar flag when `<path>.opt` exists alongside.
"""

from __future__ import annotations

import os
import tempfile
import time

import mlx.core as mx
import safetensors.mlx as stmlx

from cppmega_v4.jsonrpc.ckpt_history_method import (
    CkptListHistoryParams, ckpt_list_history,
)


def _make_ckpt(path: str, *, with_sidecar: bool = False,
               arch_hash: str = "abc123def456", step: int = 7) -> None:
    weights = {"layers.0.weight": mx.zeros((4, 4))}
    meta = {
        "cppmega_version": "test",
        "arch": '{"config_hash": "' + arch_hash + '"}',
        "train": '{"global_step": ' + str(step) + '}',
        "opt": '{"kind": "adamw", "lr": 0.001}',
    }
    stmlx.save_file(weights, path, metadata=meta)
    if with_sidecar:
        opt_state = {"opt.0.m": mx.zeros((4, 4))}
        stmlx.save_file(opt_state, path + ".opt", metadata=meta)


def test_list_history_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        res = ckpt_list_history(CkptListHistoryParams(directory=td))
    assert res.error is None
    assert res.scanned == 0
    assert res.entries == []


def test_list_history_returns_metadata() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.safetensors")
        _make_ckpt(p, arch_hash="aaa111", step=4)
        res = ckpt_list_history(CkptListHistoryParams(directory=td))
    assert res.error is None
    assert res.scanned == 1
    assert len(res.entries) == 1
    e = res.entries[0]
    assert e.path.endswith("a.safetensors")
    assert e.arch_hash == "aaa111"
    assert e.opt_kind == "adamw"
    assert e.global_step == 4
    assert e.size_bytes > 0
    assert e.has_opt_sidecar is False


def test_list_history_detects_sidecar() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "withopt.safetensors")
        _make_ckpt(p, with_sidecar=True)
        res = ckpt_list_history(CkptListHistoryParams(directory=td))
    # Sidecar file is NOT a top-level entry but IS detected on the
    # parent's has_opt_sidecar flag.
    assert res.scanned == 1
    assert len(res.entries) == 1
    assert res.entries[0].has_opt_sidecar is True


def test_list_history_sorted_by_mtime_desc() -> None:
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "older.safetensors")
        b = os.path.join(td, "newer.safetensors")
        _make_ckpt(a)
        time.sleep(0.05)  # mtime granularity
        _make_ckpt(b)
        res = ckpt_list_history(CkptListHistoryParams(directory=td))
    assert [e.path.split("/")[-1] for e in res.entries] == [
        "newer.safetensors", "older.safetensors",
    ]


def test_list_history_caps_at_max_entries() -> None:
    with tempfile.TemporaryDirectory() as td:
        for i in range(5):
            _make_ckpt(os.path.join(td, f"c{i}.safetensors"))
        res = ckpt_list_history(
            CkptListHistoryParams(directory=td, max_entries=3))
    assert res.scanned == 5
    assert len(res.entries) == 3


def test_list_history_missing_dir() -> None:
    res = ckpt_list_history(
        CkptListHistoryParams(directory="/nonexistent/dir/xyzzy"))
    assert res.error is not None
    assert "does not exist" in res.error


def test_list_history_recursive() -> None:
    with tempfile.TemporaryDirectory() as td:
        sub = os.path.join(td, "sub")
        os.makedirs(sub)
        _make_ckpt(os.path.join(td, "top.safetensors"))
        _make_ckpt(os.path.join(sub, "deep.safetensors"))
        res = ckpt_list_history(CkptListHistoryParams(directory=td))
    assert res.scanned == 2
    assert {e.path.split("/")[-1] for e in res.entries} == {
        "top.safetensors", "deep.safetensors",
    }
