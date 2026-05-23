"""E-AUDIT-01: POST /upload/parquet writes to /tmp/vbgui_uploads + TTL."""

from __future__ import annotations

import io
import os
import pathlib
import time

import pytest

from cppmega_v4.jsonrpc.server import create_app
from cppmega_v4.jsonrpc.uploads import (
    UPLOAD_ROOT, TTL_SECONDS, cleanup_stale, save_upload,
)


@pytest.fixture
def app():
    return create_app()


@pytest.fixture(autouse=True)
def _wipe_uploads():
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    for p in UPLOAD_ROOT.iterdir():
        try: p.unlink()
        except OSError: pass
    yield
    for p in UPLOAD_ROOT.iterdir():
        try: p.unlink()
        except OSError: pass


def test_e_audit_01_upload_persists_under_tmp_with_uuid_name(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    body = b"PAR1" + b"\x00" * 64  # synthetic parquet-ish blob
    resp = client.post(
        "/upload/parquet",
        files={"file": ("shard.parquet", io.BytesIO(body),
                         "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    j = resp.json()
    assert j["bytes"] == len(body)
    assert j["filename"] == "shard.parquet"
    assert j["path"].startswith(str(UPLOAD_ROOT))
    assert j["path"].endswith(".parquet")
    p = pathlib.Path(j["path"])
    assert p.is_file()
    assert p.read_bytes() == body


def test_e_audit_01_rejects_non_parquet_extension(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post(
        "/upload/parquet",
        files={"file": ("malicious.exe", io.BytesIO(b"x"),
                         "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "parquet" in resp.json()["detail"].lower()


def test_e_audit_01_rejects_empty_body(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post(
        "/upload/parquet",
        files={"file": ("empty.parquet", io.BytesIO(b""),
                         "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_e_audit_01_cleanup_stale_drops_old_files():
    """Files older than TTL_SECONDS are removed by cleanup_stale."""
    old = save_upload(b"x" * 16)
    young = save_upload(b"y" * 16)
    # Backdate `old` past TTL.
    ancient = time.time() - TTL_SECONDS - 60
    os.utime(old, (ancient, ancient))

    removed = cleanup_stale()
    assert removed >= 1
    assert not pathlib.Path(old).exists()
    assert pathlib.Path(young).exists()


def test_e_audit_01_save_upload_runs_cleanup_per_call():
    """save_upload() implicitly invokes cleanup_stale; an aged file
    from a prior call is gone after the next upload."""
    old = save_upload(b"a" * 8)
    ancient = time.time() - TTL_SECONDS - 60
    os.utime(old, (ancient, ancient))
    _new = save_upload(b"b" * 8)
    assert not pathlib.Path(old).exists()
