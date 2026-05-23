"""V7-H48: backend exposes a stable build_id used by the WS heartbeat."""

from __future__ import annotations

from cppmega_v4.jsonrpc.server import _backend_build_id


def test_v7_h48_build_id_is_stable_within_process():
    bid1 = _backend_build_id()
    bid2 = _backend_build_id()
    assert bid1 == bid2  # cached for process lifetime
    assert isinstance(bid1, str)
    assert len(bid1) > 0


def test_v7_h48_build_id_carries_sha_and_timestamp_segments():
    bid = _backend_build_id()
    # Format: "<sha>.<unix-ts>" — sha may be 'unknown' off-git.
    parts = bid.split(".")
    assert len(parts) == 2, parts
    assert parts[1].isdigit(), parts
