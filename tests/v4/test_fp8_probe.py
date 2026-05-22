"""V7-D01: fp8 probe — honest availability + reason for UI banner."""

from __future__ import annotations

from cppmega_v4.runtime.fp8_probe import probe_fp8


def test_v7_d01_probe_shape():
    r = probe_fp8()
    for k in ("available", "reason", "dtype_name"):
        assert k in r
    assert isinstance(r["available"], bool)
    assert isinstance(r["reason"], str) and len(r["reason"]) > 0


def test_v7_d01_unavailable_reason_is_non_empty():
    r = probe_fp8()
    if not r["available"]:
        # When unavailable, the reason must be a human-readable
        # explanation (so the UI banner is meaningful).
        assert any(s in r["reason"].lower()
                   for s in ("fp8", "float8", "alloc", "hardware",
                              "build"))
