"""V7-N01/N02: fusion-emitter registry in dispatch_lower."""

from __future__ import annotations

import pytest

from cppmega_mlx.nn._tilelang._engine_dispatch import (
    _FUSION_EMITTERS, emit_fusion_kernel,
    fusion_emitters_available, register_fusion_emitter,
)


def test_default_emitters_registered_at_import():
    avail = fusion_emitters_available()
    assert "gemm_softmax" in avail
    assert "qk_reduce_sm_scale" in avail


def test_register_fusion_emitter_round_trip():
    calls: list[dict] = []

    def _fake_emitter(**kw):
        calls.append(kw)
        return object()  # synthetic prim_func

    register_fusion_emitter("custom_fusion", _fake_emitter)
    try:
        assert "custom_fusion" in fusion_emitters_available()
        # Calling emit_fusion_kernel hits the registered factory; we
        # expect dispatch_lower to raise because the fake prim_func is
        # not a TileLang PrimFunc. We catch and assert the emitter ran.
        with pytest.raises(Exception):
            emit_fusion_kernel("custom_fusion", M=4, N=4, K=4)
        assert len(calls) == 1
        assert calls[0]["M"] == 4
    finally:
        _FUSION_EMITTERS.pop("custom_fusion", None)


def test_missing_emitter_raises_key_error():
    with pytest.raises(KeyError):
        emit_fusion_kernel("not_registered_anywhere",
                            M=4, N=4, K=4)
