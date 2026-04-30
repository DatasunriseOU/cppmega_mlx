from __future__ import annotations

import json
import math
import time

import mlx.core as mx
import pytest

from cppmega_mlx.training import profile as profile_mod
from cppmega_mlx.training.profile import (
    HotspotEvidence,
    KernelAdoptionBlocked,
    MemorySnapshot,
    hotspot_from_profile_metrics,
    profile_step,
)


def test_profile_step_records_json_serializable_metrics() -> None:
    values = mx.arange(8)

    with profile_step(
        "train",
        tokens=16,
        eval_args=(values,),
        extra={"phase": "unit", "not_json": object()},
    ) as prof:
        prof.add_eval_args(values + 1)

    metrics = prof.metrics
    payload = metrics.to_dict()
    json.dumps(payload)

    assert payload["label"] == "train"
    assert metrics.seconds >= 0
    assert payload["seconds"] == payload["wall_time_s"]
    assert payload["elapsed_wall_time_s"] == payload["wall_time_s"]
    assert metrics.tokens == 16
    assert metrics.tokens_per_second is not None
    assert metrics.tokens_per_second > 0
    assert metrics.evaluated is True
    assert isinstance(payload["memory"], dict)
    extra = payload["extra"]
    assert isinstance(extra, dict)
    assert extra["phase"] == "unit"
    assert isinstance(extra["not_json"], str)


def test_profile_step_metrics_available_after_context_without_tokens() -> None:
    with profile_step("eval", reset_peak=False, sync=False) as prof:
        time.sleep(0.001)

    metrics = prof.metrics
    assert metrics.label == "eval"
    assert metrics.tokens is None
    assert metrics.tokens_per_second is None
    assert metrics.seconds > 0
    assert metrics.peak_memory_reset is False
    assert metrics.synchronized is False
    assert metrics.evaluated is False


def test_memory_snapshot_feature_detects_missing_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(profile_mod.mx, "get_active_memory", raising=False)
    monkeypatch.delattr(profile_mod.mx, "get_peak_memory", raising=False)
    monkeypatch.delattr(profile_mod.mx, "get_cache_memory", raising=False)

    snapshot = MemorySnapshot.read()
    payload = snapshot.to_dict()
    json.dumps(payload)

    assert snapshot.available is False
    assert snapshot.active_bytes is None
    assert snapshot.peak_bytes is None
    assert snapshot.cache_bytes is None
    assert "mlx.core.get_active_memory unavailable" in snapshot.errors
    assert "mlx.core.get_peak_memory unavailable" in snapshot.errors
    assert "mlx.core.get_cache_memory unavailable" in snapshot.errors


def test_peak_reset_feature_detects_missing_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(profile_mod.mx, "reset_peak_memory", raising=False)

    with profile_step("no-memory-api", tokens=1) as prof:
        prof.add_eval_args(mx.array(1))

    assert prof.metrics.peak_memory_reset is False
    assert prof.metrics.evaluated is True


def test_profile_step_validates_tokens() -> None:
    with pytest.raises(ValueError, match="tokens must be non-negative"):
        profile_step(tokens=-1)


def test_metrics_reject_access_before_exit() -> None:
    prof = profile_step()
    with pytest.raises(RuntimeError, match="before context exit"):
        _ = prof.metrics


def test_memory_snapshot_uses_available_mlx_apis() -> None:
    snapshot = MemorySnapshot.read()
    payload = snapshot.to_dict()
    json.dumps(payload)

    assert isinstance(snapshot.available, bool)
    for value in (snapshot.active_bytes, snapshot.peak_bytes, snapshot.cache_bytes):
        assert value is None or value >= 0
    if snapshot.available:
        assert any(
            value is not None
            for value in (snapshot.active_bytes, snapshot.peak_bytes, snapshot.cache_bytes)
        )


def test_tokens_per_second_for_zero_elapsed_is_json_safe() -> None:
    memory = MemorySnapshot(
        active_bytes=None,
        peak_bytes=None,
        cache_bytes=None,
        available=False,
    )
    metrics = profile_mod.ProfileMetrics(
        label="manual",
        seconds=0.0,
        tokens=1,
        tokens_per_second=math.inf,
        memory=memory,
        peak_memory_reset=False,
        synchronized=False,
        evaluated=False,
    )

    payload = metrics.to_dict()
    json.dumps(payload)
    assert payload["seconds"] == payload["wall_time_s"] == payload["elapsed_wall_time_s"]
    assert payload["tokens_per_second"] is None


def test_manual_metrics_extra_is_json_safe() -> None:
    memory = MemorySnapshot(
        active_bytes=None,
        peak_bytes=None,
        cache_bytes=None,
        available=False,
    )
    metrics = profile_mod.ProfileMetrics(
        label="manual-extra",
        seconds=1.0,
        tokens=None,
        tokens_per_second=None,
        memory=memory,
        peak_memory_reset=False,
        synchronized=False,
        evaluated=False,
        extra={"bad": object(), "nested": {"nan": math.nan}},
    )

    payload = metrics.to_dict()
    json.dumps(payload)
    assert payload["seconds"] == payload["wall_time_s"] == payload["elapsed_wall_time_s"]
    extra = payload["extra"]
    assert isinstance(extra, dict)
    nested = extra["nested"]
    assert isinstance(nested, dict)
    assert isinstance(extra["bad"], str)
    assert nested["nan"] is None


def test_hotspot_evidence_summarizes_profile_metrics() -> None:
    with profile_step(
        "mamba3",
        tokens=32,
        reset_peak=False,
        sync=False,
        extra={
            "context": {
                "route": "M",
                "backend": "mlx",
                "operation": "mamba3_scan",
            },
        },
    ) as prof:
        time.sleep(0.001)

    hotspot = hotspot_from_profile_metrics(
        prof.metrics,
        total_seconds=prof.metrics.seconds * 2,
        calls=3,
    )
    summary = profile_mod.summarize_hotspots([hotspot])
    json.dumps(summary)

    assert hotspot.name == "mamba3_scan"
    assert hotspot.route == "M"
    assert hotspot.backend == "mlx"
    assert hotspot.operation == "mamba3_scan"
    assert hotspot.calls == 3
    assert 0 < hotspot.fraction <= 0.5
    assert summary["count"] == 1
    hotspots = summary["hotspots"]
    assert isinstance(hotspots, list)
    top = hotspots[0]
    assert isinstance(top, dict)
    assert top["name"] == "mamba3_scan"


def test_kernel_adoption_gate_fails_closed_without_profile_samples() -> None:
    assessment = profile_mod.assess_kernel_adoption("mamba3-metal-scan", [])

    payload = assessment.to_dict()
    json.dumps(payload)

    assert assessment.allowed is False
    assert "need at least 1 profile sample" in assessment.reason
    assert payload["allowed"] is False
    assert payload["top_hotspot"] is None
    with pytest.raises(KernelAdoptionBlocked, match="need at least 1 profile sample"):
        profile_mod.require_kernel_hotspot_evidence("mamba3-metal-scan", [])


def test_kernel_adoption_gate_blocks_weak_hotspot_fraction() -> None:
    evidence = [
        HotspotEvidence(
            name="tiny-activation",
            seconds=0.005,
            total_seconds=1.0,
            route="A",
            backend="mlx",
        )
    ]

    assessment = profile_mod.assess_kernel_adoption(
        "activation-metal",
        evidence,
        min_hotspot_fraction=0.10,
        min_hotspot_seconds=0.001,
    )

    assert assessment.allowed is False
    assert "below required 0.100" in assessment.reason
    with pytest.raises(KernelAdoptionBlocked, match="below required 0.100"):
        profile_mod.require_kernel_hotspot_evidence(
            "activation-metal",
            evidence,
            min_hotspot_fraction=0.10,
            min_hotspot_seconds=0.001,
        )


def test_kernel_adoption_gate_allows_measured_hotspot() -> None:
    evidence = [
        HotspotEvidence(
            name="m2rnn-recurrence",
            seconds=0.25,
            total_seconds=1.0,
            calls=8,
            route="R",
            backend="mlx",
            operation="m2rnn",
        ),
        HotspotEvidence(
            name="loss",
            seconds=0.05,
            total_seconds=1.0,
            calls=8,
        ),
    ]

    assessment = profile_mod.require_kernel_hotspot_evidence(
        "m2rnn-metal-recurrence",
        evidence,
        min_hotspot_fraction=0.20,
        min_hotspot_seconds=0.10,
        min_samples=2,
    )
    payload = assessment.to_dict()
    json.dumps(payload)

    assert assessment.allowed is True
    assert assessment.top_hotspot is not None
    assert assessment.top_hotspot.name == "m2rnn-recurrence"
    assert "measured hotspot" in assessment.reason
    payload_summary = payload["summary"]
    assert isinstance(payload_summary, dict)
    assert payload_summary["count"] == 2


def test_hotspot_evidence_validates_shape() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        HotspotEvidence(name="", seconds=1.0, total_seconds=1.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        HotspotEvidence(name="bad", seconds=math.nan, total_seconds=1.0)
    with pytest.raises(ValueError, match=">= seconds"):
        HotspotEvidence(name="bad", seconds=2.0, total_seconds=1.0)
    with pytest.raises(ValueError, match="calls must be positive"):
        HotspotEvidence(name="bad", seconds=1.0, total_seconds=1.0, calls=0)


def test_kernel_adoption_gate_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="candidate_kernel must be non-empty"):
        profile_mod.assess_kernel_adoption("", [])
    with pytest.raises(ValueError, match="min_hotspot_fraction"):
        profile_mod.assess_kernel_adoption("x", [], min_hotspot_fraction=math.nan)
    with pytest.raises(ValueError, match="min_hotspot_seconds"):
        profile_mod.assess_kernel_adoption("x", [], min_hotspot_seconds=-1.0)
    with pytest.raises(ValueError, match="min_samples"):
        profile_mod.assess_kernel_adoption("x", [], min_samples=0)
    with pytest.raises(ValueError, match="top_n"):
        profile_mod.summarize_hotspots([], top_n=0)
