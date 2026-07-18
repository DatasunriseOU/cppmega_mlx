"""Side-channel conditioning spec tests.

Stage A locks the pure data/config layer before any runtime conditioning
behavior is added.
"""

from __future__ import annotations

import pytest

from cppmega_v4.buildspec import (
    DataMaterializationSpec,
    FamilySpec,
    InferenceEnrichmentSpec,
    SideChannelSpec,
)


def test_side_channel_spec_defaults_are_language_neutral():
    spec = SideChannelSpec()

    assert spec.mode == "auto"
    assert tuple(spec.families) == (
        "platform",
        "syntax",
        "structure",
        "semantic_graph",
        "temporal_diff",
    )
    assert spec.families["platform"].columns == (
        "platform_ids",
        "source_platform_ids",
    )
    assert spec.families["syntax"].language_scope == ("any",)
    assert spec.inference.source == "auto"
    assert spec.inference.fail_policy == "error"


def test_side_channel_spec_json_round_trip_is_deterministic():
    spec = SideChannelSpec(
        families={
            "platform": FamilySpec(
                mode="require",
                columns=("platform_ids",),
                dropout=0.2,
                fallback="error",
            ),
            "syntax": FamilySpec(mode="off"),
        },
        inference=InferenceEnrichmentSpec(source="prompt_only", timeout_ms=250),
    )

    payload = spec.to_dict()
    assert payload["families"]["platform"]["mode"] == "require"
    assert payload["families"]["platform"]["columns"] == ["platform_ids"]
    assert payload["families"]["syntax"]["mode"] == "off"

    restored = SideChannelSpec.from_dict(payload)
    assert restored == spec
    assert restored.to_dict() == payload


def test_family_spec_rejects_invalid_policy_values():
    with pytest.raises(ValueError, match="mode"):
        FamilySpec(mode="sometimes")
    with pytest.raises(ValueError, match="dropout"):
        FamilySpec(dropout=-0.1)
    with pytest.raises(ValueError, match="dropout"):
        FamilySpec(dropout=1.1)
    with pytest.raises(ValueError, match="fallback"):
        FamilySpec(fallback="silently_copy_large_tensor")
    with pytest.raises(ValueError, match="language_scope"):
        FamilySpec(language_scope=("",))


def test_inference_enrichment_spec_rejects_invalid_values():
    with pytest.raises(ValueError, match="source"):
        InferenceEnrichmentSpec(source="magic")
    with pytest.raises(ValueError, match="fail_policy"):
        InferenceEnrichmentSpec(fail_policy="guess")
    with pytest.raises(ValueError, match="timeout_ms"):
        InferenceEnrichmentSpec(timeout_ms=-1)


def test_data_materialization_spec_locks_packing_contract():
    spec = DataMaterializationSpec()

    assert spec.packing_policy == "best_fit"
    assert spec.max_seq_len == 4096
    assert spec.pad_to_max is True
    assert spec.include_provenance is True
    assert "input_ids" in spec.required_token_fields
    assert "target_ids" in spec.required_token_fields

    with pytest.raises(ValueError, match="packing_policy"):
        DataMaterializationSpec(packing_policy="random")
    with pytest.raises(ValueError, match="max_seq_len"):
        DataMaterializationSpec(max_seq_len=0)
