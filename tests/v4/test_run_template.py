"""Tests for cppmega_v4.run_template — YAML/JSON declarative block spec."""

import json
from pathlib import Path

import pytest

from cppmega_v4.run_template import (
    HAS_YAML,
    SUPPORTED_BLOCK_KINDS,
    BlockSpec,
    MTPSpec,
    RunTemplate,
    dump_template,
    dumps_template,
    load_template,
    loads_template,
)


def _sample_template() -> RunTemplate:
    return RunTemplate(
        name="v4_test_1b",
        hidden_size=2048,
        vocab_size=32000,
        blocks=[
            BlockSpec(kind="gdn", repeat=2, params={"num_heads": 16, "head_dim_k": 128}),
            BlockSpec(kind="mla_absorb", repeat=4,
                      params={"num_heads": 16, "q_lora_rank": 1024, "kv_lora_rank": 256}),
            BlockSpec(kind="engram", repeat=1,
                      params={"num_branches": 4, "branch_dim": 512}),
            BlockSpec(kind="moe", repeat=2,
                      params={"num_experts": 64, "num_experts_per_tok": 8}),
        ],
        mtp=MTPSpec(depth=2),
    )


# ----- BlockSpec validation -----


def test_block_spec_accepts_supported_kind():
    spec = BlockSpec(kind="gdn", repeat=1, params={"x": 1})
    assert spec.kind == "gdn"


def test_block_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported block kind"):
        BlockSpec(kind="not_a_real_block")


def test_block_spec_rejects_zero_or_negative_repeat():
    with pytest.raises(ValueError, match="positive int"):
        BlockSpec(kind="gdn", repeat=0)
    with pytest.raises(ValueError, match="positive int"):
        BlockSpec(kind="gdn", repeat=-1)


def test_block_spec_rejects_non_dict_params():
    with pytest.raises(ValueError, match="must be a dict"):
        BlockSpec(kind="gdn", params="not-a-dict")  # type: ignore[arg-type]


# ----- MTPSpec validation -----


def test_mtp_spec_accepts_depth_zero():
    spec = MTPSpec(depth=0)
    assert spec.depth == 0


def test_mtp_spec_rejects_negative_depth():
    with pytest.raises(ValueError, match="non-negative"):
        MTPSpec(depth=-1)


def test_mtp_spec_rejects_zero_hidden_size_override():
    with pytest.raises(ValueError, match="positive or None"):
        MTPSpec(depth=2, hidden_size_override=0)


# ----- RunTemplate validation -----


def test_run_template_requires_non_empty_blocks():
    with pytest.raises(ValueError, match="non-empty"):
        RunTemplate(name="x", hidden_size=8, blocks=[])


def test_run_template_rejects_non_positive_hidden_size():
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        RunTemplate(name="x", hidden_size=0,
                    blocks=[BlockSpec(kind="gdn")])


def test_run_template_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        RunTemplate(name="", hidden_size=8,
                    blocks=[BlockSpec(kind="gdn")])


def test_run_template_total_blocks_sums_repeat():
    t = _sample_template()
    # 2 + 4 + 1 + 2 = 9
    assert t.total_blocks() == 9


def test_run_template_block_kinds_used():
    t = _sample_template()
    assert t.block_kinds_used() == {"gdn", "mla_absorb", "engram", "moe"}


# ----- Round-trip: dict / JSON / YAML -----


def test_round_trip_via_dict():
    t = _sample_template()
    d = t.to_dict()
    t2 = RunTemplate.from_dict(d)
    assert t2.to_dict() == d


def test_round_trip_via_json_string():
    t = _sample_template()
    s = dumps_template(t, fmt="json")
    parsed = json.loads(s)
    assert parsed["schema_version"] == 1
    t2 = loads_template(s, fmt="json")
    assert t2.to_dict() == t.to_dict()


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_round_trip_via_yaml_string():
    t = _sample_template()
    s = dumps_template(t, fmt="yaml")
    assert "name: v4_test_1b" in s
    t2 = loads_template(s, fmt="yaml")
    assert t2.to_dict() == t.to_dict()


def test_round_trip_via_json_file(tmp_path: Path):
    t = _sample_template()
    p = tmp_path / "run.json"
    dump_template(t, p)
    assert p.exists()
    t2 = load_template(p)
    assert t2.to_dict() == t.to_dict()


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_round_trip_via_yaml_file(tmp_path: Path):
    t = _sample_template()
    p = tmp_path / "run.yaml"
    dump_template(t, p)
    assert p.exists()
    t2 = load_template(p)
    assert t2.to_dict() == t.to_dict()


def test_unknown_extension_rejected(tmp_path: Path):
    p = tmp_path / "run.toml"
    with pytest.raises(ValueError, match="unsupported template extension"):
        dump_template(_sample_template(), p)


def test_schema_version_too_new_rejected():
    data = _sample_template().to_dict()
    data["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        RunTemplate.from_dict(data)


def test_supported_block_kinds_covers_all_v4_blocks():
    """All V4 ROIs are addressable by name from the template."""
    must_have = {"gdn", "kda", "mla_absorb", "engram", "moe",
                 "lightning_indexer", "nsa", "csa_hca"}
    assert must_have.issubset(SUPPORTED_BLOCK_KINDS)


def test_template_without_mtp_optional():
    t = RunTemplate(name="no-mtp", hidden_size=8,
                    blocks=[BlockSpec(kind="gdn")])
    assert t.mtp is None
    d = t.to_dict()
    assert "mtp" not in d
    assert RunTemplate.from_dict(d).mtp is None
