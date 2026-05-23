"""V8-R01 unit tests: preset_training_defaults table + build_preset_specs.

Asserts:
  * The table has >= 30 explicit paper-anchored rows.
  * Every PRESETS key (including parallel-block specs) resolves to a
    valid TrainingDefaults via the family-fallback path.
  * to_wire produces JSON-serialisable dicts (no tuples).
  * build_preset_specs RPC carries the defaults block end-to-end for
    the canonical AC preset (llama3_8b -> lr=3e-4, schedule='wsd').
"""

from __future__ import annotations

import json

import pytest

from cppmega_v4.architectures import PRESETS
from cppmega_v4.architectures.preset_training_defaults import (
    DEFAULTS, FAMILY_DEFAULTS, TrainingDefaults, get_defaults, known_keys,
    to_wire,
)
from cppmega_v4.jsonrpc.methods import build_preset_specs
from cppmega_v4.jsonrpc.schema import BuildPresetSpecsParams


def test_explicit_table_has_at_least_30_entries():
    assert len(DEFAULTS) >= 30, (
        f"R01 AC requires >=30 paper-anchored entries, got {len(DEFAULTS)}")


def test_every_explicit_entry_is_well_formed():
    """Every paper-anchored entry must pass field validation."""
    valid_schedules = {
        "constant", "linear_warmup", "cosine", "wsd", "inv_sqrt"}
    for name, td in DEFAULTS.items():
        assert isinstance(td, TrainingDefaults), name
        assert td.lr > 0 and td.lr < 1, f"{name}: lr {td.lr} out of range"
        assert td.batch_size > 0
        assert td.schedule in valid_schedules, (
            f"{name}: unknown schedule {td.schedule!r}")
        assert td.warmup_steps >= 0
        assert td.betas is None or (
            0 < td.betas[0] < 1 and 0 < td.betas[1] < 1), name
        assert td.gradient_clip > 0
        assert isinstance(td.mixed_precision, bool)
        assert td.optimizer in {
            "adamw", "muon", "muon_adamw_hybrid", "lion", "adam8bit"}
        assert td.source_paper_url.startswith(("http://", "https://"))


def test_every_preset_resolves_to_defaults():
    """No preset in PRESETS may fall off the family-fallback path."""
    for name in PRESETS:
        td = get_defaults(name)
        assert isinstance(td, TrainingDefaults), (
            f"{name} did not resolve to TrainingDefaults")
        assert td.lr > 0


def test_known_keys_returns_sorted_unique_tuple():
    keys = known_keys()
    assert keys == tuple(sorted(set(keys)))
    assert set(keys) == set(DEFAULTS)


def test_to_wire_is_json_serialisable():
    """to_wire must emit pure-JSON (lists not tuples)."""
    for name, td in DEFAULTS.items():
        wire = to_wire(td)
        # Round-trip through json — will raise if any non-JSON value.
        encoded = json.dumps(wire)
        decoded = json.loads(encoded)
        assert decoded["lr"] == td.lr
        if td.betas is not None:
            # tuples become lists in wire form
            assert isinstance(wire["betas"], list)
            assert wire["betas"] == list(td.betas)
        else:
            assert wire["betas"] is None
        # Field set is exactly the spec
        assert set(wire) == {
            "lr", "batch_size", "schedule", "warmup_steps", "betas",
            "gradient_clip", "mixed_precision", "optimizer",
            "source_paper_url"}


def test_llama3_8b_ac_row():
    """AC from R01 ticket: pick llama3_8b -> lr=3e-4, schedule=wsd."""
    td = DEFAULTS["llama3_8b"]
    assert td.lr == pytest.approx(3e-4)
    assert td.schedule == "wsd"
    assert td.optimizer == "adamw"
    assert td.mixed_precision is True


def test_family_fallback_unknown_qwen3_variant():
    """An unknown 'qwen3_*' preset must fall back to qwen3 family row."""
    td = get_defaults("qwen3_nonexistent_xyz")
    assert td.optimizer == "adamw"
    # Matches FAMILY_DEFAULTS["qwen3"] which is qwen3_dense_8b
    assert td == FAMILY_DEFAULTS["qwen3"]


def test_family_fallback_longest_prefix_wins():
    """gpt_oss_* must hit 'gpt_oss' rather than 'gpt2'."""
    td = get_defaults("gpt_oss_unknown_size")
    assert td == FAMILY_DEFAULTS["gpt_oss"]


def test_generic_fallback_for_unknown_family():
    """Totally unknown name must still resolve (never raise)."""
    td = get_defaults("zzz_invented_arch")
    assert isinstance(td, TrainingDefaults)
    assert td.lr > 0


def test_build_preset_specs_rpc_carries_defaults():
    """End-to-end: RPC result must include the defaults block."""
    res = build_preset_specs(BuildPresetSpecsParams(
        preset_name="llama3_8b", hidden_size=64))
    assert res.preset_name == "llama3_8b"
    assert res.defaults["lr"] == pytest.approx(3e-4)
    assert res.defaults["schedule"] == "wsd"
    assert res.defaults["optimizer"] == "adamw"
    assert "arxiv.org" in res.defaults["source_paper_url"]
    # specs still wired correctly (not regressed)
    assert isinstance(res.specs, list) and len(res.specs) >= 1


def test_build_preset_specs_defaults_for_family_fallback():
    """Even a preset without explicit row gets a defaults block via
    family fallback (qwen3_6_27b -> qwen3_dense_8b row)."""
    res = build_preset_specs(BuildPresetSpecsParams(
        preset_name="qwen3_6_27b", hidden_size=64))
    assert res.defaults is not None
    assert res.defaults["optimizer"] == "adamw"
    assert res.defaults["lr"] > 0
