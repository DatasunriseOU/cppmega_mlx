"""Stage B tests — descriptor_synthesizer + extended registry."""

from __future__ import annotations

import pytest

from cppmega_mlx.runtime.path_c_fusion_schedules import (
    PathCBrickScheduleDescriptor,
    PathCBrickScheduleDescriptorRegistry,
    default_path_c_brick_schedule_descriptor_registry,
)
from cppmega_v4.fusion.descriptor_synthesizer import (
    build_v4_extended_registry,
    synthesize_descriptor_for_brick,
)
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS


# ---------------------------------------------------------------------------
# Unit: per-brick synthesis
# ---------------------------------------------------------------------------


def test_synth_for_known_brick_returns_descriptor():
    d = synthesize_descriptor_for_brick("gdn")
    assert isinstance(d, PathCBrickScheduleDescriptor)
    assert d.op_name == "gdn"
    assert d.schedule_family == "linear_attn_scan"
    assert d.supports_backward is True
    assert len(d.required_codegen_steps) >= 1


def test_synth_for_sdpa_brick_is_opaque():
    d = synthesize_descriptor_for_brick("gated_attention")
    assert d.schedule_family == "opaque_sdpa_with_outputs"
    assert d.implementation_status == "opaque_brick_passthrough"
    assert d.supports_backward is False


def test_synth_for_unknown_kind_returns_safe_passthrough():
    """Synthesizer must never raise for unknown kinds — produces an
    opaque_passthrough descriptor instead."""
    d = synthesize_descriptor_for_brick("totally_made_up_kind")
    assert d.op_name == "totally_made_up_kind"
    assert d.schedule_family == "opaque_passthrough"


@pytest.mark.parametrize("kind", sorted(BLOCK_BUILDERS.keys()))
def test_synth_for_every_block_builder_kind_is_structurally_valid(kind):
    """Every kind in BLOCK_BUILDERS must produce a registry-acceptable
    descriptor (non-empty op_name, codegen steps, family)."""
    d = synthesize_descriptor_for_brick(kind)
    assert d.op_name == kind
    assert d.required_codegen_steps  # non-empty
    assert d.schedule_family
    # registry.register raises on invalid descriptors
    PathCBrickScheduleDescriptorRegistry().register(d)


# ---------------------------------------------------------------------------
# Unit: build_v4_extended_registry
# ---------------------------------------------------------------------------


def test_extended_registry_includes_v3_legacy_descriptors():
    reg = build_v4_extended_registry()
    # The v3 legacy entries from default_path_c_brick_schedule_descriptor_registry
    legacy = ("mamba3_mimo", "residual_rmsnorm", "m2rnn",
              "attention_qkv_projection", "sparse_mla_fp8_apply")
    for op in legacy:
        d = reg.descriptor_for(op)
        assert d is not None, f"legacy descriptor {op!r} dropped"
        # legacy ones have a different status — not the v4 synth tag
        assert d.implementation_status != "opaque_brick_passthrough"


def test_extended_registry_does_not_clobber_legacy_when_kind_collides():
    """If a future BLOCK_BUILDERS key matches a legacy op_name, the
    legacy descriptor wins (descriptor_for returns it, not the synth)."""
    reg = build_v4_extended_registry()
    legacy_status = default_path_c_brick_schedule_descriptor_registry()\
        .descriptor_for("residual_rmsnorm").implementation_status
    assert reg.descriptor_for("residual_rmsnorm").implementation_status == legacy_status


def test_extended_registry_covers_every_block_builder_kind():
    reg = build_v4_extended_registry()
    for kind in BLOCK_BUILDERS:
        d = reg.descriptor_for(kind)
        assert d is not None, f"missing descriptor for {kind!r}"


def test_extended_registry_aot_backward_synthesis_only_for_supporting_kinds():
    """The registry has a built-in :code:`<op>_bwd` synthesizer that only
    fires for descriptors with supports_backward=True. Verify the v4
    fallback descriptors respect that contract."""
    reg = build_v4_extended_registry()
    # gdn: supports_backward=True (linear_attn category)
    assert reg.descriptor_for("gdn_bwd") is not None
    # gated_attention: supports_backward=False (sdpa_attention category)
    assert reg.descriptor_for("gated_attention_bwd") is None


# ---------------------------------------------------------------------------
# System: signature lookup for a real Qwen3-Next-shaped chain
# ---------------------------------------------------------------------------


def test_system_descriptors_for_qwen3_next_chain():
    """Build the descriptor list for a Qwen3-Next-style 5-brick chain."""
    reg = build_v4_extended_registry()
    signature = ("gdn", "gdn", "gdn", "gated_attention", "moe")
    descriptors = reg.descriptors_for_signature(signature)
    assert descriptors is not None
    assert len(descriptors) == len(signature)
    for op_name, d in zip(signature, descriptors):
        assert d.op_name == op_name


def test_system_descriptors_for_ling26_chain():
    """Ling 2.6's 7:1 bailing_linear + bailing_mla + bailing_moe."""
    reg = build_v4_extended_registry()
    signature = ("bailing_linear",) * 7 + ("bailing_mla", "bailing_moe")
    descriptors = reg.descriptors_for_signature(signature)
    assert descriptors is not None
    assert len(descriptors) == 9
