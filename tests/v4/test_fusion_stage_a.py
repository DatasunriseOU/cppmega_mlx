"""Stage A tests — brick_graph + compatibility + dlpack_bridge.

Unit tests cover individual modules; the bottom section runs a small
system-test that walks a Qwen3-Next-shaped block sequence end-to-end
through ``from_block_specs`` and asserts the pair-compatibility map
matches the expected 3:1 hybrid pattern.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from cppmega_v4.fusion import (
    BrickGraph,
    BrickNode,
    FusionEligibility,
    can_fuse_pair,
    dlpack_available,
    from_block_specs,
    from_mlx_model,
    host_copy_fallback,
    mlx_to_tilelang,
    tilelang_to_mlx,
)


# ---------------------------------------------------------------------------
# BrickNode / BrickGraph unit tests
# ---------------------------------------------------------------------------


def test_bricknode_rejects_unknown_kind():
    with pytest.raises(ValueError, match="not in BLOCK_BUILDERS"):
        BrickNode(kind="totally_made_up", name="x")


def test_bricknode_rejects_empty_name():
    with pytest.raises(ValueError, match="name must be non-empty"):
        BrickNode(kind="gdn", name="")


def test_brickgraph_rejects_duplicate_names():
    a = BrickNode(kind="gdn", name="dup")
    b = BrickNode(kind="kda", name="dup")
    with pytest.raises(ValueError, match="duplicate brick name"):
        BrickGraph(nodes=(a, b))


def test_brickgraph_rejects_dangling_edge():
    a = BrickNode(kind="gdn", name="a")
    with pytest.raises(ValueError, match="not in node set"):
        BrickGraph(nodes=(a,), edges=(("a", "ghost"),))


def test_brickgraph_successors_predecessors():
    nodes = tuple(BrickNode(kind="gdn", name=f"n{i}") for i in range(3))
    g = BrickGraph(nodes=nodes, edges=(("n0", "n1"), ("n1", "n2")))
    assert g.successors("n0") == ("n1",)
    assert g.successors("n2") == ()
    assert g.predecessors("n0") == ()
    assert g.predecessors("n2") == ("n1",)


# ---------------------------------------------------------------------------
# from_block_specs
# ---------------------------------------------------------------------------


def test_from_block_specs_linear_chain_with_modules():
    g = from_block_specs(
        [
            {"kind": "gdn", "name": "g0", "params": {}},
            {"kind": "gdn", "name": "g1", "params": {}},
            {"kind": "moe", "name": "moe", "params": {"num_experts": 2, "top_k": 1}},
        ],
        hidden_size=64,
        instantiate=True,
    )
    assert g.names == ("g0", "g1", "moe")
    assert g.edges == (("g0", "g1"), ("g1", "moe"))
    for node in g.nodes:
        assert isinstance(node.module, nn.Module)


def test_from_block_specs_can_skip_instantiation():
    g = from_block_specs(
        [{"kind": "gdn", "name": "g0", "params": {}}],
        hidden_size=64,
        instantiate=False,
    )
    assert g.nodes[0].module is None


def test_from_block_specs_auto_names_when_missing():
    g = from_block_specs(
        [{"kind": "gdn", "params": {}}, {"kind": "gdn", "params": {}}],
        hidden_size=64,
        instantiate=False,
    )
    assert g.names == ("gdn_0", "gdn_1")


def test_from_block_specs_rejects_duplicate_names_eagerly():
    with pytest.raises(ValueError, match="duplicate name"):
        from_block_specs(
            [
                {"kind": "gdn", "name": "dup", "params": {}},
                {"kind": "gdn", "name": "dup", "params": {}},
            ],
            hidden_size=64,
            instantiate=False,
        )


# ---------------------------------------------------------------------------
# from_mlx_model
# ---------------------------------------------------------------------------


def test_from_mlx_model_picks_up_branded_children():
    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(32, 32)
            self.attn._v4_brick_kind = "gdn"  # mark for walker
            self.proj = nn.Linear(32, 32)
            self.proj._v4_brick_kind = "mlp"

    m = _Wrapper()
    g = from_mlx_model(m, attr_order=["attn", "proj"])
    assert g.names == ("attn", "proj")
    assert tuple(n.kind for n in g.nodes) == ("gdn", "mlp")
    assert g.edges == (("attn", "proj"),)


def test_from_mlx_model_skips_unbranded_children():
    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(32, 32)
            self.attn._v4_brick_kind = "gdn"
            self.bias_holder = nn.Linear(32, 32)  # not a brick

    m = _Wrapper()
    g = from_mlx_model(m, attr_order=["attn", "bias_holder"])
    assert g.names == ("attn",)


# ---------------------------------------------------------------------------
# compatibility.can_fuse_pair
# ---------------------------------------------------------------------------


def _pair(kind_a: str, kind_b: str) -> FusionEligibility:
    a = BrickNode(kind=kind_a, name="a")
    b = BrickNode(kind=kind_b, name="b")
    return can_fuse_pair(a, b)


def test_compat_linear_attn_chain_fuses_via_path_c():
    elig = _pair("gdn", "gdn")
    assert elig.can_fuse is True
    assert elig.backend == "path_c"


def test_compat_two_sdpa_dont_fuse():
    elig = _pair("gated_attention", "mla")
    assert elig.can_fuse is False
    assert elig.backend == "dlpack_handoff"


def test_compat_linear_attn_to_sdpa_uses_handoff():
    elig = _pair("gdn", "gated_attention")
    assert elig.can_fuse is False
    assert elig.backend == "dlpack_handoff"


def test_compat_sdpa_to_norm_fuses():
    elig = _pair("gated_attention", "mlp")
    assert elig.can_fuse is True
    assert elig.backend == "path_c"


def test_compat_moe_routing_is_hard_boundary():
    for partner in ("gdn", "gated_attention", "mla", "ssm"):
        elig = _pair("moe", partner)
        assert elig.can_fuse is False, partner
        assert elig.backend == "dlpack_handoff", partner


def test_compat_sparse_attn_never_fuses():
    for partner in ("gdn", "gated_attention", "mlp", "moe"):
        elig = _pair("nsa", partner)
        assert elig.can_fuse is False, partner


def test_compat_unknown_kind_yields_handoff_with_clear_reason():
    # We rely on the fact that BLOCK_BUILDERS rejects unknown kinds, but
    # the oracle itself should never raise — feed a fabricated node via
    # bypass and verify the safe-default path.
    a = BrickNode.__new__(BrickNode)
    object.__setattr__(a, "kind", "alien_kind")
    object.__setattr__(a, "name", "a")
    object.__setattr__(a, "params", {})
    object.__setattr__(a, "module", None)
    b = BrickNode(kind="gdn", name="b")
    elig = can_fuse_pair(a, b)
    assert elig.can_fuse is False
    assert "unknown category" in elig.reason


# ---------------------------------------------------------------------------
# dlpack_bridge
# ---------------------------------------------------------------------------


def test_dlpack_available_returns_bool():
    assert isinstance(dlpack_available(), bool)


def test_dlpack_roundtrip_mlx_to_tvm_to_mlx_when_available():
    if not dlpack_available():
        pytest.skip("tvm not importable on this host")
    src = mx.array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    nda = mlx_to_tilelang(src)
    back = tilelang_to_mlx(nda)
    np.testing.assert_array_equal(np.array(back), np.array(src))


def test_host_copy_fallback_accepts_mlx_passthrough():
    src = mx.array([1.0, 2.0, 3.0])
    out = host_copy_fallback(src)
    assert isinstance(out, mx.array)
    assert (out == src).all().item()


def test_host_copy_fallback_accepts_numpy_array():
    arr = np.array([1, 2, 3], dtype=np.float32)
    out = host_copy_fallback(arr)
    assert isinstance(out, mx.array)
    np.testing.assert_array_equal(np.array(out), arr)


# ---------------------------------------------------------------------------
# System test — Qwen3-Next-shaped pattern detection
# ---------------------------------------------------------------------------


def test_system_qwen3_next_pattern_3_gdn_then_attn_then_moe():
    """Build a Qwen3-Next-style 5-brick block and verify oracle classification.

    Expected boundaries: g0->g1->g2 fuse (linear-attn chain), g2->attn breaks
    (handoff to SDPA), attn->moe breaks (routing boundary).
    """
    g = from_block_specs(
        [
            {"kind": "gdn", "name": "g0"},
            {"kind": "gdn", "name": "g1"},
            {"kind": "gdn", "name": "g2"},
            {"kind": "gated_attention", "name": "attn",
             "params": {"num_attention_heads": 4, "num_key_value_heads": 2,
                        "head_dim": 16}},
            {"kind": "moe", "name": "moe",
             "params": {"num_experts": 2, "top_k": 1}},
        ],
        hidden_size=64,
        instantiate=True,
    )

    expected = {
        ("g0", "g1"): True,
        ("g1", "g2"): True,
        ("g2", "attn"): False,
        ("attn", "moe"): False,
    }
    for (p, c), can in expected.items():
        elig = can_fuse_pair(g.by_name(p), g.by_name(c))
        assert elig.can_fuse is can, f"{p}->{c}: expected {can}, got {elig}"


def test_system_ling26_pattern_7_bailing_linear_then_mla():
    """Ling 2.6's 7:1 pattern — 7 linear-attn fuse together, MLA breaks."""
    specs = [{"kind": "bailing_linear", "name": f"la{i}",
              "params": {"num_attention_heads": 4, "num_key_value_heads": 2,
                         "head_dim": 16}} for i in range(7)]
    specs.append(
        {"kind": "bailing_mla", "name": "mla",
         "params": {"num_attention_heads": 4, "num_key_value_heads": 2,
                    "head_dim": 16, "kv_lora_rank": 16, "qk_rope_head_dim": 8,
                    "qk_nope_head_dim": 16, "v_head_dim": 16}}
    )
    g = from_block_specs(specs, hidden_size=64, instantiate=True)
    # All consecutive bailing_linear pairs should fuse
    for i in range(6):
        elig = can_fuse_pair(g.nodes[i], g.nodes[i + 1])
        assert elig.can_fuse is True
    # Last bailing_linear -> bailing_mla should NOT fuse
    elig = can_fuse_pair(g.nodes[6], g.nodes[7])
    assert elig.can_fuse is False
