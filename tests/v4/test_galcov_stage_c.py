"""GalCov Stage C tests — parallel-block composition в preset DSL."""

from __future__ import annotations

import pytest

from cppmega_v4.architectures.presets import tiny_aya_parallel_specs
from cppmega_v4.fusion import from_block_specs


def test_parallel_block_emits_branch_nodes_with_fan_in_fan_out():
    specs = [
        {"kind": "mlp", "name": "pre"},
        {"parallel": [
            {"kind": "attention", "name": "branch_a", "params": {"num_heads": 4}},
            {"kind": "mlp", "name": "branch_b"},
        ]},
        {"kind": "mlp", "name": "post"},
    ]
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    names = {n.name for n in g.nodes}
    assert names == {"pre", "branch_a", "branch_b", "post"}
    edges = set(g.edges)
    # pre fans out to both branches
    assert ("pre", "branch_a") in edges
    assert ("pre", "branch_b") in edges
    # both branches fan in to post
    assert ("branch_a", "post") in edges
    assert ("branch_b", "post") in edges
    # no spurious direct edges
    assert ("pre", "post") not in edges
    assert ("branch_a", "branch_b") not in edges


def test_parallel_block_at_graph_start_has_no_inbound_edges():
    specs = [
        {"parallel": [
            {"kind": "mlp", "name": "a"},
            {"kind": "mlp", "name": "b"},
        ]},
        {"kind": "mlp", "name": "tail"},
    ]
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    assert ("a", "tail") in g.edges
    assert ("b", "tail") in g.edges
    assert len(g.edges) == 2


def test_parallel_block_at_graph_end_has_no_outbound_edges():
    specs = [
        {"kind": "mlp", "name": "head"},
        {"parallel": [
            {"kind": "mlp", "name": "a"},
            {"kind": "mlp", "name": "b"},
        ]},
    ]
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    assert ("head", "a") in g.edges
    assert ("head", "b") in g.edges
    assert len(g.edges) == 2


def test_parallel_to_parallel_cross_product():
    specs = [
        {"parallel": [
            {"kind": "mlp", "name": "a1"},
            {"kind": "mlp", "name": "a2"},
        ]},
        {"parallel": [
            {"kind": "mlp", "name": "b1"},
            {"kind": "mlp", "name": "b2"},
        ]},
    ]
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    edges = set(g.edges)
    assert edges == {("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")}


def test_parallel_block_rejects_empty_branch_list():
    specs = [{"parallel": []}]
    with pytest.raises(ValueError, match="≥1 branch"):
        from_block_specs(specs, hidden_size=64, instantiate=False)


def test_parallel_block_rejects_nested_parallel_in_branch():
    """Branch entries must be leaf-specs (no nested 'parallel' for v1)."""
    specs = [{"parallel": [{"parallel": [{"kind": "mlp", "name": "x"}]}]}]
    with pytest.raises(ValueError, match="leaf-specs"):
        from_block_specs(specs, hidden_size=64, instantiate=False)


def test_parallel_block_rejects_duplicate_branch_name():
    specs = [
        {"parallel": [
            {"kind": "mlp", "name": "dup"},
            {"kind": "mlp", "name": "dup"},
        ]},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        from_block_specs(specs, hidden_size=64, instantiate=False)


def test_tiny_aya_parallel_specs_compose_parallel_block():
    """Demo: Tiny Aya-style parallel GQA + MLP between pre/post bricks."""
    specs = tiny_aya_parallel_specs(64)
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    names = {n.name for n in g.nodes}
    assert names == {"tap_pre", "tap_gqa", "tap_mlp", "tap_moe"}
    edges = set(g.edges)
    assert ("tap_pre", "tap_gqa") in edges
    assert ("tap_pre", "tap_mlp") in edges
    assert ("tap_gqa", "tap_moe") in edges
    assert ("tap_mlp", "tap_moe") in edges
    assert ("tap_pre", "tap_moe") not in edges
    assert ("tap_gqa", "tap_mlp") not in edges


def test_tiny_aya_parallel_specs_instantiate():
    specs = tiny_aya_parallel_specs(64)
    g = from_block_specs(specs, hidden_size=64, instantiate=True)
    assert len(g.nodes) == 4


def test_leaf_only_specs_still_work_unchanged():
    """Backwards compat: existing linear-chain specs unaffected."""
    specs = [
        {"kind": "mlp", "name": "a"},
        {"kind": "mlp", "name": "b"},
        {"kind": "mlp", "name": "c"},
    ]
    g = from_block_specs(specs, hidden_size=64, instantiate=False)
    assert g.edges == (("a", "b"), ("b", "c"))
