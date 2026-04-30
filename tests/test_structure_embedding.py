import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.nn.structure_embedding import CppMegaStructureEmbedding


def to_numpy(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


def test_structure_embedding_core_returns_hidden_sized_tensor_with_zero_init():
    module = CppMegaStructureEmbedding(hidden_size=32, active_components="core", bottleneck_dim=8)
    structure_ids = mx.array([[1, 2, 3], [0, 1, 2]], dtype=mx.int64)
    dep_levels = mx.array([[0, 1, 2], [2, 1, 0]], dtype=mx.int64)

    out = module(structure_ids=structure_ids, dep_levels=dep_levels, target_dtype=mx.float32)

    assert out.shape == (2, 3, 32)
    assert np.count_nonzero(to_numpy(out)) == 0


def test_structure_embedding_parses_core_all_and_custom_components():
    assert CppMegaStructureEmbedding._parse_components("core") == ("structure", "dep_level")
    assert CppMegaStructureEmbedding._parse_components("all") == (
        "structure",
        "dep_level",
        "ast_depth",
        "sibling_index",
        "ast_node_type",
    )
    assert CppMegaStructureEmbedding._parse_components("dep_level,ast_node_type") == (
        "dep_level",
        "ast_node_type",
    )


def test_structure_embedding_unknown_or_empty_component_spec_fails_closed():
    with pytest.raises(ValueError, match="unknown"):
        CppMegaStructureEmbedding(hidden_size=16, active_components="core,unknown")
    with pytest.raises(ValueError, match="at least one"):
        CppMegaStructureEmbedding(hidden_size=16, active_components="")


def test_structure_embedding_clamps_ids_and_masks_missing_components():
    module = CppMegaStructureEmbedding(
        hidden_size=4,
        active_components="core",
        num_categories=3,
        max_dep_level=4,
        bottleneck_dim=2,
    )
    module.stacked_emb.weight = mx.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 10.0],
            [0.0, 20.0],
            [0.0, 30.0],
            [0.0, 40.0],
        ],
        dtype=mx.float32,
    )
    module.up_proj.weight = mx.ones_like(module.up_proj.weight)

    structure_only = module(
        structure_ids=mx.array([[-2, 9]], dtype=mx.int64),
        dep_levels=None,
        target_dtype=mx.float32,
    )
    both = module(
        structure_ids=mx.array([[-2, 9]], dtype=mx.int64),
        dep_levels=mx.array([[0, 99]], dtype=mx.int64),
        target_dtype=mx.float32,
    )

    assert np.allclose(to_numpy(structure_only)[0, :, 0], [0.5, 1.5])
    assert np.allclose(to_numpy(both)[0, :, 0], [5.5, 21.5])


def test_structure_embedding_validates_matching_shapes_for_present_components():
    module = CppMegaStructureEmbedding(hidden_size=16, active_components="all", bottleneck_dim=4)
    structure_ids = mx.ones((2, 3), dtype=mx.int64)
    dep_levels = mx.ones((2, 4), dtype=mx.int64)

    with pytest.raises(ValueError, match="does not match"):
        module(structure_ids=structure_ids, dep_levels=dep_levels)


def test_structure_embedding_returns_scalar_zero_when_no_active_inputs_are_present():
    module = CppMegaStructureEmbedding(hidden_size=16, active_components="core", bottleneck_dim=4)

    out = module(structure_ids=None, dep_levels=None, target_dtype=mx.float16)

    assert out.shape == ()
    assert out.dtype == mx.float16
    assert float(out.item()) == 0.0


def test_structure_embedding_gradients_reach_tables_after_projection_is_enabled():
    module = CppMegaStructureEmbedding(hidden_size=16, active_components="core", bottleneck_dim=4)
    module.up_proj.weight = mx.ones_like(module.up_proj.weight)
    structure_ids = mx.array([[1, 2, 3]], dtype=mx.int64)
    dep_levels = mx.array([[0, 1, 2]], dtype=mx.int64)

    def loss_fn():
        return mx.sum(module(structure_ids=structure_ids, dep_levels=dep_levels, target_dtype=mx.float32))

    loss, grads = nn.value_and_grad(module, loss_fn)()
    mx.eval(loss, grads)

    assert "stacked_emb" in grads
    grad = grads["stacked_emb"]["weight"]
    assert grad.shape == module.stacked_emb.weight.shape
    assert bool(mx.any(mx.isfinite(grad)).item())
    assert float(mx.sum(mx.abs(grad)).item()) > 0.0

