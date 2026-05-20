"""GalCov Stage B tests — 3 new bricks (mlstm / abs_pos_embed / per_layer_embed)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from cppmega_v4.fusion.compatibility import _CATEGORY_BY_KIND
from cppmega_v4.models.unified_superblock_v4 import BLOCK_BUILDERS
from cppmega_v4.nn.abs_pos_embed import AbsPosEmbedBlock, AbsPosEmbedConfig
from cppmega_v4.nn.mlstm import MLSTMBlock, MLSTMConfig
from cppmega_v4.nn.per_layer_embed import PerLayerEmbedBlock, PerLayerEmbedConfig
from cppmega_v4.spec import contract_for


_NEW_KINDS = ["mlstm", "abs_pos_embed", "per_layer_embed"]


@pytest.mark.parametrize("kind", _NEW_KINDS)
def test_brick_registered_in_block_builders(kind):
    assert kind in BLOCK_BUILDERS


@pytest.mark.parametrize("kind,expected_category", [
    ("mlstm", "nonlinear_rnn"),
    ("abs_pos_embed", "norm_or_proj"),
    ("per_layer_embed", "norm_or_proj"),
])
def test_brick_categorised_in_compatibility(kind, expected_category):
    assert _CATEGORY_BY_KIND[kind] == expected_category


@pytest.mark.parametrize("kind", _NEW_KINDS)
def test_brick_has_shape_contract(kind):
    c = contract_for(kind)
    assert "x" in c.inputs
    assert "y" in c.outputs


@pytest.mark.parametrize("kind", _NEW_KINDS)
def test_brick_builder_instantiates(kind):
    mod = BLOCK_BUILDERS[kind](64, {})
    assert mod is not None


def test_mlstm_forward_preserves_shape():
    block = MLSTMBlock(MLSTMConfig(hidden_size=32, head_dim=8))
    x = mx.random.normal((1, 4, 32))
    y = block(x)
    assert y.shape == (1, 4, 32)


def test_abs_pos_embed_forward_preserves_shape():
    block = AbsPosEmbedBlock(AbsPosEmbedConfig(hidden_size=32, max_position_embeddings=16))
    x = mx.random.normal((2, 8, 32))
    y = block(x)
    assert y.shape == (2, 8, 32)


def test_abs_pos_embed_rejects_oversized_seq():
    block = AbsPosEmbedBlock(AbsPosEmbedConfig(hidden_size=32, max_position_embeddings=4))
    x = mx.random.normal((1, 8, 32))
    with pytest.raises(ValueError, match="exceeds"):
        block(x)


def test_per_layer_embed_forward_preserves_shape_and_init_identity():
    block = PerLayerEmbedBlock(PerLayerEmbedConfig(hidden_size=16))
    x = mx.random.normal((1, 4, 16))
    y = block(x)
    assert y.shape == (1, 4, 16)
    # Identity at init: scale=1, bias=0 → y == x
    assert mx.allclose(y, x).item()


@pytest.mark.parametrize("kind", _NEW_KINDS)
def test_brick_kind_picked_up_in_extended_registry(kind):
    """build_v4_extended_registry must auto-include new BLOCK_BUILDERS kinds."""
    from cppmega_v4.fusion import build_v4_extended_registry
    reg = build_v4_extended_registry()
    assert reg.descriptor_for(kind) is not None
