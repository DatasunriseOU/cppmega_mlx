"""V7-B08: multi-node topology helper tests."""

from __future__ import annotations

import pytest

from cppmega_v4.runtime.multi_node_topology import (
    coord_to_rank, inter_node_ranks, intra_node_ranks, rank_to_coord,
    world_size,
)


def test_v7_b08_world_size_basic():
    assert world_size(2, 8) == 16
    assert world_size(4, 4) == 16


def test_v7_b08_rank_to_coord_round_trip():
    # 2 nodes × 4 gpus
    assert rank_to_coord(0, 2, 4) == (0, 0)
    assert rank_to_coord(3, 2, 4) == (0, 3)
    assert rank_to_coord(4, 2, 4) == (1, 0)
    assert rank_to_coord(7, 2, 4) == (1, 3)
    # Round-trip.
    for r in range(8):
        n, lr = rank_to_coord(r, 2, 4)
        assert coord_to_rank(n, lr, 4) == r


def test_v7_b08_intra_node_ranks_are_consecutive():
    assert intra_node_ranks(0, 4) == [0, 1, 2, 3]
    assert intra_node_ranks(1, 4) == [4, 5, 6, 7]


def test_v7_b08_inter_node_ranks_same_local_across_nodes():
    # local_rank=2 across 4 nodes × 4 gpus/node.
    expected = [2, 6, 10, 14]
    assert inter_node_ranks(2, 4, 4) == expected


def test_v7_b08_validation_errors():
    with pytest.raises(ValueError):
        world_size(0, 4)
    with pytest.raises(ValueError):
        rank_to_coord(99, 2, 4)
    with pytest.raises(ValueError):
        coord_to_rank(0, 99, 4)
    with pytest.raises(ValueError):
        inter_node_ranks(99, 2, 4)
