"""V7-B08: multi-node topology helper — node-grid + per-rank coords.

Pure helper: given (num_nodes, gpus_per_node) and a global rank,
returns the (node_id, local_rank) coordinate plus intra-node and
inter-node neighbour groups. Used by the FSDP/TP/PP runtime
(V7-B01) to set up nested process groups; here it's just the math.
"""

from __future__ import annotations


def world_size(num_nodes: int, gpus_per_node: int) -> int:
    if num_nodes < 1 or gpus_per_node < 1:
        raise ValueError("num_nodes and gpus_per_node must be >= 1")
    return num_nodes * gpus_per_node


def rank_to_coord(global_rank: int, num_nodes: int,
                   gpus_per_node: int) -> tuple[int, int]:
    """Return (node_id, local_rank)."""
    W = world_size(num_nodes, gpus_per_node)
    if not (0 <= global_rank < W):
        raise ValueError(f"global_rank {global_rank} out of [0, {W})")
    return (global_rank // gpus_per_node,
            global_rank % gpus_per_node)


def coord_to_rank(node_id: int, local_rank: int,
                   gpus_per_node: int) -> int:
    if node_id < 0 or local_rank < 0 or local_rank >= gpus_per_node:
        raise ValueError("invalid coord")
    return node_id * gpus_per_node + local_rank


def intra_node_ranks(node_id: int, gpus_per_node: int) -> list[int]:
    """Ranks that share `node_id`."""
    return [coord_to_rank(node_id, r, gpus_per_node)
            for r in range(gpus_per_node)]


def inter_node_ranks(local_rank: int, num_nodes: int,
                      gpus_per_node: int) -> list[int]:
    """All ranks with the same local_rank across nodes (used for
    hierarchical reductions)."""
    if not (0 <= local_rank < gpus_per_node):
        raise ValueError("invalid local_rank")
    return [coord_to_rank(n, local_rank, gpus_per_node)
            for n in range(num_nodes)]


__all__ = [
    "world_size", "rank_to_coord", "coord_to_rank",
    "intra_node_ranks", "inter_node_ranks",
]
