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


# V7-B-real: process group construction on top of mx.distributed.
# When the launcher has spawned world_size==num_nodes*gpus_per_node
# ranks, build_process_groups() returns nested Groups that match the
# coord helpers above. Single-process fallback returns None Groups so
# higher-level code can branch on group.world_size().


def build_process_groups(num_nodes: int, gpus_per_node: int
                          ) -> dict:
    """Return {'intra_node': Group|None, 'inter_node': Group|None,
    'global': Group|None, 'world_info': WorldInfo}.

    intra_node groups all ranks on the same node together; inter_node
    groups ranks with the same local_rank across nodes. Used by FSDP
    (intra-node) + TP (intra-node) + DP (inter-node) wiring.

    When mx.distributed.is_available() returns False or the actual
    world_size does not match num_nodes * gpus_per_node, every group
    falls back to None and callers must use the single-process path.
    """
    from cppmega_v4.runtime import distributed as _d

    info = _d.init()
    expected = world_size(num_nodes, gpus_per_node)
    if not info.real or info.world_size != expected:
        return {
            "intra_node": None, "inter_node": None, "global": None,
            "world_info": info, "expected_world_size": expected,
            "ok": False,
        }
    import mlx.core as mx
    node_id, local_rank = rank_to_coord(
        info.rank, num_nodes, gpus_per_node)
    global_group = mx.distributed.init()
    # mlx.distributed.Group has split(color, key) on recent builds;
    # use try/except so older mlx versions still load the module.
    intra: object | None = None
    inter: object | None = None
    try:
        intra = global_group.split(color=node_id, key=local_rank)
    except Exception:
        intra = None
    try:
        inter = global_group.split(color=local_rank, key=node_id)
    except Exception:
        inter = None
    return {
        "intra_node": intra,
        "inter_node": inter,
        "global": global_group,
        "world_info": info,
        "expected_world_size": expected,
        "ok": True,
    }


__all__ = [
    "world_size", "rank_to_coord", "coord_to_rank",
    "intra_node_ranks", "inter_node_ranks",
    "build_process_groups",
]
