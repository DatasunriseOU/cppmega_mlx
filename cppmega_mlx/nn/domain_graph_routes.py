"""Reference graph-route prior for DomainPacket edge triples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from cppmega_mlx.data.domain_packet import DomainPacket
from cppmega_mlx.data.domain_schema import DomainEdgeKind


@dataclass(frozen=True)
class DomainGraphRouteConfig:
    num_blocks: int = 64
    normalize: str = "binary"
    edge_weights: Mapping[int, float] = field(default_factory=dict)
    default_weight: float = 1.0
    max_candidates_per_query: int | None = None

    def __post_init__(self) -> None:
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >=1, got {self.num_blocks}")
        if self.normalize not in {"binary", "count"}:
            raise ValueError("normalize must be 'binary' or 'count'")
        if self.max_candidates_per_query is not None and self.max_candidates_per_query < 1:
            raise ValueError("max_candidates_per_query must be >=1 when set")


def build_domain_attention_bias(
    packet: DomainPacket,
    *,
    config: DomainGraphRouteConfig | None = None,
) -> mx.array:
    """Build a ``(num_blocks, num_blocks)`` additive prior from domain edges."""

    cfg = config or DomainGraphRouteConfig()
    prior = np.zeros((cfg.num_blocks, cfg.num_blocks), dtype=np.float32)
    if packet.graph_edges.num_edges == 0:
        return mx.array(prior)

    num_tokens = packet.token_axis_len
    block_size = max(1, (num_tokens + cfg.num_blocks - 1) // cfg.num_blocks)
    src = np.asarray(packet.graph_edges.src)
    dst = np.asarray(packet.graph_edges.dst)
    kinds = np.asarray(packet.graph_edges.kind)
    for src_i, dst_i, kind_i in zip(src, dst, kinds):
        if src_i < 0 or dst_i < 0 or src_i >= num_tokens or dst_i >= num_tokens:
            raise ValueError(f"domain edge endpoint out of range: {src_i}->{dst_i}")
        qb = min(cfg.num_blocks - 1, int(src_i) // block_size)
        kb = min(cfg.num_blocks - 1, int(dst_i) // block_size)
        weight = float(cfg.edge_weights.get(int(kind_i), cfg.default_weight))
        if cfg.normalize == "binary":
            prior[qb, kb] = max(prior[qb, kb], weight)
        else:
            prior[qb, kb] += weight
    return mx.array(prior)


def build_domain_block_candidates(
    packet: DomainPacket,
    *,
    config: DomainGraphRouteConfig | None = None,
) -> list[list[int]]:
    cfg = config or DomainGraphRouteConfig()
    bias = np.asarray(build_domain_attention_bias(packet, config=cfg))
    rows: list[list[int]] = []
    for row in range(bias.shape[0]):
        candidates = [
            (float(bias[row, idx]), int(idx))
            for idx in np.nonzero(bias[row])[0]
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if cfg.max_candidates_per_query is not None:
            candidates = candidates[: cfg.max_candidates_per_query]
        rows.append([idx for _score, idx in candidates])
    return rows


DEFAULT_EDGE_WEIGHTS: dict[int, float] = {
    int(DomainEdgeKind.BUILD_TARGET_SOURCE): 1.0,
    int(DomainEdgeKind.BUILD_TARGET_DEP): 1.0,
    int(DomainEdgeKind.SHELL_PIPE): 1.0,
    int(DomainEdgeKind.DIAG_PRIMARY_LOCATION): 2.0,
    int(DomainEdgeKind.DIAG_FIXIT): 2.0,
    int(DomainEdgeKind.LINK_UNDEFINED_SYMBOL): 2.0,
}


__all__ = [
    "DEFAULT_EDGE_WEIGHTS",
    "DomainGraphRouteConfig",
    "build_domain_attention_bias",
    "build_domain_block_candidates",
]
