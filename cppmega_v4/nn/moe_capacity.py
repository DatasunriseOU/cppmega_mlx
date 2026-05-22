"""V7-E02: capacity-bounded drop/reroute accounting for MoE routing.

V4MoE currently does unbounded top-k dispatch. With a real capacity
factor C ∈ (0, 1], each expert can absorb at most
  cap_per_expert = ceil(C * total_tokens * top_k / num_experts)
tokens. Tokens past that limit are either DROPPED (no contribution
from that slot — residual still flows) or REROUTED to the next-best
expert that still has capacity. The chosen policy must be observable
through extras.moe.{dropped_token_ratio, rerouted_token_ratio}.

This module ships the policy as a pure NumPy/python function so V4MoE
can adopt it incrementally without touching the autograd path.
"""

from __future__ import annotations

import math
from typing import Iterable


def compute_drop_reroute_stats(
    top_indices: list[list[int]],
    *,
    num_experts: int,
    capacity_factor: float,
    reroute: bool = True,
) -> dict[str, float]:
    """Apply capacity bound to a top-k dispatch matrix and report the
    fraction of dispatch slots that overflowed.

    Args:
        top_indices: list of per-token lists of expert indices, shape
            (n_tokens, top_k). Indices are 0..num_experts-1.
        num_experts: total expert count.
        capacity_factor: in (0, 1]; capacity per expert is
            ceil(capacity_factor * n_tokens * top_k / num_experts).
        reroute: when True, overflowed slots try the next-best expert
            (cyclic search) before being dropped; when False, they go
            straight to the dropped bucket.

    Returns:
        {dropped_token_ratio, rerouted_token_ratio, overflow_ratio,
         capacity_per_expert, total_slots}
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be > 0")
    if capacity_factor <= 0 or capacity_factor > 8:
        raise ValueError("capacity_factor out of sane range")
    if not top_indices:
        return {
            "dropped_token_ratio": 0.0,
            "rerouted_token_ratio": 0.0,
            "overflow_ratio": 0.0,
            "capacity_per_expert": 0,
            "total_slots": 0,
        }
    n_tokens = len(top_indices)
    top_k = len(top_indices[0])
    total_slots = n_tokens * top_k
    cap = max(1, math.ceil(
        capacity_factor * total_slots / num_experts))

    used = [0] * num_experts
    dropped = 0
    rerouted = 0
    for token_choices in top_indices:
        for orig in token_choices:
            if used[orig] < cap:
                used[orig] += 1
                continue
            # Overflow.
            placed = False
            if reroute:
                for offset in range(1, num_experts):
                    alt = (orig + offset) % num_experts
                    if used[alt] < cap:
                        used[alt] += 1
                        rerouted += 1
                        placed = True
                        break
            if not placed:
                dropped += 1
    return {
        "dropped_token_ratio": dropped / total_slots,
        "rerouted_token_ratio": rerouted / total_slots,
        "overflow_ratio": (dropped + rerouted) / total_slots,
        "capacity_per_expert": cap,
        "total_slots": total_slots,
    }


__all__ = ["compute_drop_reroute_stats"]
