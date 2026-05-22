"""V7-F03: sampling strategies for generation.

Pure-Python on lists/arrays of logits (no mx dependency in the
helpers — the gen.run RPC wraps these with mlx-to-list conversion).

  * greedy(logits)
  * temperature_sample(logits, temperature, *, rng)
  * top_k_sample(logits, k, *, rng)
  * top_p_sample(logits, p, *, rng)

For beam search a separate `beam_step` returns the top-N (token,
score) pairs so the caller can manage beam state across steps.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def greedy(logits: Sequence[float]) -> int:
    return max(range(len(logits)), key=lambda i: logits[i])


def _softmax(logits: Sequence[float]) -> list[float]:
    mx = max(logits)
    exps = [math.exp(l - mx) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def temperature_sample(logits: Sequence[float], *,
                        temperature: float,
                        rng: random.Random) -> int:
    if temperature <= 0:
        return greedy(logits)
    scaled = [l / temperature for l in logits]
    probs = _softmax(scaled)
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def top_k_sample(logits: Sequence[float], *, k: int,
                  rng: random.Random,
                  temperature: float = 1.0) -> int:
    if k <= 0 or k >= len(logits):
        return temperature_sample(logits, temperature=temperature, rng=rng)
    top_idx = sorted(range(len(logits)),
                     key=lambda i: logits[i],
                     reverse=True)[:k]
    masked = [-math.inf] * len(logits)
    for i in top_idx:
        masked[i] = logits[i]
    return temperature_sample(masked, temperature=temperature, rng=rng)


def top_p_sample(logits: Sequence[float], *, p: float,
                  rng: random.Random,
                  temperature: float = 1.0) -> int:
    if p >= 1.0:
        return temperature_sample(logits, temperature=temperature, rng=rng)
    scaled = [l / max(temperature, 1e-9) for l in logits]
    probs = _softmax(scaled)
    sorted_idx = sorted(range(len(probs)),
                        key=lambda i: probs[i], reverse=True)
    cumulative = 0.0
    keep: set[int] = set()
    for i in sorted_idx:
        cumulative += probs[i]
        keep.add(i)
        if cumulative >= p:
            break
    masked = [-math.inf] * len(logits)
    for i in keep:
        masked[i] = logits[i]
    return temperature_sample(masked, temperature=temperature, rng=rng)


def beam_step(logits: Sequence[float], *, beam_width: int
              ) -> list[tuple[int, float]]:
    """Return top-N (token_id, log_prob) pairs for beam search.

    beam_step is stateless; the caller maintains active beams,
    extends each by these candidates, ranks by cumulative log-prob,
    and keeps the top beam_width overall.
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be > 0")
    probs = _softmax(logits)
    log_probs = [math.log(max(p, 1e-30)) for p in probs]
    ranked = sorted(range(len(logits)),
                    key=lambda i: log_probs[i], reverse=True)
    return [(i, log_probs[i]) for i in ranked[:beam_width]]


__all__ = [
    "greedy", "temperature_sample",
    "top_k_sample", "top_p_sample", "beam_step",
]
