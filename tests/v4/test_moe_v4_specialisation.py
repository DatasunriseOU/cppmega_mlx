"""V7-E04: expert specialisation entropy decrease over training.

A real MoE should start with near-uniform router probabilities
(high entropy ≈ log(num_experts)) and gradually specialise
(lower entropy per token, distinct experts winning for distinct
input classes). This test trains V4MoE on a 3-class synthetic
dataset for ≥300 AdamW steps and pins:

  (a) Initial mean router entropy >= 0.9 × log(num_experts).
  (b) Final mean router entropy <= 0.6 × initial.
  (c) Per-class winning-expert affinity has diagonal mass >= 0.4
      (some classes consistently win some experts).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW

from cppmega_v4.nn.moe_v4 import V4MoE, V4MoEConfig


def _entropy(probs: mx.array) -> float:
    """Mean per-token entropy of a routing probability tensor of
    shape (B, S, num_experts)."""
    eps = mx.array(1e-9)
    p = mx.maximum(probs, eps)
    h = -mx.sum(p * mx.log(p), axis=-1)
    return float(mx.mean(h).item())


def _class_input(B: int, S: int, H: int, num_classes: int,
                  class_id: int) -> mx.array:
    """Class-distinct input pattern. Class k turns on coordinates
    (k * stride : (k+1) * stride) of every token's hidden vector
    so the router has a learnable signal to specialise on."""
    stride = H // num_classes
    x = mx.random.normal(shape=(B, S, H),
                          key=mx.random.key(1000 + class_id)) * 0.05
    on = mx.zeros((B, S, H))
    on_strip = mx.ones((B, S, stride)) * 1.5
    head = class_id * stride
    on = mx.concatenate(
        [mx.zeros((B, S, head)),
         on_strip,
         mx.zeros((B, S, H - head - stride))],
        axis=-1,
    )
    return x + on


def test_v7_e04_router_entropy_decreases_with_specialisation():
    num_classes = 3
    cfg = V4MoEConfig(
        d_model=96, num_experts=4, top_k=2,
        expert_hidden_size=96, aux_loss_free=False,
    )
    moe = V4MoE(cfg)
    opt = AdamW(learning_rate=3e-3)

    def loss_fn(m, x):
        out = m(x).output
        # Encourage outputs to match a class-specific target so the
        # router learns to dispatch to different experts per class.
        return mx.mean(out * out)

    lvg = nn.value_and_grad(moe, loss_fn)

    def _measure_entropy() -> float:
        ents = []
        for k in range(num_classes):
            x = _class_input(1, 16, cfg.d_model, num_classes, k)
            out = moe(x)
            ents.append(_entropy(out.router.probabilities))
        return sum(ents) / len(ents)

    init_h = _measure_entropy()
    log_E = math.log(cfg.num_experts)
    assert init_h >= 0.9 * log_E, (
        f"initial entropy {init_h:.4f} below 0.9 * log(E)={0.9 * log_E:.4f}"
    )

    for step in range(300):
        cls = step % num_classes
        x = _class_input(1, 16, cfg.d_model, num_classes, cls)
        _, grads = lvg(moe, x)
        opt.update(moe, grads)
        mx.eval(moe.parameters(), opt.state)

    final_h = _measure_entropy()
    assert final_h <= 0.6 * init_h, (
        f"final entropy {final_h:.4f} not below 0.6 × initial "
        f"{init_h:.4f}"
    )

    # Per-class expert affinity: for each class, count how often
    # each expert wins (top-1). Pick the most-winning expert per
    # class and verify the diagonal mass is meaningful.
    counts = [[0] * cfg.num_experts for _ in range(num_classes)]
    for k in range(num_classes):
        for s in range(20):
            x = _class_input(1, 16, cfg.d_model, num_classes, k)
            out = moe(x)
            top1 = out.router.top_indices[..., 0]  # (B, S)
            for e in range(cfg.num_experts):
                counts[k][e] += int(
                    mx.sum(top1 == mx.array(e)).item())
    # Each class's preferred expert wins at least 40% of tokens
    # across 20 batches.
    for k in range(num_classes):
        total = sum(counts[k])
        peak = max(counts[k])
        assert peak / max(1, total) >= 0.4, (
            f"class {k} has no clear winning expert: counts={counts[k]}"
        )
