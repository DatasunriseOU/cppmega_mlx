"""PR-2 numeric-equivalence check (FUSED-PIPELINE-ROADMAP §4/§5).

Verify that token-weighted grad accumulation over N microbatches produces
gradients and a loss numerically equal (within fp tolerance) to the
full-batch step. RULE #1: a mismatch here means the loop fix is wrong.

Run: python3 scratch/pr2_grad_accum_equivalence.py
"""

from __future__ import annotations

import copy
import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from cppmega_mlx.data.batch import synthetic_token_batch
from cppmega_mlx.models.hybrid_lm import HybridTinyConfig, HybridTinyLM
from cppmega_mlx.training.loss import next_token_cross_entropy
from cppmega_mlx.training.loop import one_step_train


def _build(grad_checkpoint: bool) -> HybridTinyLM:
    mx.random.seed(0)
    cfg = HybridTinyConfig(
        vocab_size=64,
        hidden_size=16,
        pattern="A",
        depth=2,
        num_attention_heads=4,
        max_seq_length=16,
        grad_checkpoint=grad_checkpoint,
    )
    model = HybridTinyLM(config=cfg)
    mx.eval(model.parameters())
    return model


def _grads_only(model: HybridTinyLM, batch) -> dict:
    loss_and_grad = nn.value_and_grad(model, next_token_cross_entropy)
    (loss, ntokens), grads = loss_and_grad(model, batch)
    mx.eval(grads, loss, ntokens)
    return loss, ntokens, grads


def _flat(grads) -> dict:
    return {k: v for k, v in tree_flatten(grads)}


def _max_rel_diff(a: dict, b: dict) -> float:
    worst = 0.0
    worst_name = ""
    for k in a:
        ga, gb = a[k], b[k]
        denom = mx.maximum(mx.abs(ga).max(), mx.array(1e-6))
        rel = float((mx.abs(ga - gb).max() / denom).item())
        if rel > worst:
            worst, worst_name = rel, k
    print(f"  worst grad rel-diff: {worst:.3e} at {worst_name}")
    return worst


def main() -> None:
    batch_size = 4
    batch = synthetic_token_batch(batch_size=batch_size, seq_length=16, vocab_size=64)

    # Reference full-batch grads (single forward/backward).
    ref_model = _build(grad_checkpoint=False)
    full_loss, full_ntok, full_grads = _grads_only(ref_model, batch)
    full_flat = _flat(full_grads)
    print(f"full-batch loss={float(full_loss.item()):.8f} ntokens={int(full_ntok.item())}")

    # Manual token-weighted accumulation matching one_step_train's math.
    acc_model = _build(grad_checkpoint=False)
    total_ntok = batch.target_mask.sum().astype(mx.float32)
    mx.eval(total_ntok)
    num_micro = 4
    base = batch_size // num_micro
    acc = None
    acc_loss = mx.array(0.0, dtype=mx.float32)
    from cppmega_mlx.training.loop import _slice_lm_batch
    loss_and_grad = nn.value_and_grad(acc_model, next_token_cross_entropy)
    for i in range(num_micro):
        micro = _slice_lm_batch(batch, i * base, (i + 1) * base)
        (ml, mn), mg = loss_and_grad(acc_model, micro)
        w = mn.astype(mx.float32) / total_ntok
        wg = {k: v * w for k, v in tree_flatten(mg)}
        if acc is None:
            acc = wg
        else:
            acc = {k: acc[k] + wg[k] for k in acc}
        acc_loss = acc_loss + ml.astype(mx.float32) * w
        mx.eval(acc, acc_loss)
    print(f"accum   loss={float(acc_loss.item()):.8f}")
    print(f"loss abs-diff: {abs(float(acc_loss.item()) - float(full_loss.item())):.3e}")
    worst = _max_rel_diff(full_flat, acc)

    tol = 2e-4
    assert abs(float(acc_loss.item()) - float(full_loss.item())) < tol, "loss mismatch"
    assert worst < tol, f"grad mismatch {worst} >= {tol}"

    # End-to-end through one_step_train (optimizer applied; compare param deltas).
    from cppmega_mlx.training.optimizers import make_adamw
    m_full = _build(grad_checkpoint=False)
    opt_full = make_adamw(learning_rate=1e-3)
    r_full = one_step_train(m_full, opt_full, batch, grad_accum_steps=1)

    m_acc = _build(grad_checkpoint=False)
    opt_acc = make_adamw(learning_rate=1e-3)
    r_acc = one_step_train(m_acc, opt_acc, batch, grad_accum_steps=4)
    print(f"one_step_train loss full={r_full.loss:.8f} accum={r_acc.loss:.8f} "
          f"diff={abs(r_full.loss - r_acc.loss):.3e}")
    pf = _flat(m_full.parameters())
    pa = _flat(m_acc.parameters())
    pworst = _max_rel_diff(pf, pa)
    assert abs(r_full.loss - r_acc.loss) < tol, "one_step_train loss mismatch"
    assert pworst < tol, f"post-step param mismatch {pworst}"
    assert r_full.ntokens == r_acc.ntokens, "ntokens mismatch"
    print("PASS: grad accumulation is numerically equivalent to full-batch step")


if __name__ == "__main__":
    main()
