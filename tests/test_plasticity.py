"""Tests for the FIRE + DASH + ReDo plasticity toolkit (port from nanochat)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cppmega_mlx.training.plasticity import (
    ReDoDiagnostics,
    apply_fire,
    dash_step,
    newton_schulz,
    recycle_dormant_neurons,
)


class _TwoLayerMLP(nn.Module):
    def __init__(self, d_in: int = 16, d_hidden: int = 64, d_out: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.embed_tokens = nn.Embedding(100, d_in)
        self.lm_head = nn.Linear(d_out, 100, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        z = self.fc1(x)
        a = nn.relu(z)
        if hasattr(self, "_redo_probe"):
            self._redo_probe(z)
        return self.lm_head(self.fc2(a))


def test_newton_schulz_preserves_frobenius_norm_and_orthogonalizes() -> None:
    rng = mx.random.normal((128, 64))
    mx.eval(rng)
    fro_before = float(mx.linalg.norm(rng).item())
    out = newton_schulz(rng)
    mx.eval(out)

    fro_after = float(mx.linalg.norm(out).item())
    assert abs(fro_after - fro_before) / fro_before < 1e-3, (
        f"Frobenius drift: {fro_before} -> {fro_after}"
    )

    inner = out.T @ out
    diag = mx.diagonal(inner)
    off = mx.abs(inner - mx.diag(diag))
    mx.eval(diag, off)
    diag_var = float(diag.std().item())
    diag_mean = float(diag.mean().item())
    off_max = float(off.max().item())
    assert diag_var / max(diag_mean, 1e-6) < 1e-3, "diagonal not uniform"
    assert off_max < 1e-3, f"residual off-diagonal: {off_max}"


def test_newton_schulz_handles_wide_matrices() -> None:
    wide = mx.random.normal((32, 96))
    mx.eval(wide)
    out = newton_schulz(wide)
    mx.eval(out)
    inner = out @ out.T  # rows orthogonal for wide input
    diag = mx.diagonal(inner)
    off = mx.abs(inner - mx.diag(diag))
    mx.eval(diag, off)
    assert float(off.max().item()) < 1e-3


def test_apply_fire_skips_embeddings_and_lm_head() -> None:
    model = _TwoLayerMLP()
    modified = apply_fire(model)
    assert "fc1.weight" in modified
    assert "fc2.weight" in modified
    assert "lm_head.weight" not in modified
    assert "embed_tokens.weight" not in modified


def test_dash_step_shrinks_only_high_cosine_rows() -> None:
    weight = mx.random.normal((4, 16))
    parallel_grad = weight  # cos_sim == 1 for every row
    out = dash_step(weight, parallel_grad, alpha=0.05, shrink_rate=0.5)
    mx.eval(out)
    ratios = mx.linalg.norm(out, axis=1) / mx.linalg.norm(weight, axis=1)
    mx.eval(ratios)
    for r in ratios.tolist():
        assert r < 0.6, f"parallel grad row not shrunk (ratio={r})"

    rng_grad = mx.random.normal(weight.shape)
    out_rng = dash_step(weight, rng_grad, alpha=0.05, shrink_rate=0.5)
    mx.eval(out_rng)
    ratios_rng = mx.linalg.norm(out_rng, axis=1) / mx.linalg.norm(weight, axis=1)
    mx.eval(ratios_rng)
    assert all(r > 0.5 for r in ratios_rng.tolist())


def test_dash_step_clamps_shrink_factor_to_half() -> None:
    weight = mx.random.normal((2, 8))
    parallel_grad = weight * 1.0
    out = dash_step(weight, parallel_grad, alpha=0.0, shrink_rate=10.0)
    mx.eval(out)
    ratios = mx.linalg.norm(out, axis=1) / mx.linalg.norm(weight, axis=1)
    mx.eval(ratios)
    for r in ratios.tolist():
        assert r >= 0.499, f"shrink dropped below 0.5 floor (r={r})"


def test_redo_diagnostics_detects_dormant_neurons() -> None:
    model = _TwoLayerMLP(d_in=8, d_hidden=16, d_out=8)
    # Kill neurons 5..15 by zeroing those rows of fc1
    keep_mask = mx.concatenate([mx.ones((5,)), mx.zeros((11,))])
    model.fc1.weight = model.fc1.weight * keep_mask[:, None]
    model.fc1.bias = model.fc1.bias * keep_mask
    mx.eval(model.fc1.weight, model.fc1.bias)

    diag = ReDoDiagnostics()
    diag.attach({"mlp": model})
    for _ in range(5):
        out = model(mx.random.normal((4, 8)))
        mx.eval(out)

    ratios = diag.get_dormant_ratio(tau=0.025)
    assert ratios["mlp"] > 0.5, f"dormant ratio too low: {ratios}"

    pre_fc1 = model.fc1.weight
    n = recycle_dormant_neurons(
        {"mlp": (model.fc1, model.fc2)},
        diag.get_stats(),
        tau=0.025,
    )
    mx.eval(model.fc1.weight, model.fc2.weight)
    assert n == 11, f"expected 11 dormant neurons recycled, got {n}"
    diff = mx.abs(model.fc1.weight - pre_fc1).sum()
    mx.eval(diff)
    assert float(diff.item()) > 0, "fc1 weights unchanged after recycle"


def test_redo_recycle_no_dormant_is_noop() -> None:
    model = _TwoLayerMLP(d_in=8, d_hidden=16, d_out=8)
    diag = ReDoDiagnostics()
    diag.attach({"mlp": model})
    for _ in range(5):
        out = model(mx.random.normal((4, 8)))
        mx.eval(out)

    pre = mx.array(model.fc1.weight)
    n = recycle_dormant_neurons(
        {"mlp": (model.fc1, model.fc2)},
        diag.get_stats(),
        tau=1e-6,
    )
    mx.eval(model.fc1.weight)
    assert n == 0
    diff = float(mx.abs(model.fc1.weight - pre).sum().item())
    assert diff == 0.0, "weights drifted on no-op recycle"
