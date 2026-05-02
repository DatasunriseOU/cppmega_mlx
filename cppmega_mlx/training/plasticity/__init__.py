"""Plasticity Toolkit (FIRE + DASH + ReDo) port from nanochat.

Source contract: ``nanochat/nanochat/fire.py``. Reference: Han et al., "FIRE:
Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity
Tradeoff", arXiv:2602.08040v1, Feb 2026.

Three independent interventions, each addressing a different failure mode of
long-horizon training:

* FIRE — one-shot Newton-Schulz orthogonalization of 2D weight matrices at
  a phase transition (default ``--fire_at_step=5000`` in nanochat). Reverts
  the spectrum to that of an isometric init while preserving the original
  Frobenius norm. Selectively wipes Adam moments only for FIRE'd params.
* DASH — per-neuron weight shrinking based on cos-sim with the gradient.
  Applied periodically during training (every ``--dash_every`` steps).
  Skips Muon-managed params (Muon already strips the parallel component).
* ReDo — dormant neuron detection and recycling for MLP layers. Periodic
  (every ``--redo_every=1000`` steps). Tracks per-neuron post-activation
  EMA; reinits below ``--redo_dormant_threshold=0.025`` of layer mean.
"""

from cppmega_mlx.training.plasticity.dash import dash_step, dash_step_tree
from cppmega_mlx.training.plasticity.fire import (
    apply_fire,
    newton_schulz,
    reset_optimizer_states_for_fired_keys,
)
from cppmega_mlx.training.plasticity.redo import (
    ReDoDiagnostics,
    recycle_dormant_neurons,
)

__all__ = [
    "ReDoDiagnostics",
    "apply_fire",
    "dash_step",
    "dash_step_tree",
    "newton_schulz",
    "recycle_dormant_neurons",
    "reset_optimizer_states_for_fired_keys",
]
