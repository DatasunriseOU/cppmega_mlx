"""V7-F03: greedy/temperature/top_k/top_p/beam samplers."""

from __future__ import annotations

import random

import pytest

from cppmega_v4.runtime.samplers import (
    beam_step, greedy, temperature_sample, top_k_sample, top_p_sample,
)


def test_v7_f03_greedy_picks_argmax():
    assert greedy([0.1, 0.9, 0.5]) == 1


def test_v7_f03_temperature_zero_equals_greedy():
    rng = random.Random(0)
    assert temperature_sample(
        [0.1, 0.9, 0.5], temperature=0.0, rng=rng) == 1


def test_v7_f03_temperature_high_spreads_choices():
    rng = random.Random(0)
    picks = {temperature_sample([0.1, 0.2, 0.3], temperature=5.0,
                                  rng=random.Random(s))
             for s in range(20)}
    # With high temperature 20 different seeds should produce >1 distinct.
    assert len(picks) > 1


def test_v7_f03_top_k_truncates_below_top_k():
    rng = random.Random(0)
    # Strong winner — top-1 always returns it.
    out = {top_k_sample([0.1, 0.2, 5.0, 0.0],
                         k=1, rng=random.Random(s))
           for s in range(10)}
    assert out == {2}


def test_v7_f03_top_p_keeps_nucleus():
    rng = random.Random(0)
    # Probabilities ≈ [0.0001, 0.0003, 0.999...] — p=0.5 keeps only
    # the dominant token.
    out = {top_p_sample([0.0, 1.0, 10.0], p=0.5,
                         rng=random.Random(s))
           for s in range(20)}
    assert out == {2}


def test_v7_f03_beam_step_returns_top_n_by_logprob():
    pairs = beam_step([0.0, 1.0, 0.5, 0.2], beam_width=2)
    assert len(pairs) == 2
    # Argmax is index 1 (logit 1.0).
    assert pairs[0][0] == 1
    # Log-probs decreasing.
    assert pairs[0][1] >= pairs[1][1]


def test_v7_f03_beam_width_zero_rejected():
    with pytest.raises(ValueError):
        beam_step([0.1, 0.2], beam_width=0)


def test_v7_f03_top_k_zero_falls_back_to_temperature():
    """k=0 means no truncation → uses plain temperature_sample."""
    rng = random.Random(7)
    result = top_k_sample([0.1, 0.9, 0.5], k=0,
                           rng=rng, temperature=1.0)
    assert 0 <= result < 3
