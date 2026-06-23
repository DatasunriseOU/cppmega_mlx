"""Fast pytest for the Stage-1 end-to-end mixed-objective training smoke.

Asserts, on the tiny ``golden_mini`` fixture with a tiny model profile:

  * mixed-objective training runs end-to-end and the loss decreases;
  * BOTH side-channel-ON steps (causal_lm / *_recovery, original token order) and
    side-channel-OFF steps (ast_fim / commit_diff / pre_to_post, reordered or
    synthesized tokens) actually occur in the drawn mix;
  * the side-channel alignment rule is enforced: a MISALIGNED channel pass RAISES
    (both at the script's ``_assert_aligned`` guard and at the model's own
    ``_check_side_channel`` guard) — never silently accepted.
"""

from __future__ import annotations

import random

import mlx.core as mx
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.fim import FIMSpecialTokenIds
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM
from cppmega_mlx.training.objectives import build_causal_lm
from cppmega_mlx.training.task_mixer import TaskKind
from scripts.train_stage1 import (
    _ALIGNED_OBJECTIVES,
    _REORDERED_OBJECTIVES,
    GOLDEN_MINI,
    _assert_aligned,
    build_train_step,
    load_code_packets,
    smoke_config,
    run_training,
)

VOCAB = 65536


@pytest.fixture(scope="module")
def smoke_results() -> dict:
    """Run the full Stage-1 smoke once (few-dozen-plus steps, tiny dims)."""

    return run_training(num_steps=140, seed=1234, verbose=False)


def test_training_runs_end_to_end_and_loss_decreases(smoke_results: dict) -> None:
    assert smoke_results["num_steps"] == 140
    # The mixed-objective model learns: final overall loss strictly below initial.
    assert smoke_results["final_loss"] < smoke_results["initial_loss"], smoke_results
    # A meaningful drop, not numerical noise.
    assert smoke_results["final_loss"] < 0.7 * smoke_results["initial_loss"]


def test_both_channel_on_and_channel_off_steps_occur(smoke_results: dict) -> None:
    dist = smoke_results["distribution"]
    # Side-channel-ON family (original token order: causal / recovery).
    assert smoke_results["aligned_steps"] > 0
    assert any(obj in _ALIGNED_OBJECTIVES for obj in dist), dist
    assert "causal_lm" in dist
    # Side-channel-OFF family (reordered / synthesized: fim / commit).
    assert smoke_results["reordered_steps"] > 0
    assert any(obj in _REORDERED_OBJECTIVES for obj in dist), dist
    # At least one FIM-permuted AND one commit-synthesized objective fired.
    assert "ast_fim" in dist, dist
    assert ("commit_diff" in dist) or ("pre_to_post" in dist), dist


def test_aligned_step_passes_channels_reordered_step_omits_them() -> None:
    packets = load_code_packets(
        sorted((GOLDEN_MINI / "code").glob("*.parquet")), vocab_size=VOCAB
    )
    packet = packets[0]

    # Aligned objective (causal_lm) -> channels ARE passed and length-matched.
    example = build_causal_lm(packet)
    aligned = build_train_step("causal_lm", example, packet)
    assert aligned.side_channels, "aligned objective must carry side-channels"
    s = int(aligned.input_ids.shape[1])
    for name, chan in aligned.side_channels.items():
        if name == "platform_ids":
            continue  # document-level (1, K), not token-aligned
        assert int(chan.shape[1]) == s, (name, chan.shape, s)

    # The aligned step runs through the real model with channels ON.
    model = DenseCppLM(smoke_config(VOCAB))
    _, loss = model(
        aligned.input_ids,
        targets=aligned.target_ids,
        loss_mask=aligned.loss_mask,
        **aligned.side_channels,
    )
    assert mx.isfinite(loss).item()


def test_misaligned_channel_pass_raises() -> None:
    """A deliberately misaligned structure channel must RAISE, not be accepted."""

    model = DenseCppLM(smoke_config(VOCAB))
    input_ids = mx.array([[1, 2, 3, 4, 5, 6]])
    good = mx.array([[0, 1, 0, 1, 0, 1]])
    bad = mx.array([[0, 1, 0, 1]])  # length 4 != 6 -> misaligned

    with pytest.raises(ValueError, match="must match input_ids"):
        model(
            input_ids,
            targets=input_ids,
            structure_ids=bad,
            dep_levels=good,
            ast_depth_ids=good,
            sibling_index_ids=good,
            node_type_ids=good,
        )


def test_assert_aligned_guard_raises_on_short_channel() -> None:
    """The script's own alignment guard rejects a too-short channel."""

    short = mx.array([0, 1, 0])  # length 3
    with pytest.raises(ValueError, match="cannot align side-channel"):
        _assert_aligned("structure_ids", short, seq_len=10)


def test_build_train_step_disables_channels_for_reordered_objective() -> None:
    """Reordered/synthesized objectives carry NO token-aligned side-channels."""

    packets = load_code_packets(
        sorted((GOLDEN_MINI / "code").glob("*.parquet")), vocab_size=VOCAB
    )
    packet = packets[0]
    from cppmega_mlx.training.objectives import build_ast_fim

    example = build_ast_fim(
        packet, rng=random.Random(0), special_token_ids=FIMSpecialTokenIds()
    )
    step = build_train_step("ast_fim", example, packet)
    assert step.objective == "ast_fim"
    assert step.side_channels == {}, "reordered objective must disable side-channels"
