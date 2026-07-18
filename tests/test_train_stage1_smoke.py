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
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cppmega_mlx.data.fim import FIMSpecialTokenIds
from cppmega_mlx.data.ast_fim import domain_preserving_document_spans
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.objectives import build_causal_lm
from scripts.train_stage1 import (
    _ALIGNED_OBJECTIVES,
    _REORDERED_OBJECTIVES,
    GOLDEN_MINI,
    _assert_aligned,
    _loss_for_step,
    build_train_step,
    load_code_packets,
    load_commit_packets,
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
    # This legacy fixture smoke intentionally has no typed commit/IFIM columns;
    # production coverage lives in test_production_objective_mixer.py.
    assert "fim" in dist, dist
    assert "ast_fim" in dist, dist
    assert "ifim" not in dist
    assert "commit_diff" not in dist
    assert dist == {"causal_lm": 112, "fim": 14, "ast_fim": 14}


def test_commit_loader_requires_and_preserves_typed_upstream_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "typed_commits.parquet"
    pq.write_table(
        pa.table(
            {
                "pre_token_ids": [[10, 11]],
                "post_token_ids": [[20, 21]],
                "diff_token_ids": [[30, 31, 32]],
                "commit_msg_token_ids": [[40, 41]],
                "repo": ["repo"],
                "filepath": ["src/demo.cc"],
                "commit_hash": ["abc123"],
            }
        ),
        path,
    )

    packets = load_commit_packets(path, vocab_size=VOCAB)

    assert len(packets) == 1
    assert np.asarray(packets[0].pre_token_ids).tolist() == [10, 11]
    assert np.asarray(packets[0].post_token_ids).tolist() == [20, 21]
    assert np.asarray(packets[0].diff_token_ids).tolist() == [30, 31, 32]
    assert np.asarray(packets[0].commit_msg).tolist() == [40, 41]


def test_code_loader_preserves_packed_document_ids_for_domain_fim(
    tmp_path: Path,
) -> None:
    path = tmp_path / "packed_code.parquet"
    token_ids = [2, 191, 41, 192, 2, 191, 42, 192]
    row = {
        "input_ids": [token_ids],
        "target_ids": [token_ids],
        "loss_mask": [[1] * len(token_ids)],
        "doc_ids": [[1, 1, 1, 1, 2, 2, 2, 2]],
        "valid_token_count": [len(token_ids)],
        "token_structure_ids": [[0] * len(token_ids)],
        "token_dep_levels": [[0] * len(token_ids)],
        "token_ast_depth": [[0] * len(token_ids)],
        "token_sibling_index": [[0] * len(token_ids)],
        "token_ast_node_type": [[0] * len(token_ids)],
        "token_symbol_ids": [[0] * len(token_ids)],
        "token_call_targets": [[0] * len(token_ids)],
        "token_type_refs": [[0] * len(token_ids)],
        "token_def_use": [[0] * len(token_ids)],
        "token_chunk_starts": [[1, 5]],
        "token_chunk_ends": [[3, 7]],
        "token_chunk_kinds": [[1, 1]],
        "token_chunk_dep_levels": [[0, 0]],
    }
    pq.write_table(pa.table(row), path)

    packet = load_code_packets([path], vocab_size=VOCAB)[0]
    assert np.asarray(packet.document_ids).tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert len(domain_preserving_document_spans(packet)) == 2


def test_commit_loader_accepts_typed_golden_fixture() -> None:
    packets = load_commit_packets(
        GOLDEN_MINI / "commits" / "commits.parquet", vocab_size=VOCAB
    )
    # ``--format both`` emits two full pre/post rows and two valid diff-only
    # rows.  The loader preserves both shapes; objective eligibility decides
    # which task can consume each packet.
    assert len(packets) == 4
    assert all(packet.commit_msg is not None for packet in packets)
    assert all(int(packet.commit_msg.shape[0]) >= 2 for packet in packets)
    assert sum(packet.post_token_ids is None for packet in packets) == 2


def test_commit_loader_preserves_partial_sections_without_synthesizing_them(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial_commits.parquet"
    pq.write_table(
        pa.table(
            {
                "pre_token_ids": [[10, 11], []],
                "post_token_ids": [[], []],
                "diff_token_ids": [[30, 31], [32]],
                "commit_msg_token_ids": [[40], []],
            }
        ),
        path,
    )

    packets = load_commit_packets(path, vocab_size=VOCAB)

    assert len(packets) == 2
    assert packets[0].post_token_ids is None
    assert np.asarray(packets[0].commit_msg).tolist() == [40]
    assert packets[1].pre_token_ids is None
    assert packets[1].commit_msg is None
    assert np.asarray(packets[1].diff_token_ids).tolist() == [32]


def test_aligned_step_passes_channels_reordered_step_omits_them() -> None:
    packets = load_code_packets(
        sorted((GOLDEN_MINI / "code").glob("*.parquet")), vocab_size=VOCAB
    )
    packet = packets[0]

    # The packed row carries domain routes and graph edges; the loader must
    # preserve both typed contracts instead of leaving them in opaque metadata.
    assert packet.domain_ids is not None
    assert packet.role_ids is not None
    assert packet.confidence_ids is not None
    assert packet.graph_batch().graphs[0].relations

    # Aligned objective (causal_lm) -> channels ARE passed and length-matched.
    example = build_causal_lm(packet)
    aligned = build_train_step("causal_lm", example, packet)
    assert aligned.side_channels, "aligned objective must carry side-channels"
    s = int(aligned.input_ids.shape[1])
    for name, chan in aligned.side_channels.items():
        if name == "platform_ids":
            continue  # document-level (1, K), not token-aligned
        assert int(chan.shape[1]) == s, (name, chan.shape, s)
    assert {"domain_ids", "role_ids", "confidence_ids"} <= set(
        aligned.side_channels
    )
    assert aligned.graph_batch is not None

    # The aligned step runs through the real model with channels ON.
    model = DenseCppLM(smoke_config(VOCAB))
    _, loss = model(
        aligned.input_ids,
        targets=aligned.target_ids,
        loss_mask=aligned.loss_mask,
        **aligned.side_channels,
    )
    assert mx.isfinite(loss).item()


def test_aligned_step_graph_routes_reach_graph_enabled_model() -> None:
    packet = load_code_packets(
        sorted((GOLDEN_MINI / "code").glob("*.parquet")), vocab_size=VOCAB
    )[0]
    step = build_train_step("causal_lm", build_causal_lm(packet), packet)
    model = DenseCppLM(
        DenseCppLMConfig(
            vocab_size=VOCAB,
            hidden_size=16,
            depth=1,
            ffn_hidden_size=32,
            max_seq_length=4096,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=8,
            graph_routes_enabled=True,
            graph_attention_bias_beta=10.0,
            ngram_hash_enabled=False,
            domain_residual_scale=1.0,
        )
    )

    loss = _loss_for_step(model, step, channels_on=True)
    mx.eval(loss)
    assert np.isfinite(float(loss.item()))


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
    assert step.graph_batch is None
    assert step.metadata_only["reason"] == (
        "objective_changes_token_order_or_synthesizes_tokens"
    )
    assert "domain_ids" in step.metadata_only["source_fields"]
    assert "type_edges" in step.metadata_only["source_fields"]
