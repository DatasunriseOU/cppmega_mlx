"""Regression test for CR-01: objective mixer matcher incompleteness.

The greedy bipartite matching in materialize_window_from_pool explores only a
narrow set of forced assignments when required_realized_assignment is set.  With
repeated quota slots the matcher always pairs a forced packet with the same
greedy-first-choice partner, so valid assignments (e.g. source indices 2, 3)
are never explored, causing a false ObjectiveQuotaUnsatisfiedError.

Reproduction: ppgp source-pool pattern, FIM objective, output_count=2,
seeds 6 and 71.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.domain_packet import DomainEdgeIndex
from cppmega_mlx.data.graph_packet import EdgeIndex
from cppmega_mlx.training.objective_mixer import (
    EligibilityAwareTaskMixer,
    ObjectiveQuotaUnsatisfiedError,
    ObjectiveSource,
)
from cppmega_mlx.training.objective_schedule import (
    assess_graph_positive_capability,
    source_has_graph_candidate,
)
from cppmega_mlx.training.task_mixer import TaskKind


def _arr(values: list[int]) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def _code_packet(*, domain_edge: tuple[int, int, int] | None = None) -> CodePacket:
    """Minimal FIM-eligible code packet with optional domain edge."""
    token_count = 8
    zeros = _arr([0] * token_count)
    domain_edges = DomainEdgeIndex.empty()
    if domain_edge is not None:
        domain_edges = DomainEdgeIndex.from_triples([domain_edge])
    return CodePacket(
        token_ids=_arr(list(range(100, 100 + token_count))),
        document_ids=_arr([1] * token_count),
        ifim_instruction_token_ids=_arr([1201, 1202]),
        structure_ids=_arr([1] * token_count),
        dep_levels=zeros,
        ast_depth=zeros,
        sibling_index=zeros,
        ast_node_type=_arr([1] * token_count),
        symbol_ids=zeros,
        call_targets=zeros,
        type_refs=zeros,
        def_use=zeros,
        domain_ids=zeros,
        role_ids=zeros,
        entity_ids=zeros,
        scope_ids=zeros,
        confidence_ids=_arr([1] * token_count),
        source_doc_ids=_arr([1] * token_count),
        source_identity_ids=_arr([1] * token_count),
        chunk_starts=_arr([0, 2, 5]),
        chunk_ends=_arr([2, 5, 8]),
        chunk_kinds=_arr([1, 1, 1]),
        chunk_dep_levels=_arr([0, 0, 0]),
        call_edges=EdgeIndex.from_pairs([], relation="call", num_nodes=3),
        type_edges=EdgeIndex.from_pairs([], relation="type", num_nodes=3),
        domain_edges=domain_edges,
        metadata={"platform_ids": [2]},
    )


def _ppgp_sources() -> list[ObjectiveSource]:
    """ppgp pattern: plain, plain, graph-positive, plain.

    All four sources are FIM-eligible code packets.  Source index 2 carries a
    domain edge making it graph-positive.
    """
    return [
        ObjectiveSource(code_packet=_code_packet()),                       # 0: plain
        ObjectiveSource(code_packet=_code_packet()),                       # 1: plain
        ObjectiveSource(code_packet=_code_packet(domain_edge=(4, 1, 60))), # 2: graph-positive
        ObjectiveSource(code_packet=_code_packet()),                       # 3: plain
    ]


@pytest.mark.parametrize("seed", [6, 71])
def test_ppgp_fim_output2_no_false_quota_error(seed: int) -> None:
    """CR-01: matcher must not raise ObjectiveQuotaUnsatisfiedError.

    With the ppgp pool, FIM objective, and output_count=2, the assignment
    (source indices 2, 3) satisfies quotas and contains a graph-positive item.
    The matcher must find a satisfying assignment rather than raising.
    """
    sources = _ppgp_sources()

    def graph_positive(source: ObjectiveSource, item) -> bool:
        receipt = assess_graph_positive_capability(
            item,
            source,
            graph_relations=("domain",),
            require_route_sidecars=False,
        )
        return bool(receipt["eligible"])

    # Must not raise ObjectiveQuotaUnsatisfiedError
    realized = EligibilityAwareTaskMixer(
        {TaskKind.FIM: 1.0},
        seed=seed,
    ).materialize_window_from_pool(
        sources,
        output_count=2,
        required_realized_assignment=graph_positive,
        candidate_assignment=lambda source, task: source_has_graph_candidate(
            source,
            task,
            graph_relations=("domain",),
        ),
    )

    # Quotas must be exactly satisfied: 2 FIM samples
    assert len(realized) == 2
    assert all(item.task == TaskKind.FIM for item in realized)

    # At least one returned assignment must be graph-positive
    graph_positive_items = [
        item
        for item in realized
        if graph_positive(sources[item.source_index], item)
    ]
    assert len(graph_positive_items) >= 1, (
        "expected at least one graph-positive realized assignment"
    )


@pytest.mark.parametrize("seed", [6, 71])
def test_ppgp_fim_output2_source_2_3_admissible(seed: int) -> None:
    """Source indices (2, 3) remain an admissible solution.

    Manually verify that the mixer can produce an assignment using sources 2
    and 3 when the search space is not artificially restricted.
    """
    sources = _ppgp_sources()

    def graph_positive(source: ObjectiveSource, item) -> bool:
        receipt = assess_graph_positive_capability(
            item,
            source,
            graph_relations=("domain",),
            require_route_sidecars=False,
        )
        return bool(receipt["eligible"])

    realized = EligibilityAwareTaskMixer(
        {TaskKind.FIM: 1.0},
        seed=seed,
    ).materialize_window_from_pool(
        sources,
        output_count=2,
        required_realized_assignment=graph_positive,
        candidate_assignment=lambda source, task: source_has_graph_candidate(
            source,
            task,
            graph_relations=("domain",),
        ),
    )

    selected = sorted(item.source_index for item in realized)
    # The assignment must include the graph-positive source (index 2)
    assert 2 in selected, (
        f"graph-positive source 2 must be selected, got {selected}"
    )
