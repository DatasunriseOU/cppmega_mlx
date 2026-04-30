from pathlib import Path

import pytest

from cppmega_mlx.config.model import (
    DEFAULT_DSA_A_LAYER_RANKS,
    DEFAULT_NAM56R_DEPTH,
    DEFAULT_NAM56R_PATTERN,
    LOCAL_PROFILE_VOCAB_SIZE,
    MEGACPP_TOKENIZER_VOCAB_SIZE,
    MoeConfig,
    VocabMetadata,
)
from cppmega_mlx.recipes.nam56r import build_nam56r_pattern


CPPMEGA_NAM56R_PATTERN = "AEMEAEMEAEMR"
CPPMEGA_NAM56R_DEPTH = 52
CPPMEGA_DSA_A_RANKS_FROM_LAUNCHERS = (1, 2, 3, 5, 6, 7, 9, 10, 11)

CPPMEGA_SOURCE_ANCHORS = (
    "cppmega/megatron/nam56r_layout.py",
    "cppmega/megatron/m2rnn_spec.py",
    "cppmega/megatron/mamba3_te_in_proj.py",
    "cppmega/megatron/mamba3_mixer.py",
    "cppmega/features/engram/ngram_hash.py",
    "cppmega/features/structure/embedding.py",
    "cppmega/megatron/custom_embedding.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cppmega_reference_root() -> Path:
    return _repo_root().parent / "cppmega"


def _cppmega_launcher_indexed_layers(
    a_layer_numbers: tuple[int, ...],
    ranks: tuple[int, ...],
) -> tuple[int, ...]:
    """Mirror cppmega H200 launcher preflight: layer_numbers = attn_nums[r]."""

    return tuple(a_layer_numbers[rank] for rank in ranks)


def test_nam56r_pattern_depth_counts_and_absolute_layers_match_cppmega():
    expanded = build_nam56r_pattern(
        pattern=CPPMEGA_NAM56R_PATTERN,
        depth=CPPMEGA_NAM56R_DEPTH,
        dsa_a_layer_ranks=(),
    )

    assert DEFAULT_NAM56R_PATTERN == CPPMEGA_NAM56R_PATTERN
    assert DEFAULT_NAM56R_DEPTH == CPPMEGA_NAM56R_DEPTH
    assert expanded.symbols == tuple((CPPMEGA_NAM56R_PATTERN * 5)[:CPPMEGA_NAM56R_DEPTH])
    assert expanded.counts == {"A": 13, "E": 22, "M": 13, "R": 4}
    assert expanded.r_layer_numbers == (12, 24, 36, 48)
    assert expanded.a_layer_numbers == (
        1,
        5,
        9,
        13,
        17,
        21,
        25,
        29,
        33,
        37,
        41,
        45,
        49,
    )


def test_cppmega_launcher_dsa_mla_anchor_uses_indexed_a_layer_contract():
    expanded = build_nam56r_pattern(
        pattern=CPPMEGA_NAM56R_PATTERN,
        depth=CPPMEGA_NAM56R_DEPTH,
        dsa_a_layer_ranks=(),
    )

    assert DEFAULT_DSA_A_LAYER_RANKS == CPPMEGA_DSA_A_RANKS_FROM_LAUNCHERS
    dsa_layers = _cppmega_launcher_indexed_layers(
        expanded.a_layer_numbers,
        CPPMEGA_DSA_A_RANKS_FROM_LAUNCHERS,
    )
    mla_layers = tuple(
        layer_number
        for rank, layer_number in enumerate(expanded.a_layer_numbers)
        if rank not in CPPMEGA_DSA_A_RANKS_FROM_LAUNCHERS
    )

    assert dsa_layers == (5, 9, 13, 21, 25, 29, 37, 41, 45)
    assert mla_layers == (1, 17, 33, 49)


def test_local_mlx_helper_dsa_rank_semantics_remain_explicit():
    expanded = build_nam56r_pattern()

    assert expanded.dsa_a_layer_ranks == CPPMEGA_DSA_A_RANKS_FROM_LAUNCHERS
    assert expanded.dsa_layer_numbers == (5, 9, 13, 21, 25, 29, 37, 41, 45)
    assert expanded.mla_layer_numbers == (1, 17, 33, 49)


def test_moe_and_vocab_constants_match_nam56r_cppmega_anchors():
    moe = MoeConfig()
    vocab = VocabMetadata()

    assert moe.num_experts == 16
    assert moe.top_k == 4
    assert moe.ffn_hidden_size == 896
    assert moe.shared_expert_intermediate_size == 1024
    assert LOCAL_PROFILE_VOCAB_SIZE == 65_536
    assert MEGACPP_TOKENIZER_VOCAB_SIZE == 131_072
    assert vocab.local_profile_vocab_size == 65_536
    assert vocab.megacpp_tokenizer_vocab_size == 131_072


def test_cppmega_source_reference_anchors_are_documented():
    doc = (_repo_root() / "docs" / "parity_anchors.md").read_text()

    for source_path in CPPMEGA_SOURCE_ANCHORS:
        assert source_path in doc


def test_cppmega_source_reference_files_exist_when_checkout_is_present():
    cppmega_root = _cppmega_reference_root()
    if not cppmega_root.exists():
        pytest.skip("../cppmega reference checkout is not present")

    for source_path in CPPMEGA_SOURCE_ANCHORS:
        assert (cppmega_root / source_path).is_file(), source_path
