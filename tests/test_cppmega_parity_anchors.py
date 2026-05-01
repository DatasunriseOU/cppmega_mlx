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
    "cppmega/recipes/nam56r_megatron.py",
    "cppmega/recipes/megatron_args.py",
    "cppmega/recipes/nam56r_launch.py",
    "cppmega/recipes/nam56r_nemo_recipe.py",
    "cppmega/megatron/nam56r_full_spec.py",
    "cppmega/megatron/nam56r_te_spec.py",
    "cppmega/megatron/nam56r_noconv_spec.py",
    "cppmega/megatron/mamba3_te_stack_spec.py",
    "cppmega/megatron/m2rnn_spec.py",
    "cppmega/megatron/mamba3_te_in_proj.py",
    "cppmega/megatron/mamba3_mixer.py",
    "cppmega/megatron/mamba3_te_mixer.py",
    "cppmega/features/engram/ngram_hash.py",
    "cppmega/features/structure/embedding.py",
    "cppmega/megatron/custom_embedding.py",
    "cppmega/megatron/structure_batch.py",
    "cppmega/megatron/custom_gpt_model.py",
    "cppmega/megatron/fastmtp_layer.py",
    "cppmega/megatron/mtp_native_hopper_ce.py",
    "cppmega/megatron/dsa_local_spec.py",
    "cppmega/megatron/dsa_sparse_attention.py",
    "cppmega/megatron/moe_dispatcher_patch.py",
    "cppmega/megatron/selective_fp8_moe_patch.py",
    "scripts/data_prep_parquet_to_megatron.py",
    "scripts/remote_production_h200_nam56r_v1.sh",
    "scripts/remote_sweep_h200_dsa_production.sh",
    "scripts/remote_smoke_h200_dsa_9_4_m.sh",
    "scripts/remote_smoke_h200_nam56r_k_pp1.sh",
    "scripts/remote_train_gb10_nam56r_single.sh",
    "scripts/remote_train_h200_nam56r_full.sh",
    "scripts/remote_train_h200_nam56r_lite.sh",
    "scripts/remote_train_h200_nam56r_grid.sh",
    "scripts/remote_train_h200_nam56r_tp2.sh",
    "scripts/remote_train_h200_nam56r_noconv.sh",
    "scripts/remote_train_h200_nam56r_europe_sweep.sh",
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


def test_parity_docs_keep_runtime_anchor_claims_fail_closed():
    doc = (_repo_root() / "docs" / "parity_anchors.md").read_text()
    porting = (_repo_root() / "docs" / "porting_plan.md").read_text()
    combined = f"{doc}\n{porting}"
    normalized = " ".join(combined.split())

    assert "Full NAM56R Megatron Recipe/Runtime Anchors" in doc
    assert "The source converter writes only token_ids" in doc
    assert "H200 scripts are source/runtime anchors for ../cppmega" in doc
    assert "not MLX-supported launchers" in normalized
    assert "local/tiny/partial" in doc
    assert "distributed Megatron behavior remains outside the MLX scaffold" in doc
    assert "distributed Megatron parity is not claimed" in doc
    assert "M4 Max vs GB10 parity is not proven" in combined

    unsupported_surfaces = (
        "Transformer Engine",
        "CUDA graph",
        "NCCL",
        "Triton",
        "TileLang",
        "native MTP",
        "native DSA",
        "sparse MLA",
        "Hopper/GB10",
        "TP/PP/VPP/EP/SP",
        "distributed optimizer",
    )
    for surface in unsupported_surfaces:
        assert surface in combined

    forbidden_overclaims = (
        "MLX supports full NAM56R",
        "MLX launcher supports H200",
        "GB10 parity is proven",
        "M4 Max parity with GB10 is proven",
        "M4 Max matches GB10",
        "M4-only rows prove GB10 parity",
        "distributed Megatron parity is proven",
        "full distributed Megatron parity",
        "H200 launchers are supported by cppmega.mlx",
    )
    for phrase in forbidden_overclaims:
        assert phrase not in combined


def test_parity_docs_track_current_mamba3_and_pattern_parser_contracts():
    doc = (_repo_root() / "docs" / "parity_anchors.md").read_text()

    assert "projected angles are not consumed" not in doc
    assert "[z,x,B,C,dd_dt,dd_A,trap,angles]" in doc
    assert "local trapezoidal input scaling from trap" in doc
    assert "cumulative projected-angle Author RoPE over B/C" in doc
    assert "source-shaped" in doc
    assert "(angle_dt, ssm, k, v) cache" in doc
    assert "exact Author TE/Triton/TileLang/CUDA SISO/MIMO kernels" in doc
    assert "fails closed on upstream-only symbols" in doc
    assert "accepts A, M, D, E, G, R, and pipe-delimited patterns" in doc


def test_mamba_m2rnn_perf_doc_does_not_overclaim_nam56r_runtime_parity():
    doc = (_repo_root() / "docs" / "perf_mamba_m2rnn.md").read_text()

    assert "source M layers map to Mamba3 positions" in doc
    assert "source R layers map to M2RNN" in doc
    assert "nam56r_full_spec.py" in doc
    assert "native MLA/MTP/DSA" in doc
    assert "H200/GB10 train launchers" in doc

    forbidden_overclaims = (
        "Mamba3 NAM56R performance parity is proven",
        "M2RNN NAM56R performance parity is proven",
        "Mamba3 matches GB10",
        "M2RNN matches GB10",
    )
    for phrase in forbidden_overclaims:
        assert phrase not in doc


def test_cppmega_source_reference_files_exist_when_checkout_is_present():
    cppmega_root = _cppmega_reference_root()
    if not cppmega_root.exists():
        pytest.skip("../cppmega reference checkout is not present")

    for source_path in CPPMEGA_SOURCE_ANCHORS:
        assert (cppmega_root / source_path).is_file(), source_path
