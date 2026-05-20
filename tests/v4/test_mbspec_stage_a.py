"""MBSpec Stage A tests — loss_spec + optim_spec data-layer."""

from __future__ import annotations

import pytest

from cppmega_v4.buildspec import (
    LOSS_BUILTINS,
    LossKind,
    LossSpec,
    OPTIM_BUILTINS,
    OptimKind,
    OptimSpec,
    ParamGroup,
    adamw,
    cross_entropy_loss,
    custom_loss,
    ifim_shaped_loss,
    mhc_attn_bias_loss,
    mtp_weighted_loss,
    muon,
    muon_adamw_hybrid,
    sgd,
)


# ---------------------------------------------------------------------------
# LossSpec validation
# ---------------------------------------------------------------------------


def test_loss_kind_rejects_non_enum_value():
    with pytest.raises(TypeError, match="LossKind"):
        LossSpec(kind="cross_entropy", head_outputs=("logits",))  # type: ignore[arg-type]


def test_loss_spec_rejects_empty_head_outputs():
    with pytest.raises(ValueError, match="head_outputs must not be empty"):
        LossSpec(kind=LossKind.CROSS_ENTROPY, head_outputs=())


def test_loss_spec_rejects_blank_head_output_name():
    with pytest.raises(ValueError, match="non-empty str"):
        LossSpec(kind=LossKind.CROSS_ENTROPY, head_outputs=("",))
    with pytest.raises(ValueError, match="non-empty str"):
        LossSpec(kind=LossKind.CROSS_ENTROPY, head_outputs=("   ",))


def test_loss_spec_rejects_bad_reduction():
    with pytest.raises(ValueError, match="reduction"):
        LossSpec(
            kind=LossKind.CROSS_ENTROPY,
            head_outputs=("logits",),
            reduction="trash",
        )


def test_loss_spec_rejects_bad_label_source():
    with pytest.raises(ValueError, match="label_source"):
        LossSpec(
            kind=LossKind.CROSS_ENTROPY,
            head_outputs=("logits",),
            label_source="fake_source",
        )


def test_mtp_weighted_requires_k_in_params():
    with pytest.raises(ValueError, match="MTP_WEIGHTED requires params\\['k'\\]"):
        LossSpec(
            kind=LossKind.MTP_WEIGHTED,
            params={},
            head_outputs=("logits_0", "logits_1"),
            label_source="next_k_tokens",
        )


def test_mtp_weighted_requires_head_count_to_match_k():
    with pytest.raises(ValueError, match="head_outputs length"):
        LossSpec(
            kind=LossKind.MTP_WEIGHTED,
            params={"k": 3, "beta_0": 1.0, "beta_1": 0.6, "beta_2": 0.4},
            head_outputs=("logits_0", "logits_1"),  # only 2, need 3
            label_source="next_k_tokens",
        )


def test_mtp_weighted_requires_each_beta_param():
    with pytest.raises(ValueError, match="beta_1"):
        LossSpec(
            kind=LossKind.MTP_WEIGHTED,
            params={"k": 2, "beta_0": 1.0},  # missing beta_1
            head_outputs=("logits_0", "logits_1"),
            label_source="next_k_tokens",
        )


def test_mtp_weighted_rejects_negative_beta():
    with pytest.raises(ValueError, match="beta_0"):
        LossSpec(
            kind=LossKind.MTP_WEIGHTED,
            params={"k": 1, "beta_0": -0.1},
            head_outputs=("logits_0",),
            label_source="next_k_tokens",
        )


def test_ifim_requires_lambda_fim():
    with pytest.raises(ValueError, match="lambda_fim"):
        LossSpec(
            kind=LossKind.IFIM_SHAPED,
            params={},
            head_outputs=("logits",),
        )


def test_mhc_requires_lambda_mhc():
    with pytest.raises(ValueError, match="lambda_mhc"):
        LossSpec(
            kind=LossKind.MHC_ATTN_BIAS,
            params={},
            head_outputs=("logits",),
        )


# ---------------------------------------------------------------------------
# LossSpec built-in factories
# ---------------------------------------------------------------------------


def test_cross_entropy_factory_returns_well_formed_spec():
    s = cross_entropy_loss()
    assert isinstance(s, LossSpec)
    assert s.kind is LossKind.CROSS_ENTROPY
    assert s.head_outputs == ("logits",)
    assert s.label_source == "next_token"


def test_cross_entropy_custom_head_name():
    s = cross_entropy_loss("my_head")
    assert s.head_outputs == ("my_head",)


def test_mtp_weighted_factory_default_k_2():
    s = mtp_weighted_loss()
    assert s.kind is LossKind.MTP_WEIGHTED
    assert int(s.params["k"]) == 2
    assert s.head_outputs == ("logits_0", "logits_1")
    assert s.params["beta_0"] == 1.0
    assert s.params["beta_1"] == 0.6
    assert s.label_source == "next_k_tokens"


def test_mtp_weighted_factory_custom_k_and_beta():
    s = mtp_weighted_loss(k=3, beta=(1.0, 0.5, 0.25))
    assert int(s.params["k"]) == 3
    assert s.head_outputs == ("logits_0", "logits_1", "logits_2")
    assert s.params["beta_2"] == 0.25


def test_mtp_weighted_factory_rejects_k_lt_1():
    with pytest.raises(ValueError, match="k must be ≥ 1"):
        mtp_weighted_loss(k=0)


def test_mtp_weighted_factory_rejects_beta_length_mismatch():
    with pytest.raises(ValueError, match="len\\(beta\\)"):
        mtp_weighted_loss(k=2, beta=(1.0,))


def test_ifim_factory_well_formed():
    s = ifim_shaped_loss(lambda_fim=0.05)
    assert s.kind is LossKind.IFIM_SHAPED
    assert s.params["lambda_fim"] == 0.05


def test_mhc_factory_well_formed():
    s = mhc_attn_bias_loss(lambda_mhc=0.03)
    assert s.kind is LossKind.MHC_ATTN_BIAS
    assert s.params["lambda_mhc"] == 0.03


def test_custom_loss_factory_accepts_arbitrary_params():
    s = custom_loss(("a", "b"), some_param=1.5, other=2.0)
    assert s.kind is LossKind.CUSTOM
    assert s.head_outputs == ("a", "b")
    assert s.params["some_param"] == 1.5


def test_loss_builtins_registry_covers_all_loss_kinds():
    """Every non-CUSTOM LossKind should have a builtin entry."""
    for k in LossKind:
        if k is LossKind.CUSTOM:
            continue
        assert k.value in LOSS_BUILTINS, k.value


# ---------------------------------------------------------------------------
# OptimSpec — ParamGroup validation
# ---------------------------------------------------------------------------


def test_param_group_rejects_blank_matcher():
    with pytest.raises(ValueError, match="non-empty"):
        ParamGroup(matcher="", lr=1e-3)


def test_param_group_rejects_unknown_matcher():
    with pytest.raises(ValueError, match="must be one of"):
        ParamGroup(matcher="totally_made_up_matcher", lr=1e-3)


def test_param_group_accepts_regex_matcher():
    g = ParamGroup(matcher="regex:.*expert.*", lr=1e-4)
    assert g.matcher.startswith("regex:")


def test_param_group_rejects_non_positive_lr():
    with pytest.raises(ValueError, match="lr"):
        ParamGroup(matcher="all", lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        ParamGroup(matcher="all", lr=-1e-3)


def test_param_group_rejects_negative_wd():
    with pytest.raises(ValueError, match="weight_decay"):
        ParamGroup(matcher="all", lr=1e-3, weight_decay=-0.01)


def test_param_group_rejects_bad_betas():
    with pytest.raises(ValueError, match="betas"):
        ParamGroup(matcher="all", lr=1e-3, betas=(1.5, 0.95))  # β1≥1
    with pytest.raises(ValueError, match="betas"):
        ParamGroup(matcher="all", lr=1e-3, betas=(0.9,))  # type: ignore[arg-type]


def test_param_group_rejects_bad_ns_steps():
    with pytest.raises(ValueError, match="ns_steps"):
        ParamGroup(matcher="all", lr=1e-3, ns_steps=0)


# ---------------------------------------------------------------------------
# OptimSpec validation
# ---------------------------------------------------------------------------


def test_optim_kind_rejects_non_enum():
    with pytest.raises(TypeError, match="OptimKind"):
        OptimSpec(
            kind="adamw",  # type: ignore[arg-type]
            groups=(ParamGroup(matcher="all", lr=1e-3, betas=(0.9, 0.95)),),
        )


def test_optim_spec_rejects_empty_groups():
    with pytest.raises(ValueError, match="groups must not be empty"):
        OptimSpec(kind=OptimKind.ADAMW, groups=())


def test_optim_spec_rejects_non_param_group_entries():
    with pytest.raises(TypeError, match="ParamGroup"):
        OptimSpec(
            kind=OptimKind.ADAMW,
            groups=({"matcher": "all", "lr": 1e-3},),  # type: ignore[arg-type]
        )


def test_optim_spec_rejects_bad_gradient_clip_norm():
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        OptimSpec(
            kind=OptimKind.ADAMW,
            groups=(ParamGroup(matcher="all", lr=1e-3, betas=(0.9, 0.95)),),
            gradient_clip_norm=0.0,
        )


def test_adamw_kind_requires_betas_on_every_group():
    with pytest.raises(ValueError, match="ADAMW group must declare betas"):
        OptimSpec(
            kind=OptimKind.ADAMW,
            groups=(ParamGroup(matcher="all", lr=1e-3),),  # no betas
        )


def test_muon_kind_requires_ns_steps_on_every_group():
    with pytest.raises(ValueError, match="MUON group must declare ns_steps"):
        OptimSpec(
            kind=OptimKind.MUON,
            groups=(ParamGroup(matcher="all", lr=1e-3),),  # no ns_steps
        )


# ---------------------------------------------------------------------------
# Built-in factories
# ---------------------------------------------------------------------------


def test_adamw_factory_well_formed():
    s = adamw()
    assert s.kind is OptimKind.ADAMW
    assert len(s.groups) == 1
    assert s.groups[0].matcher == "all"
    assert s.groups[0].lr == 3e-4
    assert s.groups[0].betas == (0.9, 0.95)


def test_muon_factory_well_formed():
    s = muon()
    assert s.kind is OptimKind.MUON
    assert s.groups[0].ns_steps == 5


def test_muon_adamw_hybrid_has_four_groups_in_order():
    s = muon_adamw_hybrid()
    assert s.kind is OptimKind.MUON_ADAMW_HYBRID
    assert len(s.groups) == 4
    matchers = [g.matcher for g in s.groups]
    assert matchers == ["moe_experts", "embeddings", "head", "all"]
    # first three groups are AdamW (carry betas), last is Muon (carries ns_steps)
    for g in s.groups[:3]:
        assert g.betas is not None
        assert g.ns_steps is None
    assert s.groups[3].ns_steps is not None
    assert s.groups[3].betas is None


def test_sgd_factory_well_formed():
    s = sgd()
    assert s.kind is OptimKind.SGD
    assert s.groups[0].betas is None
    assert s.groups[0].ns_steps is None


def test_optim_builtins_registry_covers_all_optim_kinds():
    for k in OptimKind:
        assert k.value in OPTIM_BUILTINS, k.value


# ---------------------------------------------------------------------------
# Immutability — both specs are frozen
# ---------------------------------------------------------------------------


def test_loss_spec_is_frozen():
    s = cross_entropy_loss()
    with pytest.raises((AttributeError, TypeError)):
        s.reduction = "sum"  # type: ignore[misc]


def test_optim_spec_is_frozen():
    s = adamw()
    with pytest.raises((AttributeError, TypeError)):
        s.mixed_precision = False  # type: ignore[misc]


def test_param_group_is_frozen():
    g = ParamGroup(matcher="all", lr=1e-3, betas=(0.9, 0.95))
    with pytest.raises((AttributeError, TypeError)):
        g.lr = 2e-3  # type: ignore[misc]
