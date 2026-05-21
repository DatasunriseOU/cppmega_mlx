"""Catalogue of tooltip explanations for the Visual Builder.

Every option exposed in a GUI dropdown (optimizer kind, activation
name, norm, schedule, loss, rewriter, brick) has an :class:`ExplainEntry`
keyed by (category, name). The frontend renders ``summary`` in a hover
tooltip and the full entry in :class:`ExplainModal`.

Authoritative source for *defaults* — the GUI auto-populates the
``recommended_params`` map when a user picks an option. Keep this in
sync with the factory defaults in:
  - cppmega_v4/buildspec/optim_spec.py
  - cppmega_v4/buildspec/schedules.py
  - cppmega_mlx/nn/activations.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Category = Literal[
    "optimizer", "activation", "norm", "schedule",
    "loss", "rewriter", "brick",
]


CATEGORIES: tuple[Category, ...] = (
    "optimizer", "activation", "norm", "schedule",
    "loss", "rewriter", "brick",
)


@dataclass(frozen=True)
class ExplainEntry:
    """One catalogue entry. All fields are user-facing text or
    machine-readable hints for the GUI."""

    category: Category
    name: str
    summary: str
    when_to_use: str
    when_to_avoid: str
    recommended_params: dict[str, Any] = field(default_factory=dict)
    paper_ref: str | None = None
    paper_url: str | None = None
    gotchas: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_entry(category: str, name: str) -> ExplainEntry | None:
    return CATALOG.get((category, name))


def list_options(category: str) -> list[ExplainEntry]:
    return [v for (cat, _), v in CATALOG.items() if cat == category]


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


_OPTIMIZER_ENTRIES = [
    ExplainEntry(
        category="optimizer", name="adamw",
        summary="Adam with decoupled weight decay (Loshchilov 2017).",
        when_to_use="Default for transformer pretraining. Robust to "
                    "hyperparameters; works from 100M to >100B params.",
        when_to_avoid="Memory-constrained training (2× momentum buffers "
                      "in fp32); large 2D weight matrices where Muon is "
                      "1.5-2× faster.",
        recommended_params={"lr": 3e-4, "betas": (0.9, 0.95),
                            "weight_decay": 0.01, "eps": 1e-8},
        paper_ref="Loshchilov & Hutter, 2017",
        paper_url="https://arxiv.org/abs/1711.05101",
        gotchas=("eps too small (<1e-8) → NaN on bf16 params",
                 "lr > 1e-3 typically diverges for >1B models"),
    ),
    ExplainEntry(
        category="optimizer", name="muon",
        summary="Newton-Schulz orthogonalization of 2D weight grads (Jordan 2024).",
        when_to_use="2D linear weight matrices in transformer backbone "
                    "(Q/K/V/O projections, MLP gate/up/down). 1.5-2× "
                    "faster wall-clock than AdamW on the same hardware.",
        when_to_avoid="1D parameters (biases, layer-norm gains) and "
                      "embeddings — Muon skips them; use a hybrid with "
                      "AdamW on those groups.",
        recommended_params={"lr": 2e-3, "momentum": 0.95, "ns_steps": 5,
                            "weight_decay": 0.01, "ns_carrier": "fp32"},
        paper_ref="Jordan, 2024 (Modula)",
        paper_url="https://kellerjordan.github.io/posts/muon/",
        gotchas=("ns_steps < 3 → poor orthogonalization, unstable training",
                 "fp16 ns_carrier loses precision on large matrices",
                 "1D params SKIP Muon — pair with AdamW (muon_adamw_hybrid)"),
    ),
    ExplainEntry(
        category="optimizer", name="muon_adamw_hybrid",
        summary="Muon on 2D backbone weights, AdamW on embeddings/norms/head/MoE.",
        when_to_use="Default for >1B LLM pretraining. Best of both: "
                    "Muon speed where it applies, AdamW for params Muon "
                    "skips (1D + lookup tables + MoE experts).",
        when_to_avoid="Memory-constrained or small models (<100M) — use "
                      "Lion or plain AdamW.",
        recommended_params={"muon_lr": 1e-2, "adam_lr": 3e-4,
                            "ns_steps": 5, "weight_decay": 0.01,
                            "adam_betas": (0.9, 0.95)},
        paper_ref="Jordan 2024 + Loshchilov 2017",
        paper_url="https://kellerjordan.github.io/posts/muon/",
        gotchas=("matcher misclassification → Muon applied to 1D bias → "
                 "degraded loss",
                 "MoE experts MUST be in AdamW group, not Muon"),
    ),
    ExplainEntry(
        category="optimizer", name="lion",
        summary="Sign-based momentum, 50% less state than AdamW (Chen 2023).",
        when_to_use="Memory-constrained training (single fp32 momentum "
                    "buffer = half of AdamW); works well 100M-7B params.",
        when_to_avoid="Small batch sizes (<64) — sign updates are noisy. "
                      "Models <100M params — less stable than AdamW.",
        recommended_params={"lr": 1e-4, "betas": (0.9, 0.99),
                            "weight_decay": 0.01},
        paper_ref="Chen et al., 2023",
        paper_url="https://arxiv.org/abs/2302.06675",
        gotchas=("lr > 5e-4 → NaN — gradient magnitude not used, only "
                 "sign, so lr must be 3-10× smaller than AdamW",
                 "Less stable than AdamW on <100M params"),
    ),
    ExplainEntry(
        category="optimizer", name="lion8bit",
        summary="Lion with int8 momentum + per-256-block fp32 absmax.",
        when_to_use="Same as Lion but even tighter memory budget — "
                    "quantized momentum reduces state by another 4× "
                    "compared to plain Lion.",
        when_to_avoid="Same as Lion + scale ≤ 100M params (quantization "
                      "noise dominates) and ≤ 64 batch size.",
        recommended_params={"lr": 1e-4, "betas": (0.9, 0.99),
                            "weight_decay": 0.01,
                            "quant_scheme": "linear", "block_size": 256},
        paper_ref="Chen et al., 2023 + bitsandbytes 8-bit optimizers",
        paper_url="https://arxiv.org/abs/2302.06675",
        gotchas=("lr > 5e-4 → NaN (same as Lion)",
                 "Quantization adds ~1% loss noise on small models"),
    ),
    ExplainEntry(
        category="optimizer", name="adam8bit",
        summary="AdamW with int8 quantized first + second moments.",
        when_to_use="When AdamW behaviour is desired but the full fp32 "
                    "moment buffers don't fit. Quantization preserves "
                    "update direction; only state size shrinks.",
        when_to_avoid="Apple Silicon Metal kernel suite where the int8 "
                      "ops aren't fused — overhead can offset savings.",
        recommended_params={"lr": 3e-4, "betas": (0.9, 0.999),
                            "weight_decay": 0.01,
                            "quant_scheme": "linear"},
        paper_ref="Dettmers et al., 2022",
        paper_url="https://arxiv.org/abs/2110.02861",
        gotchas=("Quantization noise visible on small models",
                 "Slight slowdown vs plain AdamW unless ops fuse"),
    ),
    ExplainEntry(
        category="optimizer", name="sgd",
        summary="Plain SGD (no adaptive scaling, no momentum buffers).",
        when_to_use="Computer-vision-style fine-tuning or RL where "
                    "Adam-family overfits the noise; or extreme memory "
                    "budgets where even Lion is too heavy.",
        when_to_avoid="LLM pretraining — Adam-family converges 2-5× "
                      "faster in wall-clock; SGD on transformers needs "
                      "carefully tuned momentum + warmup.",
        recommended_params={"lr": 1e-2, "weight_decay": 0.0},
        paper_ref=None,
        paper_url=None,
        gotchas=("No fp32 master weights → bf16 underflow during update",
                 "Sensitive to lr scaling; pair with linear_warmup"),
    ),
]


_ACTIVATION_ENTRIES = [
    ExplainEntry(
        category="activation", name="gelu",
        summary="Gaussian Error Linear Unit (Hendrycks & Gimpel 2016).",
        when_to_use="Default for GPT-2/3/Codex-style models. Smooth, "
                    "non-monotonic; good baseline for fine-tuning of "
                    "GELU-pretrained checkpoints.",
        when_to_avoid="LLaMA-family (use SiLU/SwiGLU) and ReLU-specialised "
                      "hardware (some TPU pipelines).",
        recommended_params={},
        paper_ref="Hendrycks & Gimpel, 2016",
        paper_url="https://arxiv.org/abs/1606.08415",
        gotchas=("Multiple approximations exist (tanh, erf); we use "
                 "MLX gelu_approx — differs slightly from erf-exact",),
    ),
    ExplainEntry(
        category="activation", name="relu",
        summary="Rectified Linear Unit (Nair & Hinton 2010).",
        when_to_use="Vision and classic MLPs. Very fast on Metal; sparse "
                    "activation patterns help downstream pruning.",
        when_to_avoid="LLM pretraining — dead-neuron problem hurts "
                      "convergence vs smooth alternatives.",
        recommended_params={},
        paper_ref="Nair & Hinton, 2010",
        paper_url="https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf",
        gotchas=("Dead neurons: some units may emit 0 for the entire "
                 "dataset and never receive gradient",),
    ),
    ExplainEntry(
        category="activation", name="relu2",
        summary="Squared ReLU — `square(max(0, x))` (Hua et al. 2022, T5).",
        when_to_use="TPU/GPU LLM pretraining where compute matters. "
                    "Sparser than GELU (more exact zeros) → benefits "
                    "downstream pruning and quantization.",
        when_to_avoid="Small models (<100M) — sparsity hurts capacity; "
                      "small batch sizes — variance increases.",
        recommended_params={},
        paper_ref="Hua et al., 2022 (T5-Pile)",
        paper_url="https://arxiv.org/abs/2109.08668",
        gotchas=("Larger output magnitude than ReLU → may need scaled "
                 "weight init or higher weight decay",),
    ),
    ExplainEntry(
        category="activation", name="sqrelu",
        summary="Squared ReLU via Metal kernel (training-aware fast path).",
        when_to_use="Same math as relu2 but routed through a custom "
                    "Metal kernel that fuses square+max. Use on Apple "
                    "Silicon when relu2 is the chosen activation.",
        when_to_avoid="Non-Metal runtimes — falls back to the same "
                      "Python impl as relu2, no benefit.",
        recommended_params={},
        paper_ref="Hua et al., 2022 (math) + cppmega Metal kernel",
        paper_url="https://arxiv.org/abs/2109.08668",
        gotchas=("Identical math to relu2; named separately so future "
                 "kernel iterations can diverge",),
    ),
    ExplainEntry(
        category="activation", name="silu",
        summary="SiLU / Swish — `x * sigmoid(x)` (Ramachandran et al. 2017).",
        when_to_use="Dense MLP in modern LLMs; the non-gated half of "
                    "SwiGLU. Smooth, self-gated, slightly better than "
                    "GELU on most LLM ablations.",
        when_to_avoid="When you specifically want the gated capacity of "
                      "SwiGLU — pair silu with a gate projection instead.",
        recommended_params={},
        paper_ref="Ramachandran et al., 2017",
        paper_url="https://arxiv.org/abs/1710.05941",
        gotchas=("Slightly more compute than ReLU; not a problem on "
                 "modern hardware",),
    ),
    ExplainEntry(
        category="activation", name="swiglu",
        summary="Gated SiLU — `silu(gate) * up` (Shazeer 2020). LLM default.",
        when_to_use="Best quality MLP variant across model scales. "
                    "Used in LLaMA / Qwen / DeepSeek / Mistral / Gemma. "
                    "Requires a gated_mlp brick (two input projections).",
        when_to_avoid="Memory-bound training — gated activations carry "
                      "~1.5× param footprint vs dense MLP (gate + up).",
        recommended_params={"intermediate_size": "auto (8/3*H or 4*H)"},
        paper_ref="Shazeer, 2020",
        paper_url="https://arxiv.org/abs/2002.05202",
        gotchas=("Cannot drop into a dense mlp brick — requires gate "
                 "projection; verify will block the mismatch",
                 "Convention: intermediate_size = 8/3*H rounded to 256"),
    ),
    ExplainEntry(
        category="activation", name="mish",
        summary="Mish — `x * tanh(softplus(x))` (Misra 2019). Smooth, "
                "self-regularizing.",
        when_to_use="Vision tasks; some LLM ablations show +0.5-1% "
                    "accuracy vs ReLU/GELU.",
        when_to_avoid="LLM pretraining at scale — SwiGLU dominates.",
        recommended_params={},
        paper_ref="Misra, 2019",
        paper_url="https://arxiv.org/abs/1908.08681",
        gotchas=("More compute than GELU; not significantly better on "
                 "LLM downstream metrics",),
    ),
    ExplainEntry(
        category="activation", name="geglu",
        summary="Gated GELU — `gelu(gate) * up` (Shazeer 2020).",
        when_to_use="GLM / Falcon-family pretraining; fine-tuning from "
                    "GELU-pretrained checkpoints with gating.",
        when_to_avoid="Same as SwiGLU memory caveat; pick SwiGLU if "
                      "you're starting fresh in 2025.",
        recommended_params={},
        paper_ref="Shazeer, 2020",
        paper_url="https://arxiv.org/abs/2002.05202",
        gotchas=("Requires gated_mlp brick",),
    ),
    ExplainEntry(
        category="activation", name="reglu",
        summary="Gated ReLU — `relu(gate) * up` (Shazeer 2020).",
        when_to_use="Hardware-constrained inference where ReLU is "
                    "cheaper than GELU/SiLU on the target accelerator.",
        when_to_avoid="Modern LLM pretraining — quality lags SwiGLU.",
        recommended_params={},
        paper_ref="Shazeer, 2020",
        paper_url="https://arxiv.org/abs/2002.05202",
        gotchas=("Dead-gate problem: some gate units may emit 0 "
                 "indefinitely",),
    ),
    ExplainEntry(
        category="activation", name="xielu",
        summary="Extended xGLU — `gelu(gate) * silu(up)`. Niche, "
                "appears in Megatron ablations.",
        when_to_use="Research / ablation runs; not a default choice.",
        when_to_avoid="Production pretraining — under-studied.",
        recommended_params={},
        paper_ref=None,
        paper_url=None,
        gotchas=("Combined gating + activation cost > SwiGLU; quality "
                 "rarely better",),
    ),
]


_NORM_ENTRIES = [
    ExplainEntry(
        category="norm", name="rmsnorm",
        summary="Root Mean Square normalization (Zhang & Sennrich 2019).",
        when_to_use="Default for LLM pretraining. Simpler and ~10% "
                    "faster than LayerNorm (no mean centering).",
        when_to_avoid="Models where mean offset matters (BERT-style "
                      "encoder, some vision pipelines).",
        recommended_params={"eps": 1e-6},
        paper_ref="Zhang & Sennrich, 2019",
        paper_url="https://arxiv.org/abs/1910.07467",
        gotchas=("eps < 1e-8 → NaN on bf16",),
    ),
    ExplainEntry(
        category="norm", name="layernorm",
        summary="LayerNorm — mean + variance normalization (Ba et al. 2016).",
        when_to_use="GPT-style models and fine-tuning from LayerNorm "
                    "checkpoints. Required by some legacy pretraining "
                    "recipes (BERT, T5).",
        when_to_avoid="LLaMA-family pretraining — prefer RMSNorm.",
        recommended_params={"eps": 1e-5},
        paper_ref="Ba et al., 2016",
        paper_url="https://arxiv.org/abs/1607.06450",
        gotchas=("~10% slower than RMSNorm on Metal/CUDA",),
    ),
    ExplainEntry(
        category="norm", name="none",
        summary="No normalization (raw residual stream).",
        when_to_use="Experimental / ablation; tiny models <10M params; "
                    "some specialised architectures (DeepNet, NoNorm).",
        when_to_avoid="Production LLM pretraining at any scale — without "
                      "norm the residual stream variance explodes.",
        recommended_params={},
        paper_ref=None,
        paper_url=None,
        gotchas=("pre_norm AND post_norm both 'none' → ERROR in verify "
                 "(residual grad will blow up)",),
    ),
]


_SCHEDULE_ENTRIES = [
    ExplainEntry(
        category="schedule", name="constant",
        summary="Constant base_lr at every step.",
        when_to_use="Smoke runs, ablations where you want to isolate "
                    "non-LR effects; or fine-tuning where the LR was "
                    "already tuned offline.",
        when_to_avoid="Real pretraining — without warmup the first few "
                      "steps emit huge updates; without decay the loss "
                      "plateaus and never settles.",
        recommended_params={},
        paper_ref=None,
        paper_url=None,
        gotchas=("Combine with grad_clip_norm to avoid first-step blow-up",),
    ),
    ExplainEntry(
        category="schedule", name="linear_warmup",
        summary="Linear ramp from 0 to base_lr, then constant.",
        when_to_use="Fine-tuning where total duration is open-ended; "
                    "warmup phase before plugging into a longer schedule "
                    "manually.",
        when_to_avoid="Long pretraining runs — eventually need decay; "
                      "use cosine or wsd instead.",
        recommended_params={"warmup_steps": 2000},
        paper_ref="Goyal et al., 2017 (Linear scaling rule)",
        paper_url="https://arxiv.org/abs/1706.02677",
        gotchas=("warmup_steps too small → instability in early training "
                 "(transformer literature suggests 1-5% of total)",),
    ),
    ExplainEntry(
        category="schedule", name="cosine",
        summary="Cosine annealing with optional warmup (Loshchilov & Hutter 2016).",
        when_to_use="Default for LLM pretraining (Chinchilla, GPT-NeoX). "
                    "Smooth decay, no abrupt LR drops; well-studied.",
        when_to_avoid="Checkpoint reuse mid-training — decay state hard "
                      "to resume; consider WSD for that.",
        recommended_params={"warmup_steps": 2000, "total_steps": 100_000,
                            "min_lr_ratio": 0.1},
        paper_ref="Loshchilov & Hutter, 2016 (SGDR)",
        paper_url="https://arxiv.org/abs/1608.03983",
        gotchas=("total_steps must match actual training duration — "
                 "ending early skips the cooldown",),
    ),
    ExplainEntry(
        category="schedule", name="wsd",
        summary="Warmup → Steady → linear Decay (DeepSeek-V2 default).",
        when_to_use="Long pretraining runs (>100K steps) where "
                    "checkpoints may be reused or training extended. "
                    "Steady phase gives clean checkpoint reuse.",
        when_to_avoid="Short fine-tuning (<10K steps) — cosine is "
                      "simpler and equally effective.",
        recommended_params={"warmup_steps": 2000, "decay_steps": 5000,
                            "total_steps": 100_000, "min_lr_ratio": 0.1},
        paper_ref="DeepSeek-V2 tech report, 2024",
        paper_url="https://arxiv.org/abs/2405.04434",
        gotchas=("decay_steps + warmup_steps must be ≤ total_steps "
                 "(verified at __post_init__)",),
    ),
    ExplainEntry(
        category="schedule", name="inv_sqrt",
        summary="Linear warmup + 1/sqrt(step) decay (Vaswani et al. 2017).",
        when_to_use="Classic Transformer training; encoder-decoder "
                    "models where the original AIAYN recipe is desired.",
        when_to_avoid="Modern LLM pretraining — cosine and WSD give "
                      "better final loss in most ablations.",
        recommended_params={"warmup_steps": 4000},
        paper_ref="Vaswani et al., 2017 (Attention Is All You Need)",
        paper_url="https://arxiv.org/abs/1706.03762",
        gotchas=("No floor — LR decays toward zero indefinitely",),
    ),
    ExplainEntry(
        category="schedule", name="polynomial",
        summary="Linear warmup + polynomial decay to floor.",
        when_to_use="BERT / T5 style pretraining recipes; when you want "
                    "explicit control over the decay curve shape via "
                    "the power exponent.",
        when_to_avoid="LLM pretraining where cosine is the established "
                      "default and easier to tune.",
        recommended_params={"warmup_steps": 2000, "total_steps": 100_000,
                            "power": 2.0, "min_lr_ratio": 0.1},
        paper_ref="Devlin et al., 2018 (BERT)",
        paper_url="https://arxiv.org/abs/1810.04805",
        gotchas=("power=1 = linear decay; power>1 = convex; pick "
                 "carefully or default to cosine",),
    ),
]


_LOSS_ENTRIES = [
    ExplainEntry(
        category="loss", name="cross_entropy",
        summary="Standard token-level cross-entropy (next-token prediction).",
        when_to_use="Default for LLM pretraining and finetuning. "
                    "Works with any vocabulary size.",
        when_to_avoid="When you need multi-token speculation (MTP) or "
                      "fill-in-the-middle (IFIM) — combine with the "
                      "appropriate rewriter.",
        recommended_params={"label_smoothing": 0.0,
                            "ignore_index": -100},
        paper_ref="Standard pretraining loss",
        paper_url=None,
        gotchas=("Label smoothing > 0 hurts perplexity but can help "
                 "downstream tasks",),
    ),
    ExplainEntry(
        category="loss", name="mtp_weighted",
        summary="Multi-token prediction with k-step weighting (DeepSeek-V3).",
        when_to_use="Pretraining for models with MTP heads — predicts "
                    "k future tokens jointly with declining weights.",
        when_to_avoid="Vanilla decoder-only LLMs without MTP head — "
                      "would silently downweight standard CE.",
        recommended_params={"k": 2, "beta": 0.6, "lambda_": 0.3},
        paper_ref="DeepSeek-V3 tech report, 2024",
        paper_url="https://arxiv.org/abs/2412.19437",
        gotchas=("k > 4 typically degrades quality; sweet spot is k=2-3",
                 "Requires labels_k_shifted side channel (auto-derived)"),
    ),
    ExplainEntry(
        category="loss", name="ifim_shaped",
        summary="Inverse Fill-In-the-Middle loss with PSM/SPM token shaping.",
        when_to_use="Code-generation pretraining where the model must "
                    "learn to fill arbitrary middle spans (Codex/StarCoder).",
        when_to_avoid="Tokenizers without FIM_PREFIX/MIDDLE/SUFFIX "
                      "specials — verify will block the mismatch.",
        recommended_params={"lambda_": 1.0, "psm_prob": 0.5},
        paper_ref="Bavarian et al., 2022 (Codex FIM)",
        paper_url="https://arxiv.org/abs/2207.14255",
        gotchas=("Requires tokenizer with FIM specials — use cppmega v3 "
                 "tokenizer or GPT-2 FIM variant",),
    ),
    ExplainEntry(
        category="loss", name="mhc_attn_bias",
        summary="Multi-Head Context attention bias loss (cross-document).",
        when_to_use="Pretraining where document boundaries matter "
                    "(type-edges side channel); learns cross-doc context.",
        when_to_avoid="Single-document corpora — type_edges side channel "
                      "must be present in the parquet.",
        recommended_params={"lambda_": 0.5},
        paper_ref=None,
        paper_url=None,
        gotchas=("Requires type_edges in parquet — verify will block "
                 "without it",),
    ),
    ExplainEntry(
        category="loss", name="custom",
        summary="User-supplied loss function (escape hatch).",
        when_to_use="Research recipes where none of the builtin losses "
                    "match. Provide a dotted Python path to a callable "
                    "with the standard (logits, targets, **kwargs) "
                    "signature.",
        when_to_avoid="Production pipelines — builtin losses are tested "
                      "and integrated with side channels.",
        recommended_params={"fn": "package.module:loss_fn"},
        paper_ref=None,
        paper_url=None,
        gotchas=("No automatic side-channel derivation — user must "
                 "request them via available_side_channels",),
    ),
]


_REWRITER_ENTRIES = [
    ExplainEntry(
        category="rewriter", name="MTPRewriter",
        summary="Adds k MTP heads + labels_k_shifted side channel.",
        when_to_use="When pairing the graph with mtp_weighted loss.",
        when_to_avoid="Vanilla cross_entropy — adds dead heads.",
        recommended_params={"k": 2, "share_norm": True},
        paper_ref="DeepSeek-V3, 2024",
        paper_url="https://arxiv.org/abs/2412.19437",
        gotchas=(),
    ),
    ExplainEntry(
        category="rewriter", name="IFIMRewriter",
        summary="Inserts FIM-aware token shuffling per training step.",
        when_to_use="With ifim_shaped loss for code-generation models.",
        when_to_avoid="Models that don't use FIM at inference — wastes "
                      "training tokens on a behaviour you won't use.",
        recommended_params={"psm_prob": 0.5},
        paper_ref="Bavarian et al., 2022",
        paper_url="https://arxiv.org/abs/2207.14255",
        gotchas=("Requires FIM specials in tokenizer",),
    ),
    ExplainEntry(
        category="rewriter", name="MHCRewriter",
        summary="Injects cross-document attention bias from type_edges.",
        when_to_use="With mhc_attn_bias loss and parquet with type_edges.",
        when_to_avoid="Single-document corpora.",
        recommended_params={},
        paper_ref=None,
        paper_url=None,
        gotchas=(),
    ),
]


_BRICK_ENTRIES = [
    ExplainEntry(
        category="brick", name="attention",
        summary="Standard SDPA multi-head attention.",
        when_to_use="GPT-style backbone, baseline for ablations.",
        when_to_avoid="GQA-favoured models (use gqa_sliding); MoE-heavy "
                      "designs that benefit from MLA.",
        recommended_params={"num_heads": 8, "head_dim": 64,
                            "norm_eps": 1e-6},
        paper_ref="Vaswani et al., 2017",
        paper_url="https://arxiv.org/abs/1706.03762",
        gotchas=(),
    ),
    ExplainEntry(
        category="brick", name="gated_attention",
        summary="Attention with sigmoid output gate (Qwen3-Next).",
        when_to_use="Qwen3-Next family; benefits from output gate for "
                    "selective copy operations.",
        when_to_avoid="Strict LLaMA-replica builds.",
        recommended_params={"num_attention_heads": 8, "head_dim": 64,
                            "rope_theta": 1e6},
        paper_ref="Qwen3-Next tech report, 2025",
        paper_url=None,
        gotchas=(),
    ),
    ExplainEntry(
        category="brick", name="mla",
        summary="Multi-head Latent Attention (DeepSeek-V2/V3).",
        when_to_use="Memory-efficient KV-cache for long context. "
                    "Compresses KV via low-rank latent.",
        when_to_avoid="Short-context (<2K) — overhead of latent proj.",
        recommended_params={"num_heads": 16, "q_lora_rank": 128,
                            "kv_lora_rank": 64, "qk_nope_head_dim": 32,
                            "qk_rope_head_dim": 32, "v_head_dim": 32},
        paper_ref="DeepSeek-V2 tech report, 2024",
        paper_url="https://arxiv.org/abs/2405.04434",
        gotchas=("LoRA ranks must divide head dims",),
    ),
    ExplainEntry(
        category="brick", name="gqa_sliding",
        summary="Grouped-query attention with sliding window.",
        when_to_use="Local context heads in sliding+global hybrids "
                    "(Gemma3, GPT-OSS).",
        when_to_avoid="Models that need full global context per layer.",
        recommended_params={"num_attention_heads": 8, "head_dim": 64,
                            "sliding_window_size": 4096,
                            "qk_norm": True},
        paper_ref="Mistral 7B, 2023",
        paper_url="https://arxiv.org/abs/2310.06825",
        gotchas=("Sliding window must be ≤ training context length",),
    ),
    ExplainEntry(
        category="brick", name="cca_attention",
        summary="Cross-Coarse Attention (ZAYA-1).",
        when_to_use="ZAYA-1 family; coarse+fine window blend for "
                    "long-context efficiency.",
        when_to_avoid="Replicas of non-CCA architectures.",
        recommended_params={"num_attention_heads": 8, "head_dim": 64,
                            "fine_window": 128, "coarse_block_size": 16},
        paper_ref="ZAYA-1, 2025",
        paper_url=None,
        gotchas=(),
    ),
    ExplainEntry(
        category="brick", name="gdn",
        summary="Gated Delta Net linear attention (Qwen3-Next).",
        when_to_use="Linear-attention replacement for dense attention "
                    "in long-context models.",
        when_to_avoid="Inference on Apple Metal — vjp not implemented; "
                      "no backward pass on Mac (E2E train xfail).",
        recommended_params={"num_heads": 8, "head_dim": 64,
                            "expand_v": 1.0, "use_gate": True},
        paper_ref="Qwen3-Next, 2025",
        paper_url=None,
        gotchas=("Metal backward gap — kda/gdn cells xfail in E2E train",),
    ),
    ExplainEntry(
        category="brick", name="kda",
        summary="Kimi Delta-Attention linear attention (Kimi Linear).",
        when_to_use="Kimi-Linear family; ultra-long context (>1M tokens).",
        when_to_avoid="Same Metal-backward gap as gdn.",
        recommended_params={"num_heads": 8, "head_dim": 64},
        paper_ref="Kimi-Linear, 2024",
        paper_url=None,
        gotchas=("Metal backward gap — see gdn",),
    ),
    ExplainEntry(
        category="brick", name="mamba3",
        summary="State-Space Model with selective state (Mamba-3).",
        when_to_use="Hybrid Mamba/Attention stacks (Nemotron-H).",
        when_to_avoid="Pure-attention pipelines.",
        recommended_params={"d_model": 4096},
        paper_ref="Gu & Dao, 2024 (Mamba)",
        paper_url="https://arxiv.org/abs/2312.00752",
        gotchas=("Returns (output, state) tuple — auto-unpacked",),
    ),
    ExplainEntry(
        category="brick", name="moe",
        summary="Mixture of Experts router + experts.",
        when_to_use="Sparse-activation scaling beyond dense capacity.",
        when_to_avoid="Single-device inference where router overhead "
                      "dominates.",
        recommended_params={"num_experts": 8, "top_k": 2,
                            "activation": "swiglu"},
        paper_ref="Shazeer et al., 2017 (Outrageously Large)",
        paper_url="https://arxiv.org/abs/1701.06538",
        gotchas=("MoE experts must go in AdamW group, not Muon",
                 "top_k > num_experts → router degenerates"),
    ),
    ExplainEntry(
        category="brick", name="mlp",
        summary="Dense feed-forward block (no gating).",
        when_to_use="Smallest baseline. Pair with relu/gelu/silu.",
        when_to_avoid="Modern LLMs — gated_mlp+swiglu wins on quality.",
        recommended_params={"intermediate_size": "4*H"},
        paper_ref=None,
        paper_url=None,
        gotchas=("Gated activations (swiglu/geglu) NOT compatible — "
                 "verify will block",),
    ),
    ExplainEntry(
        category="brick", name="nsa",
        summary="Native Sparse Attention.",
        when_to_use="Long-context sparse pretraining.",
        when_to_avoid="Short context (<2K) where dense is faster.",
        recommended_params={"num_heads": 8, "head_dim": 64},
        paper_ref="DeepSeek NSA, 2024",
        paper_url=None,
        gotchas=(),
    ),
    ExplainEntry(
        category="brick", name="csa_hca",
        summary="CSA/HCA cross-attention (engram retrieval).",
        when_to_use="Models with episodic retrieval (engram bricks).",
        when_to_avoid="No type_edges in parquet — verify blocks.",
        recommended_params={"num_heads": 8, "head_dim": 64,
                            "m_csa": 64, "m_hca": 32},
        paper_ref=None,
        paper_url=None,
        gotchas=("Requires type_edges side channel",),
    ),
    ExplainEntry(
        category="brick", name="engram",
        summary="Episodic memory write/read brick.",
        when_to_use="Memory-augmented models that retrieve via call_edges.",
        when_to_avoid="Parquet without call_edges side channel.",
        recommended_params={},
        paper_ref=None,
        paper_url=None,
        gotchas=("Requires call_edges side channel",),
    ),
    ExplainEntry(
        category="brick", name="lightning_indexer",
        summary="Index-vector dispatcher with rope head.",
        when_to_use="Retrieval-augmented inference paths.",
        when_to_avoid="Pure dense LLM pretraining.",
        recommended_params={"n_heads": 8, "head_dim": 64,
                            "rope_head_dim": 64, "q_lora_rank": 128,
                            "index_topk": 32},
        paper_ref=None,
        paper_url=None,
        gotchas=(),
    ),
    ExplainEntry(
        category="brick", name="abs_pos_embed",
        summary="Absolute position embedding lookup.",
        when_to_use="Models without RoPE — classic GPT-2 style.",
        when_to_avoid="LLaMA/Qwen — they use RoPE inside attention.",
        recommended_params={"max_position_embeddings": 4096},
        paper_ref=None,
        paper_url=None,
        gotchas=("Limits context to max_position_embeddings at inference",),
    ),
    ExplainEntry(
        category="brick", name="mlstm",
        summary="Matrix LSTM (xLSTM variant).",
        when_to_use="Recurrent baselines in hybrid stacks.",
        when_to_avoid="Pure transformer pipelines.",
        recommended_params={"head_dim": 64},
        paper_ref="Beck et al., 2024 (xLSTM)",
        paper_url="https://arxiv.org/abs/2405.04517",
        gotchas=(),
    ),
    ExplainEntry(
        category="brick", name="per_layer_embed",
        summary="Per-layer learnable embedding bias.",
        when_to_use="Models that depth-modulate the residual stream.",
        when_to_avoid="Standard transformer stacks.",
        recommended_params={"layer_index": 0, "num_layers": 1},
        paper_ref=None,
        paper_url=None,
        gotchas=(),
    ),
]


# Aggregate all entries into a (category, name) -> entry dict.
CATALOG: dict[tuple[str, str], ExplainEntry] = {
    (e.category, e.name): e
    for e in (
        *_OPTIMIZER_ENTRIES,
        *_ACTIVATION_ENTRIES,
        *_NORM_ENTRIES,
        *_SCHEDULE_ENTRIES,
        *_LOSS_ENTRIES,
        *_REWRITER_ENTRIES,
        *_BRICK_ENTRIES,
    )
}
