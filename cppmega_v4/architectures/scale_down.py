"""V8-R02: ``scale_down(preset, target_bytes)`` — binary search the
smallest ``(hidden_size, num_layers)`` pair whose estimated training
memory fits inside a target byte budget.

Used by the V8 GalleryScaleDownSlider UI: drag the slider to a target
(e.g. 1 GB), get a 1B-parameter llama3_8b instead of the full 8B, drop
it on the canvas, train.

Cost model: ``cppmega_v4.spec.verify_and_estimate`` with bf16 dtype +
adamw + ``training=True`` — same surface the rest of v4 uses for memory
budgets, so the slider matches the MemoryBar.

The search is deterministic and total:

  * If the smallest reachable size (``min_hidden`` × ``min_layers``)
    already exceeds the budget, returns that minimum with
    ``fits=False``.
  * Otherwise returns the largest ``(H, L)`` on a coarse-then-refine
    grid that still fits, plus the exact original ``(H, L)`` so the UI
    can show how aggressively we scaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cppmega_v4.architectures.presets import PRESETS, _attn_params
from cppmega_v4.fusion.brick_graph import from_block_specs
from cppmega_v4.spec.api import verify_and_estimate


__all__ = ["ScaleDownResult", "scale_down", "build_preset_specs_scaled"]


# Canonical full-size (hidden, layers) for each preset family.
# Source: paper/blog references threaded through preset_training_defaults.
# Used as the upper bound for the binary search and as a record of what
# was scaled down from. Family-fallback is keyed on prefix (see
# ``_canonical_size``).
_CANONICAL_SIZE: dict[str, tuple[int, int]] = {
    # Llama 3
    "llama3_8b":         (4096, 32),
    "llama3_2_1b":       (2048, 16),
    "llama3_2_3b":       (3072, 28),
    "llama4_maverick":   (8192, 32),
    # Qwen 3
    "qwen3_dense_0_6b":  (1024, 12),
    "qwen3_dense_4b":    (2560, 28),
    "qwen3_dense_8b":    (4096, 32),
    "qwen3_dense_32b":   (5120, 64),
    "qwen3_30b_a3b":     (4096, 48),
    "qwen3_235b_a22b":   (8192, 88),
    "qwen3_next":        (4096, 32),
    # DeepSeek / Kimi
    "deepseek_v3":       (7168, 61),
    "deepseek_v4_flash": (4096, 32),
    "kimi_k2":           (7168, 61),
    "kimi_linear":       (4096, 32),
    # Mistral / Phi / Granite
    "mistral_small_3_1": (4096, 32),
    "phi4":              (5120, 32),
    "granite_4_1":       (4096, 32),
    "nanbeige_4_1":      (4096, 32),
    # OLMo
    "olmo2_7b":          (4096, 32),
    "olmo3_7b":          (4096, 32),
    "olmo3_32b":         (5120, 64),
    # GLM
    "glm_45":            (4096, 32),
    "glm_47":            (4096, 32),
    "glm_5":             (5120, 60),
    # Gemma
    "gemma4":            (3072, 26),
    "gemma3_27b":        (5376, 64),
    "gemma3_270m":       (640, 18),
    "gemma4_31b":        (5376, 64),
    "gemma_4_e2b":       (2304, 26),
    "gemma_4_e4b":       (3584, 44),
    # OSS / sliding-MoE
    "gpt_oss_20b":       (4096, 32),
    "gpt_oss_120b":      (8192, 64),
    "grok25":            (6144, 48),
    # SmolLM / GPT-2 / xLSTM
    "smollm3":           (2048, 30),
    "gpt2_xl":           (1600, 48),
    "xlstm_7b":          (4096, 32),
    # Misc
    "nemotron3":         (4096, 32),
    "zaya1":             (4096, 32),
    "longcat":           (4096, 32),
    "ling25":            (4096, 32),
    "tiny_aya":          (768, 12),
    "minimax_m2":        (4096, 32),
    "mimo_v2_5":         (4096, 32),
}


def _canonical_size(preset: str) -> tuple[int, int]:
    """Resolve the canonical (H, L) for ``preset``, with family fallback."""
    if preset in _CANONICAL_SIZE:
        return _CANONICAL_SIZE[preset]
    # Longest matching prefix wins (same shape as
    # preset_training_defaults.get_defaults).
    matches = [k for k in _CANONICAL_SIZE if preset.startswith(k.split("_")[0])]
    if matches:
        return _CANONICAL_SIZE[max(matches, key=len)]
    return (4096, 32)  # generic Llama-3-shape fallback


@dataclass(frozen=True)
class ScaleDownResult:
    """Result of :func:`scale_down`.

    Attributes:
      hidden_size: chosen H (the largest that fits, or ``min_hidden``).
      num_layers: chosen L.
      estimated_bytes: cost-model estimate at ``(H, L)`` in bf16+adamw.
      target_bytes: the budget that was requested.
      fits: whether ``estimated_bytes <= target_bytes`` at the chosen size.
      scaled_down_from: canonical full-size ``(H, L)`` for the preset.
      specs: scaled-and-repeated wire-form specs (length = L). Empty when
        instantiation is skipped.
    """

    hidden_size: int
    num_layers: int
    estimated_bytes: int
    target_bytes: int
    fits: bool
    scaled_down_from: tuple[int, int]
    specs: list[dict[str, Any]]

    def to_wire(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "estimated_bytes": self.estimated_bytes,
            "target_bytes": self.target_bytes,
            "fits": self.fits,
            "scaled_down_from": {
                "hidden_size": self.scaled_down_from[0],
                "num_layers": self.scaled_down_from[1],
            },
            "specs": self.specs,
        }


# Powers-of-two H candidates between min and canonical, plus the
# canonical exactly. Layer counts are sampled coarsely to avoid blowing
# up the search.
def _h_candidates(min_h: int, max_h: int) -> list[int]:
    h = max(min_h, 64)
    cs: list[int] = []
    while h <= max_h:
        cs.append(h)
        h *= 2
    if max_h not in cs:
        cs.append(max_h)
    return cs


def _l_candidates(min_l: int, max_l: int) -> list[int]:
    cs: list[int] = []
    # Linear in log-space: min, 2*min, 4*min, ..., max
    l = max(min_l, 1)
    while l <= max_l:
        cs.append(l)
        l *= 2
    if max_l not in cs:
        cs.append(max_l)
    return cs


def _attention_only_specs(hidden_size: int) -> list[dict[str, Any]]:
    """Minimal repeat-unit: attention + mlp. Same shape used by
    every llama-like preset, so it gives a stable per-layer cost.
    """
    return [
        {"kind": "attention", "name": "attn",
         "params": _attn_params(hidden_size)},
        {"kind": "mlp", "name": "mlp"},
    ]


def _estimate_bytes(
    preset: str, hidden_size: int, num_layers: int,
) -> int:
    """Build the scaled preset's first L=num_layers repeats, run
    verify_and_estimate, and return its peak training-memory bytes."""
    factory = PRESETS.get(preset)
    if factory is None:
        unit = _attention_only_specs(hidden_size)
    else:
        unit = factory(hidden_size)
    # Stamp the unit ``num_layers`` times, renaming bricks per layer so
    # the BrickGraph remains acyclic and uniquely-named.
    specs: list[dict[str, Any]] = []
    for li in range(num_layers):
        for s in unit:
            spec = dict(s)
            base = spec.get("name") or spec.get("kind", "brick")
            spec["name"] = f"{base}_L{li}"
            # parallel-blocks carry no top-level "name"; rename leaves
            if "parallel" in spec:
                spec["parallel"] = [
                    {**leaf, "name":
                        f"{leaf.get('name', leaf.get('kind', 'leaf'))}_L{li}"}
                    for leaf in spec["parallel"]
                ]
            specs.append(spec)
    graph = from_block_specs(specs, hidden_size=hidden_size, instantiate=False)
    res = verify_and_estimate(
        graph, dim_env={"H": hidden_size, "B": 1, "S": 256},
        training=True, optimizer="adamw", dtype_bytes=2,
    )
    return int(res.memory.total_bytes)


def scale_down(
    preset: str,
    target_bytes: int,
    *,
    min_hidden: int = 64,
    min_layers: int = 1,
) -> ScaleDownResult:
    """Find the largest ``(H, L)`` ≤ canonical whose estimated training
    memory ≤ ``target_bytes``.

    Search is a coarse log-grid over both axes — exact enough for the
    UI slider (where the user sees a live estimate within ~10% of the
    budget) and cheap enough to call on every slider tick.
    """
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be > 0, got {target_bytes}")
    canon_h, canon_l = _canonical_size(preset)
    h_grid = _h_candidates(min_hidden, canon_h)
    l_grid = _l_candidates(min_layers, canon_l)

    best: tuple[int, int, int] | None = None  # (H, L, bytes)
    fallback: tuple[int, int, int] | None = None

    for h in h_grid:
        for l_ in l_grid:
            try:
                est = _estimate_bytes(preset, h, l_)
            except Exception:
                continue
            cand = (h, l_, est)
            if est <= target_bytes:
                if best is None or h * l_ > best[0] * best[1]:
                    best = cand
            else:
                # Smallest over-budget as fallback when *nothing* fits.
                if fallback is None or est < fallback[2]:
                    fallback = cand
    if best is not None:
        h, l_, est = best
        fits = True
    elif fallback is not None:
        h, l_, est = fallback
        fits = False
    else:
        # Cost model never converged — emit the minimum with fits=False.
        h, l_, est = min_hidden, min_layers, 0
        fits = False

    # Build the final scaled specs (this is what UI drops onto the canvas).
    factory = PRESETS.get(preset)
    if factory is None:
        unit = _attention_only_specs(h)
    else:
        unit = factory(h)
    specs: list[dict[str, Any]] = []
    for li in range(l_):
        for s in unit:
            spec = dict(s)
            base = spec.get("name") or spec.get("kind", "brick")
            spec["name"] = f"{base}_L{li}"
            if "parallel" in spec:
                spec["parallel"] = [
                    {**leaf, "name":
                        f"{leaf.get('name', leaf.get('kind', 'leaf'))}_L{li}"}
                    for leaf in spec["parallel"]
                ]
            specs.append(spec)
    return ScaleDownResult(
        hidden_size=h, num_layers=l_,
        estimated_bytes=est, target_bytes=target_bytes,
        fits=fits, scaled_down_from=(canon_h, canon_l),
        specs=specs,
    )


def build_preset_specs_scaled(
    preset: str, hidden_size: int, num_layers: int,
) -> list[dict[str, Any]]:
    """Convenience wrapper: stamp the preset unit num_layers times at
    a given hidden_size. Used by the auto-fit path in R04."""
    factory = PRESETS.get(preset)
    if factory is None:
        raise ValueError(f"unknown preset {preset!r}")
    unit = factory(hidden_size)
    specs: list[dict[str, Any]] = []
    for li in range(num_layers):
        for s in unit:
            spec = dict(s)
            base = spec.get("name") or spec.get("kind", "brick")
            spec["name"] = f"{base}_L{li}"
            if "parallel" in spec:
                spec["parallel"] = [
                    {**leaf, "name":
                        f"{leaf.get('name', leaf.get('kind', 'leaf'))}_L{li}"}
                    for leaf in spec["parallel"]
                ]
            specs.append(spec)
    return specs
