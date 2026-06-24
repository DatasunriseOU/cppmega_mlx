"""Shared base-model + LoRA loading and chat-template formatting for experts.

All three experts share ONE base (Qwen3-4B-Instruct-2507) and tokenizer / chat
template; they differ only by LoRA adapter + structured-output contract. This
module centralizes:

* lazy, fail-loud loading of (base + adapter) via MLX-LM (Apple-silicon path),
* applying the base's chat template (so the native tool-calling format is used),
* a single bounded text-generation entry point.

Per RULE #1 there is NO fallback: if MLX-LM is not importable, or the base /
adapter path does not exist, or generation errors, we RAISE with where+what.
We never silently return a degraded/empty completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExpertLoadError(RuntimeError):
    """Raised when an expert's base model or LoRA adapter cannot be loaded."""


class ExpertDecodeError(RuntimeError):
    """Raised when an expert's generation / structured decode fails."""


@dataclass
class LoadedExpert:
    """A loaded (base + adapter) pair plus its tokenizer."""

    model: Any
    tokenizer: Any
    base: str
    adapter_path: str | None

    def render_chat(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Apply the base chat template (native tool-calling when ``tools``)."""
        tok = self.tokenizer
        apply = getattr(tok, "apply_chat_template", None)
        if apply is None:
            raise ExpertDecodeError(
                f"tokenizer for base {self.base!r} has no apply_chat_template; "
                "cannot render the native chat/tool-call format (fail-loud)."
            )
        kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if tools is not None:
            kwargs["tools"] = tools
        return apply(messages, **kwargs)

    def generate(self, prompt: str, *, max_tokens: int = 512, temp: float = 0.0) -> str:
        """Bounded greedy/low-temp generation. RAISES on any backend error."""
        try:
            from mlx_lm import generate as mlx_generate  # type: ignore
            from mlx_lm.sample_utils import make_sampler  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ExpertLoadError(
                "mlx_lm is required for expert inference but is not importable: "
                f"{exc}"
            ) from exc
        try:
            sampler = make_sampler(temp=temp)
            out = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
        except Exception as exc:
            raise ExpertDecodeError(
                f"generation failed for base {self.base!r}: {exc}"
            ) from exc
        if not isinstance(out, str) or out == "":
            raise ExpertDecodeError(
                f"empty generation from base {self.base!r} (fail-loud; refusing "
                "to return a degraded result)."
            )
        return out


def load_expert(base: str, adapter_path: str | None) -> LoadedExpert:
    """Load (base + optional LoRA adapter) via MLX-LM. RAISES with where+what.

    ``adapter_path`` may be ``None`` to evaluate the bare base (e.g. zero-shot
    grammar-constrained routing before any adapter exists).
    """
    try:
        from mlx_lm import load as mlx_load  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ExpertLoadError(
            f"mlx_lm is required to load expert base {base!r}: {exc}"
        ) from exc

    if adapter_path is not None:
        p = Path(adapter_path)
        if not p.exists():
            raise ExpertLoadError(
                f"LoRA adapter path {adapter_path!r} does not exist (fail-loud)."
            )

    try:
        model, tokenizer = mlx_load(base, adapter_path=adapter_path)
    except Exception as exc:
        raise ExpertLoadError(
            f"failed to load base {base!r} (adapter={adapter_path!r}): {exc}"
        ) from exc
    return LoadedExpert(
        model=model, tokenizer=tokenizer, base=base, adapter_path=adapter_path
    )
