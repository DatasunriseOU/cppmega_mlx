"""YAML/JSON run-template for V4 block stacks.

Declarative spec for assembling a V4 model from named blocks. Two surfaces:

    1. RunTemplate     — pydantic-style dataclass with strict validation.
    2. load_template / dump_template — round-trip via PyYAML or stdlib json.

Schema (one top-level entry per logical block):

    name: my_v4_1b
    hidden_size: 2048
    blocks:
      - kind: gdn              # GatedDeltaNet ("L" symbol)
        repeat: 2
        params: { num_heads: 16, head_dim_k: 128, head_dim_v: 128 }
      - kind: kda              # Kimi Delta Attention ("K" symbol)
        repeat: 1
        params: { num_heads: 16, num_v_heads: 32, head_dim_k: 128, head_dim_v: 128 }
      - kind: mla_absorb       # FlashMLA absorbed attention
        repeat: 4
        params: { num_heads: 16, q_lora_rank: 1024, kv_lora_rank: 256 }
      - kind: engram           # Engram block (N symbol, doc_id-aware)
        repeat: 1
        params: { num_branches: 4, branch_dim: 512 }
      - kind: moe              # aux-loss-free V4 MoE
        repeat: 2
        params: { num_experts: 64, num_experts_per_tok: 8, intermediate_size: 4096 }
    mtp:
      depth: 2
      hidden_size_override: null   # null → use top-level hidden_size

The YAML/JSON file is the single source of truth for a run; programs
should not pass extra block kwargs at runtime. The template format is
deliberately a *flat list of named entries* (no nested groupings) so
diffs against a config under code review stay small.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

try:
    import yaml as _yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


SUPPORTED_BLOCK_KINDS = frozenset({
    "gdn",       # GatedDeltaNet (L) — cppmega_v4.nn.linear_attention
    "kda",       # Kimi Delta Attention (K) — cppmega_v4.nn.kimi_delta_attention
    "mla_absorb",  # FlashMLA absorbed — cppmega_v4.nn.mla_absorb
    "mla",         # Standard MLA — vendored mlx_lm reference
    "attention",   # Standard multi-head attention
    "engram",      # Engram block (N) — cppmega_v4.nn._external.tilekernels_engram
    "moe",         # V4 MoE — cppmega_v4.nn.moe_v4
    "mlp",         # Dense MLP
    "lightning_indexer",  # V3.2 DSA indexer (ROI 7)
    "nsa",         # Native Sparse Attention (ROI 8)
    "csa_hca",     # CSA+HCA hybrid (ROI 9)
})


@dataclass
class BlockSpec:
    """One block entry in a RunTemplate."""

    kind: str
    repeat: int = 1
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_BLOCK_KINDS:
            raise ValueError(
                f"unsupported block kind {self.kind!r}; "
                f"must be one of {sorted(SUPPORTED_BLOCK_KINDS)}"
            )
        if not isinstance(self.repeat, int) or self.repeat < 1:
            raise ValueError(f"repeat must be a positive int, got {self.repeat!r}")
        if not isinstance(self.params, dict):
            raise ValueError(f"params must be a dict, got {type(self.params).__name__}")


@dataclass
class MTPSpec:
    """Optional MTP-head spec attached to the run."""

    depth: int = 0
    hidden_size_override: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError(f"depth must be a non-negative int, got {self.depth!r}")
        if self.hidden_size_override is not None and self.hidden_size_override <= 0:
            raise ValueError(
                f"hidden_size_override must be positive or None, "
                f"got {self.hidden_size_override!r}"
            )


@dataclass
class RunTemplate:
    """Full run spec: name + dims + ordered block list + optional MTP."""

    name: str
    hidden_size: int
    blocks: list[BlockSpec]
    mtp: Optional[MTPSpec] = None
    vocab_size: Optional[int] = None
    # Schema version — bump on breaking changes; readers use this to migrate.
    schema_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size!r}")
        if not self.blocks:
            raise ValueError("blocks must be non-empty")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive or None, got {self.vocab_size!r}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunTemplate":
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        # Schema version check (forward-compatible: warn but accept older).
        ver = data.get("schema_version", 1)
        if ver > cls.schema_version:
            raise ValueError(
                f"template schema_version {ver} is newer than reader "
                f"version {cls.schema_version}; upgrade cppmega_v4"
            )
        blocks = [BlockSpec(**b) for b in data.get("blocks", [])]
        mtp = MTPSpec(**data["mtp"]) if data.get("mtp") else None
        return cls(
            name=data["name"],
            hidden_size=data["hidden_size"],
            blocks=blocks,
            mtp=mtp,
            vocab_size=data.get("vocab_size"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "name": self.name,
            "hidden_size": self.hidden_size,
            "blocks": [asdict(b) for b in self.blocks],
        }
        if self.mtp is not None:
            d["mtp"] = asdict(self.mtp)
        if self.vocab_size is not None:
            d["vocab_size"] = self.vocab_size
        return d

    def total_blocks(self) -> int:
        return sum(b.repeat for b in self.blocks)

    def block_kinds_used(self) -> set[str]:
        return {b.kind for b in self.blocks}


def load_template(path: str | Path) -> RunTemplate:
    """Load a RunTemplate from YAML or JSON, dispatching on file extension."""
    p = Path(path)
    text = p.read_text()
    return loads_template(text, fmt=_fmt_for_path(p))


def dump_template(template: RunTemplate, path: str | Path) -> None:
    """Write a RunTemplate to YAML or JSON, dispatching on file extension."""
    p = Path(path)
    p.write_text(dumps_template(template, fmt=_fmt_for_path(p)))


def loads_template(text: str, *, fmt: str = "yaml") -> RunTemplate:
    """Parse a RunTemplate from a string (fmt = 'yaml' or 'json')."""
    if fmt == "yaml":
        if not HAS_YAML:
            raise ImportError("PyYAML not installed; pip install pyyaml or use fmt='json'")
        data = _yaml.safe_load(text)
    elif fmt == "json":
        data = json.loads(text)
    else:
        raise ValueError(f"fmt must be 'yaml' or 'json', got {fmt!r}")
    return RunTemplate.from_dict(data)


def dumps_template(template: RunTemplate, *, fmt: str = "yaml") -> str:
    """Serialize a RunTemplate to a string (fmt = 'yaml' or 'json')."""
    data = template.to_dict()
    if fmt == "yaml":
        if not HAS_YAML:
            raise ImportError("PyYAML not installed; pip install pyyaml or use fmt='json'")
        return _yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    if fmt == "json":
        return json.dumps(data, indent=2, sort_keys=False)
    raise ValueError(f"fmt must be 'yaml' or 'json', got {fmt!r}")


def _fmt_for_path(p: Path) -> str:
    sfx = p.suffix.lower()
    if sfx in (".yaml", ".yml"):
        return "yaml"
    if sfx == ".json":
        return "json"
    raise ValueError(f"unsupported template extension {sfx!r}; use .yaml/.yml/.json")


__all__ = [
    "BlockSpec",
    "MTPSpec",
    "RunTemplate",
    "SUPPORTED_BLOCK_KINDS",
    "dump_template",
    "dumps_template",
    "load_template",
    "loads_template",
]
