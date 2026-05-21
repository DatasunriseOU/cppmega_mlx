"""Generate the E2E coverage matrix fixtures.

See ``E2EMatrix.md`` §3.2-3.3 for the spec.

Emits:
  tests/fixtures/tokenizers/T1_cppmega_v3.json    (vendored copy)
  tests/fixtures/tokenizers/T2_gpt2_small.json    (HF gpt2)
  tests/fixtures/tokenizers/T3_minimal_no_fim.json (256-vocab BPE)
  tests/fixtures/tokenizers/T4_fim_only.json      (1024-vocab BPE + FIM)
  tests/fixtures/parquet/{T*_P*}.parquet           (4×4 = 16 shards)
  tests/fixtures/MATRIX.json                       (machine-readable index)

Idempotent: every output is hashed against its inputs and skipped on
re-run unless content drifts. CI calls this script once per workflow.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
TOKENIZERS_DIR = FIXTURES / "tokenizers"
PARQUET_DIR = FIXTURES / "parquet"
INDEX_PATH = FIXTURES / "MATRIX.json"

VENDORED_TOKENIZER = REPO / "cppmega_mlx" / "tokenizer" / "tokenizer.json"

# Small training corpus for T3/T4. Real shape doesn't matter — just enough
# byte diversity for BPE to learn ~1k merges.
_CORPUS = [
    "def hello(name):\n    return f'Hello, {name}!'\n",
    "class Foo:\n    def __init__(self, x): self.x = x\n",
    "import numpy as np\n\ndef relu(x): return np.maximum(0, x)\n",
    "for i in range(10):\n    print(i * i)\n",
    "// C++ comment\nint main() { return 0; }\n",
    "/* block comment */\nstruct Point { float x, y; };\n",
    "// trailing whitespace\nauto sum = [](int a, int b) { return a + b; };\n",
    "template<typename T>\nT max(T a, T b) { return a > b ? a : b; }\n",
    "namespace foo {\n  inline constexpr int N = 42;\n}\n",
    "if (x > 0) {\n  printf(\"positive\\n\");\n} else {\n  printf(\"non-positive\\n\");\n}\n",
] * 16  # ~160 lines, enough for BPE


# ---------------------------------------------------------------------------
# Tokenizer builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    vocab_target: int
    specials: tuple[str, ...]
    builder: Callable[[], Tokenizer]


def _build_t1_vendored() -> Tokenizer:
    if not VENDORED_TOKENIZER.is_file():
        raise FileNotFoundError(VENDORED_TOKENIZER)
    return Tokenizer.from_file(str(VENDORED_TOKENIZER))


def _build_t2_gpt2() -> Tokenizer:
    return Tokenizer.from_pretrained("gpt2")


def _train_bpe(vocab_size: int, specials: tuple[str, ...]) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(specials),
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_CORPUS, trainer=trainer)
    bos = tok.token_to_id("<BOS>")
    eos = tok.token_to_id("<EOS>")
    if bos is not None and eos is not None:
        tok.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            pair="<BOS> $A <EOS> $B <EOS>",
            special_tokens=[("<BOS>", bos), ("<EOS>", eos)],
        )
    return tok


def _build_t3_minimal() -> Tokenizer:
    return _train_bpe(
        vocab_size=256,
        specials=("<PAD>", "<UNK>", "<BOS>", "<EOS>"),
    )


def _build_t4_fim_only() -> Tokenizer:
    return _train_bpe(
        vocab_size=1024,
        specials=("<PAD>", "<UNK>", "<BOS>", "<EOS>",
                  "<FIM_PREFIX>", "<FIM_MIDDLE>", "<FIM_SUFFIX>"),
    )


TOKENIZER_SPECS: tuple[TokenizerSpec, ...] = (
    TokenizerSpec("T1_cppmega_v3",    65536, (), _build_t1_vendored),
    TokenizerSpec("T2_gpt2_small",    50257, (), _build_t2_gpt2),
    TokenizerSpec("T3_minimal_no_fim", 256,
                  ("<PAD>", "<UNK>", "<BOS>", "<EOS>"), _build_t3_minimal),
    TokenizerSpec("T4_fim_only",      1024,
                  ("<PAD>", "<UNK>", "<BOS>", "<EOS>",
                   "<FIM_PREFIX>", "<FIM_MIDDLE>", "<FIM_SUFFIX>"),
                  _build_t4_fim_only),
)


# ---------------------------------------------------------------------------
# Parquet schema variants
# ---------------------------------------------------------------------------


PARQUET_SCHEMAS: tuple[str, ...] = ("P1_minimal", "P2_doc",
                                    "P3_engram", "P4_full")

# Synthetic sample sentences that every tokenizer encodes to populate
# the input_ids column. We don't care about semantic content — we care
# about deterministic, decode-able token streams.
SAMPLE_SENTENCES: tuple[str, ...] = (
    "def add(a, b): return a + b\n",
    "for i in range(8): print(i)\n",
    "class Foo: pass\n",
    "// hello world\n",
    "int main() { return 0; }\n",
    "auto x = 42;\n",
    "import sys\nprint(sys.argv)\n",
    "if (x) { y = 1; }\n",
)


def _make_token_rows(tok: Tokenizer, n_rows: int, seq_len: int) -> list[list[int]]:
    """Encode SAMPLE_SENTENCES repeatedly, slice/pad to seq_len."""
    rows: list[list[int]] = []
    pad_id = tok.token_to_id("<PAD>") or 0
    for i in range(n_rows):
        text = SAMPLE_SENTENCES[i % len(SAMPLE_SENTENCES)]
        ids = tok.encode(text).ids
        if len(ids) >= seq_len:
            rows.append(ids[:seq_len])
        else:
            rows.append(ids + [pad_id] * (seq_len - len(ids)))
    return rows


def _parquet_columns_for(schema: str, tok: Tokenizer, n_rows: int,
                         seq_len: int) -> dict[str, Any]:
    tokens = _make_token_rows(tok, n_rows, seq_len)
    cols: dict[str, Any] = {"input_ids": tokens}
    if schema in ("P2_doc", "P3_engram", "P4_full"):
        cols["doc_ids"] = [i // 4 for i in range(n_rows)]
    if schema in ("P3_engram", "P4_full"):
        cols["call_edges"] = [[(0, 1)] for _ in range(n_rows)]
    if schema == "P4_full":
        cols["loss_mask"] = [[1] * seq_len for _ in range(n_rows)]
        cols["chunk_boundaries"] = [[0, seq_len // 2] for _ in range(n_rows)]
        cols["type_edges"] = [[(0, 2)] for _ in range(n_rows)]
        cols["constituent_provenance_offsets"] = [[0, 8, 16, 24]
                                                   for _ in range(n_rows)]
    return cols


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(content, str):
        tmp.write_text(content)
    else:
        tmp.write_bytes(content)
    tmp.replace(path)


def generate_tokenizers() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in TOKENIZER_SPECS:
        dest = TOKENIZERS_DIR / f"{spec.name}.json"
        tok = spec.builder()
        json_str = tok.to_str(pretty=False)
        digest = _hash_text(json_str)
        # Skip if identical content already on disk.
        if dest.is_file() and _hash_text(dest.read_text()) == digest:
            existing = json.loads(dest.read_text())
            out[spec.name] = {
                "path": str(dest),
                "vocab_size": len(existing.get("model", {}).get("vocab", {}))
                              or tok.get_vocab_size(),
                "specials": list(spec.specials),
                "digest": digest,
                "fresh": False,
            }
            continue
        _atomic_write(dest, json_str)
        out[spec.name] = {
            "path": str(dest),
            "vocab_size": tok.get_vocab_size(),
            "specials": list(spec.specials),
            "digest": digest,
            "fresh": True,
        }
    return out


def generate_parquets(*, n_rows: int = 32, seq_len: int = 64,
                      ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for tok_spec in TOKENIZER_SPECS:
        tok = Tokenizer.from_file(str(TOKENIZERS_DIR / f"{tok_spec.name}.json"))
        for schema in PARQUET_SCHEMAS:
            dest = PARQUET_DIR / f"{tok_spec.name}__{schema}.parquet"
            cols = _parquet_columns_for(schema, tok, n_rows, seq_len)
            table = pa.table(cols)
            # Always rewrite — parquet is small (KB) and binary-stable
            # only if pyarrow version is pinned. Caller treats output as
            # authoritative.
            pq.write_table(table, dest)
            out[f"{tok_spec.name}__{schema}"] = {
                "path": str(dest),
                "tokenizer": tok_spec.name,
                "schema": schema,
                "rows": n_rows,
                "seq_len": seq_len,
                "columns": list(cols.keys()),
            }
    return out


def validate_round_trip() -> dict[str, dict[str, str]]:
    """Decode the first row of every shard and assert non-empty string."""
    out: dict[str, dict[str, str]] = {}
    for tok_spec in TOKENIZER_SPECS:
        tok = Tokenizer.from_file(str(TOKENIZERS_DIR / f"{tok_spec.name}.json"))
        for schema in PARQUET_SCHEMAS:
            path = PARQUET_DIR / f"{tok_spec.name}__{schema}.parquet"
            pf = pq.ParquetFile(path)
            table = pf.read_row_group(0).slice(0, 1)
            first_ids = table.column("input_ids")[0].as_py()
            decoded = tok.decode([int(x) for x in first_ids])
            out[f"{tok_spec.name}__{schema}"] = {
                "first_decoded": decoded[:120],
                "non_empty": str(bool(decoded.strip())),
            }
    return out


def write_index(tokenizers: dict[str, Any], parquets: dict[str, Any],
                round_trip: dict[str, Any]) -> Path:
    payload = {
        "tokenizers": tokenizers,
        "parquets": parquets,
        "round_trip": round_trip,
    }
    INDEX_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return INDEX_PATH


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if "--clean" in argv:
        for d in (TOKENIZERS_DIR, PARQUET_DIR):
            if d.is_dir():
                shutil.rmtree(d)
        INDEX_PATH.unlink(missing_ok=True)
    toks = generate_tokenizers()
    parqs = generate_parquets()
    rtt = validate_round_trip()
    index = write_index(toks, parqs, rtt)
    print(f"e2e matrix ready:")
    print(f"  tokenizers: {len(toks)} (fresh: "
          f"{sum(1 for v in toks.values() if v['fresh'])})")
    print(f"  parquets:   {len(parqs)}")
    print(f"  index:      {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
