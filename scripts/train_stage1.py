#!/usr/bin/env python
"""Stage-1 end-to-end multi-objective training smoke for the dense C++ LM.

This script ties the whole Stage-1 pipeline together on a tiny, fast profile:

  1. Load packed code rows from ``tests/fixtures/golden_mini/code/*.parquet``
     (optionally extended with a small slice of a real
     ``clang_semantic_4k_v10`` shard) and build one :class:`CodePacket` per
     packed row, trimmed to its ``valid_token_count`` so the real-token prefix
     and every token-aligned side-channel stay byte-aligned.
  2. Synthesize :class:`CommitPacket` objects from the paired commit rows in
     ``golden_mini/commits`` (consecutive pre/post rows of the same file) so the
     commit objectives have real ``pre``/``post``/``diff``/``commit_msg`` token
     streams to learn from.
  3. Use :class:`TaskMixer` (Stage-1 default rates, fixed seed) to draw ONE
     objective per packet, then build the ``(input_ids, target_ids, loss_mask)``
     example via :mod:`cppmega_mlx.training.objectives`.
  4. Run :class:`DenseCppLM` on a tiny smoke profile (d=256, depth=4) with AdamW
     for a few hundred steps, printing per-objective and overall loss and
     asserting the mixed-objective model learns (final overall loss < initial).

CRITICAL SIDE-CHANNEL ALIGNMENT RULE (enforced, fail-loud):

  * ``causal_lm`` and the ``*_recovery`` objectives keep the ORIGINAL token order.
    Their ``input_ids`` are exactly ``packet.token_ids[:-1]``, so the CodePacket's
    token-aligned structure/platform side-channels (sliced to the SAME prefix and
    order) are still valid and ARE passed into :class:`DenseCppLM`.
  * ``ast_fim`` / ``ifim`` / ``commit_diff`` / ``pre_to_post`` REORDER (FIM
    permutation) or SYNTHESIZE (commit splice) the token stream, so the original
    token-aligned channels no longer line up. For those steps the model is run
    with side-channels DISABLED (``structure_residual_scale=0`` /
    ``platform_residual_scale=0`` and NO channels passed). We NEVER pass a
    misaligned channel; whenever channels ARE passed we assert their length
    matches ``input_ids`` and RAISE on any mismatch.

RULE #1 (fail fast / fail loud): no silent fallbacks. A misaligned channel, a
missing required field, or a non-decreasing loss curve RAISES.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pyarrow.parquet as pq

# Make ``cppmega_mlx`` importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega_mlx.data.code_packet import CodePacket
from cppmega_mlx.data.commit_packet import CommitPacket
from cppmega_mlx.data.fim import FIMSpecialTokenIds
from cppmega_mlx.models.dense_cpp_lm import DenseCppLM, DenseCppLMConfig
from cppmega_mlx.training.objectives import ObjectiveExample
from cppmega_mlx.training.task_mixer import TaskKind, TaskMixer

GOLDEN_MINI = _REPO_ROOT / "tests" / "fixtures" / "golden_mini"

# Objectives that keep the ORIGINAL token order (side-channels stay aligned).
_ALIGNED_OBJECTIVES = frozenset(
    {
        "causal_lm",
        "symbol_recovery",
        "type_recovery",
        "callee_recovery",
    }
)
# Objectives that reorder / synthesize tokens (side-channels MUST be disabled).
_REORDERED_OBJECTIVES = frozenset(
    {
        "ast_fim",
        "ifim",
        "commit_diff",
        "pre_to_post",
    }
)

_SPECIAL = FIMSpecialTokenIds()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _col(arr) -> np.ndarray:
    return np.asarray(arr)


def _i32(values) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int32))


def load_code_packets(
    parquet_paths: Iterable[Path],
    *,
    vocab_size: int,
    min_tokens: int = 8,
) -> list[CodePacket]:
    """Build one CodePacket per packed code row, trimmed to its valid prefix.

    Each packed row stores 4096-long padded ``input_ids`` plus token-aligned
    structure/semantic channels. We trim to ``valid_token_count`` so the real
    tokens and EVERY aligned channel share the exact same length and order
    (the precondition for the side-channel alignment rule). Token ids are clamped
    into ``[0, vocab_size)`` only as an explicit, asserted invariant check — the
    golden prefix already lies in range, so no clamp is silently applied.
    """

    packets: list[CodePacket] = []
    for path in parquet_paths:
        table = pq.read_table(str(path))
        rows = table.to_pylist()
        for row_index, row in enumerate(rows):
            vtc = int(row["valid_token_count"])
            if vtc < min_tokens:
                continue
            tokens = _col(row["input_ids"])[:vtc].astype(np.int64)
            if tokens.min() < 0 or tokens.max() >= vocab_size:
                raise ValueError(
                    f"{path.name}[row={row_index}]: token id out of range "
                    f"[0,{vocab_size}); got [{int(tokens.min())},{int(tokens.max())}]"
                )

            def chan(name: str) -> mx.array:
                vals = _col(row[name])[:vtc]
                if len(vals) != vtc:
                    raise ValueError(
                        f"{path.name}[row={row_index}].{name}: length {len(vals)} "
                        f"!= valid_token_count {vtc}"
                    )
                return _i32(vals)

            # Chunk-aligned (NOT token-aligned) clang boundaries — required by the
            # AST-FIM span selector. Kept whole (chunk axis), only valid chunks
            # (chunk end within the trimmed prefix) are retained.
            chunk_kwargs = _chunk_channels(row, vtc, path.name, row_index)

            packet = CodePacket(
                token_ids=_i32(tokens),
                target_ids=chan("target_ids"),
                loss_mask=chan("loss_mask"),
                structure_ids=chan("token_structure_ids"),
                dep_levels=chan("token_dep_levels"),
                ast_depth=chan("token_ast_depth"),
                sibling_index=chan("token_sibling_index"),
                ast_node_type=chan("token_ast_node_type"),
                symbol_ids=chan("token_symbol_ids"),
                call_targets=chan("token_call_targets"),
                type_refs=chan("token_type_refs"),
                repo=str(row.get("repo")) if row.get("repo") is not None else None,
                filepath=str(row.get("filepath"))
                if row.get("filepath") is not None
                else None,
                metadata={
                    "platform_ids": _platform_ids(row, vocab_safe=True),
                    "source": path.name,
                    "row_index": row_index,
                },
                **chunk_kwargs,
            )
            packets.append(packet)
    if not packets:
        raise ValueError(f"no code packets loaded from {list(parquet_paths)!r}")
    return packets


def _chunk_channels(row, vtc: int, source: str, row_index: int) -> dict[str, mx.array]:
    """Load chunk-aligned clang boundaries, keeping only chunks within the prefix.

    Chunk channels are CHUNK-axis (one entry per clang chunk), so they are NOT
    trimmed to the token prefix; instead we drop any chunk whose end exceeds the
    trimmed ``valid_token_count`` (a partial chunk in the dropped padding tail).
    """

    starts = _col(row["token_chunk_starts"]).astype(np.int64)
    ends = _col(row["token_chunk_ends"]).astype(np.int64)
    kinds = _col(row["token_chunk_kinds"]).astype(np.int64)
    deps = _col(row["token_chunk_dep_levels"]).astype(np.int64)
    n = len(starts)
    if not (len(ends) == len(kinds) == len(deps) == n):
        raise ValueError(
            f"{source}[row={row_index}]: chunk channels have inconsistent lengths "
            f"starts={len(starts)} ends={len(ends)} kinds={len(kinds)} deps={len(deps)}"
        )
    keep = [i for i in range(n) if int(ends[i]) <= vtc and int(starts[i]) < int(ends[i])]
    if not keep:
        # No usable chunk in the prefix — omit chunk channels; AST-FIM will RAISE
        # (its required precondition), and the mixer's re-draw handles it.
        return {}
    idx = np.asarray(keep, dtype=np.int64)
    return {
        "chunk_starts": _i32(starts[idx]),
        "chunk_ends": _i32(ends[idx]),
        "chunk_kinds": _i32(kinds[idx]),
        "chunk_dep_levels": _i32(deps[idx]),
    }


def _platform_ids(row, *, vocab_safe: bool) -> mx.array | None:
    raw = row.get("platform_ids")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        return None
    # Packed per-document platform ids -> (B=1, K) document-level channel.
    return mx.array(arr[None, :])


def load_commit_packets(
    commits_path: Path,
    *,
    vocab_size: int,
) -> list[CommitPacket]:
    """Synthesize CommitPackets by pairing consecutive pre/post commit rows.

    ``golden_mini/commits`` stores tokenized before/after pairs of the SAME file
    on consecutive rows (row 2k = pre, row 2k+1 = post). We pair them into one
    CommitPacket carrying:
      * ``pre_token_ids``  = pre row token_ids
      * ``post_token_ids`` = post row token_ids
      * ``diff_token_ids`` = post row token_ids (the edited content to predict)
      * ``commit_msg``     = a short synthetic prompt: [BOS, CODE_START]
    so ``build_commit_diff`` / ``build_pre_to_post`` have real, non-empty fields.
    These objectives splice/synthesize the stream, so NO token-aligned
    side-channels travel with the CommitPacket (alignment rule).
    """

    table = pq.read_table(str(commits_path))
    rows = table.to_pylist()
    if len(rows) < 2:
        raise ValueError(
            f"{commits_path.name}: need >=2 commit rows to form a pre/post pair, "
            f"got {len(rows)}"
        )

    def toks(row, idx) -> np.ndarray:
        t = np.asarray(row["token_ids"], dtype=np.int64)
        if t.size < 2:
            raise ValueError(
                f"{commits_path.name}[row={idx}]: token_ids too short ({t.size})"
            )
        if t.min() < 0 or t.max() >= vocab_size:
            raise ValueError(
                f"{commits_path.name}[row={idx}]: token id out of range "
                f"[0,{vocab_size}); got [{int(t.min())},{int(t.max())}]"
            )
        return t

    packets: list[CommitPacket] = []
    msg = _i32([_SPECIAL.fim_prefix - 2, 7])  # [BOS=2, CODE_START=7] synthetic prompt
    for k in range(0, len(rows) - 1, 2):
        pre_row, post_row = rows[k], rows[k + 1]
        pre = toks(pre_row, k)
        post = toks(post_row, k + 1)
        packets.append(
            CommitPacket(
                pre_token_ids=_i32(pre),
                post_token_ids=_i32(post),
                diff_token_ids=_i32(post),
                commit_msg=msg,
                repo=str(pre_row.get("repo"))
                if pre_row.get("repo") is not None
                else None,
                filepath=str(pre_row.get("filepath"))
                if pre_row.get("filepath") is not None
                else None,
                metadata={"pre_row": k, "post_row": k + 1},
            )
        )
    if not packets:
        raise ValueError(f"no commit packets paired from {commits_path!r}")
    return packets


# --------------------------------------------------------------------------- #
# Side-channel alignment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainStep:
    """One materialized training step (a batch of one example)."""

    objective: str
    input_ids: mx.array  # (1, S)
    target_ids: mx.array  # (1, S)
    loss_mask: mx.array  # (1, S)
    side_channels: dict[str, mx.array]  # aligned channels, or {} when disabled


def _assert_aligned(name: str, channel: mx.array, seq_len: int) -> mx.array:
    """Slice a packet channel to the input length and assert exact alignment."""

    arr = channel
    if arr.ndim != 1:
        raise ValueError(f"{name}: expected 1-D channel, got shape {tuple(arr.shape)}")
    # causal/recovery input_ids == token_ids[:-1]; the channel must cover that.
    if int(arr.shape[0]) < seq_len:
        raise ValueError(
            f"{name}: channel length {int(arr.shape[0])} < input length {seq_len}; "
            "cannot align side-channel"
        )
    sliced = arr[:seq_len]
    if int(sliced.shape[0]) != seq_len:
        raise ValueError(
            f"{name}: aligned length {int(sliced.shape[0])} != input length {seq_len}"
        )
    return sliced[None, :]


def build_train_step(
    objective: str,
    example: ObjectiveExample,
    packet: CodePacket | CommitPacket,
) -> TrainStep:
    """Materialize a TrainStep, applying the side-channel alignment rule.

    For ALIGNED objectives we slice the CodePacket's token-aligned channels to the
    example's input length and pass them (asserting alignment). For REORDERED /
    SYNTHESIZED objectives we pass NO channels (side-channels disabled).
    """

    seq_len = int(example.input_ids.shape[0])
    input_ids = example.input_ids[None, :]
    target_ids = example.target_ids[None, :]
    loss_mask = example.loss_mask[None, :]

    if objective in _ALIGNED_OBJECTIVES:
        if not isinstance(packet, CodePacket):
            raise TypeError(
                f"aligned objective {objective!r} requires a CodePacket, got "
                f"{type(packet).__name__}"
            )
        channels: dict[str, mx.array] = {}
        channel_sources = {
            "structure_ids": packet.structure_ids,
            "dep_levels": packet.dep_levels,
            "ast_depth_ids": packet.ast_depth,
            "sibling_index_ids": packet.sibling_index,
            "node_type_ids": packet.ast_node_type,
        }
        for model_kw, source in channel_sources.items():
            if source is None:
                raise ValueError(
                    f"aligned objective {objective!r}: CodePacket.{model_kw} is "
                    "absent; cannot pass an aligned structure channel"
                )
            channels[model_kw] = _assert_aligned(model_kw, source, seq_len)
        platform = packet.metadata.get("platform_ids") if packet.metadata else None
        if platform is not None:
            channels["platform_ids"] = platform  # (1, K) document-level
        return TrainStep(objective, input_ids, target_ids, loss_mask, channels)

    if objective in _REORDERED_OBJECTIVES:
        # Side-channels DISABLED: pass nothing. The model is also driven with the
        # residual scales zeroed for this step (see run_training).
        return TrainStep(objective, input_ids, target_ids, loss_mask, {})

    raise ValueError(f"unknown objective {objective!r}: cannot classify alignment")


# --------------------------------------------------------------------------- #
# Model + training
# --------------------------------------------------------------------------- #
def smoke_config(vocab_size: int) -> DenseCppLMConfig:
    """Tiny, fast DenseCppLM profile for the smoke (d=256, depth=4)."""

    return DenseCppLMConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        depth=4,
        ffn_hidden_size=512,
        max_seq_length=4096,
        num_query_heads=8,
        num_kv_heads=2,
        head_dim=32,
        attention_mode="gqa",
        ngram_hash_table_size=50_000,
    )


def _loss_for_step(
    model: DenseCppLM, step: TrainStep, *, channels_on: bool
) -> mx.array:
    """Forward one step, honoring the side-channel on/off contract.

    When ``channels_on`` is False the residual scales are temporarily zeroed AND
    no channels are passed, so a reordered/synthesized step can NEVER fold a
    misaligned side-channel into the stream.
    """

    if channels_on:
        _, loss = model(
            step.input_ids,
            targets=step.target_ids,
            loss_mask=step.loss_mask,
            **step.side_channels,
        )
        return loss

    cfg = model.config
    saved = (
        cfg.structure_residual_scale,
        cfg.platform_residual_scale,
        cfg.ngram_residual_scale,
    )
    object.__setattr__(model, "config", replace(
        cfg,
        structure_residual_scale=0.0,
        platform_residual_scale=0.0,
        ngram_residual_scale=0.0,
    ))
    try:
        _, loss = model(
            step.input_ids,
            targets=step.target_ids,
            loss_mask=step.loss_mask,
        )
    finally:
        object.__setattr__(model, "config", replace(
            cfg,
            structure_residual_scale=saved[0],
            platform_residual_scale=saved[1],
            ngram_residual_scale=saved[2],
        ))
    return loss


def materialize_steps(
    mixer: TaskMixer,
    code_packets: list[CodePacket],
    commit_packets: list[CommitPacket],
    *,
    num_steps: int,
) -> list[TrainStep]:
    """Draw an objective per step and materialize a TrainStep deterministically.

    Code-family tasks consume a CodePacket; commit-family tasks a CommitPacket.
    Packets cycle so a small fixture supports an arbitrary step budget. Steps
    that the fixture cannot satisfy (e.g. a recovery channel with no span on a
    particular packet) are re-drawn at the NEXT step index — the mixer stays
    deterministic and we never silently fabricate a span.
    """

    steps: list[TrainStep] = []
    code_i = 0
    commit_i = 0
    step_index = 0
    attempts = 0
    max_attempts = num_steps * 50
    while len(steps) < num_steps:
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not materialize {num_steps} steps after {attempts} draws; "
                "fixture too small for the drawn objective mix"
            )
        attempts += 1
        task = mixer.draw_task(step_index)
        rng = mixer._step_rng(step_index)
        step_index += 1
        if task in (TaskKind.COMMIT_DIFF, TaskKind.PRE_TO_POST):
            if not commit_packets:
                # No commit fixture present — this objective is unavailable for the
                # whole run; advance to the next deterministic draw.
                continue
            packet = commit_packets[commit_i % len(commit_packets)]
            commit_i += 1
        else:
            packet = code_packets[code_i % len(code_packets)]
            code_i += 1
        try:
            example = mixer.build(task, packet, rng=rng)
        except ValueError as exc:
            # Only an EXPECTED data limitation may be skipped: a recovery channel
            # with no non-zero span on this packet, or a packet lacking clang
            # chunk boundaries for AST-FIM. Anything else is a real bug -> RAISE.
            if not _is_expected_unsatisfiable(exc):
                raise
            continue
        steps.append(build_train_step(example.objective, example, packet))
    return steps


_EXPECTED_UNSATISFIABLE_MARKERS = (
    "no non-zero recoverable span",
    "yields no supervised target",
    "requires CodePacket.chunk_starts and chunk_ends",
    "channel is absent",
)


def _is_expected_unsatisfiable(exc: Exception) -> bool:
    """True only for data-limitation skips (empty span / missing chunks)."""

    text = str(exc)
    return any(marker in text for marker in _EXPECTED_UNSATISFIABLE_MARKERS)


def run_training(
    *,
    num_steps: int = 220,
    seed: int = 1234,
    lr: float = 3.0e-4,
    extra_source: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the Stage-1 mixed-objective smoke and return a results dict."""

    vocab_size = 65536
    code_paths = sorted((GOLDEN_MINI / "code").glob("*.parquet"))
    if not code_paths:
        raise FileNotFoundError(f"no code parquet under {GOLDEN_MINI / 'code'}")
    if extra_source is not None:
        extra = sorted(Path(extra_source).glob("*.parquet"))[:1]
        code_paths = code_paths + extra

    code_packets = load_code_packets(code_paths, vocab_size=vocab_size)
    commits_path = GOLDEN_MINI / "commits" / "commits.parquet"
    commit_packets = (
        load_commit_packets(commits_path, vocab_size=vocab_size)
        if commits_path.exists()
        else []
    )

    # Stage-1 default mix, with the IFIM rate folded into AST_FIM. IFIM needs a
    # per-document leading comment/docstring in metadata['source_text']; the
    # packed golden_mini rows do not carry decoded source text, so IFIM is
    # genuinely unavailable on THIS fixture. We fold its 0.1 into ast_fim (the
    # FIM bucket total is preserved) rather than letting the mixer draw an
    # objective the fixture cannot satisfy. This is explicit + config-driven, not
    # a silent skip. (A real run with source_text restores the default split.)
    rates = {
        TaskKind.CAUSAL_LM: 0.5,
        TaskKind.AST_FIM: 0.2,  # 0.1 ast_fim + 0.1 folded-in ifim
        TaskKind.COMMIT_DIFF: 0.1,
        TaskKind.PRE_TO_POST: 0.1,
        TaskKind.SYMBOL_RECOVERY: 0.1 / 3.0,
        TaskKind.TYPE_RECOVERY: 0.1 / 3.0,
        TaskKind.CALLEE_RECOVERY: 0.1 / 3.0,
    }
    mixer = TaskMixer(rates, seed=seed, special_token_ids=_SPECIAL)

    steps = materialize_steps(
        mixer, code_packets, commit_packets, num_steps=num_steps
    )

    dist = Counter(s.objective for s in steps)
    aligned_steps = sum(1 for s in steps if s.objective in _ALIGNED_OBJECTIVES)
    reordered_steps = sum(1 for s in steps if s.objective in _REORDERED_OBJECTIVES)

    model = smoke_config(vocab_size)
    model = DenseCppLM(model)
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.1, betas=(0.9, 0.95))

    def step_loss(model: DenseCppLM, step: TrainStep) -> mx.array:
        return _loss_for_step(
            model, step, channels_on=step.objective in _ALIGNED_OBJECTIVES
        )

    loss_and_grad = nn.value_and_grad(model, step_loss)

    per_obj_losses: dict[str, list[float]] = {k: [] for k in dist}
    curve: list[float] = []
    window: list[float] = []

    for i, step in enumerate(steps):
        loss, grads = loss_and_grad(model, step)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        lv = float(loss.item())
        if not math.isfinite(lv):
            raise FloatingPointError(
                f"step {i} ({step.objective}) produced non-finite loss {lv}"
            )
        per_obj_losses[step.objective].append(lv)
        window.append(lv)
        if (i + 1) % max(1, num_steps // 20) == 0 or i == 0:
            recent = float(np.mean(window[-25:]))
            curve.append((i + 1, recent))
            if verbose:
                print(f"  step {i + 1:4d}/{num_steps}  loss(ma25)={recent:.4f}")

    # Compare the first vs last loss "epoch" (a window proportional to the run)
    # so the learn-down assertion is robust to per-step objective variance.
    epoch = max(5, num_steps // 20)
    initial = float(np.mean(window[:epoch]))
    final = float(np.mean(window[-epoch:]))

    results = {
        "num_steps": num_steps,
        "distribution": dict(dist),
        "aligned_steps": aligned_steps,
        "reordered_steps": reordered_steps,
        "initial_loss": initial,
        "final_loss": final,
        "curve": curve,
        "per_objective_mean": {
            k: float(np.mean(v)) for k, v in per_obj_losses.items() if v
        },
    }

    if verbose:
        print("\n=== Stage-1 smoke results ===")
        print("Per-objective step distribution:")
        for k in sorted(dist):
            on = "channels ON " if k in _ALIGNED_OBJECTIVES else "channels OFF"
            print(f"  {k:18s} {dist[k]:4d} steps  [{on}]  "
                  f"mean_loss={results['per_objective_mean'].get(k, float('nan')):.4f}")
        print(f"Aligned (channels-on) steps : {aligned_steps}")
        print(f"Reordered (channels-off)    : {reordered_steps}")
        print(f"Loss curve (initial -> final): {initial:.4f} -> {final:.4f}")

    if final >= initial:
        raise AssertionError(
            f"Stage-1 mixed-objective model did NOT learn: final loss {final:.4f} "
            f">= initial loss {initial:.4f}"
        )
    if aligned_steps == 0:
        raise AssertionError("no side-channel-ON (causal/recovery) steps occurred")
    if reordered_steps == 0:
        raise AssertionError("no side-channel-OFF (fim/commit) steps occurred")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-1 mixed-objective smoke.")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument(
        "--extra-source",
        type=str,
        default=None,
        help="optional dir with a real clang_semantic_4k_v10 shard (one *.parquet)",
    )
    args = parser.parse_args()
    extra = Path(args.extra_source) if args.extra_source else None
    run_training(
        num_steps=args.steps, seed=args.seed, lr=args.lr, extra_source=extra
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
