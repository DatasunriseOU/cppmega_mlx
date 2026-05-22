"""V7-H05: per-step training event extraction from stage_train extras.

Produces a stream of {step, loss, lr, grad_norm, mem_mb,
throughput_tok_s, ts} events derived from finalised extras. Until
stage_train emits live callbacks (deeper rewrite), the UI can render
post-hoc per-step events the moment the modal opens — same payload
shape a real /ws/train/{job_id} stream will eventually push.

Usage:
  events = list(train_events_from_extras(extras, batch=1, seq=16))
  for e in events:
      ui.append_row(e)
"""

from __future__ import annotations

import time
from typing import Iterator


def train_events_from_extras(extras: dict, *,
                              batch: int = 1, seq: int = 16,
                              start_ts: float | None = None,
                              ) -> Iterator[dict]:
    """Yield per-step training events from stage_train extras."""
    losses = list(extras.get("losses", []))
    lrs = list(extras.get("lr_trajectory", []))
    elapsed_ms = float(extras.get("elapsed_ms", 0.0)) or 1.0
    n = max(1, len(losses))
    per_step_ms = elapsed_ms / n
    tokens_per_step = max(1, batch * seq)
    throughput = tokens_per_step / max(per_step_ms / 1000.0, 1e-6)
    base_ts = start_ts if start_ts is not None else time.time()
    mem_peak = extras.get("memory_peak_bytes")
    mem_mb = (round(int(mem_peak) / (1024 * 1024), 4)
              if mem_peak else None)
    for i, loss in enumerate(losses):
        lr = float(lrs[i]) if i < len(lrs) else None
        yield {
            "step": i,
            "loss": float(loss),
            "lr": lr,
            "grad_norm": None,  # not snapshotted per-step today
            "mem_mb": mem_mb,
            "throughput_tok_s": round(throughput, 4),
            "ts": round(base_ts + (i + 1) * per_step_ms / 1000.0, 6),
        }


__all__ = ["train_events_from_extras"]
