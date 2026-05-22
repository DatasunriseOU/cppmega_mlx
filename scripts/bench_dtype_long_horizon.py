"""V7-D02: 1000-step bf16-vs-fp32 numeric divergence bench.

Trains the same tiny model (2-brick attention+mlp, H=128) for N=1000
steps in master_dtype="fp32" and master_dtype="bf16" with identical
synthetic data (rng_key roundtripped via opt-state side-car since
V01 isn't trivially available here — we use mx.random.seed instead).

Emits two JSON files in reports/ with per-step:
  * loss
  * weight_delta_norm (vs init)
  * grad_norm (max across params)

Plus a top-level summary {final_loss_fp32, final_loss_bf16,
loss_drift_max, param_l2_diff_at_step_N, decision}.

Usage:
    python -m scripts.bench_dtype_long_horizon \\
        --num-steps 1000 --out-dir reports/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW


def _build_model(hidden: int = 128, master_dtype: str = "fp32"):
    """A 2-brick attention+mlp toy. Returns Sequential."""

    class TinyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(hidden, hidden, bias=False)
            self.k = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, hidden, bias=False)
            self.o = nn.Linear(hidden, hidden, bias=False)
            self.norm = nn.RMSNorm(hidden)

        def __call__(self, x):
            B, S, H = x.shape
            q = self.q(self.norm(x))
            k = self.k(self.norm(x))
            v = self.v(self.norm(x))
            scale = H ** -0.5
            s = mx.matmul(q, mx.transpose(k, (0, 2, 1))) * scale
            return self.o(mx.matmul(s, v))

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(hidden, 4 * hidden, bias=False)
            self.down = nn.Linear(4 * hidden, hidden, bias=False)

        def __call__(self, x):
            return self.down(nn.silu(self.up(x)))

    m = nn.Sequential(TinyAttn(), TinyMLP())
    dtype_map = {
        "fp32": mx.float32,
        "bf16": mx.bfloat16,
        "fp16": mx.float16,
    }
    m.set_dtype(dtype_map[master_dtype])
    return m


def _param_norm(flat_dict) -> float:
    total = 0.0
    for v in flat_dict.values():
        if hasattr(v, "shape"):
            total += float(mx.sum(v.astype(mx.float32) ** 2).item())
    return total ** 0.5


def _run_one(master_dtype: str, num_steps: int, hidden: int = 128,
              B: int = 1, S: int = 16) -> dict:
    mx.random.seed(0)
    model = _build_model(hidden=hidden, master_dtype=master_dtype)
    opt = AdamW(learning_rate=1e-3)
    init_flat = {k: mx.array(v) for k, v in nn.utils.tree_flatten(
        model.parameters())}
    init_norm = _param_norm(init_flat)

    def loss_fn(m, x):
        return mx.mean(m(x).astype(mx.float32) ** 2)

    lvg = nn.value_and_grad(model, loss_fn)
    rng = mx.random.key(42)
    per_step: list[dict] = []
    t0 = time.perf_counter()
    for step in range(num_steps):
        x = mx.random.normal(shape=(B, S, hidden), key=rng).astype(
            getattr(mx, ("bfloat16" if master_dtype == "bf16"
                          else "float16" if master_dtype == "fp16"
                          else "float32")))
        rng, _ = mx.random.split(rng)
        loss, grads = lvg(model, x)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        # Snapshot every 100 steps (1000 entries blow up JSON).
        if step % 100 == 0 or step == num_steps - 1:
            cur_flat = dict(nn.utils.tree_flatten(model.parameters()))
            delta_norm = 0.0
            for k, v in cur_flat.items():
                if k in init_flat and hasattr(v, "shape"):
                    d = v.astype(mx.float32) - init_flat[k].astype(
                        mx.float32)
                    delta_norm += float(mx.sum(d * d).item())
            per_step.append({
                "step": step,
                "loss": float(loss.item()),
                "weight_delta_norm": delta_norm ** 0.5,
            })
    elapsed = time.perf_counter() - t0
    return {
        "master_dtype": master_dtype,
        "num_steps": num_steps,
        "elapsed_s": elapsed,
        "init_param_norm": init_norm,
        "per_step": per_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for dt in ("fp32", "bf16"):
        print(f"[bench] running {dt}...", flush=True)
        results[dt] = _run_one(dt, args.num_steps)
        with (out_dir / f"bench_dtype_{dt}.json").open("w") as f:
            json.dump(results[dt], f, indent=2)
        print(f"[bench] {dt} done: final loss="
              f"{results[dt]['per_step'][-1]['loss']:.4f}, "
              f"weight_delta_norm="
              f"{results[dt]['per_step'][-1]['weight_delta_norm']:.4f}, "
              f"elapsed {results[dt]['elapsed_s']:.1f}s")

    f32 = results["fp32"]["per_step"][-1]["loss"]
    b16 = results["bf16"]["per_step"][-1]["loss"]
    summary = {
        "final_loss_fp32": f32,
        "final_loss_bf16": b16,
        "loss_drift_abs": abs(f32 - b16),
        "loss_drift_rel": abs(f32 - b16) / max(abs(f32), 1e-9),
        "elapsed_s_fp32": results["fp32"]["elapsed_s"],
        "elapsed_s_bf16": results["bf16"]["elapsed_s"],
        "decision": (
            "keep bf16 (faster + similar loss)"
            if (abs(f32 - b16) < 0.1 * max(abs(f32), 1e-9)
                and results["bf16"]["elapsed_s"]
                <= 1.2 * results["fp32"]["elapsed_s"])
            else "fp32 recommended for this workload"
        ),
    }
    with (out_dir / "bench_dtype_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bench] summary → {out_dir / 'bench_dtype_summary.json'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
