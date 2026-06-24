#!/usr/bin/env python3
"""LoRA SFT for the per-expert adapters (GPU box; DO NOT run the 4B here).

Config-driven LoRA supervised fine-tuning over the JSONL datasets produced by
``scripts/build_expert_sft_data.py``. One adapter per expert; all share the base
Qwen3-4B-Instruct-2507 and its chat template.

    python scripts/train_expert_lora.py \
        --expert tool_router \
        --base Qwen/Qwen3-4B-Instruct-2507 \
        --data outputs/expert_sft/tool_router.jsonl \
        --out outputs/adapters/tool_router

Backends
--------
* ``--backend hf`` (default): HuggingFace ``transformers`` + ``peft`` + ``trl``
  ``SFTTrainer`` -- the standard CUDA LoRA path. Targets an NVIDIA GPU.
* ``--backend mlx``: ``mlx_lm.lora`` for Apple-silicon LoRA.

VRAM / hardware (4B base, bf16 + LoRA, r=16):
* Full fine-tune of 4B is ~64+ GB; LoRA keeps base frozen so trainable params
  are tiny. bf16 base weights ~8 GB + optimizer/activations.
* Practical: >=24 GB VRAM (RTX 4090 / A10) at seq_len 2048, micro-batch 1-2 with
  grad-accum. 16 GB works at seq_len<=1024 + grad-checkpointing. The cheap tier
  (Qwen3-1.7B) fits comfortably in 12-16 GB.

This script REFUSES to start a real 4B training run unless ``--allow-train`` is
passed (so it is safe to import / arg-parse / dry-run on a laptop). Per RULE #1
it fails loud on a missing dataset, an empty dataset, or a missing backend dep.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXPERTS = ("tool_router", "buildops", "sql")


@dataclass
class TrainConfig:
    expert: str
    base: str
    data: str
    out: str
    backend: str = "hf"
    epochs: float = 3.0
    lr: float = 2.0e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seq_len: int = 2048
    micro_batch: int = 1
    grad_accum: int = 16
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )
    seed: int = 0

    def validate(self) -> None:
        if self.expert not in EXPERTS:
            raise ValueError(f"--expert must be one of {EXPERTS}, got {self.expert!r}")
        if self.backend not in ("hf", "mlx"):
            raise ValueError(f"--backend must be hf|mlx, got {self.backend!r}")
        p = Path(self.data)
        if not p.exists():
            raise FileNotFoundError(f"--data {self.data!r} does not exist (fail-loud)")
        n = sum(1 for _ in p.open("r", encoding="utf-8"))
        if n == 0:
            raise ValueError(f"--data {self.data!r} is empty; nothing to train on")


def _count_examples(path: str) -> int:
    return sum(1 for _ in Path(path).open("r", encoding="utf-8"))


def train_hf(cfg: TrainConfig) -> None:  # pragma: no cover - GPU box only
    try:
        import torch  # noqa: F401
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            "HF LoRA backend needs torch+transformers+peft+trl+datasets: " f"{exc}"
        ) from exc

    tok = AutoTokenizer.from_pretrained(cfg.base)
    model = AutoModelForCausalLM.from_pretrained(cfg.base, torch_dtype="bfloat16")
    ds = load_dataset("json", data_files=cfg.data, split="train")

    def fmt(ex: dict[str, Any]) -> dict[str, str]:
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = ds.map(fmt, remove_columns=ds.column_names)
    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM",
    )
    sft = SFTConfig(
        output_dir=cfg.out, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.micro_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        max_seq_length=cfg.seq_len, gradient_checkpointing=True,
        bf16=True, logging_steps=10, save_strategy="epoch", seed=cfg.seed,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(cfg.out)
    tok.save_pretrained(cfg.out)


def train_mlx(cfg: TrainConfig) -> None:  # pragma: no cover - GPU/Apple box only
    import subprocess

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", cfg.base, "--train",
        "--data", str(Path(cfg.data).parent),
        "--adapter-path", cfg.out,
        "--iters", str(int(cfg.epochs * max(1, _count_examples(cfg.data)))),
        "--learning-rate", str(cfg.lr),
        "--num-layers", "16",
        "--batch-size", str(cfg.micro_batch),
        "--max-seq-length", str(cfg.seq_len),
    ]
    subprocess.run(cmd, check=True)


def run(cfg: TrainConfig, *, allow_train: bool) -> dict[str, Any]:
    cfg.validate()
    n = _count_examples(cfg.data)
    plan = {
        "expert": cfg.expert, "base": cfg.base, "backend": cfg.backend,
        "data": cfg.data, "out": cfg.out, "examples": n,
        "epochs": cfg.epochs, "lr": cfg.lr,
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout},
        "seq_len": cfg.seq_len, "micro_batch": cfg.micro_batch,
        "grad_accum": cfg.grad_accum,
        "target_modules": list(cfg.target_modules),
    }
    if not allow_train:
        plan["dry_run"] = True
        plan["note"] = (
            "DRY RUN: pass --allow-train on the GPU box to start training. The 4B "
            "base must NOT be trained on this laptop."
        )
        return plan
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    if cfg.backend == "hf":
        train_hf(cfg)
    else:
        train_mlx(cfg)
    plan["dry_run"] = False
    plan["status"] = "completed"
    return plan


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert", required=True, choices=EXPERTS)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", required=True, help="expert SFT jsonl")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--backend", default="hf", choices=("hf", "mlx"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--allow-train",
        action="store_true",
        help="actually run training (GPU box). Without it this is a dry-run.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        expert=args.expert, base=args.base, data=args.data, out=args.out,
        backend=args.backend, epochs=args.epochs, lr=args.lr,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, seq_len=args.seq_len,
        micro_batch=args.micro_batch, grad_accum=args.grad_accum, seed=args.seed,
    )
    plan = run(cfg, allow_train=args.allow_train)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
