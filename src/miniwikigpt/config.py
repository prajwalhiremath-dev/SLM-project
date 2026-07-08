from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class GPTConfig:
    """Configuration for the decoder-only Transformer."""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


@dataclass
class TrainConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 50_000
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 1_000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 1337
    use_amp: bool = True
    compile: bool = False


@dataclass
class ProjectConfig:
    data_dir: str
    tokenizer_path: str
    out_dir: str
    model: GPTConfig
    train: TrainConfig


def load_project_config(config_path: str | Path, vocab_size: int) -> ProjectConfig:
    """Load JSON config and inject tokenizer vocab_size into GPTConfig."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    model_kwargs = dict(raw["model"])
    model_kwargs["vocab_size"] = int(vocab_size)

    return ProjectConfig(
        data_dir=raw["data_dir"],
        tokenizer_path=raw["tokenizer_path"],
        out_dir=raw["out_dir"],
        model=GPTConfig(**model_kwargs),
        train=TrainConfig(**raw["train"]),
    )
