from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

# Allow running from repository root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miniwikigpt.config import ProjectConfig, load_project_config
from miniwikigpt.data import BinBatcher
from miniwikigpt.model import GPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniWikiGPT.")
    parser.add_argument("--config", type=str, default="configs/train_small.json")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_meta(data_dir: str | Path) -> dict:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_lr(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)

    if step > cfg.max_steps:
        return cfg.min_lr

    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def estimate_loss(model: GPT, batcher: BinBatcher, ctx, eval_iters: int) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = batcher.get_batch(split)  # type: ignore[arg-type]
            with ctx:
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    out_dir: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    project_cfg: ProjectConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "model_config": raw_model.config.__dict__,
        "train_config": project_cfg.train.__dict__,
    }

    latest_path = out_dir / "latest.pt"
    step_path = out_dir / f"step_{step}.pt"
    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, step_path)
    print(f"Saved checkpoint: {latest_path}")


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    meta = load_meta(raw_config["data_dir"])

    # Load config after reading vocab size from tokenized metadata.
    project_cfg = load_project_config(args.config, vocab_size=meta["vocab_size"])
    train_cfg = project_cfg.train
    model_cfg = project_cfg.model

    device = resolve_device(train_cfg.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(train_cfg.seed)

    if device_type == "cuda":
        torch.cuda.manual_seed(train_cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    batcher = BinBatcher(
        data_dir=project_cfg.data_dir,
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=meta.get("dtype", "uint16"),
    )

    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=train_cfg.weight_decay,
        learning_rate=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        device_type=device_type,
    )

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        print(f"Resumed from {args.resume} at step {start_step}")

    if train_cfg.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)  # type: ignore[assignment]

    use_amp = train_cfg.use_amp and device_type == "cuda"
    dtype = torch.float16 if use_amp else torch.float32
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    scaler = make_grad_scaler(enabled=use_amp)

    out_dir = Path(project_cfg.out_dir)
    tokens_per_step = train_cfg.batch_size * model_cfg.block_size * train_cfg.gradient_accumulation_steps
    print(f"Device: {device}")
    print(f"Tokens per optimization step: {tokens_per_step:,}")

    t0 = time.time()
    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, batcher, ctx, train_cfg.eval_iters)
            elapsed = time.time() - t0
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"lr {lr:.2e}, "
                f"elapsed {elapsed/60:.1f} min"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(out_dir, model, optimizer, step, best_val_loss, project_cfg)

        optimizer.zero_grad(set_to_none=True)
        for _ in range(train_cfg.gradient_accumulation_steps):
            xb, yb = batcher.get_batch("train")
            with ctx:
                _, loss = model(xb, yb)
                loss = loss / train_cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()

        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if step > 0 and step % train_cfg.save_interval == 0:
            save_checkpoint(out_dir, model, optimizer, step, best_val_loss, project_cfg)

    print("Training complete.")
    save_checkpoint(out_dir, model, optimizer, train_cfg.max_steps, best_val_loss, project_cfg)


if __name__ == "__main__":
    main()
