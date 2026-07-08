from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer_utils import load_tokenizer


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # torch.compile can save keys prefixed with _orig_mod.
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_generation_model(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str = "auto",
) -> tuple[GPT, object, str]:
    device = resolve_device(device)
    tokenizer = load_tokenizer(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config)
    model.load_state_dict(_clean_state_dict_keys(checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    return model, tokenizer, device


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer: object,
    prompt: str,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
) -> str:
    encoded = tokenizer.encode(prompt).ids
    idx = torch.tensor(encoded, dtype=torch.long, device=device)[None, ...]
    output = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(output[0].tolist())
