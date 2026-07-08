from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return Tokenizer.from_file(str(path))


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode_ids(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def get_vocab_size(tokenizer: Tokenizer) -> int:
    return tokenizer.get_vocab_size()
