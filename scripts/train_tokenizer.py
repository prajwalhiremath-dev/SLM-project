from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Byte-Level BPE tokenizer.")
    parser.add_argument("--input-dir", type=str, default="data/cleaned")
    parser.add_argument("--output-dir", type=str, default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(str(p) for p in input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )

    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    print(f"Tokenizer saved to: {tokenizer_json_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
