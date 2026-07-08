from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize cleaned text into train.bin and val.bin.")
    parser.add_argument("--input-dir", type=str, default="data/cleaned")
    parser.add_argument("--tokenizer-path", type=str, default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="data/tokenized")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-chars", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    train_tokens = 0
    val_tokens = 0
    documents = 0

    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        for txt_file in txt_files:
            with txt_file.open("r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"tokenizing {txt_file.name}"):
                    line = line.strip()
                    if len(line) < args.min_chars:
                        continue

                    ids = tokenizer.encode(line + "\n").ids
                    arr = np.array(ids, dtype=dtype)

                    if random.random() < args.val_ratio:
                        arr.tofile(val_f)
                        val_tokens += len(arr)
                    else:
                        arr.tofile(train_f)
                        train_tokens += len(arr)

                    documents += 1

    meta = {
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "documents": documents,
        "val_ratio": args.val_ratio,
    }

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Tokenization complete.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
