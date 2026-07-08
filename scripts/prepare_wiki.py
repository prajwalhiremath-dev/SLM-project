from __future__ import annotations

import argparse
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\t", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and clean Wikipedia text shards.")
    parser.add_argument("--dataset-name", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", type=str, default="20231101.en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--target-bytes", type=int, default=50_000_000)
    parser.add_argument("--shard-bytes", type=int, default=100_000_000)
    parser.add_argument("--output-dir", type=str, default="data/cleaned")
    parser.add_argument("--min-chars", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )

    total_bytes = 0
    shard_bytes = 0
    shard_idx = 0
    article_count = 0

    current_path = output_dir / f"wiki_{shard_idx:04d}.txt"
    current_file = current_path.open("w", encoding="utf-8")

    progress = tqdm(total=args.target_bytes, unit="B", unit_scale=True, desc="cleaned bytes")

    try:
        for row in dataset:
            raw_text = row.get("text", "")
            text = clean_text(raw_text)
            if len(text) < args.min_chars:
                continue

            line = text + "\n"
            encoded = line.encode("utf-8")
            n_bytes = len(encoded)

            if shard_bytes + n_bytes > args.shard_bytes and shard_bytes > 0:
                current_file.close()
                shard_idx += 1
                shard_bytes = 0
                current_path = output_dir / f"wiki_{shard_idx:04d}.txt"
                current_file = current_path.open("w", encoding="utf-8")

            current_file.write(line)
            total_bytes += n_bytes
            shard_bytes += n_bytes
            article_count += 1
            progress.update(n_bytes)

            if total_bytes >= args.target_bytes:
                break
    finally:
        progress.close()
        current_file.close()

    print(f"Done. Articles: {article_count:,}")
    print(f"Cleaned bytes: {total_bytes:,}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
