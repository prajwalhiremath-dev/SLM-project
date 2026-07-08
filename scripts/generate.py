from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miniwikigpt.inference import generate_text, load_generation_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a MiniWikiGPT checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--tokenizer-path", type=str, default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_generation_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
