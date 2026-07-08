from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miniwikigpt.inference import generate_text, load_generation_model


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=500)


class GenerateResponse(BaseModel):
    generated_text: str


app = FastAPI(title="MiniWikiGPT API", version="0.1.0")


@lru_cache(maxsize=1)
def get_model_bundle():
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/latest.pt")
    tokenizer_path = os.getenv("TOKENIZER_PATH", "artifacts/tokenizer/tokenizer.json")
    return load_generation_model(checkpoint_path, tokenizer_path, device="auto")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    model, tokenizer, device = get_model_bundle()
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        device=device,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return GenerateResponse(generated_text=text)
