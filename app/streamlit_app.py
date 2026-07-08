from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from miniwikigpt.inference import generate_text, load_generation_model


st.set_page_config(page_title="MiniWikiGPT", page_icon="🤖")
st.title("MiniWikiGPT")
st.caption("Small GPT-style language model trained on Wikipedia text.")


@st.cache_resource
def cached_model(checkpoint_path: str, tokenizer_path: str):
    return load_generation_model(checkpoint_path, tokenizer_path, device="auto")


checkpoint_path = st.sidebar.text_input(
    "Checkpoint path", os.getenv("CHECKPOINT_PATH", "checkpoints/latest.pt")
)
tokenizer_path = st.sidebar.text_input(
    "Tokenizer path", os.getenv("TOKENIZER_PATH", "artifacts/tokenizer/tokenizer.json")
)

prompt = st.text_area("Prompt", value="Artificial intelligence is", height=120)
max_new_tokens = st.slider("Max new tokens", min_value=10, max_value=512, value=100, step=10)
temperature = st.slider("Temperature", min_value=0.05, max_value=2.0, value=0.8, step=0.05)
top_k = st.slider("Top-k", min_value=1, max_value=500, value=50, step=1)

if st.button("Generate"):
    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
    elif not Path(tokenizer_path).exists():
        st.error(f"Tokenizer not found: {tokenizer_path}")
    else:
        with st.spinner("Generating..."):
            model, tokenizer, device = cached_model(checkpoint_path, tokenizer_path)
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        st.subheader("Output")
        st.write(output)
