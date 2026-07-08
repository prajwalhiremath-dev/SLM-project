# MiniWikiGPT: Small Language Model from Wikipedia

MiniWikiGPT is an end-to-end project for training a small GPT-style language model from scratch using Wikipedia text and PyTorch.

The project has two phases:

1. **Phase 1 — Training**
   - Download and clean Wikipedia text.
   - Train a BPE tokenizer.
   - Tokenize the cleaned text into binary training files.
   - Train a decoder-only Transformer using PyTorch/CUDA.
   - Generate text from saved checkpoints.

2. **Phase 2 — Deployment**
   - Serve the trained model using FastAPI.
   - Run an interactive text-generation UI using Streamlit.

---

## Project Structure

```text
SLM-project/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── configs/
│   └── train_small.json
├── data/
│   └── .gitkeep
├── scripts/
│   ├── prepare_wiki.py
│   ├── train_tokenizer.py
│   ├── tokenize_data.py
│   ├── train.py
│   └── generate.py
├── src/
│   └── miniwikigpt/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── inference.py
│       ├── model.py
│       └── tokenizer_utils.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 1. Setup

```bash
pip install -r requirements.txt
```

For Google Colab, mount Google Drive so checkpoints survive runtime disconnects:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 2. Prepare Wikipedia Data

Start small first to validate the full pipeline.

```bash
python scripts/prepare_wiki.py \
  --target-bytes 50000000 \
  --output-dir data/cleaned
```

For a 1 GB run:

```bash
python scripts/prepare_wiki.py \
  --target-bytes 1000000000 \
  --output-dir data/cleaned
```

This creates cleaned `.txt` shards.

---

## 3. Train BPE Tokenizer

```bash
python scripts/train_tokenizer.py \
  --input-dir data/cleaned \
  --output-dir artifacts/tokenizer \
  --vocab-size 16000
```

---

## 4. Tokenize Dataset

```bash
python scripts/tokenize_data.py \
  --input-dir data/cleaned \
  --tokenizer-path artifacts/tokenizer/tokenizer.json \
  --output-dir data/tokenized \
  --val-ratio 0.01
```

This creates:

```text
data/tokenized/train.bin
data/tokenized/val.bin
data/tokenized/meta.json
```

---

## 5. Train the SLM

```bash
python scripts/train.py \
  --config configs/train_small.json
```

Checkpoints are saved to:

```text
checkpoints/
```

Recommended for Colab: save `out_dir` inside Google Drive by editing `configs/train_small.json`.

Example:

```json
"out_dir": "/content/drive/MyDrive/MiniWikiGPT/checkpoints"
```

---

## 6. Generate Text

```bash
python scripts/generate.py \
  --checkpoint checkpoints/latest.pt \
  --tokenizer-path artifacts/tokenizer/tokenizer.json \
  --prompt "Artificial intelligence is" \
  --max-new-tokens 100
```

---

## 7. Run FastAPI

```bash
export CHECKPOINT_PATH=checkpoints/latest.pt
export TOKENIZER_PATH=artifacts/tokenizer/tokenizer.json
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Machine learning is", "max_new_tokens":100, "temperature":0.8}'
```

---

## 8. Run Streamlit

```bash
export CHECKPOINT_PATH=checkpoints/latest.pt
export TOKENIZER_PATH=artifacts/tokenizer/tokenizer.json
streamlit run app/streamlit_app.py
```

---

## Recommended 1 GB Training Config

For Colab, use this starting setup:

```text
Dataset: 1 GB cleaned Wikipedia text
Tokenizer: BPE
Vocab size: 16,000
Context length: 256
Layers: 6
Heads: 6
Embedding size: 384
Batch size: 16
Gradient accumulation: 4
Mixed precision: enabled
```

---

## Resume Bullet

```text
Built MiniWikiGPT, a GPT-style Small Language Model trained from scratch on Wikipedia data using PyTorch/CUDA, with custom BPE tokenization, causal self-attention, mixed-precision training, checkpointing, and FastAPI/Streamlit deployment.
```
