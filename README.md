# YamLLM

**YamLLM** is a professional, scratch-built Large Language Model ecosystem using PyTorch.

## Architecture

- **Tokenizer**: Custom BPE Tokenizer
- **Model**: GPT-style Decoder-only Transformer
  - Multi-Head Causal Self-Attention
  - LayerNormalizations and FeedForward Networks
- **Training**: Custom training loop with Distributed Data Parallel (DDP) support for scaling across multiple GPUs.

## Project Structure

- `src/`: Core implementation.
  - `tokenizer/`: Byte-Pair Encoding logic.
  - `model/`: Neural network layers and main transformer.
  - `data/`: Datasets and loaders.
  - `training/`: Training loop and optimizers.
  - `evaluation/`: Metrics and validation routines.
  - `utils/`: Configurations and helpers.
- `scripts/`: Entrypoints for training, evaluation, and API server.
- `tests/`: Automated unit and integration tests.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start training (example):
   ```bash
   python scripts/train.py
   ```
3. Run as an API Server (Like Ollama/vLLM):
   ```bash
   python scripts/api.py
   ```
   *Sends a POST request to `http://localhost:8000/v1/completions` with JSON body `{"prompt": "Hello", "max_tokens": 50}`*
