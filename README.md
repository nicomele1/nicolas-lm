# NicolasLM

`nicolas-lm` is a small research-oriented language modeling lab. It focuses on
character-level autoregressive models and on the tools needed to train,
evaluate, and analyze them in a controlled setting.

The current codebase includes:

- a character tokenizer and corpus utilities;
- a bigram language model;
- a decoder-only Transformer;
- a LLaMA-style decoder with RMSNorm, RoPE, and SwiGLU;
- corpus statistics and evaluation scripts;
- experiment helpers for corpus building, training, sampling, and result
  summarization.

## Installation

The project targets Python 3.10+.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Project layout

```text
src/nicolasm/        Core library code
scripts/             Training, evaluation, and analysis entry points
tests/               Unit tests for tokenizer, models, and metrics
configs/             Model and experiment configuration files
data/                Local corpora, tokenizers, and PDF references
experiments/         Run artifacts, plots, and reports
docs/                Project documentation and reading notes
```

## Typical workflow

Build corpora and tokenizers:

```bash
PYTHONPATH=src python scripts/build_corpora.py
PYTHONPATH=src python scripts/train.py
```

Train or evaluate a model:

```bash
PYTHONPATH=src python scripts/train.py
PYTHONPATH=src python scripts/evaluate.py
```

Analyze a corpus or summarize results:

```bash
PYTHONPATH=src python scripts/analyze_corpus.py
PYTHONPATH=src python scripts/run_effective_tokens.py
PYTHONPATH=src python scripts/summarize_results.py
```

Run the tests:

```bash
pytest
```

## Models

The repository currently centers on three language-model families:

- `BigramLanguageModel`: a first-order categorical model over tokens.
- `TinyTransformerLanguageModel`: a causal Transformer decoder.
- `LLaMAStyleLanguageModel`: a Transformer variant with LLaMA-style
  design choices.

Each model is implemented in `src/nicolasm/models/` and built from reusable
modules in `src/nicolasm/modules/`.
