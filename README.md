# NicolasLM

NicolasLM is a small research project combining PyTorch implementations,
controlled experiments, a paper, and a mathematical companion to study
**corpus diversity and effective tokens** in character-level autoregressive
language models.

Interactive demo: deploy instructions in [`render.yaml`](render.yaml)
(serves the four trained model checkpoints from the pilot experiment)

---

## Research question

> At fixed raw character budget, can empirical corpus diversity behave like
> a proxy for effective training tokens in small autoregressive language models?

The project isolates one measurable component of the effective-token formulation
of Chang et al. (EMNLP 2024): empirical textual diversity, as captured by n-gram
entropy, distinct-n ratios, and gzip compression ratio. It tests whether the
corpus with higher diversity yields better generalization at equal raw corpus
size.

---

## Pilot result

Two corpora of 1,000,000 characters each — *Medium* (single author, Jane Austen)
and *High* (ten authors/genres, interleaved) — were trained on two architectures
for 5,000 AdamW steps.

**Corpus diversity metrics**

| Corpus | Vocab | H₃ (nats) | Distinct-4 | Gzip ratio |
|--------|------:|----------:|-----------:|-----------:|
| Medium | 94    | 7.4058    | 0.0378     | 0.3579     |
| High   | 102   | 7.5880    | 0.0592     | 0.4066     |

**Test evaluation**
(full results in [`experiments/results/effective_tokens_1M.csv`](experiments/results/effective_tokens_1M.csv))

| Model | Corpus | Test loss | PPL | Gen. gap |
|-------|--------|----------:|----:|---------:|
| Transformer | Medium | 1.8355 | 6.27 | 0.0676 |
| Transformer | High   | 1.9597 | 7.10 | 0.0398 |
| LLaMA-style | Medium | 1.4638 | 4.32 | 0.0993 |
| LLaMA-style | High   | 1.7010 | 5.48 | 0.0742 |

In this small-scale pilot, the higher-diversity corpus produced **higher test
loss** in both architectures but **lower generalization gap**. Diversity appears
to act as implicit regularization at this scale, reducing overfitting without
reducing absolute risk. This does not establish a general claim; it is a single
pilot with one seed per configuration.

---

## Paper and mathematical companion

- **Paper** (Spanish): [`notes/main.tex`](notes/main.tex) · [`notes/main.pdf`](notes/main.pdf)
  — formalizes tokenization, autoregressive modeling, empirical risk, AdamW,
  causal attention, RMSNorm, RoPE, and SwiGLU; reports the pilot results above.
- **Mathematical companion**: [`notes/mathematical_background.tex`](notes/mathematical_background.tex) · [`notes/mathematical_background.pdf`](notes/mathematical_background.pdf)
  — long-form treatment of the probability theory, information theory, and
  optimization theory underlying the experiments.
- **Experiment protocol**: [`docs/effective_tokens_protocol.md`](docs/effective_tokens_protocol.md)

---

## What is implemented

**Tokenization and data**
- Character-level tokenizer (`CharTokenizer`) with encode/decode
- Next-token prediction dataset (`TokenDataset`) with train/val/test splitting

**Model architectures**
- `BigramLanguageModel` — first-order Markov baseline
- `TinyTransformerLanguageModel` — causal decoder with learned positional
  embeddings, multi-head causal self-attention, pre-norm blocks, GELU FFN
- `LlamaStyleLanguageModel` — same decoder structure but with RMSNorm, RoPE,
  and SwiGLU instead of LayerNorm, learned positions, and GELU FFN

**Modules** (`src/nicolasm/modules/`)
- `causal_mask` — lower-triangular boolean causal attention mask
- `CausalSelfAttentionHead`, `MultiHeadCausalSelfAttention`
- `RMSNorm`
- `apply_rope` — rotary positional embeddings
- `SwiGLU`

**Corpus metrics** (`src/nicolasm/corpus_stats.py`)
- n-gram entropy H₁, H₂, H₃
- conditional bigram entropy
- distinct-n ratio
- gzip compression ratio
- `corpus_summary` combining all of the above

**Scripts** (`scripts/`)
- `train.py` — train any of the three models on a text corpus
- `evaluate.py` — compute test loss, perplexity, and generalization gap from a checkpoint
- `sample.py` — generate text with temperature and top-k sampling
- `analyze_corpus.py` — compute diversity metrics for one or more corpora
- `run_effective_tokens.py` — full experiment pipeline (corpus preparation, training, evaluation, results CSV)
- `summarize_results.py` — compact CSV and Markdown summaries from results
- `build_corpora.py` — download and prepare Gutenberg corpora for the pilot experiment
- `app.py` — Flask web interface serving the trained checkpoints (deployed on Railway)

---

## Repository layout

```text
src/nicolasm/        Core library (tokenizer, data, models, modules, corpus stats)
scripts/             Training, evaluation, analysis, and deployment entry points
tests/               Unit and property tests
data/                Corpora and raw text (large files are gitignored)
experiments/         Result CSVs, evaluation outputs, trained checkpoints
notes/               Paper (main.tex / main.pdf) and mathematical companion
docs/                Protocol documentation and literature notes
```

---

## Installation

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

## Quickstart

Run the test suite:

```bash
pytest
```

Analyze a corpus:

```bash
python scripts/analyze_corpus.py --input data/raw/input.txt
```

Train a small model (requires `data/raw/input.txt`):

```bash
python scripts/train.py --model-name transformer --max-steps 1000
```

Sample from a trained checkpoint:

```bash
python scripts/sample.py --model-name transformer --prompt "The "
```

Evaluate a trained checkpoint:

```bash
python scripts/evaluate.py --model-name transformer
```

Run the full effective-tokens pipeline on two corpora:

```bash
python scripts/run_effective_tokens.py \
  --input data/corpora/corpus_medium.txt --name medium \
  --input data/corpora/corpus_high.txt   --name high \
  --model-name transformer \
  --model-name llama \
  --max-chars 1000000 \
  --max-steps 5000 \
  --output experiments/results/effective_tokens.csv
```

Build the Gutenberg corpora used in the pilot experiment (requires internet):

```bash
python scripts/build_corpora.py
```

---

## Reproducing the pilot experiment

The runner (`scripts/run_effective_tokens.py`) and all evaluation code are
included. The trained checkpoints for the pilot are committed under
`experiments/runs/effective_tokens/`. The prepared corpora used for training
are committed under `experiments/effective_tokens/corpora/`.

To reproduce from scratch, download the corpora with `scripts/build_corpora.py`
and run `scripts/run_effective_tokens.py` with the same hyperparameters.
Results will differ from the paper due to random seed variation with a single
run; the paper reports one seed per configuration.

---

## Tests

The test suite covers tokenizer correctness, data splitting, corpus metric
formulas, model shapes, error handling, and — in `tests/test_causal_properties.py`
— three behavioral invariants:

1. **Causal no-leakage**: logits at position t are identical when only tokens
   at positions > t differ (Transformer and LLaMA-style).
2. **Overfit smoke test**: loss decreases after 30 gradient steps on a fixed
   tiny batch.
3. **RoPE norm preservation**: `apply_rope` preserves the Euclidean norm of
   each rotated pair.

```bash
pytest          # all tests; currently 88 tests pass
pytest -v       # verbose output per test
```

---

## Limitations

- Character-level tokenization only; BPE/subword experiments not included.
- Small-scale experiments (1 million characters, ≤ 5,000 training steps).
- Single seed per configuration; reported numbers are one realization, not an
  average over runs.
- The generalization gap intervals are descriptive (overlapping character
  windows break the iid assumption for formal confidence intervals).
- This is not a full replication of scaling-law effective-token studies;
  no teacher model is used and no scaling curve is fit.
- Corpus choice matters: *High* mixes diversity with domain shift.

---

## Project status

**Implemented**
- Character tokenizer, dataset, three model families, all LLaMA-style modules
- Corpus diversity metrics
- Experiment runner with fixed-budget comparison protocol
- Flask web demo
- 88-test suite including causal correctness checks
- GitHub Actions CI

**Future work**
- Multiple seeds per configuration
- Bigram baseline in the effective-tokens pipeline
- Low-diversity corpus condition
- BPE/subword tokenization comparison
- Larger compute budget and corpus sizes
- Formal reproducibility packaging (locked dependencies, deterministic seeds)
