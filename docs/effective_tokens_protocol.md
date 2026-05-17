# Effective Tokens Experiment Protocol

## Research Question

At equal raw character/token count, does empirical textual diversity act as a
proxy for effective training tokens and improve generalization in small
autoregressive language models?

## Experimental Unit

Each row of the final results table corresponds to one pair:

- one corpus prepared to the common raw character budget;
- one model architecture trained from scratch on that corpus.

The main comparison should keep architecture and training hyperparameters fixed
while varying only the corpus.

## Recommended Models

- `bigram`: interpretable first-order Markov baseline.
- `transformer`: main small decoder-only Transformer.
- `llama`: modern Transformer variant with RMSNorm, RoPE, and SwiGLU.

Use `transformer` as the main model. Use `bigram` as the statistical baseline
and `llama` as a robustness check if time allows.

## Corpus Requirements

Prepare corpora that differ in empirical diversity but are comparable in raw
size and language:

- low diversity: repetitive text or repeated fragments;
- medium diversity: one homogeneous book or author;
- high diversity: multiple authors, genres, or sources;
- optional noisy corpus: OCR-like or dirty text.

The runner can enforce equal raw character count by truncating all corpora to
the shortest corpus:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_effective_tokens.py \
  --input data/corpora/corpus_low.txt --name low \
  --input data/corpora/corpus_medium.txt --name medium \
  --input data/corpora/corpus_high.txt --name high \
  --model-name bigram \
  --model-name transformer \
  --model-name llama \
  --truncate-to-min-chars \
  --output experiments/results/effective_tokens.csv
```

For faster pilot runs, set a smaller shared budget:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_effective_tokens.py \
  --input data/corpora/corpus_low.txt --name low \
  --input data/corpora/corpus_high.txt --name high \
  --model-name transformer \
  --max-chars 50000 \
  --max-steps 1000 \
  --output experiments/results/pilot_effective_tokens.csv
```

## Output Columns

The final CSV combines corpus metrics and model metrics:

- corpus metrics: `H1`, `H2`, `H3`, `conditional_bigram_entropy`,
  `distinct_2`, `distinct_3`, `distinct_4`, `gzip_compression_ratio`;
- training/evaluation metrics: `train_loss`, `val_loss`, `test_loss_mean`,
  `test_loss_se`, `test_ppl`, `generalization_gap`;
- model configuration: `model_name`, `block_size`, `embedding_dim`,
  `num_heads`, `num_layers`, `max_steps`.

## Interpretation

The key comparison is at fixed architecture and fixed raw corpus size:

```text
Delta_test = test_loss(high_diversity) - test_loss(low_diversity)
```

Negative `Delta_test` means the higher-diversity corpus achieved lower test
cross-entropy. Also compare perplexity and generalization gap.

The reported uncertainty interval is descriptive. It summarizes variation over
random evaluation batches, but it is not an exact iid confidence interval
because character windows overlap.
