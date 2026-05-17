# Literature Notes

These notes record how the project uses the main mathematical and empirical
references. They are meant to prevent overclaiming in the final report.

## Chang et al. 2024

Reference: Ernie Chang et al., "Scaling Parameter-Constrained Language Models
with Quality Data", EMNLP Industry Track 2024.

Main use in this project:

- Motivation for treating raw token count as an incomplete measure of training
  data value.
- Conceptual source for "effective training tokens".
- Justification for studying text diversity as a data-quality signal.

What Chang et al. actually do:

- Extend scaling-law reasoning by adding an effective-token term.
- Define effective tokens using data quantity plus text-quality indicators.
- Use two quality indicators: diversity and syntheticity measured with a teacher
  model.
- Train more than 200 decoder-only models from 25M to 1.5B parameters.
- Evaluate on eight commonsense reasoning tasks.

Important distinction:

- Chang et al. define a compression ratio as original size divided by compressed
  size, and define diversity using its inverse.
- This project directly records `gzip_compression_ratio =
  compressed_size / original_size`, so larger values mean less compressible and
  more empirically diverse text.

How we adapt it:

- We do not replicate the scaling-law fit.
- We do not estimate their constants.
- We do not use a teacher model or syntheticity score.
- We run a small controlled study: equal raw character count, fixed
  architecture/training settings, varying empirical corpus diversity.

Safe sentence for the paper:

> Inspired by Chang et al.'s effective-token view of data quality, we isolate
> one measurable component, empirical textual diversity, and test whether it is
> associated with better generalization at fixed raw corpus size in small
> character-level language models.

## Petersen and Zech

Reference: Philipp Petersen and Jakob Zech, "Mathematical theory of deep
learning", arXiv:2407.18384.

Main use in this project:

- Mathematical framing of neural-network training as empirical risk
  minimization.
- Optimization context for gradient descent, stochastic gradients, Adam, and
  backpropagation.
- Statistical-learning context for risk, empirical risk, approximation error,
  generalization error, and covering-number bounds.
- Architectural context for Transformers and causal self-attention.

Relevant chapters read for this project:

- Chapter 8: high-dimensional approximation and the curse of dimensionality.
  This is background only; it motivates why high-dimensional learning requires
  structural assumptions, but it is not central to the empirical design.
- Chapter 10: training of neural networks. Use this for the objective-function
  view, mini-batch stochastic gradients, Adam, and backpropagation.
- Chapter 14: generalization properties. Use this for risk, empirical risk,
  empirical risk minimization, and the decomposition into generalization and
  approximation terms.
- Chapter 15: overparameterized generalization. Use this only as context for why
  classical bounds do not fully explain modern deep learning.
- Chapter 17: modern architectures. Use this for embeddings, positional
  information, causal self-attention, transformer blocks, and autoregressive text
  generation.

Safe mathematical framing:

Let `C` be a text corpus and let its token sequence be
`x_0, ..., x_{N-1}`. For block size `B`, the dataset consists of windows

```text
X_i = (x_i, ..., x_{i+B-1}),
Y_i = (x_{i+1}, ..., x_{i+B}).
```

A model with parameters `theta` defines next-token probabilities

```text
p_theta(x_{t+1} | x_{t-B+1}, ..., x_t).
```

Training minimizes empirical cross-entropy:

```text
R_hat_train(theta) =
  (1 / n) sum_i -log p_theta(Y_i | X_i).
```

The study then compares empirical generalization summaries:

```text
test_loss
test_perplexity = exp(test_loss)
generalization_gap = test_loss - train_loss
```

Important limitation:

- Petersen and Zech's generalization bounds assume idealized settings such as
  i.i.d. samples and bounded hypothesis classes. Our overlapping language-model
  windows are not independent. Therefore, our batch-loss intervals are
  descriptive summaries, not exact confidence intervals.
