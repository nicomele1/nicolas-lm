from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def nats_to_bits(nats: float) -> float:
    """Convert a cross-entropy measured in nats to bits."""
    if not math.isfinite(nats):
        raise ValueError("nats must be finite.")
    return nats / math.log(2.0)


def unigram_cross_entropy(
    train_tokens: Sequence[int],
    test_tokens: Sequence[int],
    vocab_size: int,
    alpha: float = 1.0,
) -> float:
    """Evaluate a train-fitted add-alpha unigram model on held-out tokens.

    This baseline separates part of the corpus' intrinsic marginal difficulty
    from the contextual gain achieved by an autoregressive neural model.
    """
    if not train_tokens or not test_tokens:
        raise ValueError("train_tokens and test_tokens must be nonempty.")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    counts = Counter(train_tokens)
    if any(token < 0 or token >= vocab_size for token in (*counts, *test_tokens)):
        raise ValueError("token ids must lie in [0, vocab_size).")

    denominator = len(train_tokens) + alpha * vocab_size
    total_negative_log_likelihood = sum(
        -math.log((counts[token] + alpha) / denominator) for token in test_tokens
    )
    return total_negative_log_likelihood / len(test_tokens)

