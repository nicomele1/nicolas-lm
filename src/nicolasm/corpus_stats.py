from __future__ import annotations

import gzip
import math
from collections import Counter


def _validate_nonempty_text(text: str) -> None:
    if not text:
        raise ValueError("text must be nonempty.")


def _validate_ngram_args(text: str, n: int) -> None:
    _validate_nonempty_text(text)

    if n <= 0:
        raise ValueError("n must be positive.")

    if len(text) < n:
        raise ValueError("text length must be at least n.")


def _ngram_counts(text: str, n: int) -> Counter[str]:
    _validate_ngram_args(text, n)
    return Counter(text[index : index + n] for index in range(len(text) - n + 1))


def _entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())

    if total <= 0:
        raise ValueError("counts must contain at least one observation.")

    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)

    return entropy


def character_entropy(text: str) -> float:
    """
    Estimate the empirical character entropy H_1(C) in nats.
    """
    _validate_nonempty_text(text)
    return _entropy_from_counts(Counter(text))


def ngram_entropy(text: str, n: int) -> float:
    """
    Estimate the empirical n-gram entropy H_n(C) in nats.
    """
    return _entropy_from_counts(_ngram_counts(text, n))


def conditional_bigram_entropy(text: str) -> float:
    """
    Estimate H(X_{t+1} | X_t) = H(X_t, X_{t+1}) - H(X_t) in nats.
    """
    _validate_ngram_args(text, 2)

    bigram_entropy = ngram_entropy(text, 2)
    previous_char_entropy = _entropy_from_counts(Counter(text[:-1]))

    return bigram_entropy - previous_char_entropy


def distinct_ngram_ratio(text: str, n: int) -> float:
    """
    Return unique n-grams divided by total n-grams.
    """
    counts = _ngram_counts(text, n)
    total = sum(counts.values())
    return len(counts) / total


def gzip_compression_ratio(text: str) -> float:
    """
    Return compressed_size / original_size using gzip over UTF-8 bytes.

    Higher values indicate less compressible, less redundant text under this
    simple compressor.
    """
    _validate_nonempty_text(text)

    data = text.encode("utf-8")
    compressed = gzip.compress(data)

    return len(compressed) / len(data)


def corpus_summary(text: str) -> dict[str, float | int]:
    """
    Compute the diversity metrics used in the effective-tokens study.
    """
    _validate_ngram_args(text, 4)

    return {
        "num_chars": len(text),
        "vocab_size": len(set(text)),
        "H1": character_entropy(text),
        "H2": ngram_entropy(text, 2),
        "H3": ngram_entropy(text, 3),
        "conditional_bigram_entropy": conditional_bigram_entropy(text),
        "distinct_2": distinct_ngram_ratio(text, 2),
        "distinct_3": distinct_ngram_ratio(text, 3),
        "distinct_4": distinct_ngram_ratio(text, 4),
        "gzip_compression_ratio": gzip_compression_ratio(text),
    }
