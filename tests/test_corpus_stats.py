import math

import pytest

from nicolasm.corpus_stats import (
    character_entropy,
    conditional_bigram_entropy,
    corpus_summary,
    distinct_ngram_ratio,
    gzip_compression_ratio,
    ngram_entropy,
)


def test_repeated_string_has_zero_character_entropy() -> None:
    assert character_entropy("aaaaaa") == pytest.approx(0.0)


def test_balanced_two_character_string_has_log_two_entropy() -> None:
    assert character_entropy("abab") == pytest.approx(math.log(2))


def test_ngram_entropy_is_zero_for_repeated_ngram() -> None:
    assert ngram_entropy("aaaaaa", n=2) == pytest.approx(0.0)


def test_ngram_entropy_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        ngram_entropy("", n=1)

    with pytest.raises(ValueError):
        ngram_entropy("abc", n=0)

    with pytest.raises(ValueError):
        ngram_entropy("abc", n=4)


def test_conditional_bigram_entropy_is_zero_for_deterministic_transitions() -> None:
    assert conditional_bigram_entropy("ababab") == pytest.approx(0.0)


def test_conditional_bigram_entropy_rejects_too_short_text() -> None:
    with pytest.raises(ValueError):
        conditional_bigram_entropy("a")


def test_distinct_ngram_ratio_for_repeated_text() -> None:
    assert distinct_ngram_ratio("aaaa", n=2) == pytest.approx(1 / 3)


def test_distinct_ngram_ratio_for_all_distinct_ngrams() -> None:
    assert distinct_ngram_ratio("abcd", n=2) == pytest.approx(1.0)


def test_gzip_compression_ratio_is_positive() -> None:
    assert gzip_compression_ratio("abababababababab") > 0.0


def test_gzip_compression_ratio_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        gzip_compression_ratio("")


def test_corpus_summary_contains_expected_metrics() -> None:
    summary = corpus_summary("abcdabcd")

    assert summary["num_chars"] == 8
    assert summary["vocab_size"] == 4
    assert "H1" in summary
    assert "H2" in summary
    assert "H3" in summary
    assert "conditional_bigram_entropy" in summary
    assert "distinct_2" in summary
    assert "distinct_3" in summary
    assert "distinct_4" in summary
    assert "gzip_compression_ratio" in summary
