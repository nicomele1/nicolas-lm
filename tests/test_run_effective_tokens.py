import pytest

from scripts.run_effective_tokens import (
    corpus_names,
    target_char_count,
    validate_split_lengths,
)


def test_corpus_names_defaults_to_file_stems(tmp_path) -> None:
    paths = [tmp_path / "low.txt", tmp_path / "high.txt"]

    assert corpus_names(paths, names=None) == ["low", "high"]


def test_corpus_names_rejects_mismatched_names(tmp_path) -> None:
    paths = [tmp_path / "low.txt", tmp_path / "high.txt"]

    with pytest.raises(ValueError):
        corpus_names(paths, names=["low"])


def test_corpus_names_rejects_duplicates(tmp_path) -> None:
    paths = [tmp_path / "a.txt", tmp_path / "b.txt"]

    with pytest.raises(ValueError):
        corpus_names(paths, names=["same", "same"])


def test_target_char_count_accepts_equal_lengths() -> None:
    assert target_char_count(["abcd", "wxyz"], False, None) == 4


def test_target_char_count_rejects_unequal_lengths_without_truncation() -> None:
    with pytest.raises(ValueError):
        target_char_count(["abcd", "xyz"], False, None)


def test_target_char_count_uses_min_length_with_truncation() -> None:
    assert target_char_count(["abcd", "xyz"], True, None) == 3


def test_target_char_count_uses_max_chars() -> None:
    assert target_char_count(["abcd", "wxyz"], False, 2) == 2


def test_target_char_count_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        target_char_count(["abcd", ""], True, None)


def test_validate_split_lengths_accepts_sufficient_splits() -> None:
    validate_split_lengths(
        num_chars=100,
        train_fraction=0.8,
        val_fraction=0.1,
        block_size=8,
    )


def test_validate_split_lengths_rejects_too_short_split() -> None:
    with pytest.raises(ValueError):
        validate_split_lengths(
            num_chars=40,
            train_fraction=0.8,
            val_fraction=0.1,
            block_size=8,
        )
