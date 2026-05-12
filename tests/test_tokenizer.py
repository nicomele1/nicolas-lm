import pytest

from nicolasm.tokenizer import CharTokenizer


def test_encode_decode_roundtrip() -> None:
    text = "hola mundo"
    tokenizer = CharTokenizer.from_text(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_vocab_size() -> None:
    text = "banana"
    tokenizer = CharTokenizer.from_text(text)

    assert tokenizer.vocab_size == 3


def test_unknown_character_raises_error() -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError):
        tokenizer.encode("abcd")


def test_unknown_token_id_raises_error() -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError):
        tokenizer.decode([0, 1, 2, 99])


def test_empty_text_raises_error() -> None:
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")

        