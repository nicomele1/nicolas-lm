from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    """
    A minimal character-level tokenizer.

    This tokenizer builds a finite vocabulary from a text corpus. It gives two
    maps:

        stoi: character -> integer token id
        itos: integer token id -> character

    This is the simplest useful tokenizer for next-token language modeling.
    """

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """
        Build a character-level tokenizer from the characters appearing in text.
        """
        if not text:
            raise ValueError("Cannot build a tokenizer from an empty text.")

        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}

        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """
        Number of distinct tokens in the vocabulary.
        """
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string as a list of integer token ids.
        """
        ids: list[int] = []

        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Unknown character: {ch!r}")
            ids.append(self.stoi[ch])

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of integer token ids back into a string.
        """
        chars: list[str] = []

        for idx in ids:
            if idx not in self.itos:
                raise ValueError(f"Unknown token id: {idx!r}")
            chars.append(self.itos[idx])

        return "".join(chars)