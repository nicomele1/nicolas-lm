from __future__ import annotations

from pathlib import Path

import torch

from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.tokenizer import CharTokenizer


CHECKPOINT_PATH = Path("experiments/runs/bigram/model.pt")
MAX_NEW_TOKENS = 300


def load_tokenizer_from_checkpoint(checkpoint: dict) -> CharTokenizer:
    """
    Reconstruct the character tokenizer from the checkpoint.
    """
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    return CharTokenizer(stoi=stoi, itos=itos)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CHECKPOINT_PATH}. "
            "Train the bigram model first by running scripts/train.py."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    tokenizer = load_tokenizer_from_checkpoint(checkpoint)

    model = BigramLanguageModel(vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Start generation from a single newline if it exists in the vocabulary;
    # otherwise start from the first token in the vocabulary.
    if "\n" in tokenizer.stoi:
        start_token = tokenizer.stoi["\n"]
    else:
        start_token = 0

    context = torch.tensor([[start_token]], dtype=torch.long)

    generated = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)

    text = tokenizer.decode(generated[0].tolist())

    print(text)


if __name__ == "__main__":
    main()