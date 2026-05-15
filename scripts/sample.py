from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.tokenizer import CharTokenizer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "transformer"
CHECKPOINT_PATH = Path("experiments/runs") / MODEL_NAME / "model.pt"

PROMPT = "El "
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.7
TOP_K: int | None = 20


def load_tokenizer_from_checkpoint(checkpoint: dict) -> CharTokenizer:
    """
    Reconstruct the character tokenizer from the checkpoint.
    """
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    return CharTokenizer(stoi=stoi, itos=itos)


def build_model_from_checkpoint(checkpoint: dict) -> torch.nn.Module:
    """
    Reconstruct the model architecture from checkpoint metadata.
    """
    model_name = checkpoint.get("model_name", "bigram")

    if model_name == "bigram":
        return BigramLanguageModel(
            vocab_size=checkpoint["vocab_size"],
        )

    if model_name == "transformer":
        return TinyTransformerLanguageModel(
            vocab_size=checkpoint["vocab_size"],
            block_size=checkpoint["block_size"],
            embedding_dim=checkpoint["embedding_dim"],
            num_heads=checkpoint["num_heads"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )

    raise ValueError(f"Unknown model_name in checkpoint: {model_name!r}")


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Sample one token from logits.

    Parameters
    ----------
    logits:
        Tensor of shape (batch_size, vocab_size).

    temperature:
        Controls randomness. Smaller values make the distribution sharper.

    top_k:
        If not None, keep only the top_k most likely tokens before sampling.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    logits = logits / temperature

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive if provided.")

        vocab_size = logits.shape[-1]
        k = min(top_k, vocab_size)

        values, _ = torch.topk(logits, k=k, dim=-1)
        threshold = values[:, [-1]]

        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    probs = F.softmax(logits, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1)

    return next_idx


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively from a prompt.
    """
    if idx.ndim != 2:
        raise ValueError("idx must have shape (batch_size, current_length).")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")

    block_size = getattr(model, "block_size", None)

    for _ in range(max_new_tokens):
        if block_size is None:
            idx_cond = idx
        else:
            idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)

        last_logits = logits[:, -1, :]

        next_idx = sample_next_token(
            logits=last_logits,
            temperature=temperature,
            top_k=top_k,
        )

        idx = torch.cat((idx, next_idx), dim=1)

    return idx


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CHECKPOINT_PATH}. "
            "Train the model first by running scripts/train.py."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    tokenizer = load_tokenizer_from_checkpoint(checkpoint)

    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        prompt_ids = tokenizer.encode(PROMPT)
    except ValueError as exc:
        raise ValueError(
            "The prompt contains characters that were not present in the "
            "training corpus. Choose a prompt using only known characters."
        ) from exc

    context = torch.tensor([prompt_ids], dtype=torch.long)

    generated = generate(
        model=model,
        idx=context,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    text = tokenizer.decode(generated[0].tolist())

    print(text)


if __name__ == "__main__":
    main()