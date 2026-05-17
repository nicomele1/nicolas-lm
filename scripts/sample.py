from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.models.llama import LlamaStyleLanguageModel
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample text from a trained NicolásLM checkpoint."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="transformer",
        choices=["bigram", "transformer", "llama"],
        help="Name of the model checkpoint to load.",
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Directory containing model checkpoint subdirectories.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="El ",
        help="Initial text prompt.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Lower is more conservative.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Keep only the top-k most likely tokens. Use 0 to disable.",
    )

    return parser.parse_args()


def load_tokenizer_from_checkpoint(checkpoint: dict) -> CharTokenizer:
    """
    Reconstruct the character tokenizer from the checkpoint.
    """
    return CharTokenizer(
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"],
    )


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

    if model_name == "llama":
        return LlamaStyleLanguageModel(
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
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """
    Sample one token from a batch of logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    logits = logits / temperature

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive when provided.")

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
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively from an initial context.
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
    args = parse_args()

    checkpoint_path = args.runs_dir / args.model_name / "model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. "
            "Train the model first by running scripts/train.py."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    tokenizer = load_tokenizer_from_checkpoint(checkpoint)

    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except ValueError as exc:
        raise ValueError(
            "The prompt contains characters that were not present in the "
            "training corpus. Choose a prompt using only known characters."
        ) from exc

    context = torch.tensor([prompt_ids], dtype=torch.long)

    top_k = None if args.top_k == 0 else args.top_k

    generated = generate(
        model=model,
        idx=context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )

    text = tokenizer.decode(generated[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
