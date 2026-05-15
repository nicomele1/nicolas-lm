from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from nicolasm.data import TokenDataset, train_val_test_split
from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.tokenizer import CharTokenizer


DATA_PATH = Path("data/raw/input.txt")
DEFAULT_EVAL_ITERS = 100
DEFAULT_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NicolásLM checkpoint on the test split."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="transformer",
        choices=["bigram", "transformer"],
        help="Name of the model checkpoint to evaluate.",
    )

    parser.add_argument(
        "--eval-iters",
        type=int,
        default=DEFAULT_EVAL_ITERS,
        help="Number of random batches used to estimate test loss.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for evaluation.",
    )

    return parser.parse_args()


def batch_to_tensors(
    xs: list[list[int]],
    ys: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def load_tokenizer_from_checkpoint(checkpoint: dict) -> CharTokenizer:
    return CharTokenizer(
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"],
    )


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
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


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: TokenDataset,
    batch_size: int,
    eval_iters: int,
) -> float:
    if eval_iters <= 0:
        raise ValueError("eval_iters must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()

    losses: list[float] = []

    for _ in range(eval_iters):
        xs, ys = dataset.get_batch(batch_size)
        x, y = batch_to_tensors(xs, ys)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Expected loss, got None.")

        losses.append(loss.item())

    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()

    checkpoint_path = Path("experiments/runs") / args.model_name / "model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. "
            "Train the model first by running scripts/train.py."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Create data/raw/input.txt before evaluation."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    tokenizer = load_tokenizer_from_checkpoint(checkpoint)

    text = DATA_PATH.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)

    train_fraction = checkpoint.get("train_fraction", 0.8)
    val_fraction = checkpoint.get("val_fraction", 0.1)

    train_tokens, val_tokens, test_tokens = train_val_test_split(
        tokens,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    block_size = checkpoint["block_size"]
    test_dataset = TokenDataset(tokens=test_tokens, block_size=block_size)

    model = build_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = estimate_loss(
        model=model,
        dataset=test_dataset,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
    )

    print(f"Evaluating model: {args.model_name}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Vocabulary size: {checkpoint['vocab_size']}")
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Validation tokens: {len(val_tokens)}")
    print(f"Test tokens: {len(test_tokens)}")
    print(f"Block size: {block_size}")
    print()
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()