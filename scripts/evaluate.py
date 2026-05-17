from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn

from nicolasm.data import TokenDataset, train_val_test_split
from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.models.llama import LlamaStyleLanguageModel
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.tokenizer import CharTokenizer


DATA_PATH = Path("data/raw/input.txt")
DEFAULT_EVAL_ITERS = 100
DEFAULT_BATCH_SIZE = 32

CSV_FIELDNAMES = [
    "model_name",
    "checkpoint_path",
    "vocab_size",
    "train_tokens",
    "val_tokens",
    "test_tokens",
    "block_size",
    "embedding_dim",
    "num_heads",
    "num_layers",
    "max_steps",
    "batch_size",
    "eval_iters",
    "train_loss",
    "val_loss",
    "test_loss_mean",
    "test_loss_sd",
    "test_loss_se",
    "test_loss_ci_low",
    "test_loss_ci_high",
    "test_ppl",
    "generalization_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NicolásLM checkpoint on the test split."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="transformer",
        choices=["bigram", "transformer", "llama"],
        help="Name of the model checkpoint to evaluate.",
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="UTF-8 text corpus used to recreate the train/val/test split.",
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Directory containing model checkpoint subdirectories.",
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

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path where evaluation metrics are appended.",
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


@torch.no_grad()
def estimate_loss_stats(
    model: nn.Module,
    dataset: TokenDataset,
    batch_size: int,
    eval_iters: int,
) -> dict[str, float]:
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

    mean = sum(losses) / len(losses)

    if len(losses) == 1:
        sd = 0.0
    else:
        variance = sum((loss_value - mean) ** 2 for loss_value in losses)
        variance /= len(losses) - 1
        sd = math.sqrt(variance)

    se = sd / math.sqrt(len(losses))
    margin = 1.96 * se

    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def append_csv_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    args = parse_args()

    checkpoint_path = args.runs_dir / args.model_name / "model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. "
            "Train the model first by running scripts/train.py."
        )

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Could not find {args.data_path}. "
            "Create data/raw/input.txt before evaluation."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    tokenizer = load_tokenizer_from_checkpoint(checkpoint)

    text = args.data_path.read_text(encoding="utf-8")

    try:
        tokens = tokenizer.encode(text)
    except ValueError:
        raise SystemExit(
            f"The current {args.data_path} contains characters that were not "
            "present when this checkpoint was trained. Retrain the model with "
            "the current corpus before evaluating it."
        )

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

    test_loss_stats = estimate_loss_stats(
        model=model,
        dataset=test_dataset,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
    )
    test_ppl = math.exp(test_loss_stats["mean"])
    train_loss = checkpoint.get("final_train_loss")
    val_loss = checkpoint.get("final_val_loss")
    generalization_gap = None

    if train_loss is not None:
        generalization_gap = test_loss_stats["mean"] - train_loss

    row = {
        "model_name": args.model_name,
        "checkpoint_path": str(checkpoint_path),
        "vocab_size": str(checkpoint["vocab_size"]),
        "train_tokens": str(len(train_tokens)),
        "val_tokens": str(len(val_tokens)),
        "test_tokens": str(len(test_tokens)),
        "block_size": str(block_size),
        "embedding_dim": str(checkpoint.get("embedding_dim", "")),
        "num_heads": str(checkpoint.get("num_heads", "")),
        "num_layers": str(checkpoint.get("num_layers", "")),
        "max_steps": str(checkpoint.get("max_steps", "")),
        "batch_size": str(args.batch_size),
        "eval_iters": str(args.eval_iters),
        "train_loss": "" if train_loss is None else str(train_loss),
        "val_loss": "" if val_loss is None else str(val_loss),
        "test_loss_mean": str(test_loss_stats["mean"]),
        "test_loss_sd": str(test_loss_stats["sd"]),
        "test_loss_se": str(test_loss_stats["se"]),
        "test_loss_ci_low": str(test_loss_stats["ci_low"]),
        "test_loss_ci_high": str(test_loss_stats["ci_high"]),
        "test_ppl": str(test_ppl),
        "generalization_gap": (
            "" if generalization_gap is None else str(generalization_gap)
        ),
    }

    if args.output is not None:
        append_csv_row(args.output, row)

    print(f"Evaluating model: {args.model_name}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Vocabulary size: {checkpoint['vocab_size']}")
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Validation tokens: {len(val_tokens)}")
    print(f"Test tokens: {len(test_tokens)}")
    print(f"Block size: {block_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Evaluation batches: {args.eval_iters}")
    print()
    print(f"Test loss mean: {test_loss_stats['mean']:.4f}")
    print(f"Test loss sd: {test_loss_stats['sd']:.4f}")
    print(f"Test loss se: {test_loss_stats['se']:.4f}")
    print(
        "Descriptive 95% interval: "
        f"[{test_loss_stats['ci_low']:.4f}, {test_loss_stats['ci_high']:.4f}]"
    )
    print(f"Test perplexity: {test_ppl:.4f}")

    if train_loss is not None:
        print(f"Checkpoint train loss: {train_loss:.4f}")

    if val_loss is not None:
        print(f"Checkpoint validation loss: {val_loss:.4f}")

    if generalization_gap is not None:
        print(f"Generalization gap: {generalization_gap:.4f}")

    if args.output is not None:
        print(f"Saved evaluation row to {args.output}")


if __name__ == "__main__":
    main()
