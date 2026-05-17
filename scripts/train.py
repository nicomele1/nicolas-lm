from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from nicolasm.data import TokenDataset, train_val_test_split
from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.models.llama import LlamaStyleLanguageModel
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.tokenizer import CharTokenizer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("data/raw/input.txt")

MODEL_NAME = "transformer"

BLOCK_SIZE = 64
BATCH_SIZE = 32
MAX_STEPS = 15000
EVAL_INTERVAL = 700
LEARNING_RATE = 3e-4

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small character-level NicolásLM model."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        choices=["bigram", "transformer", "llama"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="UTF-8 text corpus used for training.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Directory where model checkpoints are stored.",
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)

    return parser.parse_args()


def batch_to_tensors(
    xs: list[list[int]],
    ys: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Python lists of token ids into PyTorch integer tensors.
    """
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def build_model(
    model_name: str,
    vocab_size: int,
    block_size: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """
    Build the selected language model.
    """
    if model_name == "bigram":
        return BigramLanguageModel(vocab_size=vocab_size)

    if model_name == "transformer":
        return TinyTransformerLanguageModel(
            vocab_size=vocab_size,
            block_size=block_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    if model_name == "llama":
        return LlamaStyleLanguageModel(
            vocab_size=vocab_size,
            block_size=block_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model_name: {model_name!r}")


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: TokenDataset,
    batch_size: int,
    eval_iters: int = 20,
) -> float:
    """
    Estimate the average cross-entropy loss on a dataset.
    """
    model.eval()

    losses: list[float] = []

    for _ in range(eval_iters):
        xs, ys = dataset.get_batch(batch_size)
        x, y = batch_to_tensors(xs, ys)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Expected loss, got None.")

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()

    if args.max_steps < 0:
        raise ValueError("max_steps must be nonnegative.")

    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Could not find {args.data_path}. "
            "Create data/raw/input.txt before training."
        )

    text = args.data_path.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.from_text(text)
    tokens = tokenizer.encode(text)

    train_tokens, val_tokens, test_tokens = train_val_test_split(
        tokens,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    train_dataset = TokenDataset(tokens=train_tokens, block_size=args.block_size)
    val_dataset = TokenDataset(tokens=val_tokens, block_size=args.block_size)

    model = build_model(
        model_name=args.model_name,
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    checkpoint_dir = args.runs_dir / args.model_name
    checkpoint_path = checkpoint_dir / "model.pt"
    test_fraction = 1.0 - args.train_fraction - args.val_fraction

    print(f"Training model: {args.model_name}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Validation tokens: {len(val_tokens)}")
    print(f"Test tokens: {len(test_tokens)}")
    print(f"Block size: {args.block_size}")
    print(f"Batch size: {args.batch_size}")
    print()

    train_loss = estimate_loss(model, train_dataset, args.batch_size)
    val_loss = estimate_loss(model, val_dataset, args.batch_size)
    print(
        f"step {0:04d} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f}"
    )

    for step in range(1, args.max_steps + 1):
        xs, ys = train_dataset.get_batch(args.batch_size)
        x, y = batch_to_tensors(xs, ys)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Expected loss, got None.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0:
            train_loss = estimate_loss(model, train_dataset, args.batch_size)
            val_loss = estimate_loss(model, val_dataset, args.batch_size)

            print(
                f"step {step:04d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )

    final_train_loss = estimate_loss(model, train_dataset, args.batch_size)
    final_val_loss = estimate_loss(model, val_dataset, args.batch_size)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": args.model_name,
            "model_state_dict": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "vocab_size": tokenizer.vocab_size,
            "block_size": args.block_size,
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "eval_interval": args.eval_interval,
            "learning_rate": args.learning_rate,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": test_fraction,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
        },
        checkpoint_path,
    )

    print()
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
