from __future__ import annotations

from pathlib import Path

import torch

from nicolasm.data import TokenDataset, train_val_split
from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.tokenizer import CharTokenizer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_PATH = Path("data/raw/input.txt")
CHECKPOINT_DIR = Path("experiments/runs/bigram")
CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pt"

BLOCK_SIZE = 8
BATCH_SIZE = 16
MAX_STEPS = 500
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-2


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


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    dataset: TokenDataset,
    eval_iters: int = 20,
) -> float:
    """
    Estimate the average cross-entropy loss on a dataset.
    """
    model.eval()

    losses: list[float] = []

    for _ in range(eval_iters):
        xs, ys = dataset.get_batch(BATCH_SIZE)
        x, y = batch_to_tensors(xs, ys)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Expected loss, got None.")

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Create data/raw/input.txt before training."
        )

    text = DATA_PATH.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.from_text(text)
    tokens = tokenizer.encode(text)

    train_tokens, val_tokens = train_val_split(tokens, train_fraction=0.9)

    train_dataset = TokenDataset(tokens=train_tokens, block_size=BLOCK_SIZE)
    val_dataset = TokenDataset(tokens=val_tokens, block_size=BLOCK_SIZE)

    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Training NicolásLM-0: BigramLanguageModel")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Validation tokens: {len(val_tokens)}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            train_loss = estimate_loss(model, train_dataset)
            val_loss = estimate_loss(model, val_dataset)

            print(
                f"step {step:04d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}"
            )

        xs, ys = train_dataset.get_batch(BATCH_SIZE)
        x, y = batch_to_tensors(xs, ys)

        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Expected loss, got None.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "vocab_size": tokenizer.vocab_size,
            "block_size": BLOCK_SIZE,
        },
        CHECKPOINT_PATH,
    )

    print()
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()