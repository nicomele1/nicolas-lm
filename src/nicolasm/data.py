from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenDataset:
    """
    Dataset for next-token prediction.

    Given tokens

        x_0, x_1, ..., x_{N-1},

    and a block size B, the dataset returns examples

        input  = (x_i,     x_{i+1}, ..., x_{i+B-1})
        target = (x_{i+1}, x_{i+2}, ..., x_{i+B})

    So target is input shifted one position to the left.
    """

    tokens: list[int]
    block_size: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")

        if len(self.tokens) <= self.block_size:
            raise ValueError(
                "The token sequence must have length greater than block_size."
            )

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range.")

        chunk = self.tokens[index : index + self.block_size + 1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y

    def get_batch(self, batch_size: int) -> tuple[list[list[int]], list[list[int]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        xs: list[list[int]] = []
        ys: list[list[int]] = []

        for _ in range(batch_size):
            index = random.randint(0, len(self) - 1)
            x, y = self[index]
            xs.append(x)
            ys.append(y)

        return xs, ys


def train_val_split(
    tokens: list[int],
    train_fraction: float = 0.9,
) -> tuple[list[int], list[int]]:
    """
    Split a token sequence into train and validation parts.
    """
    if not tokens:
        raise ValueError("Cannot split an empty token sequence.")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")

    split_index = int(len(tokens) * train_fraction)

    if split_index == 0 or split_index == len(tokens):
        raise ValueError("Split would produce an empty train or validation set.")

    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]

    return train_tokens, val_tokens


def train_val_test_split(
    tokens: list[int],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split a token sequence into train, validation, and test parts.

    The train set is used to update model parameters.
    The validation set is used to monitor training and tune hyperparameters.
    The test set is reserved for final evaluation.
    """
    if not tokens:
        raise ValueError("Cannot split an empty token sequence.")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie strictly between 0 and 1.")

    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1.")

    n = len(tokens)

    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]

    if not train_tokens or not val_tokens or not test_tokens:
        raise ValueError(
            "Split would produce an empty train, validation, or test set."
        )

    return train_tokens, val_tokens, test_tokens