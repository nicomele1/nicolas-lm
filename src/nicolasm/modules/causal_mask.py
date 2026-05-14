from __future__ import annotations

import torch


def causal_mask(block_size: int) -> torch.Tensor:
    """
    Return a causal attention mask of shape (block_size, block_size).

    The entry mask[i, j] is True exactly when position i is allowed to
    attend to position j.

    For autoregressive language modeling, position i may attend only to
    positions j <= i. Therefore the mask is lower triangular.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    return torch.tril(torch.ones((block_size, block_size), dtype=torch.bool))