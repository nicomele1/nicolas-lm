from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root mean square normalization.

    RMSNorm rescales each token representation by its root mean square and
    applies a learned gain. Unlike LayerNorm, it does not subtract the mean.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.embedding_dim = embedding_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected final dimension {self.embedding_dim}, "
                f"got {x.shape[-1]}."
            )

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
