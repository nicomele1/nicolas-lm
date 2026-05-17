from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """
    SwiGLU feedforward block used in LLaMA-style Transformers.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)
