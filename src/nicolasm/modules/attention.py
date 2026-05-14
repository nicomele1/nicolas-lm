from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nicolasm.modules.causal_mask import causal_mask


class CausalSelfAttentionHead(nn.Module):
    """
    A single causal self-attention head.

    Given an input tensor x of shape

        (batch_size, block_size, embedding_dim),

    this module computes queries, keys and values, applies causal scaled
    dot-product attention, and returns an output tensor of shape

        (batch_size, block_size, head_dim).
    """

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if head_dim <= 0:
            raise ValueError("head_dim must be positive.")

        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.block_size = block_size

        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        mask = causal_mask(block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal self-attention.

        Parameters
        ----------
        x:
            Tensor of shape (batch_size, block_size, embedding_dim).

        Returns
        -------
        Tensor of shape (batch_size, block_size, head_dim).
        """
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, block_size, embedding_dim)."
            )

        _, sequence_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {embedding_dim}."
            )

        if sequence_length > self.block_size:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds "
                f"block_size={self.block_size}."
            )

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = self.mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v

        return out


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    This module runs several causal self-attention heads in parallel,
    concatenates their outputs, and projects back to the embedding dimension.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    embedding_dim=embedding_dim,
                    head_dim=self.head_dim,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head causal self-attention.

        Parameters
        ----------
        x:
            Tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns
        -------
        Tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, sequence_length, embedding_dim)."
            )

        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {x.shape[-1]}."
            )

        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out