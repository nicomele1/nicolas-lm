from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nicolasm.modules.causal_mask import causal_mask
from nicolasm.modules.rmsnorm import RMSNorm
from nicolasm.modules.rope import apply_rope
from nicolasm.modules.swiglu import SwiGLU


class LlamaStyleCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE and bias-free projections.
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

        head_dim = embedding_dim // num_heads

        if head_dim % 2 != 0:
            raise ValueError("embedding_dim / num_heads must be even for RoPE.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", causal_mask(block_size))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        x = x.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, sequence_length, embedding_dim)."
            )

        batch_size, sequence_length, embedding_dim = x.shape

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

        q = apply_rope(self._split_heads(self.q_proj(x)))
        k = apply_rope(self._split_heads(self.k_proj(x)))
        v = self._split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = self.mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, sequence_length, self.embedding_dim)
        out = self.out_proj(out)

        return self.dropout(out)


class LlamaStyleTransformerBlock(nn.Module):
    """
    LLaMA-style decoder block with RMSNorm, RoPE attention, and SwiGLU.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim

        self.attention_norm = RMSNorm(embedding_dim)
        self.attention = LlamaStyleCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.feedforward_norm = RMSNorm(embedding_dim)
        self.feedforward = SwiGLU(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class LlamaStyleLanguageModel(nn.Module):
    """
    Small character-level decoder-only model inspired by LLaMA.

    It keeps the same next-token training interface as the existing models, but
    uses RMSNorm, rotary positional embeddings, and a SwiGLU feedforward block.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        if (embedding_dim // num_heads) % 2 != 0:
            raise ValueError("embedding_dim / num_heads must be even for RoPE.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[
                LlamaStyleTransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, sequence_length).")

        batch_size, sequence_length = idx.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds "
                f"block_size={self.block_size}."
            )

        x = self.token_embedding_table(idx)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx.")

            logits_flat = logits.view(batch_size * sequence_length, self.vocab_size)
            targets_flat = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, current_length).")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
