from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nicolasm.modules.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feedforward network used inside a Transformer block.

    It acts independently on each position of the sequence.
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

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A decoder-only Transformer block.

    The block has two residual sublayers:

        x <- x + causal_self_attention(layer_norm(x))
        x <- x + feedforward(layer_norm(x))

    This is a pre-norm Transformer block.
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

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=4 * embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x


class TinyTransformerLanguageModel(nn.Module):
    """
    Tiny decoder-only Transformer language model.

    Given token indices of shape

        (batch_size, sequence_length),

    the model returns logits of shape

        (batch_size, sequence_length, vocab_size).

    It models next-token probabilities of the form

        p_theta(x_{t+1} | x_0, ..., x_t).
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

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute logits and optionally cross-entropy loss.
        """
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, sequence_length).")

        batch_size, sequence_length = idx.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds "
                f"block_size={self.block_size}."
            )

        token_embeddings = self.token_embedding_table(idx)

        positions = torch.arange(sequence_length, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_final(x)

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
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        """
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, current_length).")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]

            logits, _ = self(idx_cond)

            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_idx), dim=1)

        return idx