from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BigramLanguageModel(nn.Module):
    """
    Bigram language model.

    Given a token x_t, the model returns logits for the conditional
    distribution

        p(x_{t+1} | x_t).

    If the vocabulary has size m, the model learns a matrix A in R^{m x m}.
    The row A[i] contains the logits for the next-token distribution
    conditioned on the current token being i.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute logits and optionally the cross-entropy loss.

        Parameters
        ----------
        idx:
            Tensor of shape (batch_size, block_size).

        targets:
            Optional tensor of shape (batch_size, block_size).

        Returns
        -------
        logits:
            Tensor of shape (batch_size, block_size, vocab_size).

        loss:
            Scalar cross-entropy loss if targets are provided; otherwise None.
        """
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, block_size).")

        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx.")

            batch_size, block_size, vocab_size = logits.shape

            logits_flat = logits.view(batch_size * block_size, vocab_size)
            targets_flat = targets.view(batch_size * block_size)

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

        Starting from idx, repeatedly sample

            x_{t+1} ~ p_theta(. | x_t).
        """
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, current_length).")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative.")

        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_idx), dim=1)

        return idx