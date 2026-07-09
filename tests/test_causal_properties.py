"""
Property tests for core invariants of the autoregressive models.

These tests check behavioural contracts that go beyond shape/error checks:

1. Causal no-leakage: logits at position t must not change when tokens at
   positions > t are modified. This verifies that causal masking works end-to-end
   in both the Transformer and the LLaMA-style model.

2. Overfit smoke test: a tiny artificial corpus, a few gradient steps, loss
   must decrease. This confirms the training loop is functional.

3. RoPE norm preservation: a 2-D rotation preserves the Euclidean norm of each
   (x_even, x_odd) pair. apply_rope must satisfy this for all positions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nicolasm.models.llama import LlamaStyleLanguageModel
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.modules.rope import apply_rope


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tiny_transformer(vocab_size: int = 16, block_size: int = 8) -> TinyTransformerLanguageModel:
    return TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )


def _make_tiny_llama(vocab_size: int = 16, block_size: int = 8) -> LlamaStyleLanguageModel:
    return LlamaStyleLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )


# ── Causal no-leakage tests ───────────────────────────────────────────────────

def _causal_no_leakage(model: torch.nn.Module, vocab_size: int, block_size: int) -> None:
    """
    Two sequences identical up to position split_at and different after.
    Logits at positions <= split_at must be equal for both sequences.
    """
    split_at = 3
    seq_len = 6
    assert seq_len <= block_size

    torch.manual_seed(0)
    shared = torch.randint(0, vocab_size, (1, split_at))
    suffix1 = torch.randint(0, vocab_size, (1, seq_len - split_at))
    suffix2 = torch.randint(0, vocab_size, (1, seq_len - split_at))

    while torch.equal(suffix1, suffix2):
        suffix2 = torch.randint(0, vocab_size, (1, seq_len - split_at))

    x1 = torch.cat([shared, suffix1], dim=1)
    x2 = torch.cat([shared, suffix2], dim=1)

    model.eval()
    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    assert torch.allclose(
        logits1[:, :split_at, :],
        logits2[:, :split_at, :],
        atol=1e-5,
    ), "Causal no-leakage violated: logits before split_at differ."


def test_transformer_causal_no_leakage() -> None:
    model = _make_tiny_transformer()
    _causal_no_leakage(model, vocab_size=16, block_size=8)


def test_llama_causal_no_leakage() -> None:
    model = _make_tiny_llama()
    _causal_no_leakage(model, vocab_size=16, block_size=8)


# ── Overfit smoke test ────────────────────────────────────────────────────────

def test_transformer_loss_decreases_on_tiny_corpus() -> None:
    """
    Verify that gradient descent reduces loss on a trivially small sequence.
    Uses a fixed random seed for reproducibility.
    """
    torch.manual_seed(42)
    vocab_size = 8
    block_size = 4

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    x = torch.randint(0, vocab_size, (4, block_size))
    y = torch.randint(0, vocab_size, (4, block_size))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()

    _, loss_before = model(x, y)
    assert loss_before is not None

    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, loss_after = model(x, y)

    assert loss_after is not None
    assert loss_after.item() < loss_before.item(), (
        f"Expected loss to decrease but got {loss_after.item():.4f} >= {loss_before.item():.4f}"
    )


# ── RoPE norm-preservation test ───────────────────────────────────────────────

def test_rope_preserves_pair_norms() -> None:
    """
    apply_rope applies an independent 2-D rotation to each (x[..., 2i], x[..., 2i+1]) pair.
    A 2-D rotation preserves the Euclidean norm of each pair, so the pair-wise norms
    of the output must equal those of the input up to floating-point tolerance.
    """
    torch.manual_seed(7)
    batch, heads, seq_len, head_dim = 2, 4, 6, 8

    x = torch.randn(batch, heads, seq_len, head_dim)
    x_rot = apply_rope(x)

    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    r_even = x_rot[..., 0::2]
    r_odd  = x_rot[..., 1::2]

    norms_before = (x_even ** 2 + x_odd ** 2).sqrt()
    norms_after  = (r_even ** 2 + r_odd  ** 2).sqrt()

    assert torch.allclose(norms_before, norms_after, atol=1e-5), (
        "RoPE did not preserve pair norms."
    )
