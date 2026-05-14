import pytest
import torch

from nicolasm.models.transformer import (
    FeedForward,
    TinyTransformerLanguageModel,
    TransformerBlock,
)


def test_feedforward_output_shape() -> None:
    batch_size = 2
    sequence_length = 4
    embedding_dim = 8

    ff = FeedForward(
        embedding_dim=embedding_dim,
        hidden_dim=32,
    )

    x = torch.randn(batch_size, sequence_length, embedding_dim)
    out = ff(x)

    assert out.shape == (batch_size, sequence_length, embedding_dim)


def test_transformer_block_output_shape() -> None:
    batch_size = 2
    block_size = 4
    embedding_dim = 8
    num_heads = 2

    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        block_size=block_size,
    )

    x = torch.randn(batch_size, block_size, embedding_dim)
    out = block(x)

    assert out.shape == (batch_size, block_size, embedding_dim)


def test_tiny_transformer_logits_shape() -> None:
    vocab_size = 10
    block_size = 8
    batch_size = 2
    sequence_length = 5

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    idx = torch.randint(0, vocab_size, (batch_size, sequence_length))

    logits, loss = model(idx)

    assert logits.shape == (batch_size, sequence_length, vocab_size)
    assert loss is None


def test_tiny_transformer_with_targets_returns_loss() -> None:
    vocab_size = 10
    block_size = 8
    batch_size = 2
    sequence_length = 5

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    idx = torch.randint(0, vocab_size, (batch_size, sequence_length))
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))

    logits, loss = model(idx, targets)

    assert logits.shape == (batch_size, sequence_length, vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_tiny_transformer_generate_shape() -> None:
    vocab_size = 10
    block_size = 8

    model = TinyTransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    idx = torch.tensor([[0, 1, 2]])

    generated = model.generate(idx, max_new_tokens=5)

    assert generated.shape == (1, 8)


def test_tiny_transformer_rejects_sequence_too_long() -> None:
    model = TinyTransformerLanguageModel(
        vocab_size=10,
        block_size=4,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    idx = torch.randint(0, 10, (2, 5))

    with pytest.raises(ValueError):
        model(idx)


def test_tiny_transformer_rejects_target_shape_mismatch() -> None:
    model = TinyTransformerLanguageModel(
        vocab_size=10,
        block_size=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )

    idx = torch.randint(0, 10, (2, 5))
    targets = torch.randint(0, 10, (2, 4))

    with pytest.raises(ValueError):
        model(idx, targets)


def test_tiny_transformer_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(vocab_size=0, block_size=8)

    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(vocab_size=10, block_size=0)

    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(vocab_size=10, block_size=8, embedding_dim=0)

    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=10,
            block_size=8,
            embedding_dim=16,
            num_heads=0,
        )

    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=10,
            block_size=8,
            embedding_dim=10,
            num_heads=3,
        )

    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=10,
            block_size=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=0,
        )

    with pytest.raises(ValueError):
        TinyTransformerLanguageModel(
            vocab_size=10,
            block_size=8,
            embedding_dim=16,
            num_heads=4,
            dropout=1.0,
        )