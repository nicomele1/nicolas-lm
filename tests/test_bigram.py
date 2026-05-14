import pytest
import torch

from nicolasm.models.bigram import BigramLanguageModel


def test_bigram_forward_logits_shape() -> None:
    vocab_size = 5
    model = BigramLanguageModel(vocab_size=vocab_size)

    idx = torch.tensor([[0, 1, 2], [2, 3, 4]])

    logits, loss = model(idx)

    assert logits.shape == (2, 3, vocab_size)
    assert loss is None


def test_bigram_forward_with_targets_returns_loss() -> None:
    vocab_size = 5
    model = BigramLanguageModel(vocab_size=vocab_size)

    idx = torch.tensor([[0, 1, 2], [2, 3, 4]])
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]])

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 3, vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_bigram_generate_shape() -> None:
    vocab_size = 5
    model = BigramLanguageModel(vocab_size=vocab_size)

    idx = torch.tensor([[0, 1, 2]])

    generated = model.generate(idx, max_new_tokens=4)

    assert generated.shape == (1, 7)


def test_bigram_rejects_invalid_vocab_size() -> None:
    with pytest.raises(ValueError):
        BigramLanguageModel(vocab_size=0)


def test_bigram_rejects_bad_idx_shape() -> None:
    model = BigramLanguageModel(vocab_size=5)

    idx = torch.tensor([0, 1, 2])

    with pytest.raises(ValueError):
        model(idx)


def test_bigram_rejects_target_shape_mismatch() -> None:
    model = BigramLanguageModel(vocab_size=5)

    idx = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2]])

    with pytest.raises(ValueError):
        model(idx, targets)