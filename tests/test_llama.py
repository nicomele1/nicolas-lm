import pytest
import torch

from nicolasm.models.llama import (
    LlamaStyleCausalSelfAttention,
    LlamaStyleLanguageModel,
    LlamaStyleTransformerBlock,
)
from nicolasm.modules.rmsnorm import RMSNorm
from nicolasm.modules.rope import apply_rope
from nicolasm.modules.swiglu import SwiGLU


def test_rmsnorm_output_shape() -> None:
    norm = RMSNorm(embedding_dim=8)
    x = torch.randn(2, 4, 8)

    assert norm(x).shape == x.shape


def test_swiglu_output_shape() -> None:
    swiglu = SwiGLU(embedding_dim=8, hidden_dim=32)
    x = torch.randn(2, 4, 8)

    assert swiglu(x).shape == x.shape


def test_apply_rope_preserves_shape() -> None:
    x = torch.randn(2, 3, 4, 8)

    assert apply_rope(x).shape == x.shape


def test_apply_rope_rejects_odd_head_dim() -> None:
    x = torch.randn(2, 3, 4, 7)

    with pytest.raises(ValueError):
        apply_rope(x)


def test_llama_style_attention_output_shape() -> None:
    attention = LlamaStyleCausalSelfAttention(
        embedding_dim=16,
        num_heads=4,
        block_size=8,
    )
    x = torch.randn(2, 5, 16)

    assert attention(x).shape == x.shape


def test_llama_style_block_output_shape() -> None:
    block = LlamaStyleTransformerBlock(
        embedding_dim=16,
        num_heads=4,
        block_size=8,
    )
    x = torch.randn(2, 5, 16)

    assert block(x).shape == x.shape


def test_llama_style_language_model_logits_shape() -> None:
    model = LlamaStyleLanguageModel(
        vocab_size=10,
        block_size=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )
    idx = torch.randint(0, 10, (2, 5))

    logits, loss = model(idx)

    assert logits.shape == (2, 5, 10)
    assert loss is None


def test_llama_style_language_model_with_targets_returns_loss() -> None:
    model = LlamaStyleLanguageModel(
        vocab_size=10,
        block_size=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )
    idx = torch.randint(0, 10, (2, 5))
    targets = torch.randint(0, 10, (2, 5))

    _, loss = model(idx, targets)

    assert loss is not None
    assert loss.ndim == 0


def test_llama_style_language_model_generate_shape() -> None:
    model = LlamaStyleLanguageModel(
        vocab_size=10,
        block_size=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )
    idx = torch.tensor([[0, 1, 2]])

    generated = model.generate(idx, max_new_tokens=5)

    assert generated.shape == (1, 8)


def test_llama_style_language_model_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError):
        LlamaStyleLanguageModel(vocab_size=0, block_size=8)

    with pytest.raises(ValueError):
        LlamaStyleLanguageModel(vocab_size=10, block_size=0)

    with pytest.raises(ValueError):
        LlamaStyleLanguageModel(
            vocab_size=10,
            block_size=8,
            embedding_dim=12,
            num_heads=4,
        )
