import pytest
import torch

from nicolasm.modules.attention import CausalSelfAttentionHead
from nicolasm.modules.causal_mask import causal_mask


def test_causal_mask_shape() -> None:
    mask = causal_mask(block_size=4)

    assert mask.shape == (4, 4)


def test_causal_mask_is_boolean() -> None:
    mask = causal_mask(block_size=4)

    assert mask.dtype == torch.bool


def test_causal_mask_values() -> None:
    mask = causal_mask(block_size=4)

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(mask, expected)


def test_causal_mask_rejects_nonpositive_block_size() -> None:
    with pytest.raises(ValueError):
        causal_mask(block_size=0)


def test_attention_head_output_shape() -> None:
    batch_size = 2
    block_size = 4
    embedding_dim = 8
    head_dim = 3

    head = CausalSelfAttentionHead(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        block_size=block_size,
    )

    x = torch.randn(batch_size, block_size, embedding_dim)
    out = head(x)

    assert out.shape == (batch_size, block_size, head_dim)


def test_attention_head_accepts_shorter_sequence() -> None:
    batch_size = 2
    block_size = 8
    sequence_length = 4
    embedding_dim = 8
    head_dim = 3

    head = CausalSelfAttentionHead(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        block_size=block_size,
    )

    x = torch.randn(batch_size, sequence_length, embedding_dim)
    out = head(x)

    assert out.shape == (batch_size, sequence_length, head_dim)


def test_attention_head_rejects_bad_input_rank() -> None:
    head = CausalSelfAttentionHead(
        embedding_dim=8,
        head_dim=4,
        block_size=4,
    )

    x = torch.randn(4, 8)

    with pytest.raises(ValueError):
        head(x)


def test_attention_head_rejects_wrong_embedding_dim() -> None:
    head = CausalSelfAttentionHead(
        embedding_dim=8,
        head_dim=4,
        block_size=4,
    )

    x = torch.randn(2, 4, 7)

    with pytest.raises(ValueError):
        head(x)


def test_attention_head_rejects_sequence_too_long() -> None:
    head = CausalSelfAttentionHead(
        embedding_dim=8,
        head_dim=4,
        block_size=4,
    )

    x = torch.randn(2, 5, 8)

    with pytest.raises(ValueError):
        head(x)


def test_attention_head_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttentionHead(
            embedding_dim=0,
            head_dim=4,
            block_size=4,
        )

    with pytest.raises(ValueError):
        CausalSelfAttentionHead(
            embedding_dim=8,
            head_dim=0,
            block_size=4,
        )

    with pytest.raises(ValueError):
        CausalSelfAttentionHead(
            embedding_dim=8,
            head_dim=4,
            block_size=0,
        )


def test_attention_head_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttentionHead(
            embedding_dim=8,
            head_dim=4,
            block_size=4,
            dropout=1.0,
        )