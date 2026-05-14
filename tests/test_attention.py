import pytest
import torch

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