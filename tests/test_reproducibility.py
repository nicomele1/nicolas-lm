import random

import pytest
import torch

from nicolasm.reproducibility import seed_everything


def test_seed_everything_repeats_python_and_torch_sequences() -> None:
    seed_everything(42)
    first_python = [random.random() for _ in range(3)]
    first_torch = torch.rand(3)

    seed_everything(42)

    assert [random.random() for _ in range(3)] == first_python
    assert torch.equal(torch.rand(3), first_torch)


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        seed_everything(-1)
