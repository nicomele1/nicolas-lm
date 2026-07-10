import math

import pytest

from nicolasm.metrics import nats_to_bits, unigram_cross_entropy


def test_nats_to_bits() -> None:
    assert nats_to_bits(math.log(2)) == pytest.approx(1.0)


def test_unigram_cross_entropy_matches_add_one_formula() -> None:
    # Training counts are a: 3, b: 1. Add-one probabilities are 4/6 and 2/6.
    loss = unigram_cross_entropy([0, 0, 0, 1], [0, 1], vocab_size=2)
    expected = -(math.log(4 / 6) + math.log(2 / 6)) / 2
    assert loss == pytest.approx(expected)


def test_unigram_cross_entropy_handles_unseen_test_token() -> None:
    loss = unigram_cross_entropy([0, 0], [1], vocab_size=2)
    assert loss == pytest.approx(-math.log(1 / 4))


@pytest.mark.parametrize(
    "train,test,vocab,alpha",
    [([], [0], 1, 1.0), ([0], [], 1, 1.0), ([0], [0], 0, 1.0), ([0], [0], 1, 0.0)],
)
def test_unigram_cross_entropy_validates_inputs(train, test, vocab, alpha) -> None:
    with pytest.raises(ValueError):
        unigram_cross_entropy(train, test, vocab, alpha)
