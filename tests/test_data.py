import pytest

from nicolasm.data import TokenDataset, train_val_split


def test_dataset_length() -> None:
    tokens = [0, 1, 2, 3, 4]
    dataset = TokenDataset(tokens=tokens, block_size=2)

    assert len(dataset) == 3


def test_dataset_getitem() -> None:
    tokens = [0, 1, 2, 3, 4]
    dataset = TokenDataset(tokens=tokens, block_size=3)

    x, y = dataset[0]

    assert x == [0, 1, 2]
    assert y == [1, 2, 3]


def test_dataset_getitem_later_index() -> None:
    tokens = [0, 1, 2, 3, 4]
    dataset = TokenDataset(tokens=tokens, block_size=2)

    x, y = dataset[2]

    assert x == [2, 3]
    assert y == [3, 4]


def test_dataset_index_out_of_range() -> None:
    tokens = [0, 1, 2, 3, 4]
    dataset = TokenDataset(tokens=tokens, block_size=2)

    with pytest.raises(IndexError):
        _ = dataset[3]


def test_invalid_block_size() -> None:
    with pytest.raises(ValueError):
        TokenDataset(tokens=[0, 1, 2], block_size=0)


def test_tokens_too_short_for_block_size() -> None:
    with pytest.raises(ValueError):
        TokenDataset(tokens=[0, 1, 2], block_size=3)


def test_get_batch_shapes() -> None:
    tokens = [0, 1, 2, 3, 4, 5]
    dataset = TokenDataset(tokens=tokens, block_size=2)

    xs, ys = dataset.get_batch(batch_size=4)

    assert len(xs) == 4
    assert len(ys) == 4

    for x, y in zip(xs, ys):
        assert len(x) == 2
        assert len(y) == 2


def test_train_val_split() -> None:
    tokens = list(range(10))

    train_tokens, val_tokens = train_val_split(tokens, train_fraction=0.8)

    assert train_tokens == [0, 1, 2, 3, 4, 5, 6, 7]
    assert val_tokens == [8, 9]


def test_train_val_split_rejects_empty_tokens() -> None:
    with pytest.raises(ValueError):
        train_val_split([])


def test_train_val_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        train_val_split([0, 1, 2], train_fraction=1.0)