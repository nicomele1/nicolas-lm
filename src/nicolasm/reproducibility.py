from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed every pseudo-random generator used by NicolasLM.

    The project currently relies on Python's ``random`` module for dataset
    sampling and on PyTorch for parameter initialization and token sampling.
    Keeping both in sync makes a run repeatable on the same software and
    hardware stack.
    """
    if seed < 0:
        raise ValueError("seed must be nonnegative.")

    random.seed(seed)
    torch.manual_seed(seed)

