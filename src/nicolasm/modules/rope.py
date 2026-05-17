from __future__ import annotations

import torch


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to a tensor of attention states.

    Parameters
    ----------
    x:
        Tensor of shape (..., sequence_length, head_dim). The head dimension
        must be even.
    """
    if x.ndim < 2:
        raise ValueError("x must have at least two dimensions.")

    sequence_length = x.shape[-2]
    head_dim = x.shape[-1]

    if head_dim <= 0:
        raise ValueError("head_dim must be positive.")

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even.")

    device = x.device
    dtype = x.dtype

    positions = torch.arange(sequence_length, device=device, dtype=dtype)
    frequencies = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inverse_frequencies = 1.0 / (10000.0 ** (frequencies / head_dim))
    angles = positions[:, None] * inverse_frequencies[None, :]

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
