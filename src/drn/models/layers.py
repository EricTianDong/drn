"""Shared building blocks for feedforward architectures."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A two-layer residual block.

    Computes::

        h = LeakyReLU(Linear_1(x))
        h = Linear_2(h)
        y = LeakyReLU(h + x)

    The skip path is identity-only; therefore ``in_features`` must match
    ``out_features``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        if in_features != out_features:
            raise ValueError(
                "ResidualBlock requires in_features == out_features "
                f"(got {in_features} != {out_features})"
            )
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return self.activation(out + x)


def build_hidden_layers(
    sizes: list[int],
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """Build hidden layers.

    Architecture::

        LazyLinear(sizes[0]) -> LeakyReLU -> Dropout(dropout_rate)
            -> (rectangular only) 2-layer ResidualBlock blocks
            -> (optional odd leftover) Linear -> LeakyReLU

    The first hidden layer is always an input projection. Residual blocks are
    used only for rectangular architectures (all hidden widths equal), so the
    skip path is always identity and adds no parameters. Non-rectangular
    architectures fall back to plain ``Linear -> LeakyReLU`` transitions.

    Args:
        sizes: Width of each hidden layer.
        dropout_rate: Dropout probability applied once after the input
            projection.

    Returns:
        An ``nn.Sequential`` that maps inputs to the last hidden
        representation (width ``sizes[-1]``).
    """
    if not sizes:
        raise ValueError("sizes must be non-empty")

    # Input projection: LazyLinear -> LeakyReLU -> Dropout
    layers: list[nn.Module] = [
        nn.LazyLinear(sizes[0]),
        nn.LeakyReLU(),
        nn.Dropout(dropout_rate),
    ]

    is_rectangular = len(set(sizes)) == 1
    if is_rectangular:
        i = 1
        while i + 1 < len(sizes):
            layers.append(ResidualBlock(sizes[i - 1], sizes[i], sizes[i + 1]))
            i += 2
        # Odd leftover hidden layer after pairing into residual blocks.
        if i < len(sizes):
            layers.extend([nn.Linear(sizes[i - 1], sizes[i]), nn.LeakyReLU()])
    else:
        for i in range(1, len(sizes)):
            layers.extend([nn.Linear(sizes[i - 1], sizes[i]), nn.LeakyReLU()])

    return nn.Sequential(*layers)
