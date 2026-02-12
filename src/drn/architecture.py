"""Utilities for parametrising feedforward networks by total parameter count."""

from __future__ import annotations

import math


def count_params(input_dim: int, hidden_sizes: list[int], output_dim: int) -> int:
    """Count total weights + biases for a feedforward network.

    Args:
        input_dim: Number of input features.
        hidden_sizes: Width of each hidden layer.
        output_dim: Number of output neurons.

    Returns:
        Total number of trainable parameters.
    """
    if not hidden_sizes:
        raise ValueError("hidden_sizes must be non-empty")

    total = 0
    prev = input_dim
    for h in hidden_sizes:
        total += prev * h + h  # weights + biases
        prev = h
    total += prev * output_dim + output_dim  # output layer
    return total


def _rectangular_hidden_size(
    total_params: int, input_dim: int, output_dim: int, num_layers: int
) -> int:
    """Solve for uniform hidden width h that stays within total_params budget.

    For L hidden layers the param count is:
        L=1:  (d_in + 1)*h + (h + 1)*d_out = (d_in + d_out + 1)*h + d_out
        L>=2: (d_in + 1)*h + (L-1)*(h+1)*h + (h+1)*d_out
              = (L-1)*h^2 + (d_in + L + d_out)*h + d_out

    We solve for h (flooring to remain within budget).
    """
    d_in, d_out, L, P = input_dim, output_dim, num_layers, total_params

    if L == 1:
        # Linear: (d_in + d_out + 1)*h + d_out = P  =>  h = (P - d_out) / (d_in + d_out + 1)
        h = (P - d_out) / (d_in + d_out + 1)
    else:
        # Quadratic: (L-1)*h^2 + (d_in + L + d_out)*h + (d_out - P) = 0
        a = L - 1
        b = d_in + L + d_out
        c = d_out - P
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            raise ValueError(
                f"Cannot fit {num_layers} hidden layers within {total_params} params "
                f"(input_dim={input_dim}, output_dim={output_dim})"
            )
        h = (-b + math.sqrt(discriminant)) / (2 * a)

    h = int(h)  # floor
    if h < 1:
        raise ValueError(
            f"total_params={total_params} is too small for the given architecture "
            f"(input_dim={input_dim}, output_dim={output_dim}, num_layers={num_layers})"
        )
    return h


def _funnel_hidden_sizes(
    total_params: int, input_dim: int, output_dim: int, num_layers: int
) -> list[int]:
    """Compute funnel (tapering) hidden sizes via binary search on h1.

    Layer sizes decrease linearly from h1 to h1/2 (rounded).
    For L=1, degenerates to rectangular.
    """
    if num_layers == 1:
        h = _rectangular_hidden_size(total_params, input_dim, output_dim, 1)
        return [h]

    def _sizes_for_h1(h1: int) -> list[int]:
        """Generate linearly tapering sizes from h1 down to h1//2."""
        h_last = max(h1 // 2, 1)
        if num_layers == 1:
            return [h1]
        sizes = []
        for i in range(num_layers):
            t = i / (num_layers - 1)  # 0 to 1
            sizes.append(max(round(h1 + t * (h_last - h1)), 1))
        return sizes

    # Binary search on h1
    lo, hi = 1, total_params  # generous upper bound
    best_h1 = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        sizes = _sizes_for_h1(mid)
        p = count_params(input_dim, sizes, output_dim)
        if p <= total_params:
            best_h1 = mid
            lo = mid + 1
        else:
            hi = mid - 1

    result = _sizes_for_h1(best_h1)
    if result[0] < 1:
        raise ValueError(
            f"total_params={total_params} is too small for a funnel with "
            f"{num_layers} layers (input_dim={input_dim}, output_dim={output_dim})"
        )
    return result


def compute_hidden_sizes(
    total_params: int,
    input_dim: int,
    output_dim: int,
    num_layers: int,
    shape: str = "rectangular",
) -> list[int]:
    """Convert a total parameter budget into concrete hidden layer sizes.

    Args:
        total_params: Target number of trainable parameters (weights + biases).
        input_dim: Number of input features.
        output_dim: Number of output neurons.
        num_layers: Number of hidden layers.
        shape: ``"rectangular"`` (uniform width) or ``"funnel"`` (tapering).

    Returns:
        List of hidden layer widths (length == num_layers).

    Raises:
        ValueError: If the budget is too small or the shape is unknown.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if total_params < 1:
        raise ValueError("total_params must be >= 1")

    if shape == "rectangular":
        h = _rectangular_hidden_size(total_params, input_dim, output_dim, num_layers)
        return [h] * num_layers
    elif shape == "funnel":
        return _funnel_hidden_sizes(total_params, input_dim, output_dim, num_layers)
    else:
        raise ValueError(f"Unknown shape: {shape!r}. Expected 'rectangular' or 'funnel'.")
