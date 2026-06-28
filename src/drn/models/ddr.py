from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..distributions.histogram import Histogram
from .base import BaseModel
from .layers import build_hidden_layers


class DDR(BaseModel):
    def __init__(
        self,
        cutpoints: list[float] | None = None,
        num_hidden_layers=2,
        hidden_size=100,
        dropout_rate=0.0,
        weight_decay=0.0,
        proportion=0.1,
        loss_metric="jbce",
        learning_rate=1e-3,
    ):
        """
        Args:
            x_train_shape: The shape of the training data, used to define the input size of the first layer.
            cutpoints: The cutpoints for the DDR model.
            num_hidden_layers: The number of hidden layers in the network.
            hidden_size: The number of neurons in each hidden layer.

        """
        # Can't put numpy float dtypes in the checkpoints
        if cutpoints is not None:
            cutpoints = [float(c) for c in cutpoints]  # strips numpy scalars

        self.save_hyperparameters()
        super(DDR, self).__init__()

        self.hidden_layers = build_hidden_layers(
            [hidden_size] * num_hidden_layers, dropout_rate
        )

        if cutpoints is not None:
            self.cutpoints = nn.Parameter(torch.Tensor(cutpoints), requires_grad=False)
            self.pi = nn.Linear(hidden_size, len(self.cutpoints) - 1)
        else:
            self.cutpoints = None
            self.pi = None

            self.hidden_size = hidden_size
            self.proportion = proportion

        # Assert that loss_metric is either 'jbce' or 'nll'
        if loss_metric not in ["jbce", "nll"]:
            raise ValueError(f"Unsupported loss metric: {loss_metric}")
        self.loss_metric = loss_metric
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def fit(self, X_train, y_train, *args, **kwargs) -> DDR:
        if self.cutpoints is None:
            c_0 = min(y_train.min().item() * 1.05, 0)
            c_K = y_train.max().item() * 1.05
            cutpoints = ddr_cutpoints(
                c_0=c_0, c_K=c_K, proportion=self.proportion, n=len(y_train)
            )
            self.cutpoints = nn.Parameter(torch.Tensor(cutpoints), requires_grad=False)
            self.pi = nn.Linear(self.hidden_size, len(self.cutpoints) - 1)

        return super().fit(X_train, y_train, *args, **kwargs)

    def forward(self, x):
        """
        Forward pass of the DDR model.
        Args:
            x: Input tensor.
        Returns:
            The cutpoints and probabilities for the DDR model.
        """
        if self.cutpoints is None or self.pi is None:
            raise ValueError(
                "Cutpoints must be defined before trying to make predictions."
            )

        # Pass input through the dynamically created hidden layers
        h = self.hidden_layers(x)

        # Calculate probabilities using the final layer
        probs = torch.softmax(self.pi(h), dim=1)

        return self.cutpoints, probs

    def _predict(self, x: torch.Tensor) -> Histogram:
        cutpoints, prob_masses = self(x)
        dists = Histogram(cutpoints, prob_masses)
        assert dists.batch_shape == torch.Size([x.shape[0]])
        return dists

    def loss(self, x, y):
        dists = self.predict(x)
        if self.loss_metric == "jbce":
            return jbce_loss(dists, y)
        else:
            return nll_loss(dists, y)


def jbce_loss(dists, y, alpha=0.0):
    """
    The joint binary cross entropy loss.
    Args:
        dists: the predicted distributions
        y: the observed values
        alpha: the penalty parameter
    """

    cutpoints = dists.cutpoints
    cdf_at_cutpoints = dists.cdf_at_cutpoints()

    assert cdf_at_cutpoints.shape == torch.Size([len(cutpoints), len(y)])

    n = y.shape[0]
    C = len(cutpoints)

    # The cross entropy loss can't accept 0s or 1s for the cumulative probabilities.
    epsilon = 1e-15
    cdf_at_cutpoints = cdf_at_cutpoints.clamp(epsilon, 1 - epsilon)

    # Change: C to C-1
    losses = torch.zeros(C - 1, n, device=y.device, dtype=y.dtype)

    for i in range(1, C):
        targets = (y <= cutpoints[i]).float()
        probs = cdf_at_cutpoints[i, :]
        losses[i - 1, :] = nn.functional.binary_cross_entropy(
            probs, targets, reduction="none"
        )

    return torch.mean(losses)


def ddr_loss(pred, y, alpha=0.0):
    cutpoints, prob_masses = pred
    dists = Histogram(cutpoints, prob_masses)
    return jbce_loss(dists, y, alpha)


def nll_loss(dists, y, alpha=0.0):
    losses = -(dists.log_prob(y))
    return torch.mean(losses)


def ddr_cutpoints(c_0: float, c_K: float, proportion: float, n: int) -> list[float]:
    """
    Generate cutpoints for the DDR model.
    Args:
        c_0: The minimum cutpoint.
        c_K: The maximum cutpoint.
        proportion: Number of cutpoints is this proportion of the training set size.
        n: The number of training observations.
    """
    num_cutpoints = int(np.ceil(proportion * n))
    cutpoints = list(np.linspace(c_0, c_K, num_cutpoints))

    return cutpoints
