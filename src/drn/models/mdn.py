from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from .base import BaseModel


class MDN(BaseModel):
    """
    Mixture density network that can switch between gamma and Gaussian distribution components.
    The distributional forecasts are mixtures of `num_components` specified distributions.
    """

    def __init__(
        self,
        distribution="gamma",
        num_hidden_layers=2,
        num_components=5,
        hidden_size: int | list[int] = 100,
        dropout_rate=0.2,
        learning_rate=1e-3,
    ):
        """
        Args:
            p: the number of features in the model.
            num_hidden_layers: the number of hidden layers in the network.
            num_components: the number of components in the mixture.
            hidden_size: either a single int (uniform width, repeated for
                num_hidden_layers) or a list of ints (explicit per-layer widths,
                num_hidden_layers is ignored).
            distribution: the type of distribution for the MDN ('gamma' or 'gaussian').
        """
        self.save_hyperparameters()
        super(MDN, self).__init__()
        self.num_components = num_components
        self.distribution = distribution

        # Resolve hidden layer sizes
        if isinstance(hidden_size, list):
            sizes = hidden_size
        else:
            sizes = [hidden_size] * num_hidden_layers

        layers = [nn.LazyLinear(sizes[0]), nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for i in range(1, len(sizes)):
            layers += [
                nn.Linear(sizes[i - 1], sizes[i]),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            ]
        self.hidden_layers = nn.Sequential(*layers)

        last_hidden = sizes[-1]

        # Output layers for mixture parameters
        self.logits = nn.Linear(last_hidden, num_components)
        if distribution == "gamma":
            self.log_alpha = nn.Linear(last_hidden, num_components)
            self.log_beta = nn.Linear(last_hidden, num_components)
        elif distribution == "gaussian":
            self.mu = nn.Linear(last_hidden, num_components)
            self.pre_sigma = nn.Linear(last_hidden, num_components)
        else:
            raise ValueError("Unsupported distribution: {}".format(distribution))

        self.loss_fn = gamma_mdn_loss if distribution == "gamma" else gaussian_mdn_loss
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate the parameters of the mixture components.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            A list containing the mixture weights, and distribution-specific parameters.
        """
        x = self.hidden_layers(x)
        weights = torch.softmax(self.logits(x), dim=1) + 1e-9

        if self.distribution == "gamma":
            alphas = torch.exp(self.log_alpha(x)) + 1e-6
            betas = torch.exp(self.log_beta(x)) + 1e-6
            return [weights, alphas, betas]
        else:
            mus = self.mu(x)
            sigmas = nn.Softplus()(self.pre_sigma(x)) + 1e-6 # Ensure sigma is positive
            return [weights, mus, sigmas]

    def _predict(self, x: torch.Tensor) -> MixtureSameFamily:
        """
        Create distributional forecasts for the given inputs.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            the predicted mixture distributions.
        """
        params = self(x)
        weights = params[0]
        mixture = Categorical(weights)

        if self.distribution == "gamma":
            components = torch.distributions.Gamma(params[1], params[2])
        else:
            components = torch.distributions.Normal(params[1], params[2])

        return MixtureSameFamily(mixture, components)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self(x), y)

    def mean(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Calculate the predicted means for the given observations, depending on the mixture distribution.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            the predicted means (shape: (n,))
        """
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        distributions = self.predict(x)
        return distributions.mean.detach().numpy()


def gamma_mdn_loss(out: list[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative log-likelihood loss for the mixture density network.
    Args:
        out: the mixture weights, shape parameters and rate parameters (all shape: (n, num_components))
        y: the observed values (shape: (n, 1))
    Returns:
        the negative log-likelihood loss (shape: (1,))
    """
    weights, alphas, betas = out
    dists = MixtureSameFamily(
        Categorical(weights), torch.distributions.Gamma(alphas, betas)
    )
    log_prob = dists.log_prob(y.squeeze())
    assert log_prob.ndim == 1
    return -log_prob.mean()


def gaussian_mdn_loss(out: list[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative log-likelihood loss for the mixture density network.
    Args:
        out: the mixture weights, shape parameters and rate parameters (all shape: (n, num_components))
        y: the observed values (shape: (n, 1))
    Returns:
        the negative log-likelihood loss (shape: (1,))
    """
    weights, mus, sigmas = out
    dists = MixtureSameFamily(
        Categorical(weights), torch.distributions.Normal(mus, sigmas)
    )
    log_prob = dists.log_prob(y.squeeze())
    assert log_prob.ndim == 1
    return -log_prob.mean()
