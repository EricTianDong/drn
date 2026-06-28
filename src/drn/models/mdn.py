
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from .base import BaseModel
from .layers import build_hidden_layers


class MDN(BaseModel):
    """
    Mixture density network that can switch between gamma and Gaussian distribution components.
    The distributional forecasts are mixtures of `num_components` specified distributions.
    """

    def __init__(
        self,
        distribution="gamma",
        num_hidden_layers=None,
        num_components=5,
        hidden_size: int | list[int] = 100,
        dropout_rate=0.0,
        weight_decay=0.0,
        learning_rate=1e-3,
        sigma_alpha=0.0,
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
            if num_hidden_layers is not None and num_hidden_layers != len(hidden_size):
                raise ValueError(
                    f"num_hidden_layers={num_hidden_layers} doesn't match length of hidden_size list ({len(hidden_size)}). "
                    "num_hidden_layers is not necessary when hidden_size is a list."
                )
            sizes = hidden_size
        else:
            if num_hidden_layers is None:
                num_hidden_layers = 2
            sizes = [hidden_size] * num_hidden_layers

        self.hidden_layers = build_hidden_layers(sizes, dropout_rate)

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
        self.weight_decay = weight_decay
        self.sigma_alpha = sigma_alpha

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
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
        params = self(x)
        nll = self.loss_fn(params, y)

        if self.training and self.sigma_alpha > 0:
            # Sigma activity regularisation: L2 penalty on the scale parameters.
            # params layout: [weights, alphas, betas] (gamma) or [weights, mus, sigmas] (gaussian)
            scales = params[2]  # betas for gamma, sigmas for gaussian
            nll = nll + self.sigma_alpha * (scales ** 2).sum(dim=1).mean()

        return nll

    def mean(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
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
