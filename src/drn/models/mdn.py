from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily


class MDN(L.LightningModule):
    """
    Mixture density network that can switch between gamma and Gaussian distribution components.
    The distributional forecasts are mixtures of `num_components` specified distributions.
    """

    def __init__(
        self,
        p: int,
        num_hidden_layers=2,
        num_components=5,
        hidden_size=100,
        dropout_rate=0.2,
        distribution="gamma",
        learning_rate = 1e-3,
    ):
        """
        Args:
            p: the number of features in the model.
            num_hidden_layers: the number of hidden layers in the network.
            num_components: the number of components in the mixture.
            hidden_size: the number of neurons in each hidden layer.
            distribution: the type of distribution for the MDN ('gamma' or 'gaussian').
        """
        super(MDN, self).__init__()
        self.p = p
        self.num_components = num_components
        self.distribution = distribution
        self.learning_rate = learning_rate

        layers = [nn.Linear(p, hidden_size), nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            ]
        self.hidden_layers = nn.Sequential(*layers)

        # Output layers for mixture parameters
        self.logits = nn.Linear(hidden_size, num_components)
        if distribution == "gamma":
            self.log_alpha = nn.Linear(hidden_size, num_components)
            self.log_beta = nn.Linear(hidden_size, num_components)
        elif distribution == "gaussian":
            self.mu = nn.Linear(hidden_size, num_components)
            self.pre_sigma = nn.Linear(hidden_size, num_components)
        else:
            raise ValueError("Unsupported distribution: {}".format(distribution))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate the parameters of the mixture components.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            A list containing the mixture weights, and distribution-specific parameters.
        """
        x = self.hidden_layers(x)
        weights = torch.softmax(self.logits(x), dim=1)

        if self.distribution == "gamma":
            alphas = torch.exp(self.log_alpha(x))
            betas = torch.exp(self.log_beta(x))
            return [weights, alphas, betas]
        else:
            mus = self.mu(x)
            sigmas = nn.Softplus()(self.pre_sigma(x))  # Ensure sigma is positive
            return [weights, mus, sigmas]

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)

        loss_fn = gamma_mdn_loss if self.distribution == "gamma" else gaussian_mdn_loss
        loss = loss_fn(output, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)

        loss_fn = gamma_mdn_loss if self.distribution == "gamma" else gaussian_mdn_loss
        val_loss = loss_fn(output, y)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def distributions(self, x: torch.Tensor) -> MixtureSameFamily:
        """
        Create distributional forecasts for the given inputs.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            the predicted mixture distributions.
        """
        params = self.forward(x)
        weights = params[0]
        mixture = Categorical(weights)

        if self.distribution == "gamma":
            components = torch.distributions.Gamma(params[1], params[2])
        else:
            components = torch.distributions.Normal(params[1], params[2])

        return MixtureSameFamily(mixture, components)

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
        distributions = self.distributions(x)
        return distributions.mean.detach().numpy()

    def icdf(
        self,
        x: Union[np.ndarray, torch.Tensor],
        p,
        l=None,
        u=None,
        max_iter=1000,
        tolerance=1e-7,
    ) -> torch.Tensor:
        """
        Calculate the inverse CDF (quantiles) of the distribution for the given cumulative probability.

        Args:
            p: cumulative probability values at which to evaluate icdf
            l: lower bound for the quantile search
            u: upper bound for the quantile search
            max_iter: maximum number of iterations permitted for the quantile search
            tolerance: stopping criteria for the search (precision)

        Returns:
            A tensor of shape (1, batch_shape) containing the inverse CDF values.
        """

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        dists = self.distributions(x)
        num_observations = dists.cdf(torch.Tensor([1]).unsqueeze(-1)).shape[
            1
        ]  # Dummy call to cdf to determine the batch size
        percentiles_tensor = torch.full(
            (1, num_observations), fill_value=p, dtype=torch.float32
        )

        # Initialise matrices for the bounds
        lower_bounds = (
            l if l is not None else torch.Tensor([0])
        )  # self.cutpoints[0] - (self.cutpoints[-1]-self.cutpoints[0])
        upper_bounds = (
            u if u is not None else torch.Tensor([200])
        )  # Adjust max value as needed

        lower_bounds = lower_bounds.repeat(num_observations).reshape(
            1, num_observations
        )
        upper_bounds = upper_bounds.repeat(num_observations).reshape(
            1, num_observations
        )

        for _ in range(max_iter):
            mid_points = (lower_bounds + upper_bounds) / 2
            cdf_vals = dists.cdf(mid_points)

            # Update the bounds based on where the CDF values are relative to the target percentiles
            lower_update = cdf_vals < percentiles_tensor
            upper_update = ~lower_update
            lower_bounds = torch.where(lower_update, mid_points, lower_bounds)
            upper_bounds = torch.where(upper_update, mid_points, upper_bounds)

            # Check for convergence
            if torch.max(upper_bounds - lower_bounds) < tolerance:
                break

        # Use the midpoint between the final bounds as the quantile estimate
        quantiles = (lower_bounds + upper_bounds) / 2

        return quantiles

    def quantiles(
        self,
        x: torch.Tensor,
        percentiles: list,
        l=None,
        u=None,
        max_iter=1000,
        tolerance=1e-7,
    ) -> torch.Tensor:
        """
        Calculate the quantile values for the given observations and percentiles (cumulative probabilities * 100).
        """
        l = (
            torch.Tensor([0]) if l is None else l
        )  # self.cutpoints[0] - (self.cutpoints[-1]-self.cutpoints[0]) if l is None else l
        u = torch.Tensor([200]) if u is None else u
        quantiles = [
            self.icdf(x, torch.tensor(percentile / 100.0), l, u, max_iter, tolerance)
            for percentile in percentiles
        ]

        return torch.stack(quantiles, dim=1)[0]


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
        Categorical(weights),
        torch.distributions.Gamma(alphas, betas),
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
        Categorical(weights),
        torch.distributions.Normal(mus, sigmas),
    )
    log_prob = dists.log_prob(y.squeeze())
    assert log_prob.ndim == 1
    return -log_prob.mean()
