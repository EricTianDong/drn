from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
from statsmodels.genmod.families import Gamma, Gaussian, InverseGaussian

from .base import BaseModel
from ..distributions import inverse_gaussian
from ..distributions.estimation import gamma_convert_parameters, estimate_dispersion
from ..utils import _to_numpy


class GLM(BaseModel):
    def __init__(self, distribution: str, learning_rate=1e-3):
        self.save_hyperparameters()

        if distribution not in ("gamma", "gaussian", "inversegaussian", "lognormal"):
            raise ValueError(f"Unsupported model type: {distribution}")

        super(GLM, self).__init__()
        self.distribution = distribution

        # Set default dispersion to 1 for numerical stability
        self.dispersion = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

        if self.distribution == "inversegaussian":
            self.loss_fn = inverse_gaussian_deviance_loss
        else:
            self.loss_fn = (
                gaussian_deviance_loss
                if self.distribution == "gaussian"
                else gamma_deviance_loss
            )

        self.learning_rate = learning_rate
        self.linear = nn.LazyLinear(1)

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        grad_descent: bool = False,
        *args,
        **kwargs,
    ) -> GLM:
        # If the user specifically wants to use gradient descent, we will use the base class fit method
        if grad_descent:
            super().fit(X_train, y_train, *args, **kwargs)
            self.update_dispersion(X_train, y_train)
            return self

        # Otherwise, fit via statsmodels and directly assign parameters
        X_np = _to_numpy(X_train)
        y_np = _to_numpy(y_train).flatten()

        # Log-transform for lognormal
        if self.distribution == "lognormal":
            y_np = np.log(y_np)

        # Select family
        if self.distribution == "gamma":
            family = Gamma(link=sm.families.links.Log())
        elif self.distribution in ["gaussian", "lognormal"]:
            family = Gaussian()
        elif self.distribution == "inversegaussian":
            family = InverseGaussian(link=sm.families.links.Log())

        # Add constant for intercept
        X_sm = sm.add_constant(X_np)
        model = sm.GLM(y_np, X_sm, family=family)
        results = model.fit()

        # Extract coefficients
        betas = results.params
        betas = np.asarray(betas)

        # Assign to PyTorch model
        # weights: shape (1, p), bias: intercept
        self.linear = nn.Linear(X_train.shape[1], 1)
        self.linear.weight.data = torch.Tensor(betas[1:]).unsqueeze(0)
        self.linear.bias.data = torch.Tensor([betas[0]])

        # Dispersion: scale parameter
        if self.distribution in ("gamma", "inversegaussian"):
            disp = results.scale.item()
        else:  # gaussian, lognormal
            disp = (results.scale**0.5).item()

        self.dispersion = nn.Parameter(torch.Tensor([disp]), requires_grad=False)
        self.epochs_trained = None # Epochs not applicable
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distribution in ["gamma", "inversegaussian"]:
            return torch.exp(self.linear(x)).squeeze(-1)
        return self.linear(x).squeeze(-1)

    def clone(self) -> GLM:
        """
        Create an independent copy of the model.
        """
        glm = GLM(self.distribution)
        glm.load_state_dict(self.state_dict())
        return glm

    def _predict(self, x: torch.Tensor):
        if torch.isnan(self.dispersion):
            raise RuntimeError("Dispersion parameter has not been estimated yet.")

        if self.distribution == "gamma":
            alphas, betas = gamma_convert_parameters(self(x), self.dispersion)
            return torch.distributions.Gamma(alphas, betas)
        elif self.distribution == "inversegaussian":
            return inverse_gaussian.InverseGaussian(self(x), self.dispersion)
        elif self.distribution == "lognormal":
            return torch.distributions.LogNormal(self(x), self.dispersion)
        else:
            return torch.distributions.Normal(self(x), self.dispersion)

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self(X), y)

    def update_dispersion(
        self,
        X_train: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_train: Union[np.ndarray, pd.Series, torch.Tensor],
    ) -> None:
        X = self.preprocess(X_train)
        y = self.preprocess(y_train, targets=True)
        disp = estimate_dispersion(self.distribution, self(X), y, X_train.shape[1])
        self.dispersion = nn.Parameter(torch.Tensor([disp]), requires_grad=False)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted means for the given observations.
        """
        return self(x)


def gamma_deviance_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Tweedie deviance loss for the gamma distribution.
    Args:
        y_pred: the predicted values (shape: (n,))
        y_true: the observed values (shape: (n,))
    Returns:
        the deviance loss (shape: (,))
    """
    loss = 2 * (y_true / y_pred - torch.log(y_true / y_pred) - 1)
    return torch.mean(loss)


def gaussian_deviance_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Normal deviance loss for the Gaussian distribution.
    Args:
        y_pred: the predicted values (shape: (n,))
        y_true: the observed values (shape: (n,))
    Returns:
        the deviance loss (shape: (,))
    """
    loss = (y_true - y_pred) ** 2
    return torch.mean(loss)


def inverse_gaussian_deviance_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Inverse Gaussian (Tweedie p=3) unit deviance loss.

    Unit deviance per observation:
        d(y, mu) = (y - mu)^2 / (y * mu^2)

    Args:
        y_pred: predicted mean(s) mu, shape (n,) and strictly positive
        y_true: observed y, shape (n,) and strictly positive
    Returns:
        Scalar mean deviance over observations.
    Notes:
        The IG model is only defined for y_true > 0 and y_pred > 0.
        We clamp by a tiny epsilon for numerical stability.
    """
    # numerical safety for zeros/negatives
    eps = torch.finfo(y_pred.dtype).eps if y_pred.is_floating_point() else 1e-12
    mu = torch.clamp(y_pred, min=eps)
    y = torch.clamp(y_true, min=eps)

    loss = (y - mu).pow(2) / (y * mu.pow(2))
    return loss.mean()
