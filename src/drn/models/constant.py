from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Union, Any
from .base import BaseModel
from torch.distributions import Gamma, Normal
from ..utils import _to_numpy
from ..distributions.estimation import estimate_dispersion, gamma_convert_parameters
from ..distributions import InverseGaussian


class Constant(BaseModel):
    """
    A baseline model that predicts a constant distribution, ignoring covariates.

    - Before fit(): mean = 1 (Gamma/IG), 0 (Gaussian); dispersion = 1.
    - fit(): ignores X, sets mean to y_train.mean(), estimates dispersion via estimate_dispersion.
    """

    def __init__(self, distribution: str):
        super().__init__()
        if distribution not in ("gamma", "gaussian", "inversegaussian"):
            raise ValueError(f"Unsupported distribution: {distribution}")
        self.distribution = distribution
        init = 0.0 if distribution == "gaussian" else 1.0
        self.mean_value = nn.Parameter(torch.tensor(init), requires_grad=False)
        self.dispersion = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def clone(self) -> Constant:
        """
        Create a copy of the model with the same parameters.
        """
        clone = Constant(self.distribution)
        clone.mean_value = nn.Parameter(
            self.mean_value.data.clone(), requires_grad=False
        )
        clone.dispersion = nn.Parameter(
            self.dispersion.data.clone(), requires_grad=False
        )
        return clone

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],  # ignored
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
        *args,
        **kwargs,
    ) -> Constant:
        # compute sample mean
        y_np = _to_numpy(y_train).flatten().astype(float)
        mean_y = y_np.mean()
        self.mean_value.data.fill_(mean_y)

        # estimate dispersion via shared routine (p=0 → dof = n-1)
        y_t = torch.from_numpy(y_np).float().to(self.mean_value.device)
        mu_t = torch.full_like(y_t, fill_value=mean_y)
        phi = estimate_dispersion(self.distribution, mu_t, y_t, p=0)
        self.dispersion.data.fill_(phi)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_value.expand(x.shape[0])

    def predict(self, x: Any) -> Any:
        """
        Create a (constant) distributional forecast for the given inputs.
        """
        return self._predict(x)

    def _predict(self, x: Any) -> Union[Gamma, Normal, InverseGaussian]:
        batch = self.mean_value.new_zeros(x.shape[0]) + self.mean_value
        phi = self.dispersion
        if self.distribution == "gamma":
            alpha, beta = gamma_convert_parameters(batch, phi)
            return Gamma(alpha, beta)
        if self.distribution == "inversegaussian":
            return InverseGaussian(batch, phi)
        # gaussian
        return Normal(batch, phi)

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.predict(X).log_prob(y).mean()

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted means for the given observations.
        """
        return self(x)
