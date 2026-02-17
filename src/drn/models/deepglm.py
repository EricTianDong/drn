
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import BaseModel
from .glm import gamma_deviance_loss, gaussian_deviance_loss
from ..distributions import inverse_gaussian
from ..distributions.estimation import gamma_convert_parameters, estimate_dispersion


class DeepGLM(BaseModel):
    """
    Deep Generalized Linear Model (DeepGLM).

    The model learns a nonlinear representation of the inputs via a feed-forward
    neural network and then applies a GLM head to produce the conditional mean.
    A fixed dispersion parameter is estimated after training via a classical
    deviance-based estimator (see `estimate_dispersion`).

    Supported response distributions:
        - 'gamma' (log link)
        - 'gaussian' (identity link)
        - 'inversegaussian' (log link)
        - 'lognormal' (identity on log-scale parameter; distributional head is LogNormal)
    """

    def __init__(
        self,
        distribution: str = "gamma",
        num_hidden_layers: int = 2,
        hidden_size: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        ct: object | None = None,
    ) -> None:
        self.save_hyperparameters()
        super(DeepGLM, self).__init__()

        if distribution not in ("gamma", "gaussian", "inversegaussian", "lognormal"):
            raise ValueError(f"Unsupported distribution: {distribution}")

        self.distribution = distribution
        self.learning_rate = learning_rate
        self.ct = ct  # optional ColumnTransformer used by BaseModel.preprocess

        # Representation network
        layers = [nn.LazyLinear(hidden_size), nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
            ]
        self.hidden = nn.Sequential(*layers)

        # GLM head
        self.head = nn.Linear(hidden_size, 1)

        # Non-trainable dispersion estimated post-hoc
        self.dispersion = nn.Parameter(torch.tensor([float("nan")]), requires_grad=False)

        # Loss choice (mean model only)
        if distribution == "gaussian":
            self.loss_fn = gaussian_deviance_loss
        elif distribution == "lognormal":
            self.loss_fn = gaussian_deviance_loss  # log-normal uses gaussian loss on log scale
        else:  # gamma, inversegaussian
            self.loss_fn = gamma_deviance_loss

    def fit(self, X_train, y_train, *args, **kwargs) -> "DeepGLM":
        super().fit(X_train, y_train, *args, **kwargs)
        self.update_dispersion(X_train, y_train)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        eta = self.head(h).squeeze(-1)
        if self.distribution in ("gamma", "inversegaussian"):
            return torch.exp(eta)
        return eta  # gaussian and lognormal

    def _predict(self, x: torch.Tensor):
        if torch.isnan(self.dispersion):
            raise RuntimeError("Dispersion parameter has not been estimated yet. "
                               "Call `update_dispersion(...)` after training.")

        if self.distribution == "gamma":
            alphas, betas = gamma_convert_parameters(self(x), self.dispersion)
            dists = torch.distributions.Gamma(alphas, betas)
        elif self.distribution == "inversegaussian":
            dists = inverse_gaussian.InverseGaussian(self(x), self.dispersion)
        elif self.distribution == "lognormal":
            dists = torch.distributions.LogNormal(self(x), self.dispersion)
        else:
            dists = torch.distributions.Normal(self(x), self.dispersion)

        assert dists.batch_shape == torch.Size([x.shape[0]])
        return dists

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.distribution == "lognormal":
            return self.loss_fn(self(x), torch.log(y))
        return self.loss_fn(self(x), y)

    def update_dispersion(
        self,
        X_train: np.ndarray | pd.DataFrame | torch.Tensor,
        y_train: np.ndarray | pd.Series | torch.Tensor,
    ) -> None:
        X = self.preprocess(X_train)
        y = self.preprocess(y_train, targets=True)
        # For lognormal, pass the log-transformed targets to estimate_dispersion
        if self.distribution == "lognormal":
            y_for_disp = torch.log(y)
        else:
            y_for_disp = y
        disp = estimate_dispersion(self.distribution, self(X), y_for_disp, X.shape[1])
        self.dispersion = nn.Parameter(torch.tensor([disp]), requires_grad=False)

    def mean(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self(x).detach().cpu().numpy().squeeze()

    def clone(self) -> "DeepGLM":
        # Count the actual number of linear layers in the hidden network
        linear_layers = [l for l in self.hidden if isinstance(l, nn.Linear)]
        num_hidden_layers = len(linear_layers)
        
        m = DeepGLM(
            distribution=self.distribution,
            num_hidden_layers=num_hidden_layers,
            hidden_size=self.head.in_features if isinstance(self.head, nn.Linear) else 128,
            dropout_rate=next((l.p for l in self.hidden if isinstance(l, nn.Dropout)), 0.0),
            learning_rate=self.learning_rate,
            ct=self.ct,
        )
        # Need to initialize the lazy layer first by doing a forward pass
        # Get the input size from the first linear layer's weight
        if hasattr(self.hidden[0], 'weight') and self.hidden[0].weight is not None:
            input_size = self.hidden[0].weight.shape[1]
            dummy_input = torch.zeros(1, input_size)
            m(dummy_input)  # Initialize lazy layer
        
        m.load_state_dict(self.state_dict())
        return m
