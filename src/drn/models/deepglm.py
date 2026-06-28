
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import BaseModel
from .glm import gamma_deviance_loss, gaussian_deviance_loss
from .layers import build_hidden_layers
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
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-3,
        ct: object | None = None,
    ) -> None:
        self.save_hyperparameters()
        super(DeepGLM, self).__init__()

        if distribution not in ("gamma", "gaussian", "inversegaussian", "lognormal"):
            raise ValueError(f"Unsupported distribution: {distribution}")

        self.distribution = distribution
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ct = ct  # optional ColumnTransformer used by BaseModel.preprocess

        # Representation network
        self.hidden_layers = build_hidden_layers(
            [hidden_size] * num_hidden_layers, dropout_rate
        )

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
        h = self.hidden_layers(x)
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
        m = DeepGLM(
            distribution=self.hparams.distribution,
            num_hidden_layers=self.hparams.num_hidden_layers,
            hidden_size=self.hparams.hidden_size,
            dropout_rate=self.hparams.dropout_rate,
            weight_decay=self.hparams.weight_decay,
            learning_rate=self.hparams.learning_rate,
            ct=self.ct,
        )
        # Initialize the lazy layer so state_dict keys match
        first_layer = self.hidden_layers[0]
        if hasattr(first_layer, 'weight') and first_layer.weight is not None:
            input_size = first_layer.weight.shape[1]
            dummy_input = torch.zeros(1, input_size)
            m(dummy_input)

        m.load_state_dict(self.state_dict())
        return m
