from typing import Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..distributions.extended_histogram import ExtendedHistogram
from .ddr import jbce_loss, nll_loss
from .base import BaseModel


class DRN(BaseModel):
    def __init__(
        self,
        glm,
        cutpoints: Optional[list[float]] = None,
        num_hidden_layers=2,
        hidden_size=75,
        dropout_rate=0.2,
        baseline_start=False,
        proportion=0.1,
        min_obs=1,
        loss_metric="jbce",
        kl_alpha=0.0,
        mean_alpha=0.0,
        tv_alpha=0.0,
        dv_alpha=0.0,
        kl_direction="forwards",
        learning_rate=1e-3,
        debug=False,
    ):
        """
        Args:
            glm: A Generalized Linear Model (GLM) that DRN will adjust.
            cutpoints: Cutpoints for the DRN model.
            num_hidden_layers: Number of hidden layers in the DRN network.
            hidden_size: Number of neurons in each hidden layer.
        """
        self.save_hyperparameters()
        super(DRN, self).__init__()
        self.glm = glm.clone()

        for param in self.glm.parameters():
            param.requires_grad = False

        layers = [
            nn.Linear(glm.p, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))

        self.hidden_layers = nn.Sequential(*layers)

        if cutpoints is not None:
            self.cutpoints = nn.Parameter(torch.Tensor(cutpoints), requires_grad=False)
            self.fc_output = nn.Linear(hidden_size, len(self.cutpoints) - 1)
            # Initialize weights and biases for fc_output to zero
            if baseline_start:
                nn.init.constant_(self.fc_output.weight, 0)
                nn.init.constant_(self.fc_output.bias, 0)

        else:
            self.cutpoints = None
            self.fc_output = None
            self.hidden_size = hidden_size
            self.proportion = proportion
            self.min_obs = min_obs
            self.baseline_start = baseline_start

        self.loss_metric = loss_metric
        self.kl_alpha = kl_alpha
        self.mean_alpha = mean_alpha
        self.tv_alpha = tv_alpha
        self.dv_alpha = dv_alpha
        self.kl_direction = kl_direction
        self.learning_rate = learning_rate
        self.debug = debug

    def fit(self, X_train, y_train, *args, **kwargs):
        if self.cutpoints is None:
            c_0 = min(y_train.min().item() * 1.05, 0)
            c_K = y_train.max().item() * 1.05
            cutpoints = drn_cutpoints(
                c_0, c_K, y_train, proportion=self.proportion, min_obs=self.min_obs
            )
            self.cutpoints = nn.Parameter(torch.Tensor(cutpoints), requires_grad=False)
            self.fc_output = nn.Linear(self.hidden_size, len(self.cutpoints) - 1)
            if self.baseline_start:
                nn.init.constant_(self.fc_output.weight, 0)
                nn.init.constant_(self.fc_output.bias, 0)

        super().fit(X_train, y_train, *args, **kwargs)

    def log_adjustments(self, x):
        """
        Estimates log adjustments using the neural network.
        Args:
            x: Input features.
        Returns:
            Log adjustments for the DRN model.
        """
        # Pass input through the hidden layers
        z = self.hidden_layers(x)
        # Compute log adjustments
        log_adjustments = self.fc_output(z)
        return log_adjustments  # - torch.mean(log_adjustments, dim=1, keepdim=True)

    def forward(self, x):
        if self.cutpoints is None or self.fc_output is None:
            raise ValueError(
                "Cutpoints must be defined before trying to make predictions."
            )

        if self.debug:
            num_cutpoints = len(self.cutpoints)
            num_regions = len(self.cutpoints) - 1

        with torch.no_grad():
            baseline_dists = self.glm.distributions(x)

            baseline_cdfs = baseline_dists.cdf(self.cutpoints.unsqueeze(-1)).T
            if self.debug:
                assert baseline_cdfs.shape == (x.shape[0], num_cutpoints)

            baseline_probs = torch.diff(baseline_cdfs, dim=1)
            if self.debug:
                assert baseline_probs.shape == (x.shape[0], num_regions)

            # Sometimes the GLM probabilities are 0 simply due to numerical problems.
            # DRN cannot adjust regions with 0 probability, so we ensure 0's become
            # an incredibly small number just to avoid this issue.
            mass = torch.sum(baseline_probs, dim=1, keepdim=True)
            baseline_probs = torch.clip(baseline_probs, min=1e-10, max=1.0)
            baseline_probs = (
                baseline_probs / torch.sum(baseline_probs, dim=1, keepdim=True) * mass
            )

        drn_logits = torch.log(baseline_probs) + self.log_adjustments(x)
        drn_pmf = torch.softmax(drn_logits, dim=1)

        if self.debug:
            assert drn_pmf.shape == (x.shape[0], num_regions)

            # Sometimes we get nan value in here. Otherwise, it should sum to 1.
            assert torch.isnan(drn_pmf).any() or torch.allclose(
                torch.sum(drn_pmf, dim=1), torch.ones(x.shape[0], device=x.device)
            )

        return baseline_dists, self.cutpoints, baseline_probs, drn_pmf

    def distributions(
        self, x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
    ) -> ExtendedHistogram:
        x = self._to_tensor(x)
        baseline_dists, cutpoints, baseline_probs, drn_pmf = self.forward(x)
        return ExtendedHistogram(baseline_dists, cutpoints, drn_pmf, baseline_probs)

    def loss(self, x, y):
        if self.training:
            return drn_loss(
                self.forward(x),
                y,
                kind=self.loss_metric,
                kl_alpha=self.kl_alpha,
                mean_alpha=self.mean_alpha,
                tv_alpha=self.tv_alpha,
                dv_alpha=self.dv_alpha,
                kl_direction=self.kl_direction,
            )
        else:
            # Disable regularization during evaluation phase (e.g. validation set loss)
            return drn_loss(self.forward(x), y, kind=self.loss_metric)


def drn_loss(
    pred,
    y,
    kind="jbce",
    kl_alpha=0.0,
    mean_alpha=0.0,
    tv_alpha=0.0,
    dv_alpha=0.0,
    kl_direction="forwards",
):
    baseline_dists, cutpoints, baseline_probs, drn_pmf = pred
    dists = ExtendedHistogram(baseline_dists, cutpoints, drn_pmf, baseline_probs)

    if kind == "jbce":
        losses = jbce_loss(dists, y)
    else:
        losses = nll_loss(dists, y)

    reg_loss = 0.0
    epsilon = 1e-30
    a_i = dists.real_adjustments()
    b_i = baseline_probs

    if kl_alpha > 0:
        if kl_direction == "forwards":
            kl = -(torch.log(a_i + epsilon) * b_i)
        else:
            kl = torch.log(a_i + epsilon) * a_i * b_i
        reg_loss += torch.mean(torch.sum(kl, dim=1)) * kl_alpha

    if mean_alpha > 0:
        mean_penalty = torch.mean((baseline_dists.mean - dists.mean) ** 2)
        reg_loss += mean_alpha * mean_penalty

    if tv_alpha > 0 or dv_alpha > 0:
        drn_density = a_i * b_i / torch.diff(cutpoints)
        first_diffs = torch.diff(drn_density, dim=1)

        if tv_alpha > 0:
            tv_penalty = torch.mean(torch.sum(torch.abs(first_diffs), dim=1))
            reg_loss += tv_alpha * tv_penalty

        if dv_alpha > 0:
            second_diffs = torch.diff(first_diffs, dim=1)
            dv_penalty = torch.mean(torch.sum(second_diffs**2, dim=1))
            reg_loss += dv_alpha * dv_penalty

    return losses + reg_loss


def merge_cutpoints(cutpoints: list[float], y: np.ndarray, min_obs: int) -> list[float]:
    # Ensure cutpoints are sorted and unique to start with
    cutpoints = sorted(np.unique(cutpoints).tolist())
    assert len(cutpoints) >= 2

    new_cutpoints = [cutpoints[0]]  # Start with the first cutpoint
    left = 0

    for right in range(1, len(cutpoints) - 1):
        num_in_region = np.sum((y >= cutpoints[left]) & (y < cutpoints[right]))
        num_after_region = np.sum((y >= cutpoints[right]) & (y < cutpoints[-1]))

        if num_in_region >= min_obs and num_after_region >= min_obs:
            new_cutpoints.append(cutpoints[right])
            left = right

    new_cutpoints.append(cutpoints[-1])  # End with the last cutpoint

    return new_cutpoints


def drn_cutpoints(
    c_0: float,
    c_K: float,
    y: Union[np.ndarray, torch.Tensor],
    proportion: Optional[float] = None,
    num_cutpoints: Optional[int] = None,
    min_obs=1,
):
    if proportion is None and num_cutpoints is None:
        raise ValueError(
            "Either a proportion or a specific num_cutpoints must be provided."
        )

    if num_cutpoints is None and proportion is not None:
        num_cutpoints = int(np.ceil(proportion * len(y)))

    uniform_cutpoints = np.linspace(c_0, c_K, num_cutpoints).tolist()

    return merge_cutpoints(uniform_cutpoints, np.asarray(y), min_obs)
