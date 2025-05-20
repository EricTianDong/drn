import numpy as np
import torch
import torch.nn as nn
import lightning as L

from ..distributions.extended_histogram import ExtendedHistogram
from .ddr import jbce_loss, nll_loss


class DRN(L.LightningModule):
    def __init__(
        self,
        num_features,
        cutpoints,
        glm,
        num_hidden_layers=2,
        hidden_size=75,
        dropout_rate=0.2,
        baseline_start=False,
        learning_rate=1e-3,
        loss_fn="jbce",
        kl_alpha=0.0,
        mean_alpha=0.0,
        tv_alpha=0.0,
        dv_alpha=0.0,
        kl_direction="forwards",
        debug=False,
    ):
        """
        Args:
            num_features: Number of features in the input dataset.
            cutpoints: Cutpoints for the DRN model.
            glm: A Generalized Linear Model (GLM) that DRN will adjust.
            num_hidden_layers: Number of hidden layers in the DRN network.
            hidden_size: Number of neurons in each hidden layer.
        """
        super(DRN, self).__init__()
        self.cutpoints = nn.Parameter(torch.Tensor(cutpoints), requires_grad=False)
        # Assuming glm.clone() is a method to clone the glm model; ensure glm has a clone method.
        self.glm = glm.clone() if hasattr(glm, "clone") else glm

        for param in self.glm.parameters():
            param.requires_grad = False
            # assert param.requires_grad == False

        layers = [
            nn.Linear(num_features, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.fc_output = nn.Linear(hidden_size, len(self.cutpoints) - 1)
        self.batch_norm = nn.BatchNorm1d(len(cutpoints) - 1)

        # Initialize weights and biases for fc_output to zero
        if baseline_start:
            nn.init.constant_(self.fc_output.weight, 0)
            nn.init.constant_(self.fc_output.bias, 0)
        
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.kl_alpha = kl_alpha
        self.mean_alpha = mean_alpha
        self.tv_alpha = tv_alpha
        self.dv_alpha = dv_alpha
        self.kl_direction = kl_direction
        self.debug = debug

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
        return log_adjustments - torch.mean(log_adjustments, axis=1, keepdim=True)

        # normalized_log_adjustments = self.batch_norm(log_adjustments)
        # return normalized_log_adjustments

    def forward(self, x):
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
            mass = torch.sum(baseline_probs, axis=1, keepdim=True)
            baseline_probs = torch.clip(baseline_probs, min=1e-10, max=1.0)
            baseline_probs = (
                baseline_probs / torch.sum(baseline_probs, axis=1, keepdim=True) * mass
            )

        drn_logits = torch.log(baseline_probs) + self.log_adjustments(x)
        drn_pmf = torch.softmax(drn_logits, dim=1)

        if self.debug:
            assert drn_pmf.shape == (x.shape[0], num_regions)

            # Sometimes we get nan value in here. Otherwise, it should sum to 1.
            assert torch.isnan(drn_pmf).any() or torch.allclose(
                torch.sum(drn_pmf, axis=1),
                torch.ones(x.shape[0], device=x.device),
            )

        return baseline_dists, baseline_probs, drn_pmf

    def regularization(self, dists, baseline_dists, baseline_probs):
        reg_loss = 0.0
        epsilon = 1e-30
        a_i = dists.real_adjustments()
        b_i = baseline_probs

        if self.kl_alpha > 0:
            if self.kl_direction == "forwards":
                kl = -(torch.log(a_i + epsilon) * b_i)
            else:
                kl = torch.log(a_i + epsilon) * a_i * b_i
            reg_loss += torch.mean(torch.sum(kl, axis=1)) * self.kl_alpha

        if self.mean_alpha > 0:
            mean_penalty = torch.mean((baseline_dists.mean - dists.mean) ** 2)
            reg_loss += self.mean_alpha * mean_penalty

        if self.tv_alpha > 0 or self.dv_alpha > 0:
            drn_density = a_i * b_i / torch.diff(self.cutpoints)
            first_diffs = torch.diff(drn_density, dim=1)

            if self.tv_alpha > 0:
                tv_penalty = torch.mean(torch.sum(torch.abs(first_diffs), dim=1))
                reg_loss += self.tv_alpha * tv_penalty

            if self.dv_alpha > 0:
                second_diffs = torch.diff(first_diffs, dim=1)
                dv_penalty = torch.mean(torch.sum(second_diffs ** 2, dim=1))
                reg_loss += self.dv_alpha * dv_penalty

        return reg_loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        baseline_dists, baseline_probs, drn_pmf = self.forward(x)
        dists = ExtendedHistogram(baseline_dists, self.cutpoints, drn_pmf, baseline_probs)

        if self.loss_fn == "jbce":
            loss = jbce_loss(dists, y)
        else:
            loss = nll_loss(dists, y)

        if self.kl_alpha or self.mean_alpha or self.tv_alpha or self.dv_alpha:
            loss += self.regularization(dists, baseline_dists, baseline_probs)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        baseline_dists, baseline_probs, drn_pmf = self.forward(x)
        dists = ExtendedHistogram(baseline_dists, self.cutpoints, drn_pmf, baseline_probs)

        if self.loss_fn == "jbce":
            val_loss = jbce_loss(dists, y)
        else:
            val_loss = nll_loss(dists, y)

        # Regularization penalties intentionally omitted from validation

        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


    def distributions(self, x):
        baseline_dists, baseline_probs, drn_pmf = self.forward(x)
        return ExtendedHistogram(baseline_dists, self.cutpoints, drn_pmf, baseline_probs)


def merge_cutpoints(cutpoints: list[float], y: np.ndarray, min_obs: int) -> list[float]:
    # Ensure cutpoints are sorted and unique to start with
    cutpoints = sorted(list(np.unique(cutpoints)))
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


def drn_cutpoints(c_0, c_K, y, proportion=None, num_cutpoints=None, min_obs=1):
    if proportion is None and num_cutpoints is None:
        raise ValueError(
            "Either a proportion p or a specific num_cutpoints must be provided."
        )

    if proportion is not None:
        num_cutpoints = int(np.ceil(proportion * len(y)))

    uniform_cutpoints = list(np.linspace(c_0, c_K, num_cutpoints))

    return merge_cutpoints(uniform_cutpoints, y, min_obs)
