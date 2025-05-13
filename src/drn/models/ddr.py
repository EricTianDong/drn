import numpy as np
import torch
import torch.nn as nn
import lightning as L

from ..distributions.histogram import Histogram


class DDR(L.LightningModule):
    def __init__(
        self, p: int, cutpoints, num_hidden_layers=2, hidden_size=100, dropout_rate=0.2, learning_rate=1e-3, loss_fn="jbce",
    ):
        """
        Args:
            x_train_shape: The shape of the training data, used to define the input size of the first layer.
            cutpoints: The cutpoints for the DDR model.
            num_hidden_layers: The number of hidden layers in the network.
            hidden_size: The number of neurons in each hidden layer.
        """
        super(DDR, self).__init__()
        self.cutpoints = nn.Parameter(torch.Tensor(cutpoints), requires_grad=False)
        self.p = p
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

        layers = [
            nn.Linear(self.p, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Use nn.Sequential to chain the layers together
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer for the pi values
        self.pi = nn.Linear(hidden_size, len(self.cutpoints) - 1)

    def forward(self, x):
        """
        Forward pass of the DDR model.
        Args:
            x: Input tensor.
        Returns:
            The cutpoints and probabilities for the DDR model.
        """
        # Pass input through the dynamically created hidden layers
        h = self.hidden_layers(x)

        # Calculate probabilities using the final layer
        probs = torch.softmax(self.pi(h), dim=1)

        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self.forward(x)
        dists = Histogram(self.cutpoints, probs)

        if self.loss_fn == "jbce":
            loss = jbce_loss(dists, y)
        else:
            loss = nll_loss(dists, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self.forward(x)
        dists = Histogram(self.cutpoints, probs)

        if self.loss_fn == "jbce":
            val_loss = jbce_loss(dists, y)
        else:
            val_loss = nll_loss(dists, y)

        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def distributions(self, x):
        prob_masses = self.forward(x)
        dists = Histogram(self.cutpoints, prob_masses)
        assert dists.batch_shape == torch.Size([x.shape[0]])
        return dists


def jbce_loss(dists, y):
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

def nll_loss(dists, y):
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

def distribution_prediction_kl_dists(ddr_model, glm_model, X, kl_direction="forwards"):
    """
    Compare DDR and GLM models by producing distributional predictions and calculating CDFs.
    
    Args:
        ddr_model: The DDR model.
        glm_model: The GLM model.
        X: Batch of covariates.
    
    Returns:
        A tuple of CDFs from DDR and GLM models.
    """
    # Get DDR model predictions
    ddr_dists = ddr_model.distributions(X)
    ddr_cdfs = ddr_dists.cdf_at_cutpoints()
    ddr_probs = torch.diff(ddr_cdfs, dim=1)

    # Get GLM model predictions
    glm_dists = glm_model.distributions(X)
    glm_cdfs = glm_dists.cdf_at_cutpoints()
    glm_probs = torch.diff(glm_cdfs, dim=1)

    # Rename the variables for clarity
    a_i = ddr_probs / glm_probs
    b_i = glm_probs

    if kl_direction == "forwards":
        kl = -(torch.log(a_i) * b_i)
    else:
        kl = torch.log(a_i) * a_i * b_i
    
    return torch.mean(torch.sum(kl, axis=1)) 