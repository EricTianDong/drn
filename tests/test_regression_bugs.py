"""Regression tests for previously-discovered bugs."""

import numpy as np
import pandas as pd
import torch
from synthetic_dataset import generate_synthetic_tensordataset

from drn import GLM, train
from drn.metrics import quantile_score, rmse
from drn.models.glm import gaussian_deviance_loss


# ── GLM lognormal must use gaussian (not gamma) deviance loss ────────────

def test_glm_lognormal_uses_gaussian_loss():
    """GLM('lognormal') should use gaussian_deviance_loss, not gamma_deviance_loss."""
    glm = GLM("lognormal")
    assert glm.loss_fn is gaussian_deviance_loss


def test_glm_lognormal_loss_log_transforms_targets():
    """GLM('lognormal').loss() should compare predictions against log(y)."""
    glm = GLM("lognormal")
    # Initialise the lazy linear layer
    x = torch.randn(5, 2)
    _ = glm(x)

    y = torch.exp(torch.randn(5).abs())  # positive targets

    # Manually compute expected: gaussian deviance on log-scale
    with torch.no_grad():
        preds = glm(x)
        expected = torch.mean((torch.log(y) - preds) ** 2)
        actual = glm.loss(x, y)

    assert torch.allclose(actual, expected), (
        f"lognormal loss should operate on log(y); got {actual} vs expected {expected}"
    )


# ── train() epochs_trained must equal actual epochs run ──────────────────

def test_train_epochs_trained_no_early_stop():
    """After training for N epochs without early stopping, epochs_trained == N."""
    X_train, Y_train, train_ds, val_ds = generate_synthetic_tensordataset()

    torch.manual_seed(1)
    glm = GLM("gamma")
    epochs = 3
    train(glm, train_ds, val_ds, epochs=epochs, patience=epochs + 100,
          print_details=False)

    assert glm.epochs_trained == epochs, (
        f"Expected epochs_trained={epochs}, got {glm.epochs_trained}"
    )


def test_train_epochs_trained_with_early_stop():
    """When early stopping fires, epochs_trained must be <= requested epochs."""
    X_train, Y_train, train_ds, val_ds = generate_synthetic_tensordataset()

    torch.manual_seed(1)
    glm = GLM("gamma")
    epochs = 500
    train(glm, train_ds, val_ds, epochs=epochs, patience=2, print_details=False)

    assert 1 <= glm.epochs_trained <= epochs, (
        f"epochs_trained={glm.epochs_trained} out of range [1, {epochs}]"
    )


# ── metrics must accept numpy arrays, pandas, and tensors ────────────────

def test_quantile_score_accepts_numpy():
    """quantile_score should work with numpy array inputs."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    score = quantile_score(y_true, y_pred, p=0.5)
    assert isinstance(score, torch.Tensor)
    assert score.isfinite()


def test_quantile_score_accepts_pandas():
    """quantile_score should work with pandas Series inputs."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.1, 1.9, 3.2])
    score = quantile_score(y_true, y_pred, p=0.5)
    assert isinstance(score, torch.Tensor)
    assert score.isfinite()


def test_quantile_score_consistent_across_types():
    """quantile_score should return the same value for numpy, pandas, and tensor."""
    vals_true = [1.0, 2.0, 3.0]
    vals_pred = [1.1, 1.9, 3.2]
    p = 0.5

    score_np = quantile_score(np.array(vals_true), np.array(vals_pred), p)
    score_pd = quantile_score(pd.Series(vals_true), pd.Series(vals_pred), p)
    score_pt = quantile_score(torch.tensor(vals_true), torch.tensor(vals_pred), p)

    assert torch.allclose(score_np, score_pd)
    assert torch.allclose(score_np, score_pt)


def test_rmse_accepts_numpy():
    """rmse should work with numpy array inputs."""
    y = np.array([1.0, 2.0, 3.0])
    y_hat = torch.tensor([1.1, 1.9, 3.2])
    score = rmse(y, y_hat)
    assert isinstance(score, torch.Tensor)
    assert score.isfinite()
