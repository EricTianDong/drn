from typing import Union
import numpy as np
import pandas as pd
import torch


def _to_tensor(arr: Union[pd.DataFrame, pd.Series, np.ndarray]) -> torch.Tensor:
    """Convert pandas or numpy array to float32 torch.Tensor."""
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        vals = arr.values
    else:
        vals = arr
    return torch.as_tensor(vals, dtype=torch.float32)


def crps(
    obs: Union[np.ndarray, pd.Series, torch.Tensor],
    grid: torch.Tensor,
    cdf_on_grid: torch.Tensor,
):
    """
    Compute CRPS using the provided grid and CDF values with PyTorch tensors.

    :param obs: observed value(s)
    :param grid: a grid over y values
    :param cdf_on_grid: tensor of corresponding CDF values or a 2D tensor where each column is a CDF
    :return: CRPS value(s) as a PyTorch tensor
    """
    # Ensure obs and cdf_on_grid are at least 1D and 2D tensors, respectively
    obs = _to_tensor(obs)
    obs = obs.unsqueeze(0) if obs.ndim == 0 else obs
    cdf_on_grid = cdf_on_grid.unsqueeze(1) if cdf_on_grid.ndim == 1 else cdf_on_grid
    cdf_on_grid = cdf_on_grid.T

    # Compute the difference between grid points (assuming uniform spacing)
    dy = grid[1] - grid[0]

    # Calculate the Heaviside step function values for each x and y_grid value
    heaviside_matrix = (grid >= obs.unsqueeze(1)).type(torch.float32).squeeze()

    # Compute the CRPS values for each x and CDF_grid pair
    crps_values = torch.sum((cdf_on_grid - heaviside_matrix) ** 2, dim=1) * dy

    # If x was a scalar, return a scalar. Otherwise, return a tensor.
    return crps_values if crps_values.numel() > 1 else crps_values.item()


def quantile_score(y_true, y_pred, p, mean_tensor=True):
    """
    Compute the quantile score for predictions at a specific quantile.

    :param y_true: Actual target values as a Pandas Series or PyTorch tensor.
    :param y_pred: Predicted target values as a numpy array or PyTorch tensor.
    :param p: The cumulative probability as a float
    :return: The quantile score as a PyTorch tensor.
    """
    # Ensure that y_true and y_pred are PyTorch tensors
    y_true = (
        torch.Tensor(y_true.values) if not isinstance(y_true, torch.Tensor) else y_true
    )
    y_pred = torch.Tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    # Reshape y_pred to match y_true if necessary and compute the error
    e = y_true - y_pred.reshape(y_true.shape)
    # Compute the quantile score
    if mean_tensor:
        return torch.where(y_true >= y_pred, p * e, (1 - p) * -e).mean()
    else:
        return torch.where(y_true >= y_pred, p * e, (1 - p) * -e)


def quantile_losses(
    p,
    model,
    model_name,
    X,
    y,
    max_iter=1000,
    tolerance=5e-5,
    l=None,
    u=None,
    print_score=True,
):
    """
    Calculate and optionally print the quantile loss for the given data and model.

    :param p: The cumulative probability ntile as a float
    :param model: The trained model.
    :param model_name: The name of the trained model.
    :param X: Input features as a Pandas DataFrame or numpy array.
    :param y: True target values as a Pandas Series or numpy array.
    :param max_iter: The maximum number of iterations for the quantile search algorithm.
    :param tolerance: The tolerance for convergence of the the quantile search algorithm.
    :param l: The lower bound for the quantile search
    :param u: The upper bound for the quantile search
    :param print_score: A boolean indicating whether to print the score.
    :return: The quantile loss as a PyTorch tensor.
    """
    # Predict quantiles based on the model name
    if model_name in ["GLM", "MDN", "CANN"]:
        predicted_quantiles = model.quantiles(
            X, [p * 100], max_iter=max_iter, tolerance=tolerance, l=l, u=u
        )
    elif model_name in ["DDR", "DRN"]:
        predicted_quantiles = model.predict(X).quantiles(
            [p * 100], max_iter=max_iter, tolerance=tolerance, l=l, u=u
        )

    # Compute the quantile score
    score = quantile_score(y, predicted_quantiles, p)

    # Print the score if requested
    if print_score:
        print(f"{model_name}: {score:.5f}")

    return score


def rmse(y, y_hat):
    """
    Compute the Root Mean Square Error (RMSE) between the true values and predictions.

    :param y: True target values. Can be a Pandas Series or a PyTorch tensor.
    :param y_hat: Predicted target values. Should be a PyTorch tensor.
    :return: The RMSE as a PyTorch tensor.
    """
    # Convert y to a PyTorch tensor if it is not already one
    if not isinstance(y, torch.Tensor):
        if hasattr(y, "values"):  # pandas Series/DataFrame
            y = torch.Tensor(y.values)
        else:  # numpy array or other array-like
            y = torch.Tensor(y)
    # Calculate the RMSE
    return torch.sqrt(torch.mean((y.squeeze() - y_hat.squeeze()) ** 2))
