import torch


def gamma_estimate_dispersion(mu: torch.Tensor, y: torch.Tensor, p: int) -> float:
    """
    For a gamma GLM, the dispersion parameter is estimated using the method of moments.
    Args:
        mu: the predicted means for the gamma distributions (shape: (n, 1))
        y: the observed values (shape: (n, 1))
        p: the number of features (not including the intercept)
    """
    n = mu.shape[0]
    dof = n - (p + 1)
    assert dof > 0, "Degrees of freedom must be positive to estimate dispersion (i.e. need n > p)."

    # We want to estimate dispersion = (torch.sum((y - mu) ** 2 / mu**2) / dof).item()
    # however, if y or mu is large then (y - mu) ** 2 can become infinite leading to NaN dispersion.
    r = y / mu - 1.0
    phi = r.square().mean() * (n / dof)
    return phi.item()


def gamma_convert_parameters(
    mu: torch.Tensor, phi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Our models predict the mean of the gamma distribution, but we need the shape and rate parameters.
    This function converts the mean and dispersion parameter into the shape and rate parameters.
    Args:
        mu: the predicted means for the gamma distributions (shape: (n,))
        phi: the dispersion parameter
    Returns:
        alpha: the shape parameter (shape: (n,))
        beta: the rate parameter (shape: (n,))
    """
    beta = 1.0 / (mu * phi)
    alpha = (1.0 / phi) * torch.ones_like(beta)
    return alpha, beta


def gaussian_estimate_sigma(mu: torch.Tensor, y: torch.Tensor) -> float:
    """
    For a Gaussian GLM, the dispersion parameter is estimated using the method of moments.
    Args:
        mu: the predicted means for the Gaussian distributions (shape: (n, 1))
        y: the observed values (shape: (n, 1))
        p: the number of features
    """
    n = mu.shape[0]
    variance_estimate = torch.sum((y - mu) ** 2) / (n - 1)
    return (torch.sqrt(variance_estimate)).item()


def estimate_dispersion(distribution: str, mu: torch.Tensor, y: torch.Tensor, p: int):
    """
    Estimate the dispersion parameter for different distributions.

    Parameters:
    distribution (str): The type of distribution ("gamma", "gaussian", "inversegaussian", "lognormal").
    mu (torch.Tensor): The predicted mean values.
    y (torch.Tensor): The observed target values.
    p (int): The number of model parameters.

    Returns:
    torch.Tensor: The estimated dispersion parameter.
    """
    if distribution == "gamma":
        disp = gamma_estimate_dispersion(mu, y, p)
    elif distribution == "gaussian":
        disp = gaussian_estimate_sigma(mu, y)
    elif distribution == "lognormal":
        disp = gaussian_estimate_sigma(mu, y)  # lognormal uses same estimation as gaussian
    elif distribution == "inversegaussian":
        disp = inversegaussian_estimate_dispersion(mu, y, p)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    assert torch.isnan(torch.tensor(disp)) == False, "Estimated dispersion is NaN." 
    assert disp >= 0.0, "Estimated dispersion must be non-negative."
    return disp


def inversegaussian_estimate_dispersion(
    mu: torch.Tensor, y: torch.Tensor, p: int
) -> float:
    n = mu.shape[0]
    dof = n - (p + 1)
    return (torch.sum(((y - mu) ** 2) / (mu**3)) / dof).item()
