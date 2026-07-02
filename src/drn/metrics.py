import numpy as np
import pandas as pd
import torch


def _to_tensor(arr: pd.DataFrame | pd.Series | np.ndarray) -> torch.Tensor:
    """Convert pandas or numpy array to float32 torch.Tensor."""
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        vals = arr.values
    else:
        vals = arr
    return torch.as_tensor(vals, dtype=torch.float32)


def crps(
    obs: np.ndarray | pd.Series | torch.Tensor,
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


# ---------------------------------------------------------------------------
# Semi-closed-form CRPS for the DRN ExtendedHistogram distribution.
#
# See docs/crps_extended_histogram.tex for the full derivation. The histogram
# body is integrated exactly (piecewise-linear CDF -> squared-affine integrals),
# and the two tails are reduced to one-dimensional integrals of the baseline CDF
# evaluated with Gauss-Legendre quadrature.
# ---------------------------------------------------------------------------

_GL_NODES = 100


def _gauss_legendre(n: int, device, dtype):
    """Gauss-Legendre nodes/weights on [0, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)  # on [-1, 1]
    nodes = torch.as_tensor(0.5 * (nodes + 1.0), device=device, dtype=dtype)
    weights = torch.as_tensor(0.5 * weights, device=device, dtype=dtype)
    return nodes, weights


def _integrate_finite(f, a, b, nodes, weights):
    """Integrate f over [a, b] (a, b broadcastable tensors) via Gauss-Legendre.

    Nodes have shape (Q,); a, b have shape (n,). Returns shape (n,).
    """
    a = a.unsqueeze(0)  # (1, n)
    b = b.unsqueeze(0)  # (1, n)
    u = nodes.unsqueeze(1)  # (Q, 1)
    x = a + (b - a) * u  # (Q, n)
    vals = f(x)  # (Q, n)
    return ((b - a) * torch.einsum("q,qn->qn", weights, vals).sum(dim=0)).squeeze(0)


def _integrate_semi_infinite(f, t, nodes, weights, direction):
    """Integrate f over [t, +inf) (direction=+1) or (-inf, t] (direction=-1).

    Uses x = t + direction * u / (1 - u), dx = du / (1 - u)^2.
    t has shape (n,); returns shape (n,).
    """
    t = t.unsqueeze(0)  # (1, n)
    u = nodes.unsqueeze(1)  # (Q, 1)
    one_minus = 1.0 - u
    x = t + direction * u / one_minus  # (Q, n)
    jac = 1.0 / (one_minus ** 2)  # (Q, 1)
    vals = f(x) * jac  # (Q, n)
    return torch.einsum("q,qn->qn", weights, vals).sum(dim=0)


def crps_extended_histogram(dist, obs) -> torch.Tensor:
    """CRPS for an ExtendedHistogram prediction.

    The histogram body is always integrated exactly (piecewise-linear CDF). For
    Gamma and Normal baselines the tails are handled by the baseline's
    closed-form CRPS, leaving only a single bounded body integral; for other
    baselines the tails fall back to Gauss-Legendre quadrature. See
    ``docs/crps_extended_histogram.tex``.

    :param dist: an ``ExtendedHistogram`` (batched over the observations)
    :param obs: observed value(s); shape must match the batch of ``dist``
    :return: per-observation CRPS as a 1D tensor
    """
    baseline = dist.baseline
    if isinstance(baseline, (torch.distributions.Normal, torch.distributions.Gamma)):
        return _crps_eh_closed(dist, obs)
    return _crps_eh_quadrature(dist, obs)


def _crps_eh_quadrature(dist, obs) -> torch.Tensor:
    """Baseline-agnostic CRPS: exact body, tails via Gauss-Legendre quadrature."""
    cutpoints = dist.cutpoints  # (K+1,)
    device, dtype = cutpoints.device, cutpoints.dtype

    y = _to_tensor(obs).to(device=device, dtype=dtype).reshape(-1)  # (n,)

    c0 = cutpoints[0]
    cK = cutpoints[-1]
    widths = cutpoints[1:] - cutpoints[:-1]  # (K,)

    # CDF at cutpoints: shape (K+1, n) -> (n, K+1)
    G = dist.cdf_at_cutpoints().T
    G_lo = G[:, :-1]  # (n, K) value of F at each bin's left edge
    G_hi = G[:, 1:]  # (n, K)
    slopes = (G_hi - G_lo) / widths.unsqueeze(0)  # (n, K)

    # --- Body term M (exact) ---
    a_row = cutpoints[:-1].unsqueeze(0)  # (1, K)
    b_row = cutpoints[1:].unsqueeze(0)  # (1, K)
    y_col = y.unsqueeze(1)  # (n, 1)

    u_star = torch.clamp(y_col - a_row, min=torch.zeros_like(a_row), max=widths.unsqueeze(0))
    # ^ length of the indicator=0 stretch inside each bin

    def Q(P, s, L):
        return P * P * L + P * s * L * L + (s * s * L * L * L) / 3.0

    L0 = u_star
    L1 = widths.unsqueeze(0) - u_star
    P_switch = G_lo + slopes * u_star  # value of F at the switch point
    body = Q(G_lo, slopes, L0) + Q(P_switch - 1.0, slopes, L1)
    M = body.sum(dim=1)  # (n,)

    # --- Tail terms (semi-closed) ---
    nodes, weights = _gauss_legendre(_GL_NODES, device, dtype)

    # Quadrature nodes in the tail transforms can fall outside the baseline's
    # support (e.g. negative x for a Gamma). There the CDF is 0 (left) or 1
    # (right), so clamp into support and mask, bypassing validate_args.
    support = getattr(dist.baseline, "support", None)
    lower = getattr(support, "lower_bound", None)
    if lower is not None:
        lower = torch.as_tensor(float(lower), device=device, dtype=dtype)

    def _safe_cdf(x):
        if lower is None:
            return dist.baseline.cdf(x)
        below = x < lower
        xx = torch.where(below, lower, x)
        c = dist.baseline.cdf(xx)
        return torch.where(below, torch.zeros_like(c), c)

    def Fb_sq(x):
        return _safe_cdf(x) ** 2

    def one_minus_Fb_sq(x):
        return (1.0 - _safe_cdf(x)) ** 2

    t_L = torch.clamp(y, max=c0)  # min(y, c0)
    t_R = torch.clamp(y, min=cK)  # max(y, cK)

    c0_vec = c0.expand_as(y)
    cK_vec = cK.expand_as(y)

    # Left tail: A(t_L) + int_{t_L}^{c0} (1 - Fb)^2
    A_tL = _integrate_semi_infinite(Fb_sq, t_L, nodes, weights, direction=-1.0)
    left_finite = _integrate_finite(one_minus_Fb_sq, t_L, c0_vec, nodes, weights)
    T_L = A_tL + left_finite

    # Right tail: int_{cK}^{t_R} Fb^2 + B(t_R)
    right_finite = _integrate_finite(Fb_sq, cK_vec, t_R, nodes, weights)
    B_tR = _integrate_semi_infinite(one_minus_Fb_sq, t_R, nodes, weights, direction=1.0)
    T_R = right_finite + B_tR

    return T_L + M + T_R


# ---- Closed-form baseline pieces (Normal, Gamma) --------------------------

_SQRT_PI = float(np.sqrt(np.pi))


def _normal_std(x, mu, sigma):
    return (x - mu) / sigma


def _std_normal_pdf(z):
    return torch.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _std_normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))


def _baseline_crps(baseline, y):
    """Closed-form CRPS of the baseline at y (shape (n,))."""
    if isinstance(baseline, torch.distributions.Normal):
        mu, sigma = baseline.loc, baseline.scale
        omega = (y - mu) / sigma
        Phi = _std_normal_cdf(omega)
        phi = _std_normal_pdf(omega)
        return sigma * (omega * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / _SQRT_PI)
    if isinstance(baseline, torch.distributions.Gamma):
        alpha, beta = baseline.concentration, baseline.rate
        F1 = baseline.cdf(torch.clamp(y, min=0.0))
        F1 = torch.where(y > 0, F1, torch.zeros_like(F1))
        F2 = torch.distributions.Gamma(alpha + 1.0, beta).cdf(torch.clamp(y, min=0.0))
        F2 = torch.where(y > 0, F2, torch.zeros_like(F2))
        # Constant term 1 / (beta * B(1/2, alpha)) = Gamma(alpha+1/2)/(beta*sqrt(pi)*Gamma(alpha))
        const = torch.exp(torch.lgamma(alpha + 0.5) - torch.lgamma(alpha)) / (beta * _SQRT_PI)
        return y * (2.0 * F1 - 1.0) - (alpha / beta) * (2.0 * F2 - 1.0) - const
    raise TypeError(f"No closed-form CRPS for baseline {type(baseline).__name__}")


def _baseline_partial_first_moment(baseline, a, b):
    """int_a^b x f_b(x) dx, closed form (a, b broadcastable tensors)."""
    if isinstance(baseline, torch.distributions.Normal):
        mu, sigma = baseline.loc, baseline.scale
        za, zb = (a - mu) / sigma, (b - mu) / sigma
        return mu * (_std_normal_cdf(zb) - _std_normal_cdf(za)) - sigma * (
            _std_normal_pdf(zb) - _std_normal_pdf(za)
        )
    if isinstance(baseline, torch.distributions.Gamma):
        alpha, beta = baseline.concentration, baseline.rate
        g1 = torch.distributions.Gamma(alpha + 1.0, beta)
        return (alpha / beta) * (g1.cdf(b) - g1.cdf(a))
    raise TypeError(f"No closed-form partial moment for {type(baseline).__name__}")


def _baseline_cdf_integral(baseline, a, b):
    """int_a^b F_b(x) dx = [x F_b]_a^b - int_a^b x f_b dx (closed form)."""
    Fa, Fb = baseline.cdf(a), baseline.cdf(b)
    return b * Fb - a * Fa - _baseline_partial_first_moment(baseline, a, b)


def _crps_eh_closed(dist, obs) -> torch.Tensor:
    """Fully-analytic-tail CRPS for Gamma/Normal ExtendedHistogram predictions.

    Uses  CRPS(F,y) = CRPS(F_b,y) + int_{c0}^{cK}(F^2 - F_b^2)
                       - 2 int_{c0}^{cK} 1{x>=y}(F - F_b),
    with the baseline CRPS in closed form. The only non-elementary piece is the
    bounded body integral S = int_{c0}^{cK} F_b^2, done by Gauss-Legendre
    quadrature over [c0, cK] (well-conditioned; no infinite tails).
    """
    baseline = dist.baseline
    cutpoints = dist.cutpoints
    device, dtype = cutpoints.device, cutpoints.dtype
    y = _to_tensor(obs).to(device=device, dtype=dtype).reshape(-1)  # (n,)

    c0, cK = cutpoints[0], cutpoints[-1]
    widths = cutpoints[1:] - cutpoints[:-1]  # (K,)

    G = dist.cdf_at_cutpoints().T  # (n, K+1)
    G_lo = G[:, :-1]
    slopes = (G[:, 1:] - G_lo) / widths.unsqueeze(0)  # (n, K)

    a_row = cutpoints[:-1].unsqueeze(0)  # (1, K)
    w_row = widths.unsqueeze(0)
    u_star = torch.clamp(y.unsqueeze(1) - a_row, min=torch.zeros_like(a_row), max=w_row)

    # int_{c0}^{cK} F^2 dx  (F piecewise linear; no indicator)
    def Q(P, s, L):
        return P * P * L + P * s * L * L + (s * s * L * L * L) / 3.0

    MF2 = Q(G_lo, slopes, w_row).sum(dim=1)  # (n,)

    # int_{c0}^{cK} 1{x>=y} F dx = sum_k int_{u*}^{w}(G_k + s_k u) du
    MFI = (
        G_lo * (w_row - u_star) + slopes * (w_row * w_row - u_star * u_star) / 2.0
    ).sum(dim=1)

    # int_{c0}^{cK} 1{x>=y} F_b dx = int_{y'}^{cK} F_b dx,  y' = clip(y, c0, cK)
    y_clip = torch.clamp(y, min=c0, max=cK)
    MFbI = _baseline_cdf_integral(baseline, y_clip, cK.expand_as(y))

    # S = int_{c0}^{cK} F_b^2 dx  (bounded quadrature)
    nodes, weights = _gauss_legendre(_GL_NODES, device, dtype)
    S = _integrate_finite(
        lambda x: baseline.cdf(x) ** 2, c0.expand_as(y), cK.expand_as(y), nodes, weights
    )

    crps_b = _baseline_crps(baseline, y)
    return crps_b + (MF2 - S) - 2.0 * (MFI - MFbI)


def quantile_score(y_true, y_pred, p, mean_tensor=True):
    """
    Compute the quantile score for predictions at a specific quantile.

    :param y_true: Actual target values as a Pandas Series or PyTorch tensor.
    :param y_pred: Predicted target values as a numpy array or PyTorch tensor.
    :param p: The cumulative probability as a float
    :return: The quantile score as a PyTorch tensor.
    """
    # Ensure that y_true and y_pred are PyTorch tensors
    y_true = _to_tensor(y_true)
    y_pred = _to_tensor(y_pred)
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
    y = _to_tensor(y)
    # Calculate the RMSE
    return torch.sqrt(torch.mean((y.squeeze() - y_hat.squeeze()) ** 2))


def nll(dists, y, alpha=0.0):
    """
    Compute the mean negative log-likelihood of observations under a distribution.

    :param dists: A (batched) torch distribution exposing ``log_prob``.
    :param y: Observed target values. Can be a Pandas Series/DataFrame, a NumPy
        array, or a PyTorch tensor.
    :return: The mean negative log-likelihood as a PyTorch tensor.
    """
    y = _to_tensor(y)
    losses = -(dists.log_prob(y))
    return torch.mean(losses)
