import pytest
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from drn import Constant, InverseGaussian, GLM
from torch.distributions import Gamma, Normal


def test_initial_state_values():
    # Gamma and IG should start with mean=1, dispersion=1; Gaussian with mean=0
    c_gamma = Constant("gamma")
    assert c_gamma.mean_value.item() == pytest.approx(1.0)
    assert c_gamma.dispersion.item() == pytest.approx(1.0)

    c_ig = Constant("inversegaussian")
    assert c_ig.mean_value.item() == pytest.approx(1.0)
    assert c_ig.dispersion.item() == pytest.approx(1.0)

    c_gauss = Constant("gaussian")
    assert c_gauss.mean_value.item() == pytest.approx(0.0)
    assert c_gauss.dispersion.item() == pytest.approx(1.0)


def test_distributions_initial():
    # Use a dummy input of batch size 4
    x = torch.zeros((4, 2))
    # Gamma
    c = Constant("gamma")
    dist = c.predict(x)
    assert isinstance(dist, Gamma)
    # mean = alpha/beta = (1/phi)/(mean_value) = 1/1 = 1
    assert torch.allclose(dist.mean, torch.tensor([1.0] * 4))

    # IG
    c = Constant("inversegaussian")
    dist = c.predict(x)
    assert isinstance(dist, InverseGaussian)
    # mean property for custom IG
    assert torch.allclose(dist.mean, torch.tensor([1.0] * 4))

    # Gaussian
    c = Constant("gaussian")
    dist = c.predict(x)
    assert isinstance(dist, Normal)
    assert torch.allclose(dist.mean, torch.tensor([0.0] * 4))


def test_fit_updates_mean_and_dispersion_gamma():
    # Create sample y with non-equal values for Gamma
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
    c = Constant("gamma")
    # fit ignores X
    c.fit(X_train=None, y_train=y)
    # mean should update
    assert c.mean_value.item() == pytest.approx(5.0)
    # dispersion = sum((y-5)^2/5^2)/(n-1)
    expected_phi = np.sum((y - 5.0) ** 2 / 5.0**2) / (len(y) - 1)
    assert c.dispersion.item() == pytest.approx(expected_phi)


def test_fit_updates_mean_and_dispersion_gaussian():
    # Gaussian dispersion is sigma (std dev)
    y = np.array([1.0, 3.0, 5.0], dtype=float)
    c = Constant("gaussian")
    c.fit(X_train=None, y_train=y)
    # mean = 3.0
    assert c.mean_value.item() == pytest.approx(3.0)
    # sigma = sqrt(sum((y-3)^2)/(n-1)) = sqrt((4+0+4)/2) = sqrt(4) = 2
    assert c.dispersion.item() == pytest.approx(2.0)


def test_loss_decreases_after_fit():
    # For a simple dataset, loss before fit is higher than after fit
    # Note: we need y to have some variance for dispersion estimation to be > 0
    y = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
    x = torch.zeros((4, 1))
    c = Constant("gamma")
    # initial loss with mean=1
    loss_before = c.loss(x, torch.tensor(y))
    c.fit(X_train=None, y_train=y)
    loss_after = c.loss(x, torch.tensor(y))
    assert loss_after < loss_before


def test_invalid_distribution_raises():
    with pytest.raises(ValueError):
        Constant("unsupported")


def test_equivalence_with_zeroed_glm():
    # Dummy input batch of size 3
    p = 2
    x = torch.randn((3, p))

    # Constant without fit should match GLM without fit (if we zero-out the parameters)
    for dist in ("gamma", "gaussian", "inversegaussian"):
        const = Constant(dist)
        glm = GLM(dist)

        glm.linear = nn.Linear(p, 1, bias=True)
        glm.linear.weight.data.fill_(0.0)
        glm.linear.bias.data.fill_(0.0)

        # forward matches
        out_c = const(x)
        out_g = glm(x)
        assert torch.allclose(out_c, out_g)

        # dispersion matches
        assert const.dispersion.item() == pytest.approx(glm.dispersion.item())
        # distribution parameters match
        d1 = const.predict(x)
        d2 = glm.predict(x)
        if dist == "gamma":
            assert torch.allclose(d1.concentration, d2.concentration)
            assert torch.allclose(d1.rate, d2.rate)
        elif dist == "gaussian":
            assert torch.allclose(d1.loc, d2.loc)
            assert torch.allclose(d1.scale, d2.scale)
        else:
            assert torch.allclose(d1.mean, d2.mean)
            assert const.dispersion.item() == pytest.approx(glm.dispersion.item())


def test_quantiles_method_exists():
    """Test that the quantiles method exists and is callable."""
    for dist in ("gamma", "gaussian", "inversegaussian"):
        const = Constant(dist)
        assert hasattr(const, "quantiles")
        assert callable(getattr(const, "quantiles"))


def test_quantiles_shape_and_monotonicity():
    """Test that quantiles return correct shape and are monotonically increasing."""
    for dist in ("gamma", "gaussian", "inversegaussian"):
        # Create and fit model
        const = Constant(dist)

        # Create training data
        np.random.seed(42)
        if dist == "gamma":
            y_train = np.random.gamma(2, 2, 50)
        elif dist == "gaussian":
            y_train = np.random.normal(5, 2, 50)
        else:  # inversegaussian
            y_train = np.random.gamma(2, 2, 50)  # Use gamma as proxy

        X_train = np.random.randn(50, 3)
        const.fit(X_train, y_train)

        # Test data
        X_test = np.random.randn(5, 3)
        percentiles = [10, 25, 50, 75, 90]

        # Compute quantiles
        quantiles = const.quantiles(X_test, percentiles)

        # Check shape: should be (n_percentiles, n_samples)
        assert quantiles.shape == (
            5,
            5,
        ), f"Expected shape (5, 5), got {quantiles.shape}"

        # Check that quantiles are monotonically increasing for each sample
        for j in range(quantiles.shape[1]):  # for each sample
            col = quantiles[:, j]  # get quantiles for this sample
            for i in range(len(col) - 1):
                assert (
                    col[i] <= col[i + 1]
                ), f"Quantiles not monotonic for {dist} at sample {j}"


def test_quantiles_with_different_inputs():
    """Test quantiles with different input types (numpy, pandas, torch)."""
    const = Constant("gamma")
    y_train = np.random.gamma(2, 2, 30)
    X_train = np.random.randn(30, 2)
    const.fit(X_train, y_train)

    percentiles = [25, 50, 75]

    # Test with numpy array
    X_np = np.random.randn(3, 2)
    q_np = const.quantiles(X_np, percentiles)

    # Test with torch tensor
    X_torch = torch.from_numpy(X_np).float()
    q_torch = const.quantiles(X_torch, percentiles)

    # Test with pandas DataFrame
    X_pd = pd.DataFrame(X_np, columns=["feat1", "feat2"])
    q_pd = const.quantiles(X_pd, percentiles)

    # Results should be very similar (allowing for small numerical differences)
    assert torch.allclose(q_np, q_torch, atol=1e-6)
    assert torch.allclose(q_np, q_pd, atol=1e-6)


def test_quantiles_values_reasonable():
    """Test that quantile values are reasonable for known distributions."""
    # Test Gaussian case where we can verify results analytically
    const = Constant("gaussian")

    # Fit with known data
    y_train = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # mean = 4.0
    X_train = np.zeros((5, 1))
    const.fit(X_train, y_train)

    X_test = np.zeros((1, 1))

    # Test 50th percentile (median) should be close to mean for normal distribution
    median = const.quantiles(X_test, [50])
    assert (
        torch.abs(median[0, 0] - 4.0) < 0.1
    ), f"Median {median[0, 0]} not close to mean 4.0"

    # Test that lower percentiles < higher percentiles
    q_vals = const.quantiles(X_test, [10, 25, 50, 75, 90])
    for i in range(4):
        assert q_vals[i, 0] < q_vals[i + 1, 0], "Quantiles not properly ordered"


def test_icdf_method_exists():
    """Test that the icdf method exists and works."""
    for dist in ("gamma", "gaussian", "inversegaussian"):
        const = Constant(dist)
        assert hasattr(const, "icdf")
        assert callable(getattr(const, "icdf"))

        # Fit model
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_train = np.zeros((5, 2))
        const.fit(X_train, y_train)

        # Test icdf
        X_test = np.zeros((3, 2))
        result = const.icdf(X_test, 0.5)  # median

        # Should return tensor of shape (1, n_samples)
        assert result.shape == (1, 3), f"Expected shape (1, 3), got {result.shape}"

        # Values should be positive for gamma/inversegaussian
        if dist != "gaussian":
            assert torch.all(result > 0), f"Expected positive values for {dist}"


def test_quantiles_edge_cases():
    """Test quantiles with edge cases."""
    const = Constant("gamma")
    y_train = np.array([1.5, 2.0, 2.5])  # small variation to avoid numerical issues
    X_train = np.zeros((3, 1))
    const.fit(X_train, y_train)

    X_test = np.zeros((2, 1))

    # Test with single percentile
    q_single = const.quantiles(X_test, [50])
    assert q_single.shape == (1, 2)  # (n_percentiles, n_samples)

    # Test with extreme percentiles
    q_extreme = const.quantiles(X_test, [5, 95])  # Use less extreme percentiles
    assert q_extreme.shape == (2, 2)  # (n_percentiles, n_samples)
    # For gamma distribution, 5th percentile should be < 95th percentile
    assert torch.all(q_extreme[0, :] <= q_extreme[1, :])  # 5th <= 95th percentile


def test_quantiles_consistency_with_distributions():
    """Test that quantiles are consistent with the underlying distribution."""
    const = Constant("gaussian")
    y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    X_train = np.zeros((5, 1))
    const.fit(X_train, y_train)

    X_test = np.zeros((1, 1))

    # Get quantile using our method
    q50 = const.quantiles(X_test, [50])[0, 0]  # First percentile, first sample

    # Get quantile directly from distribution
    dist = const.predict(torch.zeros(1, 1))
    q50_direct = dist.icdf(torch.tensor(0.5))[0]

    # Should be very close
    assert torch.abs(q50 - q50_direct) < 1e-6, "Quantile inconsistent with distribution"
