import torch
import pytest
import warnings
from drn.distributions.estimation import (
    gamma_estimate_dispersion,
    gaussian_estimate_sigma,
    inversegaussian_estimate_dispersion,
)


class TestGammaEstimateDispersion:
    """Test that numerically stable gamma dispersion matches simple formula on reasonable inputs."""

    def simple_gamma_dispersion(self, mu: torch.Tensor, y: torch.Tensor, p: int, unbiased: bool = True) -> float:
        """Simple/naive formula for gamma dispersion estimation."""
        n = mu.shape[0]
        dof = n - (p + 1) if (unbiased and n - (p + 1) > 0) else n
        return (torch.sum((y - mu) ** 2 / mu**2) / dof).item()

    def test_small_values(self):
        """Test with small reasonable values."""
        torch.manual_seed(42)
        n = 100
        p = 5
        mu = torch.rand(n, 1) * 10  # values between 0 and 10
        y = torch.rand(n, 1) * 10

        stable_result = gamma_estimate_dispersion(mu, y, p, unbiased=True)
        simple_result = self.simple_gamma_dispersion(mu, y, p, unbiased=True)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_unbiased_false(self):
        """Test with unbiased=False."""
        torch.manual_seed(123)
        n = 50
        p = 3
        mu = torch.rand(n, 1) * 5
        y = torch.rand(n, 1) * 5

        stable_result = gamma_estimate_dispersion(mu, y, p, unbiased=False)
        simple_result = self.simple_gamma_dispersion(mu, y, p, unbiased=False)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_various_sizes(self):
        """Test with different sample sizes."""
        torch.manual_seed(456)
        for n in [20, 50, 100, 200]:
            p = 5
            mu = torch.rand(n, 1) * 20
            y = torch.rand(n, 1) * 20

            stable_result = gamma_estimate_dispersion(mu, y, p, unbiased=True)
            simple_result = self.simple_gamma_dispersion(mu, y, p, unbiased=True)

            assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)


class TestGaussianEstimateSigma:
    """Test that numerically stable gaussian sigma matches simple formula on reasonable inputs."""

    def simple_gaussian_sigma(self, mu: torch.Tensor, y: torch.Tensor, p: int, unbiased: bool = True) -> float:
        """Simple/naive formula for gaussian sigma estimation."""
        n = mu.shape[0]
        dof = n - (p + 1) if (unbiased and n - (p + 1) > 0) else n
        variance_estimate = torch.sum((y - mu) ** 2) / dof
        return torch.sqrt(variance_estimate).item()

    def test_small_values(self):
        """Test with small reasonable values."""
        torch.manual_seed(42)
        n = 100
        p = 5
        mu = torch.randn(n, 1) * 2  # Normal values around 0
        y = torch.randn(n, 1) * 2

        stable_result = gaussian_estimate_sigma(mu, y, p, unbiased=True)
        simple_result = self.simple_gaussian_sigma(mu, y, p, unbiased=True)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_unbiased_false(self):
        """Test with unbiased=False."""
        torch.manual_seed(123)
        n = 50
        p = 3
        mu = torch.randn(n, 1) * 5
        y = torch.randn(n, 1) * 5

        stable_result = gaussian_estimate_sigma(mu, y, p, unbiased=False)
        simple_result = self.simple_gaussian_sigma(mu, y, p, unbiased=False)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_positive_values(self):
        """Test with positive values (for cases where mu should be positive)."""
        torch.manual_seed(789)
        n = 80
        p = 4
        mu = torch.rand(n, 1) * 10 + 1  # values between 1 and 11
        y = torch.rand(n, 1) * 10 + 1

        stable_result = gaussian_estimate_sigma(mu, y, p, unbiased=True)
        simple_result = self.simple_gaussian_sigma(mu, y, p, unbiased=True)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_various_sizes(self):
        """Test with different sample sizes."""
        torch.manual_seed(456)
        for n in [20, 50, 100, 200]:
            p = 5
            mu = torch.randn(n, 1) * 3
            y = torch.randn(n, 1) * 3

            stable_result = gaussian_estimate_sigma(mu, y, p, unbiased=True)
            simple_result = self.simple_gaussian_sigma(mu, y, p, unbiased=True)

            assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)


class TestInverseGaussianEstimateDispersion:
    """Test that numerically stable inverse gaussian dispersion matches simple formula on reasonable inputs."""

    def simple_inversegaussian_dispersion(self, mu: torch.Tensor, y: torch.Tensor, p: int, unbiased: bool = True) -> float:
        """Simple/naive formula for inverse gaussian dispersion estimation."""
        n = mu.shape[0]
        dof = n - (p + 1) if (unbiased and n - (p + 1) > 0) else n
        return (torch.sum(((y - mu) ** 2) / (mu**3)) / dof).item()

    def test_small_values(self):
        """Test with small reasonable values."""
        torch.manual_seed(42)
        n = 100
        p = 5
        mu = torch.rand(n, 1) * 5 + 0.5  # values between 0.5 and 5.5
        y = torch.rand(n, 1) * 5 + 0.5

        stable_result = inversegaussian_estimate_dispersion(mu, y, p, unbiased=True)
        simple_result = self.simple_inversegaussian_dispersion(mu, y, p, unbiased=True)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_unbiased_false(self):
        """Test with unbiased=False."""
        torch.manual_seed(123)
        n = 50
        p = 3
        mu = torch.rand(n, 1) * 3 + 1  # values between 1 and 4
        y = torch.rand(n, 1) * 3 + 1

        stable_result = inversegaussian_estimate_dispersion(mu, y, p, unbiased=False)
        simple_result = self.simple_inversegaussian_dispersion(mu, y, p, unbiased=False)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_larger_values(self):
        """Test with larger but still reasonable values."""
        torch.manual_seed(789)
        n = 80
        p = 4
        mu = torch.rand(n, 1) * 20 + 5  # values between 5 and 25
        y = torch.rand(n, 1) * 20 + 5

        stable_result = inversegaussian_estimate_dispersion(mu, y, p, unbiased=True)
        simple_result = self.simple_inversegaussian_dispersion(mu, y, p, unbiased=True)

        assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)

    def test_various_sizes(self):
        """Test with different sample sizes."""
        torch.manual_seed(456)
        for n in [20, 50, 100, 200]:
            p = 5
            mu = torch.rand(n, 1) * 10 + 1  # values between 1 and 11
            y = torch.rand(n, 1) * 10 + 1

            stable_result = inversegaussian_estimate_dispersion(mu, y, p, unbiased=True)
            simple_result = self.simple_inversegaussian_dispersion(mu, y, p, unbiased=True)

            assert torch.isclose(torch.tensor(stable_result), torch.tensor(simple_result), rtol=1e-5)


class TestWarnings:
    """Test that warnings are issued when unbiased estimation is requested but not possible."""

    def test_gamma_warning_insufficient_dof(self):
        """Test gamma warning when n <= p."""
        torch.manual_seed(42)
        n = 5
        p = 10  # p > n, so unbiased is not possible
        mu = torch.rand(n, 1) * 5
        y = torch.rand(n, 1) * 5

        with pytest.warns(UserWarning, match="Unbiased dispersion estimation requested but insufficient degrees of freedom"):
            gamma_estimate_dispersion(mu, y, p, unbiased=True)

    def test_gamma_no_warning_sufficient_dof(self):
        """Test gamma doesn't warn when n > p."""
        torch.manual_seed(42)
        n = 20
        p = 5
        mu = torch.rand(n, 1) * 5
        y = torch.rand(n, 1) * 5

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            gamma_estimate_dispersion(mu, y, p, unbiased=True)  # Should not raise

    def test_gamma_no_warning_when_unbiased_false(self):
        """Test gamma doesn't warn when unbiased=False even if n <= p."""
        torch.manual_seed(42)
        n = 5
        p = 10
        mu = torch.rand(n, 1) * 5
        y = torch.rand(n, 1) * 5

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gamma_estimate_dispersion(mu, y, p, unbiased=False)  # Should not raise

    def test_gaussian_warning_insufficient_dof(self):
        """Test gaussian warning when n <= p."""
        torch.manual_seed(42)
        n = 5
        p = 10
        mu = torch.randn(n, 1) * 2
        y = torch.randn(n, 1) * 2

        with pytest.warns(UserWarning, match="Unbiased sigma estimation requested but insufficient degrees of freedom"):
            gaussian_estimate_sigma(mu, y, p, unbiased=True)

    def test_gaussian_no_warning_sufficient_dof(self):
        """Test gaussian doesn't warn when n > p."""
        torch.manual_seed(42)
        n = 20
        p = 5
        mu = torch.randn(n, 1) * 2
        y = torch.randn(n, 1) * 2

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gaussian_estimate_sigma(mu, y, p, unbiased=True)  # Should not raise

    def test_gaussian_no_warning_when_unbiased_false(self):
        """Test gaussian doesn't warn when unbiased=False even if n <= p."""
        torch.manual_seed(42)
        n = 5
        p = 10
        mu = torch.randn(n, 1) * 2
        y = torch.randn(n, 1) * 2

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gaussian_estimate_sigma(mu, y, p, unbiased=False)  # Should not raise

    def test_inversegaussian_warning_insufficient_dof(self):
        """Test inverse gaussian warning when n <= p."""
        torch.manual_seed(42)
        n = 5
        p = 10
        mu = torch.rand(n, 1) * 5 + 1
        y = torch.rand(n, 1) * 5 + 1

        with pytest.warns(UserWarning, match="Unbiased dispersion estimation requested but insufficient degrees of freedom"):
            inversegaussian_estimate_dispersion(mu, y, p, unbiased=True)

    def test_inversegaussian_no_warning_sufficient_dof(self):
        """Test inverse gaussian doesn't warn when n > p."""
        torch.manual_seed(42)
        n = 20
        p = 5
        mu = torch.rand(n, 1) * 5 + 1
        y = torch.rand(n, 1) * 5 + 1

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inversegaussian_estimate_dispersion(mu, y, p, unbiased=True)  # Should not raise

    def test_inversegaussian_no_warning_when_unbiased_false(self):
        """Test inverse gaussian doesn't warn when unbiased=False even if n <= p."""
        torch.manual_seed(42)
        n = 5
        p = 10
        mu = torch.rand(n, 1) * 5 + 1
        y = torch.rand(n, 1) * 5 + 1

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inversegaussian_estimate_dispersion(mu, y, p, unbiased=False)  # Should not raise
