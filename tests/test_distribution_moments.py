import torch
import pytest
from torch.distributions import Normal, Gamma

from drn.distributions.histogram import Histogram
from drn.distributions.extended_histogram import ExtendedHistogram


class TestHistogramMoments:
    """Test Histogram mean and variance against Monte Carlo estimates."""

    def test_histogram_mean_simple(self):
        """Test histogram mean with a simple uniform-like distribution."""
        # Create a histogram with 3 bins
        cutpoints = torch.tensor([0.0, 1.0, 2.0, 3.0])
        prob_masses = torch.tensor([[0.3, 0.5, 0.2]])  # One distribution

        hist = Histogram(cutpoints, prob_masses)

        # Analytical mean
        analytical_mean = hist.mean

        # Monte Carlo estimate
        n_samples = 100000
        samples = hist.sample((n_samples,))
        mc_mean = samples.mean(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_mean, mc_mean, atol=0.01, rtol=0.01)

    def test_histogram_variance_simple(self):
        """Test histogram variance with a simple uniform-like distribution."""
        # Create a histogram with 3 bins
        cutpoints = torch.tensor([0.0, 1.0, 2.0, 3.0])
        prob_masses = torch.tensor([[0.3, 0.5, 0.2]])  # One distribution

        hist = Histogram(cutpoints, prob_masses)

        # Analytical variance
        analytical_var = hist.variance

        # Monte Carlo estimate
        n_samples = 100000
        samples = hist.sample((n_samples,))
        mc_var = samples.var(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_var, mc_var, atol=0.02, rtol=0.02)

    def test_histogram_mean_batch(self):
        """Test histogram mean with multiple distributions (batch)."""
        # Create a histogram with 4 bins and 3 different distributions
        cutpoints = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        prob_masses = torch.tensor([
            [0.25, 0.25, 0.25, 0.25],  # Uniform
            [0.1, 0.2, 0.3, 0.4],       # Increasing
            [0.4, 0.3, 0.2, 0.1],       # Decreasing
        ])

        hist = Histogram(cutpoints, prob_masses)

        # Analytical mean
        analytical_mean = hist.mean

        # Monte Carlo estimate
        n_samples = 100000
        samples = hist.sample((n_samples,))
        mc_mean = samples.mean(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_mean, mc_mean, atol=0.01, rtol=0.01)

    def test_histogram_variance_batch(self):
        """Test histogram variance with multiple distributions (batch)."""
        # Create a histogram with 4 bins and 3 different distributions
        cutpoints = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        prob_masses = torch.tensor([
            [0.25, 0.25, 0.25, 0.25],  # Uniform
            [0.1, 0.2, 0.3, 0.4],       # Increasing
            [0.4, 0.3, 0.2, 0.1],       # Decreasing
        ])

        hist = Histogram(cutpoints, prob_masses)

        # Analytical variance
        analytical_var = hist.variance

        # Monte Carlo estimate
        n_samples = 100000
        samples = hist.sample((n_samples,))
        mc_var = samples.var(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_var, mc_var, atol=0.02, rtol=0.02)


class TestHistogramVectorized:
    """Test Histogram in vectorized setting (multiple distributions at once)."""

    def test_histogram_vectorized_shapes(self):
        """Test that shapes are correct in vectorized setting."""
        # Simulate predictions for 10 samples
        batch_size = 10
        cutpoints = torch.tensor([0.0, 1.0, 2.0, 3.0])
        prob_masses = torch.rand(batch_size, 3)
        prob_masses = prob_masses / prob_masses.sum(dim=1, keepdim=True)  # Normalize

        hist = Histogram(cutpoints, prob_masses)

        # Check batch shape
        assert hist.batch_shape == torch.Size([batch_size])

        # Check mean and variance shapes
        assert hist.mean.shape == torch.Size([batch_size])
        assert hist.variance.shape == torch.Size([batch_size])

        # Check single sample shape
        sample_single = hist.sample()
        assert sample_single.shape == torch.Size([batch_size])

        # Check multiple samples shape
        n_samples = 1000
        samples = hist.sample((n_samples,))
        assert samples.shape == torch.Size([n_samples, batch_size])

    def test_histogram_vectorized_mean_variance(self):
        """Test that vectorized sampling produces correct statistics for each distribution."""
        # Create 5 different distributions
        cutpoints = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        prob_masses = torch.tensor([
            [0.4, 0.3, 0.2, 0.1],   # Skewed left
            [0.1, 0.2, 0.3, 0.4],   # Skewed right
            [0.25, 0.25, 0.25, 0.25],  # Uniform
            [0.1, 0.4, 0.4, 0.1],   # Centered
            [0.5, 0.2, 0.2, 0.1],   # Heavy left
        ])

        hist = Histogram(cutpoints, prob_masses)

        # Analytical moments
        analytical_mean = hist.mean
        analytical_var = hist.variance

        # Monte Carlo estimates
        n_samples = 100000
        samples = hist.sample((n_samples,))

        # Compute statistics for each distribution separately
        mc_mean = samples.mean(dim=0)
        mc_var = samples.var(dim=0)

        # Each distribution should match independently
        assert torch.allclose(analytical_mean, mc_mean, atol=0.02, rtol=0.02)
        assert torch.allclose(analytical_var, mc_var, atol=0.03, rtol=0.03)

    def test_histogram_vectorized_independence(self):
        """Test that samples from different distributions are independent."""
        # Create 2 very different distributions
        cutpoints = torch.tensor([0.0, 1.0, 2.0])
        prob_masses = torch.tensor([
            [0.9, 0.1],  # Almost all in first bin
            [0.1, 0.9],  # Almost all in second bin
        ])

        hist = Histogram(cutpoints, prob_masses)

        # Sample many times
        n_samples = 10000
        samples = hist.sample((n_samples,))

        # Check that distribution 0 has samples mostly in [0, 1]
        samples_0 = samples[:, 0]
        assert (samples_0 < 1.0).sum() / n_samples > 0.85  # Should be ~90%

        # Check that distribution 1 has samples mostly in [1, 2]
        samples_1 = samples[:, 1]
        assert (samples_1 >= 1.0).sum() / n_samples > 0.85  # Should be ~90%


class TestExtendedHistogramMoments:
    """Test ExtendedHistogram mean and variance against Monte Carlo estimates."""

    def test_extended_histogram_mean_normal_baseline(self):
        """Test extended histogram mean with Normal baseline."""
        # Create baseline distribution
        baseline = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

        # Create histogram in the middle
        cutpoints = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        prob_masses = torch.tensor([[0.2, 0.3, 0.3, 0.2]])

        ext_hist = ExtendedHistogram(baseline, cutpoints, prob_masses)

        # Analytical mean
        analytical_mean = ext_hist.mean

        # Monte Carlo estimate
        n_samples = 100000
        samples = ext_hist.sample((n_samples,))
        mc_mean = samples.mean(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_mean, mc_mean, atol=0.02, rtol=0.02)

    def test_extended_histogram_variance_normal_baseline(self):
        """Test extended histogram variance with Normal baseline."""
        # Create baseline distribution
        baseline = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

        # Create histogram in the middle
        cutpoints = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        prob_masses = torch.tensor([[0.2, 0.3, 0.3, 0.2]])

        ext_hist = ExtendedHistogram(baseline, cutpoints, prob_masses)

        # Analytical variance
        analytical_var = ext_hist.variance

        # Monte Carlo estimate
        n_samples = 100000
        samples = ext_hist.sample((n_samples,))
        mc_var = samples.var(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_var, mc_var, atol=0.03, rtol=0.03)

    def test_extended_histogram_mean_gamma_baseline(self):
        """Test extended histogram mean with Gamma baseline (positive support)."""
        # Create baseline distribution (Gamma has positive support)
        baseline = Gamma(
            concentration=torch.tensor([2.0]),
            rate=torch.tensor([1.0])
        )

        # Create histogram in the middle of the support
        cutpoints = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
        prob_masses = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

        ext_hist = ExtendedHistogram(baseline, cutpoints, prob_masses)

        # Analytical mean
        analytical_mean = ext_hist.mean

        # Monte Carlo estimate
        n_samples = 100000
        samples = ext_hist.sample((n_samples,))
        mc_mean = samples.mean(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_mean, mc_mean, atol=0.03, rtol=0.03)

    def test_extended_histogram_variance_gamma_baseline(self):
        """Test extended histogram variance with Gamma baseline (positive support)."""
        # Create baseline distribution (Gamma has positive support)
        baseline = Gamma(
            concentration=torch.tensor([2.0]),
            rate=torch.tensor([1.0])
        )

        # Create histogram in the middle of the support
        cutpoints = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
        prob_masses = torch.tensor([[0.25, 0.25, 0.25, 0.25]])

        ext_hist = ExtendedHistogram(baseline, cutpoints, prob_masses)

        # Analytical variance
        analytical_var = ext_hist.variance

        # Monte Carlo estimate
        n_samples = 100000
        samples = ext_hist.sample((n_samples,))
        mc_var = samples.var(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_var, mc_var, atol=0.05, rtol=0.05)

    def test_extended_histogram_mean_batch(self):
        """Test extended histogram mean with batch of distributions."""
        # Create baseline distributions (batch of 3)
        baseline = Normal(
            loc=torch.tensor([0.0, 1.0, -1.0]),
            scale=torch.tensor([1.0, 0.5, 1.5])
        )

        # Create histogram in the middle
        cutpoints = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        prob_masses = torch.tensor([
            [0.2, 0.3, 0.3, 0.2],
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.4, 0.4, 0.1],
        ])

        ext_hist = ExtendedHistogram(baseline, cutpoints, prob_masses)

        # Analytical mean
        analytical_mean = ext_hist.mean

        # Monte Carlo estimate
        n_samples = 100000
        samples = ext_hist.sample((n_samples,))
        mc_mean = samples.mean(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_mean, mc_mean, atol=0.03, rtol=0.03)

    def test_extended_histogram_variance_batch(self):
        """Test extended histogram variance with batch of distributions."""
        # Create baseline distributions (batch of 3)
        baseline = Normal(
            loc=torch.tensor([0.0, 1.0, -1.0]),
            scale=torch.tensor([1.0, 0.5, 1.5])
        )

        # Create histogram in the middle
        cutpoints = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        prob_masses = torch.tensor([
            [0.2, 0.3, 0.3, 0.2],
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.4, 0.4, 0.1],
        ])

        ext_hist = ExtendedHistogram(baseline, cutpoints, prob_masses)

        # Analytical variance
        analytical_var = ext_hist.variance

        # Monte Carlo estimate
        n_samples = 100000
        samples = ext_hist.sample((n_samples,))
        mc_var = samples.var(dim=0)

        # Check they're close (with generous tolerance for MC)
        assert torch.allclose(analytical_var, mc_var, atol=0.05, rtol=0.05)
