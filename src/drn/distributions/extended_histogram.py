import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Distribution

from .histogram import Histogram
from ..utils import binary_search_icdf


class ExtendedHistogram(Distribution):
    """
    This class represents a splicing of a supplied distribution with a histogram distribution.
    The histogram part is defined by K regions with boundaries -infty < c_0 < c_1 < ... < c_K < infty.
    The final density before c_0 & after c_K is the same as the original distribution.
    The density between c_k & c_{k+1} is defined by the histogram distribution.
    """

    def __init__(
        self,
        baseline: Distribution,
        cutpoints: torch.Tensor,
        pmf: torch.Tensor,
        baseline_probs: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            baseline: the original distribution
            cutpoints: the bin boundaries (shape: (K+1,))
            pmf: the refined (cond.) probability for landing in each region (shape: (n, K))
            baseline_probs: the baseline's probability for landing in each region (shape: (n, K))
        """
        self.baseline = baseline
        self.cutpoints = cutpoints
        self.prob_masses = pmf
        self.baseline_probs = baseline_probs
        self.histogram = Histogram(cutpoints, pmf)
        self.scale_down_hist = baseline.cdf(cutpoints[-1]) - baseline.cdf(cutpoints[0])

        assert self.scale_down_hist.shape == torch.Size([self.histogram.batch_shape[0]])

        super(ExtendedHistogram, self).__init__(
            batch_shape=self.histogram.batch_shape, validate_args=False
        )

    def baseline_prob_between_cutpoints(self) -> torch.Tensor:
        """
        Calculate the baseline probability vector
        """
        if self.baseline_probs is None:
            baseline_cdfs = self.baseline.cdf(self.cutpoints.unsqueeze(-1)).T
            self.baseline_probs = torch.diff(baseline_cdfs, dim=1)

        return self.baseline_probs

    def real_adjustments(self) -> torch.Tensor:
        """
        Calculate the real adjustment factors a_k's
        """
        return self.prob_masses / self.baseline_prob_between_cutpoints()

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability densities of `values`.
        """

        orig_ndim = value.ndim

        # Ensure the last dimension of value matches the batch_shape
        if value.shape[-1] != self.batch_shape[0]:
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            value = value.expand(-1, self.batch_shape[0])

        # Ensure value is 2D
        if value.ndim == 1:
            value = value.unsqueeze(0)

        baseline_prob = torch.exp(self.baseline.log_prob(value))
        baseline_prob = torch.clip(baseline_prob, min=1e-10, max=1.0)
        hist_prob = self.histogram.prob(value) * (self.scale_down_hist + 1e-10)

        in_hist = (value >= self.histogram.cutpoints[0]) & (
            value < self.histogram.cutpoints[-1]
        )
        in_baseline = ~in_hist

        probabilities = torch.zeros_like(baseline_prob)
        probabilities[in_baseline] = baseline_prob[in_baseline]
        probabilities[in_hist] = hist_prob[in_hist]

        return probabilities

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob(value))

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cumulative distribution function for the given values.
        """
        baseline_cdf = self.baseline.cdf(value)
        hist_cdf = self.histogram.cdf(value) * self.scale_down_hist
        in_hist = (value >= self.histogram.cutpoints[0]) & (
            value < self.histogram.cutpoints[-1]
        )
        in_hist = (
            in_hist.expand(value.shape[0], self.batch_shape[0])
            if in_hist.ndim > 1
            else in_hist
        )
        in_baseline = ~in_hist

        lower_cdf = self.baseline.cdf(self.histogram.cutpoints[0])
        cdf_values = torch.zeros_like(baseline_cdf)

        cdf_values[in_baseline] = baseline_cdf[in_baseline]
        cdf_values[in_hist] = (lower_cdf + hist_cdf)[in_hist]

        return cdf_values

    def cdf_at_cutpoints(self) -> torch.Tensor:
        """
        Calculate the cumulative distribution function at each cutpoint.
        """
        hist_at_cutpoints = (
            self.histogram.cdf_at_cutpoints() * self.scale_down_hist.unsqueeze(0)
        )
        lower_cdf = self.baseline.cdf(self.histogram.cutpoints[0]).unsqueeze(0)
        out = lower_cdf + hist_at_cutpoints
        return out

    @property
    def mean(self) -> torch.Tensor:
        """
        Calculate the mean of the distribution.
        Returns:
            the mean (shape: (batch_shape,))
        """
        middle_of_bins = (self.cutpoints[1:] + self.cutpoints[:-1]) / 2
        return torch.sum(self.prob_masses * middle_of_bins, dim=1)

    def icdf(self, p, l=None, u=None, max_iter=1000, tolerance=1e-7) -> torch.Tensor:
        """
        Calculate the inverse CDF (quantiles) using shared binary search implementation.
        """
        return binary_search_icdf(self, p, l, u, max_iter, tolerance)

    def quantiles(
        self, percentiles: list, l=None, u=None, max_iter=1000, tolerance=1e-7
    ) -> torch.Tensor:
        """
        Calculate the quantile values for the given percentiles (cumulative probabilities * 100).
        """
        quantiles = [
            self.icdf(percentile / 100.0, l, u, max_iter, tolerance)
            for percentile in percentiles
        ]
        return torch.stack(quantiles, dim=1)[0]

    def __repr__(self):
        base = self.baseline
        return (
            f"{self.__class__.__name__}("
            f"baseline: {base}, "
            f"cutpoints: {self.cutpoints.shape}, "
            f"prob_masses: {self.prob_masses.shape})"
        )
