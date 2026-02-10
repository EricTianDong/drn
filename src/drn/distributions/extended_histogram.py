import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Distribution, Categorical

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
        DRN mean using Histogram for the middle block and baseline for tails.

        Implements:
            E[Y] = p_L * mu_L  +  p_mid * mu_mid  +  p_R * mu_R

        where
            p_L   = F_beta(c_0)
            p_mid = F_beta(c_K) - F_beta(c_0)   (== self.scale_down_hist)
            p_R   = 1 - F_beta(c_K)

            mu_mid = E[Y | c_0 ≤ Y < c_K] from the Histogram (conditional on middle)
            mu_L, mu_R = baseline truncated means over the left/right tails
                         (left here in terms of conditional expectations; computed
                          numerically via _truncated_moments if sampling is available).
        """
        # Cutpoints as 0-D tensors (keep device/dtype)
        c0 = self.cutpoints[0]
        cK = self.cutpoints[-1]

        # Region masses from baseline CDF
        p_left  = self.baseline.cdf(c0)          # shape: (batch,)
        p_right = 1 - self.baseline.cdf(cK)      # shape: (batch,)
        p_mid   = self.scale_down_hist           # shape: (batch,)

        # Middle conditional mean from verified Histogram implementation
        mu_mid = self.histogram.mean             # shape: (batch,)

        # Tail truncated means (keep integrals; MC fallback if sampling available)
        if hasattr(self.baseline, "sample"):
            c0_f = float(c0.item())
            cK_f = float(cK.item())
            span = float((cK - c0).item())
            # Left: (-inf, c0)
            mu_L = self._truncated_moments(float("-inf"), c0_f, order=1)
            # Right: [cK, +inf) approximated with a large finite bound
            upper_approx = cK_f + 10.0 * span
            mu_R = self._truncated_moments(cK_f, upper_approx, order=1)
        else:
            # Rough deterministic fallbacks if baseline sampling isn't available
            mu_L = c0 / 2
            mu_R = cK * 1.5

        # Combine pieces
        # E[Y] = p_L * mu_L + p_mid * mu_mid + p_R * mu_R
        return p_left * mu_L + p_mid * mu_mid + p_right * mu_R

    def _truncated_moments(self, lower_bound: float, upper_bound: float, order: int = 1, n_samples: int = 100000) -> torch.Tensor:
        """
        Compute truncated moments of the baseline distribution using Monte Carlo sampling.

        Args:
        lower_bound: Lower bound for truncation
        upper_bound: Upper bound for truncation
        order: Moment order (1 for mean, 2 for second moment)
        n_samples: Number of samples for Monte Carlo estimation

        Returns:
        Truncated moment E[X^order | lower_bound ≤ X < upper_bound]
        """
        # Sample from baseline
        samples = self.baseline.sample((n_samples,))

        # Filter samples within bounds
        mask = (samples >= lower_bound) & (samples < upper_bound)

        # Compute moment for each batch element
        moments = torch.zeros(self.batch_shape[0], device=samples.device)

        for i in range(self.batch_shape[0]):
            valid_samples = samples[mask[:, i], i]
            if len(valid_samples) > 0:
                moments[i] = (valid_samples**order).mean()
            else:
                # Fallback: use finite approximation or boundary
                # If lower_bound is -inf, use upper_bound - span
                # If upper_bound is +inf, use lower_bound + span
                if lower_bound == float("-inf") and upper_bound != float("inf"):
                    # Left tail: use value slightly below upper bound
                    fallback_val = upper_bound - abs(upper_bound) if upper_bound != 0 else -1.0
                elif upper_bound == float("inf") and lower_bound != float("-inf"):
                    # Right tail: use value slightly above lower bound
                    fallback_val = lower_bound + abs(lower_bound) if lower_bound != 0 else 1.0
                else:
                    # Finite bounds: use midpoint
                    fallback_val = (lower_bound + upper_bound) / 2
                moments[i] = fallback_val**order

        return moments


    @property
    def variance(self) -> torch.Tensor:
        """
        DRN variance using Histogram for the middle block and baseline for tails.

        Implements:
            E[Y^2] = p_L * M2_L
                   + p_mid * (var_mid + mu_mid^2)
                   + p_R * M2_R

            Var(Y) = E[Y^2] - (E[Y])^2

        where
            p_L   = F_beta(c_0)
            p_mid = F_beta(c_K) - F_beta(c_0)   (== self.scale_down_hist)
            p_R   = 1 - F_beta(c_K)

            mu_mid, var_mid come from the Histogram (conditional on middle)
            M2_L  = E[Y^2 | Y < c_0] (baseline, left as an integral)
            M2_R  = E[Y^2 | Y ≥ c_K] (baseline, left as an integral)
        """
        # Cutpoints as 0-D tensors (keep device/dtype)
        c0 = self.cutpoints[0]
        cK = self.cutpoints[-1]

        # Region masses from baseline CDF
        p_left  = self.baseline.cdf(c0)          # shape: (batch,)
        p_right = 1 - self.baseline.cdf(cK)      # shape: (batch,)
        p_mid   = self.scale_down_hist           # shape: (batch,)

        # Middle conditional moments from verified Histogram implementation
        mu_mid  = self.histogram.mean            # E[Y | middle]
        var_mid = self.histogram.variance        # Var(Y | middle)
        M2_mid  = var_mid + mu_mid**2            # E[Y^2 | middle]

        # Tail truncated moments (keep integrals; MC fallback if sampling available)
        if hasattr(self.baseline, "sample"):
            c0_f = float(c0.item())
            cK_f = float(cK.item())
            span = float((cK - c0).item())
            # Left: (-inf, c0)
            mu_L  = self._truncated_moments(float("-inf"), c0_f, order=1)
            M2_L  = self._truncated_moments(float("-inf"), c0_f, order=2)
            # Right: [cK, +inf) approximated with a large finite bound
            upper_approx = cK_f + 10.0 * span
            mu_R  = self._truncated_moments(cK_f, upper_approx, order=1)
            M2_R  = self._truncated_moments(cK_f, upper_approx, order=2)
        else:
            # Rough deterministic fallbacks if baseline sampling isn't available
            mu_L = c0 / 2
            M2_L = (c0**2) / 3
            mu_R = cK * 1.5
            M2_R = cK**2 * 2

        # First moment (mean) using the same composition as in .mean
        # E[Y] = p_L * mu_L + p_mid * mu_mid + p_R * mu_R
        EY  = p_left * mu_L + p_mid * mu_mid + p_right * mu_R

        # Second moment composition:
        # E[Y^2] = p_L * M2_L + p_mid * M2_mid + p_R * M2_R
        EY2 = p_left * M2_L + p_mid * M2_mid + p_right * M2_R

        # Var(Y) = E[Y^2] - (E[Y])^2
        return EY2 - EY**2

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

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample from the extended histogram distribution.

        The distribution is a splice of:
        - Left tail: baseline distribution for Y < c_0
        - Middle: histogram distribution for c_0 <= Y < c_K
        - Right tail: baseline distribution for Y >= c_K

        Uses PyTorch's Categorical to select regions, then samples from
        the appropriate distribution for each region.

        Args:
            sample_shape: The shape of samples to draw

        Returns:
            Tensor of shape sample_shape + batch_shape
        """
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape])
        elif not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        total_samples = sample_shape.numel() if len(sample_shape) > 0 else 1
        batch_size = self.batch_shape[0]

        c0 = self.cutpoints[0]
        cK = self.cutpoints[-1]

        # Calculate region probabilities using Categorical distribution
        p_left = self.baseline.cdf(c0)  # Shape: (batch_size,)
        p_right = 1 - self.baseline.cdf(cK)  # Shape: (batch_size,)
        p_mid = self.scale_down_hist  # Shape: (batch_size,)

        # Stack probabilities: (batch_size, 3) where 3 regions are [left, middle, right]
        region_probs = torch.stack([p_left, p_mid, p_right], dim=1)

        # Use Categorical to sample regions for all batch elements at once
        # Expand to (total_samples, batch_size, 3) for sampling
        region_probs_expanded = region_probs.unsqueeze(0).expand(total_samples, -1, -1)

        # Sample regions: shape (total_samples, batch_size)
        region_cat = Categorical(probs=region_probs_expanded)
        region_indices = region_cat.sample()

        # Initialize output
        samples = torch.zeros(total_samples, batch_size, device=self.cutpoints.device)

        # Process each batch element
        for i in range(batch_size):
            regions_i = region_indices[:, i]

            # Left tail (region 0): rejection sample from baseline for Y < c_0
            left_mask = regions_i == 0
            if left_mask.any():
                samples[left_mask, i] = self._rejection_sample_tail(
                    n_samples=left_mask.sum().item(),
                    batch_idx=i,
                    lower_bound=float('-inf'),
                    upper_bound=c0.item()
                )

            # Middle (region 1): sample from histogram
            mid_mask = regions_i == 1
            if mid_mask.any():
                samples[mid_mask, i] = self.histogram.sample((mid_mask.sum().item(),))[:, i]

            # Right tail (region 2): rejection sample from baseline for Y >= c_K
            right_mask = regions_i == 2
            if right_mask.any():
                samples[right_mask, i] = self._rejection_sample_tail(
                    n_samples=right_mask.sum().item(),
                    batch_idx=i,
                    lower_bound=cK.item(),
                    upper_bound=float('inf')
                )

        # Reshape to match sample_shape + batch_shape
        if len(sample_shape) > 0:
            samples = samples.reshape(sample_shape + self.batch_shape)
        else:
            samples = samples.squeeze(0)

        return samples

    def _rejection_sample_tail(self, n_samples: int, batch_idx: int,
                                lower_bound: float, upper_bound: float) -> torch.Tensor:
        """
        Helper method for rejection sampling from baseline in tail regions.

        Args:
            n_samples: Number of samples needed
            batch_idx: Index of batch element to sample from
            lower_bound: Lower bound for acceptance (-inf for left tail)
            upper_bound: Upper bound for acceptance (inf for right tail)

        Returns:
            Tensor of shape (n_samples,) with samples from the truncated baseline
        """
        collected = []
        while len(collected) < n_samples:
            # Oversample to reduce iterations (10x oversample)
            n_needed = n_samples - len(collected)
            n_to_sample = max(n_needed * 10, 100)

            # Sample from baseline
            baseline_samples = self.baseline.sample((n_to_sample,))[:, batch_idx]

            # Filter to valid range
            valid = (baseline_samples >= lower_bound) & (baseline_samples < upper_bound)
            collected.extend(baseline_samples[valid].tolist())

        return torch.tensor(collected[:n_samples], device=self.cutpoints.device)

    def __repr__(self):
        base = self.baseline
        return (
            f"{self.__class__.__name__}("
            f"baseline: {base}, "
            f"cutpoints: {self.cutpoints.shape}, "
            f"prob_masses: {self.prob_masses.shape})"
        )
