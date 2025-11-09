import torch
from torch.distributions import Distribution, Categorical, Uniform, MixtureSameFamily, Independent
from ..utils import binary_search_icdf


class Histogram(Distribution):
    """
    This class represents a histogram distribution.
    Basically, the distribution is a composite of uniform distributions over the bins.
    """

    def __init__(self, cutpoints: torch.Tensor, prob_masses: torch.Tensor):
        """
        Args:
            regions: the bin boundaries (shape: (K+1,))
            prob_masses: the probability for landing in each regions (shape: (n, K))
        """

        # Constructed regions T_k \in [c_k, c_{k+1}) for all k \in \{0, ..., K-1\}
        self.cutpoints = cutpoints
        self.num_regions = len(self.cutpoints) - 1

        # Predicted probabilities vector: (Pr(c_0<Y<c_1|x,w_{Histogram}),..., Pr(c_{K-1}<Y<c_K|x,w_{Histogram}))
        self.prob_masses = prob_masses
        assert torch.allclose(
            torch.sum(self.prob_masses, dim=1),
            torch.ones(self.prob_masses.shape[0], device=cutpoints.device),
        )
        assert self.prob_masses.shape[1] == self.num_regions

        # Compute the bin widths for later use, i.e., (T_1 = c_1 - c_0,..., T_K = c_K - c_{K-1})
        self.bin_widths = self.cutpoints[1:] - self.cutpoints[:-1]
        assert torch.all(self.bin_widths > 0)

        # Compute the PDF values using the probability masses and bin widths, i.e., normalising
        # Pr(Y\in T_k|x,w_{Histogram})/T_k * Pr(c_0<Y<c_K|x,w_{Baseline})
        self.prob_densities = self.prob_masses / self.bin_widths

        super(Histogram, self).__init__(
            batch_shape=torch.Size([prob_masses.shape[0]]), validate_args=False
        )

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

        # Initialize CDF to zeros with the same shape as value
        probabilities = torch.zeros_like(value)

        # Go through each observation vector and calculate pdf over all batch_size distributions
        for i in range(value.shape[0]):
            # Calculate the pdf for the `y` batch
            y = value[i, :]

            # Iterate over each bin
            for r in range(self.num_regions):
                in_bin = (y >= self.cutpoints[r]) & (y < self.cutpoints[r + 1])
                probabilities[i, in_bin] = self.prob_densities[in_bin, r]

        # If we added a leading dimension, remove it
        if orig_ndim == 1 and probabilities.ndim == 2:
            probabilities = probabilities.squeeze(0)

        return probabilities

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log probability densities of `values`.
        """
        return torch.log(self.prob(value))

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cumulative distribution function for the given values.
        """
        orig_ndim = value.ndim
        # Ensure the last dimension of value matches the batch_shape
        if value.shape[-1] != self.batch_shape[0]:  # and value.shape[-1] == 1:
            return self.cdf_same_eval(value)

        # Ensure value is 2D
        if value.ndim == 1:
            value = value.unsqueeze(0)

        # Initialize CDF to zeros with the same shape as value
        cdf_values = torch.zeros_like(value)

        # Original cdf
        cumulative_cdf = torch.cumsum(self.prob_masses, dim=-1).T

        # Iterate over each bin
        for j in range(value.shape[0]):

            y = value[j, :]

            # Conditions now compare each element in y against the cutpoints
            # This assumes the intent is to apply the condition across all dimensions uniformly
            condition_above_last_cutpoint = y >= self.cutpoints[-1]
            condition_below_first_cutpoint = y <= self.cutpoints[0]

            # Since y is a vector, the assignment needs to respect the condition per element
            cdf_values[j, condition_above_last_cutpoint] = 1.0
            cdf_values[j, condition_below_first_cutpoint] = 0.0
            # print(condition_above_last_cutpoint, y, cdf_values)

            # Determine the index of the cutpoints
            y_expanded = y.unsqueeze(0)  # Now y has shape [1, n]
            cutpoints_below = self.cutpoints[:-1].unsqueeze(
                1
            )  # Now self.cutpoints[:-1] has shape [K, 1]
            cutpoints_above = self.cutpoints[1:].unsqueeze(1)

            # Perform comparison to determine bins
            comparison_result_below = y_expanded >= cutpoints_below  # [K, n]
            comparison_result_above = y_expanded < cutpoints_above  # [K, n]
            valid_bins = comparison_result_below & comparison_result_above  # [K, n]

            # Find the last valid bin for each element
            last_bin_idx = valid_bins.long().argmax(dim=0)  # [n]
            # Ensure elements outside the cutpoints are handled correctly
            below_min_mask = y_expanded.squeeze() <= self.cutpoints[0]
            above_max_mask = y_expanded.squeeze() >= self.cutpoints[-1]

            # Update last_bin_idx for values outside the cutpoints
            last_bin_idx[below_min_mask] = (
                0  # First bin for values below the minimum cutpoint
            )
            last_bin_idx[above_max_mask] = (
                len(self.cutpoints) - 2
            )  # Last valid bin index for values above the maximum cutpoint

            # Determine next_bin_idx based on last_bin_idx
            next_bin_idx = last_bin_idx + 1
            # Ensure next_bin_idx does not exceed the number of bins
            next_bin_idx[above_max_mask] = len(self.cutpoints) - 1

            # Initialize 'last_cdfs' and 'next_cdfs' with zeros and ones, respectively
            zeros = torch.zeros(size=(1, value.shape[1]))
            ones = torch.ones(size=(1, value.shape[1]))

            # Ensure y is properly shaped for broadcasting. If y is already [1, n], this step might be redundant
            y = y.reshape(1, -1)  # Ensure y has shape [1, n]

            # Reshape or select cutpoints for broadcasting
            # If comparing against a single cutpoint, ensure it's shaped for broadcasting
            cutpoint_for_comparison_lower = self.cutpoints[1].reshape(
                1, -1
            )  # Ensure cutpoint is shaped [1, 1] or similar for broadcasting
            cutpoint_for_comparison_upper = self.cutpoints[-2].reshape(1, -1)

            # Perform the comparison
            condition_last_cdfs = (
                y >= cutpoint_for_comparison_lower
            )  # This should now result in a shape [1, n]
            condition_next_cdfs = y < cutpoint_for_comparison_upper

            # Update 'last_cdfs' based on condition, considering 'cumulative_cdf' indexing
            # Ensure 'last_bin_idx' and 'next_bin_idx' are correctly broadcasted or indexed to match 'y's dimensions
            last_cdfs = torch.where(
                condition_last_cdfs,
                cumulative_cdf[
                    (last_bin_idx - 1).unsqueeze(0),
                    torch.arange(cumulative_cdf.shape[1]),
                ],
                zeros,
            )
            next_cdfs = torch.where(
                condition_next_cdfs,
                cumulative_cdf[
                    (next_bin_idx - 1).unsqueeze(0),
                    torch.arange(cumulative_cdf.shape[1]),
                ],
                ones,
            )

            last_bin_idx_expanded = last_bin_idx.unsqueeze(0).expand(y.shape[0], -1)
            next_bin_idx_expanded = next_bin_idx.unsqueeze(0).expand(y.shape[0], -1)

            # Determine cutpoints based on conditions
            cutpoint_low = torch.where(
                y < self.cutpoints[1].unsqueeze(0),
                self.cutpoints[0],
                self.cutpoints[last_bin_idx_expanded],
            )
            cutpoint_high = torch.where(
                y >= self.cutpoints[-2].unsqueeze(0),
                self.cutpoints[-1],
                self.cutpoints[next_bin_idx_expanded],
            )

            # Determine bin_width based on conditions
            bin_width = torch.where(
                y < self.cutpoints[1].unsqueeze(0),
                self.bin_widths[0],
                self.bin_widths[last_bin_idx_expanded],
            )

            # Compute bin_fraction for each feature across all observations
            bin_fraction = (y - cutpoint_low) / bin_width

            # Compute cdf_values using the previously obtained last_cdfs and next_cdfs for each feature across all observations
            # Ensure last_cdfs and next_cdfs are retrieved using the approach described in previous steps and have the correct shape
            cdf_values[j, :] = last_cdfs + (next_cdfs - last_cdfs) * bin_fraction

        cdf_values = torch.clamp(cdf_values, max=1.0, min=0.0)

        # If we added a leading dimension, remove it
        if orig_ndim == 1 and cdf_values.ndim == 2:
            cdf_values = cdf_values.squeeze(0)

        return cdf_values

    def cdf_same_eval(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cumulative distribution function for the same value across the batch.
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

        # Initialize CDF to zeros with the same shape as value
        cdf_values = torch.zeros_like(value)

        # Original cdf
        cumulative_cdf = torch.cumsum(self.prob_masses, dim=-1).T

        # Iterate over each bin
        for j in range(value.shape[0]):
            y = value[j, 0]
            if y >= self.cutpoints[-1]:
                cdf_values[j, :] = torch.ones(size=(1, value.shape[1]))
                continue

            if y <= self.cutpoints[0]:
                cdf_values[j, :] = torch.zeros(size=(1, value.shape[1]))
                continue

            # Determine the index of the cutpoints
            last_bin_idx = (
                (y >= self.cutpoints[:-1]) & (y < self.cutpoints[1:])
            ).nonzero(as_tuple=True)[0]
            next_bin_idx = last_bin_idx + 1

            # Set cdfs for the lower and upper bounds
            last_cdfs = (
                torch.zeros(size=(1, value.shape[1]), device=value.device)
                if y < self.cutpoints[1]
                else cumulative_cdf[last_bin_idx - 1, :]
            )
            next_cdfs = (
                torch.ones(size=(1, value.shape[1]), device=value.device)
                if y >= self.cutpoints[-2]
                else cumulative_cdf[next_bin_idx - 1, :]
            )

            # Compute cdf_values
            cutpoint_low = (
                self.cutpoints[0]
                if y < self.cutpoints[1]
                else self.cutpoints[last_bin_idx]
            )
            cutpoint_high = (
                self.cutpoints[-1]
                if y >= self.cutpoints[-2]
                else self.cutpoints[next_bin_idx]
            )
            bin_width = (
                self.bin_widths[0]
                if y < self.cutpoints[1]
                else self.bin_widths[last_bin_idx]
            )
            bin_fraction = (y - cutpoint_low) / bin_width
            cdf_values[j, :] = last_cdfs + (next_cdfs - last_cdfs) * bin_fraction

        cdf_values = torch.clamp(cdf_values, max=1.0, min=0.0)

        # If we added a leading dimension, remove it
        if orig_ndim == 1 and cdf_values.ndim == 2:
            cdf_values = cdf_values.squeeze(0)

        return cdf_values

    def cdf_at_cutpoints(self) -> torch.Tensor:
        """
        Calculate the cumulative distribution function at each cutpoint.
        """
        # Want to use cumsum but add a 0 to the beginning, torch.cumsum(self.prob_masses, dim=1)
        return torch.cat(
            [
                torch.zeros(
                    self.prob_masses.shape[0], 1, device=self.prob_masses.device
                ),
                torch.cumsum(self.prob_masses, dim=1),
            ],
            dim=1,
        ).T

    @property
    def mean(self) -> torch.Tensor:
        """
        Calculate the mean of the distribution.
        Returns:
            the mean (shape: (batch_shape,))
        """
        middle_of_bins = (self.cutpoints[1:] + self.cutpoints[:-1]) / 2
        return torch.sum(self.prob_masses * middle_of_bins, dim=1)

    @property
    def variance(self) -> torch.Tensor:
        """
        Calculate the variance of the distribution.

        For a piecewise uniform distribution, the variance is computed as:
        Var(X) = E[X²] - E[X]²

        For uniform distribution over [a, b]:
        E[X²] = (a² + ab + b²) / 3

        Returns:
            the variance (shape: (batch_shape,))
        """
        lower = self.cutpoints[:-1]
        upper = self.cutpoints[1:]

        # E[X²] for uniform over [a,b] = (a² + ab + b²)/3
        second_moment_bins = (lower**2 + lower * upper + upper**2) / 3
        second_moment = torch.sum(self.prob_masses * second_moment_bins, dim=1)

        mean = self.mean
        return second_moment - mean**2

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
        Sample from the histogram distribution using PyTorch's MixtureSameFamily.

        For each batch element, we:
        1. Sample which bin to use based on prob_masses (Categorical)
        2. Sample uniformly within that bin (Uniform)

        Args:
            sample_shape: The shape of samples to draw

        Returns:
            Tensor of shape sample_shape + batch_shape
        """
        # Create categorical distribution for bin selection
        # prob_masses has shape (batch_size, num_regions)
        mix = Categorical(probs=self.prob_masses)

        # Create uniform distributions for each bin
        # Need to expand cutpoints to shape (batch_size, num_regions)
        lower = self.cutpoints[:-1].unsqueeze(0).expand(self.batch_shape[0], -1)
        upper = self.cutpoints[1:].unsqueeze(0).expand(self.batch_shape[0], -1)

        # Create batch of Uniform distributions with batch_shape=(batch_size, num_regions)
        comp = Uniform(lower, upper)

        # Create mixture distribution
        mixture = MixtureSameFamily(mix, comp)

        # Sample from the mixture
        return mixture.sample(sample_shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutpoints: {self.cutpoints.shape}, prob_masses: {self.prob_masses.shape})"
