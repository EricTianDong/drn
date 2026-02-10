"""
Test suite to verify equivalence and performance of fast merge_cutpoints implementation.
"""
import numpy as np
import pytest
from time import time
from drn.models.drn import merge_cutpoints


def merge_cutpoints_old(cutpoints: list[float], y: np.ndarray, min_obs: int) -> list[float]:
    """Original (slow) implementation of merge_cutpoints for comparison."""
    print("Merging cutpoints (old)", flush=True)
    from time import time
    start_time = time()
    # Ensure cutpoints are sorted and unique to start with
    cutpoints = sorted(np.unique(cutpoints).tolist())
    assert len(cutpoints) >= 2

    new_cutpoints = [cutpoints[0]]  # Start with the first cutpoint
    left = 0

    for right in range(1, len(cutpoints) - 1):
        num_in_region = np.sum((y >= cutpoints[left]) & (y < cutpoints[right]))
        num_after_region = np.sum((y >= cutpoints[right]) & (y < cutpoints[-1]))

        if num_in_region >= min_obs and num_after_region >= min_obs:
            new_cutpoints.append(cutpoints[right])
            left = right

    new_cutpoints.append(cutpoints[-1])  # End with the last cutpoint

    elapsed = time() - start_time
    print(f"Merge cutpoints took {elapsed}", flush=True)

    return new_cutpoints


class TestMergeCutpointsEquivalence:
    """Test that fast implementation produces identical results to original."""

    def test_simple_case(self):
        """Test with a simple dataset where no merging is needed."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutpoints = [0.0, 2.5, 6.0]
        min_obs = 1

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_requires_merging(self):
        """Test case where some cutpoints need to be merged."""
        y = np.array([1.0, 1.1, 5.0, 5.1, 5.2, 9.0, 9.1])
        cutpoints = [0.0, 2.0, 4.0, 6.0, 10.0]
        min_obs = 2

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_minimum_cutpoints(self):
        """Test with minimum number of cutpoints (2)."""
        y = np.array([1.0, 2.0, 3.0])
        cutpoints = [0.0, 5.0]
        min_obs = 1

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_many_cutpoints_sparse_data(self):
        """Test with many cutpoints but sparse data."""
        y = np.array([1.0, 10.0])
        cutpoints = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        min_obs = 1

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_uniform_distribution(self):
        """Test with uniformly distributed data."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=1000)
        cutpoints = np.linspace(0, 105, 50).tolist()
        min_obs = 10

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_skewed_distribution(self):
        """Test with skewed distribution."""
        np.random.seed(42)
        y = np.random.exponential(scale=2.0, size=1000)
        cutpoints = np.linspace(0, y.max() * 1.1, 100).tolist()
        min_obs = 5

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_clustered_data(self):
        """Test with clustered data."""
        np.random.seed(42)
        cluster1 = np.random.normal(10, 1, 100)
        cluster2 = np.random.normal(50, 2, 100)
        cluster3 = np.random.normal(90, 1.5, 100)
        y = np.concatenate([cluster1, cluster2, cluster3])
        cutpoints = np.linspace(0, 100, 50).tolist()
        min_obs = 3

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_duplicate_cutpoints(self):
        """Test with duplicate cutpoints in input."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutpoints = [0.0, 2.5, 2.5, 2.5, 6.0]
        min_obs = 1

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_unsorted_cutpoints(self):
        """Test with unsorted cutpoints."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cutpoints = [6.0, 0.0, 2.5, 4.0]
        min_obs = 1

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"

    def test_high_min_obs(self):
        """Test with high min_obs that forces aggressive merging."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=100)
        cutpoints = np.linspace(0, 105, 30).tolist()
        min_obs = 20

        result_old = merge_cutpoints_old(cutpoints, y, min_obs)
        result_new = merge_cutpoints(cutpoints, y, min_obs)

        assert result_old == result_new, \
            f"Results differ: old={result_old}, new={result_new}"


class TestMergeCutpointsPerformance:
    """Test performance comparison between implementations."""

    def _benchmark(self, func, cutpoints, y, min_obs, n_runs=5):
        """Run a function multiple times and return average time."""
        times = []
        for _ in range(n_runs):
            start = time()
            result = func(cutpoints, y, min_obs)
            elapsed = time() - start
            times.append(elapsed)
        return np.mean(times), np.std(times), result

    def test_performance_small(self):
        """Benchmark with small dataset (1K observations, 50 cutpoints)."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=1000)
        cutpoints = np.linspace(0, 105, 50).tolist()
        min_obs = 10

        time_old, std_old, result_old = self._benchmark(merge_cutpoints_old, cutpoints, y, min_obs)
        time_new, std_new, result_new = self._benchmark(merge_cutpoints, cutpoints, y, min_obs)

        speedup = time_old / time_new

        print(f"\n--- Small dataset (1K obs, 50 cutpoints) ---")
        print(f"Old: {time_old*1000:.2f} ± {std_old*1000:.2f} ms")
        print(f"New: {time_new*1000:.2f} ± {std_new*1000:.2f} ms")
        print(f"Speedup:  {speedup:.2f}x")

        assert result_old == result_new, "Results must be identical"
        assert speedup > 1.0, f"New version should be faster (speedup={speedup:.2f}x)"

    def test_performance_medium(self):
        """Benchmark with medium dataset (10K observations, 100 cutpoints)."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=10000)
        cutpoints = np.linspace(0, 105, 100).tolist()
        min_obs = 50

        time_old, std_old, result_old = self._benchmark(merge_cutpoints_old, cutpoints, y, min_obs)
        time_new, std_new, result_new = self._benchmark(merge_cutpoints, cutpoints, y, min_obs)

        speedup = time_old / time_new

        print(f"\n--- Medium dataset (10K obs, 100 cutpoints) ---")
        print(f"Old: {time_old*1000:.2f} ± {std_old*1000:.2f} ms")
        print(f"New: {time_new*1000:.2f} ± {std_new*1000:.2f} ms")
        print(f"Speedup:  {speedup:.2f}x")

        assert result_old == result_new, "Results must be identical"
        assert speedup > 1.0, f"New version should be faster (speedup={speedup:.2f}x)"

    def test_performance_large(self):
        """Benchmark with large dataset (100K observations, 200 cutpoints)."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=100000)
        cutpoints = np.linspace(0, 105, 200).tolist()
        min_obs = 100

        time_old, std_old, result_old = self._benchmark(merge_cutpoints_old, cutpoints, y, min_obs, n_runs=3)
        time_new, std_new, result_new = self._benchmark(merge_cutpoints, cutpoints, y, min_obs, n_runs=3)

        speedup = time_old / time_new

        print(f"\n--- Large dataset (100K obs, 200 cutpoints) ---")
        print(f"Old: {time_old*1000:.2f} ± {std_old*1000:.2f} ms")
        print(f"New: {time_new*1000:.2f} ± {std_new*1000:.2f} ms")
        print(f"Speedup:  {speedup:.2f}x")

        assert result_old == result_new, "Results must be identical"
        assert speedup > 1.0, f"New version should be faster (speedup={speedup:.2f}x)"

    def test_performance_very_large(self):
        """Benchmark with very large dataset (1M observations, 500 cutpoints)."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=1000000)
        cutpoints = np.linspace(0, 105, 500).tolist()
        min_obs = 500

        time_old, std_old, result_old = self._benchmark(merge_cutpoints_old, cutpoints, y, min_obs, n_runs=2)
        time_new, std_new, result_new = self._benchmark(merge_cutpoints, cutpoints, y, min_obs, n_runs=2)

        speedup = time_old / time_new

        print(f"\n--- Very large dataset (1M obs, 500 cutpoints) ---")
        print(f"Old: {time_old:.3f} ± {std_old:.3f} s")
        print(f"New: {time_new:.3f} ± {std_new:.3f} s")
        print(f"Speedup:  {speedup:.2f}x")

        assert result_old == result_new, "Results must be identical"
        assert speedup > 1.0, f"New version should be faster (speedup={speedup:.2f}x)"


class TestMergeCutpointsCorrectness:
    """Test that both implementations maintain required properties."""

    def test_min_obs_property(self):
        """Verify that all regions have at least min_obs observations."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=1000)
        cutpoints = np.linspace(0, 105, 50).tolist()
        min_obs = 20

        for func, name in [(merge_cutpoints_old, "old"), (merge_cutpoints, "new")]:
            result = func(cutpoints, y, min_obs)

            # Check all interior regions
            for i in range(len(result) - 1):
                count = np.sum((y >= result[i]) & (y < result[i + 1]))
                assert count >= min_obs or i == len(result) - 2, \
                    f"{name}: Region [{result[i]}, {result[i+1]}) has {count} obs, expected >= {min_obs}"

    def test_maintains_order(self):
        """Verify that cutpoints remain sorted."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=1000)
        cutpoints = np.random.uniform(0, 105, 50).tolist()
        min_obs = 10

        for func, name in [(merge_cutpoints_old, "old"), (merge_cutpoints, "new")]:
            result = func(cutpoints, y, min_obs)
            assert result == sorted(result), f"{name}: Result is not sorted: {result}"

    def test_preserves_boundaries(self):
        """Verify that first and last cutpoints are preserved."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=1000)
        cutpoints = np.linspace(-5, 110, 50).tolist()
        min_obs = 10

        for func, name in [(merge_cutpoints_old, "old"), (merge_cutpoints, "new")]:
            result = func(cutpoints, y, min_obs)
            assert result[0] == cutpoints[0], f"{name}: First cutpoint changed"
            assert result[-1] == sorted(np.unique(cutpoints))[-1], f"{name}: Last cutpoint changed"

    def test_at_least_two_cutpoints(self):
        """Verify that result always has at least 2 cutpoints."""
        np.random.seed(42)
        y = np.random.uniform(0, 100, size=10)
        cutpoints = np.linspace(0, 105, 20).tolist()
        min_obs = 100  # Impossibly high min_obs

        for func, name in [(merge_cutpoints_old, "old"), (merge_cutpoints, "new")]:
            result = func(cutpoints, y, min_obs)
            assert len(result) >= 2, f"{name}: Result has fewer than 2 cutpoints: {result}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
