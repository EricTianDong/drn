"""Tests for drn.architecture — param counting and hidden-size solvers."""

import pytest
import torch
from drn.architecture import count_params, compute_hidden_sizes
from drn.models.mdn import MDN
from drn.models.drn import DRN
from drn.models.glm import GLM


def _materialise(model, input_dim, n=2):
    """Pass fake data through a model to materialise any LazyLinear layers."""
    x = torch.randn(n, input_dim)
    model.eval()
    with torch.no_grad():
        model(x)


def _trainable_param_count(model):
    """Count trainable (requires_grad) parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------- count_params ----------

class TestCountParams:
    def test_single_layer(self):
        # input=10, hidden=[20], output=5
        # Layer 1: 10*20 + 20 = 220
        # Output:  20*5 + 5 = 105
        assert count_params(10, [20], 5) == 220 + 105

    def test_two_uniform_layers(self):
        # input=10, hidden=[20, 20], output=5
        # Layer 1: 10*20 + 20 = 220
        # Layer 2: 20*20 + 20 = 420
        # Output:  20*5 + 5 = 105
        assert count_params(10, [20, 20], 5) == 220 + 420 + 105

    def test_variable_width(self):
        # input=4, hidden=[8, 6], output=3
        # Layer 1 (input proj): 4*8 + 8 = 40
        # Layer 2 (leftover):   8*6 + 6 = 54
        # Output:               6*3 + 3 = 21
        assert count_params(4, [8, 6], 3) == 40 + 54 + 21

    def test_three_layers(self):
        # input=5, hidden=[10, 8, 4], output=2
        # Layer 1 (input proj): 5*10 + 10 = 60
        # Layer 2:              10*8 + 8 = 88
        # Layer 3:              8*4 + 4 = 36
        # Output:               4*2 + 2 = 10
        assert count_params(5, [10, 8, 4], 2) == 60 + 88 + 36 + 10

    def test_four_layers_with_leftover(self):
        # input=3, hidden=[7, 5, 4, 2], output=1
        # Layer 1 (input proj): 3*7 + 7 = 28
        # Layer 2:              7*5 + 5 = 40
        # Layer 3:              5*4 + 4 = 24
        # Layer 4:              4*2 + 2 = 10
        # Output:               2*1 + 1 = 3
        assert count_params(3, [7, 5, 4, 2], 1) == 28 + 40 + 24 + 10 + 3

    def test_empty_hidden_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            count_params(10, [], 5)


# ---------- compute_hidden_sizes — rectangular ----------

class TestRectangular:
    def test_single_layer_fits_within_budget(self):
        sizes = compute_hidden_sizes(
            total_params=500, input_dim=10, output_dim=5, num_layers=1, shape="rectangular"
        )
        assert len(sizes) == 1
        assert all(s == sizes[0] for s in sizes)
        assert count_params(10, sizes, 5) <= 500

    def test_two_layers_fits_within_budget(self):
        sizes = compute_hidden_sizes(
            total_params=2000, input_dim=10, output_dim=5, num_layers=2, shape="rectangular"
        )
        assert len(sizes) == 2
        assert sizes[0] == sizes[1]
        assert count_params(10, sizes, 5) <= 2000

    def test_three_layers_fits_within_budget(self):
        sizes = compute_hidden_sizes(
            total_params=10000, input_dim=20, output_dim=15, num_layers=3, shape="rectangular"
        )
        assert len(sizes) == 3
        assert sizes[0] == sizes[1] == sizes[2]
        assert count_params(20, sizes, 15) <= 10000

    def test_budget_is_tight(self):
        """The next larger h should exceed the budget."""
        sizes = compute_hidden_sizes(
            total_params=5000, input_dim=10, output_dim=5, num_layers=2, shape="rectangular"
        )
        h = sizes[0]
        actual = count_params(10, sizes, 5)
        over = count_params(10, [h + 1] * 2, 5)
        assert actual <= 5000
        assert over > 5000

    def test_too_small_budget_raises(self):
        with pytest.raises(ValueError):
            compute_hidden_sizes(
                total_params=5, input_dim=100, output_dim=50, num_layers=2, shape="rectangular"
            )


# ---------- compute_hidden_sizes — funnel ----------

class TestFunnel:
    def test_single_layer_same_as_rectangular(self):
        rect = compute_hidden_sizes(
            total_params=500, input_dim=10, output_dim=5, num_layers=1, shape="rectangular"
        )
        funnel = compute_hidden_sizes(
            total_params=500, input_dim=10, output_dim=5, num_layers=1, shape="funnel"
        )
        assert rect == funnel

    def test_sizes_are_decreasing(self):
        sizes = compute_hidden_sizes(
            total_params=10000, input_dim=20, output_dim=15, num_layers=3, shape="funnel"
        )
        assert len(sizes) == 3
        assert sizes[0] >= sizes[1] >= sizes[2]

    def test_last_approx_half_of_first(self):
        sizes = compute_hidden_sizes(
            total_params=10000, input_dim=20, output_dim=15, num_layers=4, shape="funnel"
        )
        assert len(sizes) == 4
        # Last should be approximately first // 2
        assert sizes[-1] == max(sizes[0] // 2, 1)

    def test_fits_within_budget(self):
        sizes = compute_hidden_sizes(
            total_params=5000, input_dim=10, output_dim=5, num_layers=3, shape="funnel"
        )
        assert count_params(10, sizes, 5) <= 5000

    def test_budget_is_tight(self):
        """Increasing h1 by 1 should exceed the budget (binary search is exact)."""
        sizes = compute_hidden_sizes(
            total_params=5000, input_dim=10, output_dim=5, num_layers=3, shape="funnel"
        )
        actual = count_params(10, sizes, 5)
        assert actual <= 5000
        # Verify we can't go bigger — construct sizes with h1+1
        h1_bigger = sizes[0] + 1
        h_last = max(h1_bigger // 2, 1)
        bigger_sizes = [
            max(round(h1_bigger + i / 2 * (h_last - h1_bigger)), 1) for i in range(3)
        ]
        assert count_params(10, bigger_sizes, 5) > 5000


# ---------- compute_hidden_sizes — dispatcher ----------

class TestDispatcher:
    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            compute_hidden_sizes(
                total_params=1000, input_dim=10, output_dim=5, num_layers=2, shape="pyramid"
            )

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            compute_hidden_sizes(
                total_params=1000, input_dim=10, output_dim=5, num_layers=0
            )

    def test_negative_params_raises(self):
        with pytest.raises(ValueError, match="total_params"):
            compute_hidden_sizes(
                total_params=-1, input_dim=10, output_dim=5, num_layers=2
            )


# ---------- auto layer selection (num_layers=None) ----------

class TestAutoLayerSelection:
    def test_last_hidden_ge_output_dim(self):
        """Auto-selected architecture must have last hidden >= output_dim."""
        for total_params in [1000, 5000, 10000, 50000]:
            sizes = compute_hidden_sizes(
                total_params=total_params, input_dim=20, output_dim=15
            )
            assert sizes[-1] >= 15, f"last hidden {sizes[-1]} < output_dim 15 for {total_params} params"

    def test_prefers_deeper(self):
        """With more params, should be able to afford more layers."""
        small = compute_hidden_sizes(total_params=500, input_dim=10, output_dim=5)
        large = compute_hidden_sizes(total_params=50000, input_dim=10, output_dim=5)
        assert len(large) >= len(small)

    def test_adding_one_more_layer_would_violate_constraint(self):
        """The next deeper architecture should have last_hidden < output_dim."""
        output_dim = 15
        sizes = compute_hidden_sizes(
            total_params=5000, input_dim=20, output_dim=output_dim
        )
        L = len(sizes)
        if L < 5:  # only testable if we didn't hit the cap
            try:
                deeper = compute_hidden_sizes(
                    total_params=5000, input_dim=20, output_dim=output_dim, num_layers=L + 1
                )
                assert deeper[-1] < output_dim
            except ValueError:
                pass  # also fine — budget too small

    def test_within_budget(self):
        sizes = compute_hidden_sizes(total_params=10000, input_dim=20, output_dim=15)
        assert count_params(20, sizes, 15) <= 10000

    def test_large_input_dim_fewer_layers(self):
        """With many features and modest budget, should pick fewer layers."""
        sizes = compute_hidden_sizes(total_params=5000, input_dim=300, output_dim=15)
        # Most params consumed by input→first hidden, so few layers expected
        assert len(sizes) <= 3

    def test_small_budget_raises(self):
        """Budget too small for even 1 layer with last_hidden >= output_dim."""
        with pytest.raises(ValueError, match="too small"):
            compute_hidden_sizes(total_params=10, input_dim=100, output_dim=50)

    def test_rectangular_auto(self):
        """Auto-selection also works with rectangular shape."""
        sizes = compute_hidden_sizes(
            total_params=10000, input_dim=10, output_dim=5, shape="rectangular"
        )
        assert all(s == sizes[0] for s in sizes)
        assert sizes[-1] >= 5
        assert count_params(10, sizes, 5) <= 10000

    def test_funnel_auto_decreasing(self):
        """Auto-selected funnel should have non-increasing layer widths."""
        sizes = compute_hidden_sizes(
            total_params=10000, input_dim=20, output_dim=15, shape="funnel"
        )
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1]


# ---------- count_params vs actual MDN models ----------

class TestCountParamsMatchesMDN:
    """Build real MDN models, materialise lazy layers, and verify count_params agrees."""

    @pytest.mark.parametrize("hidden_size, num_components, distribution, input_dim", [
        ([100, 100], 5, "gamma", 10),
        ([64], 3, "gamma", 4),
        ([128, 64, 32], 4, "gaussian", 20),
        ([50, 50], 2, "gaussian", 8),
        ([200], 7, "gamma", 15),
        ([80, 60, 40, 20], 3, "gamma", 6),
    ])
    def test_list_hidden_size(self, hidden_size, num_components, distribution, input_dim):
        mdn = MDN(
            hidden_size=hidden_size,
            num_components=num_components,
            distribution=distribution,
        )
        _materialise(mdn, input_dim)

        output_dim = 3 * num_components  # logits + 2 params per component
        expected = count_params(input_dim, hidden_size, output_dim)
        assert _trainable_param_count(mdn) == expected

    @pytest.mark.parametrize("hidden_size, num_layers, num_components, distribution, input_dim", [
        (100, 2, 5, "gamma", 10),
        (75, 3, 3, "gaussian", 4),
        (50, 1, 2, "gamma", 20),
    ])
    def test_uniform_layers(self, hidden_size, num_layers, num_components, distribution, input_dim):
        """Verify the legacy hidden_size + num_hidden_layers path also matches."""
        mdn = MDN(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_components=num_components,
            distribution=distribution,
        )
        _materialise(mdn, input_dim)

        sizes = [hidden_size] * num_layers
        output_dim = 3 * num_components
        expected = count_params(input_dim, sizes, output_dim)
        assert _trainable_param_count(mdn) == expected


# ---------- count_params vs actual DRN models ----------

class TestCountParamsMatchesDRN:
    """Build real DRN models, materialise lazy layers, and verify count_params agrees."""

    @staticmethod
    def _make_baseline(input_dim):
        """Create a materialised GLM baseline for DRN."""
        glm = GLM("gamma")
        # Materialise the GLM's lazy layer
        x = torch.randn(2, input_dim)
        with torch.no_grad():
            glm(x)
        # DRN needs dispersion set for predict; just set a dummy value
        glm.dispersion = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        return glm

    @pytest.mark.parametrize("hidden_size, cutpoints, input_dim", [
        ([75, 75], [0.0, 1.0, 2.0, 3.0], 10),
        ([64], [0.0, 0.5, 1.0, 1.5, 2.0], 4),
        ([128, 64, 32], [0.0, 1.0, 2.0], 20),
        ([50, 50], [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], 8),
        ([100, 80, 60, 40], [0.0, 1.0, 2.0, 3.0, 4.0], 6),
    ])
    def test_list_hidden_size(self, hidden_size, cutpoints, input_dim):
        baseline = self._make_baseline(input_dim)
        drn = DRN(
            baseline=baseline,
            cutpoints=cutpoints,
            hidden_size=hidden_size,
        )
        _materialise(drn, input_dim)

        output_dim = len(cutpoints) - 1
        expected = count_params(input_dim, hidden_size, output_dim)
        assert _trainable_param_count(drn) == expected

    @pytest.mark.parametrize("hidden_size, num_layers, cutpoints, input_dim", [
        (75, 2, [0.0, 1.0, 2.0, 3.0], 10),
        (100, 1, [0.0, 0.5, 1.0, 1.5, 2.0], 4),
        (50, 3, [0.0, 1.0, 2.0], 20),
    ])
    def test_uniform_layers(self, hidden_size, num_layers, cutpoints, input_dim):
        """Verify the legacy hidden_size + num_hidden_layers path also matches."""
        baseline = self._make_baseline(input_dim)
        drn = DRN(
            baseline=baseline,
            cutpoints=cutpoints,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
        )
        _materialise(drn, input_dim)

        sizes = [hidden_size] * num_layers
        output_dim = len(cutpoints) - 1
        expected = count_params(input_dim, sizes, output_dim)
        assert _trainable_param_count(drn) == expected
