"""Regression test: DRN.quantiles must respect the model's own support.

When a DRN is built on a GLM baseline supported on [0, inf) with cutpoints
starting at 0.0, calling ``quantiles`` with the default lower bound ``l``
should not push negative values through the baseline distribution's cdf.
Previously the default ``l`` was derived from the data spread and could go
negative, raising a ValueError from the Gamma baseline. Passing ``l=0.0``
explicitly was the workaround.
"""

import numpy as np
import pandas as pd
import torch

import drn


def _build_model():
    X = pd.DataFrame({"x": np.linspace(0.1, 1.0, 64)})
    y = pd.Series(np.linspace(0.5, 5.0, 64))

    glm = drn.GLM("gamma").fit(X, y)  # baseline supported on [0, inf)
    cutpoints = drn.default_drn_cutpoints(y, 0.1, 1)  # cutpoints[0] == 0.0
    assert cutpoints[0] == 0.0
    model = drn.DRN(glm, cutpoints=cutpoints).eval()  # no DRN training needed
    return model, X


def test_drn_quantiles_default_bounds_do_not_raise():
    """quantiles() with default bounds must not raise on a [0, inf) baseline."""
    model, X = _build_model()

    # Workaround (explicit l=0.0) is known to work.
    workaround = model.quantiles(X, [90.0], l=0.0)

    # Default call should behave the same, not raise a support ValueError.
    default = model.quantiles(X, [90.0])

    assert torch.allclose(default, workaround, atol=1e-4)
