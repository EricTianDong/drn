import copy
import numpy as np
import torch
import pytest

# Import your package API
from drn import GLM, CANN


def _rng_data(n=256, p=3, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    # Positive target for gamma/log-like models
    y = (np.exp(0.6 * X[:, 0] - 0.25 * X[:, 1]) + 0.05 * rng.normal(size=n)).astype(np.float32)
    return X, y


def _params_vec(module: torch.nn.Module) -> torch.Tensor:
    """Flatten all parameters to a single detached CPU tensor."""
    if not any(True for _ in module.parameters()):
        # If baseline has no Parameters, this test isn't meaningful.
        raise AssertionError("Baseline GLM has no trainable parameters.")
    return torch.nn.utils.parameters_to_vector(
        [p.detach().cpu().clone() for p in module.parameters()]
    )
    

def _preds(module, X_np, take_first=64):
    """Get stable predictions in eval mode on a small slice."""
    module.eval()
    with torch.no_grad():
        X = torch.from_numpy(X_np[:take_first])
        return module(X).detach().cpu().clone()


def test_cann_train_glm_freezes_baseline_when_flag_false():
    """Baseline coefficients and predictions must stay unchanged when train_glm=False."""
    torch.manual_seed(7)
    X, y = _rng_data(n=128, p=3, seed=7)

    # Fit a baseline GLM once
    baseline_fitted = GLM("gamma").fit(X, y)

    # Work on a deep copy so the original isn't mutated
    base0 = copy.deepcopy(baseline_fitted)

    # Snapshot parameters & predictions BEFORE training the CANN
    before_params = _params_vec(base0)
    before_preds = _preds(base0, X)

    # Train a small CANN that must NOT update the baseline
    cann_no = CANN(
        baseline=base0,
        hidden_size=4,
        num_hidden_layers=2,
        dropout_rate=0.1,
        learning_rate=1e-2,
        train_glm=False,
    ).fit(X, y, epochs=2, batch_size=32)

    # After training, baseline params & preds should be identical
    after_params = _params_vec(cann_no.baseline)
    after_preds = _preds(cann_no.baseline, X)

    assert torch.allclose(before_params, after_params), \
        "Baseline GLM parameters changed despite train_glm=False."
    assert torch.allclose(before_preds, after_preds), \
        "Baseline GLM predictions changed despite train_glm=False."

    # Also check requires_grad toggling for clarity
    assert all(not p.requires_grad for p in cann_no.baseline.parameters()), \
        "Baseline params should have requires_grad=False when train_glm=False."


def test_cann_train_glm_updates_baseline_when_flag_true():
    """Baseline coefficients must update when train_glm=True."""
    torch.manual_seed(42)
    X, y = _rng_data(n=256, p=3, seed=42)

    # Fresh baseline starting from the same fitted state
    baseline_fitted = GLM("gamma").fit(X, y)
    base1 = copy.deepcopy(baseline_fitted)

    before_params = _params_vec(base1)
    before_preds = _preds(base1, X)

    cann_yes = CANN(
        baseline=base1,
        hidden_size=4,
        num_hidden_layers=2,
        dropout_rate=0.1,
        learning_rate=1e-2,
        train_glm=True,
    ).fit(X, y, epochs=2, batch_size=32)

    after_params = _params_vec(cann_yes.baseline)
    after_preds = _preds(cann_yes.baseline, X)

    assert not torch.allclose(before_params, after_params), \
        "Baseline GLM parameters did not change with train_glm=True."
    # Usually predictions will change too; allow tiny tolerance, but expect a difference
    assert not torch.allclose(before_preds, after_preds, rtol=1e-7, atol=1e-8), \
        "Baseline GLM predictions did not change with train_glm=True."

    # requires_grad should be True for at least one baseline param
    assert any(p.requires_grad for p in cann_yes.baseline.parameters()), \
        "Baseline params should have requires_grad=True when train_glm=True."
