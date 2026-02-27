import pytest
import numpy as np
import pandas as pd
import torch

from drn import DeepGLM
from drn.distributions import inverse_gaussian


def test_gamma_forward_positive():
    """Test that gamma distribution forward pass produces positive values."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 5)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=32)).astype(np.float32)

    m = DeepGLM(distribution='gamma', num_hidden_layers=2, hidden_size=16, dropout_rate=0.0, learning_rate=1e-2)
    m.fit(X, y, epochs=2, batch_size=16)
    mu = m(torch.tensor(X)).detach().numpy()
    assert np.all(mu > 0), 'Gamma means must be positive.'


def test_predict_returns_distribution():
    """Test that predict returns a valid distribution object."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=8)).astype(np.float32)

    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8, dropout_rate=0.0, learning_rate=1e-2)
    m.fit(X, y, epochs=2, batch_size=8)
    d = m.predict(X)
    assert hasattr(d, "sample"), "Prediction should be a torch.distributions object."
    assert d.batch_shape[0] == X.shape[0]


def test_invalid_distribution_raises():
    """Test that invalid distribution raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported distribution"):
        DeepGLM(distribution="invalid")


def test_dispersion_not_estimated_error():
    """Test that predict fails if dispersion not estimated."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma')
    # Skip fit to avoid dispersion estimation
    with pytest.raises(RuntimeError, match="Dispersion parameter has not been estimated"):
        m.predict(X)


def test_gamma_distribution():
    """Test gamma distribution functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=len(X))).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X, y, epochs=2)
    
    # Test forward pass produces positive values
    pred_means = m(torch.tensor(X))
    assert torch.all(pred_means > 0)
    
    # Test prediction returns Gamma distribution
    dist = m.predict(X)
    assert isinstance(dist, torch.distributions.Gamma)
    assert dist.batch_shape[0] == len(X)


def test_gaussian_distribution():
    """Test gaussian distribution functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=len(X))).astype(np.float32)
    
    m = DeepGLM(distribution='gaussian', num_hidden_layers=1, hidden_size=8)
    m.fit(X, y, epochs=2)
    
    # Test prediction returns Normal distribution
    dist = m.predict(X)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.batch_shape[0] == len(X)


def test_inversegaussian_distribution():
    """Test inverse gaussian distribution functionality."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=len(X))).astype(np.float32)
    
    m = DeepGLM(distribution='inversegaussian', num_hidden_layers=1, hidden_size=8)
    m.fit(X, y, epochs=2)
    
    # Test forward pass produces positive values
    pred_means = m(torch.tensor(X))
    assert torch.all(pred_means > 0)
    
    # Test prediction returns InverseGaussian distribution
    dist = m.predict(X)
    assert isinstance(dist, inverse_gaussian.InverseGaussian)
    assert dist.batch_shape[0] == len(X)


def test_lognormal_distribution():
    """Test lognormal distribution functionality."""
    # Skip lognormal for now - not currently used
    pytest.skip("Lognormal distribution not currently supported")


def test_fit_updates_dispersion():
    """Test that fit properly updates dispersion parameter."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=50)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    
    # Before fit, dispersion should be NaN
    assert torch.isnan(m.dispersion)
    
    m.fit(X, y, epochs=2)
    
    # After fit, dispersion should be estimated
    assert not torch.isnan(m.dispersion)
    assert m.dispersion.item() > 0


def test_fit_with_validation():
    """Test fitting with validation data."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(40, 3)).astype(np.float32)
    y_train = np.exp(X_train[:, 0] + 0.1 * rng.normal(size=40)).astype(np.float32)
    X_val = rng.normal(size=(10, 3)).astype(np.float32)
    y_val = np.exp(X_val[:, 0] + 0.1 * rng.normal(size=10)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X_train, y_train, X_val, y_val, epochs=2)
    
    # Should complete without error
    assert not torch.isnan(m.dispersion)


def test_loss_decreases_with_training():
    """Test that loss decreases during training."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=100)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=16, learning_rate=1e-2)
    
    # Compute initial loss
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    initial_loss = m.loss(X_tensor, y_tensor).item()
    
    # Train model
    m.fit(X, y, epochs=10)
    
    # Loss should decrease
    final_loss = m.loss(X_tensor, y_tensor).item()
    assert final_loss < initial_loss


def test_numpy_input():
    """Test with numpy array inputs."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=20)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X, y, epochs=2)
    
    pred = m.predict(X)
    assert pred.batch_shape[0] == len(X)


def test_pandas_input():
    """Test with pandas DataFrame and Series inputs."""
    rng = np.random.default_rng(42)
    X_np = rng.normal(size=(20, 3)).astype(np.float32)
    y_np = np.exp(X_np[:, 0] + 0.1 * rng.normal(size=20)).astype(np.float32)
    
    X_df = pd.DataFrame(X_np, columns=['feat1', 'feat2', 'feat3'])
    y_series = pd.Series(y_np, name='target')
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X_df, y_series, epochs=2)
    
    pred = m.predict(X_df)
    assert pred.batch_shape[0] == len(X_df)


def test_torch_tensor_input():
    """Test with torch tensor inputs."""
    rng = np.random.default_rng(42)
    X_np = rng.normal(size=(20, 3)).astype(np.float32)
    X = torch.tensor(X_np)
    y_np = np.exp(X_np[:, 0] + 0.1 * rng.normal(size=20)).astype(np.float32)
    y = torch.tensor(y_np)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X, y, epochs=2)
    
    pred = m.predict(X)
    assert pred.batch_shape[0] == len(X)


def test_mean_method():
    """Test the mean method with different input types."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=10)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X, y, epochs=2)
    
    # Test numpy input
    means_np = m.mean(X)
    assert isinstance(means_np, np.ndarray)
    assert means_np.shape == (10,)
    
    # Test torch input
    means_torch = m.mean(torch.tensor(X))
    assert isinstance(means_torch, np.ndarray)
    assert np.allclose(means_np, means_torch)


def test_clone_method():
    """Test the clone method."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=20)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=2, hidden_size=16, dropout_rate=0.2, learning_rate=1e-3)
    m.fit(X, y, epochs=2)
    
    # Clone the model
    m_clone = m.clone()
    
    # Should be different objects
    assert m is not m_clone
    
    # Should have same hyperparameters
    assert m_clone.distribution == m.distribution
    assert m_clone.learning_rate == m.learning_rate
    
    # Should have same state dict
    original_state = m.state_dict()
    cloned_state = m_clone.state_dict()
    assert original_state.keys() == cloned_state.keys()
    for key in original_state.keys():
        assert torch.allclose(original_state[key], cloned_state[key])


def test_hyperparameters_saved():
    """Test that hyperparameters are properly saved."""
    m = DeepGLM(
        distribution='inversegaussian',
        num_hidden_layers=3,
        hidden_size=64,
        dropout_rate=0.3,
        learning_rate=5e-4
    )
    
    # Should be able to access hyperparameters
    assert hasattr(m, 'hparams')
    assert m.hparams.distribution == 'inversegaussian'
    assert m.hparams.num_hidden_layers == 3
    assert m.hparams.hidden_size == 64
    assert m.hparams.dropout_rate == 0.3
    assert m.hparams.learning_rate == 5e-4


def test_small_dataset():
    """Test with small dataset."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(5, 3)).astype(np.float32)  # Use 5 samples minimum
    y = np.exp(X[:, 0] + 0.1 * rng.normal(size=5)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X, y, epochs=2, batch_size=5)  # Use batch size that matches data size
    
    pred = m.predict(X)
    # Check that prediction works and has correct batch shape
    assert hasattr(pred, 'mean')
    assert pred.batch_shape[0] == 5


def test_different_batch_sizes():
    """Test training and prediction with different batch sizes."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(50, 3)).astype(np.float32)
    y_train = np.exp(X_train[:, 0] + 0.1 * rng.normal(size=50)).astype(np.float32)
    
    X_test = rng.normal(size=(30, 3)).astype(np.float32)
    
    m = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    m.fit(X_train, y_train, epochs=2, batch_size=16)
    
    pred = m.predict(X_test)
    assert pred.batch_shape[0] == 30


def test_network_construction():
    """Test that network is constructed properly with different layer configurations."""
    from drn.models.layers import ResidualBlock

    # Test with 1 hidden layer: input projection only, no residual blocks
    m1 = DeepGLM(distribution='gamma', num_hidden_layers=1, hidden_size=8)
    assert len([l for l in m1.hidden_layers if isinstance(l, torch.nn.Linear)]) == 1
    assert len([l for l in m1.hidden_layers if isinstance(l, ResidualBlock)]) == 0

    # Test with 3 hidden layers: input projection + one 2-layer residual block
    m3 = DeepGLM(distribution='gamma', num_hidden_layers=3, hidden_size=16)
    assert len([l for l in m3.hidden_layers if isinstance(l, torch.nn.Linear)]) == 1
    assert len([l for l in m3.hidden_layers if isinstance(l, ResidualBlock)]) == 1

    # Test with 4 hidden layers: one residual block + one leftover plain layer
    m4 = DeepGLM(distribution='gamma', num_hidden_layers=4, hidden_size=16)
    assert len([l for l in m4.hidden_layers if isinstance(l, torch.nn.Linear)]) == 2
    assert len([l for l in m4.hidden_layers if isinstance(l, ResidualBlock)]) == 1
