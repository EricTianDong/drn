# Quick Start Guide

This guide shows how to use DRN with pandas and numpy. The library handles all tensor conversions internally, so you can work with familiar data structures.

## Installation

Install DRN using pip:

```bash
pip install drn
```

DRN requires Python 3.8+ and will automatically install PyTorch, pandas, numpy, scikit-learn, and other dependencies.

## 1. Basic Example

```python
from drn import GLM, DRN
import pandas as pd
import numpy as np

# Generate sample data (pandas/numpy)
np.random.seed(42)
n_samples = 50

X_train = pd.DataFrame({
    'age': np.random.randint(20, 70, n_samples),
    'income': np.random.randint(20000, 100000, n_samples)
})
# Generate target with some relationship to features
y_train = pd.Series(
    800 + 20 * (X_train['age'] - 40) + 0.01 * X_train['income'] + np.random.normal(0, 200, n_samples)
)
# Ensure positive values (e.g., for insurance claims or sales)
y_train = y_train.clip(lower=100)

# 1. Train baseline model (GLM handles data conversion internally)
baseline = GLM('gamma')  # Good for positive targets like insurance claims
baseline.fit(X_train, y_train)

# 2. Train DRN (automatically handles data conversion)
drn_model = DRN(baseline)
drn_model.fit(X_train, y_train)

# 3. Make predictions (returns distribution objects)
predictions = drn_model.predict(X_train[:5])  # Predict first 5 samples
mean_predictions = predictions.mean
quantiles = predictions.quantiles([10, 50, 90])

print(f"Mean predictions: {mean_predictions}")
print(f"Quantiles (10th, 50th, 90th): {quantiles}")
```

## 2. Complete Example with Realistic Data

Let's create a more complete example using tested patterns from the codebase:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from drn import GLM, DRN
from drn.metrics import rmse

def generate_realistic_data(n=1000, seed=42):
    """Generate data similar to the tested synthetic dataset."""
    np.random.seed(seed)
    
    # Create features (based on actual test patterns)
    X = np.random.random(size=(n, 4))
    
    # Create complex target relationship (from test_fit_models_synthetic.py)
    means = np.exp(
        0
        - 0.5 * X[:, 0]
        + 0.5 * X[:, 1]
        + np.sin(np.pi * X[:, 0])
        - np.sin(np.pi * np.log(X[:, 2] + 1))
        + np.cos(X[:, 1] * X[:, 2])
    ) + np.cos(X[:, 1])
    
    # Add noise
    epsilon = np.random.normal(0, 0.2, n)
    y = means + epsilon**2
    
    # Convert to pandas
    X_df = pd.DataFrame(X, columns=['feature_0', 'feature_1', 'feature_2', 'feature_3'])
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

# Generate synthetic data
X, y = generate_realistic_data(n=1000)
print(f"Dataset shape: {X.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
```

## 3. Split Data (Simple Approach)

```python
# Simple train/test split (pandas stays as pandas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

## 4. Train Baseline GLM

```python
# Train baseline GLM (uses pandas/numpy internally)
baseline = GLM('gamma')  # Gamma distribution for positive targets
baseline.fit(X_train, y_train)

# Check baseline performance
baseline_pred = baseline.predict(X_test)
baseline_rmse = rmse(y_test, baseline_pred.mean)
print(f"Baseline RMSE: {baseline_rmse:.4f}")
```

## 5. Train DRN

```python
# Create DRN (automatically determines cutpoints)
drn_model = DRN(baseline)
drn_model.fit(X_train, y_train)

# Check DRN performance
drn_pred = drn_model.predict(X_test)
drn_rmse = rmse(y_test, drn_pred.mean)

print(f"DRN RMSE: {drn_rmse:.4f}")
print(f"Improvement: {((baseline_rmse - drn_rmse) / baseline_rmse * 100):.1f}%")
```

## 6. Explore Predictions

```python
# Get distributional properties
single_prediction = drn_model.predict(X_test.iloc[:1])

print(f"Mean prediction: {single_prediction.mean.item():.3f}")
print(f"True value: {y_test.iloc[0]:.3f}")

# Get risk measures
risk_quantiles = drn_model.quantiles(X_test.iloc[:1], [5, 25, 50, 75, 95])
print(f"5th-95th percentile range: [{risk_quantiles[0][0]:.3f}, {risk_quantiles[0][-1]:.3f}]")

# Compare multiple models
models = {'Baseline': baseline, 'DRN': drn_model}
for name, model in models.items():
    pred = model.predict(X_test)
    test_rmse = rmse(y_test, pred.mean)
    print(f"{name} RMSE: {test_rmse:.4f}")
```

## 7. Working with Different Data Types

### Mixed Data Types (Pandas)
```python
# DRN handles mixed data automatically
mixed_data = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [30000, 50000, 70000, 90000],
    'region': ['North', 'South', 'East', 'West'],  # Categorical
    'policy_type': ['Basic', 'Premium', 'Basic', 'Premium']  # Categorical
})

# DRN will handle categorical encoding internally
glm_mixed = GLM('gamma')
# Note: For categorical data, you may want to use preprocessing utilities
```

### NumPy Arrays
```python
# DRN also works with pure numpy
X_numpy = X_train.values
y_numpy = y_train.values

glm_numpy = GLM('gamma')
glm_numpy.fit(X_numpy, y_numpy)
```

