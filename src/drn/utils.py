from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch


def split_and_preprocess(
    features, target, num_features, cat_features, seed=42, num_standard=True
):
    # Before preprocessing split
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features, target, random_state=seed, train_size=0.8, shuffle=True
    )

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_raw, y_train, random_state=seed, train_size=0.75, shuffle=True
    )

    # Determine the full set of categories for each feature
    all_categories = {feature: set() for feature in cat_features}
    for feature in cat_features:
        all_categories[feature].update(features[feature].unique())

    # Convert each categorical feature to a categorical type with all possible categories
    for feature in cat_features:
        # Sort the categories to ensure consistent order
        sorted_categories = sorted(all_categories[feature])
        x_train_raw[feature] = pd.Categorical(
            x_train_raw[feature], categories=sorted_categories
        )
        x_val_raw[feature] = pd.Categorical(
            x_val_raw[feature], categories=sorted_categories
        )
        x_test_raw[feature] = pd.Categorical(
            x_test_raw[feature], categories=sorted_categories
        )
        features[feature] = pd.Categorical(
            features[feature], categories=sorted_categories
        )

    # One-hot Encoding
    features_one_hot = pd.get_dummies(features, columns=cat_features, drop_first=True)
    features_one_hot = features_one_hot.astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        features_one_hot, target, random_state=seed, train_size=0.8, shuffle=True
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, random_state=seed, train_size=0.75, shuffle=True
    )

    if num_standard:
        # Standarise the numeric features.
        ct = ColumnTransformer(
            [("standardize", StandardScaler(), num_features)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        x_train = ct.fit_transform(x_train)
        x_val = ct.transform(x_val)
        x_test = ct.transform(x_test)
    else:
        ct = None

    x_train = pd.DataFrame(
        x_train, index=x_train_raw.index, columns=features_one_hot.columns
    )
    x_val = pd.DataFrame(x_val, index=x_val_raw.index, columns=features_one_hot.columns)
    x_test = pd.DataFrame(
        x_test, index=x_test_raw.index, columns=features_one_hot.columns
    )

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        x_train_raw,
        x_val_raw,
        x_test_raw,
        num_features,
        cat_features,
        all_categories,
        ct,
    )


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int = 42,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split features and target into train, validation, and test sets based on fractions of the entire dataset.

    Args:
        features: DataFrame of predictors.
        target: Series of labels.
        seed: Random seed for reproducibility.
        train_size: Fraction of data for training.
        val_size: Fraction of data for validation.
            (test_size is computed as 1 - train_size - val_size)
    Returns:
        x_train_raw, x_val_raw, x_test_raw,
        y_train, y_val, y_test
    """
    # Compute test fraction
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError(
            f"train_size + val_size must be < 1. Got {train_size + val_size}"
        )

    # Split off test set
    x_train_val, x_test_raw, y_train_val, y_test = train_test_split(
        features, target, test_size=test_size, random_state=seed, shuffle=True
    )
    # Split train+val into train and val
    relative_val_size = val_size / (train_size + val_size)
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )
    return x_train_raw, x_val_raw, x_test_raw, y_train, y_val, y_test


def replace_rare_categories(
    df: pd.DataFrame,
    threshold: int | float = 10,
    placeholder: str = "OTHER",
    cat_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Replace rare categories in specified categorical columns with a placeholder category.

    Parameters:
    - df: The input DataFrame.
    - threshold: Minimum number of occurrences for a category to be kept.
               If >= 1: treated as absolute count (e.g., 10 means at least 10 occurrences).
               If < 1: treated as percentage of total rows (e.g., 0.01 means at least 1%).
    - placeholder: Name to assign to rare categories.
    - cat_features: If specified, only apply to these columns.

    Raises:
    - ValueError: If the placeholder value already exists in any of the target columns,
                 or if threshold is invalid (negative or zero).

    Returns:
    - pd.DataFrame: A new DataFrame with rare categories replaced.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    df = df.copy()
    columns = (
        cat_features
        if cat_features is not None
        else df.select_dtypes(include=["object", "category"]).columns
    )

    # Check for placeholder conflicts
    for col in columns:
        if placeholder in df[col].unique():
            raise ValueError(
                f"The placeholder value '{placeholder}' already exists in column '{col}'."
            )

    # Convert percentage threshold to absolute count if needed
    if threshold < 1:
        # Threshold is a percentage
        absolute_threshold = int(np.ceil(threshold * len(df)))
    else:
        # Threshold is already an absolute count
        absolute_threshold = int(threshold)

    for col in columns:
        value_counts = df[col].value_counts()
        rare_categories = value_counts[value_counts < absolute_threshold].index
        df[col] = df[col].apply(lambda x: placeholder if x in rare_categories else x)
        if df[col].dtype.name != "category":
            df[col] = df[col].astype("category")
    return df


def generate_categories(
    x_train_raw: pd.DataFrame,
    x_val_raw: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_features: list[str],
) -> dict[str, list]:
    """
    Create a mapping of categorical features to their full category lists:
      - Initialize from training split
      - Detect new categories in val/test, print a warning, and extend
    Returns:
        all_categories: feature -> sorted list of categories
    """
    all_categories = {
        feature: sorted(x_train_raw[feature].dropna().unique())
        for feature in cat_features
    }
    for split_name, df in [("validation", x_val_raw), ("test", x_test)]:
        for feature in cat_features:
            seen = set(all_categories[feature])
            unique_vals = set(df[feature].dropna().unique())
            new_vals = unique_vals - seen
            if new_vals:
                print(
                    f"New categories for '{feature}' in {split_name} split: {new_vals}"
                )
                all_categories[feature] = sorted(
                    all_categories[feature] + list(new_vals)
                )
    return all_categories


def preprocess_data(
    x_train_raw: pd.DataFrame,
    x_val_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    num_standard: bool = True,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer, dict[str, list]
]:
    """
    Fit a ColumnTransformer on x_train_raw and transform raw splits.
    - Numeric features are optionally standardized.
    - Categorical features are one-hot encoded, using full categories detected from splits.

    Returns:
        x_train, x_val, x_test, fitted ColumnTransformer, all_categories mapping
    """
    # Prepare category mapping
    all_categories = generate_categories(
        x_train_raw, x_val_raw, x_test_raw, cat_features
    )

    # OneHotEncoder with fixed categories
    ohe = OneHotEncoder(
        categories=[all_categories[f] for f in cat_features],
        handle_unknown="error",
        sparse_output=False,
        drop="first",
    )

    # Build transformers list
    transformers = [
        ("num", StandardScaler() if num_standard else "passthrough", num_features),
        ("cat", ohe, cat_features),
    ]
    ct = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    # Fit & transform splits
    x_train_arr = ct.fit_transform(x_train_raw)
    x_val_arr = ct.transform(x_val_raw)
    x_test_arr = ct.transform(x_test_raw)

    # Build DataFrames with proper feature names
    feature_names = ct.get_feature_names_out()
    x_train = pd.DataFrame(x_train_arr, columns=feature_names, index=x_train_raw.index)
    x_val = pd.DataFrame(x_val_arr, columns=feature_names, index=x_val_raw.index)
    x_test = pd.DataFrame(x_test_arr, columns=feature_names, index=x_test_raw.index)

    return x_train, x_val, x_test, ct, all_categories


def _to_numpy(data):
    """Convert input data to numpy array with float32 precision."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values.astype(np.float32)
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    else:
        return np.asarray(data, dtype=np.float32)


def binary_search_icdf(
    distribution: torch.distributions.Distribution, p: float, l=None, u=None, max_iter=1000, tolerance=1e-7
) -> torch.Tensor:
    """
    Generic binary search implementation for inverse CDF (quantiles).

    This function can be used by any distribution that has a `cdf` method
    but doesn't have its own `icdf` implementation.

    Args:
        distribution: Distribution object with a `cdf` method
        p: cumulative probability value at which to evaluate icdf
        l: lower bound for the quantile search
        u: upper bound for the quantile search
        max_iter: maximum number of iterations
        tolerance: stopping criteria for convergence

    Returns:
        A tensor of shape (1, batch_shape) containing the inverse CDF values.
    """
    # Get batch size by doing a dummy CDF call
    if hasattr(distribution, "batch_shape"):
        num_observations = distribution.batch_shape[0]
    else:
        # Fallback: try to infer from a dummy CDF call
        dummy_val = torch.tensor([1.0])
        try:
            dummy_cdf = distribution.cdf(
                dummy_val.unsqueeze(-1)
                if hasattr(distribution, "cutpoints")
                else dummy_val
            )
            num_observations = dummy_cdf.shape[-1] if dummy_cdf.ndim > 0 else 1
        except:
            num_observations = 1

    percentiles_tensor = torch.full(
        (1, num_observations), fill_value=p, dtype=torch.float32
    )

    # Initialize bounds with distribution-aware defaults
    if l is None:
        if hasattr(distribution, "cutpoints"):
            # For histogram-like distributions
            l = distribution.cutpoints[0] - (
                distribution.cutpoints[-1] - distribution.cutpoints[0]
            )
        else:
            # Try to get reasonable bounds from the distribution
            try:
                # Use distribution support if available
                if hasattr(distribution, "support"):
                    support = distribution.support
                    if hasattr(support, "lower_bound"):
                        l = support.lower_bound
                    else:
                        l = torch.tensor(
                            -10.0
                        )  # Default for unbounded below (e.g., Gaussian)
                else:
                    # Try to infer from distribution type
                    if hasattr(distribution, "concentration") and hasattr(
                        distribution, "rate"
                    ):
                        # Gamma-like distribution, should be positive
                        l = torch.tensor(0.001)
                    else:
                        # Assume could be negative (e.g., Gaussian)
                        l = torch.tensor(-10.0)
            except:
                l = torch.tensor(0.0)  # Conservative fallback
    else:
        l = torch.tensor(l) if not isinstance(l, torch.Tensor) else l

    if u is None:
        if hasattr(distribution, "cutpoints"):
            # For histogram-like distributions
            u = distribution.cutpoints[-1] + (
                distribution.cutpoints[-1] - distribution.cutpoints[0]
            )
        else:
            # Try to get reasonable bounds from the distribution
            try:
                # Use distribution support if available
                if hasattr(distribution, "support"):
                    support = distribution.support
                    if hasattr(support, "upper_bound"):
                        u = support.upper_bound
                    else:
                        u = torch.tensor(10.0)  # Default for unbounded above
                else:
                    # Conservative upper bound
                    u = torch.tensor(10.0)
            except:
                u = torch.tensor(200.0)  # Conservative fallback
    else:
        u = torch.tensor(u) if not isinstance(u, torch.Tensor) else u

    # Adaptive bounds: if the CDF at bounds doesn't bracket the target percentile,
    # expand the bounds
    try:
        cdf_lower = distribution.cdf(l.repeat(num_observations))
        cdf_upper = distribution.cdf(u.repeat(num_observations))

        # If p is outside [cdf_lower, cdf_upper], expand bounds
        if torch.any(p < cdf_lower):
            # Need to expand lower bound
            l = l - torch.abs(l) - 1.0
        if torch.any(p > cdf_upper):
            # Need to expand upper bound
            u = u + torch.abs(u) + 1.0
    except:
        # If CDF evaluation fails, stick with original bounds
        pass

    # Ensure l and u are tensors
    l = torch.tensor(l) if not isinstance(l, torch.Tensor) else l
    u = torch.tensor(u) if not isinstance(u, torch.Tensor) else u

    lower_bounds = l.repeat(num_observations).reshape(1, num_observations)
    upper_bounds = u.repeat(num_observations).reshape(1, num_observations)

    # Binary search for quantiles
    for _ in range(max_iter):
        mid_points = (lower_bounds + upper_bounds) / 2

        # Call the distribution's CDF method
        cdf_values = distribution.cdf(mid_points.squeeze(0))

        # Ensure cdf_values has the right shape for comparison
        if cdf_values.ndim == 0:
            cdf_values = cdf_values.unsqueeze(0)
        if (
            len(cdf_values.shape) == 1
            and cdf_values.shape[0] != percentiles_tensor.shape[1]
        ):
            cdf_values = cdf_values.expand(percentiles_tensor.shape[1])

        # Update bounds based on CDF comparison
        mask = cdf_values < percentiles_tensor.squeeze(0)
        lower_bounds = torch.where(mask.unsqueeze(0), mid_points, lower_bounds)
        upper_bounds = torch.where(mask.unsqueeze(0), upper_bounds, mid_points)

        # Check convergence
        if torch.all(torch.abs(upper_bounds - lower_bounds) < tolerance):
            break

    return (lower_bounds + upper_bounds) / 2
