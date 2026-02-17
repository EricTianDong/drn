import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from synthetic_dataset import generate_synthetic_tensordataset

from drn import *

# Check if SHAP is available (optional dependency)
try:
    import shap
    from drn.kernel_shap_explainer import KernelSHAP_DRN
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@pytest.mark.skipif(not HAS_SHAP, reason="SHAP not installed - install with: pip install drn[shap]")
def test_plot_adjustment_factors():

    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()
    y_train = Y_train.cpu().numpy()

    x_train = pd.DataFrame(
        X_train.detach().cpu().numpy(),
        columns=[f"X_{i}" for i in range(X_train.shape[1])],
    )

    cutpoints = drn_cutpoints(0, 1, y_train, 0.1, 2)

    glm = GLM("gamma").fit(X_train, Y_train)

    drn = DRN(glm, cutpoints)
    train(drn, train_dataset, val_dataset, epochs=2)

    # Make a column transformer that standardizes the data
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), ["X_0", "X_1", "X_2", "X_3"])]
    )
    preprocessor.fit(x_train)

    drn_explainer = DRNExplainer(
        drn, glm, cutpoints, x_train, preprocessor=preprocessor
    )

    # Put in a DataFrame with different columns as x_train
    instance = pd.DataFrame(
        np.array([[0.0, 1.0, 2.0, 3.0]]), columns=["X_1", "X_2", "X_3", "X_4"]
    )

    with pytest.raises(ValueError, match="X_0"):
        _ = drn_explainer.preprocessor.transform(instance)

    instance.columns = ["X_0", "X_1", "X_2", "X_3"]
    _ = drn_explainer.preprocessor.transform(instance)
    assert hasattr(drn_explainer.preprocessor, "feature_names_in_")

    # Check that the default drn_explainer.preprocessor.transform works with DataFrame and has a feature_names_in attribute
    drn_explainer = DRNExplainer(drn, glm, cutpoints, x_train)

    transformed_instance_df = drn_explainer.preprocessor.transform(instance)
    assert hasattr(drn_explainer.preprocessor, "feature_names_in_")

    drn_explainer.plot_adjustment_factors(
        instance,
        num_interpolations=3000,
        plot_adjustments_labels=False,
        plot_mean_adjustment=True,
    )

    instance = np.array([[0.0, 1.0, 2.0, 3.0]])

    drn_explainer.plot_adjustment_factors(
        instance,
        num_interpolations=3000,
        plot_adjustments_labels=False,
        plot_mean_adjustment=True,
    )

    drn_explainer.cdf_plot(
        instance,
        method="Kernel",
        nsamples_background_fraction=0.5,
        top_K_features=3,
        labelling_gap=0.15,
        dist_property="90% Quantile",
        x_range=(2.5, 4.5),
        y_range=(0.87, 0.93),
        density_transparency=0.9,
        adjustment=True,
        shap_fontsize=40,
        plot_title="90% Quantile Adjustment Explanation",
        figsize=(20, 20),
    )
