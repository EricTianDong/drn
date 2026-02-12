# __init__.py for drn package
from .models import *
from .distributions import *
from .architecture import compute_hidden_sizes, count_params
from .interpretability import DRNExplainer

try:
    from .kernel_shap_explainer import KernelSHAP_DRN
except ImportError:
    pass
from .train import train
from .metrics import crps, quantile_score, quantile_losses, rmse
from .utils import (
    split_and_preprocess,
    split_data,
    preprocess_data,
    replace_rare_categories,
)

from . import models, distributions
from .models import __all__ as _models_all
from .distributions import __all__ as _dist_all

__all__ = [
    *_models_all,
    *_dist_all,
    "DRNExplainer",
    "compute_hidden_sizes",
    "count_params",
    "crps",
    "preprocess_data",
    "replace_rare_categories",
    "rmse",
    "split_and_preprocess",
    "split_data",
    "train",
    "quantile_score",
    "quantile_losses",
]  # type: ignore[reportUnsupportedDunderAll]
