# Training

Training framework and utilities for DRN models, including the main training loop, loss functions, and optimization helpers.

## Main Training Function

::: drn.train.train
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Loss Functions

### DRN Loss

::: drn.models.drn_loss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### JBCE Loss

::: drn.models.ddr.jbce_loss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### NLL Loss

::: drn.metrics.nll
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Model Utilities

### Cutpoint Generation

::: drn.models.drn_cutpoints
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### GLM Utilities

::: drn.models.glm.gaussian_deviance_loss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: drn.models.glm.gamma_deviance_loss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3