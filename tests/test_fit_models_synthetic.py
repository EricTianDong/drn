import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from synthetic_dataset import generate_synthetic_tensordataset

from drn import *


def check_crps(model, X_train, Y_train, grid_size=3000):
    grid = torch.linspace(0, Y_train.max().item() * 1.1, grid_size).unsqueeze(-1)
    grid = grid.to(X_train.device)
    dists = model.predict(X_train)
    cdfs = dists.cdf(grid)
    grid = grid.squeeze()
    crps_ = crps(Y_train, grid, cdfs)
    assert crps_.shape == Y_train.shape
    assert crps_.mean() > 0


def test_glm():
    print("\n\nTraining GLM\n")
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()

    torch.manual_seed(1)
    glm = GLM("gamma")
    train(glm, train_dataset, val_dataset, epochs=2)

    glm.update_dispersion(X_train, Y_train)

    check_crps(glm, X_train, Y_train)


def test_cann():
    print("\n\nTraining CANN\n")
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()

    torch.manual_seed(2)
    glm = GLM("gamma")
    train(glm, train_dataset, val_dataset, epochs=2)

    cann = CANN(glm, num_hidden_layers=2, hidden_size=100)
    train(cann, train_dataset, val_dataset, epochs=2)

    cann.update_dispersion(X_train, Y_train)

    check_crps(cann, X_train, Y_train)


def test_mdn():
    print("\n\nTraining MDN\n")
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()

    torch.manual_seed(3)
    mdn = MDN("gamma", num_components=4)
    train(mdn, train_dataset, val_dataset, epochs=2)

    check_crps(mdn, X_train, Y_train)


def setup_cutpoints(Y_train):
    max_y = torch.max(Y_train).item()
    len_y = torch.numel(Y_train)
    c_0 = 0.0
    c_K = max_y * 1.01
    p = 0.1
    num_cutpoints = int(np.ceil(p * len_y))
    cutpoints_ddr = list(np.linspace(c_0, c_K, num_cutpoints))
    assert len(cutpoints_ddr) >= 2
    return cutpoints_ddr


def test_ddr():
    print("\n\nTraining DDR\n")
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()

    cutpoints_ddr = setup_cutpoints(Y_train)

    torch.manual_seed(4)
    ddr = DDR(cutpoints_ddr, hidden_size=100)
    train(ddr, train_dataset, val_dataset, epochs=2)

    check_crps(ddr, X_train, Y_train)


def test_drn():
    print("\n\nTraining DRN\n")
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()
    y_train = Y_train.cpu().numpy()

    cutpoints_ddr = setup_cutpoints(Y_train)
    cutpoints_drn = merge_cutpoints(cutpoints_ddr, y_train, min_obs=2)
    assert len(cutpoints_drn) >= 2

    torch.manual_seed(5)
    glm = GLM("gamma")
    train(glm, train_dataset, val_dataset, epochs=2)
    glm.update_dispersion(X_train, Y_train)

    drn = DRN(
        glm,
        cutpoints_drn,
        hidden_size=100,
        loss_metric="jbce",
        kl_alpha=1.0,
        mean_alpha=1.0,
        dv_alpha=1.0,
        tv_alpha=1.0,
    )

    # Try both loss functions with their regularisation terms enabled
    train(drn, train_dataset, val_dataset, epochs=2)

    drn = DRN(
        glm,
        cutpoints_drn,
        hidden_size=100,
        loss_metric="nll",
        kl_alpha=1.0,
        mean_alpha=1.0,
        dv_alpha=1.0,
        tv_alpha=1.0,
    )

    train(drn, train_dataset, val_dataset, epochs=2)

    check_crps(drn, X_train, Y_train)


def test_torch():
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()

    cutpoints = setup_cutpoints(Y_train)

    torch.manual_seed(5)
    glm = GLM("gamma").fit(X_train, Y_train)

    hs = 5
    drn = DRN(glm, cutpoints, num_hidden_layers=2, hidden_size=hs)
    train(drn, train_dataset, val_dataset, epochs=2)

    # Calculate the expected number of weights & biases given two layers of hs hidden units
    expected_num_weights = hs * X_train.shape[1] + hs + hs * hs + hs
    num_weights = sum([p.numel() for p in drn.hidden_layers.parameters()])
    assert num_weights == expected_num_weights

    # Try again using np.int64 instead of ints for the hyperparameters
    drn = DRN(glm, cutpoints, num_hidden_layers=np.int64(2), hidden_size=np.int64(hs))

    train(drn, train_dataset, val_dataset, epochs=2)

    num_weights = sum([p.numel() for p in drn.hidden_layers.parameters()])
    assert num_weights == expected_num_weights

    # Check dropout is working as intended
    drn = DRN(glm, cutpoints, num_hidden_layers=2, hidden_size=hs, dropout_rate=0.5)
    train(drn, train_dataset, val_dataset, epochs=2)

    # Make sure two different predictions (which in drn.train mode) are different
    drn.train()
    preds1 = drn(X_train)[-1]
    preds2 = drn(X_train)[-1]
    assert not torch.allclose(preds1, preds2)

    # Make sure two different predictions (which in drn.eval mode) are the same
    drn.eval()
    preds1 = drn(X_train)[-1]
    preds2 = drn(X_train)[-1]
    assert torch.allclose(preds1, preds2)


# Just made all the fit methods return self
# Just run constructor then fit method and save to a variable
# to see if they all work
def test_fit_chain():
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_tensordataset()

    torch.manual_seed(6)
    glm = GLM("gamma").fit(X_train, Y_train)

    cann = CANN(glm).fit(X_train, Y_train)

    mdn = MDN("gamma", num_components=5).fit(X_train, Y_train)

    ddr = DDR().fit(X_train, Y_train)

    drn = DRN(glm).fit(X_train, Y_train)

    constant = Constant("gamma").fit(X_train, Y_train)

    # Check that none are None
    assert glm is not None, "GLM fit returned None"
    assert cann is not None, "CANN fit returned None"
    assert mdn is not None, "MDN fit returned None"
    assert ddr is not None, "DDR fit returned None"
    assert drn is not None, "DRN fit returned None"
    assert constant is not None, "Constant fit returned None"
