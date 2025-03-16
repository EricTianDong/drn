import os
import torch
import numpy as np
import pandas as pd
import lightning as L
from synthetic_dataset import generate_synthetic_data
from drn import *
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


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


def check_crps(model, X_train, Y_train, grid_size=3000):
    grid = torch.linspace(0, Y_train.max().item() * 1.1, grid_size).unsqueeze(-1).to(X_train.device)
    dists = model.distributions(X_train)
    cdfs = dists.cdf(grid)
    grid = grid.squeeze()
    crps_ = crps(Y_train, grid, cdfs)
    assert crps_.shape == Y_train.shape
    assert crps_.mean() > 0


def test_glm():
    print("\n\nTraining GLM with Lightning\n")
    X_train, Y_train, train_loader, val_loader = generate_synthetic_data()

    torch.manual_seed(1)
    glm_model = GLM(X_train.shape[1], distribution="gamma")
    trainer = L.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(glm_model, train_loader, val_loader)

    glm_model.update_dispersion(X_train, Y_train)
    check_crps(glm_model, X_train, Y_train)


def test_cann():
    print("\n\nTraining CANN with Lightning\n")
    X_train, Y_train, train_loader, val_loader = generate_synthetic_data()

    torch.manual_seed(2)
    glm = GLM(X_train.shape[1], distribution="gamma")
    trainer = L.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(glm, train_loader, val_loader)

    cann_model = CANN(glm, num_hidden_layers=2, hidden_size=100)
    trainer.fit(cann_model, train_loader, val_loader)

    cann_model.update_dispersion(X_train, Y_train)
    check_crps(cann_model, X_train, Y_train)


def test_mdn():
    print("\n\nTraining MDN with Lightning\n")
    X_train, Y_train, train_loader, val_loader = generate_synthetic_data()

    torch.manual_seed(3)
    mdn_model = MDN(X_train.shape[1], num_components=5, distribution="gamma")
    trainer = L.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(mdn_model, train_loader, val_loader)

    check_crps(mdn_model, X_train, Y_train)


def test_ddr():
    print("\n\nTraining DDR with Lightning\n")
    X_train, Y_train, train_loader, val_loader = generate_synthetic_data()

    cutpoints_ddr = setup_cutpoints(Y_train)

    torch.manual_seed(4)
    ddr_model = DDR(X_train.shape[1], cutpoints_ddr, hidden_size=100)
    trainer = L.Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(ddr_model, train_loader, val_loader)

    check_crps(ddr_model, X_train, Y_train)


def test_drn():
    print("\n\nTraining DRN with Lightning\n")
    X_train, Y_train, train_loader, val_loader = generate_synthetic_data()

    cutpoints_ddr = setup_cutpoints(Y_train)
    cutpoints_drn = merge_cutpoints(cutpoints_ddr, Y_train.cpu().numpy(), min_obs=2)

    torch.manual_seed(5)
    glm = GLM(X_train.shape[1], distribution="gamma")
    trainer = L.Trainer(max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu") # MPS error
    trainer.fit(glm, train_loader, val_loader)
    glm.update_dispersion(X_train, Y_train)

    drn_model = DRN(X_train.shape[1], cutpoints_drn, glm, hidden_size=100)
    trainer.fit(drn_model, train_loader, val_loader)

    check_crps(drn_model, X_train, Y_train)


def test_torch():
    print("\n\nTraining DRN (Torch compatibility check) with Lightning\n")
    X_train, Y_train, train_loader, val_loader = generate_synthetic_data()

    cutpoints = setup_cutpoints(Y_train)

    torch.manual_seed(5)
    glm = GLM.from_statsmodels(X_train, Y_train, distribution="gamma")

    hs = 5
    drn_model = DRN(X_train.shape[1], cutpoints, glm, num_hidden_layers=2, hidden_size=hs)
    trainer = L.Trainer(max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu") # MPS error
    trainer.fit(drn_model, train_loader, val_loader)

    check_crps(drn_model, X_train, Y_train)