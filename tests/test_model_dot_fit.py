import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pytest
import torch
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from synthetic_dataset import generate_synthetic_data

from drn import (
    GLM,
    CANN,
    MDN,
    DDR,
    DRN,
    gamma_estimate_dispersion,
    merge_cutpoints,
    crps,
    ddr_cutpoints,
    drn_cutpoints,
    default_drn_cutpoints,
)


def _to_tensor(arr):
    """Convert NumPy array or pandas to torch.FloatTensor."""
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        arr = arr.values
    return torch.Tensor(arr)


def check_crps(model, X_train, y_train, grid_size=3000):
    X = _to_tensor(X_train)
    Y = _to_tensor(y_train).squeeze()
    grid = torch.linspace(0, Y.max().item() * 1.1, grid_size).unsqueeze(-1).to(X.device)
    dists = model.predict(X)
    cdfs = dists.cdf(grid)
    grid = grid.squeeze()
    crps_val = crps(Y, grid, cdfs)
    assert crps_val.shape == Y.shape
    assert crps_val.mean() > 0


def test_glm():
    print("\n\nTraining GLM\n")
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    # --- 1) NUMPY inputs ---
    torch.manual_seed(1)
    glm_np = GLM("gamma")
    glm_np.fit(X_train, y_train, X_val, y_val, epochs=2)
    glm_np.update_dispersion(X_train, y_train)
    check_crps(glm_np, X_train, y_train)

    # --- 2) PANDAS inputs (sanity check) ---
    Xtr_df = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    ytr_sr = pd.Series(y_train, name="Y")
    Xvl_df = pd.DataFrame(X_val, columns=Xtr_df.columns)
    yvl_sr = pd.Series(y_val, name="Y")

    torch.manual_seed(1)
    glm_pd = GLM("gamma")
    # one short epoch just to confirm no errors
    glm_pd.fit(Xtr_df, ytr_sr, Xvl_df, yvl_sr, epochs=1)
    glm_pd.update_dispersion(Xtr_df, ytr_sr)
    check_crps(glm_pd, Xtr_df, ytr_sr)


def test_glm_fit_with_statsmodels():
    X_train, y_train, _, _ = generate_synthetic_data()

    # tensor‐based
    glm1 = GLM("gamma").fit(_to_tensor(X_train), _to_tensor(y_train))
    d1 = gamma_estimate_dispersion(
        glm1(_to_tensor(X_train)), _to_tensor(y_train), X_train.shape[1]
    )
    assert np.isclose(d1, glm1.dispersion.item())

    # numpy-based
    glm2 = GLM("gamma").fit(X_train, y_train)
    assert np.isclose(d1, glm2.dispersion.item())

    # pandas‐based, compare to statsmodels predictions
    X_df = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    y_sr = pd.Series(y_train, name="Y")
    glm3 = GLM("gamma").fit(X_df, y_sr)

    sm_mod = sm.GLM(
        y_sr, sm.add_constant(X_df), family=Gamma(link=sm.families.links.Log())
    ).fit()
    sm_pred = sm_mod.predict(sm.add_constant(X_df))
    our_pred = glm3(_to_tensor(X_df)).cpu().detach().numpy()
    assert np.allclose(sm_pred, our_pred)


def test_update_dispersion_dataframe_targets_matches_statsmodels():
    """A DataFrame target should survive preprocess without corrupting dispersion."""

    X_train, y_train, _, _ = generate_synthetic_data()
    X_df = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    y_df = pd.DataFrame(y_train, columns=["Y"])

    sm_results = sm.GLM(
        y_df["Y"].values,
        sm.add_constant(X_df.values),
        family=Gamma(link=sm.families.links.Log()),
    ).fit()

    glm = GLM("gamma")
    glm.linear = torch.nn.Linear(X_df.shape[1], 1)
    with torch.no_grad():
        glm.linear.weight.copy_(
            torch.tensor(sm_results.params[1:], dtype=torch.float32).unsqueeze(0)
        )
        glm.linear.bias.copy_(
            torch.tensor([sm_results.params[0]], dtype=torch.float32)
        )

    glm.update_dispersion(X_df, y_df)

    assert glm.dispersion.item() == pytest.approx(sm_results.scale, rel=1e-3)


def test_cann():
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    torch.manual_seed(2)
    base = GLM("gamma")
    base.fit(X_train, y_train, X_val, y_val, epochs=2)

    cann = CANN(base, num_hidden_layers=2, hidden_size=100)
    cann.fit(X_train, y_train, X_val, y_val, epochs=2)

    cann.update_dispersion(X_train, y_train)
    check_crps(cann, X_train, y_train)


def test_fit_works_for_pd_series_targets():
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    # Convert to pd.Series
    y_train = pd.Series(y_train, name="Y")
    y_val = pd.Series(y_val, name="Y")

    torch.manual_seed(2)
    base = GLM("gamma")
    base.fit(X_train, y_train, X_val, y_val, epochs=2)

    cann = CANN(base, num_hidden_layers=2, hidden_size=100)
    cann.fit(X_train, y_train, X_val, y_val, epochs=2)

    cann.update_dispersion(X_train, y_train)
    check_crps(cann, X_train, y_train)


def test_fit_works_for_pd_dataframe_targets():
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    # Convert to pd.DataFrame
    y_train = pd.DataFrame(y_train, columns=["Y"])
    y_val = pd.DataFrame(y_val, columns=["Y"])

    torch.manual_seed(2)
    base = GLM("gamma")
    base.fit(X_train, y_train, X_val, y_val, epochs=2)

    cann = CANN(base, num_hidden_layers=2, hidden_size=100)
    cann.fit(X_train, y_train, X_val, y_val, epochs=2)

    # cann.update_dispersion(X_train, y_train)
    check_crps(cann, X_train, y_train)


def test_mdn():
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    torch.manual_seed(3)
    mdn = MDN("gamma", num_components=5)
    mdn.fit(X_train, y_train, X_val, y_val, epochs=2)

    check_crps(mdn, X_train, y_train)


def setup_cutpoints(y_train):
    Y = _to_tensor(y_train)
    max_y = Y.max().item()
    n = Y.numel()
    cps = np.linspace(0.0, max_y * 1.01, int(np.ceil(0.1 * n)))
    assert len(cps) >= 2
    return cps.tolist()


def test_ddr():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    cps = setup_cutpoints(y_train)

    torch.manual_seed(4)
    ddr = DDR(cps, hidden_size=100)
    ddr.fit(X_train, y_train, X_val, y_val, epochs=2)

    check_crps(ddr, X_train, y_train)


def test_drn():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    cps0 = setup_cutpoints(y_train)
    cps1 = merge_cutpoints(cps0, y_train, min_obs=2)
    assert len(cps1) >= 2

    torch.manual_seed(5)
    base = GLM("gamma")
    base.fit(X_train, y_train, X_val, y_val, epochs=2)

    for loss_fn in ("jbce", "nll"):
        drn = DRN(
            base,
            cps1,
            hidden_size=100,
            loss_metric=loss_fn,
            kl_alpha=1.0,
            mean_alpha=1.0,
            dv_alpha=1.0,
            tv_alpha=1.0,
        )
        drn.fit(X_train, y_train, X_val, y_val, epochs=2)
        check_crps(drn, X_train, y_train)


def test_torch_api():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    cps = setup_cutpoints(y_train)

    torch.manual_seed(5)
    glm = GLM("gamma").fit(X_train, y_train)

    hs = 5
    drn1 = DRN(glm, cps, num_hidden_layers=2, hidden_size=hs)
    drn1.fit(X_train, y_train, X_val, y_val, epochs=2)
    total_w = sum(p.numel() for p in drn1.hidden_layers.parameters())
    expected = hs * X_train.shape[1] + hs + hs * hs + hs
    assert total_w == expected

    # numpy‐int hyperparameters
    drn2 = DRN(glm, cps, num_hidden_layers=np.int64(2), hidden_size=np.int64(hs))
    drn2.fit(X_train, y_train, X_val, y_val, epochs=2)
    total_w2 = sum(p.numel() for p in drn2.hidden_layers.parameters())
    assert total_w2 == expected and len(drn2.hidden_layers) // 3 == 2

    # dropout behaviour
    drn3 = DRN(glm, cps, num_hidden_layers=2, hidden_size=hs, dropout_rate=0.5)
    drn3.fit(X_train, y_train, X_val, y_val, epochs=2)

    drn3.train()
    p1 = drn3(_to_tensor(X_train))[-1]
    p2 = drn3(_to_tensor(X_train))[-1]
    assert not torch.allclose(p1, p2)

    drn3.eval()
    q1 = drn3(_to_tensor(X_train))[-1]
    q2 = drn3(_to_tensor(X_train))[-1]
    assert torch.allclose(q1, q2)


def test_ddr_supplied_vs_default_cutpoints_equivalence():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    p = X_train.shape[1]
    proportion = 0.1

    # Compute what the default cutpoints *should* be
    c0 = min(y_train.min() * 1.05, 0)
    cK = y_train.max() * 1.05
    expected_cps = ddr_cutpoints(c0, cK, proportion=proportion, n=len(y_train))

    # 1) Model with SUPPLIED cutpoints
    torch.manual_seed(42)
    ddr_sup = DDR(cutpoints=expected_cps, proportion=proportion)
    ddr_sup.fit(X_train, y_train, X_val, y_val, epochs=3)

    # 2) Model with DEFAULT cutpoints
    torch.manual_seed(42)
    ddr_def = DDR(cutpoints=None, proportion=proportion)
    ddr_def.fit(X_train, y_train, X_val, y_val, epochs=3)

    # => their cutpoints should now be equal
    assert torch.allclose(
        ddr_sup.cutpoints, ddr_def.cutpoints
    ), "Auto-generated cutpoints differ from manually supplied ones"

    # => and all parameters should match
    params_sup = dict(ddr_sup.named_parameters())
    params_def = dict(ddr_def.named_parameters())
    for name in params_sup:
        p1, p2 = params_sup[name], params_def[name]
        assert torch.allclose(p1, p2), f"Parameter {name} mismatch"

    # => finally, they should give identical CRPS
    check_crps(ddr_sup, X_train, y_train)
    check_crps(ddr_def, X_train, y_train)


def test_ddr_default_cutpoints_match_function():
    # verify that DDR.fit actually uses ddr_cutpoints under the hood
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    p = X_train.shape[1]
    proportion = 0.2  # try a different proportion

    # compute expected cutpoints
    c0 = min(y_train.min() * 1.05, 0)
    cK = y_train.max() * 1.05
    expected = torch.Tensor(
        ddr_cutpoints(c0, cK, proportion=proportion, n=len(y_train))
    )

    ddr = DDR(cutpoints=None, proportion=proportion)
    # trigger auto-generation
    ddr.fit(X_train, y_train, X_val, y_val, epochs=1)

    assert torch.allclose(
        ddr.cutpoints, expected
    ), "DDR.fit did not generate cutpoints matching ddr_cutpoints()"


def test_drn_supplied_vs_default_cutpoints_equivalence():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    p = X_train.shape[1]
    proportion = 0.1
    min_obs = 1

    # Compute what the default DRN cutpoints should be:
    expected_cuts = default_drn_cutpoints(
        y_train, proportion=proportion, min_obs=min_obs
    )

    # 1) DRN with SUPPLIED cutpoints
    torch.manual_seed(123)
    glm1 = GLM("gamma")
    glm1.fit(X_train, y_train, X_val, y_val, epochs=3)
    drn_sup = DRN(glm1, cutpoints=expected_cuts)
    drn_sup.fit(X_train, y_train, X_val, y_val, epochs=3)

    # 2) DRN with DEFAULT cutpoints
    torch.manual_seed(123)
    glm2 = GLM("gamma")
    glm2.fit(X_train, y_train, X_val, y_val, epochs=3)
    drn_def = DRN(glm2, cutpoints=None)
    drn_def.fit(X_train, y_train, X_val, y_val, epochs=3)

    # => their cutpoints should now be equal
    assert torch.allclose(
        drn_sup.cutpoints, drn_def.cutpoints
    ), "Auto‐generated DRN cutpoints differ from manually supplied ones"

    # => and all other parameters should match
    sup_params = dict(drn_sup.named_parameters())
    def_params = dict(drn_def.named_parameters())
    for name in sup_params:
        if name == "cutpoints":
            continue
        p1, p2 = sup_params[name], def_params[name]
        assert torch.allclose(p1, p2), f"DRN parameter {name} mismatch"

    # => finally, they should give identical CRPS
    check_crps(drn_sup, X_train, y_train)
    check_crps(drn_def, X_train, y_train)


def test_drn_default_cutpoints_match_function():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    p = X_train.shape[1]
    proportion = 0.2
    min_obs = 3

    # compute expected cutpoints via ddr_cutpoints + merge_cutpoints
    c0 = min(y_train.min() * 1.05, 0)
    cK = y_train.max() * 1.05
    ddr_cps = ddr_cutpoints(c0, cK, proportion=proportion, n=len(y_train))
    expected = torch.tensor(merge_cutpoints(ddr_cps, y_train, min_obs=min_obs))

    # trigger auto-generation
    glm = GLM("gamma")
    glm.fit(X_train, y_train, X_val, y_val, epochs=1)

    drn = DRN(glm, cutpoints=None, proportion=proportion, min_obs=min_obs)
    drn.fit(X_train, y_train, X_val, y_val, epochs=1)

    assert torch.allclose(
        drn.cutpoints, expected
    ), "DRN.fit did not generate cutpoints matching merge_cutpoints(ddr_cutpoints(...))"


def test_pandas_and_tensor_inputs_agree():
    # This error only showed up with a larger dataset when the cutpoints are
    # sometimes setup using float64 and sometimes float32 values.
    X_train_np, y_train_np, X_val_np, y_val_np = generate_synthetic_data(n=10_000)

    X_train_pd = pd.DataFrame(
        X_train_np, columns=[f"X{i}" for i in range(X_train_np.shape[1])]
    )
    X_val_pd = pd.DataFrame(X_val_np, columns=X_train_pd.columns)
    y_train_pd = pd.Series(y_train_np, name="Y")
    y_val_pd = pd.Series(y_val_np, name="Y")

    X_train_tensor = torch.Tensor(X_train_pd.values)
    X_val_tensor = torch.Tensor(X_val_pd.values)
    y_train_tensor = torch.Tensor(y_train_pd.values).flatten()
    y_val_tensor = torch.Tensor(y_val_pd.values).flatten()

    glm = GLM("gamma").fit(X_train_pd, y_train_pd)

    torch.manual_seed(23)
    drn_model = DRN(
        glm, cutpoints=None, hidden_size=128, num_hidden_layers=2, baseline_start=False
    )

    drn_model.fit(
        X_train_pd,
        y_train_pd,
        X_val_pd,
        y_val_pd,
        batch_size=256,
        epochs=1,
        enable_progress_bar=False,
    )

    # Now compare with tensor inputs
    torch.manual_seed(23)
    drn_model_tensor = DRN(
        glm, cutpoints=None, hidden_size=128, num_hidden_layers=2, baseline_start=False
    )

    drn_model_tensor.fit(
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        batch_size=256,
        epochs=1,
        enable_progress_bar=False,
    )

    # Compare parameters
    params1 = dict(drn_model.named_parameters())
    params2 = dict(drn_model_tensor.named_parameters())

    for name in params1:
        if name == "cutpoints":
            continue
        p1, p2 = params1[name], params2[name]
        assert torch.allclose(p1, p2, atol=1e-5), f"DRN parameter {name} mismatch"
