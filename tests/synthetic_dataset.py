import numpy as np
import torch


def generate_synthetic_data(n=1000, seed=1) -> tuple[torch.Tensor, torch.Tensor, torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    rng = np.random.default_rng(seed)
    x_all = rng.random(size=(n, 4))
    epsilon = rng.normal(0, 0.2, n)

    means = np.exp(
        0
        - 0.5 * x_all[:, 0]
        + 0.5 * x_all[:, 1]
        + np.sin(np.pi * x_all[:, 0])
        - np.sin(np.pi * np.log(x_all[:, 2] + 1))
        + np.cos(x_all[:, 1] * x_all[:, 2])
    ) + np.cos(x_all[:, 1])
    dispersion = 0.5

    y_lognormal = np.exp(rng.normal(means / 4, dispersion))
    y_gamma = rng.gamma(1 / dispersion, scale=dispersion * means / 4)

    y_all = y_gamma * 0.5 + y_lognormal * 0.5 + epsilon**2

    num_val = int(n * 0.2)
    num_test = int(n * 0.2)
    num_train = n - num_val - num_test

    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_val, y_val = (
        x_all[num_train : num_train + num_val],
        y_all[num_train : num_train + num_val],
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Convert NumPy arrays to PyTorch tensors and move them to the specified device
    X_train = torch.tensor(x_train).float().to(device)
    Y_train = torch.tensor(y_train).float().to(device)
    X_val = torch.tensor(x_val).float().to(device)
    Y_val = torch.tensor(y_val).float().to(device)

    # Create Tensor datasets for training and validation
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

    # Create DataLoader objects for training and validation

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False)

    return X_train, Y_train, train_loader, val_loader
