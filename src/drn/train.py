from __future__ import annotations
import copy
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

if TYPE_CHECKING:
    from .models.base import BaseModel


def train(
    model: BaseModel,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    epochs=200,
    patience=5,
    lr: Optional[float] = None,
    device: Optional[torch.device] = None,
    log_interval=10,
    batch_size=128,
    optimizer=torch.optim.Adam,
    print_details=True,
    keep_best=True,
    gradient_clipping=False,
) -> None:
    """
    A generic neural network training function given a model and datasets.
    Args:
        model: The model to train.
        train_dataset: Dataset for training.
        val_dataset: Dataset for validation.
        epochs: Number of epochs to train for.
        patience: Number of epochs with no improvement after which training will be stopped.
        lr: Learning rate for the optimizer.
        device: Device to use for training (default is determined automatically).
        log_interval: How often to log training progress.
        batch_size: Batch size for training and validation.
        optimizer: Optimizer class to use (default is Adam).
        print_details: Whether to print detailed logs during training.
        keep_best: Whether to return the best model found during training.
        gradient_clipping: Whether to apply gradient clipping.
    """

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False
    )

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        # Temporarily disable MPS which doesn't support some operations we need.
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # If both learning rate is supplied, and the model has a learning rate attribute, use the supplied one.
    if lr is not None:
        if print_details:
            if hasattr(model, "learning_rate") and model.learning_rate is not None:
                print("Ignoring model's learning rate.")
        learning_rate = lr
    elif hasattr(model, "learning_rate") and model.learning_rate is not None:
        learning_rate = model.learning_rate
    else:
        raise ValueError("Need a learning rate to train the model.")

    try:
        if print_details:
            print(f"Using device: {device}")
        model.to(device)

        optimizer = optimizer(model.parameters(), lr=learning_rate)

        no_improvement = 0
        best_loss = torch.Tensor([float("inf")]).to(device)
        best_model = model.state_dict()

        range_selection = trange if print_details else range
        for epoch in range_selection(1, epochs + 1):
            epochs_run = epoch + 1
            model.train()

            for data, target in train_loader:
                x = data.to(device)
                y = target.to(device)
                optimizer.zero_grad()
                loss_batch = model.loss(x, y)
                loss_batch.backward()

                # Check for NaN gradients
                for name, param in model.named_parameters():
                    # Parameters might not have gradients if they are not trainable
                    if param.grad is not None and torch.isnan(param.grad).any():
                        if print_details:
                            tqdm.write("Stopping training as NaN gradient detected")
                        raise ValueError(
                            f"Gradient NaN detected in {name}. Try smaller learning rates."
                        )

                if gradient_clipping:
                    # If no error is raised, proceed with gradient clipping and optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            model.eval()

            loss_val = torch.zeros(1, device=device)

            for data, target in val_loader:
                x = data.to(device)
                y = target.to(device)
                loss_val += model.loss(x, y)

            if loss_val < best_loss:
                best_loss = loss_val
                best_model = copy.deepcopy(model.state_dict())
                no_improvement = 0

            else:
                no_improvement += 1

            if print_details and epoch % log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch} \t| batch train loss: {loss_batch.item():.4f}"
                    + f"\t| validation loss:  {loss_val.item():.4f}"
                    + f"\t| no improvement: {no_improvement}"
                )

            if no_improvement > patience:
                if print_details:
                    tqdm.write("Stopping early!")
                break

    finally:
        # If we never improved the loss, throw an error.
        if best_loss == float("inf"):
            raise ValueError("Training failed.")

        if keep_best:
            # If requested, return tbe best model found during training.
            model.load_state_dict(best_model)
        
        model.epochs_trained = epochs_run

        # Make sure dropout is always disabled after training
        model.eval()
