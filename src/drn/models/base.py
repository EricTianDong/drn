from __future__ import annotations
import abc
import tempfile
from pathlib import Path
from typing import Any, Optional, Union
import warnings
import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from lightning.pytorch.utilities import disable_possible_user_warnings
from ..utils import binary_search_icdf

disable_possible_user_warnings()


class BaseModel(L.LightningModule, abc.ABC):
    learning_rate: float

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    def predict(self, x_raw: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Any:
        """Creates a distributional prediction for the input data.
        Here, `x_raw` is raw data, and this method will apply any model-specific preprocessing.
        """
        x = self.preprocess(x_raw)
        return self._predict(x)

    @abc.abstractmethod
    def _predict(self, x: torch.Tensor) -> Any: ...

    def training_step(self, batch: Any, idx: int) -> torch.Tensor:
        x, y = batch
        loss_batch = self.loss(x, y)
        self.log("train_loss", loss_batch, prog_bar=True)
        return loss_batch

    def validation_step(self, batch: Any, idx: int) -> torch.Tensor:
        x, y = batch
        loss_val = self.loss(x, y)
        self.log("val_loss", loss_val, prog_bar=True)
        return loss_val

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        batch_size: int = 128,
        epochs: int = 10,
        patience: int = 5,
        **trainer_kwargs,
    ) -> BaseModel:
        # Set some default trainer arguments
        trainer_kwargs.setdefault("max_epochs", epochs)
        trainer_kwargs.setdefault("accelerator", "cpu")
        trainer_kwargs.setdefault("devices", 1)
        trainer_kwargs.setdefault("logger", False)
        trainer_kwargs.setdefault("deterministic", True)
        trainer_kwargs.setdefault("enable_progress_bar", True)
        trainer_kwargs.setdefault("enable_model_summary", True)

        has_val = X_val is not None and y_val is not None

        # Build training DataLoader
        train_tensor = TensorDataset(
            self.preprocess(X_train), self.preprocess(y_train, targets=True).squeeze()
        )
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

        # Simple train if no validation provided
        if not has_val:
            trainer_kwargs.setdefault("enable_checkpointing", False)
            trainer = L.Trainer(**trainer_kwargs)
            trainer.fit(self, train_loader)
            # Record epochs run
            self.epochs_trained = int(getattr(trainer, "current_epoch", 0))
            self.eval()
            return self

        # Build validation DataLoader
        val_tensor = TensorDataset(
            self.preprocess(X_val), self.preprocess(y_val, targets=True).squeeze()
        )
        val_loader = DataLoader(val_tensor, batch_size=len(val_tensor), shuffle=False)

        if trainer_kwargs.get("enable_checkpointing") is False:
            warnings.warn(
                "Early stopping requires checkpointing. "
                "Overriding enable_checkpointing=True.",
                UserWarning,
            )
        # force checkpointing on, ignore user override
        trainer_kwargs["enable_checkpointing"] = True

        # Validation: checkpointing + early stopping
        # OPTIMIZATION: Use local .tmp directory instead of system temp for better I/O performance
        local_tmpdir = Path(".tmp_checkpoints")
        local_tmpdir.mkdir(exist_ok=True)

        with tempfile.TemporaryDirectory(dir=local_tmpdir) as tmpdir:
            # OPTIMIZATION: Reduce checkpoint frequency - only save on validation, not training
            ckpt_cb = ModelCheckpoint(
                dirpath=tmpdir,
                filename="best",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                save_on_train_epoch_end=False,  # Only save on validation, not every training epoch
            )
            early_cb = EarlyStopping(
                monitor="val_loss", mode="min", patience=patience, verbose=False
            )

            trainer = L.Trainer(callbacks=[ckpt_cb, early_cb], **trainer_kwargs)
            trainer.fit(self, train_loader, val_loader)

            # Restore best weights
            best_path = ckpt_cb.best_model_path
            if best_path:
                ckpt = torch.load(
                    best_path, map_location=lambda s, loc: s, weights_only=False
                )
                state = ckpt.get("state_dict", ckpt)
                self.load_state_dict(state)
            
            # Record epochs actually run
            self.epochs_trained = int(getattr(trainer, "current_epoch", 0))

        self.eval()
        return self

    def preprocess(
        self, x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor], targets=False
    ) -> torch.Tensor:
        """
        Convert input data to a PyTorch tensor. Apply any neural network preprocessing (if applicable).
        """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)

        # If the model has a .ct attribute (for column transformer), only apply to covariates
        if hasattr(self, "ct") and self.ct is not None and not targets:
            x = self.ct.transform(x)

        # Convert the DataFrame/Series to numpy array
        x_np = np.asarray(x)
        if targets:
            x_np = x_np.reshape(-1)

        return torch.Tensor(x_np).to(self.device)

    def icdf(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
        p: float,
        l=None,
        u=None,
        max_iter=1000,
        tolerance=1e-7,
    ) -> torch.Tensor:
        """
        Calculate the inverse CDF (quantiles) of the distribution for the given cumulative probability.

        This is a fallback implementation for PyTorch distributions that don't have icdf implemented.

        Args:
            x: Input features
            p: cumulative probability value at which to evaluate icdf
            l: lower bound for the quantile search
            u: upper bound for the quantile search
            max_iter: maximum number of iterations
            tolerance: stopping criteria

        Returns:
            A tensor of shape (1, batch_shape) containing the inverse CDF values.
        """
        x = self.preprocess(x)
        dists = self.predict(x)

        # Try to use PyTorch distribution's icdf method first
        try:
            quantiles = dists.icdf(torch.tensor(p))
            return quantiles.unsqueeze(0)
        except (AttributeError, NotImplementedError, RuntimeError):
            # Use shared binary search implementation
            return binary_search_icdf(dists, p, l, u, max_iter, tolerance)

    def quantiles(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series],
        percentiles: list,
        l=None,
        u=None,
        max_iter=1000,
        tolerance=1e-7,
    ) -> torch.Tensor:
        """
        Calculate the quantile values for the given observations and percentiles (cumulative probabilities * 100).

        This unified implementation first checks if the distribution has its own quantiles method,
        then falls back to icdf-based approach.
        """
        dists = self.predict(x)

        # Check if the distribution has its own quantiles method (e.g., Histogram, ExtendedHistogram)
        if hasattr(dists, "quantiles") and callable(getattr(dists, "quantiles")):
            return dists.quantiles(percentiles, l, u, max_iter, tolerance)

        # Fallback to icdf-based approach
        quantiles = [
            self.icdf(x, percentile / 100.0, l, u, max_iter, tolerance)
            for percentile in percentiles
        ]
        return torch.stack(quantiles, dim=1)[0]
