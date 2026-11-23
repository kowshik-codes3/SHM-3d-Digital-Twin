"""Inference engine and diagnostic plotting for the trained LSTMAutoencoder."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ReconstructionEvaluator:
    """Runs forward-pass inference and computes per-window and per-channel MSE residuals.

    Usage
    -----
    evaluator = ReconstructionEvaluator(model, device)
    mse_vals, x_true, x_hat = evaluator.evaluate(dataloader)
    evaluator.plot_convergence(history, output_dir)
    evaluator.plot_reconstruction(x_true, x_hat, output_dir)
    """

    CHANNEL_NAMES = [
        "Abutment_Left",
        "Pier_Left",
        "MidSpan_Girder",
        "Pier_Right",
        "Ambient_Temp",
    ]

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    # ──────────────────────────────────────────────────────────────────────
    def evaluate(
        self, loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs full inference pass; returns per-window MSE and raw arrays.

        Args:
            loader: DataLoader yielding tensors of shape (B, T, C).

        Returns:
            mse_per_window: (M,) – scalar MSE for each window.
            x_true_all:     (M, T, C) – concatenated ground-truth windows.
            x_hat_all:      (M, T, C) – concatenated reconstructed windows.
        """
        self.model.eval()
        x_true_list, x_hat_list = [], []

        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device)           # (B, T, C)
                x_hat, _ = self.model(x)            # (B, T, C)
                x_true_list.append(x.cpu().numpy())
                x_hat_list.append(x_hat.cpu().numpy())

        x_true_all = np.concatenate(x_true_list, axis=0)  # (M, T, C)
        x_hat_all  = np.concatenate(x_hat_list, axis=0)   # (M, T, C)

        # Per-window MSE: mean over T and C axes
        mse_per_window = np.mean((x_true_all - x_hat_all) ** 2, axis=(1, 2))  # (M,)
        return mse_per_window, x_true_all, x_hat_all

    # ──────────────────────────────────────────────────────────────────────
    def per_channel_mse(self, x_true: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """Computes mean per-channel MSE averaged over all windows and timesteps.

        $e_c = \frac{1}{M \cdot T} \sum_{m,t} (X_{m,t,c} - \hat{X}_{m,t,c})^2$

        Returns:
            (C,) float array of per-channel MSE values.
        """
        return np.mean((x_true - x_hat) ** 2, axis=(0, 1))  # (C,)

    # ──────────────────────────────────────────────────────────────────────
    def plot_convergence(
        self,
        history: Dict[str, list],
        output_dir: str = "reports/phase4",
    ) -> None:
        """Saves train vs. val loss convergence curve.

        Args:
            history:    Dict with keys 'train_loss', 'val_loss'.
            output_dir: Output folder path.
        """
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=1.5)
        ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#ff7f0e", lw=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("LSTM Autoencoder: Training & Validation Loss Convergence")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(p / "loss_convergence.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p / 'loss_convergence.png'}")

    # ──────────────────────────────────────────────────────────────────────
    def plot_reconstruction(
        self,
        x_true: np.ndarray,
        x_hat: np.ndarray,
        output_dir: str = "reports/phase4",
        sample_idx: int = 0,
        title_prefix: str = "Healthy",
    ) -> None:
        """Overlays true vs. reconstructed signal for a single sample window.

        Args:
            x_true:       (M, T, C) true windows.
            x_hat:        (M, T, C) reconstructed windows.
            output_dir:   Output folder path.
            sample_idx:   Which window index to visualise.
            title_prefix: Label for the plot title (e.g. 'Healthy' or 'Damaged').
        """
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        n_channels = x_true.shape[2]
        t = np.arange(x_true.shape[1])

        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)
        for c, ax in enumerate(axes):
            name = self.CHANNEL_NAMES[c] if c < len(self.CHANNEL_NAMES) else f"Ch{c}"
            ax.plot(t, x_true[sample_idx, :, c], label="True",   color="steelblue",  lw=1.5)
            ax.plot(t, x_hat[sample_idx,  :, c], label="Recon.", color="darkorange", lw=1.2, linestyle="--")
            ax.set_ylabel(name, fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.4)

        axes[-1].set_xlabel("Time Step")
        fig.suptitle(f"Signal Reconstruction – {title_prefix} Window #{sample_idx}", fontsize=12)
        fig.tight_layout()

        fname = f"signal_reconstruction_{title_prefix.lower()}.png"
        fig.savefig(p / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p / fname}")
