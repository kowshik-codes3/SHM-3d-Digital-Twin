"""Per-channel dynamic threshold calibration from healthy validation residuals.

Computes mean (μ_c), std (σ_c), and Upper Control Limit (τ_c = μ_c + k·σ_c)
for each sensor channel using the LSTM Autoencoder applied to healthy val data.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader


CHANNEL_NAMES: List[str] = [
    "Sensor_Abutment_Left",
    "Sensor_Pier_Left",
    "Sensor_MidSpan_Girder",
    "Sensor_Pier_Right",
    "Sensor_Ambient_Temp",
]


@dataclass
class ChannelThreshold:
    """Calibrated statistical control limits for a single sensor channel."""

    name: str
    mu: float       # baseline mean residual
    sigma: float    # baseline residual standard deviation
    tau: float      # Upper Control Limit  τ = μ + k·σ
    warn: float     # Warning level        = μ + 2·σ
    k: float        # multiplier used


@dataclass
class ThresholdConfig:
    """Full calibrated threshold configuration for all channels."""

    calibration_timestamp: float
    k_multiplier: float
    channels: Dict[str, ChannelThreshold]

    def to_json(self) -> dict:
        out = {
            "calibration_timestamp": self.calibration_timestamp,
            "k_multiplier": self.k_multiplier,
            "channels": {
                name: asdict(ch) for name, ch in self.channels.items()
            },
        }
        return out

    def save(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def from_json(cls, data: dict) -> "ThresholdConfig":
        channels = {
            name: ChannelThreshold(**ch_data)
            for name, ch_data in data["channels"].items()
        }
        return cls(
            calibration_timestamp=data["calibration_timestamp"],
            k_multiplier=data["k_multiplier"],
            channels=channels,
        )


class ThresholdCalibrator:
    """Calibrates per-channel dynamic UCL thresholds from healthy validation data.

    Parameters
    ----------
    model     : Trained LSTMAutoencoder (eval mode).
    device    : Torch device for inference.
    k         : UCL multiplier (default 3.0 → 99.73% Gaussian coverage).
    channel_names : Ordered list of sensor channel names (C entries).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        k: float = 3.0,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.k = k
        self.channel_names = channel_names or CHANNEL_NAMES

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _per_channel_mse_batch(
        x: torch.Tensor, x_hat: torch.Tensor
    ) -> np.ndarray:
        """Time-averaged MSE per window per channel.

        Args:
            x     : (B, T, C) ground-truth
            x_hat : (B, T, C) reconstruction

        Returns:
            (B, C) float32 MSE matrix
        """
        # (B, T, C) → mean over T → (B, C)
        return ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()  # (B, C)

    # ──────────────────────────────────────────────────────────────────────
    def calibrate(self, val_loader: DataLoader) -> ThresholdConfig:
        """Forward-pass over val_loader and derive per-channel UCL thresholds.

        Args:
            val_loader: Healthy validation DataLoader.

        Returns:
            Populated ThresholdConfig dataclass.
        """
        self.model.eval()
        residual_buffers: List[np.ndarray] = []  # accumulates (B, C) batches

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
                x_hat, _ = self.model(x)            # (B, T, C)
                residual_buffers.append(
                    self._per_channel_mse_batch(x, x_hat)
                )

        # Stack → (M_val, C)
        all_residuals = np.concatenate(residual_buffers, axis=0)

        channels: Dict[str, ChannelThreshold] = {}
        for c, name in enumerate(self.channel_names):
            col = all_residuals[:, c]
            mu = float(np.mean(col))
            sigma = float(np.std(col, ddof=1))  # unbiased sample std
            tau = mu + self.k * sigma
            warn = mu + 2.0 * sigma
            channels[name] = ChannelThreshold(
                name=name, mu=mu, sigma=sigma, tau=tau, warn=warn, k=self.k
            )

        config = ThresholdConfig(
            calibration_timestamp=time.time(),
            k_multiplier=self.k,
            channels=channels,
        )
        self._config = config
        self._all_residuals = all_residuals
        return config

    # ──────────────────────────────────────────────────────────────────────
    def save(
        self,
        artifact_dir: str = "artifacts",
        filename: str = "threshold_config.json",
    ) -> Path:
        """Persist the threshold config as JSON.

        Returns:
            Path to the saved file.
        """
        p = Path(artifact_dir)
        p.mkdir(parents=True, exist_ok=True)
        out_path = p / filename
        with open(out_path, "w") as f:
            json.dump(self._config.to_json(), f, indent=2)
        return out_path

    @staticmethod
    def load(path: str) -> ThresholdConfig:
        """Load a previously persisted ThresholdConfig from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return ThresholdConfig.from_json(data)

    # ──────────────────────────────────────────────────────────────────────
    def plot_distributions(self, output_dir: str = "reports/phase5") -> None:
        """Save per-channel residual histograms with Gaussian overlays and τ lines.

        Must be called after :meth:`calibrate`.
        """
        assert hasattr(self, "_all_residuals"), \
            "Call calibrate() before plot_distributions()."

        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)

        n_ch = len(self.channel_names)
        fig, axes = plt.subplots(1, n_ch, figsize=(4 * n_ch, 4), sharey=False)
        if n_ch == 1:
            axes = [axes]

        for c, (name, ax) in enumerate(zip(self.channel_names, axes)):
            col = self._all_residuals[:, c]
            ch = self._config.channels[name]

            ax.hist(col, bins=50, density=True, color="steelblue",
                    alpha=0.6, label="Val residuals")

            # Fitted Gaussian overlay (guard against zero-variance channels)
            x_fit = np.linspace(col.min(), col.max(), 300)
            if ch.sigma > 1e-9:
                ax.plot(x_fit, norm.pdf(x_fit, ch.mu, ch.sigma),
                        "k--", lw=1.5, label="Gaussian fit")
            else:
                ax.axvline(ch.mu, color="black", lw=1.5, linestyle="--",
                           label="Gaussian fit (zero-var)")

            # Control lines
            ax.axvline(ch.warn, color="orange", lw=1.2,
                       linestyle=":", label=f"Warn (2σ={ch.warn:.4f})")
            ax.axvline(ch.tau,  color="crimson", lw=1.5,
                       linestyle="--", label=f"UCL τ={ch.tau:.4f}")

            ax.set_title(name.replace("Sensor_", ""), fontsize=9)
            ax.set_xlabel("MSE residual")
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.4)

        fig.suptitle("Per-Channel Residual Distributions & Dynamic UCL Thresholds",
                     fontsize=11)
        fig.tight_layout()
        outfile = p / "threshold_distributions.png"
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outfile}")
