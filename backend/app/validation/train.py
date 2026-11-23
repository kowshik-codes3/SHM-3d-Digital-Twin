"""Training pipeline for LSTM Autoencoder on real KW51 validation benchmark.

Supports standard autoencoding as well as Stage 2 Conditional Autoencoding
(e.g., conditioning on ambient temperature without reconstructing it).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import LSTMAutoencoder
from app.ml.threshold_calibrator import ThresholdCalibrator, ThresholdConfig
from app.validation.dataset import DatasetSplits
from app.validation.evt import EVTCalibrationConfig, EVTCalibrator


def train_autoencoder(
    splits: DatasetSplits,
    config: dict,
    model_name: str = "model_best.pt",
    device: Optional[torch.device] = None,
    target_size: Optional[int] = None,
    evt_decorrelation: str = "none",
) -> Tuple[LSTMAutoencoder, np.ndarray, EVTCalibrationConfig, ThresholdConfig]:
    """Trains LSTMAutoencoder on splits.windows_train and calibrates on splits.windows_val.

    Args:
        splits: Prepared DatasetSplits
        config: Validation configuration dict
        model_name: Filename for model weights
        device: Torch compute device
        target_size: If specified and < len(channels), trains a Conditional Autoencoder
                     reconstructing only the first target_size channels.
        evt_decorrelation: 'none', 'strided', or 'block_maxima'

    Returns:
        (trained_model, val_residuals, evt_config, threshold_config)
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_size = len(splits.channel_names)
    effective_target_size = target_size or input_size
    seq_len = config["windowing"]["window_size"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]
    lr = config["training"]["lr"]

    model = LSTMAutoencoder(
        input_size=input_size,
        target_size=effective_target_size,
        enc_hidden1=config["training"]["hidden1"],
        enc_hidden2=config["training"]["hidden2"],
        dec_hidden1=config["training"]["hidden1"],
        dec_hidden2=config["training"]["hidden1"],
        seq_len=seq_len,
    ).to(dev)

    train_ds = TensorDataset(splits.windows_train)
    val_ds = TensorDataset(splits.windows_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    print(f"Starting training: {epochs} epochs, batch_size={batch_size}, in={input_size}, target={effective_target_size} ({dev})...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(dev)
            target_x = batch_x[:, :, :effective_target_size]
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, target_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(splits.windows_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(dev)
                target_x = batch_x[:, :, :effective_target_size]
                recon, _ = model(batch_x)
                loss = criterion(recon, target_x)
                val_loss += loss.item() * len(batch_x)

        val_loss /= len(splits.windows_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            is_best = True
        else:
            patience_counter += 1
            is_best = False

        if epoch % 5 == 0 or is_best or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{' *' if is_best else ''}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.5f}")
            break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.to(dev)

    # Compute validation per-channel MSE residuals: shape (M_val, target_channels)
    model.eval()
    eval_channel_names = splits.channel_names[:effective_target_size]
    val_residual_list = []
    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(dev)
            target_x = batch_x[:, :, :effective_target_size]
            recon, _ = model(batch_x)
            mse_batch = ((target_x - recon) ** 2).mean(dim=1).cpu().numpy()
            val_residual_list.append(mse_batch)

    val_residuals = np.concatenate(val_residual_list, axis=0)  # (M_val, target_size)

    # Calibrate EVT + Percentile + SPC thresholds
    evt_calibrator = EVTCalibrator(
        tail_quantile=config["thresholds"]["evt_tail_quantile"],
        target_prob=config["thresholds"]["evt_target_prob"],
        decorrelation=evt_decorrelation,
        channel_names=eval_channel_names,
    )
    evt_config = evt_calibrator.calibrate(val_residuals, eval_channel_names)

    # Convert to production ThresholdConfig
    threshold_config = evt_config.to_threshold_config(threshold_type="evt")

    # Persist artifacts
    art_dir = Path(config["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), art_dir / model_name)
    splits.scaler.save(str(art_dir / "scaler.joblib"))
    evt_config.save(art_dir / "evt_calibration.json")
    threshold_config.save(str(art_dir / "threshold_config.json"))

    print(f"Artifacts successfully saved to {art_dir}")
    return model, val_residuals, evt_config, threshold_config
