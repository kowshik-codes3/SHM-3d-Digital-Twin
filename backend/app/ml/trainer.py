"""Class-based training engine for LSTMAutoencoder with early stopping and checkpointing."""

from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class Trainer:
    """Manages the full training lifecycle for the LSTMAutoencoder.

    Features
    --------
    - AdamW optimizer with weight decay.
    - ReduceLROnPlateau scheduler monitoring validation loss.
    - Gradient clipping (max_norm=1.0) to stabilise BPTT.
    - Early stopping on validation loss plateau.
    - Best-checkpoint saving (lowest val loss).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 60,
        early_stop_patience: int = 10,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        max_grad_norm: float = 1.0,
        checkpoint_path: str = "artifacts/lstm_autoencoder_best.pt",
        device: Optional[str] = None,
    ) -> None:
        # ── Device auto-detection ──────────────────────────────────────────
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()

        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )

        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    # ──────────────────────────────────────────────────────────────────────
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        """Runs one full epoch; returns mean MSE loss."""
        self.model.train(train)
        total_loss = 0.0
        n_batches = 0

        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for batch in loader:
                x = batch.to(self.device)               # (B, T, C)

                if train:
                    self.optimizer.zero_grad()

                x_hat, _ = self.model(x)                # (B, T, C)
                loss = self.criterion(x_hat, x)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    # ──────────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Train model until max_epochs or early stopping.

        Args:
            train_loader: Healthy baseline training DataLoader.
            val_loader:   Healthy baseline validation DataLoader.

        Returns:
            history dict with train_loss, val_loss, lr lists.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"  Training on device: {self.device}")
        print(f"  Max epochs: {self.max_epochs}  |  Early-stop patience: {self.early_stop_patience}")
        print(f"  Checkpoint: {self.checkpoint_path}")

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss   = self._run_epoch(val_loader, train=False)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            # Checkpoint on improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{self.max_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Patience: {patience_counter}/{self.early_stop_patience}"
                )

            if patience_counter >= self.early_stop_patience:
                print(f"\n  Early stopping at epoch {epoch} (best val loss={best_val_loss:.6f})")
                break

        print(f"\n  Training complete. Best val loss: {best_val_loss:.6f}")
        return self.history
