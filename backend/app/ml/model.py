"""LSTM Autoencoder architecture for unsupervised SHM anomaly detection.

Input / Output shape: (Batch, SeqLen=50, Channels=5)
Latent bottleneck dim: 32
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """2-layer stacked LSTM encoder compressing (B, T, 5) → latent z (B, 32).

    Architecture
    ------------
    Layer 1 : LSTM  input=5   → hidden=64,  dropout=0.1
    Layer 2 : LSTM  input=64  → hidden=32
    Output  : final hidden state h_T  shape (B, 32)
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 5)
        Returns:
            z: (B, 32) – final hidden state from layer 2
        """
        out1, _ = self.lstm1(x)               # (B, T, 64)
        out1 = self.dropout(out1)
        _, (h_n, _) = self.lstm2(out1)         # h_n: (1, B, 32)
        z = h_n.squeeze(0)                     # (B, 32)
        return z


class LSTMDecoder(nn.Module):
    """2-layer stacked LSTM decoder expanding latent z (B, 32) → (B, T, 5).

    Architecture
    ------------
    RepeatVector : tile z across T=50 steps → (B, T, 32)
    Layer 1      : LSTM  input=32 → hidden=64
    Layer 2      : LSTM  input=64 → hidden=64
    Projection   : TimeDistributed Linear  64 → 5
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden1: int = 64,
        hidden2: int = 64,
        output_size: int = 5,
        seq_len: int = 50,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            num_layers=1,
            batch_first=True,
        )
        # Time-distributed linear: applied across every time step independently
        self.output_proj = nn.Linear(hidden2, output_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 32)
        Returns:
            x_hat: (B, T, 5)
        """
        # RepeatVector: (B, 32) → (B, T, 32)
        z_rep = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, 32)

        out1, _ = self.lstm1(z_rep)             # (B, T, 64)
        out2, _ = self.lstm2(out1)              # (B, T, 64)
        x_hat = self.output_proj(out2)          # (B, T, 5)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """Full encoder-decoder LSTM autoencoder for SHM sequence reconstruction.

    Forward pass returns (reconstruction, latent_z) to allow downstream
    anomaly scoring directly from the latent space if desired.
    Supports conditional autoencoding where input_size > target_size
    (e.g. conditioning on temperature while only reconstructing vibration).
    """

    def __init__(
        self,
        input_size: int = 5,
        target_size: Optional[int] = None,
        enc_hidden1: int = 64,
        enc_hidden2: int = 32,
        dec_hidden1: int = 64,
        dec_hidden2: int = 64,
        dropout: float = 0.1,
        seq_len: int = 50,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size or input_size
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden1=enc_hidden1,
            hidden2=enc_hidden2,
            dropout=dropout,
        )
        self.decoder = LSTMDecoder(
            latent_dim=enc_hidden2,
            hidden1=dec_hidden1,
            hidden2=dec_hidden2,
            output_size=self.target_size,
            seq_len=seq_len,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C)
        Returns:
            x_hat: (B, T, C)  – reconstructed sequence
            z:     (B, 32)    – latent bottleneck vector
        """
        z = self.encoder(x)       # (B, 32)
        x_hat = self.decoder(z)   # (B, T, C)
        return x_hat, z
