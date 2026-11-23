"""High-performance circular sliding window buffer for streaming telemetry ingestion.

Buffers 1D multi-channel sensor frames into sliding windows of length T=50,
applies feature scaling, and constructs standardized PyTorch tensors (1, 50, C).
"""

from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .scaler_utils import StructuralDataScaler


class SensorStreamBuffer:
    """Circular buffer for real-time 1D multi-channel telemetry vectors.

    Parameters
    ----------
    window_size : Number of timesteps per window tensor (default 50).
    num_channels: Number of sensor channels (default 5).
    """

    def __init__(self, window_size: int = 50, num_channels: int = 5) -> None:
        self.window_size = window_size
        self.num_channels = num_channels
        self._buffer: deque = deque(maxlen=window_size)

    def push_frame(self, frame: Union[List[float], np.ndarray]) -> None:
        """Push a single 1D sensor vector into the circular buffer.

        Args:
            frame: 1D array/list of length num_channels (e.g. 5 entries).
        """
        arr = np.asarray(frame, dtype=np.float32)
        assert arr.shape == (self.num_channels,), (
            f"Expected frame shape ({self.num_channels},), got {arr.shape}"
        )
        self._buffer.append(arr)

    def is_full(self) -> bool:
        """Check if buffer has accumulated window_size timesteps."""
        return len(self._buffer) == self.window_size

    def get_raw_window(self) -> np.ndarray:
        """Extract unscaled raw window array of shape (T, C)."""
        assert self.is_full(), "Buffer is not full yet."
        return np.array(self._buffer, dtype=np.float32)  # (50, C)

    def get_standardized_tensor(
        self,
        scaler: StructuralDataScaler,
        device: torch.device,
    ) -> torch.Tensor:
        """Standardize current window and return PyTorch tensor (1, T, C).

        Args:
            scaler: Fitted StructuralDataScaler instance.
            device: Target torch device (CPU / CUDA).

        Returns:
            torch.Tensor of shape (1, 50, 5) float32.
        """
        raw_window = self.get_raw_window()  # (50, 5)
        # StructuralDataScaler transforms (50, 5) matrix directly
        scaled_window = scaler.transform(raw_window)  # (50, 5)
        tensor = torch.tensor(scaled_window, dtype=torch.float32, device=device)
        return tensor.unsqueeze(0)  # (1, 50, 5)

    def reset(self) -> None:
        """Clear all buffered timesteps."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
