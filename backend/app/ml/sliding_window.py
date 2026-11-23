"""High-speed vectorized sliding window generator using NumPy sliding_window_view."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def generate_sliding_windows(
    data: np.ndarray, 
    window_size: int = 50, 
    stride: int = 5
) -> np.ndarray:
    """
    Transforms 2D array (N, C) into 3D array (M, T, C) using zero-copy striding.
    
    Parameters:
        data: 2D array of shape (N, Channels)
        window_size: Length of sequence (T)
        stride: Step size (S)
        
    Returns:
        3D numpy array of shape (M, window_size, Channels) where float32 is enforced.
    """
    if data.ndim != 2:
        raise ValueError(f"Input data must be 2D (N, C), got shape {data.shape}")
        
    n_samples, n_channels = data.shape
    if n_samples < window_size:
        raise ValueError(f"Sample length ({n_samples}) must be >= window_size ({window_size})")

    # Fast sliding window across axis 0. Results in shape (M, Channels, window_size)
    windows = sliding_window_view(data, window_shape=window_size, axis=0)
    
    # Transpose to standard (M, window_size, Channels) matching sequence modeling inputs
    windows = np.swapaxes(windows, 1, 2)
    
    # Subsample according to stride
    if stride > 1:
        windows = windows[::stride]
        
    return np.ascontiguousarray(windows, dtype=np.float32)
