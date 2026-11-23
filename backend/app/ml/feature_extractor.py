"""Module to extract statistical and frequency-domain features from 3D window tensors."""

import numpy as np
from scipy import stats
from typing import Tuple


class WindowFeatureExtractor:
    """Extracts time-domain and frequency-domain features from 3D window tensors."""
    
    @staticmethod
    def extract_features(windows: np.ndarray, fs: float = 50.0) -> np.ndarray:
        """
        Transforms 3D array (M, T, C) into 2D feature matrix (M, C * 8).
        
        Features per channel:
        1. Mean
        2. Standard deviation
        3. Peak-to-peak amplitude (ptp)
        4. RMS (Root Mean Square)
        5. Crest factor
        6. Skewness
        7. Kurtosis
        8. Peak energy in 0.5 - 5.0 Hz structural band (FFT)
        
        Args:
            windows: 3D numpy array of shape (M, T, C).
            fs: Sampling frequency in Hz. Default is 50.0.
            
        Returns:
            2D numpy array of shape (M, C * 8) in float32.
        """
        M, T, C = windows.shape
        
        # 1. Time-Domain Metrics
        mean = np.mean(windows, axis=1)                          # (M, C)
        std = np.std(windows, axis=1)                            # (M, C)
        p2p = np.ptp(windows, axis=1)                            # (M, C)
        rms = np.sqrt(np.mean(windows**2, axis=1))               # (M, C)
        crest = np.max(np.abs(windows), axis=1) / (rms + 1e-8)   # (M, C)
        skew = stats.skew(windows, axis=1)                       # (M, C)
        kurt = stats.kurtosis(windows, axis=1)                   # (M, C)
        
        # 2. Frequency-Domain Metrics (FFT Energy in 0.5 - 5.0 Hz band)
        # Using real FFT along temporal axis
        fft_vals = np.abs(np.fft.rfft(windows, axis=1))          # (M, T//2 + 1, C)
        freqs = np.fft.rfftfreq(T, d=1.0/fs)
        band_mask = (freqs >= 0.5) & (freqs <= 5.0)
        band_energy = np.sum(fft_vals[:, band_mask, :], axis=1)  # (M, C)
        
        # Concatenate all features: 8 features per channel -> (M, C, 8)
        stacked = np.stack([mean, std, p2p, rms, crest, skew, kurt, band_energy], axis=-1)
        
        # Flatten to (M, C * 8)
        return stacked.reshape(M, -1).astype(np.float32)
