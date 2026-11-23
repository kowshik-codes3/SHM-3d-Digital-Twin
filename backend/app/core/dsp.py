"""Digital Signal Processing (DSP) and spectral analysis utilities for SHM."""

from typing import Tuple, Union
import numpy as np
import pandas as pd
import scipy.signal


def apply_bandpass_filter(
    data: Union[np.ndarray, pd.Series],
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter.
    
    Args:
        data: 1D array or Series representing the signal.
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        order: Filter order (will be zero-phase, so effective order is doubled).
        
    Returns:
        Filtered 1D numpy array.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Scipy design bandpass filter
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    # Zero-phase filtfilt to maintain exact temporal correspondence with no delay
    return scipy.signal.filtfilt(b, a, data)


def detrend_signal(data: Union[np.ndarray, pd.Series], type: str = 'linear') -> np.ndarray:
    """
    Removes linear trend or constant offset from a signal.
    
    Args:
        data: 1D time-series data.
        type: Trend type to remove ('linear' or 'constant').
        
    Returns:
        Detrended 1D numpy array.
    """
    return scipy.signal.detrend(data, type=type)


def compute_fft(data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the real Fast Fourier Transform (FFT).
    
    Args:
        data: 1D real-valued signal.
        fs: Sampling frequency in Hz.
        
    Returns:
        Tuple of (frequencies, spectral_magnitudes).
    """
    n = len(data)
    fft_vals = np.fft.rfft(data)
    fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
    # Normalized magnitude
    magnitudes = np.abs(fft_vals) / n
    # Multiply by 2 for real single-sided spectrum (excluding DC and Nyquist components)
    magnitudes[1:-1] *= 2.0
    return fft_freqs, magnitudes


def compute_psd(data: np.ndarray, fs: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Power Spectral Density (PSD) using Welch's method.
    
    Args:
        data: 1D signal.
        fs: Sampling frequency in Hz.
        nperseg: Length of each segment. Default is 1024.
        
    Returns:
        Tuple of (frequencies, PSD values).
    """
    freqs, psd_vals = scipy.signal.welch(data, fs=fs, nperseg=nperseg)
    return freqs, psd_vals
