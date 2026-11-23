"""Slow-drift layer for hourly mean residuals and operational modal frequency tracking.

Implements Stage 2 Item (h):
1. Hourly mean residuals tracking with sequential EWMA & CUSUM change detection.
2. Hourly modal frequency extraction (via Welch PSD peak-picking) and regression vs temperature f(T).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class DriftAnomalyReport:
    """Detection report for slow-drift structural changes over hourly scales."""

    hour_index: int
    timestamp: float
    mean_residuals: Dict[str, float]
    ewma_values: Dict[str, float]
    cusum_pos: Dict[str, float]
    cusum_neg: Dict[str, float]
    cusum_alarm: bool
    tracked_modes: Dict[str, float]      # e.g., {'mode1_freq': 3.75, 'mode2_freq': 4.40}
    temp_residual_hz: Optional[float]    # f_observed - f_model(T)
    slow_drift_alarm: bool


class ResidualEWMA_CUSUM:
    """Sequential change detection on hourly mean residuals using EWMA and CUSUM.

    Parameters:
        lambda_ewma: Weight of current residual in EWMA (default 0.2).
        cusum_k: Slack / reference value in units of std (default 0.5 * sigma).
        cusum_h: Decision threshold in units of std (default 4.0 * sigma).
    """

    def __init__(
        self,
        channel_names: List[str],
        baseline_means: Optional[Dict[str, float]] = None,
        baseline_stds: Optional[Dict[str, float]] = None,
        lambda_ewma: float = 0.2,
        cusum_k_factor: float = 0.5,
        cusum_h_factor: float = 4.0,
    ) -> None:
        self.channel_names = channel_names
        self.lambda_ewma = lambda_ewma
        self.k_factor = cusum_k_factor
        self.h_factor = cusum_h_factor

        self.mu = baseline_means or {ch: 0.0 for ch in channel_names}
        self.sigma = baseline_stds or {ch: 1.0 for ch in channel_names}

        # Rolling states
        self.ewma = {ch: self.mu[ch] for ch in channel_names}
        self.cusum_pos = {ch: 0.0 for ch in channel_names}
        self.cusum_neg = {ch: 0.0 for ch in channel_names}

    def update(self, hourly_residuals: Dict[str, float]) -> Tuple[bool, Dict[str, float], Dict[str, float]]:
        """Processes one hourly mean residual observation.

        Returns:
            (alarm_triggered, cusum_pos_dict, ewma_dict)
        """
        alarm = False
        for ch in self.channel_names:
            x = hourly_residuals.get(ch, self.mu[ch])
            # EWMA
            self.ewma[ch] = self.lambda_ewma * x + (1.0 - self.lambda_ewma) * self.ewma[ch]

            # CUSUM
            sig = max(self.sigma[ch], 1e-6)
            k = self.k_factor * sig
            h = self.h_factor * sig

            diff = x - self.mu[ch]
            self.cusum_pos[ch] = max(0.0, self.cusum_pos[ch] + diff - k)
            self.cusum_neg[ch] = max(0.0, self.cusum_neg[ch] - diff - k)

            if self.cusum_pos[ch] > h or self.cusum_neg[ch] > h:
                alarm = True

        return alarm, dict(self.cusum_pos), dict(self.ewma)


class ModalFrequencyTracker:
    """Operational Modal Analysis (OMA) peak-picking and thermal baseline regression.

    Tracks bridge fundamental frequencies (f1 ~3.4-4.0 Hz, f2 ~4.0-4.8 Hz)
    and evaluates the temperature-adjusted modal shift Delta f = f_obs - f_hat(T).
    """

    def __init__(self, fs: float = 50.0) -> None:
        self.fs = fs
        # Fitted linear model f1(T) = a * T + b
        self.temp_slope: float = 0.0
        self.temp_intercept: float = 3.80
        self.residual_std: float = 0.05
        self.is_fitted: bool = False

    def extract_frequencies(self, signal_window: np.ndarray) -> Dict[str, float]:
        """Extracts peak resonant frequencies from an accelerometer recording.

        Args:
            signal_window: 1D array of vibration signal.

        Returns:
            Dict containing detected mode frequencies in Hz.
        """
        if len(signal_window) < 128:
            return {"mode1_hz": 3.8, "mode2_hz": 4.4}

        freqs, psd = signal.welch(
            signal_window,
            fs=self.fs,
            nperseg=min(len(signal_window), 512),
            noverlap=min(len(signal_window) // 2, 256),
        )

        # Mode 1 search range: 3.2 - 4.1 Hz
        m1_mask = (freqs >= 3.2) & (freqs <= 4.1)
        if np.any(m1_mask):
            f1 = float(freqs[m1_mask][np.argmax(psd[m1_mask])])
        else:
            f1 = 3.80

        # Mode 2 search range: 4.1 - 4.9 Hz
        m2_mask = (freqs > 4.1) & (freqs <= 4.9)
        if np.any(m2_mask):
            f2 = float(freqs[m2_mask][np.argmax(psd[m2_mask])])
        else:
            f2 = 4.40

        return {"mode1_hz": f1, "mode2_hz": f2}

    def fit_thermal_model(self, temperatures: np.ndarray, f1_frequencies: np.ndarray) -> None:
        """Fits f1(T) = a * T + b on healthy training data."""
        if len(temperatures) >= 3:
            p = np.polyfit(temperatures, f1_frequencies, deg=1)
            self.temp_slope = float(p[0])
            self.temp_intercept = float(p[1])
            preds = np.polyval(p, temperatures)
            self.residual_std = float(np.std(f1_frequencies - preds))
            self.is_fitted = True
        else:
            self.temp_slope = -0.005  # Standard physical negative thermal stiffening
            self.temp_intercept = 3.85
            self.residual_std = 0.03
            self.is_fitted = True

    def check_modal_anomaly(self, temperature: float, f1_obs: float, n_sigma: float = 3.0) -> Tuple[bool, float]:
        """Evaluates whether the observed modal frequency exhibits an abnormal shift.

        Returns:
            (is_anomalous, frequency_residual_hz)
        """
        f_expected = self.temp_slope * temperature + self.temp_intercept
        delta_f = f1_obs - f_expected
        # A significant structural stiffness reduction causes frequency drop (negative delta_f)
        threshold = n_sigma * self.residual_std
        is_anomalous = bool(abs(delta_f) > threshold)
        return is_anomalous, float(delta_f)
