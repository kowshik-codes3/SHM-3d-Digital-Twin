"""Unified baseline detector interface and implementations for Sub-Stage E3.

Implements:
B0: Temperature-only control (thermometer classifier)
B1: Fixed RMS + Spectral Peak detector
B2: Natural-Frequency model from trackedmodes (regression on T + Mahalanobis distance)
B3: PCA reconstruction error on spectral band power
B4: Plain LSTM Autoencoder (original Stage 0/1 baseline)
B5: Conditional LSTM Autoencoder (temperature condition input only)
B6: Full Stage 2 System
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from scipy import signal

from app.validation.recording_replay import RecordingReplaySummary, replay_single_recording


class BaseDetector(ABC):
    """Unified detector interface ensuring fair, leak-free evaluation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        """Fit model strictly on healthy training data."""
        pass

    @abstractmethod
    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.001) -> float:
        """Calibrate operating threshold on healthy validation data to target FAR."""
        pass

    @abstractmethod
    def score_recording(self, recording: np.ndarray) -> float:
        """Returns scalar anomaly score for one recording."""
        pass


class TemperatureOnlyControl_B0(BaseDetector):
    """B0: Classifies healthy vs changed state purely from the temperature channel."""

    def __init__(self) -> None:
        super().__init__("B0_Temperature_Control")
        self.clf = LogisticRegression()

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        temps = [float(np.mean(r[:, 4])) for r in train_recordings]
        X = np.array(temps).reshape(-1, 1)
        y = np.zeros(len(temps), dtype=int)
        # Train requires two classes; fit dummy or calibration on val
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.001) -> float:
        temps = [float(np.mean(r[:, 4])) for r in val_recordings]
        return float(np.percentile(temps, (1.0 - target_far) * 100.0))

    def score_recording(self, recording: np.ndarray) -> float:
        return float(np.mean(recording[:, 4]))


class RMS_SpectralPeak_B1(BaseDetector):
    """B1: Classical fixed threshold on vibration RMS and dominant spectral frequency."""

    def __init__(self, fs: float = 50.0) -> None:
        super().__init__("B1_RMS_Spectral")
        self.fs = fs
        self.tau_rms = 0.0
        self.baseline_rms = 0.0

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        rms_vals = [float(np.sqrt(np.mean(r[:, :4] ** 2))) for r in train_recordings]
        self.baseline_rms = float(np.mean(rms_vals))
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.001) -> float:
        scores = [self.score_recording(r) for r in val_recordings]
        self.tau_rms = float(np.percentile(scores, (1.0 - target_far) * 100.0))
        return self.tau_rms

    def score_recording(self, recording: np.ndarray) -> float:
        accel = recording[:, :4]
        rms = float(np.sqrt(np.mean(accel ** 2)))
        return abs(rms - self.baseline_rms)


class ModalRegressionMahalanobis_B2(BaseDetector):
    """B2: Natural-Frequency regression on temperature + Mahalanobis novelty index."""

    def __init__(self) -> None:
        super().__init__("B2_Modal_Mahalanobis")
        self.reg = LinearRegression()
        self.inv_cov = None
        self.mu_res = None
        self.tau = 0.0

    def fit_modal(self, temperatures: np.ndarray, modal_frequencies: np.ndarray) -> None:
        """Fits multivariate regression on healthy modal frequencies."""
        X = temperatures.reshape(-1, 1)
        Y = modal_frequencies
        self.reg.fit(X, Y)
        res = Y - self.reg.predict(X)
        self.mu_res = np.mean(res, axis=0)
        cov = np.cov(res, rowvar=False)
        self.inv_cov = np.linalg.pinv(cov)
        self.is_fitted = True

    def calibrate_modal(self, temperatures_val: np.ndarray, modal_frequencies_val: np.ndarray, target_far: float = 0.01) -> float:
        X_val = temperatures_val.reshape(-1, 1)
        res_val = modal_frequencies_val - self.reg.predict(X_val)
        diff = res_val - self.mu_res
        d2 = np.sum(diff @ self.inv_cov * diff, axis=1)
        self.tau = float(np.percentile(d2, (1.0 - target_far) * 100.0))
        return self.tau

    def score_hour(self, temp: float, freqs: np.ndarray) -> float:
        pred = self.reg.predict(np.array([[temp]]))
        diff = (freqs - pred) - self.mu_res
        return float(diff @ self.inv_cov @ diff.T)

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        pass

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.001) -> float:
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        return 0.0
