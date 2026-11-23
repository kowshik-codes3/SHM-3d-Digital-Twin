"""Sub-Stage F4: Streaming Spectral Tracking Detector ("S1").

Direct physical vibration frequency tracking from raw streaming sensor data.
Unified with BaseDetector interface for plug-and-play evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
from sklearn.linear_model import HuberRegressor, LinearRegression

from app.evaluation.detectors.suite import BaseDetector


class SpectralTracker_S1(BaseDetector):
    """S1: Streaming Spectral Tracking Detector.
    
    Extracts bridge natural frequencies from streaming raw acceleration records via
    Welch PSD and parabolic sub-bin peak interpolation, compensates for ambient temperature
    effects using robust regression fitted on healthy-train, and flags stiffness degradation
    via multivariate Mahalanobis distance with strided calibration and persistence.
    """

    def __init__(
        self,
        fs: float = 50.0,
        nperseg: int = 2048,
        window_sec: float = 300.0,
        temp_model_type: str = "piecewise_freezing", # 'linear' or 'piecewise_freezing'
        target_far: float = 0.01,
        persistence_k: int = 2,
    ) -> None:
        super().__init__("S1_Spectral_Tracker")
        self.fs = fs
        self.nperseg = nperseg
        self.window_sec = window_sec
        self.temp_model_type = temp_model_type
        self.target_far = target_far
        self.persistence_k = persistence_k

        # Frequency search bands (Hz)
        self.bands = [
            {"mode": 1, "band": (0.48, 0.55), "channels": "az"}, # Mode 1: vertical
            {"mode": 2, "band": (1.18, 1.28), "channels": "ay"}, # Mode 2: lateral
            {"mode": 3, "band": (1.80, 2.00), "channels": "az"}, # Mode 3: vertical
            {"mode": 4, "band": (2.35, 2.52), "channels": "az"}, # Mode 4: vertical
        ]

        # Learned model parameters
        self.regressors: List[Union[LinearRegression, HuberRegressor]] = []
        self.cov: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.tau: float = 0.0

        # Streaming ring buffer
        self.buffer_size = int(window_sec * fs)
        self.ring_buffer = np.zeros((0, 5), dtype=np.float32)
        self.consecutive_alerts = 0

    def extract_frequencies(self, recording: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """Extracts sub-bin peak frequencies and temperature from recording.
        
        Returns:
            (freqs, temp, status_str):
            status_str in {'OK', 'SENSOR_FAULT_FLATLINE', 'SENSOR_FAULT_SPIKE', 'SENSOR_FAULT_NAN', 'INSUFFICIENT_DATA'}
        """
        if recording.shape[0] < self.nperseg:
            return None, None, "INSUFFICIENT_DATA"

        vib = recording[:, :4]
        temp = float(np.nanmean(recording[:, 4]))

        # 1. Sensor Fault Checks
        if np.isnan(vib).any() or np.isnan(temp):
            return None, None, "SENSOR_FAULT_NAN"
        
        std_per_ch = np.std(vib, axis=0)
        if (std_per_ch < 1e-5).any():
            return None, None, "SENSOR_FAULT_FLATLINE"
        
        peak_per_ch = np.max(np.abs(vib - np.mean(vib, axis=0)), axis=0)
        if (peak_per_ch > 12.0 * np.maximum(std_per_ch, 1e-6)).any():
            return None, None, "SENSOR_FAULT_SPIKE"

        # 2. Welch PSD computation
        # Channels: 0: Ay1, 1: Az1, 2: Ay2, 3: Az2
        ay_sig = 0.5 * (vib[:, 0] + vib[:, 2])
        az_sig = 0.5 * (vib[:, 1] + vib[:, 3])

        freqs_ay, psd_ay = signal.welch(ay_sig, fs=self.fs, nperseg=self.nperseg, window="hann")
        freqs_az, psd_az = signal.welch(az_sig, fs=self.fs, nperseg=self.nperseg, window="hann")

        extracted_f = []
        df_bin = freqs_az[1] - freqs_az[0]

        for b in self.bands:
            f_axis = freqs_az if b["channels"] == "az" else freqs_ay
            psd = psd_az if b["channels"] == "az" else psd_ay
            f_min, f_max = b["band"]
            
            idx = np.where((f_axis >= f_min) & (f_axis <= f_max))[0]
            if len(idx) == 0:
                extracted_f.append(0.5 * (f_min + f_max))
                continue
                
            k_max = idx[np.argmax(psd[idx])]
            
            # Sub-bin parabolic peak interpolation
            if 0 < k_max < len(psd) - 1:
                alpha, beta, gamma = psd[k_max - 1], psd[k_max], psd[k_max + 1]
                denom = alpha - 2.0 * beta + gamma
                if abs(denom) > 1e-12:
                    delta_k = 0.5 * (alpha - gamma) / denom
                    delta_k = float(np.clip(delta_k, -0.5, 0.5))
                else:
                    delta_k = 0.0
                f_refined = f_axis[0] + (k_max + delta_k) * df_bin
            else:
                f_refined = f_axis[k_max]
                
            extracted_f.append(float(f_refined))

        return np.array(extracted_f, dtype=np.float64), temp, "OK"

    def _prepare_temp_features(self, temps: np.ndarray) -> np.ndarray:
        """Constructs temperature basis features."""
        T = np.asarray(temps, dtype=np.float64).reshape(-1, 1)
        if self.temp_model_type == "piecewise_freezing":
            # Hinge term for freezing nonlinear stiffening below 0 deg C
            freezing_hinge = np.maximum(0.0, -T)
            return np.hstack([T, freezing_hinge])
        return T

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        """Fits robust temperature compensation models and baseline covariance strictly on healthy-train."""
        f_list, t_list = [], []
        for r in train_recordings:
            f_vec, temp, status = self.extract_frequencies(r)
            if status == "OK" and f_vec is not None:
                f_list.append(f_vec)
                t_list.append(temp)

        if len(f_list) < 10:
            raise ValueError(f"Insufficient valid recordings for S1 fit: {len(f_list)}")

        F_train = np.array(f_list) # (N, 4)
        T_train = np.array(t_list) # (N,)
        X_train = self._prepare_temp_features(T_train)

        self.regressors = []
        residuals = []
        for m in range(F_train.shape[1]):
            reg = HuberRegressor(epsilon=1.35, max_iter=200)
            reg.fit(X_train, F_train[:, m])
            self.regressors.append(reg)
            res_m = F_train[:, m] - reg.predict(X_train)
            residuals.append(res_m)

        R_train = np.column_stack(residuals) # (N, 4)
        self.cov = np.cov(R_train, rowvar=False)
        # Regularized inverse covariance to guarantee numerical stability
        reg_cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.pinv(reg_cov)
        self.is_fitted = True

    def _compute_mahalanobis(self, f_vec: np.ndarray, temp: float) -> float:
        """Computes squared Mahalanobis distance of temperature-compensated residuals."""
        X = self._prepare_temp_features(np.array([temp]))
        res = []
        for m, reg in enumerate(self.regressors):
            f_pred = float(reg.predict(X)[0])
            res.append(f_vec[m] - f_pred)
        r = np.array(res, dtype=np.float64)
        d2 = float(r @ self.inv_cov @ r)
        return d2

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        """Calibrates Mahalanobis distance threshold to target FAR strictly on healthy-val."""
        scores = []
        for r in val_recordings:
            s = self.score_recording(r)
            if s >= 0:
                scores.append(s)

        if not scores:
            raise ValueError("No valid scores generated on validation set")

        self.target_far = target_far
        # Calibrate to target FAR (e.g. 99th percentile for 1% FAR)
        self.tau = float(np.percentile(scores, (1.0 - target_far) * 100.0))
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        """Computes scalar novelty score for a single recording."""
        f_vec, temp, status = self.extract_frequencies(recording)
        if status != "OK" or f_vec is None or temp is None:
            return -1.0 # Sensor fault or missing data
        return self._compute_mahalanobis(f_vec, temp)

    def update(self, frame: np.ndarray) -> Dict:
        """Streaming state machine update on individual 1-second frames.
        
        Args:
            frame: (50, 5) acceleration and temperature frame.
        Returns:
            Dict containing state ('HEALTHY', 'CRITICAL', 'SENSOR_FAULT', 'WARMING_UP'),
            novelty score, threshold, and tracked modal residuals.
        """
        # Append frame to ring buffer
        if len(self.ring_buffer) == 0:
            self.ring_buffer = frame.copy()
        else:
            self.ring_buffer = np.vstack([self.ring_buffer, frame])
            if len(self.ring_buffer) > self.buffer_size:
                self.ring_buffer = self.ring_buffer[-self.buffer_size :]

        if len(self.ring_buffer) < self.buffer_size:
            return {
                "state": "WARMING_UP",
                "score": 0.0,
                "tau": self.tau,
                "consecutive_alerts": 0,
                "frequencies": None,
                "status_str": "BUFFERING",
            }

        # Analyze current rolling window
        f_vec, temp, status = self.extract_frequencies(self.ring_buffer)
        if status.startswith("SENSOR_FAULT"):
            return {
                "state": "SENSOR_FAULT",
                "score": -1.0,
                "tau": self.tau,
                "consecutive_alerts": 0,
                "frequencies": None,
                "status_str": status,
            }

        score = self._compute_mahalanobis(f_vec, temp)
        if score >= self.tau:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = max(0, self.consecutive_alerts - 1)

        state = "CRITICAL" if self.consecutive_alerts >= self.persistence_k else "HEALTHY"
        return {
            "state": state,
            "score": score,
            "tau": self.tau,
            "consecutive_alerts": self.consecutive_alerts,
            "frequencies": [round(x, 4) for x in f_vec],
            "temperature": round(temp, 2),
            "status_str": status,
        }
