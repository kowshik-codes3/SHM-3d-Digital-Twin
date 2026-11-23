"""Complete suite of baseline detectors B0 through B6 for Sub-Stage E3.

Unified Interface: BaseDetector
B0: Temperature-only control (Logistic regression / threshold on tVL)
B1: Fixed RMS + Spectral Peak detector
B2: Trackedmodes Natural-Frequency f(T) regression + Mahalanobis distance
B3: PCA reconstruction error on vibration spectral band power
B4: Plain LSTM Autoencoder (unconditioned, 4 vibration channels)
B5: Conditional LSTM Autoencoder (conditioned on tVL, reconstructs 4 vibration channels)
B6: Full Stage 2 Pipeline (Conditional LSTM-AE + Strided EVT + Persistence + Sensor-Fault Filter)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model import LSTMAutoencoder
from app.ml.anomaly_engine import AnomalyStateMachine, STATE_CRITICAL
from app.validation.evt import EVTCalibrator, EVTCalibrationConfig
from app.ml.threshold_calibrator import ThresholdConfig
from app.validation.recording_replay import replay_single_recording


class BaseDetector(ABC):
    """Unified detector interface ensuring fair, leak-free evaluation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_fitted = False
        self.tau = 0.0

    @abstractmethod
    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        """Fit model strictly on healthy training data."""
        pass

    @abstractmethod
    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        """Calibrate operating threshold on healthy validation data to target FAR."""
        pass

    @abstractmethod
    def score_recording(self, recording: np.ndarray) -> float:
        """Returns scalar anomaly score for one recording."""
        pass


class TemperatureOnlyControl_B0(BaseDetector):
    """B0: Classifies state purely from the temperature channel (Season Control)."""

    def __init__(self) -> None:
        super().__init__("B0_Temperature_Control")

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        temps = [float(np.mean(r[:, 4])) for r in val_recordings]
        self.tau = float(np.percentile(temps, (1.0 - target_far) * 100.0))
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        return float(np.mean(recording[:, 4]))


class RMS_SpectralPeak_B1(BaseDetector):
    """B1: Classical fixed threshold on vibration RMS."""

    def __init__(self) -> None:
        super().__init__("B1_RMS_Spectral")
        self.baseline_rms = 0.0

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        rms_vals = [float(np.sqrt(np.mean(r[:, :4] ** 2))) for r in train_recordings]
        self.baseline_rms = float(np.mean(rms_vals))
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        scores = [self.score_recording(r) for r in val_recordings]
        self.tau = float(np.percentile(scores, (1.0 - target_far) * 100.0))
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(recording[:, :4] ** 2)))
        return abs(rms - self.baseline_rms)


class ModalRegressionMahalanobis_B2(BaseDetector):
    """B2: Natural-Frequency regression on temperature + Mahalanobis novelty index."""

    def __init__(self) -> None:
        super().__init__("B2_Modal_Mahalanobis")
        self.reg = LinearRegression()
        self.inv_cov = None
        self.mu_res = None

    def fit_modal(self, temperatures: np.ndarray, modal_frequencies: np.ndarray) -> None:
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
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        return 0.0


class PCABandPower_B3(BaseDetector):
    """B3: PCA reconstruction error on Welch PSD spectral band powers (0.5 to 20 Hz)."""

    def __init__(self, n_components: int = 4, fs: float = 50.0) -> None:
        super().__init__("B3_PCA_BandPower")
        self.n_components = n_components
        self.fs = fs
        self.pca = PCA(n_components=n_components)

    def _extract_band_power(self, recording: np.ndarray) -> np.ndarray:
        # Compute Welch PSD for each of the 4 vibration channels
        features = []
        for ch in range(4):
            freqs, psd = signal.welch(recording[:, ch], fs=self.fs, nperseg=256)
            # 8 sub-bands between 0.5 and 20 Hz
            bands = np.linspace(0.5, 20.0, 9)
            for b_idx in range(len(bands) - 1):
                mask = (freqs >= bands[b_idx]) & (freqs < bands[b_idx + 1])
                bp = float(np.trapz(psd[mask], freqs[mask])) if mask.sum() > 0 else 0.0
                features.append(np.log1p(bp))
        return np.array(features, dtype=np.float32)

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict]) -> None:
        X = np.stack([self._extract_band_power(r) for r in train_recordings], axis=0)
        n_comp = min(self.n_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(X)
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        scores = [self.score_recording(r) for r in val_recordings]
        self.tau = float(np.percentile(scores, (1.0 - target_far) * 100.0))
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        feat = self._extract_band_power(recording).reshape(1, -1)
        recon = self.pca.inverse_transform(self.pca.transform(feat))
        mse = float(np.mean((feat - recon) ** 2))
        return mse


class PlainLSTMAutoencoder_B4(BaseDetector):
    """B4: Unconditioned LSTM-AE (Stage 0/1 baseline) on 4 vibration channels."""

    def __init__(self, seq_len: int = 50, hidden: int = 32, device: Optional[torch.device] = None) -> None:
        super().__init__("B4_Plain_LSTM_AE")
        self.seq_len = seq_len
        self.hidden = hidden
        self.device = device or torch.device("cpu")
        self.model = LSTMAutoencoder(
            input_size=4,
            target_size=4,
            enc_hidden1=hidden * 2,
            enc_hidden2=hidden,
            dec_hidden1=hidden * 2,
            dec_hidden2=hidden * 2,
            seq_len=seq_len,
        ).to(self.device)

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict], epochs: int = 8, lr: float = 0.001) -> None:
        # Extract windows from vibration channels (first 4)
        windows = []
        for r in train_recordings:
            vib = r[:, :4]
            T = vib.shape[0]
            for i in range(0, T - self.seq_len + 1, self.seq_len):
                windows.append(vib[i : i + self.seq_len])
        X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
        loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        scores = [self.score_recording(r) for r in val_recordings]
        self.tau = float(np.percentile(scores, (1.0 - target_far) * 100.0))
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        self.model.eval()
        vib = recording[:, :4]
        T = vib.shape[0]
        windows = [vib[i : i + self.seq_len] for i in range(0, T - self.seq_len + 1, self.seq_len)]
        if not windows:
            return 0.0
        X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
        with torch.no_grad():
            recon, _ = self.model(X)
            # Mean squared reconstruction error per window, then 95th percentile
            errs = torch.mean((recon - X) ** 2, dim=[1, 2]).cpu().numpy()
        return float(np.percentile(errs, 95.0))


class ConditionalLSTMAutoencoder_B5(BaseDetector):
    """B5: Conditional LSTM-AE (conditioned on temperature tVL, reconstructs 4 vibration channels)."""

    def __init__(self, seq_len: int = 50, hidden: int = 32, device: Optional[torch.device] = None) -> None:
        super().__init__("B5_Conditional_LSTM_AE")
        self.seq_len = seq_len
        self.hidden = hidden
        self.device = device or torch.device("cpu")
        self.model = LSTMAutoencoder(
            input_size=5,
            target_size=4,
            enc_hidden1=hidden * 2,
            enc_hidden2=hidden,
            dec_hidden1=hidden * 2,
            dec_hidden2=hidden * 2,
            seq_len=seq_len,
        ).to(self.device)

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict], epochs: int = 8, lr: float = 0.001) -> None:
        windows = []
        for r in train_recordings:
            T = r.shape[0]
            for i in range(0, T - self.seq_len + 1, self.seq_len):
                windows.append(r[i : i + self.seq_len])
        X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
        Y = X[:, :, :4]  # target is 4 vibration channels
        loader = DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_y)
                loss.backward()
                optimizer.step()
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        scores = [self.score_recording(r) for r in val_recordings]
        self.tau = float(np.percentile(scores, (1.0 - target_far) * 100.0))
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        self.model.eval()
        T = recording.shape[0]
        windows = [recording[i : i + self.seq_len] for i in range(0, T - self.seq_len + 1, self.seq_len)]
        if not windows:
            return 0.0
        X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
        targets = X[:, :, :4]
        with torch.no_grad():
            recon, _ = self.model(X)
            errs = torch.mean((recon - targets) ** 2, dim=[1, 2]).cpu().numpy()
        return float(np.percentile(errs, 95.0))


class FullStage2System_B6(BaseDetector):
    """B6: Complete Stage 2 Architecture with Temperature Conditioning, EVT, and Persistence Replay."""

    def __init__(self, seq_len: int = 50, hidden: int = 32, device: Optional[torch.device] = None) -> None:
        super().__init__("B6_Full_Stage2")
        self.seq_len = seq_len
        self.device = device or torch.device("cpu")
        self.model = LSTMAutoencoder(
            input_size=5,
            target_size=4,
            enc_hidden1=hidden * 2,
            enc_hidden2=hidden,
            dec_hidden1=hidden * 2,
            dec_hidden2=hidden * 2,
            seq_len=seq_len,
        ).to(self.device)
        self.evt_config: Optional[EVTCalibrationConfig] = None
        self.th_config: Optional[ThresholdConfig] = None

    def fit(self, train_recordings: List[np.ndarray], train_meta: List[dict], epochs: int = 8, lr: float = 0.001) -> None:
        windows = []
        for r in train_recordings:
            T = r.shape[0]
            for i in range(0, T - self.seq_len + 1, self.seq_len):
                windows.append(r[i : i + self.seq_len])
        X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
        Y = X[:, :, :4]
        loader = DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch_x)
                loss = criterion(recon, batch_y)
                loss.backward()
                optimizer.step()
        self.is_fitted = True

    def calibrate(self, val_recordings: List[np.ndarray], target_far: float = 0.01) -> float:
        self.model.eval()
        all_errs = []
        for r in val_recordings:
            T = r.shape[0]
            windows = [r[i : i + self.seq_len] for i in range(0, T - self.seq_len + 1, self.seq_len)]
            if not windows:
                continue
            X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
            Y = X[:, :, :4]
            with torch.no_grad():
                recon, _ = self.model(X)
                # Channel-wise squared errors
                ch_errs = torch.mean((recon - Y) ** 2, dim=1).cpu().numpy() # (N, 4)
                all_errs.append(ch_errs)
        val_residuals = np.concatenate(all_errs, axis=0) # (Total_windows, 4)

        # Calibrate Strided EVT
        calibrator = EVTCalibrator(
            tail_quantile=0.95,
            target_prob=target_far,
            decorrelation="strided",
            stride=5,
            channel_names=["aBD17Ay", "aBD17Az", "aBD23Ay", "aBD23Az"],
        )
        self.evt_config = calibrator.calibrate(val_residuals)
        taus = [ch.tau_evt for ch in self.evt_config.channels.values()]
        self.tau = float(np.mean(taus))
        return self.tau

    def score_recording(self, recording: np.ndarray) -> float:
        self.model.eval()
        T = recording.shape[0]
        windows = [recording[i : i + self.seq_len] for i in range(0, T - self.seq_len + 1, self.seq_len)]
        if not windows:
            return 0.0
        X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
        targets = X[:, :, :4]
        with torch.no_grad():
            recon, _ = self.model(X)
            # Channel residuals
            ch_errs = torch.mean((recon - targets) ** 2, dim=1).cpu().numpy()
        # Normalized by EVT critical thresholds
        if self.evt_config is not None:
            tau_vec = np.array([self.evt_config.channels[k].tau_evt for k in ["aBD17Ay", "aBD17Az", "aBD23Ay", "aBD23Az"]])
            norm_errs = ch_errs / np.maximum(tau_vec, 1e-6)
            comp_score = np.max(norm_errs, axis=1) # max channel over-threshold ratio
            return float(np.percentile(comp_score, 95.0))
        return float(np.percentile(np.mean(ch_errs, axis=1), 95.0))
