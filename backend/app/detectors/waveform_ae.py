"""Waveform Autoencoder Detector (Stage 2 LSTM-AE Pipeline).

DOCUMENTED SCOPE & LIMITATIONS:
- Retained strictly for abrupt, high-frequency transients, mechanical impact shocks,
  train passage waveforms, and hardware sensor faults.
- MATHEMATICALLY REFUTED & NOT CLAIMED for detecting small sub-bin natural frequency
  shifts or slow stiffness degradation (demonstrated 0.0% recall on KW51 shift under 10%).
- Runs at high frequency (default: 1.0 s cadence, 50-sample window at 50 Hz).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.detectors.base import BaseDetector, DetectorOutput, DetectorState, get_git_hash
from app.ml.anomaly_engine import AnomalyStateMachine, STATE_CRITICAL, STATE_NORMAL, STATE_SENSOR_FAULT, STATE_WARNING
from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.threshold_calibrator import ChannelThreshold, ThresholdCalibrator, ThresholdConfig


class WaveformAEDetector(BaseDetector):
    """Stage 2 Conditional LSTM Autoencoder Anomaly Detector."""

    SCOPE_STATEMENT = (
        "High-frequency transient anomaly and impact detector. Not capable of "
        "sub-bin natural frequency shift tracking."
    )

    def __init__(
        self,
        detector_id: str = "waveform_ae",
        version: str = "v1",
        update_cadence_seconds: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            detector_id=detector_id,
            version=version,
            update_cadence_seconds=update_cadence_seconds,
            params=params,
        )
        p = self.params
        self.seq_len: int = p.get("seq_len", 50)
        self.input_size: int = p.get("input_size", 5)
        self.target_size: int = p.get("target_size", 4)
        self.hidden: int = p.get("hidden", 32)
        device_str = p.get("device", "cpu")
        self.device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
        self.persistence_k: int = p.get("persistence_k", 3)
        self.t_persist: float = p.get("t_persist", 0.04)
        self.min_channels: int = int(p.get("min_channels", 2))
        self.required_channels: List[str] = list(p.get("required_channels", []))
        self._last_usable_mask: Optional[List[bool]] = None

        self.channel_names = channel_names or [f"ch_{i}" for i in range(self.input_size)]
        self.target_channel_names = self.channel_names[: self.target_size]

        # Model, Scaler, State Machine
        self.model: LSTMAutoencoder = LSTMAutoencoder(
            input_size=self.input_size,
            target_size=self.target_size,
            enc_hidden1=self.hidden * 2,
            enc_hidden2=self.hidden,
            dec_hidden1=self.hidden * 2,
            dec_hidden2=self.hidden * 2,
            seq_len=self.seq_len,
        ).to(self.device)

        self.scaler: Optional[StructuralDataScaler] = None
        self.threshold_config: Optional[ThresholdConfig] = None
        self.state_machine: Optional[AnomalyStateMachine] = None

        # Rolling raw buffer
        self.buffer: np.ndarray = np.zeros((0, self.input_size), dtype=np.float32)
        self.step_counter: int = 0

    def fit(
        self,
        train_recordings: List[np.ndarray],
        val_recordings: Optional[List[np.ndarray]] = None,
        train_meta: Optional[List[dict]] = None,
        epochs: int = 5,
        lr: float = 0.001,
        **kwargs: Any,
    ) -> None:
        """Fit scaler and LSTM autoencoder weights on healthy training recordings."""
        if not train_recordings:
            raise ValueError("train_recordings cannot be empty")

        # Fit Scaler
        stacked_raw = np.vstack(train_recordings)
        self.scaler = StructuralDataScaler(feature_names=self.channel_names)
        self.scaler.scaler.fit(stacked_raw)
        self.scaler.is_fitted = True

        # Extract sequence windows
        windows = []
        for r in train_recordings:
            T = r.shape[0]
            for i in range(0, T - self.seq_len + 1, self.seq_len):
                w = r[i : i + self.seq_len]
                w_scaled = self.scaler.transform(w)
                windows.append(w_scaled)

        if not windows:
            raise ValueError(f"No valid windows of length {self.seq_len} extracted")

        X = torch.tensor(np.array(windows, dtype=np.float32), device=self.device)
        Y = X[:, :, : self.target_size]
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                recon, _ = self.model(bx)
                loss = criterion(recon, by)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        self.healthy_hours_trained = float(stacked_raw.shape[0] / (self.params.get("fs", 50.0) * 3600.0))

        if val_recordings:
            self.calibrate(val_recordings)

    def calibrate(
        self,
        val_recordings: List[np.ndarray],
        target_far: float = 0.01,
        **kwargs: Any,
    ) -> float:
        """Calibrate channel thresholds on healthy validation recordings."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")

        self.model.eval()
        all_channel_mses = []
        with torch.no_grad():
            for r in val_recordings:
                T = r.shape[0]
                for i in range(0, T - self.seq_len + 1, self.seq_len):
                    w = r[i : i + self.seq_len]
                    w_scaled = self.scaler.transform(w)
                    bx = torch.tensor(w_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)
                    by = bx[:, :, : self.target_size]
                    recon, _ = self.model(bx)
                    mse_vec = torch.mean((recon - by) ** 2, dim=1).squeeze(0).cpu().numpy()
                    all_channel_mses.append(mse_vec)

        if not all_channel_mses:
            raise ValueError("No validation windows extracted for calibration")

        mse_matrix = np.array(all_channel_mses)  # (N_win, target_size)
        th_map: Dict[str, ChannelThreshold] = {}
        for idx, ch_name in enumerate(self.target_channel_names):
            pct_val = float(np.percentile(mse_matrix[:, idx], (1.0 - target_far) * 100.0))
            th_map[ch_name] = ChannelThreshold(
                channel_name=ch_name,
                tau=pct_val,
                tau_warning=pct_val * 0.75,
                lambda_baseline=float(np.mean(mse_matrix[:, idx])),
            )

        self.threshold_config = ThresholdConfig(channels=th_map)
        self.tau = float(np.mean([ct.tau for ct in th_map.values()]))
        self.state_machine = AnomalyStateMachine(
            config=self.threshold_config,
            persistence=self.persistence_k,
            t_persist=self.t_persist,
        )
        return self.tau

    def reset_state(self, clear_persistence: bool = True) -> None:
        """Reset sliding buffer and optionally anomaly state machine at recording boundaries."""
        self.buffer = np.zeros((0, self.input_size), dtype=np.float32)
        self.step_counter = 0
        if clear_persistence and self.threshold_config is not None:
            self.state_machine = AnomalyStateMachine(
                config=self.threshold_config,
                persistence=self.persistence_k,
                t_persist=self.t_persist,
            )

    def update(
        self,
        frame_or_chunk: np.ndarray,
        usable_mask: Optional[List[bool]] = None,
        temperature_ok: bool = True,
    ) -> DetectorOutput:
        """Ingests live frame(s) with bad-sensor isolation. Returns DetectorOutput."""
        self.last_update_timestamp = time.time()
        self.step_counter += 1

        if frame_or_chunk.ndim == 1:
            frame_or_chunk = frame_or_chunk.reshape(1, -1)

        if usable_mask is not None:
            self._last_usable_mask = list(usable_mask)
        active_mask = usable_mask if usable_mask is not None else self._last_usable_mask

        # Append to ring buffer
        if len(self.buffer) == 0:
            self.buffer = frame_or_chunk.copy()
        else:
            self.buffer = np.vstack([self.buffer, frame_or_chunk])

        if len(self.buffer) > self.seq_len:
            self.buffer = self.buffer[-self.seq_len :]

        # Guardrail: temperature quality failure when input expects temp feature
        if not temperature_ok and self.input_size > self.target_size:
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.LOW_CONFIDENCE,
                score=0.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={"temperature_ok": False},
                reason="TEMPERATURE_UNAVAILABLE: Temperature sensor fault/offline",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        if active_mask is not None:
            usable_targets = [
                i for i in range(min(self.target_size, len(active_mask))) if active_mask[i]
            ]
            if len(usable_targets) < min(self.min_channels, self.target_size):
                return DetectorOutput(
                    detector_id=self.detector_id,
                    version=self.version,
                    timestamp=self.last_update_timestamp,
                    state=DetectorState.LOW_CONFIDENCE,
                    score=0.0,
                    tau=self.tau,
                    update_cadence_seconds=self.update_cadence_seconds,
                    details={"usable_channels": len(usable_targets), "min_channels": self.min_channels},
                    reason=f"INSUFFICIENT_USABLE_CHANNELS: {len(usable_targets)}/{self.min_channels}",
                    provenance={"scope": self.SCOPE_STATEMENT},
                )
            if self.required_channels:
                missing_req = [
                    sid
                    for sid in self.required_channels
                    if sid in self.channel_names
                    and self.channel_names.index(sid) < len(active_mask)
                    and not active_mask[self.channel_names.index(sid)]
                ]
                if missing_req:
                    return DetectorOutput(
                        detector_id=self.detector_id,
                        version=self.version,
                        timestamp=self.last_update_timestamp,
                        state=DetectorState.LOW_CONFIDENCE,
                        score=0.0,
                        tau=self.tau,
                        update_cadence_seconds=self.update_cadence_seconds,
                        details={"missing_required_channels": missing_req},
                        reason=f"REQUIRED_CHANNELS_UNAVAILABLE: {missing_req}",
                        provenance={"scope": self.SCOPE_STATEMENT},
                    )
        else:
            usable_targets = list(range(self.target_size))

        if len(self.buffer) < self.seq_len:
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.LOW_CONFIDENCE,
                score=0.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={"buffered_samples": len(self.buffer), "required_samples": self.seq_len},
                reason=f"BUFFERING: {len(self.buffer)}/{self.seq_len} samples",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        # Full window available
        window = self.buffer.copy()

        # Isolate excluded (FAULT/OFFLINE) channels with neutral baseline fill before model forward
        if active_mask is not None:
            for ch_idx in range(min(window.shape[1], len(active_mask))):
                if not active_mask[ch_idx]:
                    fill_val = 0.0
                    if self.scaler is not None and hasattr(self.scaler, "mean_") and ch_idx < len(self.scaler.mean_):
                        fill_val = float(self.scaler.mean_[ch_idx])
                    window[:, ch_idx] = fill_val

        # Check for hardware sensor faults strictly on usable target channels
        check_cols = usable_targets if usable_targets else list(range(window.shape[1]))
        usable_win = window[:, check_cols]
        if np.isnan(usable_win).any() or np.isinf(usable_win).any() or np.isnan(window).any() or np.isinf(window).any():
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.SENSOR_FAULT,
                score=-1.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                reason="SENSOR_FAULT: NaN or Inf detected in usable window",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        std_per_ch = np.std(usable_win, axis=0)
        if (std_per_ch < 1e-6).any():
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.SENSOR_FAULT,
                score=-1.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                reason="SENSOR_FAULT: Usable channel flatline detected",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        # Scale and forward pass
        try:
            if self.scaler is not None and self.scaler.is_fitted:
                scaled_win = self.scaler.transform(window)
            else:
                scaled_win = window

            bx = torch.tensor(scaled_win, dtype=torch.float32, device=self.device).unsqueeze(0)
            by = bx[:, :, : self.target_size]
            self.model.eval()
            with torch.no_grad():
                recon, _ = self.model(bx)
                mse_vec = torch.mean((recon - by) ** 2, dim=1).squeeze(0).cpu().numpy()

            # Zero out any MSE contribution from excluded channels
            if active_mask is not None:
                for ch_idx in range(min(len(mse_vec), len(active_mask))):
                    if not active_mask[ch_idx]:
                        mse_vec[ch_idx] = 0.0

            score = float(np.mean(mse_vec[usable_targets])) if usable_targets else float(np.mean(mse_vec))

            # Step state machine
            if self.state_machine is not None:
                event = self.state_machine.step(
                    window_index=self.step_counter,
                    timestamp=self.last_update_timestamp,
                    mse_vector=mse_vec,
                )
                status_str = event.system_status
                defect_node = event.localized_defect_node
            else:
                status_str = STATE_NORMAL if score < self.tau else STATE_CRITICAL
                defect_node = None

            state_map = {
                STATE_NORMAL: DetectorState.NORMAL,
                STATE_WARNING: DetectorState.WARNING,
                STATE_CRITICAL: DetectorState.CRITICAL,
                STATE_SENSOR_FAULT: DetectorState.SENSOR_FAULT,
            }
            state = state_map.get(status_str, DetectorState.NORMAL)
            reason = f"Defect suspected at node: {defect_node}" if defect_node else None

            details = {
                "mean_mse": score,
                "per_channel_mse": {
                    name: float(mse_vec[i])
                    for i, name in enumerate(self.target_channel_names)
                },
                "defect_node": defect_node,
            }

            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=state,
                score=score,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details=details,
                reason=reason,
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        except Exception as exc:
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.SENSOR_FAULT,
                score=-1.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                reason=f"DETECTOR_EXCEPTION: {str(exc)}",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "model.pt")
        if self.scaler is not None:
            joblib.dump(self.scaler, p / "scaler.joblib")
        if self.threshold_config is not None:
            with open(p / "threshold_config.json", "w", encoding="utf-8") as f:
                json.dump(self.threshold_config.to_dict(), f, indent=2)
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata(), f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        model_file = p / "model.pt"
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file not found at {model_file}")
        self.model.load_state_dict(torch.load(model_file, map_location=self.device, weights_only=True))
        self.model.eval()

        scaler_file = p / "scaler.joblib"
        if scaler_file.is_file():
            self.scaler = joblib.load(scaler_file)

        thresh_file = p / "threshold_config.json"
        if thresh_file.is_file():
            with open(thresh_file, "r", encoding="utf-8") as f:
                th_data = json.load(f)
            self.threshold_config = ThresholdConfig.from_dict(th_data)
            self.tau = float(np.mean([ct.tau for ct in self.threshold_config.channels.values()]))
            self.state_machine = AnomalyStateMachine(
                config=self.threshold_config,
                persistence=self.persistence_k,
                t_persist=self.t_persist,
            )

        meta_file = p / "metadata.json"
        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.healthy_hours_trained = meta.get("healthy_hours_trained", 0.0)
            self.training_temp_range = tuple(meta.get("training_temp_range", (-999.0, 999.0)))

        self.is_fitted = True
