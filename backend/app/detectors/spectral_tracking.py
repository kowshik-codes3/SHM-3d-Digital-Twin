"""Streaming Spectral Tracking Detector (Production "S1" Architecture).

Direct physical vibration frequency tracking from raw streaming sensor data.
Implements the frozen parameters from FROZEN_S1.md and strict scientific guardrails:
- Welch PSD with parabolic 3-point sub-bin peak interpolation.
- Piecewise-freezing robust temperature compensation (Huber regression, hinge at 0 deg C).
- Regularized multivariate Mahalanobis distance.
- K = 2 persistence temporal verification.
- Parallel EWMA slow-drift trend tracker (span = 20).
- STRICT LOW_CONFIDENCE state: Triggers whenever ambient temperature leaves the
  calibrated baseline range [T_min, T_max], whenever a modal peak is missing, or
  whenever the ring buffer lacks a full 5-minute analysis window. NEVER triggers
  CRITICAL alarms from mathematical extrapolation.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import scipy.signal as signal
from sklearn.linear_model import HuberRegressor, LinearRegression

from app.detectors.base import BaseDetector, DetectorOutput, DetectorState, get_git_hash
from app.schemas.structure_config import ModeConfig


class SpectralTrackingDetector(BaseDetector):
    """S1 Pluggable Spectral Tracking SHM Detector."""

    SCOPE_STATEMENT = (
        "Sub-bin modal natural frequency tracking and stiffness degradation detector. "
        "Operates on 5-minute rolling windows (300.0 s cadence). Physical detection delay is >= 10 min."
    )

    def __init__(
        self,
        detector_id: str = "spectral_tracking",
        version: str = "v1",
        update_cadence_seconds: float = 300.0,
        params: Optional[Dict[str, Any]] = None,
        modes: Optional[List[ModeConfig]] = None,
        fs: float = 50.0,
        temp_channel_index: int = 4,
        channel_axes: Optional[List[str]] = None,
        sensor_ids: Optional[List[str]] = None,
        min_channels: int = 2,
        required_channels: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            detector_id=detector_id,
            version=version,
            update_cadence_seconds=update_cadence_seconds,
            params=params,
        )
        p = self.params
        self.fs: float = p.get("fs", fs)
        self.nperseg: int = p.get("nperseg", 2048)
        self.window_sec: float = p.get("window_sec", 300.0)
        self.temp_model_type: str = p.get("temp_model_type", "piecewise_freezing")
        self.target_far: float = p.get("target_far", 0.01)
        self.persistence_k: int = p.get("persistence_k", 2)
        self.tau: float = p.get("operating_threshold_tau", 28.91)
        self.slow_drift_ewma_span: int = p.get("slow_drift_ewma_span", 20)
        self.temp_channel_index: int = temp_channel_index
        self.channel_axes: Optional[List[str]] = channel_axes
        self.sensor_ids: Optional[List[str]] = sensor_ids
        self.min_channels: int = int(p.get("min_channels", min_channels))
        self.required_channels: List[str] = list(p.get("required_channels", required_channels or []))
        self._last_usable_mask: Optional[List[bool]] = None
        self._last_temperature_ok: bool = True

        # Modes definition
        if modes:
            self.modes_cfg = [
                {"name": m.name, "band": m.band_hz, "channel_type": m.channel_type or "az"}
                for m in modes
            ]
        else:
            # Default to KW51 4 modes from FROZEN_S1.md
            self.modes_cfg = [
                {"name": "Mode 1 (Vertical)", "band": (0.48, 0.55), "channel_type": "az"},
                {"name": "Mode 2 (Lateral)", "band": (1.18, 1.28), "channel_type": "ay"},
                {"name": "Mode 3 (Vertical)", "band": (1.80, 2.00), "channel_type": "az"},
                {"name": "Mode 4 (Vertical)", "band": (2.35, 2.52), "channel_type": "az"},
            ]

        # Learned regression models and covariance
        self.regressors: List[Union[HuberRegressor, LinearRegression]] = []
        self.cov: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None

        # Streaming state variables
        self.buffer_size = int(self.window_sec * self.fs)
        self.ring_buffer = np.zeros((0, self.temp_channel_index + 1), dtype=np.float32)
        self.consecutive_alerts: int = 0
        self.ewma_score: Optional[float] = None
        self.last_modal_frequencies: Optional[List[float]] = None
        self.last_residuals: Optional[List[float]] = None
        self.last_temperature: Optional[float] = None

    def _prepare_temp_features(self, temps: np.ndarray) -> np.ndarray:
        """Constructs temperature basis features."""
        T = np.asarray(temps, dtype=np.float64).reshape(-1, 1)
        if self.temp_model_type == "piecewise_freezing":
            freezing_hinge = np.maximum(0.0, -T)
            return np.hstack([T, freezing_hinge])
        return T

    def extract_frequencies(
        self,
        recording: np.ndarray,
        usable_mask: Optional[List[bool]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[float], str, Optional[str]]:
        """Extracts sub-bin peak natural frequencies and temperature from raw vibration record.
        
        Returns:
            (freqs, temp, status_str, reason_detail)
        """
        n_samples = recording.shape[0]
        actual_nperseg = min(self.nperseg, n_samples)
        if actual_nperseg < 32:
            return None, None, "INSUFFICIENT_DATA", f"Need at least 32 samples for PSD, got {n_samples}"

        n_channels = recording.shape[1]
        temp_idx = min(self.temp_channel_index, n_channels - 1) if self.temp_channel_index >= 0 else n_channels - 1
        all_vib_indices = [i for i in range(n_channels) if i != temp_idx]
        if usable_mask is not None and len(usable_mask) == n_channels:
            vib_indices = [i for i in all_vib_indices if usable_mask[i]]
        else:
            vib_indices = all_vib_indices

        if len(vib_indices) < max(1, min(self.min_channels, len(all_vib_indices))):
            if len(vib_indices) == 0:
                return None, None, "LOW_CONFIDENCE", "No usable vibration channels remaining"
            if len(vib_indices) < self.min_channels:
                return (
                    None,
                    None,
                    "LOW_CONFIDENCE",
                    f"Usable channels {len(vib_indices)} < min_channels {self.min_channels}",
                )

        vib = recording[:, vib_indices]
        temp = float(np.nanmean(recording[:, temp_idx]))

        # 1. Sensor fault checks strictly across active usable vibration channels
        if np.isnan(vib).any() or np.isnan(temp) or np.isinf(vib).any():
            return None, None, "SENSOR_FAULT_NAN", "NaN or Inf present in usable channels"

        std_per_ch = np.std(vib, axis=0)
        if (std_per_ch < 1e-5).any():
            return None, None, "SENSOR_FAULT_FLATLINE", "Usable sensor channel flatline (< 1e-5 std)"

        if usable_mask is None:
            peak_per_ch = np.max(np.abs(vib - np.mean(vib, axis=0)), axis=0)
            if (peak_per_ch > 12.0 * np.maximum(std_per_ch, 1e-6)).any():
                return None, None, "SENSOR_FAULT_SPIKE", "Impulse noise spike (> 12 sigma)"

        # 2. Spectral Welch PSD decomposition using axis metadata when available
        axes = self.channel_axes if (self.channel_axes and len(self.channel_axes) == n_channels) else None
        if axes is not None:
            ay_cols = [k for k, orig_idx in enumerate(vib_indices) if str(axes[orig_idx]).lower() == "ay"]
            az_cols = [k for k, orig_idx in enumerate(vib_indices) if str(axes[orig_idx]).lower() == "az"]
            if not ay_cols and any(m.get("channel_type") == "ay" for m in self.modes_cfg):
                return None, temp, "MISSING_PEAK", "No healthy lateral (ay) sensor available"
            if not az_cols and any(m.get("channel_type") == "az" for m in self.modes_cfg):
                return None, temp, "MISSING_PEAK", "No healthy vertical (az) sensor available"
            ay_sig = np.mean(vib[:, ay_cols], axis=1) if ay_cols else vib[:, 0]
            az_sig = np.mean(vib[:, az_cols], axis=1) if az_cols else vib[:, 0]
        elif vib.shape[1] >= 4:
            ay_sig = 0.5 * (vib[:, 0] + vib[:, 2])
            az_sig = 0.5 * (vib[:, 1] + vib[:, 3])
        elif vib.shape[1] >= 2:
            ay_sig = vib[:, 0]
            az_sig = vib[:, 1]
        else:
            ay_sig = vib[:, 0]
            az_sig = vib[:, 0]

        freqs_ay, psd_ay = signal.welch(ay_sig, fs=self.fs, nperseg=actual_nperseg, window="hann")
        freqs_az, psd_az = signal.welch(az_sig, fs=self.fs, nperseg=actual_nperseg, window="hann")

        extracted_f = []
        df_bin = freqs_az[1] - freqs_az[0]

        for m_cfg in self.modes_cfg:
            ch_type = m_cfg.get("channel_type", "az")
            f_axis = freqs_az if ch_type == "az" else freqs_ay
            psd = psd_az if ch_type == "az" else psd_ay
            f_min, f_max = m_cfg["band"]

            idx = np.where((f_axis >= f_min) & (f_axis <= f_max))[0]
            if len(idx) < 3:
                return None, temp, "MISSING_PEAK", f"Search band {f_min}-{f_max} Hz lacks spectral bins"

            band_psd = psd[idx]
            max_in_band = np.argmax(band_psd)
            k_max = idx[max_in_band]

            # Check peak prominence: peak must be at least 1.1x higher than band average
            mean_band_power = np.mean(band_psd)
            if band_psd[max_in_band] < 1.10 * mean_band_power:
                return None, temp, "MISSING_PEAK", f"No distinct peak in band {f_min}-{f_max} Hz"

            # 3-point parabolic sub-bin peak refinement
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

        return np.array(extracted_f, dtype=np.float64), temp, "OK", None

    def fit(
        self,
        train_recordings: List[np.ndarray],
        val_recordings: Optional[List[np.ndarray]] = None,
        train_meta: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Fits robust temperature compensation models and baseline covariance on healthy recordings."""
        f_list, t_list = [], []
        for r in train_recordings:
            f_vec, temp, status, _ = self.extract_frequencies(r)
            if status == "OK" and f_vec is not None and temp is not None:
                f_list.append(f_vec)
                t_list.append(temp)

        if len(f_list) < 3:
            raise ValueError(f"Insufficient valid recordings for S1 fit: {len(f_list)}")

        F_train = np.array(f_list)
        T_train = np.array(t_list)

        # Record exact healthy temperature training bounds
        self.training_temp_range = (float(np.min(T_train)), float(np.max(T_train)))
        self.healthy_hours_trained = float(len(train_recordings) * (self.window_sec / 3600.0))

        X_train = self._prepare_temp_features(T_train)
        self.regressors = []
        residuals = []

        for m in range(F_train.shape[1]):
            reg = HuberRegressor(epsilon=1.35, max_iter=200)
            reg.fit(X_train, F_train[:, m])
            self.regressors.append(reg)
            res_m = F_train[:, m] - reg.predict(X_train)
            residuals.append(res_m)

        R_train = np.column_stack(residuals)
        self.cov = np.cov(R_train, rowvar=False)
        reg_cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.pinv(reg_cov)
        self.is_fitted = True

        if val_recordings:
            self.calibrate(val_recordings)

    def calibrate(
        self,
        val_recordings: List[np.ndarray],
        target_far: float = 0.01,
        **kwargs: Any,
    ) -> float:
        """Calibrates Mahalanobis distance operating threshold to target FAR."""
        scores = []
        for r in val_recordings:
            f_vec, temp, status, _ = self.extract_frequencies(r)
            if status == "OK" and f_vec is not None and temp is not None:
                d2, _ = self._compute_mahalanobis(f_vec, temp)
                scores.append(d2)

        if not scores:
            raise ValueError("No valid scores generated on validation set")

        self.target_far = target_far
        self.tau = float(np.percentile(scores, (1.0 - target_far) * 100.0))
        return self.tau

    def _compute_mahalanobis(self, f_vec: np.ndarray, temp: float) -> Tuple[float, List[float]]:
        """Computes squared Mahalanobis distance and temperature-compensated residuals."""
        X = self._prepare_temp_features(np.array([temp]))
        residuals = []
        for m, reg in enumerate(self.regressors):
            f_pred = float(reg.predict(X)[0])
            residuals.append(float(f_vec[m] - f_pred))
        r = np.array(residuals, dtype=np.float64)
        d2 = float(r @ self.inv_cov @ r)
        return d2, residuals

    def reset_state(self, clear_persistence: bool = True) -> None:
        """Reset streaming buffer and optionally persistence counters at recording boundaries."""
        self.ring_buffer = np.zeros((0, self.ring_buffer.shape[1] if self.ring_buffer.ndim == 2 else self.temp_channel_index + 1), dtype=np.float32)
        if clear_persistence:
            self.consecutive_alerts = 0
            self.ewma_score = None

    def update(
        self,
        frame_or_chunk: np.ndarray,
        usable_mask: Optional[List[bool]] = None,
        temperature_ok: bool = True,
    ) -> DetectorOutput:
        """Streaming state update with strict LOW_CONFIDENCE & bad-sensor guardrails."""
        self.last_update_timestamp = time.time()

        if frame_or_chunk.ndim == 1:
            frame_or_chunk = frame_or_chunk.reshape(1, -1)

        if usable_mask is not None:
            self._last_usable_mask = list(usable_mask)
        self._last_temperature_ok = bool(temperature_ok)

        # Append to ring buffer
        if len(self.ring_buffer) == 0:
            self.ring_buffer = frame_or_chunk.copy()
        else:
            self.ring_buffer = np.vstack([self.ring_buffer, frame_or_chunk])

        if len(self.ring_buffer) > self.buffer_size:
            self.ring_buffer = self.ring_buffer[-self.buffer_size :]

        # 0. Guardrail: Quality-layer temperature fault or insufficient usable channels
        if not temperature_ok:
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.LOW_CONFIDENCE,
                score=0.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={"temperature_ok": False},
                reason="TEMPERATURE_UNAVAILABLE: Temperature sensor fault/offline; refusing uncompensated evaluation",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        active_mask = usable_mask if usable_mask is not None else self._last_usable_mask
        if active_mask is not None:
            n_ch = self.ring_buffer.shape[1]
            t_idx = min(self.temp_channel_index, n_ch - 1) if self.temp_channel_index >= 0 else n_ch - 1
            usable_vib_count = sum(
                1 for i in range(min(n_ch, len(active_mask))) if i != t_idx and active_mask[i]
            )
            if usable_vib_count < self.min_channels:
                return DetectorOutput(
                    detector_id=self.detector_id,
                    version=self.version,
                    timestamp=self.last_update_timestamp,
                    state=DetectorState.LOW_CONFIDENCE,
                    score=0.0,
                    tau=self.tau,
                    update_cadence_seconds=self.update_cadence_seconds,
                    details={"usable_channels": usable_vib_count, "min_channels": self.min_channels},
                    reason=f"INSUFFICIENT_USABLE_CHANNELS: {usable_vib_count}/{self.min_channels} usable vibration sensors",
                    provenance={"scope": self.SCOPE_STATEMENT},
                )
            if self.required_channels and self.sensor_ids:
                missing_req = [
                    sid
                    for sid in self.required_channels
                    if sid in self.sensor_ids
                    and self.sensor_ids.index(sid) < len(active_mask)
                    and not active_mask[self.sensor_ids.index(sid)]
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

        # 1. Guardrail: Insufficient buffer length -> LOW_CONFIDENCE
        if len(self.ring_buffer) < self.buffer_size:
            cur_n = len(self.ring_buffer)
            req_n = self.buffer_size
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.LOW_CONFIDENCE,
                score=0.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={"buffered_samples": cur_n, "required_samples": req_n},
                reason=f"INSUFFICIENT_WINDOW: {cur_n}/{req_n} samples (warming up)",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        # 2. Guardrail: Temperature extrapolation -> LOW_CONFIDENCE (NEVER CRITICAL)
        temp_idx = min(self.temp_channel_index, self.ring_buffer.shape[1] - 1)
        temp = float(np.nanmean(self.ring_buffer[:, temp_idx]))
        t_min, t_max = self.training_temp_range
        if temp < t_min or temp > t_max:
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.LOW_CONFIDENCE,
                score=0.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={
                    "temperature": temp,
                    "calibrated_range": [t_min, t_max],
                },
                reason=(
                    f"TEMPERATURE_OUT_OF_BOUNDS: Ambient temp {temp:.1f}°C is outside "
                    f"calibrated healthy training range [{t_min:.1f}, {t_max:.1f}]°C. "
                    f"Refusing extrapolation to prevent false alarms."
                ),
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        # 3. Extract frequencies (excluding bad/offline channels via usable_mask)
        f_vec, temp, status, reason_detail = self.extract_frequencies(
            self.ring_buffer, usable_mask=active_mask
        )

        # 4. Guardrail: Hardware fault -> SENSOR_FAULT
        if status.startswith("SENSOR_FAULT"):
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.SENSOR_FAULT,
                score=-1.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={"status_code": status},
                reason=f"HARDWARE_FAULT: {reason_detail or status}",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        # 5. Guardrail: Missing modal peak or insufficient usable channels -> LOW_CONFIDENCE
        if status in ("MISSING_PEAK", "LOW_CONFIDENCE") or f_vec is None or temp is None:
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.LOW_CONFIDENCE,
                score=0.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={"status_code": status},
                reason=f"MODAL_PEAK_UNRESOLVED: {reason_detail}" if status == "MISSING_PEAK" else (reason_detail or status),
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        # 5b. Guardrail: Detector baseline not yet fitted -> LOW_CONFIDENCE (no matmul on None)
        if not self.is_fitted or self.inv_cov is None or not self.regressors:
            self.last_modal_frequencies = [round(float(x), 4) for x in f_vec]
            self.last_temperature = round(float(temp), 2)
            return DetectorOutput(
                detector_id=self.detector_id,
                version=self.version,
                timestamp=self.last_update_timestamp,
                state=DetectorState.LOW_CONFIDENCE,
                score=0.0,
                tau=self.tau,
                update_cadence_seconds=self.update_cadence_seconds,
                details={
                    "status_code": "NOT_FITTED",
                    "modal_frequencies_hz": self.last_modal_frequencies,
                },
                reason="DETECTOR_NOT_FITTED: Spectral regression baseline not calibrated",
                provenance={"scope": self.SCOPE_STATEMENT},
            )

        # 6. Model Evaluation (Temperature within valid range)
        score, residuals = self._compute_mahalanobis(f_vec, temp)

        # Update EWMA slow-drift layer
        alpha = 2.0 / (self.slow_drift_ewma_span + 1.0)
        if self.ewma_score is None:
            self.ewma_score = score
        else:
            self.ewma_score = alpha * score + (1.0 - alpha) * self.ewma_score

        # Persistence Filter (K = 2 consecutive exceedances)
        if score >= self.tau:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = max(0, self.consecutive_alerts - 1)

        if self.consecutive_alerts >= self.persistence_k:
            state = DetectorState.CRITICAL
            reason = f"CRITICAL: Persistence threshold met ({self.consecutive_alerts}/{self.persistence_k})"
        elif self.consecutive_alerts > 0:
            state = DetectorState.WARNING
            reason = f"WARNING: Transient threshold exceedance ({self.consecutive_alerts}/{self.persistence_k})"
        else:
            state = DetectorState.NORMAL
            reason = None

        self.last_modal_frequencies = [round(x, 4) for x in f_vec]
        self.last_residuals = [round(x, 4) for x in residuals]
        self.last_temperature = round(temp, 2)

        details = {
            "instantaneous_score": round(score, 4),
            "ewma_score": round(self.ewma_score, 4) if self.ewma_score is not None else None,
            "consecutive_alerts": self.consecutive_alerts,
            "persistence_k": self.persistence_k,
            "temperature": self.last_temperature,
            "calibrated_range": [t_min, t_max],
            "modal_frequencies_hz": {
                self.modes_cfg[i]["name"]: self.last_modal_frequencies[i]
                for i in range(len(self.modes_cfg))
            },
            "modal_residuals_hz": {
                self.modes_cfg[i]["name"]: self.last_residuals[i]
                for i in range(len(self.modes_cfg))
            },
        }

        return DetectorOutput(
            detector_id=self.detector_id,
            version=self.version,
            timestamp=self.last_update_timestamp,
            state=state,
            score=round(score, 4),
            tau=self.tau,
            update_cadence_seconds=self.update_cadence_seconds,
            details=details,
            reason=reason,
            provenance={"scope": self.SCOPE_STATEMENT},
        )

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.regressors, p / "regressors.joblib")
        if self.cov is not None and self.inv_cov is not None:
            np.savez(p / "covariance.npz", cov=self.cov, inv_cov=self.inv_cov)
        meta = self.metadata()
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        p = Path(path)
        reg_file = p / "regressors.joblib"
        if not reg_file.is_file():
            raise FileNotFoundError(f"Regressors file not found at {reg_file}")
        self.regressors = joblib.load(reg_file)

        cov_file = p / "covariance.npz"
        if cov_file.is_file():
            data = np.load(cov_file)
            self.cov = data["cov"]
            self.inv_cov = data["inv_cov"]

        meta_file = p / "metadata.json"
        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.tau = meta.get("tau", self.tau)
            self.training_temp_range = tuple(meta.get("training_temp_range", (-999.0, 999.0)))
            self.healthy_hours_trained = meta.get("healthy_hours_trained", 0.0)

        self.is_fitted = True
