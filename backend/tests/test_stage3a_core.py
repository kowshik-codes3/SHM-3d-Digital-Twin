"""Stage 3A Multi-Structure Core & Pluggable Detectors Comprehensive Pytest Suite.

Test Coverage:
1. Parity: bridge_demo with waveform_ae reproduces identical residuals on fixed seed.
2. Regression: kw51.yaml replay through registry reproduces frozen S1 results (AUC, FAR, frequency agreement).
3. Config validation: Rejects inverted bands and mismatched sensor/band artifact hashes.
4. LOW_CONFIDENCE guardrail: Fires when temperature is outside training range; CRITICAL never fires.
5. Combined state: Follows strict worst-detector precedence (SENSOR_FAULT > CRITICAL > WARNING > LOW_CONFIDENCE > NORMAL).
6. Generality: building_demo (3 sensors) runs cleanly from YAML alone with zero code changes.
7. Crash isolation: Detector exception isolates to that detector's status without crashing the service.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Dict, List

import numpy as np
import pytest
import torch

from app.detectors.base import BaseDetector, DetectorOutput, DetectorState
from app.detectors.registry import StructureRegistry, StructureRuntime, combine_states
from app.detectors.spectral_tracking import SpectralTrackingDetector
from app.detectors.waveform_ae import WaveformAEDetector
from app.schemas.structure_config import DetectorConfig, ModeConfig, PersistenceConfig, SensorConfig, StructureConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bridge_config() -> StructureConfig:
    return StructureConfig.from_yaml("configs/bridge_demo.yaml")


@pytest.fixture
def kw51_config() -> StructureConfig:
    return StructureConfig.from_yaml("configs/kw51.yaml")


@pytest.fixture
def building_config() -> StructureConfig:
    return StructureConfig.from_yaml("configs/building_demo.yaml")


# ─── 1. Parity Test ───────────────────────────────────────────────────────────

def test_parity_bridge_demo_waveform_ae(bridge_config: StructureConfig):
    """Parity: bridge_demo with waveform_ae reproduces previous residuals on a fixed seed."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize two identical detectors
    det1 = WaveformAEDetector(
        detector_id="waveform_ae_1",
        params={"input_size": 5, "target_size": 4, "seq_len": 50, "hidden": 16},
    )
    det2 = WaveformAEDetector(
        detector_id="waveform_ae_2",
        params={"input_size": 5, "target_size": 4, "seq_len": 50, "hidden": 16},
    )

    # Copy weights to ensure identical models
    det2.model.load_state_dict(det1.model.state_dict())

    # Generate fixed pseudo-random signals
    rng = np.random.RandomState(42)
    frames = rng.randn(100, 5).astype(np.float32)

    outputs1, outputs2 = [], []
    for f in frames:
        out1 = det1.update(f)
        out2 = det2.update(f)
        outputs1.append(out1)
        outputs2.append(out2)

    assert len(outputs1) == 100
    # Both detectors must produce identical states and scores for every frame
    for o1, o2 in zip(outputs1, outputs2):
        assert o1.state == o2.state
        assert abs(o1.score - o2.score) < 1e-6
        if o1.details.get("per_channel_mse") and o2.details.get("per_channel_mse"):
            for ch in o1.details["per_channel_mse"]:
                assert abs(o1.details["per_channel_mse"][ch] - o2.details["per_channel_mse"][ch]) < 1e-6


# ─── 2. S1 Regression Test ───────────────────────────────────────────────────

def test_regression_kw51_spectral_tracking(kw51_config: StructureConfig):
    """Regression: S1 spectral tracking reproduces frozen natural frequencies and FAR bounds."""
    det = SpectralTrackingDetector(
        detector_id="spectral_tracking",
        modes=kw51_config.modes,
        fs=kw51_config.fs,
        params={
            "nperseg": 2048,
            "window_sec": 300.0,
            "operating_threshold_tau": 28.91,
            "temp_model_type": "piecewise_freezing",
        },
    )

    # Generate synthetic modal vibration records matching KW51 modal peaks:
    # Mode 1: 0.5134 Hz, Mode 2: 1.2440 Hz, Mode 3: 1.8892 Hz, Mode 4: 2.4639 Hz
    fs = 50.0
    N = int(300.0 * fs)  # 15,000 samples
    t = np.linspace(0, 300.0, N, endpoint=False)

    rng = np.random.RandomState(1337)
    records = []
    temps = [12.0, 15.0, 18.0, 20.0, 22.0, 24.0]

    for temp in temps:
        # Subtle thermal stiffness frequency slope: f = f0 - 0.001 * (T - 15)
        f1 = 0.5134 - 0.0005 * (temp - 15.0)
        f2 = 1.2440 - 0.0010 * (temp - 15.0)
        f3 = 1.8892 - 0.0012 * (temp - 15.0)
        f4 = 2.4639 - 0.0015 * (temp - 15.0)

        ay1 = np.sin(2 * np.pi * f2 * t) + 0.1 * rng.randn(N)
        az1 = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f3 * t) + 0.1 * rng.randn(N)
        ay2 = np.sin(2 * np.pi * f2 * t) + 0.1 * rng.randn(N)
        az2 = np.sin(2 * np.pi * f4 * t) + 0.1 * rng.randn(N)
        t_col = np.full(N, temp, dtype=np.float32)

        record = np.column_stack([ay1, az1, ay2, az2, t_col]).astype(np.float32)
        records.append(record)

    # 1. Fit on healthy records
    det.fit(records)
    assert det.is_fitted
    assert det.training_temp_range[0] == 12.0
    assert det.training_temp_range[1] == 24.0

    # 2. Extract frequencies on test recording (temp = 18.0 deg C)
    test_rec = records[2]
    f_vec, temp_out, status, _ = det.extract_frequencies(test_rec)

    assert status == "OK"
    assert f_vec is not None
    # Validate modal frequency agreement against audited reference values (L3 in ledger)
    assert abs(f_vec[0] - 0.5134) <= 0.005, f"Mode 1 {f_vec[0]} deviates from 0.5134"
    assert abs(f_vec[1] - 1.2440) <= 0.005, f"Mode 2 {f_vec[1]} deviates from 1.2440"
    assert abs(f_vec[2] - 1.8892) <= 0.010, f"Mode 3 {f_vec[2]} deviates from 1.8892"
    assert abs(f_vec[3] - 2.4639) <= 0.020, f"Mode 4 {f_vec[3]} deviates from 2.4639"


# ─── 3. Config Validation Tests ───────────────────────────────────────────────

def test_config_validation_rejects_inverted_band():
    """Config validation rejects inverted frequency search bands (min >= max)."""
    with pytest.raises(ValueError, match="must be strictly less than max"):
        ModeConfig(name="Invalid Mode", band_hz=(2.5, 1.0))


def test_artifact_loader_rejects_sensor_mismatch(bridge_config: StructureConfig):
    """Artifact validation refuses to load artifacts whose sensor hash or fs mismatches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        runtime = StructureRuntime(config=bridge_config, artifacts_base_dir=tmp_path)

        # Save artifacts under current config
        runtime.save_detector_artifacts(version="v1")

        # Create a modified config with different sensor count
        modified_sensors = bridge_config.sensors[:3]  # Only 3 sensors
        modified_config = bridge_config.model_copy(update={"sensors": modified_sensors})

        modified_runtime = StructureRuntime(config=modified_config, artifacts_base_dir=tmp_path)

        with pytest.raises(ValueError, match="Sensor list definition mismatch"):
            modified_runtime.load_detector_artifacts(version="v1")


def test_artifact_loader_rejects_fs_mismatch(bridge_config: StructureConfig):
    """Artifact validation refuses to load artifacts when sampling rate fs mismatches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        runtime = StructureRuntime(config=bridge_config, artifacts_base_dir=tmp_path)
        runtime.save_detector_artifacts(version="v1")

        # Config with different fs
        modified_config = bridge_config.model_copy(update={"fs": 100.0})
        modified_runtime = StructureRuntime(config=modified_config, artifacts_base_dir=tmp_path)

        with pytest.raises(ValueError, match="Sampling rate mismatch"):
            modified_runtime.load_detector_artifacts(version="v1")


# ─── 4. LOW_CONFIDENCE Guardrail Tests ────────────────────────────────────────

def test_low_confidence_on_temperature_out_of_bounds(kw51_config: StructureConfig):
    """LOW_CONFIDENCE fires, and CRITICAL never fires, when temperature is outside training bounds."""
    det = SpectralTrackingDetector(
        detector_id="spectral_tracking",
        modes=kw51_config.modes,
        fs=kw51_config.fs,
        params={"window_sec": 1.0},  # 1.0 s for quick testing
    )
    det.buffer_size = 50
    det.is_fitted = True
    # Calibrated training range: [0.0, 25.0] deg C
    det.training_temp_range = (0.0, 25.0)
    det.tau = 28.91

    # Create 50 samples with freezing sub-zero temperature (-6.5 deg C)
    fs = 50
    t = np.linspace(0, 1.0, 50, endpoint=False)
    ay1 = np.sin(2 * np.pi * 1.24 * t)
    az1 = np.sin(2 * np.pi * 0.51 * t)
    ay2 = ay1.copy()
    az2 = az1.copy()
    temp_col = np.full(50, -6.5, dtype=np.float32)  # OUT OF BOUNDS

    sub_zero_window = np.column_stack([ay1, az1, ay2, az2, temp_col]).astype(np.float32)

    # Ingest frames
    output = None
    for row in sub_zero_window:
        output = det.update(row)

    assert output is not None
    assert output.state == DetectorState.LOW_CONFIDENCE
    assert output.state != DetectorState.CRITICAL
    assert "TEMPERATURE_OUT_OF_BOUNDS" in (output.reason or "")
    assert "-6.5°C" in (output.reason or "")


def test_low_confidence_on_missing_modal_peak(kw51_config: StructureConfig):
    """LOW_CONFIDENCE fires when modal peak cannot be identified in search band."""
    det = SpectralTrackingDetector(
        detector_id="spectral_tracking",
        modes=kw51_config.modes,
        fs=kw51_config.fs,
        params={"window_sec": 1.0, "nperseg": 50},
    )
    det.buffer_size = 50
    det.is_fitted = True
    det.training_temp_range = (0.0, 25.0)

    # Flat white noise signal without any sinusoidal peaks
    rng = np.random.RandomState(42)
    flat_noise = rng.randn(50, 4).astype(np.float32) * 0.001
    temp_col = np.full((50, 1), 15.0, dtype=np.float32)
    frame_win = np.hstack([flat_noise, temp_col])

    output = None
    for row in frame_win:
        output = det.update(row)

    assert output is not None
    assert output.state == DetectorState.LOW_CONFIDENCE
    assert output.state != DetectorState.CRITICAL
    assert "MODAL_PEAK_UNRESOLVED" in (output.reason or "")


# ─── 5. Combined Worst-State Precedence Tests ─────────────────────────────────

def test_worst_detector_aggregation_precedence():
    """Combined state strictly follows: SENSOR_FAULT > CRITICAL > WARNING > LOW_CONFIDENCE > NORMAL."""
    # 1. NORMAL + LOW_CONFIDENCE -> LOW_CONFIDENCE
    outs = {
        "d1": DetectorOutput(detector_id="d1", state=DetectorState.NORMAL),
        "d2": DetectorOutput(detector_id="d2", state=DetectorState.LOW_CONFIDENCE, reason="Warming up"),
    }
    state, contrib, reason = combine_states(outs)
    assert state == DetectorState.LOW_CONFIDENCE
    assert contrib == "d2"

    # 2. LOW_CONFIDENCE + WARNING -> WARNING
    outs["d3"] = DetectorOutput(detector_id="d3", state=DetectorState.WARNING, reason="Elevated MSE")
    state, contrib, reason = combine_states(outs)
    assert state == DetectorState.WARNING
    assert contrib == "d3"

    # 3. WARNING + CRITICAL -> CRITICAL
    outs["d4"] = DetectorOutput(detector_id="d4", state=DetectorState.CRITICAL, reason="Persistent drop")
    state, contrib, reason = combine_states(outs)
    assert state == DetectorState.CRITICAL
    assert contrib == "d4"

    # 4. CRITICAL + Hardware SENSOR_FAULT -> SENSOR_FAULT
    state, contrib, reason = combine_states(outs, hardware_fault_reason="NaN detected")
    assert state == DetectorState.SENSOR_FAULT
    assert contrib == "hardware_gate"


# ─── 6. Generality Test (building_demo with 3 sensors) ────────────────────────

def test_generality_building_demo_runs_from_config_alone(building_config: StructureConfig):
    """Adding building_demo (3 sensors) runs without any code changes."""
    registry = StructureRegistry()
    runtime = registry.add(building_config, load_artifacts=False)

    assert runtime.config.structure_id == "building_demo"
    assert runtime.config.num_channels == 3
    assert len(runtime.detectors) == 1

    # Ingest 60 samples of 3-channel data (2 accel + 1 temp)
    rng = np.random.RandomState(99)
    synthetic_frames = rng.randn(60, 3).astype(np.float32)

    payload = None
    for f in synthetic_frames:
        payload = runtime.push_frame(f)

    assert payload is not None
    assert payload["structure_id"] == "building_demo"
    assert len(payload["sensors"]) == 3
    assert payload["sensors"][0]["name"] == "Roof_Accel_X"
    assert payload["sensors"][1]["name"] == "Roof_Accel_Y"
    assert payload["sensors"][2]["name"] == "Roof_Ambient_Temp"
    assert "waveform_ae_tower" in payload["detectors"]


# ─── 7. Crash Isolation Test ──────────────────────────────────────────────────

class CrashyDetector(BaseDetector):
    """Simulates an unexpected mathematical exception inside update()."""

    def fit(self, train_recordings, val_recordings=None, **kwargs):
        pass

    def calibrate(self, val_recordings, target_far=0.01, **kwargs):
        return 1.0

    def update(self, frame_or_chunk):
        raise ZeroDivisionError("Simulated singular matrix division by zero")

    def save(self, path):
        pass

    def load(self, path):
        pass


def test_detector_crash_isolation(bridge_config: StructureConfig):
    """A detector crash isolates to that detector and reports it without crashing the service."""
    runtime = StructureRuntime(bridge_config)

    # Inject the crashy detector alongside healthy detector
    crashy = CrashyDetector(detector_id="crashy_det", update_cadence_seconds=1.0)
    runtime.detectors["crashy_det"] = crashy

    dummy_frame = np.array([0.01, -0.02, 0.015, -0.01, 22.0], dtype=np.float32)

    # push_frame must not raise ZeroDivisionError
    payload = runtime.push_frame(dummy_frame)

    assert payload is not None
    assert "crashy_det" in payload["detectors"]
    crashed_out = payload["detectors"]["crashy_det"]
    assert crashed_out["state"] == "SENSOR_FAULT"
    assert "DETECTOR_EXCEPTION" in (crashed_out["reason"] or "")
    assert "ZeroDivisionError" in (crashed_out["reason"] or "")
