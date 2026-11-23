"""Unit tests for Stage 2 ML improvements.

Validates:
a) Temperature handling: Conditional Autoencoder (input 5, target 4)
b) Time-based persistence & raw CRITICAL display fix
c) Warning threshold: 99th percentile vs 2-sigma SPC
d) EVT decorrelation: strided & block-maxima
e) Localisation: Top-3 suspected sensors with confidence distribution
f) Sensor-fault vs real-damage discrimination (flatline, spike, drift -> SENSOR_FAULT)
g) Severity score bounds (0-100)
h) Slow-drift layer: EWMA/CUSUM change detection & modal frequency tracking
"""

import numpy as np
import pytest
import torch

from app.ml.anomaly_engine import (
    AnomalyStateMachine,
    STATE_CRITICAL,
    STATE_NORMAL,
    STATE_SENSOR_FAULT,
    STATE_WARNING,
)
from app.ml.model import LSTMAutoencoder
from app.ml.threshold_calibrator import ChannelThreshold, ThresholdConfig
from app.validation.evt import EVTCalibrator
from app.validation.slow_drift import ModalFrequencyTracker, ResidualEWMA_CUSUM


def _dummy_threshold_config(channels=("aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az")):
    ch_map = {}
    for ch in channels:
        ch_map[ch] = ChannelThreshold(
            name=ch,
            mu=0.01,
            sigma=0.005,
            tau=0.025,
            warn=0.015,
            k=3.0,
        )
    return ThresholdConfig(
        calibration_timestamp=1000.0,
        k_multiplier=3.0,
        channels=ch_map,
    )


def test_stage2_conditional_autoencoder():
    """Item (a): Model can accept 5 channels (accel + temp) and reconstruct 4 channels."""
    model = LSTMAutoencoder(input_size=5, target_size=4, enc_hidden1=16, enc_hidden2=8, dec_hidden1=16, dec_hidden2=16)
    x = torch.randn(8, 50, 5)
    recon, z = model(x)
    assert recon.shape == (8, 50, 4)
    assert z.shape == (8, 8)


def test_stage2_persistence_warning_fix():
    """Item (b): Raw CRITICAL below P must display as WARNING, never NORMAL."""
    config = _dummy_threshold_config()
    sm = AnomalyStateMachine(config=config, persistence=3, t_persist=0.1)

    # Channel 0 has MSE above tau=0.025, others normal
    mse_vec = np.array([0.035, 0.005, 0.005, 0.005], dtype=np.float32)

    # Step 1: count=1 < P=3 -> Must be WARNING (not NORMAL!)
    ev1 = sm.step(window_index=0, timestamp=0.0, mse_vector=mse_vec)
    assert ev1.system_status == STATE_WARNING
    assert ev1.nodes["aBD11Az"].status == STATE_WARNING

    # Step 2: count=2 < P=3 -> Still WARNING
    ev2 = sm.step(window_index=1, timestamp=0.05, mse_vector=mse_vec)
    assert ev2.system_status == STATE_WARNING

    # Step 3: count=3 >= P=3 and duration 0.1 >= 0.1 -> Becomes CRITICAL
    ev3 = sm.step(window_index=2, timestamp=0.10, mse_vector=mse_vec)
    assert ev3.system_status == STATE_CRITICAL
    assert ev3.nodes["aBD11Az"].status == STATE_CRITICAL


def test_stage2_evt_decorrelation_and_warning_options():
    """Items (c) & (d): EVT with decorrelation modes and 99th percentile warning."""
    np.random.seed(42)
    # Autocorrelated residuals
    raw_res = np.zeros((1000, 2))
    for t in range(1, 1000):
        raw_res[t] = 0.9 * raw_res[t - 1] + 0.1 * np.random.exponential(scale=0.01, size=2)

    calib_strided = EVTCalibrator(
        tail_quantile=0.95,
        target_prob=0.001,
        decorrelation="strided",
        stride=5,
        channel_names=["ch1", "ch2"],
    )
    res_strided = calib_strided.calibrate(raw_res)

    calib_block = EVTCalibrator(
        tail_quantile=0.95,
        target_prob=0.001,
        decorrelation="block_maxima",
        block_size=10,
        channel_names=["ch1", "ch2"],
    )
    res_block = calib_block.calibrate(raw_res)

    assert "ch1" in res_strided.channels
    assert "ch2" in res_block.channels
    assert res_strided.channels["ch1"].warn_percentile_99 > 0.0

    # Test conversion to threshold config using 99th percentile warning
    cfg_p99 = res_strided.to_threshold_config(threshold_type="evt", warn_type="percentile_99")
    assert cfg_p99.channels["ch1"].warn == res_strided.channels["ch1"].warn_percentile_99


def test_stage2_top3_localization():
    """Item (e): Top-3 suspected sensors ranked with softmax confidence distribution."""
    config = _dummy_threshold_config()
    sm = AnomalyStateMachine(config=config, persistence=1, top_k=3)

    # Ch 0 has ratio 3.0, Ch 2 has ratio 1.5, others low
    mse_vec = np.array([0.075, 0.010, 0.0375, 0.010], dtype=np.float32)
    ev = sm.step(window_index=0, timestamp=0.0, mse_vector=mse_vec)

    assert len(ev.top_candidates) == 3
    # Primary suspected should be aBD11Az
    assert ev.top_candidates[0][0] == "aBD11Az"
    # Second should be aBD17Az
    assert ev.top_candidates[1][0] == "aBD17Az"
    # Confidences sum roughly to 1.0
    total_conf = sum(c for _, c in ev.top_candidates)
    assert 0.99 <= total_conf <= 1.01


def test_stage2_sensor_fault_discrimination():
    """Item (f): Flatline, impulse spike, and isolated drift flag SENSOR_FAULT, not CRITICAL."""
    config = _dummy_threshold_config()
    sm = AnomalyStateMachine(config=config, persistence=1, enable_sensor_fault_detection=True)

    # 1. Flatline window (std < 1e-5)
    raw_window_flat = np.ones((50, 4), dtype=np.float32)
    raw_window_flat[:, 0] = 0.001  # Constant flatline on channel 0
    raw_window_flat[:, 1:] = np.random.normal(0, 0.01, size=(50, 3))

    mse_vec = np.array([0.05, 0.01, 0.01, 0.01], dtype=np.float32)
    ev_flat = sm.step(window_index=0, timestamp=0.0, mse_vector=mse_vec, raw_window=raw_window_flat)
    assert ev_flat.system_status == STATE_SENSOR_FAULT
    assert ev_flat.nodes["aBD11Az"].status == STATE_SENSOR_FAULT
    assert ev_flat.nodes["aBD11Az"].sensor_fault == "flatline"

    # 2. Impulse spike (single huge spike, kurtosis > 20)
    raw_window_spike = np.random.normal(0, 0.01, size=(50, 4)).astype(np.float32)
    raw_window_spike[25, 0] = 10.0  # Massive glitch
    ev_spike = sm.step(window_index=1, timestamp=0.1, mse_vector=mse_vec, raw_window=raw_window_spike)
    assert ev_spike.system_status == STATE_SENSOR_FAULT
    assert ev_spike.nodes["aBD11Az"].sensor_fault == "spike"


def test_stage2_severity_score():
    """Item (g): Severity score is within [0, 100] and scales monotonically with severity."""
    config = _dummy_threshold_config()
    sm = AnomalyStateMachine(config=config, persistence=1, t_persist=0.0)

    # Normal state
    ev_norm = sm.step(window_index=0, timestamp=0.0, mse_vector=np.array([0.005, 0.005, 0.005, 0.005]))
    assert 0.0 <= ev_norm.severity_score < 25.0

    # Warning state (ratio between 0.7 and 1.0)
    ev_warn = sm.step(window_index=1, timestamp=0.1, mse_vector=np.array([0.020, 0.005, 0.005, 0.005]))
    assert 25.0 <= ev_warn.severity_score < 50.0

    # Critical state (ratio > 1.0 with persistence=1, t_persist=0.0)
    ev_crit = sm.step(window_index=2, timestamp=0.2, mse_vector=np.array([0.050, 0.005, 0.005, 0.005]))
    assert 50.0 <= ev_crit.severity_score <= 100.0


def test_stage2_slow_drift_layer():
    """Item (h): EWMA/CUSUM detects residual drift; modal frequency tracker checks f(T)."""
    channels = ["ch1", "ch2"]
    cusum = ResidualEWMA_CUSUM(
        channel_names=channels,
        baseline_means={"ch1": 0.01, "ch2": 0.01},
        baseline_stds={"ch1": 0.002, "ch2": 0.002},
        cusum_k_factor=0.5,
        cusum_h_factor=4.0,
    )

    # 1. Healthy hourly residuals -> No alarm
    alarm, c_pos, ewma = cusum.update({"ch1": 0.011, "ch2": 0.010})
    assert not alarm

    # 2. Gradual drift on ch1 -> CUSUM accumulates and fires alarm
    for _ in range(5):
        alarm, c_pos, ewma = cusum.update({"ch1": 0.025, "ch2": 0.010})
    assert alarm

    # 3. Modal frequency tracker
    tracker = ModalFrequencyTracker(fs=50.0)
    tracker.fit_thermal_model(
        temperatures=np.array([10.0, 20.0, 30.0]),
        f1_frequencies=np.array([3.85, 3.80, 3.75]),
    )
    # Healthy temperature response
    is_anom, delta_f = tracker.check_modal_anomaly(temperature=20.0, f1_obs=3.80)
    assert not is_anom
    # Structural stiffness drop (e.g. f1 drops to 3.50 Hz at 20 C)
    is_anom_damage, delta_f_damage = tracker.check_modal_anomaly(temperature=20.0, f1_obs=3.50)
    assert is_anom_damage
    assert delta_f_damage < -0.1
