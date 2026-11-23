"""Unit and regression tests for Phase 3: Time-Based Persistence and Spatial Graph Localization."""

import numpy as np
import pytest

from app.core.structures import DEFAULT_BRIDGE_CONFIG
from app.ml.anomaly_engine import AnomalyStateMachine, STATE_CRITICAL, STATE_NORMAL, STATE_WARNING
from app.ml.threshold_calibrator import ChannelThreshold, ThresholdConfig


@pytest.fixture
def dummy_threshold_config():
    channels = {
        name: ChannelThreshold(
            name=name,
            mu=0.01,
            sigma=0.002,
            tau=0.04,
            warn=0.02,
            k=3.0,
        )
        for name in DEFAULT_BRIDGE_CONFIG.channel_names
    }
    return ThresholdConfig(
        calibration_timestamp=1700000000.0,
        k_multiplier=3.0,
        channels=channels,
    )


def test_spatial_adjacency_matrix():
    """Verify Gaussian spatial adjacency matrix properly smooths across accelerometer channels."""
    adj = DEFAULT_BRIDGE_CONFIG.compute_adjacency_matrix(sigma=10.0)
    W = adj["matrix"]
    names = adj["channel_names"]

    assert W.shape == (5, 5)
    # Row normalization check
    row_sums = W.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(5), rtol=1e-5)

    # Temperature channel (index 4) is isolated from vibration spatial diffusion
    temp_idx = names.index("Sensor_Ambient_Temp")
    assert W[temp_idx, temp_idx] == 1.0
    for j in range(5):
        if j != temp_idx:
            assert W[temp_idx, j] == 0.0


def test_time_based_persistence(dummy_threshold_config):
    """Verify single-frame transient spikes do NOT trigger CRITICAL, but continuous duration does."""
    sm = AnomalyStateMachine(
        config=dummy_threshold_config,
        persistence=3,
        t_persist=0.04,  # Requires 0.04s duration and >= 3 frames
    )

    t0 = 1000.0
    normal_mse = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    spike_mse = np.array([0.01, 0.01, 0.15, 0.01, 0.01])  # Spike on Sensor_MidSpan_Girder

    # Frame 1: Single spike at t=1000.0 -> Should remain NORMAL or WARNING, NOT CRITICAL
    ev1 = sm.step(1, t0, spike_mse)
    assert ev1.system_status != STATE_CRITICAL

    # Frame 2: Immediately returns to normal -> Spike cleared
    ev2 = sm.step(2, t0 + 0.02, normal_mse)
    assert ev2.system_status == STATE_NORMAL
    assert sm._exceedance_counts["Sensor_MidSpan_Girder"] == 0

    # Continuous sustained fault across 0.14s (7 frames at 50Hz)
    times = [t0 + 0.1 + i * 0.02 for i in range(7)]
    for i, t in enumerate(times, start=3):
        ev = sm.step(i, t, spike_mse)
        if i >= 6:  # Duration >= 0.04s and frames >= 3
            assert ev.system_status == STATE_CRITICAL
            assert ev.localized_defect_node == "Sensor_MidSpan_Girder"


def test_physics_informed_spatial_localization(dummy_threshold_config):
    """Verify spatial graph diffusion correctly identifies defect node even with minor channel noise."""
    adj = DEFAULT_BRIDGE_CONFIG.compute_adjacency_matrix(sigma=8.0)
    sm = AnomalyStateMachine(
        config=dummy_threshold_config,
        persistence=2,
        t_persist=0.02,
        adjacency_matrix=adj["matrix"],
    )

    t = 2000.0
    # True defect on Pier_Left (index 1), with adjacent sensors picking up diffused vibration
    diffused_fault = np.array([
        0.05,  # Abutment_Left (adjacent)
        0.18,  # Pier_Left (epicenter)
        0.06,  # MidSpan_Girder (adjacent)
        0.01,  # Pier_Right (far)
        0.01,  # Temp
    ])

    sm.step(1, t, diffused_fault)
    ev2 = sm.step(2, t + 0.03, diffused_fault)

    assert ev2.system_status == STATE_CRITICAL
    assert ev2.localized_defect_node == "Sensor_Pier_Left"
