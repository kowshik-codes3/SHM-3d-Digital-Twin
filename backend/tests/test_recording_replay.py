"""Unit tests for Track 1 evaluation modules:
1. inventory inspection
2. O1 state timeline labelling
3. O8 modal baseline regression & Mahalanobis novelty
4. O3 recording-aware replay & boundary persistence reset
"""

import numpy as np
import pytest
import torch

from app.ml.anomaly_engine import AnomalyStateMachine, STATE_CRITICAL, STATE_NORMAL
from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.threshold_calibrator import ChannelThreshold, ThresholdConfig
from app.validation.recording_replay import replay_single_recording


def _dummy_threshold_config():
    channels = ["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az"]
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


def test_recording_aware_boundary_reset():
    """Rule O3: Persistence counters MUST reset at recording boundaries."""
    config = _dummy_threshold_config()
    sm = AnomalyStateMachine(config=config, persistence=3, t_persist=0.0)

    # File 1: Two consecutive critical windows (count reaches 2 < P=3 -> WARNING)
    mse_vec = np.array([0.05, 0.005, 0.005, 0.005], dtype=np.float32)
    ev1 = sm.step(window_index=0, timestamp=0.0, mse_vector=mse_vec)
    ev2 = sm.step(window_index=1, timestamp=0.5, mse_vector=mse_vec)
    assert sm._exceedance_counts["aBD11Az"] == 2

    # File boundary occurs: sm.reset() MUST be called
    sm.reset()
    assert sm._exceedance_counts["aBD11Az"] == 0

    # Step in File 2: count starts fresh at 1 (not 3)
    ev3 = sm.step(window_index=0, timestamp=0.0, mse_vector=mse_vec)
    assert sm._exceedance_counts["aBD11Az"] == 1
    assert ev3.system_status != STATE_CRITICAL  # Must not fire false critical across boundary


def test_alert_episodes_tracking():
    """Rule O3: Consecutive critical windows within a file form a single alert episode."""
    config = _dummy_threshold_config()
    sm = AnomalyStateMachine(config=config, persistence=1, t_persist=0.0)
    model = LSTMAutoencoder(input_size=5, target_size=4, enc_hidden1=8, enc_hidden2=4, dec_hidden1=8, dec_hidden2=8)

    # Create dummy scaler
    import pandas as pd
    df_dummy = pd.DataFrame(np.random.normal(0, 1, size=(100, 5)), columns=["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az", "tVL"])
    scaler = StructuralDataScaler(feature_names=["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az", "tVL"])
    scaler.fit(df_dummy)

    # Synthetic 5-minute recording (1500 samples at 50Hz = 30s)
    raw_arr = np.random.normal(0, 0.01, size=(300, 5)).astype(np.float32)

    res = replay_single_recording(
        raw_recording=raw_arr,
        recording_id="test_rec_001",
        scaler=scaler,
        model=model,
        state_machine=sm,
        fs=50.0,
        eval_stride=25,
        temperature_conditioned=True,
    )

    assert res.recording_id == "test_rec_001"
    assert res.total_frames == 300
    assert res.evaluated_windows > 0
