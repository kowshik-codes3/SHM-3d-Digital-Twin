"""Unit tests for real-data validation module (KW51 loader, EVT, splits, replay)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from app.ml.anomaly_engine import AnomalyStateMachine, STATE_CRITICAL, STATE_NORMAL
from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.stream_buffer import SensorStreamBuffer
from app.validation.dataset import create_sliding_windows
from app.validation.evaluate import replay_stream
from app.validation.evt import EVTCalibrator
from app.validation.loader import KW51Loader


def test_sliding_windows_shape():
    """Verify sliding window generator produces correct (N, window_size, C) shapes."""
    data = np.random.randn(1000, 5).astype(np.float32)
    windows = create_sliding_windows(data, window_size=50, stride=25)
    # Expected: (1000 - 50) // 25 + 1 = 950 // 25 + 1 = 38 + 1 = 39 windows
    assert windows.shape == (39, 50, 5)


def test_evt_calibrator_fitting():
    """Verify EVTCalibrator computes tail parameter, EVT threshold, and percentiles."""
    np.random.seed(42)
    # Log-normal distribution produces positive heavy tails typical of squared prediction errors
    residuals = np.random.lognormal(mean=0.0, sigma=0.4, size=(2000, 3)).astype(np.float32)
    channel_names = ["ch1", "ch2", "ch3"]

    calibrator = EVTCalibrator(tail_quantile=0.95, target_prob=0.001)
    cfg = calibrator.calibrate(residuals, channel_names)

    for ch in channel_names:
        c_thresh = cfg.channels[ch]
        assert c_thresh.tail_u > 0.0
        assert c_thresh.tau_evt >= c_thresh.tail_u
        assert c_thresh.tau_percentile_999 >= c_thresh.tail_u
        assert c_thresh.tau_spc_3sigma > 0.0

    # Convert to production ThresholdConfig
    prod_cfg = cfg.to_threshold_config(threshold_type="evt")
    assert len(prod_cfg.channels) == 3
    assert prod_cfg.channels["ch1"].tau == cfg.channels["ch1"].tau_evt


def test_streaming_replay_integration():
    """Verify streaming evaluation through real SensorStreamBuffer and AnomalyStateMachine."""
    np.random.seed(42)
    channel_names = ["s1", "s2", "s3", "s4", "s5"]
    raw_data = np.random.randn(200, 5).astype(np.float32)

    import pandas as pd
    df = pd.DataFrame(raw_data, columns=channel_names)
    scaler = StructuralDataScaler(channel_names)
    scaler.fit(df)

    # Simple model
    model = LSTMAutoencoder(input_size=5, seq_len=50)
    model.eval()

    # Synthetic calibrator config
    calibrator = EVTCalibrator(tail_quantile=0.95, target_prob=0.001)
    dummy_residuals = np.random.exponential(scale=1.0, size=(500, 5)).astype(np.float32)
    evt_cfg = calibrator.calibrate(dummy_residuals, channel_names)
    thresh_cfg = evt_cfg.to_threshold_config(threshold_type="evt")

    asm = AnomalyStateMachine(config=thresh_cfg, persistence=3, t_persist=0.0)

    results = replay_stream(
        raw_timeseries=raw_data,
        scaler=scaler,
        model=model,
        state_machine=asm,
        device=torch.device("cpu"),
        fs=50.0,
        eval_stride=25,
    )

    assert results.total_frames == 200
    assert results.evaluated_windows > 0
    assert len(results.system_states) == results.evaluated_windows
    assert results.per_channel_mse.shape[1] == 5
