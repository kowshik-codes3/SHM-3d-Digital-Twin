"""Streaming evaluation replaying test data through the REAL production pipeline.

Streams raw multi-channel samples frame-by-frame through:
1. SensorStreamBuffer (circular window buffer)
2. StructuralDataScaler (StandardScaler standardization)
3. LSTMAutoencoder (reconstruction inference)
4. AnomalyStateMachine (persistence filtering & spatial defect localization)

Strictly uses the real production classes with zero separate mock evaluation paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from app.ml.anomaly_engine import AnomalyEvent, AnomalyStateMachine
from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.stream_buffer import SensorStreamBuffer


@dataclass
class ReplayResults:
    """Output summary of streaming continuous data through the real pipeline."""

    events: List[AnomalyEvent]
    total_frames: int
    evaluated_windows: int
    elapsed_time_sec: float
    system_states: List[str]          # 'NORMAL', 'WARNING', 'CRITICAL', 'SENSOR_FAULT'
    localized_nodes: List[Optional[str]]
    max_ratios: List[float]
    per_channel_mse: np.ndarray       # (evaluated_windows, C)


def replay_stream(
    raw_timeseries: np.ndarray,
    scaler: StructuralDataScaler,
    model: LSTMAutoencoder,
    state_machine: AnomalyStateMachine,
    device: Optional[torch.device] = None,
    fs: float = 50.0,
    eval_stride: int = 25,
    start_timestamp: float = 0.0,
    temperature_conditioned: bool = False,
) -> ReplayResults:
    """Replays continuous time-series data through SensorStreamBuffer and AnomalyStateMachine.

    Args:
        raw_timeseries: (T, C) raw unscaled sensor recordings.
        scaler: Fitted StructuralDataScaler.
        model: Trained LSTMAutoencoder in eval mode.
        state_machine: AnomalyStateMachine instance (will be reset before run).
        device: Torch device (CPU / CUDA).
        fs: Sampling rate in Hz (default 50 Hz).
        eval_stride: Step size in samples between evaluations (default 25 = 0.5s at 50Hz).
        start_timestamp: Starting epoch or elapsed seconds.
        temperature_conditioned: If True, model takes (B, T, 5) but outputs (B, T, 4).
                                 Residuals are calculated against the first 4 channels.

    Returns:
        ReplayResults with all emitted AnomalyEvent instances.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    state_machine.reset()

    n_samples, n_channels = raw_timeseries.shape
    buffer = SensorStreamBuffer(window_size=50, num_channels=n_channels)
    dt = 1.0 / fs

    events: List[AnomalyEvent] = []
    system_states: List[str] = []
    localized_nodes: List[Optional[str]] = []
    max_ratios: List[float] = []
    mse_history: List[np.ndarray] = []

    t_start = time.perf_counter()
    window_idx = 0

    with torch.no_grad():
        for i in range(n_samples):
            frame = raw_timeseries[i]
            current_time = start_timestamp + i * dt
            buffer.push_frame(frame)

            # Evaluate when buffer is full and stride interval is reached
            if buffer.is_full() and (i % eval_stride == 0):
                x_tensor = buffer.get_standardized_tensor(scaler, dev)  # (1, 50, C)
                recon, _ = model(x_tensor)                             # (1, 50, C_target)

                if temperature_conditioned:
                    # x_target is accelerometer channels [0:4]
                    x_target = x_tensor[:, :, :4]
                    raw_accel_window = buffer.get_raw_window()[:, :4]
                else:
                    x_target = x_tensor
                    raw_accel_window = buffer.get_raw_window()

                # Channel MSE: mean over time axis 1 -> (C_target,)
                mse_vec = ((x_target - recon) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

                event = state_machine.step(
                    window_index=window_idx,
                    timestamp=current_time,
                    mse_vector=mse_vec,
                    raw_window=raw_accel_window,
                )

                events.append(event)
                system_states.append(event.system_status)
                localized_nodes.append(event.localized_defect_node)
                max_ratio = max(ns.ratio for ns in event.nodes.values())
                max_ratios.append(float(max_ratio))
                mse_history.append(mse_vec)
                window_idx += 1

    elapsed = time.perf_counter() - t_start

    num_eval_channels = 4 if temperature_conditioned else n_channels
    return ReplayResults(
        events=events,
        total_frames=n_samples,
        evaluated_windows=len(events),
        elapsed_time_sec=elapsed,
        system_states=system_states,
        localized_nodes=localized_nodes,
        max_ratios=max_ratios,
        per_channel_mse=np.array(mse_history, dtype=np.float32) if mse_history else np.empty((0, num_eval_channels)),
    )
