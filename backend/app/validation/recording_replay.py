"""Recording-aware streaming replay for KW51 ambient recordings.

Enforces Dataset-Specific Rule O3:
1. Windows never cross recording boundaries.
2. AnomalyStateMachine exceedance counters and duration timers reset at every file boundary.
3. False Alarm Rates are computed strictly per recorded hour of observation.
4. Alerts are grouped into alert episodes within each recording.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.ml.anomaly_engine import AnomalyEvent, AnomalyStateMachine, STATE_CRITICAL, STATE_SENSOR_FAULT, STATE_WARNING
from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.stream_buffer import SensorStreamBuffer


@dataclass
class AlertEpisode:
    """Contiguous run of confirmed CRITICAL windows inside a single recording."""

    recording_id: str
    start_window: int
    end_window: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    max_severity: float
    suspected_node: Optional[str]


@dataclass
class RecordingReplaySummary:
    """Summary of replaying one discrete recording file."""

    recording_id: str
    file_duration_sec: float
    recorded_hours: float
    total_frames: int
    evaluated_windows: int
    episodes: List[AlertEpisode]
    is_flagged: bool              # True if at least 1 alert episode occurred
    sensor_fault_count: int
    events: List[AnomalyEvent]
    first_alert_time_sec: Optional[float]  # Time to first alert from recording start


@dataclass
class MultiRecordingResults:
    """Aggregated results across an entire partition of discrete recordings."""

    partition_name: str
    total_recordings: int
    flagged_recordings: int
    recording_level_far_or_recall: float
    total_recorded_hours: float
    total_episodes: int
    episodes_per_recorded_hour: float
    recordings: List[RecordingReplaySummary]


def replay_single_recording(
    raw_recording: np.ndarray,
    recording_id: str,
    scaler: StructuralDataScaler,
    model: LSTMAutoencoder,
    state_machine: AnomalyStateMachine,
    device: Optional[torch.device] = None,
    fs: float = 50.0,
    eval_stride: int = 25,
    temperature_conditioned: bool = True,
    quiet_period_windows: int = 5,
) -> RecordingReplaySummary:
    """Replays a single 5-minute recording file with strict boundary reset."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Rule O3: Persistence counters MUST reset at recording boundaries
    state_machine.reset()

    n_samples, n_channels = raw_recording.shape
    buffer = SensorStreamBuffer(window_size=50, num_channels=n_channels)
    dt = 1.0 / fs
    file_duration_sec = n_samples * dt

    events: List[AnomalyEvent] = []
    episodes: List[AlertEpisode] = []

    current_episode_start_idx: Optional[int] = None
    current_episode_start_time: Optional[float] = None
    current_episode_severities: List[float] = []
    current_episode_nodes: List[Optional[str]] = []
    first_alert_time: Optional[float] = None
    sensor_fault_count = 0
    quiet_counter = 0

    window_idx = 0

    with torch.no_grad():
        for i in range(n_samples):
            frame = raw_recording[i]
            t_rel = i * dt
            buffer.push_frame(frame)

            if buffer.is_full() and (i % eval_stride == 0):
                x_tensor = buffer.get_standardized_tensor(scaler, dev)
                recon, _ = model(x_tensor)

                if temperature_conditioned:
                    x_target = x_tensor[:, :, :4]
                    raw_eval_window = buffer.get_raw_window()[:, :4]
                else:
                    x_target = x_tensor
                    raw_eval_window = buffer.get_raw_window()

                mse_vec = ((x_target - recon) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

                event = state_machine.step(
                    window_index=window_idx,
                    timestamp=t_rel,
                    mse_vector=mse_vec,
                    raw_window=raw_eval_window,
                )
                events.append(event)

                if event.system_status == STATE_SENSOR_FAULT:
                    sensor_fault_count += 1

                # Track alert episodes (runs of confirmed CRITICAL windows)
                if event.system_status == STATE_CRITICAL:
                    if first_alert_time is None:
                        first_alert_time = t_rel

                    if current_episode_start_idx is None:
                        current_episode_start_idx = window_idx
                        current_episode_start_time = t_rel
                        current_episode_severities = [event.severity_score]
                        current_episode_nodes = [event.localized_defect_node]
                    else:
                        current_episode_severities.append(event.severity_score)
                        current_episode_nodes.append(event.localized_defect_node)
                    quiet_counter = 0
                else:
                    if current_episode_start_idx is not None:
                        quiet_counter += 1
                        if quiet_counter >= quiet_period_windows:
                            # End current episode
                            episodes.append(
                                AlertEpisode(
                                    recording_id=recording_id,
                                    start_window=current_episode_start_idx,
                                    end_window=window_idx - quiet_counter,
                                    start_time_sec=current_episode_start_time or 0.0,
                                    end_time_sec=t_rel - (quiet_counter * eval_stride * dt),
                                    duration_sec=max(0.1, (t_rel - (quiet_counter * eval_stride * dt)) - (current_episode_start_time or 0.0)),
                                    max_severity=max(current_episode_severities) if current_episode_severities else 0.0,
                                    suspected_node=max(set(current_episode_nodes), key=current_episode_nodes.count) if current_episode_nodes else None,
                                )
                            )
                            current_episode_start_idx = None
                            current_episode_start_time = None
                            current_episode_severities = []
                            current_episode_nodes = []
                            quiet_counter = 0

                window_idx += 1

    # End episode at boundary of recording if still open
    if current_episode_start_idx is not None:
        episodes.append(
            AlertEpisode(
                recording_id=recording_id,
                start_window=current_episode_start_idx,
                end_window=window_idx - 1,
                start_time_sec=current_episode_start_time or 0.0,
                end_time_sec=file_duration_sec,
                duration_sec=file_duration_sec - (current_episode_start_time or 0.0),
                max_severity=max(current_episode_severities) if current_episode_severities else 0.0,
                suspected_node=max(set(current_episode_nodes), key=current_episode_nodes.count) if current_episode_nodes else None,
            )
        )

    return RecordingReplaySummary(
        recording_id=recording_id,
        file_duration_sec=file_duration_sec,
        recorded_hours=file_duration_sec / 3600.0,
        total_frames=n_samples,
        evaluated_windows=len(events),
        episodes=episodes,
        is_flagged=len(episodes) > 0,
        sensor_fault_count=sensor_fault_count,
        events=events,
        first_alert_time_sec=first_alert_time,
    )


def replay_recording_partition(
    recordings_list: List[Tuple[str, np.ndarray]],
    partition_name: str,
    scaler: StructuralDataScaler,
    model: LSTMAutoencoder,
    state_machine: AnomalyStateMachine,
    device: Optional[torch.device] = None,
    fs: float = 50.0,
    eval_stride: int = 25,
    temperature_conditioned: bool = True,
) -> MultiRecordingResults:
    """Replays an entire sequence of discrete recordings with file-aware statistics."""
    summaries = []
    for rec_id, raw_arr in recordings_list:
        summary = replay_single_recording(
            raw_recording=raw_arr,
            recording_id=rec_id,
            scaler=scaler,
            model=model,
            state_machine=state_machine,
            device=device,
            fs=fs,
            eval_stride=eval_stride,
            temperature_conditioned=temperature_conditioned,
        )
        summaries.append(summary)

    total_recs = len(summaries)
    flagged_recs = sum(1 for s in summaries if s.is_flagged)
    total_hours = sum(s.recorded_hours for s in summaries)
    total_episodes = sum(len(s.episodes) for s in summaries)
    episodes_per_hour = (total_episodes / total_hours) if total_hours > 0 else 0.0
    rate = (flagged_recs / total_recs) if total_recs > 0 else 0.0

    return MultiRecordingResults(
        partition_name=partition_name,
        total_recordings=total_recs,
        flagged_recordings=flagged_recs,
        recording_level_far_or_recall=rate,
        total_recorded_hours=total_hours,
        total_episodes=total_episodes,
        episodes_per_recorded_hour=episodes_per_hour,
        recordings=summaries,
    )
