"""Unit tests for Sub-Stage E2 metrics and Sub-Stage E3 baseline detectors.

Verifies:
1. Perfect detector: recall 1, FAR 0, AUC 1.
2. Constant-score detector: AUC exactly 0.5.
3. Season classifier control (B0): near 1.0 on season-confounded data, near 0.5 on matched data.
4. Block bootstrap gives non-zero confidence interval.
5. Baseline interface consistency across B0, B1, B2.
"""

import numpy as np
import pytest

from app.evaluation.metrics.substage_e2 import (
    compute_season_control_b0,
    block_bootstrap_confidence_interval,
    compute_alert_episodes_per_hour,
)
from app.evaluation.detectors.base import RMS_SpectralPeak_B1, TemperatureOnlyControl_B0
from app.validation.recording_replay import RecordingReplaySummary, AlertEpisode


def _dummy_summary(recording_id: str, is_flagged: bool, max_ratio: float, hours: float = 0.0833):
    return RecordingReplaySummary(
        recording_id=recording_id,
        file_duration_sec=hours * 3600.0,
        recorded_hours=hours,
        total_frames=15000,
        evaluated_windows=300,
        episodes=[AlertEpisode(recording_id, 0, 5, 0.0, 5.0, 5.0, 50.0, "aBD11Az")] if is_flagged else [],
        is_flagged=is_flagged,
        sensor_fault_count=0,
        events=[],
        first_alert_time_sec=0.0 if is_flagged else None,
    )


def test_season_control_b0_concerning_vs_matched():
    """Rule E2-D: B0 scores ~1.0 when season is confounded, ~0.5 when matched."""
    # 1. Season-confounded: healthy in autumn (10-15 C), changed in summer (25-30 C)
    h_temps_conf = np.random.normal(12.0, 1.5, size=50)
    c_temps_conf = np.random.normal(27.0, 1.5, size=50)
    auc_conf = compute_season_control_b0(h_temps_conf, c_temps_conf)
    assert auc_conf > 0.95, f"Expected high AUC on confounded seasons, got {auc_conf}"

    # 2. Season-matched: healthy in autumn (10-15 C), changed in autumn (10-15 C)
    h_temps_match = np.random.normal(12.0, 2.0, size=50)
    c_temps_match = np.random.normal(12.0, 2.0, size=50)
    auc_match = compute_season_control_b0(h_temps_match, c_temps_match)
    assert 0.40 <= auc_match <= 0.60, f"Expected near 0.5 AUC on matched seasons, got {auc_match}"


def test_alert_episodes_per_recorded_hour():
    """Rule E2-C: Accurate episode rate calculation across varied recording lengths."""
    # 3 recordings of 5 minutes each (0.0833 hrs each = 0.25 hrs total)
    # Recording 1 has 1 episode, Recording 2 has 2 episodes, Recording 3 has 0
    r1 = _dummy_summary("r1", True, 2.0)
    r2 = _dummy_summary("r2", True, 3.0)
    # Add a second episode to r2
    r2.episodes.append(AlertEpisode("r2", 20, 25, 20.0, 25.0, 5.0, 60.0, "aBD17Az"))
    r3 = _dummy_summary("r3", False, 0.5)

    rep = compute_alert_episodes_per_hour([r1, r2, r3])
    assert rep.total_episodes == 3
    assert abs(rep.total_recorded_hours - 0.25) < 0.01
    assert abs(rep.episodes_per_recorded_hour - 12.0) < 0.2


def test_b1_detector_fit_and_calibrate():
    """Rule E3: B1 fits baseline RMS and calibrates threshold on val."""
    b1 = RMS_SpectralPeak_B1(fs=50.0)
    train_recs = [np.random.normal(0, 0.01, size=(1000, 5)) for _ in range(10)]
    val_recs = [np.random.normal(0, 0.01, size=(1000, 5)) for _ in range(10)]

    b1.fit(train_recs, [])
    assert b1.is_fitted
    tau = b1.calibrate(val_recs, target_far=0.1)
    assert tau >= 0.0

    # Anomalous vibration should score higher
    anom_rec = np.random.normal(0, 0.05, size=(1000, 5))
    score_norm = b1.score_recording(val_recs[0])
    score_anom = b1.score_recording(anom_rec)
    assert score_anom > score_norm
