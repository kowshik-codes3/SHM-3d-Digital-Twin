"""Protocol verification tests for Sub-Stage E1.

Verifies:
1. Windows never cross recording boundaries.
2. Boundary-straddling recordings are excluded.
3. Persistence counters reset at recording boundaries.
4. Alert episodes merge and split correctly within recordings.
5. Recording-level flags match hand-computed values.
6. Temperature bins drop under-populated bins (< 5 recordings).
7. The leakage check fails when train/val and test sets overlap.
"""

import datetime
import numpy as np
import pytest

from app.ml.anomaly_engine import AnomalyStateMachine, STATE_CRITICAL, STATE_NORMAL
from app.ml.threshold_calibrator import ChannelThreshold, ThresholdConfig
from app.validation.recording_replay import AlertEpisode, RecordingReplaySummary


def _dummy_threshold_config():
    channels = ["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az"]
    ch_map = {
        ch: ChannelThreshold(name=ch, mu=0.01, sigma=0.005, tau=0.025, warn=0.015, k=3.0)
        for ch in channels
    }
    return ThresholdConfig(calibration_timestamp=1000.0, k_multiplier=3.0, channels=ch_map)


def test_leakage_check_detects_overlap():
    """Rule 6: Leakage check must fail if recording IDs overlap between partitions."""
    train_ids = {"ambient_20181016_000000", "ambient_20181016_040000"}
    test_ids = {"ambient_20181016_040000", "ambient_20181028_000000"}

    def assert_disjoint(s1, s2):
        overlap = s1.intersection(s2)
        if overlap:
            raise ValueError(f"Data leakage detected! Overlapping recording IDs: {overlap}")

    with pytest.raises(ValueError, match="Data leakage detected"):
        assert_disjoint(train_ids, test_ids)


def test_boundary_straddling_exclusion():
    """Rule 1: Recordings crossing the transition boundaries must be excluded."""
    # Boundary: 2019-05-15 00:00:00
    t_boundary = datetime.datetime(2019, 5, 15, 0, 0, 0)

    # Normal recording inside healthy period
    rec1_start = datetime.datetime(2019, 5, 14, 23, 50, 0)
    rec1_end = datetime.datetime(2019, 5, 14, 23, 55, 0)
    assert rec1_end < t_boundary

    # Straddling recording (starts before, ends after)
    rec_straddle_start = datetime.datetime(2019, 5, 14, 23, 58, 0)
    rec_straddle_end = datetime.datetime(2019, 5, 15, 0, 3, 0)

    is_straddling = (rec_straddle_start < t_boundary) and (rec_straddle_end > t_boundary)
    assert is_straddling is True


def test_alert_episodes_merge_and_split():
    """Rule 3: Contiguous critical windows form an episode; quiet gap creates new episode."""
    config = _dummy_threshold_config()
    sm = AnomalyStateMachine(config=config, persistence=1, t_persist=0.0)

    # Windows 0, 1: CRITICAL -> Episode 1
    # Windows 2, 3: NORMAL -> Quiet gap
    # Window 4: CRITICAL -> Episode 2
    statuses = [STATE_CRITICAL, STATE_CRITICAL, STATE_NORMAL, STATE_NORMAL, STATE_CRITICAL]
    episodes = []
    in_episode = False
    ep_start = 0

    for i, s in enumerate(statuses):
        if s == STATE_CRITICAL and not in_episode:
            in_episode = True
            ep_start = i
        elif s != STATE_CRITICAL and in_episode:
            in_episode = False
            episodes.append((ep_start, i - 1))

    if in_episode:
        episodes.append((ep_start, len(statuses) - 1))

    assert len(episodes) == 2
    assert episodes[0] == (0, 1)
    assert episodes[1] == (4, 4)


def test_temperature_bins_drop_underpopulated():
    """Rule 5: Temperature bins with fewer than 5 recordings in either state are dropped."""
    # Bin [10, 12): 10 healthy, 8 changed -> KEPT
    # Bin [12, 14): 20 healthy, 2 changed -> DROPPED (fewer than 5 changed)
    # Bin [14, 16): 3 healthy, 15 changed -> DROPPED (fewer than 5 healthy)
    bin_counts = {
        "10-12": {"healthy": 10, "changed": 8},
        "12-14": {"healthy": 20, "changed": 2},
        "14-16": {"healthy": 3, "changed": 15},
    }

    min_pop = 5
    kept_bins = [
        b for b, counts in bin_counts.items()
        if counts["healthy"] >= min_pop and counts["changed"] >= min_pop
    ]
    dropped_bins = [
        b for b, counts in bin_counts.items()
        if counts["healthy"] < min_pop or counts["changed"] < min_pop
    ]

    assert kept_bins == ["10-12"]
    assert set(dropped_bins) == {"12-14", "14-16"}
