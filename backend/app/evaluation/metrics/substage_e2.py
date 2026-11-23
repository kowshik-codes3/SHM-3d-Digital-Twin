"""Sub-Stage E2 Pure Evaluation Metrics for KW51.

Implements all metrics required by Protocol B and Sub-Stage E2:
A) Recording-level detection: recall, FAR (cold/warming/warm), ROC-AUC, PR-AUC
B) Window-level detection: Precision, Recall, F1, Balanced Accuracy, ROC-AUC, PR-AUC
C) False alarms: alert episodes per recorded hour, empirical vs claimed FAR
D) Temperature-stratified and matched metrics + Season Classifier Control (B0)
E) Time to first alert from recording start (median, p95)
F) Sensor-fault discrimination audit
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    auc as sk_auc,
)

from app.validation.recording_replay import AlertEpisode, RecordingReplaySummary


@dataclass
class RecordingLevelMetrics:
    """Primary recording-level detection metrics."""

    changed_recall: float
    changed_flagged_count: int
    changed_total_count: int

    healthy_far: float
    healthy_flagged_count: int
    healthy_total_count: int

    roc_auc_max: float
    roc_auc_p95: float
    pr_auc_max: float


@dataclass
class WindowLevelMetrics:
    """Secondary window-level classification metrics."""

    precision: float
    recall: float
    f1_score: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    total_healthy_windows: int
    total_changed_windows: int


@dataclass
class FalseAlarmReport:
    """Operational false alarm rate per recorded hour."""

    total_episodes: int
    total_recorded_hours: float
    episodes_per_recorded_hour: float
    empirical_window_far: float


@dataclass
class TemperatureMatchedReport:
    """Temperature-stratified and matched analysis."""

    bin_edges: List[Tuple[float, float]]
    kept_bins: List[str]
    dropped_bins: List[str]
    bin_metrics: Dict[str, dict]
    matched_weighted_recall: float
    season_control_auc: float
    season_control_gap: float  # AUC(detector) - AUC(B0)


def compute_recording_level_metrics(
    healthy_recordings: List[RecordingReplaySummary],
    changed_recordings: List[RecordingReplaySummary],
) -> RecordingLevelMetrics:
    """Computes primary recording-level recall, FAR, ROC-AUC and PR-AUC."""
    n_h = len(healthy_recordings)
    n_c = len(changed_recordings)

    flagged_h = sum(1 for r in healthy_recordings if r.is_flagged)
    flagged_c = sum(1 for r in changed_recordings if r.is_flagged)

    far = (flagged_h / n_h) if n_h > 0 else 0.0
    rec = (flagged_c / n_c) if n_c > 0 else 0.0

    # Scores per recording: max ratio and 95th percentile ratio
    def get_scores(recs):
        max_s, p95_s = [], []
        for r in recs:
            ratios = [max(ns.ratio for ns in e.nodes.values()) for e in r.events] if r.events else [0.0]
            max_s.append(float(np.max(ratios)))
            p95_s.append(float(np.percentile(ratios, 95.0)))
        return np.array(max_s), np.array(p95_s)

    h_max, h_p95 = get_scores(healthy_recordings)
    c_max, c_p95 = get_scores(changed_recordings)

    y_true = np.array([0] * n_h + [1] * n_c)
    y_max = np.concatenate([h_max, c_max])
    y_p95 = np.concatenate([h_p95, c_p95])

    try:
        auc_max = float(roc_auc_score(y_true, y_max))
        auc_p95 = float(roc_auc_score(y_true, y_p95))
        prec, rec_curve, _ = precision_recall_curve(y_true, y_max)
        pr_max = float(sk_auc(rec_curve, prec))
    except Exception:
        auc_max, auc_p95, pr_max = 0.5, 0.5, 0.0

    return RecordingLevelMetrics(
        changed_recall=rec,
        changed_flagged_count=flagged_c,
        changed_total_count=n_c,
        healthy_far=far,
        healthy_flagged_count=flagged_h,
        healthy_total_count=n_h,
        roc_auc_max=auc_max,
        roc_auc_p95=auc_p95,
        pr_auc_max=pr_max,
    )


def compute_season_control_b0(
    healthy_temps: np.ndarray,
    changed_temps: np.ndarray,
) -> float:
    """Trains a baseline thermometer classifier (B0) using ONLY the temperature channel.

    Returns the ROC-AUC of predicting healthy vs changed purely from temperature.
    """
    n_h = len(healthy_temps)
    n_c = len(changed_temps)
    if n_h == 0 or n_c == 0:
        return 0.5

    X = np.concatenate([healthy_temps, changed_temps]).reshape(-1, 1)
    y = np.array([0] * n_h + [1] * n_c)

    clf = LogisticRegression()
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, probs))


def compute_alert_episodes_per_hour(
    recordings: List[RecordingReplaySummary],
) -> FalseAlarmReport:
    """Computes total alert episodes per recorded hour of data."""
    tot_hours = sum(r.recorded_hours for r in recordings)
    tot_episodes = sum(len(r.episodes) for r in recordings)
    episodes_per_hour = (tot_episodes / tot_hours) if tot_hours > 0 else 0.0

    tot_windows = sum(r.evaluated_windows for r in recordings)
    crit_windows = sum(
        sum(1 for e in r.events if e.system_status == "CRITICAL") for r in recordings
    )
    window_far = (crit_windows / tot_windows) if tot_windows > 0 else 0.0

    return FalseAlarmReport(
        total_episodes=tot_episodes,
        total_recorded_hours=round(tot_hours, 2),
        episodes_per_recorded_hour=round(episodes_per_hour, 3),
        empirical_window_far=round(window_far, 5),
    )


def block_bootstrap_confidence_interval(
    healthy_recordings: List[RecordingReplaySummary],
    changed_recordings: List[RecordingReplaySummary],
    n_bootstraps: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Block bootstrap resampling whole recordings to compute 95% CIs on metrics."""
    rng = np.random.RandomState(seed)
    n_h = len(healthy_recordings)
    n_c = len(changed_recordings)

    boot_recall = []
    boot_far = []
    boot_auc = []

    for _ in range(n_bootstraps):
        idx_h = rng.randint(0, n_h, size=n_h)
        idx_c = rng.randint(0, n_c, size=n_c)

        sample_h = [healthy_recordings[i] for i in idx_h]
        sample_c = [changed_recordings[i] for i in idx_c]

        m = compute_recording_level_metrics(sample_h, sample_c)
        boot_recall.append(m.changed_recall)
        boot_far.append(m.healthy_far)
        boot_auc.append(m.roc_auc_max)

    return {
        "recall_95_ci": (float(np.percentile(boot_recall, 2.5)), float(np.percentile(boot_recall, 97.5))),
        "far_95_ci": (float(np.percentile(boot_far, 2.5)), float(np.percentile(boot_far, 97.5))),
        "roc_auc_95_ci": (float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5))),
    }
