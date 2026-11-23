"""Metric computation and evaluation analytics for SHM real-data validation.

Calculates:
- False alarm rate (FAR) per hour and per window on healthy-test data.
- Detection delay (seconds from fault onset to first confirmed CRITICAL state).
- Window-level Precision, Recall, F1-Score, and ROC-AUC.
- Spatial localization node distribution / confusion counts.
- Empirical FAR comparison across EVT POT, 99.9th percentile, and SPC 3-sigma.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from app.ml.anomaly_engine import STATE_CRITICAL, STATE_NORMAL, STATE_WARNING
from app.validation.evaluate import ReplayResults
from app.validation.evt import EVTCalibrationConfig


@dataclass
class ValidationMetrics:
    """Full metric report for a validation experiment run."""

    experiment_name: str
    channels: List[str]
    has_temperature: bool
    threshold_method: str  # 'evt', 'percentile_999', 'spc_3sigma'

    # Operational metrics
    healthy_test_hours: float
    healthy_test_windows: int
    false_alarms_total: int
    false_alarms_per_hour: float
    empirical_far_per_window: float

    # Fault detection metrics
    damaged_test_hours: float
    damaged_test_windows: int
    detection_delay_seconds: Optional[float]
    detection_delay_windows: Optional[int]
    is_detected: bool

    # Classification metrics
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

    # Defect localization
    localization_counts: Dict[str, int]
    primary_localized_node: Optional[str]

    # Threshold comparison
    threshold_far_comparison: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_threshold_far_comparison(
    val_residuals: np.ndarray,
    test_h_residuals: np.ndarray,
    evt_config: EVTCalibrationConfig,
    channel_names: List[str],
) -> Dict[str, float]:
    """Compares empirical FAR on healthy test residuals using EVT vs 99.9th percentile vs 3-sigma."""
    n_test = len(test_h_residuals)
    if n_test == 0:
        return {"evt": 0.0, "percentile_999": 0.0, "spc_3sigma": 0.0}

    # An alert occurs for a window if any channel exceeds its calibrated threshold
    evt_exceed = np.zeros(n_test, dtype=bool)
    perc_exceed = np.zeros(n_test, dtype=bool)
    spc_exceed = np.zeros(n_test, dtype=bool)

    for c, name in enumerate(channel_names):
        ch = evt_config.channels[name]
        col = test_h_residuals[:, c]
        evt_exceed |= (col > ch.tau_evt)
        perc_exceed |= (col > ch.tau_percentile_999)
        spc_exceed |= (col > ch.tau_spc_3sigma)

    return {
        "evt_empirical_far": float(np.mean(evt_exceed)),
        "percentile_999_empirical_far": float(np.mean(perc_exceed)),
        "spc_3sigma_empirical_far": float(np.mean(spc_exceed)),
    }


def compute_validation_metrics(
    healthy_results: ReplayResults,
    damaged_results: ReplayResults,
    evt_config: EVTCalibrationConfig,
    channel_names: List[str],
    has_temperature: bool,
    experiment_name: str = "kw51_validation",
    threshold_method: str = "evt",
    fs: float = 50.0,
    eval_stride: int = 25,
) -> ValidationMetrics:
    """Computes operational and classification metrics from streaming replay results."""
    # 1. Healthy test durations and false alarms
    h_duration_sec = healthy_results.total_frames / fs
    h_duration_hours = h_duration_sec / 3600.0
    h_windows = healthy_results.evaluated_windows

    # Count false alarms (confirmed CRITICAL state during healthy test)
    false_critical_count = sum(1 for s in healthy_results.system_states if s == STATE_CRITICAL)
    far_per_hour = (false_critical_count / h_duration_hours) if h_duration_hours > 0 else 0.0
    far_per_window = (false_critical_count / h_windows) if h_windows > 0 else 0.0

    # 2. Damaged test metrics and detection delay
    d_duration_sec = damaged_results.total_frames / fs
    d_duration_hours = d_duration_sec / 3600.0
    d_windows = damaged_results.evaluated_windows

    first_critical_idx: Optional[int] = None
    for idx, s in enumerate(damaged_results.system_states):
        if s == STATE_CRITICAL:
            first_critical_idx = idx
            break

    if first_critical_idx is not None:
        # Time to detection = window index * stride in seconds
        detection_delay_sec = float((first_critical_idx * eval_stride) / fs)
        is_detected = True
    else:
        detection_delay_sec = None
        is_detected = False

    # 3. Window-level classification metrics
    # Healthy windows = label 0, Damaged windows = label 1
    y_true = np.array([0] * h_windows + [1] * d_windows, dtype=int)
    
    # Binary predictions: CRITICAL = 1, WARNING/NORMAL = 0
    preds_h = [1 if s == STATE_CRITICAL else 0 for s in healthy_results.system_states]
    preds_d = [1 if s == STATE_CRITICAL else 0 for s in damaged_results.system_states]
    y_pred = np.array(preds_h + preds_d, dtype=int)

    # Anomaly continuous scores: max channel ratio (e_{b, c} / \tau_c)
    scores_h = healthy_results.max_ratios
    scores_d = damaged_results.max_ratios
    y_scores = np.array(scores_h + scores_d, dtype=float)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.0

    # 4. Localization analysis: tabulate defect nodes during damaged critical states
    loc_counts: Dict[str, int] = {ch: 0 for ch in channel_names}
    for node, state in zip(damaged_results.localized_nodes, damaged_results.system_states):
        if state == STATE_CRITICAL and node is not None:
            loc_counts[node] = loc_counts.get(node, 0) + 1

    primary_node = None
    if any(loc_counts.values()):
        primary_node = max(loc_counts.keys(), key=lambda k: loc_counts[k])

    # 5. Threshold comparison
    threshold_comp = compute_threshold_far_comparison(
        val_residuals=np.empty((0, len(channel_names))),
        test_h_residuals=healthy_results.per_channel_mse,
        evt_config=evt_config,
        channel_names=channel_names,
    )

    return ValidationMetrics(
        experiment_name=experiment_name,
        channels=channel_names,
        has_temperature=has_temperature,
        threshold_method=threshold_method,
        healthy_test_hours=round(h_duration_hours, 4),
        healthy_test_windows=h_windows,
        false_alarms_total=false_critical_count,
        false_alarms_per_hour=round(far_per_hour, 4),
        empirical_far_per_window=round(far_per_window, 6),
        damaged_test_hours=round(d_duration_hours, 4),
        damaged_test_windows=d_windows,
        detection_delay_seconds=round(detection_delay_sec, 4) if detection_delay_sec is not None else None,
        detection_delay_windows=first_critical_idx,
        is_detected=is_detected,
        precision=round(float(prec), 4),
        recall=round(float(rec), 4),
        f1_score=round(float(f1), 4),
        roc_auc=round(float(auc), 4),
        localization_counts=loc_counts,
        primary_localized_node=primary_node,
        threshold_far_comparison=threshold_comp,
    )
