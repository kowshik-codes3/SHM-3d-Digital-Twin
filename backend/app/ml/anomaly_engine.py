"""Multi-state Anomaly State Machine with persistence filtering and spatial localization.

Consumes per-channel MSE residuals from the LSTM Autoencoder and applies
statistical process control to classify structural health states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .threshold_calibrator import ChannelThreshold, ThresholdConfig

# ── State constants ────────────────────────────────────────────────────────────
STATE_NORMAL       = "NORMAL"
STATE_WARNING      = "WARNING"
STATE_CRITICAL     = "CRITICAL"
STATE_SENSOR_FAULT = "SENSOR_FAULT"


@dataclass
class NodeStatus:
    """Per-node residual report for one evaluation window."""

    name: str
    mse: float
    threshold: float     # τ_c
    ratio: float         # e_{b,c} / τ_c
    status: str          # NORMAL / WARNING / CRITICAL / SENSOR_FAULT
    sensor_fault: Optional[str] = None  # None, 'flatline', 'spike', 'drift'


@dataclass
class AnomalyEvent:
    """Full structural health event emitted for one evaluation window."""

    window_index: int
    timestamp: float
    system_status: str                          # NORMAL / WARNING / CRITICAL / SENSOR_FAULT
    localized_defect_node: Optional[str]        # Primary suspected node (None when NORMAL)
    nodes: Dict[str, NodeStatus]
    raw_mse: np.ndarray = field(repr=False, compare=False)  # (C,) vector
    top_candidates: List[Tuple[str, float]] = field(default_factory=list)  # Top-3 (node, confidence)
    severity_score: float = 0.0                 # 0-100 severity index
    sensor_faults: Dict[str, str] = field(default_factory=dict)  # Faulty sensor -> fault type


class AnomalyStateMachine:
    """Sliding-window state machine with P-window persistence filter.

    Parameters
    ----------
    config           : Calibrated ThresholdConfig (from ThresholdCalibrator).
    persistence      : Number of consecutive exceedances before CRITICAL is
                       declared (anti-flapping guard). Default = 3.
    """

    def __init__(
        self,
        config: ThresholdConfig,
        persistence: int = 3,
        t_persist: float = 0.05,
        adjacency_matrix: Optional[np.ndarray] = None,
        enable_sensor_fault_detection: bool = True,
        top_k: int = 3,
    ) -> None:
        self.config = config
        self.persistence = persistence
        self.t_persist = t_persist  # Required continuous exceedance duration in seconds
        self.adjacency_matrix = adjacency_matrix
        self.enable_sensor_fault_detection = enable_sensor_fault_detection
        self.top_k = top_k
        self.channel_names: List[str] = list(config.channels.keys())

        # Per-channel rolling exceedance counters and timestamp tracking
        self._exceedance_counts: Dict[str, int] = {
            name: 0 for name in self.channel_names
        }
        self._exceedance_start_times: Dict[str, Optional[float]] = {
            name: None for name in self.channel_names
        }

    # ──────────────────────────────────────────────────────────────────────
    def detect_sensor_fault(self, name: str, window_data: np.ndarray) -> Optional[str]:
        """Detects hardware sensor faults: flatline, spike/glitch, or DC drift.

        Args:
            name: Sensor channel name
            window_data: 1D array of length T (raw sensor readings)

        Returns:
            Fault type string ('flatline', 'spike', 'drift') or None if healthy/structural.
        """
        arr = np.asarray(window_data, dtype=np.float64)
        v_std = float(np.std(arr))
        if v_std < 1e-5:
            return "flatline"

        diff = np.diff(arr)
        max_jump = float(np.max(np.abs(diff))) if len(diff) > 0 else 0.0
        if v_std > 0:
            m = float(np.mean(arr))
            m4 = float(np.mean((arr - m) ** 4))
            kurt = (m4 / (v_std ** 4)) if v_std > 0 else 0.0
            if kurt > 20.0 or (max_jump / v_std) > 6.0:
                return "spike"

        v_mean = float(np.mean(arr))
        if v_std > 0 and abs(v_mean) > 8.0 * v_std:
            return "drift"

        return None

    # ──────────────────────────────────────────────────────────────────────
    def _classify_channel(
        self,
        name: str,
        mse: float,
    ) -> str:
        """Single-channel state before applying persistence filter."""
        ch: ChannelThreshold = self.config.channels[name]
        if mse > ch.tau:
            return STATE_CRITICAL
        elif mse > ch.warn:
            return STATE_WARNING
        return STATE_NORMAL

    # ──────────────────────────────────────────────────────────────────────
    def step(
        self,
        window_index: int,
        timestamp: float,
        mse_vector: np.ndarray,         # (C,) per-channel MSE
        raw_window: Optional[np.ndarray] = None,  # (T, C) optional raw data for sensor-fault checks
    ) -> AnomalyEvent:
        """Evaluate one window and emit an AnomalyEvent.

        Args:
            window_index : Sequential index of the sliding window.
            timestamp    : Unix timestamp of the window.
            mse_vector   : Per-channel MSE array of shape (C,).
            raw_window   : Optional raw (T, C) buffer for hardware sensor fault checks.

        Returns:
            AnomalyEvent with per-node statuses, system status, top-3 candidates, and severity.
        """
        assert len(mse_vector) == len(self.channel_names), (
            f"Expected {len(self.channel_names)} channels, got {len(mse_vector)}"
        )

        nodes: Dict[str, NodeStatus] = {}
        confirmed_critical: List[str] = []
        sensor_faults: Dict[str, str] = {}

        for c, name in enumerate(self.channel_names):
            ch = self.config.channels[name]
            mse_c = float(mse_vector[c])
            raw_state = self._classify_channel(name, mse_c)

            # ── 1. Sensor-Fault vs Structural Change Discrimination ────────
            ch_fault = None
            if self.enable_sensor_fault_detection and raw_window is not None:
                ch_fault = self.detect_sensor_fault(name, raw_window[:, c])

            if ch_fault is not None:
                sensor_faults[name] = ch_fault
                confirmed_state = STATE_SENSOR_FAULT
                # Hardware failure: reset structural exceedance tracking
                self._exceedance_counts[name] = 0
                self._exceedance_start_times[name] = None
            else:
                # ── 2. Time-based + frame persistence filter ───────────────
                if raw_state == STATE_CRITICAL:
                    self._exceedance_counts[name] += 1
                    if self._exceedance_start_times[name] is None:
                        self._exceedance_start_times[name] = timestamp
                elif raw_state == STATE_NORMAL:
                    self._exceedance_counts[name] = 0
                    self._exceedance_start_times[name] = None

                duration = (timestamp - self._exceedance_start_times[name]) if self._exceedance_start_times[name] is not None else 0.0
                is_persistent = (
                    self._exceedance_counts[name] >= self.persistence
                    and (self.t_persist <= 0.0 or duration >= self.t_persist)
                )

                if is_persistent:
                    confirmed_state = STATE_CRITICAL
                    confirmed_critical.append(name)
                elif raw_state in (STATE_CRITICAL, STATE_WARNING):
                    # While CRITICAL count is building towards P (or duration building), show WARNING (never NORMAL)
                    confirmed_state = STATE_WARNING
                else:
                    confirmed_state = STATE_NORMAL

            ratio = mse_c / ch.tau if ch.tau > 0 else 0.0
            nodes[name] = NodeStatus(
                name=name,
                mse=mse_c,
                threshold=ch.tau,
                ratio=round(ratio, 4),
                status=confirmed_state,
                sensor_fault=ch_fault,
            )

        # ── 3. Spatial Localization & Top-3 Candidates with Confidence ───────
        smoothed_ratios: Dict[str, float] = {}
        if self.adjacency_matrix is not None:
            r_vec = np.array([nodes[n].ratio for n in self.channel_names], dtype=np.float32)
            smoothed_vec = self.adjacency_matrix @ r_vec
            smoothed_ratios = {self.channel_names[i]: float(smoothed_vec[i]) for i in range(len(self.channel_names))}
        else:
            smoothed_ratios = {n: nodes[n].ratio for n in self.channel_names}

        # Rank channels by anomaly ratio
        ranked = sorted(
            [(n, smoothed_ratios[n]) for n in self.channel_names],
            key=lambda x: x[1],
            reverse=True,
        )
        top_k_list = ranked[: min(self.top_k, len(self.channel_names))]
        r_vals = np.array([max(0.01, r) for _, r in top_k_list], dtype=np.float32)
        exp_r = np.exp(np.clip(r_vals - np.max(r_vals), -10, 10))
        confs = exp_r / (np.sum(exp_r) + 1e-9)
        top_candidates = [
            (top_k_list[i][0], round(float(confs[i]), 4))
            for i in range(len(top_k_list))
        ]

        defect_node = top_candidates[0][0] if confirmed_critical else None

        # ── 4. Continuous Severity Score 0-100 ──────────────────────────────
        max_ratio = max(nodes[n].ratio for n in self.channel_names)
        max_duration = max(
            [(timestamp - self._exceedance_start_times[n]) for n in self.channel_names if self._exceedance_start_times[n] is not None]
            or [0.0]
        )
        n_affected = len(confirmed_critical)
        n_ch = len(self.channel_names)

        if confirmed_critical:
            s_ratio = min(25.0, 12.5 * max(0.0, max_ratio - 1.0))
            s_dur = min(15.0, 15.0 * (max_duration / 5.0))
            s_spatial = 10.0 * (n_affected / n_ch)
            severity = min(100.0, 50.0 + s_ratio + s_dur + s_spatial)
        elif any(ns.status == STATE_WARNING for ns in nodes.values()):
            severity = min(49.0, 25.0 + 24.0 * max(0.0, (max_ratio - 0.7) / 0.3))
        elif sensor_faults:
            severity = 30.0  # Hardware issue
        else:
            severity = min(24.0, 24.0 * max(0.0, max_ratio / 0.7))

        # ── 5. System Status Resolution ──────────────────────────────────────
        if confirmed_critical:
            system_status = STATE_CRITICAL
        elif any(ns.status == STATE_WARNING for ns in nodes.values()):
            system_status = STATE_WARNING
        elif sensor_faults:
            system_status = STATE_SENSOR_FAULT
        else:
            system_status = STATE_NORMAL

        return AnomalyEvent(
            window_index=window_index,
            timestamp=timestamp,
            system_status=system_status,
            localized_defect_node=defect_node,
            nodes=nodes,
            raw_mse=mse_vector.copy(),
            top_candidates=top_candidates,
            severity_score=round(float(severity), 2),
            sensor_faults=sensor_faults,
        )

    # ──────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset all persistence counters (use between distinct test sequences)."""
        for name in self._exceedance_counts:
            self._exceedance_counts[name] = 0
            self._exceedance_start_times[name] = None
            self._exceedance_start_times[name] = None


# ── Heatmap plotting utility ──────────────────────────────────────────────────

def plot_localization_heatmap(
    mse_matrix: np.ndarray,           # (M, C) per-window per-channel MSE
    channel_names: List[str],
    tau_values: np.ndarray,           # (C,) UCL thresholds
    output_dir: str = "reports/phase5",
) -> None:
    """Save a coloured heatmap of per-channel MSE over test time with threshold overlays.

    Args:
        mse_matrix    : (M, C) residual matrix over test windows.
        channel_names : Channel label list of length C.
        tau_values    : (C,) threshold array for normalisation.
        output_dir    : Output folder.
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    # Normalise by τ_c so 1.0 = threshold crossing
    ratio_matrix = mse_matrix / (tau_values[None, :] + 1e-9)   # (M, C)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(
        ratio_matrix.T,              # (C, M) – channels on y-axis
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=3.0,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(channel_names)))
    ax.set_yticklabels(
        [n.replace("Sensor_", "") for n in channel_names], fontsize=9
    )
    ax.set_xlabel("Sliding Window Index")
    ax.set_title("Per-Channel MSE / τ Ratio Over Test Stream\n"
                 "(1.0 = threshold; red = critical)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("MSE / τ ratio")
    fig.tight_layout()

    outfile = p / "localization_heatmap.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outfile}")
