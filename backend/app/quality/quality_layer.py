"""Stage 4 Data-Quality Layer (app/quality/quality_layer.py).

Performs 10 configurable, baseline-calibrated data-quality checks before sensor
frames reach structural detectors:
1. Gaps/dropouts (missing seq, timestamp gaps, zero/NaN dropouts)
2. Flatline and stuck-at-nonzero (accelerometer/strain only; never temperature)
3. Saturation/clipping (range bounds & flat-top clipping)
4. Spikes (calibrated robust z-score + kurtosis)
5. Drift/offset (DC bias relative to calibrated baseline)
6. Noise and gain change (high-freq PSD noise floor & sibling-relative gain)
7. Out-of-order, duplicate, and late frames
8. Clock offset, clock drift, clock jump, and multi-sensor sync skew
9. Sensor offline timeout & explicit source-down/unplug handling
10. Temperature & non-vibrational plausibility (range, rate-of-change, NaN/loss)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import uuid

import numpy as np
from scipy import signal

from app.ingestion.base import EventType, Frame, LifecycleEvent
from app.quality.models import (
    QualityEvent,
    QualityReport,
    QualityStatus,
    SensorBaselineStats,
    SensorQualityState,
)
from app.schemas.structure_config import (
    QualityThresholdsConfig,
    SensorConfig,
    SensorType,
    StructureConfig,
)


def _robust_mad(x: np.ndarray) -> float:
    """Scaled median absolute deviation consistent with Gaussian std."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


def _excess_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis safely for constant or tiny arrays."""
    if x.size < 4:
        return 0.0
    centered = x - float(np.mean(x))
    var = float(np.mean(centered * centered))
    if var <= 1e-24:
        return 0.0
    m4 = float(np.mean((centered * centered) ** 2))
    return (m4 / (var * var)) - 3.0


def _high_freq_noise_power(x: np.ndarray, fs: float, cutoff_hz: float = 15.0) -> float:
    """Estimate mean spectral power density above cutoff_hz."""
    n = x.size
    if n < 16 or fs <= 0:
        return 0.0
    nyquist = fs * 0.5
    if nyquist <= cutoff_hz * 1.05:
        cutoff_hz = nyquist * 0.6
    nperseg = min(n, 256)
    freqs, psd = signal.welch(x - float(np.mean(x)), fs=fs, nperseg=nperseg)
    mask = freqs >= cutoff_hz
    if not np.any(mask):
        return float(np.median(psd)) if psd.size else 0.0
    return float(np.mean(psd[mask]))


@dataclass
class _ReorderEntry:
    seq: int
    t0: int
    frame: Frame

    def __lt__(self, other: "_ReorderEntry") -> bool:
        return (self.t0, self.seq) < (other.t0, other.seq)


@dataclass
class _SensorTracker:
    last_seen_wall: Optional[float] = None
    last_seen_data_ns: Optional[int] = None
    last_temp_val: Optional[float] = None
    last_temp_time_sec: Optional[float] = None
    dc_ema: Optional[float] = None
    active_fault: Optional[str] = None
    active_status: QualityStatus = QualityStatus.OK


class DataQualityLayer:
    """Stateful per-structure data-quality evaluator and baseline calibrator."""

    def __init__(self, config: StructureConfig) -> None:
        self.config = config
        self.structure_id = config.structure_id
        self.thresholds: QualityThresholdsConfig = config.quality
        self.sensors_by_id: Dict[str, SensorConfig] = {s.id: s for s in config.sensors}
        self.sensor_order: List[str] = [s.id for s in config.sensors]
        self.baselines: Dict[str, SensorBaselineStats] = {}
        self.calibration_metadata: Dict[str, Any] = {}

        # Per-source sequencing, timing & reorder state
        self._seen_keys: Deque[Tuple[str, int, int]] = deque(maxlen=4096)
        self._seen_key_set: set[Tuple[str, int, int]] = set()
        self._last_seq_by_source: Dict[str, int] = {}
        self._max_seq_seen_by_source: Dict[str, int] = {}
        self._expected_next_t0_by_source: Dict[str, int] = {}
        self._reorder_heap: List[_ReorderEntry] = []

        # Clock drift/jump tracking (source_id -> (first_t0_ns, first_wall_ns, prev_t0_ns, prev_wall_ns, prev_n, prev_fs))
        self._clock_ref: Dict[str, Tuple[int, int, int, int, int, float]] = {}

        # Per-sensor live state & event log
        self._trackers: Dict[str, _SensorTracker] = {
            sid: _SensorTracker() for sid in self.sensor_order
        }
        self.events: Deque[QualityEvent] = deque(maxlen=1000)
        self.counters: Dict[str, int] = {
            "frames_in": 0,
            "frames_out": 0,
            "duplicates_dropped": 0,
            "out_of_order_frames": 0,
            "late_frames_dropped": 0,
            "missing_seq_gaps": 0,
            "timestamp_gaps": 0,
            "clock_jumps": 0,
            "clock_drift_warnings": 0,
            "quality_events_emitted": 0,
        }
        self._gap_lengths_sec: Deque[float] = deque(maxlen=500)
        self.last_report: Optional[QualityReport] = None
        self.source_online: bool = True

    # ------------------------------------------------------------------
    # Calibration (train / val healthy data only, never tuned on test)
    # ------------------------------------------------------------------

    def calibrate(
        self,
        recordings: List[np.ndarray],
        sensor_ids: Optional[List[str]] = None,
        fs: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, SensorBaselineStats]:
        """Compute per-sensor healthy baseline stats from training/validation arrays."""
        if not recordings:
            return self.baselines
        ids = sensor_ids or self.sensor_order
        eff_fs = float(fs or self.config.effective_detector_fs)

        per_sensor_dc: Dict[str, List[float]] = {sid: [] for sid in ids}
        per_sensor_rms: Dict[str, List[float]] = {sid: [] for sid in ids}
        per_sensor_hf: Dict[str, List[float]] = {sid: [] for sid in ids}
        per_sensor_samples: Dict[str, List[np.ndarray]] = {sid: [] for sid in ids}
        sample_counts: Dict[str, int] = {sid: 0 for sid in ids}

        block_len = max(int(round(eff_fs * 1.0)), 32)
        for rec in recordings:
            arr = np.asarray(rec, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            n_rows, n_cols = arr.shape
            for col_idx, sid in enumerate(ids[:n_cols]):
                col = arr[:, col_idx]
                finite = col[np.isfinite(col)]
                if finite.size == 0:
                    continue
                sample_counts[sid] += int(finite.size)
                per_sensor_samples[sid].append(finite[: min(finite.size, 5000)])
                for start in range(0, finite.size, block_len):
                    blk = finite[start : start + block_len]
                    if blk.size < 16:
                        continue
                    mu = float(np.mean(blk))
                    per_sensor_dc[sid].append(mu)
                    per_sensor_rms[sid].append(float(np.sqrt(np.mean((blk - mu) ** 2))))
                    per_sensor_hf[sid].append(_high_freq_noise_power(blk, eff_fs))

        calibrated: Dict[str, SensorBaselineStats] = {}
        for sid in ids:
            sensor_cfg = self.sensors_by_id.get(sid)
            stype = (
                str(sensor_cfg.type.value if hasattr(sensor_cfg.type, "value") else sensor_cfg.type)
                if sensor_cfg
                else "accel"
            )
            dcs = np.asarray(per_sensor_dc[sid], dtype=np.float64)
            rmss = np.asarray(per_sensor_rms[sid], dtype=np.float64)
            hfs = np.asarray(per_sensor_hf[sid], dtype=np.float64)
            cat = (
                np.concatenate(per_sensor_samples[sid])
                if per_sensor_samples[sid]
                else np.zeros(1, dtype=np.float64)
            )
            stats = SensorBaselineStats(
                sensor_id=sid,
                sensor_type=stype,
                dc_mean=float(np.median(dcs)) if dcs.size else 0.0,
                dc_std=max(float(np.std(dcs)) if dcs.size else 1e-4, 1e-6),
                median=float(np.median(cat)),
                mad=max(_robust_mad(cat), 1e-6),
                rms_median=max(float(np.median(rmss)) if rmss.size else 1e-3, 1e-6),
                rms_p99=max(float(np.percentile(rmss, 99)) if rmss.size else 1e-2, 1e-5),
                rms_max=max(float(np.max(rmss)) if rmss.size else 5e-2, 1e-5),
                high_freq_noise_floor=max(float(np.median(hfs)) if hfs.size else 1e-7, 1e-10),
                calibrated_samples=sample_counts[sid],
            )
            calibrated[sid] = stats

        self.baselines.update(calibrated)
        self.calibration_metadata = {
            "structure_id": self.structure_id,
            "calibrated_at": time.time(),
            "fs": eff_fs,
            "n_recordings": len(recordings),
            **(metadata or {}),
        }
        return calibrated

    def save_calibration(self, path: str | Path) -> None:
        """Persist baseline statistics and metadata to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.calibration_metadata,
            "baselines": {sid: st.model_dump() for sid, st in self.baselines.items()},
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_calibration(self, path: str | Path) -> None:
        """Load baseline statistics and metadata from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.calibration_metadata = payload.get("metadata", {})
        self.baselines = {
            sid: SensorBaselineStats(**vals)
            for sid, vals in payload.get("baselines", {}).items()
        }

    # ------------------------------------------------------------------
    # Lifecycle & boundary handling
    # ------------------------------------------------------------------

    def handle_lifecycle_event(self, event: LifecycleEvent) -> None:
        """Adjust internal sequence/gap expectations on recording or source events."""
        if event.event_type in (EventType.RECORDING_START, EventType.RECORDING_END):
            self._expected_next_t0_by_source.clear()
            self._clock_ref.clear()
            self._reorder_heap.clear()
        elif event.event_type == EventType.SOURCE_DOWN:
            self.source_online = False
            for sid in self.sensor_order:
                self._transition_sensor_state(
                    sid,
                    QualityStatus.OFFLINE,
                    "source_down",
                    event.reason or "Ingestion source went offline",
                    0.0,
                )
        elif event.event_type == EventType.SOURCE_UP:
            self.source_online = True

    def reset_state(self) -> None:
        """Reset streaming state while preserving calibrated baselines."""
        self._seen_keys.clear()
        self._seen_key_set.clear()
        self._last_seq_by_source.clear()
        self._max_seq_seen_by_source.clear()
        self._expected_next_t0_by_source.clear()
        self._reorder_heap.clear()
        self._clock_ref.clear()
        self._trackers = {sid: _SensorTracker() for sid in self.sensor_order}
        self._gap_lengths_sec.clear()

    # ------------------------------------------------------------------
    # Reordering, Duplicate, Late, and Sequencing Checks
    # ------------------------------------------------------------------

    def ingest_and_order(self, frame: Frame) -> List[Tuple[Frame, Dict[str, float]]]:
        """Accept a Frame into duplicate/late/reorder handling and return ordered ready frames."""
        self.counters["frames_in"] += 1
        key = (frame.source_id, int(frame.seq), int(frame.t0))
        if key in self._seen_key_set:
            self.counters["duplicates_dropped"] += 1
            return []

        if len(self._seen_keys) == self._seen_keys.maxlen and self._seen_keys:
            evicted = self._seen_keys.popleft()
            self._seen_key_set.discard(evicted)
        self._seen_keys.append(key)
        self._seen_key_set.add(key)

        max_seq_seen = self._max_seq_seen_by_source.get(frame.source_id)
        is_ooo = False
        if max_seq_seen is not None and frame.seq < max_seq_seen:
            is_ooo = True
            self.counters["out_of_order_frames"] += 1
            # Reject if strictly older than late_arrival_tolerance_sec vs expected timeline
            expected_t0 = self._expected_next_t0_by_source.get(frame.source_id)
            if expected_t0 is not None:
                lateness_sec = (expected_t0 - frame.t0) / 1e9
                if lateness_sec > self.thresholds.late_arrival_tolerance_sec:
                    self.counters["late_frames_dropped"] += 1
                    return []
        if max_seq_seen is None or frame.seq > max_seq_seen:
            self._max_seq_seen_by_source[frame.source_id] = frame.seq

        # Push into bounded reorder heap; flush immediately if reorder_window <= 1 or in-order
        win = max(1, int(self.thresholds.reorder_window_frames))
        heapq.heappush(self._reorder_heap, _ReorderEntry(frame.seq, frame.t0, frame))

        ready: List[Tuple[Frame, Dict[str, float]]] = []
        # If incoming frame is strictly monotonic and heap has >1 or win == 1, drain ready prefix
        if win <= 1:
            while self._reorder_heap:
                item = heapq.heappop(self._reorder_heap)
                ready.append((item.frame, self._update_timing_stats(item.frame, is_ooo=is_ooo)))
        else:
            while len(self._reorder_heap) > win:
                item = heapq.heappop(self._reorder_heap)
                ready.append((item.frame, self._update_timing_stats(item.frame, is_ooo=is_ooo)))
            # Flush monotonically ONLY when next expected seq sits at heap head (hold forward gaps briefly)
            while self._reorder_heap:
                head = self._reorder_heap[0]
                cur_last = self._last_seq_by_source.get(head.frame.source_id)
                if cur_last is None or head.seq <= cur_last + 1:
                    item = heapq.heappop(self._reorder_heap)
                    ready.append((item.frame, self._update_timing_stats(item.frame, is_ooo=is_ooo)))
                else:
                    break
        return ready

    def flush_reorder_buffer(self) -> List[Tuple[Frame, Dict[str, float]]]:
        """Flush all remaining buffered frames in timestamp/seq order."""
        out: List[Tuple[Frame, Dict[str, float]]] = []
        while self._reorder_heap:
            item = heapq.heappop(self._reorder_heap)
            out.append((item.frame, self._update_timing_stats(item.frame, is_ooo=False)))
        return out

    def _update_timing_stats(self, frame: Frame, is_ooo: bool) -> Dict[str, float]:
        """Compute per-frame sequence gap, timestamp gap, clock offset, drift ppm, and jump."""
        src = frame.source_id
        n_samples = int(frame.samples.shape[0])
        fs = max(float(frame.fs), 1e-6)
        dur_ns = int(round((n_samples / fs) * 1e9))
        wall_ns = int(frame.flags.get("received_wall_ns") or time.time_ns())
        is_historical = bool(frame.flags.get("historical_replay", False)) or (
            "recording_id" in frame.flags and not frame.flags.get("enforce_wall_clock", False)
        )

        # Seq gap
        missing_seqs = 0
        last_seq = self._last_seq_by_source.get(src)
        if last_seq is not None and frame.seq > last_seq + 1:
            missing_seqs = frame.seq - last_seq - 1
            self.counters["missing_seq_gaps"] += missing_seqs
        if last_seq is None or frame.seq > last_seq:
            self._last_seq_by_source[src] = frame.seq

        # Timestamp gap
        ts_gap_sec = 0.0
        exp_t0 = self._expected_next_t0_by_source.get(src)
        if exp_t0 is not None:
            diff_sec = (frame.t0 - exp_t0) / 1e9
            if diff_sec > self.thresholds.max_gap_interpolate_sec:
                ts_gap_sec = diff_sec
                self.counters["timestamp_gaps"] += 1
                self._gap_lengths_sec.append(diff_sec)
        self._expected_next_t0_by_source[src] = frame.t0 + dur_ns

        # Clock offset, jump, drift (comparing sample timestamps against reference/wall clock)
        clock_offset_sec = 0.0
        clock_jump_sec = 0.0
        clock_drift_ppm = 0.0

        if not is_historical:
            clock_offset_sec = abs(wall_ns - (frame.t0 + dur_ns)) / 1e9

        ref = self._clock_ref.get(src)
        if ref is None:
            self._clock_ref[src] = (frame.t0, wall_ns, frame.t0, wall_ns, n_samples, fs)
        else:
            t0_first, wall_first, prev_t0, prev_wall, prev_n, prev_fs = ref
            dt_t0_sec = (frame.t0 - prev_t0) / 1e9
            expected_step_sec = prev_n / max(prev_fs, 1e-6)
            step_error_sec = dt_t0_sec - expected_step_sec

            if is_historical:
                # For historical streams, unexpected negative jump or explicit discontinuity without boundary
                if dt_t0_sec < -1e-6:
                    clock_jump_sec = abs(dt_t0_sec)
            else:
                dt_wall_sec = max((wall_ns - prev_wall) / 1e9, 1e-6)
                # Explicit clock jump check: large discrepancy between dt_t0 and elapsed step/wall
                if missing_seqs == 0 and abs(step_error_sec) > self.thresholds.clock_jump_max_sec:
                    clock_jump_sec = abs(step_error_sec)
                elif abs((dt_t0_sec - dt_wall_sec)) > self.thresholds.clock_jump_max_sec and missing_seqs == 0:
                    clock_jump_sec = abs(dt_t0_sec - dt_wall_sec)

                # Cumulative drift ppm over >= 2 s elapsed wall time
                total_wall_sec = (wall_ns - wall_first) / 1e9
                total_t0_sec = (frame.t0 - t0_first) / 1e9
                if total_wall_sec >= 2.0 and clock_jump_sec <= self.thresholds.clock_jump_max_sec:
                    clock_drift_ppm = abs(total_t0_sec - total_wall_sec) / total_wall_sec * 1e6

            if clock_jump_sec > self.thresholds.clock_jump_max_sec:
                self.counters["clock_jumps"] += 1
            if clock_drift_ppm > self.thresholds.clock_drift_max_ppm:
                self.counters["clock_drift_warnings"] += 1

            self._clock_ref[src] = (t0_first, wall_first, frame.t0, wall_ns, n_samples, fs)

        # Multi-sensor sync skew (optional channel_t0_ns metadata in flags)
        sync_skew_sec = 0.0
        ch_t0 = frame.flags.get("channel_t0_ns")
        if isinstance(ch_t0, dict) and len(ch_t0) >= 2:
            vals = [int(v) for v in ch_t0.values()]
            sync_skew_sec = (max(vals) - min(vals)) / 1e9
        elif isinstance(ch_t0, (list, tuple)) and len(ch_t0) >= 2:
            vals = [int(v) for v in ch_t0]
            sync_skew_sec = (max(vals) - min(vals)) / 1e9

        return {
            "missing_seqs": float(missing_seqs),
            "ts_gap_sec": float(ts_gap_sec),
            "clock_offset_sec": float(clock_offset_sec),
            "clock_jump_sec": float(clock_jump_sec),
            "clock_drift_ppm": float(clock_drift_ppm),
            "sync_skew_sec": float(sync_skew_sec),
            "is_ooo": 1.0 if is_ooo else 0.0,
        }

    # ------------------------------------------------------------------
    # Main Per-Frame Evaluation
    # ------------------------------------------------------------------

    def evaluate_frame(
        self, frame: Frame, wall_now: Optional[float] = None
    ) -> Tuple[List[Tuple[Frame, QualityReport]], Optional[QualityReport]]:
        """Run ingest ordering + 10-check quality evaluation.

        Returns (list of (ordered_frame, report), latest_report).
        """
        now_sec = wall_now if wall_now is not None else time.time()
        ready = self.ingest_and_order(frame)
        results: List[Tuple[Frame, QualityReport]] = []
        latest: Optional[QualityReport] = None
        for ord_frame, timing in ready:
            report = self._score_frame(ord_frame, timing, now_sec)
            self.counters["frames_out"] += 1
            self.last_report = report
            latest = report
            results.append((ord_frame, report))
        return results, latest

    def evaluate_direct(
        self, frame: Frame, wall_now: Optional[float] = None
    ) -> QualityReport:
        """Evaluate a frame directly without waiting in reorder buffer (useful for unit tests)."""
        now_sec = wall_now if wall_now is not None else time.time()
        timing = self._update_timing_stats(frame, is_ooo=False)
        report = self._score_frame(frame, timing, now_sec)
        self.counters["frames_out"] += 1
        self.last_report = report
        return report

    def evaluate_offline(self, now_sec: Optional[float] = None) -> QualityReport:
        """Check all configured sensors for offline timeout even when no frames arrive."""
        now = now_sec if now_sec is not None else time.time()
        sensors_out: Dict[str, SensorQualityState] = {}
        usable: List[str] = []
        faulted: Dict[str, str] = {}
        offline: List[str] = []
        temp_ok = True
        temp_reason: Optional[str] = None

        for sid in self.sensor_order:
            tr = self._trackers[sid]
            scfg = self.sensors_by_id[sid]
            timed_out = (
                tr.last_seen_wall is not None
                and (now - tr.last_seen_wall) > self.thresholds.offline_timeout_sec
            )
            if timed_out:
                self._transition_sensor_state(
                    sid,
                    QualityStatus.OFFLINE,
                    "offline",
                    f"No data for {now - tr.last_seen_wall:.1f}s (>{self.thresholds.offline_timeout_sec}s)",
                    0.0,
                )
                offline.append(sid)
                if scfg.type == SensorType.TEMP or sid == self.config.temperature_sensor_id:
                    temp_ok = False
                    temp_reason = "temperature_sensor_offline"
                sensors_out[sid] = SensorQualityState(
                    sensor_id=sid,
                    status=QualityStatus.OFFLINE,
                    quality_score=0.0,
                    completeness=0.0,
                    timing_score=0.0,
                    signal_health=0.0,
                    reasons=[f"Sensor offline for {now - tr.last_seen_wall:.1f}s"],
                    fault_type="offline",
                    last_seen_ns=tr.last_seen_data_ns,
                )
            elif tr.active_status == QualityStatus.FAULT:
                faulted[sid] = tr.active_fault or "fault"
                if scfg.type == SensorType.TEMP or sid == self.config.temperature_sensor_id:
                    temp_ok = False
                    temp_reason = f"temperature_fault:{tr.active_fault}"
                sensors_out[sid] = SensorQualityState(
                    sensor_id=sid,
                    status=QualityStatus.FAULT,
                    quality_score=0.0,
                    reasons=[tr.active_fault or "fault"],
                    fault_type=tr.active_fault,
                    last_seen_ns=tr.last_seen_data_ns,
                )
            else:
                if scfg.type != SensorType.TEMP:
                    usable.append(sid)
                sensors_out[sid] = SensorQualityState(
                    sensor_id=sid,
                    status=tr.active_status,
                    quality_score=1.0 if tr.active_status == QualityStatus.OK else 0.7,
                    last_seen_ns=tr.last_seen_data_ns,
                )

        overall = QualityStatus.OK
        if offline or faulted:
            overall = QualityStatus.DEGRADED if usable else QualityStatus.OFFLINE
        scores = [s.quality_score for s in sensors_out.values()]
        overall_q = float(np.mean(scores)) if scores else 0.0
        report = QualityReport(
            structure_id=self.structure_id,
            timestamp=now,
            overall_status=overall,
            overall_quality=overall_q,
            usable_channels=usable,
            faulted_channels=faulted,
            offline_channels=offline,
            temperature_ok=temp_ok,
            temperature_reason=temp_reason,
            sensors=sensors_out,
            counters=dict(self.counters),
            gap_stats=self._gap_stats_dict(),
        )
        self.last_report = report
        return report

    def _score_frame(
        self, frame: Frame, timing: Dict[str, float], now_sec: float
    ) -> QualityReport:
        samples = np.asarray(frame.samples, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples[:, None]
        n_rows, n_cols = samples.shape
        fs = max(float(frame.fs), 1e-6)
        explicit_offline = set(frame.flags.get("offline_sensors", []))

        # Map sensor_id (and configured sensor display name alias) -> column index in incoming frame
        name_to_id = {s.name: s.id for s in self.config.sensors}
        col_map: Dict[str, int] = {}
        for idx, raw_sid in enumerate(frame.sensor_ids[:n_cols]):
            canonical = name_to_id.get(raw_sid, raw_sid)
            col_map[canonical] = idx

        # Precompute per-channel stats for cross-sensor sibling gain consistency
        rel_rms_by_accel: Dict[str, float] = {}
        hf_ratio_by_accel: Dict[str, float] = {}
        raw_stats: Dict[str, Dict[str, Any]] = {}

        for sid in self.sensor_order:
            scfg = self.sensors_by_id[sid]
            col_idx = col_map.get(sid)
            if sid in explicit_offline or col_idx is None:
                continue
            col = samples[:, col_idx]
            finite_mask = np.isfinite(col)
            n_finite = int(np.sum(finite_mask))
            if n_finite == 0:
                continue
            finite = col[finite_mask]
            mu = float(np.mean(finite))
            std = float(np.std(finite))
            rms = float(np.sqrt(np.mean((finite - mu) ** 2)))
            hf = _high_freq_noise_power(finite, fs) if finite.size >= 16 else 0.0
            raw_stats[sid] = {
                "col": col,
                "finite_mask": finite_mask,
                "finite": finite,
                "mean": mu,
                "std": std,
                "rms": rms,
                "hf": hf,
            }
            if scfg.type == SensorType.ACCEL:
                base = self.baselines.get(sid)
                base_rms = base.rms_median if base else max(rms, 1e-4)
                rel_rms_by_accel[sid] = rms / max(base_rms, 1e-9)
                base_hf = base.high_freq_noise_floor if base else max(hf, 1e-9)
                hf_ratio_by_accel[sid] = hf / max(base_hf, 1e-12)

        sibling_med_rel_rms = (
            float(np.median(list(rel_rms_by_accel.values())))
            if len(rel_rms_by_accel) >= 2
            else 1.0
        )

        # Timing score (shared baseline across frame)
        timing_penalties = 0.0
        timing_reasons: List[str] = []
        if timing["clock_offset_sec"] > self.thresholds.clock_offset_max_sec:
            timing_penalties += 0.35
            timing_reasons.append(f"clock_offset={timing['clock_offset_sec']:.2f}s")
        if timing["clock_jump_sec"] > self.thresholds.clock_jump_max_sec:
            timing_penalties += 0.45
            timing_reasons.append(f"clock_jump={timing['clock_jump_sec']:.2f}s")
        if timing["clock_drift_ppm"] > self.thresholds.clock_drift_max_ppm:
            timing_penalties += 0.30
            timing_reasons.append(f"clock_drift={timing['clock_drift_ppm']:.0f}ppm")
        if timing["sync_skew_sec"] > self.thresholds.clock_sync_skew_max_sec:
            timing_penalties += 0.30
            timing_reasons.append(f"sync_skew={timing['sync_skew_sec']:.3f}s")
        if timing["is_ooo"] > 0.5:
            timing_penalties += 0.15
            timing_reasons.append("out_of_order")
        frame_timing_score = max(0.0, 1.0 - timing_penalties)

        sensors_out: Dict[str, SensorQualityState] = {}
        usable: List[str] = []
        faulted: Dict[str, str] = {}
        offline: List[str] = []
        temp_ok = True
        temp_reason: Optional[str] = None

        for sid in self.sensor_order:
            scfg = self.sensors_by_id[sid]
            tr = self._trackers[sid]
            is_temp = scfg.type == SensorType.TEMP or sid == self.config.temperature_sensor_id

            # Explicit offline or missing from frame
            if sid in explicit_offline or sid not in col_map:
                timed_out = sid in explicit_offline or (
                    tr.last_seen_wall is not None
                    and (now_sec - tr.last_seen_wall) > self.thresholds.offline_timeout_sec
                )
                status = QualityStatus.OFFLINE if timed_out else QualityStatus.DEGRADED
                reason = "Channel missing / unplugged" if status == QualityStatus.OFFLINE else "Awaiting channel"
                if status == QualityStatus.OFFLINE:
                    offline.append(sid)
                    if is_temp:
                        temp_ok = False
                        temp_reason = "temperature_offline"
                    self._transition_sensor_state(sid, QualityStatus.OFFLINE, "offline", reason, 0.0)
                sensors_out[sid] = SensorQualityState(
                    sensor_id=sid,
                    status=status,
                    quality_score=0.0 if status == QualityStatus.OFFLINE else 0.5,
                    completeness=0.0,
                    timing_score=frame_timing_score,
                    signal_health=0.0,
                    reasons=[reason],
                    fault_type="offline" if status == QualityStatus.OFFLINE else None,
                    last_seen_ns=tr.last_seen_data_ns,
                )
                continue

            st = raw_stats.get(sid)
            if st is None:
                # All NaNs/Infs in frame -> unplugged/loss
                tr.last_seen_wall = now_sec
                tr.last_seen_data_ns = int(frame.t0)
                offline.append(sid)
                if is_temp:
                    temp_ok = False
                    temp_reason = "temperature_all_nan"
                self._transition_sensor_state(
                    sid, QualityStatus.OFFLINE, "packet_loss", "All samples non-finite (NaN/Inf)", 0.0
                )
                sensors_out[sid] = SensorQualityState(
                    sensor_id=sid,
                    status=QualityStatus.OFFLINE,
                    quality_score=0.0,
                    completeness=0.0,
                    timing_score=frame_timing_score,
                    signal_health=0.0,
                    reasons=["All samples non-finite"],
                    fault_type="packet_loss",
                    last_seen_ns=int(frame.t0),
                )
                continue

            tr.last_seen_wall = now_sec
            tr.last_seen_data_ns = int(frame.t0)
            col = st["col"]
            finite = st["finite"]
            finite_frac = float(finite.size / max(n_rows, 1))

            # Completeness check
            completeness = finite_frac
            if timing["missing_seqs"] > 0:
                completeness = min(completeness, max(0.0, 1.0 - 0.1 * timing["missing_seqs"]))
            if timing["ts_gap_sec"] > self.thresholds.max_gap_interpolate_sec:
                completeness = min(completeness, 0.5)

            reasons: List[str] = list(timing_reasons)
            fault_type: Optional[str] = None
            signal_health = 1.0
            metrics: Dict[str, float] = {
                "mean": st["mean"],
                "std": st["std"],
                "rms": st["rms"],
                "hf_power": st["hf"],
                "finite_frac": finite_frac,
            }

            if is_temp:
                # Check 10: Temperature plausibility, ROC, packet loss
                if finite_frac < 0.5:
                    fault_type = "temp_packet_loss"
                    reasons.append(f"Temp completeness {finite_frac:.0%} < 50%")
                    signal_health = 0.0
                else:
                    tmin, tmax = self.thresholds.temp_range_c
                    t_mean = st["mean"]
                    metrics["temp_c"] = t_mean
                    if t_mean < tmin or t_mean > tmax:
                        fault_type = "temp_out_of_range"
                        reasons.append(f"Temp {t_mean:.2f}C outside [{tmin}, {tmax}]C")
                        signal_health = 0.0
                    else:
                        data_t_sec = frame.t0 / 1e9
                        if tr.last_temp_val is not None and tr.last_temp_time_sec is not None:
                            dt_min = (data_t_sec - tr.last_temp_time_sec) / 60.0
                            if dt_min > 1e-4:
                                roc = abs(t_mean - tr.last_temp_val) / dt_min
                                metrics["roc_c_per_min"] = roc
                                if roc > self.thresholds.temp_max_roc_c_per_min:
                                    fault_type = "temp_roc_exceeded"
                                    reasons.append(
                                        f"Temp ROC {roc:.1f}C/min > {self.thresholds.temp_max_roc_c_per_min}"
                                    )
                                    signal_health = 0.0
                        tr.last_temp_val = t_mean
                        tr.last_temp_time_sec = data_t_sec
            else:
                # Non-temperature sensor (accel / strain) checks 1-6
                n_f = finite.size
                base = self.baselines.get(sid)
                range_min = (
                    scfg.range_min
                    if scfg.range_min is not None
                    else self.thresholds.default_accel_range[0]
                )
                range_max = (
                    scfg.range_max
                    if scfg.range_max is not None
                    else self.thresholds.default_accel_range[1]
                )

                # Dropout / exact zero fraction
                zero_frac = float(np.sum(np.abs(finite) <= 1e-12) / max(n_f, 1))
                metrics["zero_frac"] = zero_frac
                if finite_frac < 0.5 or zero_frac > 0.5:
                    fault_type = "dropout"
                    reasons.append(f"Dropout finite={finite_frac:.0%}, zero={zero_frac:.0%}")
                    signal_health = 0.0
                elif zero_frac > self.thresholds.dropout_zero_fraction:
                    reasons.append(f"Partial zero dropout {zero_frac:.1%}")
                    completeness = min(completeness, max(0.0, 1.0 - zero_frac))
                    signal_health = min(signal_health, 0.5)

                # Check 2: Flatline / stuck-at-nonzero
                uniq_ratio = float(np.unique(np.round(finite, 8)).size / max(n_f, 1))
                metrics["unique_ratio"] = uniq_ratio
                if fault_type is None and st["std"] <= self.thresholds.flatline_min_std:
                    fault_type = "stuck_nonzero" if abs(st["mean"]) > 1e-4 else "flatline"
                    reasons.append(
                        f"{fault_type}: std={st['std']:.2e} <= {self.thresholds.flatline_min_std:.1e}"
                    )
                    signal_health = 0.0
                elif (
                    fault_type is None
                    and n_f >= 16
                    and uniq_ratio <= self.thresholds.stuck_min_unique_ratio
                    and abs(st["mean"]) > 1e-3
                ):
                    fault_type = "stuck_nonzero"
                    reasons.append(f"Stuck quantized/nonzero: unique_ratio={uniq_ratio:.3f}")
                    signal_health = 0.0

                # Check 3: Saturation / clipping (range rails OR flat-top clipping)
                if fault_type is None and n_f >= 10:
                    sat_frac = float(
                        np.mean((finite >= range_max - 1e-6) | (finite <= range_min + 1e-6))
                    )
                    clip_hi = float(np.mean(finite >= float(np.max(finite)) - 1e-9))
                    clip_lo = float(np.mean(finite <= float(np.min(finite)) + 1e-9))
                    max_clip_frac = max(sat_frac, clip_hi, clip_lo)
                    metrics["clip_frac"] = max_clip_frac
                    if max_clip_frac > self.thresholds.saturation_max_fraction:
                        fault_type = "saturation"
                        reasons.append(
                            f"Clipping/saturation fraction {max_clip_frac:.1%} > {self.thresholds.saturation_max_fraction:.1%}"
                        )
                        signal_health = 0.0

                # Check 4: Spikes (robust z-score + kurtosis + local impulse check)
                if fault_type is None and n_f >= 16:
                    scale_mad = max(_robust_mad(finite), base.mad if base else 1e-5, 1e-6)
                    med = float(np.median(finite))
                    max_rz = float(np.max(np.abs(finite - med)) / scale_mad)
                    kurt = _excess_kurtosis(finite)
                    # Local impulse: compare points against neighbor median to isolate electrical spikes
                    padded = np.pad(finite, 1, mode="edge")
                    nbr_med = 0.5 * (padded[:-2] + padded[2:])
                    impulse_rz = float(np.max(np.abs(finite - nbr_med)) / scale_mad)
                    metrics["max_robust_z"] = max_rz
                    metrics["kurtosis"] = kurt
                    metrics["impulse_rz"] = impulse_rz
                    if (
                        max_rz > self.thresholds.spike_robust_z_threshold
                        and kurt > self.thresholds.spike_kurtosis_threshold
                        and impulse_rz > self.thresholds.spike_robust_z_threshold * 0.6
                    ):
                        fault_type = "spikes"
                        reasons.append(
                            f"Spikes: z={max_rz:.1f}, kurt={kurt:.1f}, impulse_z={impulse_rz:.1f}"
                        )
                        signal_health = 0.0

                # Check 5: Drift / DC offset against baseline
                if fault_type is None:
                    base_dc = base.dc_mean if base else 0.0
                    base_dc_std = max(base.dc_std if base else 1e-3, 1e-4)
                    # Update EMA slowly over healthy frames; compute offset against baseline
                    dc_dev = abs(st["mean"] - base_dc)
                    dc_thresh = max(
                        self.thresholds.drift_sigma_multiplier * base_dc_std,
                        3.0 * (base.rms_p99 if base else 1e-2) if n_f < 250 else 0.02,
                    )
                    if tr.dc_ema is None:
                        tr.dc_ema = st["mean"]
                    elif abs(st["mean"] - base_dc) <= dc_thresh:
                        tr.dc_ema = 0.9 * tr.dc_ema + 0.1 * st["mean"]
                    else:
                        tr.dc_ema = 0.5 * tr.dc_ema + 0.5 * st["mean"]
                    sustained_dev = abs(tr.dc_ema - base_dc)
                    metrics["dc_offset"] = dc_dev
                    metrics["dc_thresh"] = dc_thresh
                    if dc_dev > dc_thresh and (n_f >= 200 or sustained_dev > dc_thresh * 0.8):
                        fault_type = "drift"
                        reasons.append(
                            f"DC drift {dc_dev:.4f} (ema={sustained_dev:.4f}) > {dc_thresh:.4f}"
                        )
                        signal_health = 0.0

                # Check 6: Noise floor & gain change
                if fault_type is None and base is not None and n_f >= 16:
                    hf_ratio = hf_ratio_by_accel.get(sid, 1.0)
                    metrics["hf_noise_ratio"] = hf_ratio
                    if hf_ratio > self.thresholds.noise_floor_ratio:
                        fault_type = "noise_increase"
                        reasons.append(
                            f"High-freq noise floor {hf_ratio:.1f}x > {self.thresholds.noise_floor_ratio}x"
                        )
                        signal_health = 0.0
                    elif sid in rel_rms_by_accel:
                        rel_to_siblings = rel_rms_by_accel[sid] / max(sibling_med_rel_rms, 1e-6)
                        abs_vs_p99 = st["rms"] / max(base.rms_p99, 1e-6)
                        metrics["gain_ratio_vs_siblings"] = rel_to_siblings
                        metrics["rms_vs_p99"] = abs_vs_p99
                        if (
                            len(rel_rms_by_accel) >= 2
                            and rel_to_siblings > self.thresholds.gain_high_ratio
                        ):
                            fault_type = "gain_error"
                            reasons.append(
                                f"Gain anomaly {rel_to_siblings:.2f}x vs siblings (>{self.thresholds.gain_high_ratio}x)"
                            )
                            signal_health = 0.0
                        elif (
                            len(rel_rms_by_accel) >= 2
                            and rel_to_siblings < self.thresholds.gain_low_ratio
                        ):
                            fault_type = "gain_error"
                            reasons.append(
                                f"Low gain anomaly {rel_to_siblings:.2f}x vs siblings (<{self.thresholds.gain_low_ratio}x)"
                            )
                            signal_health = 0.0
                        elif len(rel_rms_by_accel) == 1 and abs_vs_p99 > self.thresholds.gain_high_ratio:
                            fault_type = "gain_error"
                            reasons.append(
                                f"Single-channel RMS {abs_vs_p99:.2f}x > {self.thresholds.gain_high_ratio}x p99"
                            )
                            signal_health = 0.0

            # Weighted quality score in [0, 1]
            wc = self.thresholds.weight_completeness
            wt = self.thresholds.weight_timing
            ws = self.thresholds.weight_signal_health
            wsum = max(wc + wt + ws, 1e-9)
            q_score = float(
                np.clip(
                    (wc * completeness + wt * frame_timing_score + ws * signal_health) / wsum,
                    0.0,
                    1.0,
                )
            )

            if fault_type is not None:
                status = QualityStatus.FAULT
                q_score = min(q_score, 0.25)
                faulted[sid] = fault_type
                if is_temp:
                    temp_ok = False
                    temp_reason = f"temperature_fault:{fault_type}"
                self._transition_sensor_state(
                    sid, QualityStatus.FAULT, fault_type, "; ".join(reasons), q_score
                )
            elif reasons or q_score < 0.85:
                status = QualityStatus.DEGRADED
                if not is_temp:
                    usable.append(sid)
                self._transition_sensor_state(sid, QualityStatus.DEGRADED, None, "; ".join(reasons), q_score)
            else:
                status = QualityStatus.OK
                if not is_temp:
                    usable.append(sid)
                self._transition_sensor_state(sid, QualityStatus.OK, None, "", q_score)

            sensors_out[sid] = SensorQualityState(
                sensor_id=sid,
                status=status,
                quality_score=q_score,
                completeness=completeness,
                timing_score=frame_timing_score,
                signal_health=signal_health,
                reasons=reasons,
                fault_type=fault_type,
                last_seen_ns=int(frame.t0),
                metrics=metrics,
            )

        if not faulted and not offline and not any(
            s.status == QualityStatus.DEGRADED or s.status == "DEGRADED" for s in sensors_out.values()
        ):
            overall = QualityStatus.OK
        elif usable or temp_ok:
            overall = QualityStatus.DEGRADED if (faulted or offline or timing_reasons) else QualityStatus.OK
        else:
            overall = QualityStatus.FAULT if faulted else QualityStatus.OFFLINE

        scores = [s.quality_score for s in sensors_out.values()]
        overall_q = float(np.mean(scores)) if scores else 0.0

        return QualityReport(
            structure_id=self.structure_id,
            timestamp=now_sec,
            overall_status=overall,
            overall_quality=overall_q,
            usable_channels=usable,
            faulted_channels=faulted,
            offline_channels=offline,
            temperature_ok=temp_ok,
            temperature_reason=temp_reason,
            sensors=sensors_out,
            counters=dict(self.counters),
            gap_stats=self._gap_stats_dict(),
            clock_stats={
                "offset_sec": timing["clock_offset_sec"],
                "jump_sec": timing["clock_jump_sec"],
                "drift_ppm": timing["clock_drift_ppm"],
                "sync_skew_sec": timing["sync_skew_sec"],
            },
        )

    def _transition_sensor_state(
        self,
        sensor_id: str,
        new_status: QualityStatus,
        fault_type: Optional[str],
        reason: str,
        quality_score: float,
    ) -> None:
        tr = self._trackers[sensor_id]
        prev_status = tr.active_status
        prev_fault = tr.active_fault

        if new_status in (QualityStatus.FAULT, QualityStatus.OFFLINE):
            if prev_status != new_status or prev_fault != fault_type:
                ev = QualityEvent(
                    event_id=f"qe-{uuid.uuid4().hex[:10]}",
                    structure_id=self.structure_id,
                    sensor_id=sensor_id,
                    status=new_status,
                    fault_type=fault_type or new_status.value.lower(),
                    reason=reason,
                    quality_score=quality_score,
                )
                self.events.append(ev)
                self.counters["quality_events_emitted"] += 1
        elif prev_status in (QualityStatus.FAULT, QualityStatus.OFFLINE):
            # Mark last open event for this sensor resolved
            now = time.time()
            for ev in reversed(self.events):
                if ev.sensor_id == sensor_id and not ev.resolved:
                    ev.resolved = True
                    ev.resolved_at = now
                    break

        tr.active_status = new_status
        tr.active_fault = fault_type

    def _gap_stats_dict(self) -> Dict[str, float]:
        if not self._gap_lengths_sec:
            return {"gap_count": 0.0, "max_gap_sec": 0.0, "total_gap_sec": 0.0}
        arr = np.asarray(self._gap_lengths_sec, dtype=np.float64)
        return {
            "gap_count": float(arr.size),
            "max_gap_sec": float(np.max(arr)),
            "total_gap_sec": float(np.sum(arr)),
        }

    def get_events(self, limit: int = 100, unresolved_only: bool = False) -> List[QualityEvent]:
        evs = list(self.events)
        if unresolved_only:
            evs = [e for e in evs if not e.resolved]
        return list(reversed(evs))[:limit]
