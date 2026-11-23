"""Structure Runtime and Multi-Structure Registry for Pluggable SHM Detectors.

Key Responsibilities:
1. StructureRegistry: Manages multiple civil structures keyed by structure_id.
2. StructureRuntime: Encapsulates shared buffer, shared data-quality sensor-fault gate,
   multi-cadence detector dispatch, and worst-state aggregation.
3. Artifact Validation: Verifies sensor list hash, band hash, and sampling rate.
   Strictly refuses to load mismatched artifacts.
4. Telemetry Formatting: Publishes dual-cadence, non-hardcoded payloads with
   per-detector cadence, last update, confidence, and calibration coverage.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from app.detectors.base import BaseDetector, DetectorOutput, DetectorState
from app.detectors.spectral_tracking import SpectralTrackingDetector
from app.detectors.waveform_ae import WaveformAEDetector
from app.ingestion.base import EventType, Frame, IngestionSource, IngestionStreamItem, LifecycleEvent
from app.ingestion.resampler import EdgeResampler
from app.ingestion.spool import StructureSpool
from app.quality.models import QualityReport, QualityStatus
from app.quality.quality_layer import DataQualityLayer
from app.schemas.structure_config import DetectorConfig, SensorType, StructureConfig

logger = logging.getLogger("SHM.Registry")

# Precedence order: SENSOR_FAULT > CRITICAL > WARNING > LOW_CONFIDENCE > NORMAL
STATE_PREFERENCE_ORDER = [
    DetectorState.SENSOR_FAULT,
    DetectorState.CRITICAL,
    DetectorState.WARNING,
    DetectorState.LOW_CONFIDENCE,
    DetectorState.NORMAL,
]


def combine_states(
    detector_outputs: Dict[str, DetectorOutput],
    hardware_fault_reason: Optional[str] = None,
) -> Tuple[DetectorState, Optional[str], Optional[str]]:
    """Worst-detector aggregation rule with contributing detector and reason.
    
    Returns:
        (combined_state, contributing_detector_id, reason)
    """
    if hardware_fault_reason:
        return DetectorState.SENSOR_FAULT, "hardware_gate", hardware_fault_reason

    if not detector_outputs:
        return DetectorState.NORMAL, None, "No active detectors"

    for candidate_state in STATE_PREFERENCE_ORDER:
        candidate_val = candidate_state.value if hasattr(candidate_state, "value") else str(candidate_state)
        for det_id, out in detector_outputs.items():
            out_val = out.state.value if hasattr(out.state, "value") else str(out.state)
            if out_val == candidate_val:
                return candidate_state, det_id, out.reason

    return DetectorState.NORMAL, None, "All detectors normal"


class StructureRuntime:
    """Manages runtime ingestion, multi-cadence detector execution, and telemetry for one structure."""

    def __init__(
        self,
        config: StructureConfig,
        artifacts_base_dir: Optional[Path] = None,
        broadcast_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self.config = config
        self.artifacts_base_dir = artifacts_base_dir or (Path(__file__).resolve().parent.parent.parent / "artifacts")
        self.broadcast_fn = broadcast_fn

        self.detectors: Dict[str, BaseDetector] = {}
        self.latest_outputs: Dict[str, DetectorOutput] = {}
        self.detector_last_run: Dict[str, float] = {}

        # Shared circular buffer
        # Buffer capacity = at least 5 minutes at configured detector fs
        eff_fs = config.effective_detector_fs
        self.buffer_capacity = max(int(eff_fs * 300.0), 1000)
        self.shared_buffer = np.zeros((0, config.num_channels), dtype=np.float32)

        # Stage 4 edge resampling, data-quality layer, and rotating crash-recovery spool
        temp_indices = [
            s.channel_index
            for s in config.sensors
            if s.type == SensorType.TEMP and s.channel_index is not None
        ]
        self.resampler = EdgeResampler(
            native_fs=config.effective_native_fs,
            target_fs=eff_fs,
            filter_type=config.resampling.filter_type,
            order=config.resampling.order,
            cutoff_hz=config.resampling.cutoff_hz,
            ripple_db=config.resampling.ripple_db,
            mode=config.resampling.mode,
            max_gap_interpolate_sec=config.resampling.max_gap_interpolate_sec,
            temp_channel_indices=temp_indices,
        )
        self.quality_layer = DataQualityLayer(config)
        self.last_quality_report: Optional[QualityReport] = None
        spool_root = Path(__file__).resolve().parent.parent.parent / "data" / "spool"
        self.spool = StructureSpool(structure_id=config.structure_id, spool_dir=spool_root)

        # Bounded backpressure queue & live ingestion telemetry
        self._queue: Optional[asyncio.Queue[IngestionStreamItem]] = None
        self._max_queue_size: int = max(8, int(config.ingestion.max_queue_size))
        self._worker_task: Optional[asyncio.Task[Any]] = None
        self._source_task: Optional[asyncio.Task[Any]] = None
        self._active_source: Optional[IngestionSource] = None
        self.source_status: str = "IDLE"  # IDLE | UP | DEGRADED | DOWN
        self.last_data_timestamp: Optional[float] = None
        self.dropped_frames_total: int = 0
        self.frames_processed: int = 0
        self.samples_processed: int = 0
        self._rate_window: Deque[Tuple[float, int, float]] = deque(maxlen=600)
        self._last_metric_log_wall: float = 0.0
        self.effective_frames_per_sec: float = 0.0
        self.effective_samples_per_sec: float = 0.0
        self.processing_lag_sec: float = 0.0

        # Scheduled-recording boundaries & confirmation delay tracking
        self.recordings_completed: int = 0
        self.hours_observed: float = 0.0
        self._current_recording_id: Optional[str] = None
        self._current_recording_samples: int = 0
        self._alert_onset_time: Optional[float] = None
        self._alert_onset_recording_idx: Optional[int] = None
        self._samples_since_det_run: Dict[str, int] = {}

        # Hardware triage gate status
        self.hardware_fault_reason: Optional[str] = None
        self.frame_counter: int = 0
        self.start_time: float = time.time()

        # Initialize detector instances from config
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Instantiate configured detectors."""
        channel_names = self.config.channel_names
        channel_axes = [
            (s.axis or ("temp" if s.type == SensorType.TEMP else "az")).lower()
            for s in self.config.sensors
        ]
        temp_sensor = self.config.get_temperature_sensor()
        temp_idx = (
            temp_sensor.channel_index
            if temp_sensor and temp_sensor.channel_index is not None
            else self.config.num_channels - 1
        )
        eff_fs = self.config.effective_detector_fs

        for d_cfg in self.config.detectors:
            cadence = d_cfg.update_cadence_seconds or 1.0
            min_ch = d_cfg.min_channels if d_cfg.min_channels is not None else 2
            req_ch = d_cfg.required_channels or []
            if d_cfg.type == "waveform_ae":
                det = WaveformAEDetector(
                    detector_id=d_cfg.detector_id,
                    version="v1",
                    update_cadence_seconds=cadence,
                    params={
                        "fs": eff_fs,
                        "input_size": self.config.num_channels,
                        "target_size": max(1, self.config.num_channels - (1 if temp_sensor else 0)),
                        "min_channels": min_ch,
                        "required_channels": req_ch,
                        **d_cfg.params,
                    },
                    channel_names=channel_names,
                )
            elif d_cfg.type == "spectral_tracking":
                det = SpectralTrackingDetector(
                    detector_id=d_cfg.detector_id,
                    version="v1",
                    update_cadence_seconds=cadence,
                    params={
                        "fs": eff_fs,
                        "operating_threshold_tau": 28.91,
                        "min_channels": min_ch,
                        "required_channels": req_ch,
                        **d_cfg.params,
                    },
                    modes=self.config.modes,
                    fs=eff_fs,
                    temp_channel_index=temp_idx,
                    channel_axes=channel_axes,
                    sensor_ids=channel_names,
                    min_channels=min_ch,
                    required_channels=req_ch,
                )
            else:
                logger.warning(f"Unknown detector type '{d_cfg.type}' in config for {self.config.structure_id}")
                continue

            self.detectors[d_cfg.detector_id] = det
            self.detector_last_run[d_cfg.detector_id] = 0.0
            self._samples_since_det_run[d_cfg.detector_id] = 0

    def load_detector_artifacts(self, version: str = "v1") -> None:
        """Loads and strictly validates artifacts for all detectors.
        
        Refuses to load artifacts if sensor definitions, bands, or fs mismatch.
        """
        sensor_hash = self.config.compute_sensor_hash()
        band_hash = self.config.compute_band_hash()

        for det_id, det in self.detectors.items():
            artifact_dir = self.artifacts_base_dir / self.config.structure_id / det_id / version
            if not artifact_dir.exists():
                logger.info(f"No existing artifact directory at {artifact_dir} (starting fresh)")
                continue

            meta_file = artifact_dir / "metadata.json"
            if meta_file.is_file():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # Strict Guardrail 1: Sampling rate match
                art_fs = meta.get("fs")
                if art_fs is not None and abs(art_fs - self.config.fs) > 1e-3:
                    raise ValueError(
                        f"Sampling rate mismatch for {self.config.structure_id}/{det_id}: "
                        f"Config fs={self.config.fs} Hz vs Artifact fs={art_fs} Hz"
                    )

                # Strict Guardrail 2: Sensor list hash match
                art_sensor_hash = meta.get("sensor_hash")
                if art_sensor_hash is not None and art_sensor_hash != sensor_hash:
                    raise ValueError(
                        f"Sensor list definition mismatch for {self.config.structure_id}/{det_id}: "
                        f"Current sensor hash does not match artifact metadata"
                    )

                # Strict Guardrail 3: Band hash match for spectral tracking
                art_band_hash = meta.get("band_hash")
                if art_band_hash is not None and band_hash and art_band_hash != band_hash:
                    raise ValueError(
                        f"Modal band definition mismatch for {self.config.structure_id}/{det_id}: "
                        f"Current band hash does not match artifact metadata"
                    )

            det.load(artifact_dir)
            logger.info(f"Successfully loaded and verified artifacts for {self.config.structure_id}/{det_id}")

    def save_detector_artifacts(self, version: str = "v1") -> None:
        """Saves models, scalers, and metadata with provenance hashes."""
        sensor_hash = self.config.compute_sensor_hash()
        band_hash = self.config.compute_band_hash()

        for det_id, det in self.detectors.items():
            artifact_dir = self.artifacts_base_dir / self.config.structure_id / det_id / version
            artifact_dir.mkdir(parents=True, exist_ok=True)
            det.save(artifact_dir)

            # Augment metadata with sensor & band hashes
            meta_file = artifact_dir / "metadata.json"
            if meta_file.is_file():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["sensor_hash"] = sensor_hash
                meta["band_hash"] = band_hash
                meta["fs"] = self.config.fs
                meta["structure_id"] = self.config.structure_id
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

    def check_hardware_fault(self, frame: np.ndarray) -> Optional[str]:
        """Hardware triage check across all incoming channels."""
        if np.isnan(frame).any() or np.isinf(frame).any():
            return "NaN or Inf values detected in input frame"

        # Check flatline across recent buffer (last 50 frames)
        if len(self.shared_buffer) >= 50:
            recent = self.shared_buffer[-50:]
            std_ch = np.std(recent, axis=0)
            for i, s in enumerate(std_ch):
                if s < 1e-6:
                    ch_name = self.config.channel_names[i]
                    return f"Channel flatline detected on {ch_name} (std={s:.2e} < 1e-6)"

        return None

    def push_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Ingest live frame, step detectors according to cadence, and return telemetry payload."""
        self.frame_counter += 1
        now = time.time()

        if frame.ndim == 1:
            frame_2d = frame.reshape(1, -1)
        else:
            frame_2d = frame

        # Append to shared circular buffer
        if len(self.shared_buffer) == 0:
            self.shared_buffer = frame_2d.copy()
        else:
            self.shared_buffer = np.vstack([self.shared_buffer, frame_2d])

        if len(self.shared_buffer) > self.buffer_capacity:
            self.shared_buffer = self.shared_buffer[-self.buffer_capacity :]

        # Hardware triage check
        self.hardware_fault_reason = self.check_hardware_fault(frame)

        # Run detectors according to their configured cadence
        for det_id, det in self.detectors.items():
            cadence = det.update_cadence_seconds
            elapsed = now - self.detector_last_run.get(det_id, 0.0)

            if elapsed >= cadence or self.frame_counter == 1:
                # Isolate detector crash inside try/except
                try:
                    out = det.update(frame_2d)
                    self.latest_outputs[det_id] = out
                    self.detector_last_run[det_id] = now
                except Exception as exc:
                    logger.error(f"Detector {det_id} crashed: {exc}", exc_info=True)
                    self.latest_outputs[det_id] = DetectorOutput(
                        detector_id=det_id,
                        version=det.version,
                        timestamp=now,
                        state=DetectorState.SENSOR_FAULT,
                        score=-1.0,
                        tau=det.tau,
                        update_cadence_seconds=cadence,
                        reason=f"DETECTOR_EXCEPTION ({exc.__class__.__name__}): {str(exc)}",
                        provenance={"error": str(exc)},
                    )
                    self.detector_last_run[det_id] = now

        # Format combined telemetry payload
        payload = self.format_telemetry(frame)

        # Broadcast if callback provided
        if self.broadcast_fn is not None:
            if asyncio.iscoroutinefunction(self.broadcast_fn):
                asyncio.create_task(self.broadcast_fn(payload))
            else:
                self.broadcast_fn(payload)

        return payload

    # ------------------------------------------------------------------
    # Stage 4 Unified Ingestion & Quality-Aware Pipeline
    # ------------------------------------------------------------------

    def _align_frame_channels(self, frame: Frame) -> Frame:
        """Reorder or pad incoming frame columns to match config sensor order (accepting id or name)."""
        target_ids = [s.id for s in self.config.sensors]
        target_names = self.config.channel_names
        if (frame.sensor_ids == target_ids or frame.sensor_ids == target_names) and frame.samples.shape[1] == len(target_ids):
            return Frame(
                structure_id=frame.structure_id,
                sensor_ids=target_ids,
                t0=frame.t0,
                fs=frame.fs,
                samples=frame.samples,
                seq=frame.seq,
                source_id=frame.source_id,
                flags=dict(frame.flags),
            )

        n_rows = frame.samples.shape[0]
        aligned = np.full((n_rows, len(target_ids)), np.nan, dtype=np.float32)
        src_map = {sid: idx for idx, sid in enumerate(frame.sensor_ids[: frame.samples.shape[1]])}
        for dst_idx, s_cfg in enumerate(self.config.sensors):
            if s_cfg.id in src_map:
                aligned[:, dst_idx] = frame.samples[:, src_map[s_cfg.id]]
            elif s_cfg.name in src_map:
                aligned[:, dst_idx] = frame.samples[:, src_map[s_cfg.name]]
        return Frame(
            structure_id=frame.structure_id,
            sensor_ids=target_ids,
            t0=frame.t0,
            fs=frame.fs,
            samples=aligned,
            seq=frame.seq,
            source_id=frame.source_id,
            flags=dict(frame.flags),
        )

    def _reset_for_recording_boundary(self, reason: str = "recording_boundary") -> None:
        """Clear resampler and detector buffers cleanly at scheduled recording boundaries."""
        if not self.config.ingestion.reset_on_recording_boundary:
            return
        self.resampler.reset()
        self.shared_buffer = np.zeros((0, self.config.num_channels), dtype=np.float32)
        for det in self.detectors.values():
            det.reset_state(clear_persistence=True)
        for det_id in self._samples_since_det_run:
            self._samples_since_det_run[det_id] = 0
        logger.debug("Structure %s reset pipeline state (%s)", self.config.structure_id, reason)

    def handle_lifecycle_event(self, event: LifecycleEvent) -> None:
        """Process stream lifecycle event (recording boundaries, source status, gaps)."""
        self.quality_layer.handle_lifecycle_event(event)
        if event.event_type == EventType.SOURCE_UP:
            self.source_status = "UP"
        elif event.event_type == EventType.SOURCE_DOWN:
            self.source_status = "DOWN"
            self.last_quality_report = self.quality_layer.evaluate_offline()
        elif event.event_type == EventType.RECORDING_START:
            rec_id = str(event.details.get("recording_id", ""))
            if self._current_recording_id is not None and rec_id != self._current_recording_id:
                self._reset_for_recording_boundary(f"start:{rec_id}")
            self._current_recording_id = rec_id or None
            self._current_recording_samples = 0
        elif event.event_type == EventType.RECORDING_END:
            if self._current_recording_samples > 0:
                self.recordings_completed += 1
                self.hours_observed += (
                    self._current_recording_samples / max(self.config.effective_detector_fs, 1e-6)
                ) / 3600.0
            self._reset_for_recording_boundary(f"end:{event.details.get('recording_id', '')}")
            self._current_recording_id = None
            self._current_recording_samples = 0
        elif event.event_type == EventType.GAP:
            self.dropped_frames_total += int(event.details.get("dropped_frames", 1))
            if self.source_status == "UP":
                self.source_status = "DEGRADED"

    def push_ingestion_item(
        self,
        item: IngestionStreamItem,
        spool_write: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Process one Frame or LifecycleEvent through Resampler -> QualityLayer -> Detectors."""
        if isinstance(item, LifecycleEvent):
            self.handle_lifecycle_event(item)
            return None

        aligned = self._align_frame_channels(item)
        if spool_write:
            try:
                self.spool.append(aligned)
            except Exception as exc:
                logger.warning("Spool append failed for %s: %s", self.config.structure_id, exc)

        # Edge resampling (native fs -> detector fs)
        target_fs = self.config.effective_detector_fs
        if abs(float(aligned.fs) - target_fs) > 1e-3 and self.config.resampling.enabled:
            res_items = self.resampler.process_frame(aligned)
            resampled_frames: List[Frame] = []
            for ri in res_items:
                if isinstance(ri, LifecycleEvent):
                    self.handle_lifecycle_event(ri)
                elif isinstance(ri, Frame) and ri.samples.shape[0] > 0:
                    resampled_frames.append(ri)
            if not resampled_frames:
                return None
        else:
            resampled_frames = [aligned]

        # Data-quality evaluation (10 checks + ordering/dedup/late handling)
        evaluated_pairs: List[Tuple[Frame, QualityReport]] = []
        for rf in resampled_frames:
            pairs, _ = self.quality_layer.evaluate_frame(rf)
            evaluated_pairs.extend(pairs)
        if not evaluated_pairs:
            return None

        latest_payload: Optional[Dict[str, Any]] = None
        now = time.time()
        self.source_status = "UP" if self.source_status in ("IDLE", "DOWN", "UP") else self.source_status
        self.last_data_timestamp = now

        for ord_frame, q_report in evaluated_pairs:
            self.last_quality_report = q_report
            arr = ord_frame.samples
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            n_new = int(arr.shape[0])
            self.frame_counter += 1
            self.frames_processed += 1
            self.samples_processed += n_new
            self._current_recording_samples += n_new

            # Append to shared circular buffer
            if len(self.shared_buffer) == 0:
                self.shared_buffer = arr.astype(np.float32, copy=True)
            else:
                self.shared_buffer = np.vstack([self.shared_buffer, arr.astype(np.float32, copy=False)])
            if len(self.shared_buffer) > self.buffer_capacity:
                self.shared_buffer = self.shared_buffer[-self.buffer_capacity :]

            # Compute per-channel usable_mask from QualityReport (never pass FAULT/OFFLINE into detectors)
            usable_mask: List[bool] = []
            for sid in self.config.channel_names:
                st = q_report.sensors.get(sid)
                is_bad = st is not None and str(st.status) in ("FAULT", "OFFLINE", QualityStatus.FAULT.value, QualityStatus.OFFLINE.value)
                usable_mask.append(not is_bad)

            # Hardware fault reason only blocks all detectors when zero accel channels remain usable
            if not q_report.usable_channels:
                self.hardware_fault_reason = (
                    f"All accelerometer channels faulted/offline: "
                    f"faulted={list(q_report.faulted_channels.keys())}, offline={q_report.offline_channels}"
                )
            else:
                self.hardware_fault_reason = None

            # Update lag & 1-minute throughput telemetry
            end_data_sec = (ord_frame.t0 / 1e9) + (n_new / max(ord_frame.fs, 1e-6))
            is_hist = bool(ord_frame.flags.get("historical_replay", False)) or ("recording_id" in ord_frame.flags)
            lag_sec = 0.0 if is_hist else max(0.0, now - end_data_sec)
            self._rate_window.append((now, n_new, lag_sec))
            cutoff_60 = now - 60.0
            while self._rate_window and self._rate_window[0][0] < cutoff_60:
                self._rate_window.popleft()
            if len(self._rate_window) >= 2:
                span = max(now - self._rate_window[0][0], 1e-3)
                self.effective_frames_per_sec = len(self._rate_window) / span
                self.effective_samples_per_sec = sum(x[1] for x in self._rate_window) / span
                self.processing_lag_sec = float(np.mean([x[2] for x in self._rate_window]))
            else:
                self.processing_lag_sec = lag_sec

            if now - self._last_metric_log_wall >= 60.0:
                self._last_metric_log_wall = now
                if self.dropped_frames_total > 0 or self.processing_lag_sec > 5.0:
                    self.source_status = "DEGRADED"
                logger.info(
                    "Ingestion metrics [%s]: fps=%.2f sps=%.1f lag=%.3fs dropped=%d quality=%.2f",
                    self.config.structure_id,
                    self.effective_frames_per_sec,
                    self.effective_samples_per_sec,
                    self.processing_lag_sec,
                    self.dropped_frames_total,
                    q_report.overall_quality,
                )

            # Run detectors cleanly at data-time cadence (so both 1x live and fast replay trigger accurately)
            for det_id, det in self.detectors.items():
                self._samples_since_det_run[det_id] = self._samples_since_det_run.get(det_id, 0) + n_new
                cadence_samples = max(1, int(round(det.update_cadence_seconds * target_fs)))
                buf_attr = "ring_buffer" if hasattr(det, "ring_buffer") else ("buffer" if hasattr(det, "buffer") else None)
                req_len = getattr(det, "buffer_size", getattr(det, "seq_len", 0))
                cur_buf_len = len(getattr(det, buf_attr)) if buf_attr else 0
                warming_up = bool(buf_attr and cur_buf_len < req_len)
                should_run = (
                    self.frames_processed == 1
                    or self._samples_since_det_run[det_id] >= cadence_samples
                    or (warming_up and cur_buf_len + n_new >= req_len)
                )
                try:
                    if not should_run and not warming_up and buf_attr and q_report.temperature_ok:
                        # Append to detector ring buffer between evaluation cadences without re-stepping persistence
                        existing_buf = getattr(det, buf_attr)
                        combined_buf = np.vstack([existing_buf, arr]) if len(existing_buf) else arr.copy()
                        setattr(det, buf_attr, combined_buf[-req_len:])
                    else:
                        out = det.update(arr, usable_mask=usable_mask, temperature_ok=q_report.temperature_ok)
                        self.latest_outputs[det_id] = out
                        if should_run:
                            self._samples_since_det_run[det_id] = 0
                            self.detector_last_run[det_id] = now
                except Exception as exc:
                    logger.error("Detector %s crashed: %s", det_id, exc, exc_info=True)
                    self.latest_outputs[det_id] = DetectorOutput(
                        detector_id=det_id,
                        version=det.version,
                        timestamp=now,
                        state=DetectorState.SENSOR_FAULT,
                        score=-1.0,
                        tau=det.tau,
                        update_cadence_seconds=det.update_cadence_seconds,
                        reason=f"DETECTOR_EXCEPTION ({exc.__class__.__name__}): {exc}",
                        provenance={"error": str(exc)},
                    )
                    self.detector_last_run[det_id] = now

            latest_payload = self.format_telemetry(arr[-1])
            if self.broadcast_fn is not None:
                if asyncio.iscoroutinefunction(self.broadcast_fn):
                    try:
                        asyncio.create_task(self.broadcast_fn(latest_payload))
                    except RuntimeError:
                        pass
                else:
                    self.broadcast_fn(latest_payload)

        return latest_payload

    def enqueue_item(self, item: IngestionStreamItem) -> bool:
        """Push an item onto the bounded ingestion queue with drop-oldest backpressure."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        dropped = False
        if self._queue.full():
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
                dropped = True
                self.dropped_frames_total += 1
                self.source_status = "DEGRADED"
                self.quality_layer.handle_lifecycle_event(
                    LifecycleEvent(
                        event_type=EventType.GAP,
                        structure_id=self.config.structure_id,
                        source_id="backpressure_queue",
                        reason="Bounded ingestion queue full; dropped oldest frame",
                        details={"dropped_frames": 1},
                    )
                )
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(item)
        return not dropped

    async def start_worker(self) -> None:
        """Start background queue consumer task for this structure."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        if self._worker_task and not self._worker_task.done():
            return

        async def _consume() -> None:
            assert self._queue is not None
            while True:
                item = await self._queue.get()
                try:
                    self.push_ingestion_item(item)
                except Exception as exc:
                    logger.error("Queue worker error for %s: %s", self.config.structure_id, exc, exc_info=True)
                finally:
                    self._queue.task_done()

        self._worker_task = asyncio.create_task(_consume(), name=f"worker-{self.config.structure_id}")

    async def run_ingestion_source(self, source: IngestionSource, use_queue: bool = True) -> None:
        """Consume an async IngestionSource forever with exception isolation."""
        self._active_source = source
        if use_queue:
            await self.start_worker()
        try:
            async for item in source:
                if use_queue:
                    self.enqueue_item(item)
                else:
                    self.push_ingestion_item(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.source_status = "DOWN"
            logger.error("Ingestion source %s crashed for %s: %s", source.source_id, self.config.structure_id, exc, exc_info=True)
            self.handle_lifecycle_event(
                LifecycleEvent(
                    event_type=EventType.SOURCE_DOWN,
                    structure_id=self.config.structure_id,
                    source_id=source.source_id,
                    reason=f"Source error: {exc}",
                )
            )

    async def stop_pipeline(self) -> None:
        """Stop source and queue consumer tasks gracefully."""
        if self._active_source is not None:
            await self._active_source.stop()
        if self._source_task and not self._source_task.done():
            self._source_task.cancel()
            try:
                await self._source_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass

    def get_ingestion_health(self) -> Dict[str, Any]:
        """Return ingestion throughput, lag, queue depth, quality summary, and confirmation metrics."""
        q_depth = self._queue.qsize() if self._queue is not None else 0
        q_report = self.last_quality_report
        confirmation_delay_recordings = (
            max(d.persistence_k for d in self.detectors.values()) if self.detectors else 1
        )
        rec_hours = (self.hours_observed / max(self.recordings_completed, 1)) if self.recordings_completed > 0 else (300.0 / 3600.0)
        return {
            "structure_id": self.config.structure_id,
            "source_type": self.config.ingestion.type,
            "source_status": self.source_status,
            "acquisition_mode": str(self.config.acquisition_mode.value if hasattr(self.config.acquisition_mode, "value") else self.config.acquisition_mode),
            "native_fs": self.config.effective_native_fs,
            "detector_fs": self.config.effective_detector_fs,
            "last_data_timestamp": self.last_data_timestamp,
            "effective_frames_per_sec": round(self.effective_frames_per_sec, 3),
            "effective_samples_per_sec": round(self.effective_samples_per_sec, 2),
            "processing_lag_sec": round(self.processing_lag_sec, 4),
            "queue_depth": q_depth,
            "max_queue_size": self._max_queue_size,
            "dropped_frames_total": self.dropped_frames_total,
            "frames_processed": self.frames_processed,
            "samples_processed": self.samples_processed,
            "recordings_completed": self.recordings_completed,
            "hours_observed": round(self.hours_observed, 3),
            "confirmation_delay": {
                "recordings": confirmation_delay_recordings,
                "hours": round(confirmation_delay_recordings * rec_hours, 3),
            },
            "quality": q_report.model_dump() if q_report else None,
            "quality_events": [e.model_dump() for e in self.quality_layer.get_events(limit=25)],
        }

    def format_telemetry(self, current_frame: np.ndarray) -> Dict[str, Any]:
        """Formats rich domain-agnostic telemetry payload."""
        now = time.time()
        combined_state, contrib_det, reason = combine_states(
            self.latest_outputs,
            self.hardware_fault_reason,
        )

        # Per-sensor metadata, quality disposition, and values
        sensors_list = []
        nodes_dict = {}
        q_rep = self.last_quality_report
        for i, s_cfg in enumerate(self.config.sensors):
            val = float(current_frame[i]) if i < len(current_frame) and np.isfinite(current_frame[i]) else 0.0
            q_state = q_rep.sensors.get(s_cfg.id) if q_rep else None
            if q_state is not None:
                q_st_val = str(q_state.status)
                if q_st_val == QualityStatus.OFFLINE.value:
                    s_status = "OFFLINE"
                elif q_st_val == QualityStatus.FAULT.value:
                    s_status = "SENSOR_FAULT"
                else:
                    s_status = "DEGRADED" if q_st_val == QualityStatus.DEGRADED.value else "NORMAL"
                s_quality = round(q_state.quality_score, 3)
                s_reasons = list(q_state.reasons)
                s_fault = q_state.fault_type
                s_last_seen = q_state.last_seen_ns
            else:
                s_status = "NORMAL" if not self.hardware_fault_reason else "SENSOR_FAULT"
                s_quality = 1.0
                s_reasons = []
                s_fault = None
                s_last_seen = None
            sensors_list.append({
                "id": s_cfg.id,
                "name": s_cfg.name,
                "type": str(s_cfg.type),
                "unit": s_cfg.unit,
                "axis": s_cfg.axis,
                "current_value": round(val, 4),
                "position_xyz": list(s_cfg.position_xyz) if s_cfg.position_xyz else None,
                "mesh_name": s_cfg.mesh_name,
                "status": s_status,
                "quality_score": s_quality,
                "fault_type": s_fault,
                "reasons": s_reasons,
                "last_seen_ns": s_last_seen,
            })
            # Also populate nodes_dict for backward compatibility
            nodes_dict[s_cfg.name] = {
                "mse": round(val, 6),
                "threshold": 0.04,
                "ratio": 1.0,
                "status": s_status,
            }

        # Per-detector summary
        detectors_dict = {}
        localized_node = None
        for det_id, out in self.latest_outputs.items():
            detectors_dict[det_id] = {
                "state": out.state.value if isinstance(out.state, DetectorState) else str(out.state),
                "score": round(out.score, 4),
                "tau": round(out.tau, 4),
                "update_cadence_seconds": out.update_cadence_seconds,
                "last_update_timestamp": round(out.timestamp, 3),
                "reason": out.reason,
                "details": out.details,
            }
            if out.details.get("defect_node"):
                localized_node = out.details.get("defect_node")

        # Spectral summary & calibration coverage
        spectral_summary = {}
        temp_val = None
        calib_coverage = True
        temp_sensor = self.config.get_temperature_sensor()
        if temp_sensor and temp_sensor.channel_index is not None:
            if temp_sensor.channel_index < len(current_frame):
                temp_val = float(current_frame[temp_sensor.channel_index])

        spec_det = self.detectors.get("spectral_tracking")
        if isinstance(spec_det, SpectralTrackingDetector):
            t_min, t_max = spec_det.training_temp_range
            if temp_val is not None:
                calib_coverage = bool(t_min <= temp_val <= t_max)
            spectral_summary = {
                "temperature": round(temp_val, 2) if temp_val is not None else None,
                "calibrated_range": [round(t_min, 1), round(t_max, 1)],
                "modal_frequencies_hz": spec_det.last_modal_frequencies,
                "modal_residuals_hz": spec_det.last_residuals,
                "ewma_score": round(spec_det.ewma_score, 4) if spec_det.ewma_score is not None else None,
            }

        state_str = combined_state.value if isinstance(combined_state, DetectorState) else str(combined_state)

        return {
            "structure_id": self.config.structure_id,
            "structure_name": self.config.name,
            "structure_type": self.config.structure_type,
            "timestamp": now,
            "combined_state": state_str,
            "system_status": state_str,  # backward compatibility
            "localized_defect_node": localized_node,
            "contributing_detector": contrib_det,
            "primary_reason": reason,
            "detectors": detectors_dict,
            "sensors": sensors_list,
            "nodes": nodes_dict,  # backward compatibility
            "spectral_summary": spectral_summary,
            "calibration_coverage": {
                "in_calibrated_bounds": calib_coverage,
                "current_temperature": round(temp_val, 2) if temp_val is not None else None,
            },
            "healthy_hours_trained": sum(d.healthy_hours_trained for d in self.detectors.values()),
            "ingestion": {
                "source_status": self.source_status,
                "last_data_timestamp": self.last_data_timestamp,
                "effective_frames_per_sec": round(self.effective_frames_per_sec, 3),
                "effective_samples_per_sec": round(self.effective_samples_per_sec, 2),
                "processing_lag_sec": round(self.processing_lag_sec, 4),
                "dropped_frames_total": self.dropped_frames_total,
                "recordings_completed": self.recordings_completed,
                "hours_observed": round(self.hours_observed, 3),
            },
            "quality": q_rep.model_dump() if q_rep else None,
        }


class StructureRegistry:
    """Central registry keyed by structure_id."""

    def __init__(self, artifacts_base_dir: Optional[Path] = None) -> None:
        self.artifacts_base_dir = artifacts_base_dir or (Path(__file__).resolve().parent.parent.parent / "artifacts")
        self._runtimes: Dict[str, StructureRuntime] = {}
        self._configs: Dict[str, StructureConfig] = {}

    @property
    def runtimes(self) -> Dict[str, StructureRuntime]:
        """Dictionary of registered structure runtimes."""
        return self._runtimes

    def add(
        self,
        config: StructureConfig,
        broadcast_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        load_artifacts: bool = True,
        version: str = "v1",
    ) -> StructureRuntime:
        """Register a structure and instantiate its runtime."""
        runtime = StructureRuntime(
            config=config,
            artifacts_base_dir=self.artifacts_base_dir,
            broadcast_fn=broadcast_fn,
        )
        if load_artifacts:
            runtime.load_detector_artifacts(version=version)

        self._runtimes[config.structure_id] = runtime
        self._configs[config.structure_id] = config
        logger.info(f"Registered structure '{config.structure_id}' ({config.name}) with {len(runtime.detectors)} detectors.")
        return runtime

    def get(self, structure_id: str) -> Optional[StructureRuntime]:
        return self._runtimes.get(structure_id)

    def get_config(self, structure_id: str) -> Optional[StructureConfig]:
        return self._configs.get(structure_id)

    def remove(self, structure_id: str) -> bool:
        """Remove a structure runtime."""
        if structure_id in self._runtimes:
            del self._runtimes[structure_id]
            self._configs.pop(structure_id, None)
            logger.info(f"Removed structure '{structure_id}' from registry.")
            return True
        return False

    def list(self) -> List[Dict[str, Any]]:
        """List all registered structures."""
        out = []
        for sid, cfg in self._configs.items():
            rt = self._runtimes.get(sid)
            detectors_info = [
                {
                    "detector_id": d_id,
                    "type": d.__class__.__name__,
                    "cadence_seconds": d.update_cadence_seconds,
                    "is_fitted": d.is_fitted,
                    "tau": d.tau,
                }
                for d_id, d in (rt.detectors.items() if rt else [])
            ]
            out.append({
                "structure_id": cfg.structure_id,
                "name": cfg.name,
                "structure_type": cfg.structure_type,
                "fs": cfg.fs,
                "num_channels": cfg.num_channels,
                "sensors": [s.model_dump() for s in cfg.sensors],
                "detectors": detectors_info,
            })
        return out

    def load_from_configs_dir(self, configs_dir: Union[str, Path]) -> int:
        """Loads and registers all .yaml configurations in the directory."""
        p = Path(configs_dir)
        if not p.is_dir():
            return 0
        count = 0
        for yaml_file in sorted(p.glob("*.yaml")):
            try:
                cfg = StructureConfig.from_yaml(yaml_file)
                self.add(cfg, load_artifacts=False)
                count += 1
            except Exception as exc:
                logger.error(f"Failed to load config from {yaml_file}: {exc}")
        return count
