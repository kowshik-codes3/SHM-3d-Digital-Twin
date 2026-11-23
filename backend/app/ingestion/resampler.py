"""Edge Anti-Alias Filtering, Rational Decimation & Jitter Resampling.

Handles:
- Configurable anti-alias filtering ('butter', 'cheby1', 'none') and polyphase decimation.
- KW51 raw (~825.806 Hz) -> 50 Hz exact rational parity (up=31, down=512).
- Jittery timestamps and non-integer sampling ratios onto uniform target grids.
- Strict gap-interpolation guard: NEVER interpolates across gaps exceeding max_gap_interpolate_sec.
"""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from app.ingestion.base import EventType, Frame, IngestionStreamItem, LifecycleEvent


class EdgeResampler:
    """Configurable edge anti-alias filter and resampler for streaming/recording ingestion."""

    def __init__(
        self,
        native_fs: float,
        target_fs: float,
        filter_type: str = "butter",
        order: int = 8,
        cutoff_hz: float = 20.0,
        ripple_db: float = 0.5,
        mode: str = "block_zero_phase",
        max_gap_interpolate_sec: float = 0.25,
        temp_channel_indices: Optional[List[int]] = None,
    ) -> None:
        self.native_fs = float(native_fs)
        self.target_fs = float(target_fs)
        self.filter_type = filter_type.lower().strip()
        self.order = int(order)
        self.cutoff_hz = float(cutoff_hz)
        self.ripple_db = float(ripple_db)
        self.mode = mode.lower().strip()
        self.max_gap_interpolate_sec = float(max_gap_interpolate_sec)
        self.temp_channel_indices = set(temp_channel_indices or [])

        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._up: int = 1
        self._down: int = 1
        self._last_t_end_ns: Optional[int] = None

        self._setup_filter_and_ratio()

    def _setup_filter_and_ratio(self) -> None:
        if self.target_fs <= 0 or self.native_fs <= 0:
            raise ValueError("native_fs and target_fs must be positive")

        if abs(self.native_fs - self.target_fs) < 1e-6:
            self._up, self._down = 1, 1
            self._sos = None
            return

        # Exact match for KW51 825.8064516129032 Hz -> 50.0 Hz (up=31, down=512)
        if abs(self.native_fs - (50.0 * 512.0 / 31.0)) < 1e-2 and abs(self.target_fs - 50.0) < 1e-4:
            self._up, self._down = 31, 512
        else:
            frac = Fraction(self.target_fs / self.native_fs).limit_denominator(1024)
            self._up, self._down = frac.numerator, frac.denominator

        nyq = 0.5 * self.native_fs
        if self.cutoff_hz >= nyq or self.filter_type == "none":
            self._sos = None
            return

        wn = min(0.99, self.cutoff_hz / nyq)
        eff_order = max(2, self.order // 2) if self.mode == "block_zero_phase" else max(2, self.order)

        if self.filter_type == "butter":
            self._sos = signal.butter(eff_order, wn, btype="low", output="sos")
        elif self.filter_type in ("cheby1", "chebyshev"):
            self._sos = signal.cheby1(eff_order, self.ripple_db, wn, btype="low", output="sos")
        else:
            raise ValueError(f"Unsupported resampling filter_type: '{self.filter_type}'")

    def reset(self) -> None:
        """Reset stateful streaming filter memory (e.g., at recording boundaries or gaps)."""
        self._zi = None
        self._last_t_end_ns = None

    def resample_array(self, data: np.ndarray, pad_for_zero_phase: bool = True) -> np.ndarray:
        """Filter and resample a contiguous 2D array [n_samples, n_channels]."""
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_samples, n_channels = arr.shape
        if n_samples == 0 or (self._up == 1 and self._down == 1 and self._sos is None):
            return arr.astype(np.float32)

        vib_cols = [i for i in range(n_channels) if i not in self.temp_channel_indices]
        temp_cols = [i for i in range(n_channels) if i in self.temp_channel_indices]

        out_cols: Dict[int, np.ndarray] = {}

        if vib_cols:
            vib = arr[:, vib_cols]
            if self._sos is not None:
                if self.mode == "block_zero_phase" and pad_for_zero_phase and vib.shape[0] > 3 * (self._sos.shape[0] * 2 + 1):
                    filtered = signal.sosfiltfilt(self._sos, vib, axis=0)
                else:
                    if self._zi is None or self._zi.shape[1] != len(vib_cols):
                        zi_base = signal.sosfilt_zi(self._sos)
                        self._zi = np.repeat(zi_base[:, None, :], len(vib_cols), axis=1) * vib[0:1, :]
                    filtered, self._zi = signal.sosfilt(self._sos, vib, axis=0, zi=self._zi)
            else:
                filtered = vib

            if self._up != self._down:
                resampled = signal.resample_poly(filtered, up=self._up, down=self._down, axis=0)
            else:
                resampled = filtered

            for j, c_idx in enumerate(vib_cols):
                out_cols[c_idx] = resampled[:, j]

        out_len = len(next(iter(out_cols.values()))) if out_cols else int(math.ceil(n_samples * self._up / self._down))

        for c_idx in temp_cols:
            t_vals = arr[:, c_idx]
            mean_t = float(np.nanmean(t_vals)) if not np.all(np.isnan(t_vals)) else 0.0
            out_cols[c_idx] = np.full(out_len, mean_t, dtype=np.float64)

        result = np.column_stack([out_cols[i] for i in range(n_channels)]).astype(np.float32)
        return result

    def resample_irregular(
        self,
        timestamps_ns: np.ndarray,
        samples: np.ndarray,
        structure_id: str,
        sensor_ids: List[str],
        source_id: str,
        seq_start: int = 0,
    ) -> List[IngestionStreamItem]:
        """Resample jittery / irregularly-spaced timestamps onto a uniform target_fs grid.

        Never interpolates across gaps exceeding `max_gap_interpolate_sec`; emits
        `LifecycleEvent(GAP)` and yields separated uniform Frame segments instead.
        """
        ts = np.asarray(timestamps_ns, dtype=np.int64)
        arr = np.asarray(samples, dtype=np.float32)
        if len(ts) == 0:
            return []
        if len(ts) == 1:
            return [
                Frame(
                    structure_id=structure_id,
                    sensor_ids=sensor_ids,
                    t0=int(ts[0]),
                    fs=self.target_fs,
                    samples=arr.reshape(1, -1),
                    seq=seq_start,
                    source_id=source_id,
                    flags={"resampled": True},
                )
            ]

        dt_sec = np.diff(ts) / 1_000_000_000.0
        split_indices = np.where(dt_sec > self.max_gap_interpolate_sec)[0] + 1

        seg_bounds = [0] + list(split_indices) + [len(ts)]
        items: List[IngestionStreamItem] = []
        seq = seq_start
        grid_dt_ns = int(round(1_000_000_000.0 / self.target_fs))

        for seg_idx in range(len(seg_bounds) - 1):
            s_i, e_i = seg_bounds[seg_idx], seg_bounds[seg_idx + 1]
            if seg_idx > 0:
                gap_sec = float((ts[s_i] - ts[s_i - 1]) / 1_000_000_000.0)
                self.reset()
                items.append(
                    LifecycleEvent(
                        event_type=EventType.GAP,
                        structure_id=structure_id,
                        source_id=source_id,
                        timestamp_ns=int(ts[s_i]),
                        reason=f"Timestamp gap {gap_sec:.3f}s exceeds interpolation limit ({self.max_gap_interpolate_sec}s)",
                        details={"gap_seconds": gap_sec},
                    )
                )

            seg_ts = ts[s_i:e_i]
            seg_data = arr[s_i:e_i]
            if len(seg_ts) < 2:
                continue

            # Uniform grid directly at target_fs (or native_fs if filter needed first)
            dur_ns = seg_ts[-1] - seg_ts[0]
            n_out = max(1, int(dur_ns // grid_dt_ns) + 1)
            uniform_ts = seg_ts[0] + np.arange(n_out, dtype=np.int64) * grid_dt_ns
            uniform_ts = uniform_ts[uniform_ts <= seg_ts[-1]]

            interp_cols = []
            for c in range(seg_data.shape[1]):
                interp_cols.append(np.interp(uniform_ts.astype(np.float64), seg_ts.astype(np.float64), seg_data[:, c].astype(np.float64)))
            uniform_arr = np.column_stack(interp_cols).astype(np.float32)

            items.append(
                Frame(
                    structure_id=structure_id,
                    sensor_ids=sensor_ids,
                    t0=int(seg_ts[0]),
                    fs=self.target_fs,
                    samples=uniform_arr,
                    seq=seq,
                    source_id=source_id,
                    flags={"resampled": True, "jitter_corrected": True},
                )
            )
            seq += 1

        return items

    def process_frame(self, frame: Frame) -> List[IngestionStreamItem]:
        """Process an incoming Frame chunk through edge resampling if frame.fs differs from target_fs."""
        items: List[IngestionStreamItem] = []

        if self._last_t_end_ns is not None:
            gap_sec = (frame.t0 - self._last_t_end_ns) / 1_000_000_000.0
            if gap_sec > self.max_gap_interpolate_sec:
                self.reset()
                items.append(
                    LifecycleEvent(
                        event_type=EventType.GAP,
                        structure_id=frame.structure_id,
                        source_id=frame.source_id,
                        timestamp_ns=frame.t0,
                        reason=f"Inter-frame gap {gap_sec:.3f}s exceeds limit ({self.max_gap_interpolate_sec}s)",
                        details={"gap_seconds": gap_sec},
                    )
                )

        if abs(frame.fs - self.target_fs) < 1e-4:
            self._last_t_end_ns = frame.t_end
            items.append(frame)
            return items

        # Re-align filter ratio dynamically if frame.fs changed
        if abs(frame.fs - self.native_fs) > 1e-3:
            self.native_fs = frame.fs
            self._setup_filter_and_ratio()

        resampled = self.resample_array(frame.samples)
        flags = dict(frame.flags)
        flags["resampled"] = True
        flags["native_fs"] = frame.fs

        out_frame = Frame(
            structure_id=frame.structure_id,
            sensor_ids=frame.sensor_ids,
            t0=frame.t0,
            fs=self.target_fs,
            samples=resampled,
            seq=frame.seq,
            source_id=frame.source_id,
            flags=flags,
        )
        self._last_t_end_ns = out_frame.t_end
        items.append(out_frame)
        return items
