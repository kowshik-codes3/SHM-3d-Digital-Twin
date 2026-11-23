"""Replay Ingestion Source supporting HDF5 (KW51 caches), Parquet, CSV, and raw MAT files.

Features:
- Emits `RECORDING_START` and `RECORDING_END` events at recording boundaries.
- Guarantees no Frame or analysis chunk ever spans two separate recordings.
- Supports speed factor (`1.0`, `10.0`, `"max"` / `<= 0` for maximum throughput) and `loop`.
- Deadline-compensated timing loop for accurate sustained playback rate.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from app.ingestion.base import EventType, Frame, IngestionSource, IngestionStreamItem, LifecycleEvent

logger = logging.getLogger("SHM.ReplaySource")


def _parse_iso_to_ns(ts_str: str) -> int:
    try:
        dt = datetime.datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return time.time_ns()


class ReplaySource(IngestionSource):
    """Streams recorded telemetry from HDF5, Parquet, CSV, or raw KW51 .mat files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        fs: float = 50.0,
        speed_multiplier: Union[float, str] = 1.0,
        loop: bool = False,
        structure_id: str = "kw51",
        sensor_ids: Optional[List[str]] = None,
        chunk_samples: int = 50,
        recording_ids: Optional[List[str]] = None,
        include_temperature: bool = True,
    ) -> None:
        super().__init__(name="Replay", structure_id=structure_id)
        self.file_path = Path(file_path)
        self.fs = float(fs)
        if isinstance(speed_multiplier, str) and speed_multiplier.strip().lower() == "max":
            self.speed_multiplier: float = 0.0
        else:
            val = float(speed_multiplier)
            self.speed_multiplier = 0.0 if val <= 0 else val
        self.loop = bool(loop)
        self.chunk_samples = max(1, int(chunk_samples))
        self.recording_ids = recording_ids
        self.include_temperature = include_temperature
        self.sensor_ids = list(sensor_ids) if sensor_ids else []

        if not self.file_path.exists():
            raise FileNotFoundError(f"Replay path not found: {self.file_path}")

        self._recordings: List[Tuple[str, int, float, np.ndarray]] = []
        self._load_recordings()

    def _load_recordings(self) -> None:
        """Load recordings as tuples of (recording_id, t0_ns, fs, data [N, C])."""
        p = self.file_path
        if p.is_dir():
            h5_files = sorted(p.glob("*_50hz.h5")) + sorted(p.glob("*.hdf5"))
            if h5_files:
                for hp in h5_files:
                    self._load_h5_file(hp)
                return
            mat_files = sorted(p.glob("*.mat"))
            if mat_files:
                for mp in mat_files:
                    self._load_mat_file(mp)
                return
            raise FileNotFoundError(f"No supported replay files (.h5/.mat) in directory: {p}")

        ext = p.suffix.lower()
        if ext in (".h5", ".hdf5"):
            self._load_h5_file(p)
        elif ext == ".mat":
            self._load_mat_file(p)
        elif ext == ".parquet":
            df = pd.read_parquet(p)
            self._load_dataframe(df, recording_id=p.stem)
        elif ext == ".csv":
            df = pd.read_csv(p)
            self._load_dataframe(df, recording_id=p.stem)
        else:
            raise ValueError(f"Unsupported replay file extension: {ext}")

        if not self._recordings:
            raise ValueError(f"No recordings loaded from {self.file_path}")

    def _load_h5_file(self, h5_path: Path) -> None:
        with h5py.File(h5_path, "r") as f:
            file_fs = float(f.attrs.get("fs_hz", self.fs))
            self.fs = file_fs
            raw_chs = f.attrs.get("channels")
            if raw_chs is not None and not self.sensor_ids:
                self.sensor_ids = [
                    c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(c)
                    for c in raw_chs
                ]

            if "recordings" in f:
                grp = f["recordings"]
                keys = sorted(grp.keys())
                if self.recording_ids is not None:
                    keys = [k for k in keys if k in set(self.recording_ids)]
                for rid in keys:
                    ds = grp[rid]
                    arr = ds[:].astype(np.float32)
                    t0_ns = _parse_iso_to_ns(ds.attrs.get("timestamp", ""))
                    if not self.sensor_ids:
                        self.sensor_ids = [f"ch_{i}" for i in range(arr.shape[1])]
                    self._recordings.append((rid, t0_ns, file_fs, arr))
            else:
                for key in sorted(f.keys()):
                    if isinstance(f[key], h5py.Dataset):
                        arr = f[key][:].astype(np.float32)
                        if not self.sensor_ids:
                            self.sensor_ids = [f"ch_{i}" for i in range(arr.shape[1])]
                        self._recordings.append((key, time.time_ns(), file_fs, arr))

    def _load_mat_file(self, mat_path: Path) -> None:
        import scipy.io

        mat = scipy.io.loadmat(mat_path)
        pa = mat["predat_a"]
        pe = mat.get("predat_env")
        raw_lbl_arr = pa["labels"][0, 0]
        raw_lbls = [str(raw_lbl_arr[i, 0][0]) for i in range(raw_lbl_arr.shape[0])]
        tdata = pa["tdata"][0, 0]
        raw_fs = float(pa["fs"][0, 0][0, 0]) if "fs" in pa.dtype.names else self.fs
        self.fs = raw_fs

        if not self.sensor_ids:
            # Keep standard KW51 subset if matching or all accel + optional temp
            self.sensor_ids = list(raw_lbls)
            if self.include_temperature and pe is not None:
                self.sensor_ids.append("tVL")

        accel_cols = [s for s in self.sensor_ids if s in raw_lbls]
        indices = [raw_lbls.index(c) for c in accel_cols]
        vib = tdata[:, indices].astype(np.float32)

        cols = [vib]
        if self.include_temperature and "tVL" in self.sensor_ids and pe is not None:
            env_lbls_arr = pe["labels"][0, 0]
            env_lbls = [str(env_lbls_arr[i, 0][0]) for i in range(env_lbls_arr.shape[0])]
            t_idx = env_lbls.index("tVL") if "tVL" in env_lbls else 0
            t_raw = pe["data"][0, 0][:, t_idx]
            mean_temp = float(np.nanmean(t_raw)) if not np.all(np.isnan(t_raw)) else 12.0
            cols.append(np.full((vib.shape[0], 1), mean_temp, dtype=np.float32))

        arr = np.hstack(cols).astype(np.float32)
        parts = mat_path.stem.split("_")
        if len(parts) >= 3:
            try:
                dt = datetime.datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
                dt = dt.replace(tzinfo=datetime.timezone.utc)
                t0_ns = int(dt.timestamp() * 1_000_000_000)
            except Exception:
                t0_ns = time.time_ns()
        else:
            t0_ns = time.time_ns()

        self._recordings.append((mat_path.stem, t0_ns, raw_fs, arr))

    def _load_dataframe(self, df: pd.DataFrame, recording_id: str) -> None:
        if not self.sensor_ids:
            self.sensor_ids = list(df.columns)
        else:
            cols_present = [c for c in self.sensor_ids if c in df.columns]
            if len(cols_present) == len(self.sensor_ids):
                df = df[cols_present]
        arr = df.values.astype(np.float32)
        self._recordings.append((recording_id, time.time_ns(), self.fs, arr))

    @property
    def num_recordings(self) -> int:
        return len(self._recordings)

    @property
    def total_samples(self) -> int:
        return sum(r[3].shape[0] for r in self._recordings)

    async def stream(self) -> AsyncIterator[IngestionStreamItem]:
        """Yield RECORDING_START, bounded Frame chunks, and RECORDING_END events."""
        self._is_running = True
        seq = 0
        yield LifecycleEvent(
            event_type=EventType.SOURCE_UP,
            structure_id=self.structure_id,
            source_id=self.source_id,
        )

        prev_rec_end_ns: Optional[int] = None
        next_deadline = time.perf_counter()

        try:
            while self._is_running:
                for rec_id, rec_t0_ns, rec_fs, data in self._recordings:
                    if not self._is_running:
                        break

                    if prev_rec_end_ns is not None and rec_t0_ns > prev_rec_end_ns:
                        gap_sec = (rec_t0_ns - prev_rec_end_ns) / 1_000_000_000.0
                        if gap_sec > (2.0 / max(rec_fs, 1.0)):
                            yield LifecycleEvent(
                                event_type=EventType.GAP,
                                structure_id=self.structure_id,
                                source_id=self.source_id,
                                timestamp_ns=rec_t0_ns,
                                recording_id=rec_id,
                                reason=f"Inter-recording gap of {gap_sec:.2f}s before {rec_id}",
                                details={"gap_seconds": gap_sec},
                            )

                    yield LifecycleEvent(
                        event_type=EventType.RECORDING_START,
                        structure_id=self.structure_id,
                        source_id=self.source_id,
                        timestamp_ns=rec_t0_ns,
                        recording_id=rec_id,
                        details={"fs": rec_fs, "n_samples": int(data.shape[0])},
                    )

                    n_total = data.shape[0]
                    sample_ns = int(round(1_000_000_000 / rec_fs))

                    for start_idx in range(0, n_total, self.chunk_samples):
                        if not self._is_running:
                            break
                        end_idx = min(n_total, start_idx + self.chunk_samples)
                        chunk = data[start_idx:end_idx]
                        chunk_t0_ns = rec_t0_ns + start_idx * sample_ns

                        yield Frame(
                            structure_id=self.structure_id,
                            sensor_ids=self.sensor_ids,
                            t0=chunk_t0_ns,
                            fs=rec_fs,
                            samples=chunk,
                            seq=seq,
                            source_id=self.source_id,
                            flags={"recording_id": rec_id},
                        )
                        seq += 1

                        if self.speed_multiplier > 0.0:
                            dt_chunk = (end_idx - start_idx) / (rec_fs * self.speed_multiplier)
                            next_deadline += dt_chunk
                            sleep_dur = next_deadline - time.perf_counter()
                            if sleep_dur > 0:
                                await asyncio.sleep(sleep_dur)
                            else:
                                if sleep_dur < -2.0:
                                    next_deadline = time.perf_counter()
                                await asyncio.sleep(0)
                        else:
                            # Max speed: yield control briefly every chunk so event loop stays responsive
                            await asyncio.sleep(0)

                    rec_end_ns = rec_t0_ns + n_total * sample_ns
                    prev_rec_end_ns = rec_end_ns
                    yield LifecycleEvent(
                        event_type=EventType.RECORDING_END,
                        structure_id=self.structure_id,
                        source_id=self.source_id,
                        timestamp_ns=rec_end_ns,
                        recording_id=rec_id,
                    )

                if not self.loop:
                    break
        finally:
            self._is_running = False
            yield LifecycleEvent(
                event_type=EventType.SOURCE_DOWN,
                structure_id=self.structure_id,
                source_id=self.source_id,
                reason="Replay finished" if not self.loop else "Replay stopped",
            )

    def stop(self) -> None:
        self._is_running = False
