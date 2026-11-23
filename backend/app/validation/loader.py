"""Data loader and preprocessing for KW51 bridge vibration dataset.

Channel Mapping:
----------------
Ch 0: aBD11Az - Bridge deck Section 11, Track A, vertical (left quarter-span)
Ch 1: aBD17Ay - Bridge deck Section 17, Track A, transverse (midspan transverse)
Ch 2: aBD17Az - Bridge deck Section 17, Track A, vertical (midspan vertical bending)
Ch 3: aBD23Az - Bridge deck Section 23, Track A, vertical (right quarter-span)
Ch 4: tVL     - Ambient temperature at bridge site in °C (from Vlierzele RMI station)

Resampling Pipeline:
--------------------
Raw sample rate: 825.806 Hz (247,742 samples per 5-minute file)
Anti-aliasing: 8th-order Butterworth lowpass filter, cutoff = 20.0 Hz (forward-backward zero-phase)
Decimation: Polyphase rational resampling with up=31, down=512 -> exact 50.0 Hz (15,001 samples)
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as sio
import scipy.signal as signal


def resolve_path(p: Union[str, Path]) -> str:
    """Resolves path relative to cwd or repository root."""
    path = Path(p)
    if path.is_absolute() and path.exists():
        return str(path)
    if path.exists():
        return str(path.resolve())
    # Try relative to repo root (3 levels up from this file)
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / path
    if candidate.exists():
        return str(candidate.resolve())
    return str(path)


CHANNEL_DESCRIPTIONS: Dict[str, str] = {
    "aBD11Az": "Left quarter-span deck (Track A, vertical)",
    "aBD17Ay": "Midspan deck (Track A, transverse)",
    "aBD17Az": "Midspan deck (Track A, vertical bending)",
    "aBD23Az": "Right quarter-span deck (Track A, vertical)",
    "tVL": "Ambient temperature at bridge site (°C)",
}


@dataclass
class BridgeFileMetadata:
    """Metadata for a single 5-minute ambient vibration recording."""

    file_name: str
    source_type: str  # 'zip' or 'disk'
    timestamp_str: str  # e.g. '20181016_000000'
    raw_fs: float
    raw_samples: int
    resampled_samples: int
    temperature: float


class KW51Loader:
    """Loads, filters, and decimates KW51 MAT files to 50 Hz."""

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        include_temperature: bool = True,
        raw_fs: float = 825.8064516129032,
        target_fs: float = 50.0,
        lowpass_cutoff: float = 20.0,
        filter_order: int = 8,
        polyphase_up: int = 31,
        polyphase_down: int = 512,
    ) -> None:
        if channels is None:
            self.accel_channels = ["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az"]
        else:
            self.accel_channels = [c for c in channels if c != "tVL"]

        self.include_temperature = include_temperature and ("tVL" in (channels or ["tVL"]))
        self.raw_fs = raw_fs
        self.target_fs = target_fs
        self.lowpass_cutoff = lowpass_cutoff
        self.filter_order = filter_order
        self.polyphase_up = polyphase_up
        self.polyphase_down = polyphase_down

        # Precompute Butterworth filter in second-order sections (SOS) for numerical stability
        self._sos = signal.butter(
            self.filter_order,
            self.lowpass_cutoff,
            btype="low",
            fs=self.raw_fs,
            output="sos",
        )

    @property
    def output_channels(self) -> List[str]:
        """List of active channel names in output order."""
        chans = list(self.accel_channels)
        if self.include_temperature:
            chans.append("tVL")
        return chans

    def _read_mat_structs(self, file_source: Union[str, Path, io.BytesIO]) -> Tuple[object, object]:
        """Loads predat_a and predat_env from file or buffer."""
        mat = sio.loadmat(file_source, squeeze_me=True, struct_as_record=False)
        return mat["predat_a"], mat["predat_env"]

    def process_mat_data(
        self,
        predat_a: object,
        predat_env: object,
        file_name: str = "",
    ) -> Tuple[np.ndarray, BridgeFileMetadata]:
        """Filters, decimates and stacks vibration + temperature into [time, channels]."""
        raw_labels = list(predat_a.labels)
        raw_tdata = predat_a.tdata

        # Locate accelerometer indices
        indices = [raw_labels.index(c) for c in self.accel_channels]
        selected_raw = raw_tdata[:, indices].astype(np.float64)

        # Zero-phase anti-aliasing lowpass filter
        filtered = signal.sosfiltfilt(self._sos, selected_raw, axis=0)

        # Decimate to 50 Hz via polyphase resampling
        resampled_accel = signal.resample_poly(
            filtered,
            self.polyphase_up,
            self.polyphase_down,
            axis=0,
        ).astype(np.float32)

        # Extract temperature
        env_labels = list(predat_env.labels)
        if "tVL" in env_labels:
            t_idx = env_labels.index("tVL")
            temp_val = float(predat_env.data[t_idx])
            if np.isnan(temp_val) and "tBD31A" in env_labels:
                alt_idx = env_labels.index("tBD31A")
                temp_val = float(predat_env.data[alt_idx])
        else:
            temp_val = 15.0

        if np.isnan(temp_val):
            temp_val = 15.0

        n_samples = resampled_accel.shape[0]

        if self.include_temperature:
            temp_col = np.full((n_samples, 1), temp_val, dtype=np.float32)
            out_matrix = np.hstack([resampled_accel, temp_col])
        else:
            out_matrix = resampled_accel

        meta = BridgeFileMetadata(
            file_name=file_name,
            source_type="disk" if os.path.exists(file_name) else "zip",
            timestamp_str=os.path.basename(file_name).split(".")[0],
            raw_fs=float(predat_a.fs),
            raw_samples=len(raw_tdata),
            resampled_samples=n_samples,
            temperature=temp_val,
        )

        return out_matrix, meta

    def load_from_zip(
        self,
        zip_path: str,
        internal_file: str,
    ) -> Tuple[np.ndarray, BridgeFileMetadata]:
        """Loads and processes a single recording from a zip archive."""
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(internal_file) as f:
                buf = io.BytesIO(f.read())
                pa, pe = self._read_mat_structs(buf)
                return self.process_mat_data(pa, pe, file_name=internal_file)

    def load_from_file(
        self,
        file_path: str,
    ) -> Tuple[np.ndarray, BridgeFileMetadata]:
        """Loads and processes a single recording from a local .mat file."""
        pa, pe = self._read_mat_structs(file_path)
        return self.process_mat_data(pa, pe, file_name=file_path)


def list_partition_files(
    zip_2018: str,
    dir_2019: str,
    splits_cfg: dict,
) -> Dict[str, List[Tuple[str, str]]]:
    """Partitions available files into time-based splits.

    Returns dict mapping split_name -> list of (source_type, path_or_entry).
    """
    resolved_zip = resolve_path(zip_2018)
    resolved_dir = resolve_path(dir_2019)

    splits: Dict[str, List[Tuple[str, str]]] = {
        "healthy_train": [],
        "healthy_val": [],
        "healthy_test": [],
        "damaged_test": [],
    }

    # 2018 files from zip
    if os.path.exists(resolved_zip):
        with zipfile.ZipFile(resolved_zip, "r") as z:
            all_entries = sorted([e for e in z.namelist() if e.endswith(".mat")])

            for entry in all_entries:
                base = os.path.basename(entry)
                # Format: ambient_20181016_000000.mat -> date: 2018-10-16
                parts = base.split("_")
                if len(parts) >= 2 and len(parts[1]) == 8:
                    date_str = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"
                    # Check partitions
                    for split_name in ["healthy_train", "healthy_val", "healthy_test"]:
                        cfg = splits_cfg[split_name]
                        if cfg["start_date"] <= date_str <= cfg["end_date"]:
                            splits[split_name].append(("zip", entry))
                            break

    # 2019 files from directory
    if os.path.exists(resolved_dir):
        disk_entries = sorted([
            os.path.join(resolved_dir, f)
            for f in os.listdir(resolved_dir)
            if f.endswith(".mat")
        ])
        for fpath in disk_entries:
            base = os.path.basename(fpath)
            parts = base.split("_")
            if len(parts) >= 2 and len(parts[1]) == 8:
                date_str = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}"
                cfg = splits_cfg["damaged_test"]
                if cfg["start_date"] <= date_str <= cfg["end_date"]:
                    splits["damaged_test"].append(("disk", fpath))

    return splits
