"""Protocol B Data Loader for 50 Hz HDF5 Caches.

Strict time-ordered splits:
- healthy-train: 2018-10-10 to 2018-12-10 (343 recordings)
- healthy-val: 2018-12-11 to 2018-12-31 (124 recordings)
- changed-val: 2019-09-27 to 2019-10-31 (205 recordings)
- healthy-test-cold: 2019-01-01 to 2019-01-31 (185 recordings) [LOCKED]
- healthy-test-warming: 2019-02-01 to 2019-04-30 (504 recordings) [LOCKED]
- healthy-test-warm: 2019-05-01 to 2019-05-14 (84 recordings) [LOCKED]
- transition: 2019-05-15 to 2019-09-26 (639 recordings) [LOCKED]
- changed-test: 2019-11-01 to 2020-01-15 (438 recordings) [LOCKED]

Causal integrity rules:
- StructuralDataScaler fitted STRICTLY on healthy-train.
- Windows NEVER cross recording boundaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

from app.ml.scaler_utils import StructuralDataScaler


DEFAULT_INVENTORY_PATH = "results/data/inventory_50hz.csv"
CACHE_DIR = Path("data/kw51/cache_50hz")

CHANNELS = ["aBD17Ay", "aBD17Az", "aBD23Ay", "aBD23Az", "tVL"]
VIBRATION_CHANNELS = ["aBD17Ay", "aBD17Az", "aBD23Ay", "aBD23Az"]


class ProtocolBSplitsLoader:
    def __init__(self, inventory_path: str = DEFAULT_INVENTORY_PATH, cache_dir: Path = CACHE_DIR) -> None:
        self.inventory_df = pd.read_csv(inventory_path)
        self.inventory_df["ts"] = pd.to_datetime(self.inventory_df["timestamp"])
        self.cache_dir = cache_dir
        self.scaler = StructuralDataScaler(feature_names=CHANNELS)
        self.is_fitted = False

    def get_recording_ids_for_split(self, split_name: str) -> List[dict]:
        df = self.inventory_df
        if split_name == "healthy-train":
            mask = (df["ts"] >= "2018-10-10") & (df["ts"] <= "2018-12-10 23:59:59")
        elif split_name == "healthy-val":
            mask = (df["ts"] >= "2018-12-11") & (df["ts"] <= "2018-12-31 23:59:59")
        elif split_name == "changed-val":
            mask = (df["ts"] >= "2019-09-27") & (df["ts"] <= "2019-10-31 23:59:59")
        elif split_name == "healthy-test-cold":
            mask = (df["ts"] >= "2019-01-01") & (df["ts"] <= "2019-01-31 23:59:59")
        elif split_name == "healthy-test-warming":
            mask = (df["ts"] >= "2019-02-01") & (df["ts"] <= "2019-04-30 23:59:59")
        elif split_name == "healthy-test-warm":
            mask = (df["ts"] >= "2019-05-01") & (df["ts"] <= "2019-05-14 23:59:59")
        elif split_name == "transition":
            mask = (df["ts"] >= "2019-05-15") & (df["ts"] <= "2019-09-26 23:59:59")
        elif split_name == "changed-test":
            mask = (df["ts"] >= "2019-11-01") & (df["ts"] <= "2020-01-15 23:59:59")
        else:
            raise ValueError(f"Unknown split name: {split_name}")
        return df[mask].to_dict(orient="records")

    def load_recordings(self, records: List[dict], max_count: Optional[int] = None) -> List[np.ndarray]:
        if max_count:
            records = records[:max_count]
        
        # Group by month file for optimal HDF5 access
        by_month: Dict[str, List[str]] = {}
        for r in records:
            by_month.setdefault(r["month"], []).append(r["recording_id"])
            
        data_map: Dict[str, np.ndarray] = {}
        for month, rec_ids in by_month.items():
            h5_path = self.cache_dir / f"{month}_50hz.h5"
            with h5py.File(h5_path, "r") as h:
                rec_grp = h["recordings"]
                for rid in rec_ids:
                    if rid in rec_grp:
                        data_map[rid] = rec_grp[rid][:].astype(np.float32)
                        
        ordered_data = [data_map[r["recording_id"]] for r in records if r["recording_id"] in data_map]
        return ordered_data

    def fit_scaler(self, train_recordings: List[np.ndarray]) -> None:
        """Fits StructuralDataScaler on 4 vibration channels + temperature."""
        stacked = np.concatenate(train_recordings, axis=0)
        # Scaler expects DataFrame with column names
        df = pd.DataFrame(stacked, columns=CHANNELS)
        self.scaler.fit(df)
        self.is_fitted = True

    def transform_recordings(self, recordings: List[np.ndarray]) -> List[np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on healthy-train before transforming.")
        scaled = []
        for r in recordings:
            df = pd.DataFrame(r, columns=CHANNELS)
            scaled_arr = self.scaler.transform(df)
            scaled.append(np.asarray(scaled_arr, dtype=np.float32))
        return scaled

    @staticmethod
    def create_recording_windows(
        recordings: List[np.ndarray],
        window_size: int = 50,
        stride: int = 50,
    ) -> torch.Tensor:
        """Extracts sliding windows strictly WITHIN each recording. Never crosses boundaries."""
        all_windows = []
        for rec in recordings:
            T, C = rec.shape
            if T < window_size:
                continue
            num_win = (T - window_size) // stride + 1
            for i in range(num_win):
                all_windows.append(rec[i * stride : i * stride + window_size])
        if not all_windows:
            return torch.empty((0, window_size, recordings[0].shape[1]), dtype=torch.float32)
        return torch.tensor(np.array(all_windows, dtype=np.float32), dtype=torch.float32)
