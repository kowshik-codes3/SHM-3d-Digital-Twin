"""Time-based non-shuffled dataset splits and sliding-window generation for KW51.

Strictly follows time-series causal integrity:
- Splits are strictly ordered in time (healthy-train -> healthy-val -> healthy-test -> damaged-test).
- StructuralDataScaler is fitted strictly on healthy-train.
- Val and test splits are transformed using the train-fitted scaler parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ml.scaler_utils import StructuralDataScaler
from app.validation.loader import KW51Loader, list_partition_files, resolve_path


@dataclass
class DatasetSplits:
    """Container for preprocessed, scaled arrays and windowed PyTorch tensors."""

    channel_names: List[str]
    scaler: StructuralDataScaler

    # Continuous 2D arrays [time, channels]
    raw_train: np.ndarray
    raw_val: np.ndarray
    raw_test_healthy: np.ndarray
    raw_test_damaged: np.ndarray

    scaled_train: np.ndarray
    scaled_val: np.ndarray
    scaled_test_healthy: np.ndarray
    scaled_test_damaged: np.ndarray

    # Sliding window 3D tensors (N, seq_len, channels)
    windows_train: torch.Tensor
    windows_val: torch.Tensor
    windows_test_healthy: torch.Tensor
    windows_test_damaged: torch.Tensor

    # Timestamps / file metadata
    meta_train: list
    meta_val: list
    meta_test_healthy: list
    meta_test_damaged: list


def create_sliding_windows(
    data: np.ndarray,
    window_size: int = 50,
    stride: int = 25,
) -> np.ndarray:
    """Converts (T, C) time-series into (N, window_size, C) sliding windows."""
    n_samples, n_channels = data.shape
    if n_samples < window_size:
        return np.empty((0, window_size, n_channels), dtype=np.float32)

    num_windows = (n_samples - window_size) // stride + 1
    # Use strided slice for speed and memory efficiency
    sub_windows = [
        data[i * stride : i * stride + window_size]
        for i in range(num_windows)
    ]
    return np.array(sub_windows, dtype=np.float32)


def build_splits(
    config: dict,
    include_temperature: bool = True,
    channels: Optional[List[str]] = None,
) -> DatasetSplits:
    """Loads all split files, fits scaler on train, and generates windowed tensors.

    Args:
        config: Loaded validation YAML dictionary.
        include_temperature: Whether to include temperature channel.
        channels: Override channel list.

    Returns:
        Populated DatasetSplits object.
    """
    selected_channels = channels or (
        config["channels"]["primary"] if include_temperature else config["channels"]["no_temp"]
    )

    loader = KW51Loader(
        channels=selected_channels,
        include_temperature=include_temperature,
        raw_fs=config["resampling"]["raw_fs"],
        target_fs=config["resampling"]["target_fs"],
        lowpass_cutoff=config["resampling"]["lowpass_cutoff_hz"],
        filter_order=config["resampling"]["filter_order"],
        polyphase_up=config["resampling"]["polyphase_up"],
        polyphase_down=config["resampling"]["polyphase_down"],
    )

    active_channels = loader.output_channels

    # Partition file paths
    partitions = list_partition_files(
        zip_2018=config["dataset"]["zip_2018"],
        dir_2019=config["dataset"]["dir_2019"],
        splits_cfg=config["splits"],
    )

    resolved_zip = resolve_path(config["dataset"]["zip_2018"])

    def load_partition(file_list: List[Tuple[str, str]]) -> Tuple[np.ndarray, list]:
        arrays: List[np.ndarray] = []
        meta_list: list = []
        for src_type, f_item in file_list:
            if src_type == "zip":
                arr, meta = loader.load_from_zip(resolved_zip, f_item)
            else:
                arr, meta = loader.load_from_file(f_item)
            arrays.append(arr)
            meta_list.append(meta)

        if not arrays:
            empty_arr = np.empty((0, len(active_channels)), dtype=np.float32)
            return empty_arr, []

        return np.vstack(arrays), meta_list

    # 1. Load raw arrays chronologically
    raw_train, meta_train = load_partition(partitions["healthy_train"])
    raw_val, meta_val = load_partition(partitions["healthy_val"])
    raw_test_healthy, meta_test_healthy = load_partition(partitions["healthy_test"])
    raw_test_damaged, meta_test_damaged = load_partition(partitions["damaged_test"])

    assert len(raw_train) > 0, "healthy_train partition is empty!"
    assert len(raw_val) > 0, "healthy_val partition is empty!"
    assert len(raw_test_healthy) > 0, "healthy_test partition is empty!"
    assert len(raw_test_damaged) > 0, "damaged_test partition is empty!"

    # 2. Fit scaler STRICTLY on healthy_train
    df_train = pd.DataFrame(raw_train, columns=active_channels)
    scaler = StructuralDataScaler(feature_names=active_channels)
    scaler.fit(df_train)

    # 3. Transform all splits using train-fitted scaler
    scaled_train = scaler.transform(raw_train)
    scaled_val = scaler.transform(raw_val)
    scaled_test_healthy = scaler.transform(raw_test_healthy)
    scaled_test_damaged = scaler.transform(raw_test_damaged)

    # 4. Create sliding windows
    w_size = config["windowing"]["window_size"]
    stride_train = config["windowing"]["stride_train"]
    stride_val = config["windowing"]["stride_val"]
    stride_test = config["windowing"]["stride_test"]

    win_train = create_sliding_windows(scaled_train, window_size=w_size, stride=stride_train)
    win_val = create_sliding_windows(scaled_val, window_size=w_size, stride=stride_val)
    win_test_h = create_sliding_windows(scaled_test_healthy, window_size=w_size, stride=stride_test)
    win_test_d = create_sliding_windows(scaled_test_damaged, window_size=w_size, stride=stride_test)

    return DatasetSplits(
        channel_names=active_channels,
        scaler=scaler,
        raw_train=raw_train,
        raw_val=raw_val,
        raw_test_healthy=raw_test_healthy,
        raw_test_damaged=raw_test_damaged,
        scaled_train=scaled_train,
        scaled_val=scaled_val,
        scaled_test_healthy=scaled_test_healthy,
        scaled_test_damaged=scaled_test_damaged,
        windows_train=torch.from_numpy(win_train),
        windows_val=torch.from_numpy(win_val),
        windows_test_healthy=torch.from_numpy(win_test_h),
        windows_test_damaged=torch.from_numpy(win_test_d),
        meta_train=meta_train,
        meta_val=meta_val,
        meta_test_healthy=meta_test_healthy,
        meta_test_damaged=meta_test_damaged,
    )
