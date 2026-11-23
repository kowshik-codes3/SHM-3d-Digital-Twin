"""Dataset and DataLoader wrappers for SHM sequence processing."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

from .scaler_utils import StructuralDataScaler
from .sliding_window import generate_sliding_windows


class SHMTimeSeriesDataset(Dataset):
    """Memory-efficient PyTorch Dataset for SHM multi-channel sequences."""
    
    def __init__(self, tensor_data: np.ndarray):
        """
        Initializes the dataset with 3D standard arrays.
        
        Args:
            tensor_data: 3D numpy array of shape (M, T, C).
        """
        # Store as contiguous float32 PyTorch tensor
        self.data = torch.from_numpy(tensor_data).type(torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns sequence tensor of shape (T, C)
        return self.data[idx]


class SHMDataModule:
    """End-to-end data manager handling splitting, scaling, and DataLoaders."""

    def __init__(
        self,
        train_parquet: str,
        test_parquet: str,
        channels: List[str],
        window_size: int = 50,
        train_stride: int = 5,
        eval_stride: int = 1,
        batch_size: int = 64,
        train_split_ratio: float = 0.75,
        artifacts_dir: str = "artifacts"
    ):
        """
        Initializes standard parameters for partitioning and tokenization.
        
        Args:
            train_parquet: File path to healthy training parquet dataset.
            test_parquet: File path to damaged testing parquet dataset.
            channels: Mapped physical columns inside target datasets.
            window_size: Continuous sequence window steps. Default is 50.
            train_stride: Training phase stride resolution. Default is 5.
            eval_stride: Evaluation/testing phase stride resolution. Default is 1.
            batch_size: PyTorch loader batch dimension size. Default is 64.
            train_split_ratio: Splitting threshold on chronological data index. Default is 0.75.
            artifacts_dir: Joblib binary persistence folder path. Default is 'artifacts'.
        """
        self.train_parquet = train_parquet
        self.test_parquet = test_parquet
        self.channels = channels
        self.window_size = window_size
        self.train_stride = train_stride
        self.eval_stride = eval_stride
        self.batch_size = batch_size
        self.train_split_ratio = train_split_ratio
        self.artifacts_dir = Path(artifacts_dir)
        
        self.scaler = StructuralDataScaler(feature_names=self.channels)

    def setup(self) -> None:
        """Ingets parquets, splits baseline historically, scales features, and creates windows."""
        # 1. Ingest raw DataFrames
        df_healthy = pd.read_parquet(self.train_parquet)
        df_damaged = pd.read_parquet(self.test_parquet)

        # 2. Chronological Split on baseline healthy data (no shuffling to prevent leakage)
        split_idx = int(len(df_healthy) * self.train_split_ratio)
        df_train = df_healthy.iloc[:split_idx]
        df_val = df_healthy.iloc[split_idx:]

        # 3. Fit scaler ONLY on train, transform all partitions to prevent data leakage
        train_scaled = self.scaler.fit_transform(df_train)
        val_scaled = self.scaler.transform(df_val)
        test_scaled = self.scaler.transform(df_damaged)

        # Persist scaler parameters to artifacts path
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.scaler.save(self.artifacts_dir / "scaler.joblib")

        # 4. Generate 3D sliding window tensors
        self.train_windows = generate_sliding_windows(train_scaled, self.window_size, self.train_stride)
        self.val_windows = generate_sliding_windows(val_scaled, self.window_size, self.eval_stride)
        self.test_windows = generate_sliding_windows(test_scaled, self.window_size, self.eval_stride)

        # 5. Instantiate Datasets
        self.train_ds = SHMTimeSeriesDataset(self.train_windows)
        self.val_ds = SHMTimeSeriesDataset(self.val_windows)
        self.test_ds = SHMTimeSeriesDataset(self.test_windows)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,          # Shuffles temporal windows across batches
            drop_last=True,        # Ensures deterministic batch shapes for PyTorch sequence modeling
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True
        )
