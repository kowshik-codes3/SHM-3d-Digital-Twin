"""Loader for the Z24 Bridge Progressive Damage Benchmark.

Extracts healthy training/validation subsets from Scenario 0 and progressive damage test
subsets from Scenarios 1 through 16.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class Z24BenchmarkLoader:
    """Loads standardized Z24 benchmark features and partition splits."""

    def __init__(self, data_dir: str = "data/z24/features") -> None:
        self.data_dir = Path(data_dir)
        self.X_train_path = self.data_dir / "X_train_new.npy"
        self.y_train_path = self.data_dir / "labels_train_new.npy"
        self.X_test_path = self.data_dir / "X_test_new.npy"
        self.y_test_path = self.data_dir / "labels_test_new.npy"

        if not self.X_train_path.exists():
            raise FileNotFoundError(f"Missing Z24 dataset files at {self.data_dir}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_tr = np.load(self.X_train_path)
        y_tr = np.load(self.y_train_path)
        X_te = np.load(self.X_test_path)
        y_te = np.load(self.y_test_path)
        return X_tr, y_tr, X_te, y_te

    def get_benchmark_splits(self, seed: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Constructs leak-free splits:
        - healthy-train: 75% of Scenario 0
        - healthy-val: 25% of Scenario 0
        - damage-test: Scenarios 1-16 (Progressive Damage Tests)
        """
        X_tr, y_tr, X_te, y_te = self.load_data()
        X_all = np.vstack([X_tr, X_te])
        y_all = np.concatenate([y_tr, y_te])

        # Filter Scenario 0 (healthy undamaged)
        idx_healthy = np.where(y_all == 0)[0]
        idx_damage = np.where(y_all > 0)[0]

        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx_healthy)
        split_pt = int(0.75 * len(perm))

        train_idx = perm[:split_pt]
        val_idx = perm[split_pt:]

        return {
            "healthy-train": (X_all[train_idx], y_all[train_idx]),
            "healthy-val": (X_all[val_idx], y_all[val_idx]),
            "damage-test": (X_all[idx_damage], y_all[idx_damage]),
        }
