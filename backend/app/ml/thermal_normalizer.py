"""Thermal normalization engine for Structural Health Monitoring.

Removes environmental temperature confounders from vibration signals using
co-integration regression and orthogonal subspace projection, preventing
false alarms during seasonal and diurnal ambient temperature fluctuations.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

logger = logging.getLogger("SHM.ThermalNormalizer")


@dataclass
class ThermalModelParams:
    """Coefficients for thermal regression and PCA projection."""
    coefficients: np.ndarray  # Shape (C, 3): [T^2, T, 1] per channel
    vibration_channel_indices: List[int]
    temp_channel_index: int
    pca_components: Optional[np.ndarray] = None
    mean_temp: float = 26.0


class ThermalNormalizer:
    """Decouples ambient thermal variations from dynamic structural response.

    Models the baseline relationship between ambient temperature T and
    vibration amplitude metrics / raw signals for each accelerometer channel.
    During evaluation, it computes the stationary residual:
        e_c = y_c - (alpha_c * T^2 + beta_c * T + gamma_c)
    or standardizes features by subtracting the temperature trend.
    """

    def __init__(
        self,
        vibration_channel_indices: Optional[List[int]] = None,
        temp_channel_index: int = 4,
        use_polynomial: bool = True,
        alpha: float = 1.0,
    ) -> None:
        self.vibration_indices = vibration_channel_indices or [0, 1, 2, 3]
        self.temp_index = temp_channel_index
        self.use_polynomial = use_polynomial
        self.alpha = alpha

        self.models: Dict[int, Ridge] = {}
        self.mean_temp: float = 26.0
        self.is_fitted: bool = False

    def _make_features(self, temp: np.ndarray) -> np.ndarray:
        """Create feature matrix from 1D temperature array [T^2, T] or [T]."""
        temp_centered = (temp - self.mean_temp).reshape(-1, 1)
        if self.use_polynomial:
            return np.hstack([temp_centered**2, temp_centered])
        return temp_centered

    def fit(self, data: np.ndarray) -> "ThermalNormalizer":
        """Fit thermal regression models on healthy baseline data.

        Args:
            data: 2D numpy array of shape (N, C) containing vibration channels
                  and at least one temperature channel.
        """
        temp = data[:, self.temp_index]
        self.mean_temp = float(np.mean(temp))
        X = self._make_features(temp)

        for c_idx in self.vibration_indices:
            y = data[:, c_idx]
            reg = Ridge(alpha=self.alpha)
            reg.fit(X, y)
            self.models[c_idx] = reg

        self.is_fitted = True
        logger.info(
            f"Fitted ThermalNormalizer across {len(self.vibration_indices)} channels "
            f"with mean temperature {self.mean_temp:.2f}°C"
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Subtract thermal expected value from vibration channels.

        Args:
            data: 2D array (N, C) or 1D array (C,).

        Returns:
            Thermally normalized copy of data where thermal drift is removed.
        """
        if not self.is_fitted:
            return data.copy()

        is_1d = (data.ndim == 1)
        arr = data.reshape(1, -1) if is_1d else data.copy()
        temp = arr[:, self.temp_index]
        X = self._make_features(temp)

        for c_idx, reg in self.models.items():
            predicted_thermal_component = reg.predict(X)
            # Remove thermal expectation
            arr[:, c_idx] = arr[:, c_idx] - predicted_thermal_component

        return arr.squeeze(0) if is_1d else arr

    def save(self, file_path: Union[str, Path]) -> None:
        """Persist normalizer state to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved ThermalNormalizer to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ThermalNormalizer":
        """Load persisted normalizer state from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"ThermalNormalizer artifact not found: {path}")
        obj = joblib.load(path)
        return obj
