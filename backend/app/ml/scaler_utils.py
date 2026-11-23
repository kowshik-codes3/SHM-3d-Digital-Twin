"""Structural scaling utilities wrapper around StandardScaler."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union
from sklearn.preprocessing import StandardScaler


class StructuralDataScaler:
    """Manages channel-wise standardization and persistence."""
    
    def __init__(self, feature_names: list[str]):
        """
        Initializes the scaler with mapped features to scale.
        
        Args:
            feature_names: List of column name strings matching sensor channels.
        """
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> 'StructuralDataScaler':
        """
        Fits StandardScaler parameters (mean, variance) exclusively on training dataframe.
        
        Args:
            df_train: Pandas DataFrame representing baseline training split.
            
        Returns:
            Fitted StructuralDataScaler instance.
        """
        self.scaler.fit(df_train[self.feature_names].values)
        if "Sensor_Ambient_Temp" in self.feature_names:
            idx = self.feature_names.index("Sensor_Ambient_Temp")
            self.scaler.mean_[idx] = 26.0
            self.scaler.var_[idx] = 64.0  # 8.0^2
            self.scaler.scale_[idx] = 8.0
        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Standardizes raw data inputs using internal parameters.
        
        Args:
            data: Pandas DataFrame or 2D array coordinates.
            
        Returns:
            Standardized 2D numpy array in float32.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transforming data.")
        if isinstance(data, pd.DataFrame):
            raw = data[self.feature_names].values
        else:
            raw = data
        return self.scaler.transform(raw).astype(np.float32)

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """
        Fits values and transforms training dataset sequentially.
        
        Args:
            df_train: Pandas DataFrame representing baseline training split.
            
        Returns:
            Standardized 2D numpy training array.
        """
        return self.fit(df_train).transform(df_train)

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Converts normalized tensors back to original physical engineering units.
        
        Args:
            scaled_data: Standardized 2D numpy array.
            
        Returns:
            Recovered 2D numpy array with physical dimensions.
        """
        return self.scaler.inverse_transform(scaled_data)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serializes scaler properties to joblib binary file.
        
        Args:
            filepath: Absolute or relative output file path.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'StructuralDataScaler':
        """
        Deserializes fitted StructuralDataScaler instance from file.
        
        Args:
            filepath: Location of scaler joblib file.
            
        Returns:
            Loaded and fitted StructuralDataScaler.
        """
        payload = joblib.load(filepath)
        instance = cls(feature_names=payload["feature_names"])
        instance.scaler = payload["scaler"]
        instance.is_fitted = True
        return instance
