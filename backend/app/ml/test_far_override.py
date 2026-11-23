import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.dataset import SHMDataModule
from app.ml.feature_extractor import WindowFeatureExtractor
from app.ml.baseline_models import BaselineStructuralDetector
from app.ml.evaluator import BaselineEvaluator

# Custom scaler that handles Temperature scaling differently
class CustomStructuralDataScaler:
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> 'CustomStructuralDataScaler':
        self.scaler.fit(df_train[self.feature_names].values)
        if "Sensor_Ambient_Temp" in self.feature_names:
            idx = self.feature_names.index("Sensor_Ambient_Temp")
            self.scaler.mean_[idx] = 26.0
            self.scaler.var_[idx] = 64.0  # 8.0^2
            self.scaler.scale_[idx] = 8.0
        self.is_fitted = True
        return self

    def transform(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            raw = data[self.feature_names].values
        else:
            raw = data
        return self.scaler.transform(raw).astype(np.float32)

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        return self.fit(df_train).transform(df_train)

def test_override():
    backend_dir = Path(__file__).resolve().parent.parent.parent
    train_parquet = backend_dir / "data" / "baseline_healthy.parquet"
    test_parquet = backend_dir / "data" / "test_damaged.parquet"
    
    channels = [
        "Sensor_Abutment_Left", "Sensor_Pier_Left", 
        "Sensor_MidSpan_Girder", "Sensor_Pier_Right", "Sensor_Ambient_Temp"
    ]
    
    # Manually setup to use Custom Scale
    df_healthy = pd.read_parquet(train_parquet)
    df_damaged = pd.read_parquet(test_parquet)
    
    split_idx = int(len(df_healthy) * 0.75)
    df_train = df_healthy.iloc[:split_idx]
    df_val = df_healthy.iloc[split_idx:]
    
    cust_scaler = CustomStructuralDataScaler(channels)
    
    train_scaled = cust_scaler.fit_transform(df_train)
    val_scaled = cust_scaler.transform(df_val)
    test_scaled = cust_scaler.transform(df_damaged)
    
    from app.ml.sliding_window import generate_sliding_windows
    train_windows = generate_sliding_windows(train_scaled, 50, 5)
    val_windows = generate_sliding_windows(val_scaled, 50, 1)
    test_windows = generate_sliding_windows(test_scaled, 50, 1)
    
    X_train_feats = WindowFeatureExtractor.extract_features(train_windows)
    X_val_feats   = WindowFeatureExtractor.extract_features(val_windows)
    X_test_feats  = WindowFeatureExtractor.extract_features(test_windows)
    
    # We will exclude temperature features in the feature set or keep them?
    # Let's test with all 40 features first, but with updated temperature scaler!
    detector = BaselineStructuralDetector(pca_components=2, contamination=0.02)
    detector.fit(X_train_feats)
    
    val_res = detector.score(X_val_feats)
    far = np.mean(val_res["iso_preds"]) * 100
    print(f"CASE 3 (40 Features, Custom Scale): Validation FAR = {far:.2f}%")
    
    damage_start_idx = int(len(X_test_feats) * 0.8)
    y_test_true = np.zeros(len(X_test_feats), dtype=int)
    y_test_true[damage_start_idx:] = 1
    
    test_res = detector.score(X_test_feats)
    metrics = BaselineEvaluator.evaluate_performance(y_test_true, test_res["iso_scores"])
    print(f"CASE 3 test ROC-AUC = {metrics['roc_auc']:.4f}")
    
    # What if we exclude temperature channels in Feature Extractor?
    # i.e., we extract features from all channels, but only train on the vibration channels (index 0-3)
    X_train_feats_sub = X_train_feats[:, :32]
    X_val_feats_sub   = X_val_feats[:, :32]
    X_test_feats_sub  = X_test_feats[:, :32]
    
    detector_sub = BaselineStructuralDetector(pca_components=2, contamination=0.02)
    detector_sub.fit(X_train_feats_sub)
    
    val_res_sub = detector_sub.score(X_val_feats_sub)
    far_sub = np.mean(val_res_sub["iso_preds"]) * 100
    print(f"CASE 4 (32 Features, Temp Excluded from Classifier): Validation FAR = {far_sub:.2f}%")
    
    test_res_sub = detector_sub.score(X_test_feats_sub)
    metrics_sub = BaselineEvaluator.evaluate_performance(y_test_true, test_res_sub["iso_scores"])
    print(f"CASE 4 test ROC-AUC = {metrics_sub['roc_auc']:.4f}")

if __name__ == "__main__":
    test_override()
