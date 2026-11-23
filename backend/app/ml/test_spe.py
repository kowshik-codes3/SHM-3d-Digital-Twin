import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.dataset import SHMDataModule
from app.ml.feature_extractor import WindowFeatureExtractor
from app.ml.baseline_models import BaselineStructuralDetector
from app.ml.evaluator import BaselineEvaluator

def test_spe():
    backend_dir = Path(__file__).resolve().parent.parent.parent
    train_parquet = backend_dir / "data" / "baseline_healthy.parquet"
    test_parquet = backend_dir / "data" / "test_damaged.parquet"
    
    channels = [
        "Sensor_Abutment_Left", "Sensor_Pier_Left", 
        "Sensor_MidSpan_Girder", "Sensor_Pier_Right", "Sensor_Ambient_Temp"
    ]
    
    dm = SHMDataModule(
        train_parquet=str(train_parquet),
        test_parquet=str(test_parquet),
        channels=channels,
        window_size=50,
        train_stride=5,
        eval_stride=1,
        batch_size=64
    )
    dm.setup()
    
    X_train_feats = WindowFeatureExtractor.extract_features(dm.train_windows)
    X_val_feats   = WindowFeatureExtractor.extract_features(dm.val_windows)
    X_test_feats  = WindowFeatureExtractor.extract_features(dm.test_windows)
    
    # Exclude temp
    X_train_32 = X_train_feats[:, :32]
    X_val_32 = X_val_feats[:, :32]
    X_test_32 = X_test_feats[:, :32]
    
    # Fit detector
    detector = BaselineStructuralDetector(pca_components=2, contamination=0.02)
    detector.fit(X_train_32)
    
    val_res = detector.score(X_val_32)
    test_res = detector.score(X_test_32)
    
    # Ground truth
    damage_start_idx = int(len(X_test_feats) * 0.8)
    y_test_true = np.zeros(len(X_test_feats), dtype=int)
    y_test_true[damage_start_idx:] = 1
    
    # Compute SPE ROC-AUC
    spe_metrics = BaselineEvaluator.evaluate_performance(y_test_true, test_res["spe"])
    print(f"Vibration-only PCA SPE ROC-AUC = {spe_metrics['roc_auc']:.4f}")
    
    # Compare with Isolation Forest
    iso_metrics = BaselineEvaluator.evaluate_performance(y_test_true, test_res["iso_scores"])
    print(f"Vibration-only IsoForest ROC-AUC = {iso_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    test_spe()
