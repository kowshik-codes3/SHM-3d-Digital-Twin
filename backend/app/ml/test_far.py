import os
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.dataset import SHMDataModule
from app.ml.feature_extractor import WindowFeatureExtractor
from app.ml.baseline_models import BaselineStructuralDetector
from app.ml.evaluator import BaselineEvaluator

def test_far():
    current_dir = Path(__file__).resolve().parent
    backend_dir = current_dir.parent.parent.parent / "backend" # d:\Desktop\Other\Projects\General Project\Interactive 3D Digital Twin Integration\backend
    
    train_parquet = backend_dir / "data" / "baseline_healthy.parquet"
    test_parquet = backend_dir / "data" / "test_damaged.parquet"
    artifacts_dir = backend_dir / "artifacts"
    
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
        batch_size=64,
        artifacts_dir=str(artifacts_dir)
    )
    dm.setup()
    
    X_train_feats = WindowFeatureExtractor.extract_features(dm.train_windows)
    X_val_feats   = WindowFeatureExtractor.extract_features(dm.val_windows)
    X_test_feats  = WindowFeatureExtractor.extract_features(dm.test_windows)
    
    # Run test 1: All 40 features
    detector_40 = BaselineStructuralDetector(pca_components=2, contamination=0.02)
    detector_40.fit(X_train_feats)
    val_res_40 = detector_40.score(X_val_feats)
    far_40 = np.mean(val_res_40["iso_preds"]) * 100
    print(f"CASE 1 (40 Features): Validation FAR = {far_40:.2f}%")
    
    # Run test 2: Exclude temperature features (columns 32-39)
    X_train_32 = X_train_feats[:, :32]
    X_val_32   = X_val_feats[:, :32]
    X_test_32  = X_test_feats[:, :32]
    
    detector_32 = BaselineStructuralDetector(pca_components=2, contamination=0.02)
    detector_32.fit(X_train_32)
    val_res_32 = detector_32.score(X_val_32)
    far_32 = np.mean(val_res_32["iso_preds"]) * 100
    print(f"CASE 2 (32 Features, Temp Excluded): Validation FAR = {far_32:.2f}%")
    
    damage_start_idx = int(len(X_test_feats) * 0.8)
    y_test_true = np.zeros(len(X_test_feats), dtype=int)
    y_test_true[damage_start_idx:] = 1
    
    test_res_32 = detector_32.score(X_test_32)
    metrics_32 = BaselineEvaluator.evaluate_performance(y_test_true, test_res_32["iso_scores"])
    print(f"CASE 2 test ROC-AUC = {metrics_32['roc_auc']:.4f}")

if __name__ == "__main__":
    test_far()
