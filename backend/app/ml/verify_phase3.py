"""Defensive verification pipeline running classical ML benchmarks on structural features."""

import os
import sys
from pathlib import Path
import numpy as np

# Add backend directory to path to allow direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.dataset import SHMDataModule
from app.ml.feature_extractor import WindowFeatureExtractor
from app.ml.baseline_models import BaselineStructuralDetector
from app.ml.evaluator import BaselineEvaluator


def run_phase3_verification() -> None:
    """Combines dataset components, extracts feature representations and tests performance."""
    print("=" * 60)
    print("STARTING PHASE 3 VERIFICATION FOR SHM OFFLINE BASELINES")
    print("=" * 60)
    
    # Resolve directory paths dynamically
    current_dir = Path(__file__).resolve().parent  # backend/app/ml/
    backend_dir = current_dir.parent.parent        # backend/
    
    train_parquet = backend_dir / "data" / "baseline_healthy.parquet"
    test_parquet = backend_dir / "data" / "test_damaged.parquet"
    artifacts_dir = backend_dir / "artifacts"
    
    # Workspace root reports directory
    workspace_dir = backend_dir.parent
    reports_dir = workspace_dir / "reports" / "phase3"
    
    print(f"Data directory: {backend_dir / 'data'}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Reports directory: {reports_dir}")
    print("-" * 60)

    channels = [
        "Sensor_Abutment_Left", 
        "Sensor_Pier_Left", 
        "Sensor_MidSpan_Girder", 
        "Sensor_Pier_Right", 
        "Sensor_Ambient_Temp"
    ]
    
    # 1. Ingest DataLoaders and Tensors from Phase 2
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
    
    # 2. Extract Tabular Features from Sliding Windows
    print("\n--- [1] Extracting Statistical & Spectral Features ---")
    X_train_feats = WindowFeatureExtractor.extract_features(dm.train_windows)
    X_val_feats   = WindowFeatureExtractor.extract_features(dm.val_windows)
    X_test_feats  = WindowFeatureExtractor.extract_features(dm.test_windows)
    print(f"Train Feature Matrix : {X_train_feats.shape} (40 engineered features)")
    print(f"Val Feature Matrix   : {X_val_feats.shape}")
    print(f"Test Feature Matrix  : {X_test_feats.shape}")

    # 3. Fit Baseline Unsupervised Models
    print("\n--- [2] Training Isolation Forest & PCA Models ---")
    detector = BaselineStructuralDetector(pca_components=2, contamination=0.02)
    detector.fit(X_train_feats)
    detector.save(str(artifacts_dir))
    print(f"Calibrated Decision Threshold (μ + 3σ): {detector.threshold:.4f}")

    # 4. Create Ground Truth Mask on Test Set for Benchmarking
    # In Phase 1, damage was injected in the final 20% of the timeline
    damage_start_idx = int(len(X_test_feats) * 0.8)
    y_test_true = np.zeros(len(X_test_feats), dtype=int)
    y_test_true[damage_start_idx:] = 1

    # 5. Evaluate Performance & Generate Reports
    print("\n--- [3] Evaluating Anomaly Detection Metrics ---")
    val_results = detector.score(X_val_feats)
    test_results = detector.score(X_test_feats)
    
    # False Alarm Rate on Validation Set
    far = np.mean(val_results["iso_preds"]) * 100
    metrics = BaselineEvaluator.evaluate_performance(y_test_true, test_results["iso_scores"])
    
    print(f"Validation False Alarm Rate (FAR): {far:.2f}% (Target: ≤ 5.0%)")
    print(f"Test Set ROC-AUC Score          : {metrics['roc_auc']:.4f} (Target: > 0.95)")
    print(f"Test Set PR-AUC Score           : {metrics['pr_auc']:.4f}")

    # Metrics assertions enforcing baseline acceptability criteria
    assert metrics['roc_auc'] >= 0.95, f"ROC-AUC below threshold: {metrics['roc_auc']:.4f}"
    assert far <= 5.0, f"False Alarm Rate too high: {far:.2f}%"
    print(" [PASS] Isolation Forest detection metrics and FAR requirements verified.")

    # 6. Generate Diagnostic Plots
    print("\n--- [4] Generating Visual Diagnostic Plots ---")
    reports_dir.mkdir(parents=True, exist_ok=True)
    BaselineEvaluator.generate_report_plots(
        pca_latents=test_results["pca_latents"],
        y_true=y_test_true,
        scores=test_results["iso_scores"],
        threshold=detector.threshold,
        output_dir=str(reports_dir)
    )
    print(f"  Plots saved in: {reports_dir}")
    print(" [PASS] Visual reporting logs saved output files.")
    
    print("=" * 60)
    print("PHASE 3 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_phase3_verification()
    except AssertionError as e:
        print(f"\n[FAIL] Assertion check failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
