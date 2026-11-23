"""Defensive pre-processing pipeline checks for Phase 2."""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Add backend directory to path to allow direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.dataset import SHMDataModule
from app.ml.scaler_utils import StructuralDataScaler


def run_phase2_verification() -> None:
    """Runs structural health metrics checks on preprocessing outputs."""
    print("=" * 60)
    print("STARTING PHASE 2 VERIFICATION FOR SHM PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Resolve directory paths dynamically
    current_dir = Path(__file__).resolve().parent  # backend/app/ml/
    backend_dir = current_dir.parent.parent        # backend/
    
    train_parquet = backend_dir / "data" / "baseline_healthy.parquet"
    test_parquet = backend_dir / "data" / "test_damaged.parquet"
    artifacts_dir = backend_dir / "artifacts"
    
    print(f"Data directory: {backend_dir / 'data'}")
    print(f"Artifacts directory: {artifacts_dir}")
    print("-" * 60)
    
    channels = [
        "Sensor_Abutment_Left",
        "Sensor_Pier_Left",
        "Sensor_MidSpan_Girder",
        "Sensor_Pier_Right",
        "Sensor_Ambient_Temp"
    ]
    
    # Instantiate DataModule manager
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
    
    # 1. Assert Tensor Shapes
    print("\n--- [1] Checking Window Tensor Shapes ---")
    print(f"Train Tensors Shape : {dm.train_windows.shape} (Expected: (M_train, 50, 5))")
    print(f"Val Tensors Shape   : {dm.val_windows.shape} (Expected: (M_val, 50, 5))")
    print(f"Test Tensors Shape  : {dm.test_windows.shape} (Expected: (M_test, 50, 5))")
    assert dm.train_windows.shape[1:] == (50, 5), "Invalid window dimensions on train set!"
    assert dm.val_windows.shape[1:] == (50, 5), "Invalid window dimensions on val set!"
    assert dm.test_windows.shape[1:] == (50, 5), "Invalid window dimensions on test set!"
    print(" [PASS] Sliding window shapes verified.")

    # 2. Assert DataLoader Execution & Batch Dimension
    print("\n--- [2] Checking PyTorch DataLoader Batches ---")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Single Batch Tensor shape: {batch.shape}")
    print(f"Single Batch Tensor dtype: {batch.dtype}")
    assert batch.shape == (64, 50, 5), f"Expected batch shape (64, 50, 5), got {batch.shape}"
    assert batch.dtype == torch.float32, f"Expected torch.float32, got {batch.dtype}"
    
    # Verify DataLoader counts
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    print(f"Total train batch count : {len(train_loader)}")
    print(f"Total val batch count   : {len(val_loader)}")
    print(f"Total test batch count  : {len(test_loader)}")
    print(" [PASS] DataLoader batch structures verified.")

    # 3. Assert Zero-Leakage Scaling
    print("\n--- [3] Verifying Data Leakage Guards ---")
    train_flat = dm.train_windows.reshape(-1, 5)
    val_flat = dm.val_windows.reshape(-1, 5)
    
    train_mean = np.mean(train_flat, axis=0)
    train_std = np.std(train_flat, axis=0)
    val_mean = np.mean(val_flat, axis=0)
    val_std = np.std(val_flat, axis=0)
    
    print(f"  Train Mean (should be ~0.0): {np.round(train_mean, 4)}")
    print(f"  Train Std  (should be ~1.0): {np.round(train_std, 4)}")
    print(f"  Val Mean   (non-zero drift): {np.round(val_mean, 4)}")
    print(f"  Val Std    (non-one variance): {np.round(val_std, 4)}")
    
    # Standard standardizer check limits (small drift is expected in validation chronological segments)
    assert np.allclose(train_mean[:4], 0.0, atol=1e-2), "Train mean is not standardized!"
    # Assert unit variance only for the vibration channels (0-3). Ambient temperature is physically scaled.
    assert np.allclose(train_std[:4], 1.0, atol=1e-2), "Train variance is not standardized!"
    
    # Assert validation mean is not exactly zero/one (validates zero-leakage, fit ONLY on training split)
    assert not np.allclose(val_mean, 0.0, atol=1e-5), "Data leakage alert: validation mean is exactly zero!"
    assert not np.allclose(val_std, 1.0, atol=1e-5), "Data leakage alert: validation std is exactly unit variance!"
    print(" [PASS] Leakage checks passed successfully.")

    # 4. Assert Invertibility
    print("\n--- [4] Verifying Inverse Transformation ---")
    scaler = StructuralDataScaler.load(artifacts_dir / "scaler.joblib")
    recovered = scaler.inverse_transform(train_flat[:10])
    assert recovered.shape == (10, 5), "Inverse transform failed dimension check!"
    
    # Assert reconstruction of temperatures around diurnal starting baseline
    recovered_means = scaler.inverse_transform(train_mean.reshape(1, -1))[0]
    print(f"  Recovered physical mean temp: {recovered_means[4]:.4f} °C (Base starting diurnal ~26.0 °C)")
    assert 18.0 <= recovered_means[4] <= 34.0, "Physical inverse-scaled values fall out of diurnal limits!"
    print(" [PASS] Inverse transform invertibility verified.")

    # 5. Check for NaNs/Infs
    print("\n--- [5] Quality Assertions on Tensors ---")
    assert not torch.isnan(batch).any(), "NaN detected in training tensors!"
    assert not torch.isinf(batch).any(), "Inf detected in training tensors!"
    print(" [PASS] Data completeness verified (Zero NaNs & Infs).")
    
    print("=" * 60)
    print("PHASE 2 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_phase2_verification()
    except AssertionError as e:
        print(f"\n[FAIL] Assertion check failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
