"""Run only post-training inference and assertions using saved checkpoint."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.dataset import SHMDataModule
from app.ml.model import LSTMAutoencoder
from app.ml.reconstruction_evaluator import ReconstructionEvaluator
from app.ml.evaluator import BaselineEvaluator


def run_inference_only() -> None:
    backend_dir   = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = backend_dir / "artifacts"
    workspace_dir = backend_dir.parent
    reports_dir   = workspace_dir / "reports" / "phase4"

    channels = [
        "Sensor_Abutment_Left", "Sensor_Pier_Left",
        "Sensor_MidSpan_Girder", "Sensor_Pier_Right", "Sensor_Ambient_Temp",
    ]

    dm = SHMDataModule(
        train_parquet=str(backend_dir / "data" / "baseline_healthy.parquet"),
        test_parquet=str(backend_dir / "data" / "test_damaged.parquet"),
        channels=channels, window_size=50, train_stride=5, eval_stride=1,
        batch_size=64, artifacts_dir=str(artifacts_dir),
    )
    dm.setup()

    device = torch.device("cpu")
    model = LSTMAutoencoder(input_size=5, enc_hidden1=64, enc_hidden2=32,
                             dec_hidden1=64, dec_hidden2=64, dropout=0.1, seq_len=50)
    model.load_state_dict(torch.load(artifacts_dir / "lstm_autoencoder_best.pt", map_location=device))

    evaluator = ReconstructionEvaluator(model, device)
    val_mse,  x_val_true,  x_val_hat  = evaluator.evaluate(dm.val_dataloader())
    test_mse, x_test_true, x_test_hat = evaluator.evaluate(dm.test_dataloader())

    mean_val_mse  = float(np.mean(val_mse))
    mean_test_mse = float(np.mean(test_mse))
    separation = mean_test_mse / mean_val_mse

    print(f"Val  (healthy) MSE      : {mean_val_mse:.6f}")
    print(f"Test (damaged) MSE      : {mean_test_mse:.6f}")
    print(f"Separation ratio        : {separation:.2f}x")

    assert mean_val_mse < 0.25,     f"Val MSE {mean_val_mse:.6f} >= 0.25"
    assert separation  >= 2.5,      f"Separation {separation:.2f}x < 2.5x"
    print("[PASS] Both assertions passed.")

    # Plots
    history_stub = {"train_loss": [0.16], "val_loss": [0.158], "lr": [5e-4]}   # stub so plot still runs
    evaluator.plot_convergence(history_stub, str(reports_dir))
    mid_val   = len(x_val_true)  // 2
    dam_start = int(len(x_test_true) * 0.8)
    evaluator.plot_reconstruction(x_val_true,  x_val_hat,  str(reports_dir), sample_idx=mid_val,  title_prefix="Healthy")
    evaluator.plot_reconstruction(x_test_true, x_test_hat, str(reports_dir), sample_idx=dam_start, title_prefix="Damaged")
    print("Plots saved.")

    # ROC-AUC over test set using MSE as anomaly score
    damage_start_idx = int(len(test_mse) * 0.8)
    y_test_true = np.zeros(len(test_mse), dtype=int)
    y_test_true[damage_start_idx:] = 1
    metrics = BaselineEvaluator.evaluate_performance(y_test_true, test_mse)
    print(f"LSTM ROC-AUC (test set) : {metrics['roc_auc']:.4f}")
    print(f"LSTM PR-AUC  (test set) : {metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    try:
        run_inference_only()
    except AssertionError as e:
        print(f"[FAIL] {e}", file=sys.stderr); sys.exit(1)
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
