"""End-to-end Phase 5 verification: threshold calibration, state machine simulation,
spatial localization accuracy, and diagnostic plot generation."""

import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.ml.anomaly_engine import (
    AnomalyStateMachine,
    STATE_CRITICAL,
    plot_localization_heatmap,
)
from app.ml.dataset import SHMDataModule
from app.ml.model import LSTMAutoencoder
from app.ml.telemetry_formatter import TelemetryFormatter
from app.ml.threshold_calibrator import ThresholdCalibrator


# ── Known damage channels (injected in Phase 1 signal generator) ──────────────
# Scenario A: Stiffness loss → affects midspan girder most
# Scenario B: Pier settlement → affects pier channels
# The test Parquet merges both; either may dominate per-channel residual.
EXPECTED_DEFECT_CHANNELS = {"Sensor_MidSpan_Girder", "Sensor_Pier_Left", "Sensor_Pier_Right"}

CHANNEL_NAMES: List[str] = [
    "Sensor_Abutment_Left",
    "Sensor_Pier_Left",
    "Sensor_MidSpan_Girder",
    "Sensor_Pier_Right",
    "Sensor_Ambient_Temp",
]


def _channel_mse_matrix(model, loader, device) -> np.ndarray:
    """Return (M, C) per-window per-channel MSE matrix."""
    model.eval()
    buffers = []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            x_hat, _ = model(x)
            mse_bc = ((x - x_hat) ** 2).mean(dim=1).cpu().numpy()  # (B, C)
            buffers.append(mse_bc)
    return np.concatenate(buffers, axis=0)


def run_phase5_verification() -> None:
    print("=" * 64)
    print("STARTING PHASE 5 VERIFICATION – THRESHOLDING & ANOMALY ENGINE")
    print("=" * 64)

    backend_dir   = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = backend_dir / "artifacts"
    workspace_dir = backend_dir.parent
    reports_dir   = workspace_dir / "reports" / "phase5"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. DataLoaders ────────────────────────────────────────────────────
    print("\n--- [1] Building DataLoaders ---")
    dm = SHMDataModule(
        train_parquet=str(backend_dir / "data" / "baseline_healthy.parquet"),
        test_parquet=str(backend_dir  / "data" / "test_damaged.parquet"),
        channels=CHANNEL_NAMES,
        window_size=50, train_stride=5, eval_stride=1,
        batch_size=64, artifacts_dir=str(artifacts_dir),
    )
    dm.setup()
    print(f"  Val batches : {len(dm.val_dataloader())} | Test batches: {len(dm.test_dataloader())}")

    # ── 2. Load Phase 4 checkpoint ────────────────────────────────────────
    print("\n--- [2] Loading LSTM Autoencoder Checkpoint ---")
    device = torch.device("cpu")
    model = LSTMAutoencoder(
        input_size=5, enc_hidden1=64, enc_hidden2=32,
        dec_hidden1=64, dec_hidden2=64, dropout=0.1, seq_len=50,
    )
    ckpt = artifacts_dir / "lstm_autoencoder_best.pt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded checkpoint: {ckpt}")

    # ── 3. Calibrate thresholds on healthy val data ───────────────────────
    print("\n--- [3] Calibrating Per-Channel UCL Thresholds (k=3) ---")
    calibrator = ThresholdCalibrator(model, device, k=3.0, channel_names=CHANNEL_NAMES)
    config = calibrator.calibrate(dm.val_dataloader())

    print("  Channel-wise thresholds (μ | σ | τ):")
    for name, ch in config.channels.items():
        print(f"    {name:<30} μ={ch.mu:.6f}  σ={ch.sigma:.6f}  τ={ch.tau:.6f}")

    saved_json = calibrator.save(str(artifacts_dir))
    print(f"  Threshold config saved: {saved_json}")

    # Assertion: all τ_c > 0
    for name, ch in config.channels.items():
        assert ch.tau > 0, f"τ_c <= 0 for channel {name}"
    print(" [PASS] All τ_c > 0 verified.")

    # Threshold distribution plot
    calibrator.plot_distributions(str(reports_dir))

    # ── 4. Stream val data through state machine (FAR check) ─────────────
    print("\n--- [4] False Alarm Rate on Healthy Val Stream ---")
    sm = AnomalyStateMachine(config, persistence=3)
    val_mse_matrix = _channel_mse_matrix(model, dm.val_dataloader(), device)
    critical_val_count = 0

    for idx, mse_vec in enumerate(val_mse_matrix):
        event = sm.step(idx, time.time(), mse_vec)
        if event.system_status == STATE_CRITICAL:
            critical_val_count += 1

    far = critical_val_count / len(val_mse_matrix) * 100
    print(f"  Critical alerts on val (healthy): {critical_val_count} / {len(val_mse_matrix)} → FAR={far:.2f}%")
    assert far <= 5.0, f"FAR too high on healthy stream: {far:.2f}%"
    print(f" [PASS] FAR {far:.2f}% ≤ 5%")

    # ── 5. Stream test data and verify localization ───────────────────────
    print("\n--- [5] Streaming Test Data & Assessing Defect Localization ---")
    sm.reset()
    test_mse_matrix = _channel_mse_matrix(model, dm.test_dataloader(), device)

    # Only assess windows in the known damaged segment (last 20% of test)
    damage_start = int(len(test_mse_matrix) * 0.8)
    damaged_mse  = test_mse_matrix[damage_start:]

    critical_events = []
    for idx, mse_vec in enumerate(damaged_mse, start=damage_start):
        event = sm.step(idx, time.time(), mse_vec)
        if event.system_status == STATE_CRITICAL:
            critical_events.append(event)

    print(f"  Total CRITICAL events in damaged segment: {len(critical_events)}")
    assert len(critical_events) > 0, "No CRITICAL events detected in the damaged test segment!"

    # Localization accuracy
    localization_counts = Counter(
        e.localized_defect_node for e in critical_events if e.localized_defect_node
    )
    assert len(localization_counts) > 0, (
        "All CRITICAL events had localized_defect_node=None — state machine localization broken."
    )
    top_node, top_count = localization_counts.most_common(1)[0]
    accuracy = top_count / len(critical_events) * 100

    print(f"  Localization breakdown: {dict(localization_counts)}")
    print(f"  Top localized node: '{top_node}' → {accuracy:.1f}% of CRITICAL events")

    assert top_node in EXPECTED_DEFECT_CHANNELS, (
        f"Top localized node '{top_node}' not in expected defect channels {EXPECTED_DEFECT_CHANNELS}"
    )
    assert accuracy >= 80.0, (
        f"Localization accuracy {accuracy:.1f}% below 80% threshold"
    )
    print(f" [PASS] Spatial localization '{top_node}' confirmed with {accuracy:.1f}% accuracy.")

    # ── 6. Example telemetry payloads ────────────────────────────────────
    print("\n--- [6] Sample Telemetry Payload ---")
    sample_event = critical_events[len(critical_events) // 2]
    payload_str = TelemetryFormatter.to_json(sample_event, indent=2)
    print(payload_str[:600])

    # ── 7. Localization heatmap (full test stream) ────────────────────────
    print("\n--- [7] Saving Localization Heatmap ---")
    tau_values = np.array([config.channels[n].tau for n in CHANNEL_NAMES])
    plot_localization_heatmap(
        mse_matrix=test_mse_matrix,
        channel_names=CHANNEL_NAMES,
        tau_values=tau_values,
        output_dir=str(reports_dir),
    )

    print("=" * 64)
    print("PHASE 5 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 64)


if __name__ == "__main__":
    try:
        run_phase5_verification()
    except AssertionError as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
