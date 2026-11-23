"""Sub-Stage E3 Baseline Suite Evaluation Runner on Protocol B.

Strict Training & Calibration Rules:
- Fit models strictly on `healthy-train` (2018-10-10 to 2018-12-10).
- Calibrate operating thresholds on `healthy-val` (2018-12-11 to 2018-12-31) to FAR = 1.0%.
- Evaluate performance on:
  1. `healthy-val` (In-distribution false alarm audit)
  2. `changed-val` (Matched-season October 2019 retrofit recall audit)
- Test sets (`healthy-test-*`, `changed-test`) remain STRICTLY LOCKED.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure backend root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.evaluation.loader_protocol_b import ProtocolBSplitsLoader
from app.evaluation.detectors.suite import (
    BaseDetector,
    TemperatureOnlyControl_B0,
    RMS_SpectralPeak_B1,
    ModalRegressionMahalanobis_B2,
    PCABandPower_B3,
    PlainLSTMAutoencoder_B4,
    ConditionalLSTMAutoencoder_B5,
    FullStage2System_B6,
)
from app.evaluation.metrics.substage_e2 import (
    compute_recording_level_metrics,
    compute_season_control_b0,
    block_bootstrap_confidence_interval,
)


def run_e3_baselines_on_validation() -> pd.DataFrame:
    print("=== SUB-STAGE E3: UNIFIED BASELINE SUITE (B0–B6) BENCHMARK ===")
    print("Fitting strictly on healthy-train; calibrating on healthy-val; evaluating on changed-val.")
    
    loader = ProtocolBSplitsLoader()
    
    # 1. Load Train metadata & recordings
    train_meta = loader.get_recording_ids_for_split("healthy-train")
    print(f"Loading {len(train_meta)} healthy-train recordings...")
    t0 = time.time()
    train_raw = loader.load_recordings(train_meta)
    print(f"Loaded train in {time.time() - t0:.1f}s. Fitting scaler...")
    loader.fit_scaler(train_raw)
    train_scaled = loader.transform_recordings(train_raw)
    
    # 2. Load Healthy Validation
    val_meta = loader.get_recording_ids_for_split("healthy-val")
    print(f"Loading {len(val_meta)} healthy-val recordings...")
    val_raw = loader.load_recordings(val_meta)
    val_scaled = loader.transform_recordings(val_raw)
    
    # 3. Load Changed Validation (Oct 2019 - Matched Season)
    cval_meta = loader.get_recording_ids_for_split("changed-val")
    print(f"Loading {len(cval_meta)} changed-val recordings...")
    cval_raw = loader.load_recordings(cval_meta)
    cval_scaled = loader.transform_recordings(cval_raw)

    print(f"\nData loaded: {len(train_scaled)} Train, {len(val_scaled)} Healthy-Val, {len(cval_scaled)} Changed-Val.")

    detectors: List[BaseDetector] = [
        TemperatureOnlyControl_B0(),
        RMS_SpectralPeak_B1(),
        PCABandPower_B3(n_components=4),
        PlainLSTMAutoencoder_B4(seq_len=50, hidden=32),
        ConditionalLSTMAutoencoder_B5(seq_len=50, hidden=32),
        FullStage2System_B6(seq_len=50, hidden=32),
    ]

    results = []

    for det in detectors:
        print(f"\n--- Benchmarking {det.name} ---")
        t_fit = time.time()
        det.fit(train_scaled, train_meta)
        fit_sec = time.time() - t_fit
        
        # Calibrate on healthy validation to target FAR 1% (tau_0.01)
        tau = det.calibrate(val_scaled, target_far=0.01)
        print(f"[{det.name}] Fitted in {fit_sec:.1f}s. Calibrated tau={tau:.4f}")

        # Score healthy-val
        val_scores = np.array([det.score_recording(r) for r in val_scaled])
        # Score changed-val
        cval_scores = np.array([det.score_recording(r) for r in cval_scaled])

        # Evaluate performance
        val_labels = np.zeros(len(val_scores), dtype=int)
        cval_labels = np.ones(len(cval_scores), dtype=int)
        
        y_true = np.concatenate([val_labels, cval_labels])
        y_scores = np.concatenate([val_scores, cval_scores])

        # Metrics
        empirical_far = float(np.mean(val_scores >= tau))
        recall = float(np.mean(cval_scores >= tau))
        
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = 0.5
            
        try:
            pr_auc = float(average_precision_score(y_true, y_scores))
        except Exception:
            pr_auc = 0.0

        print(f"[{det.name}] Val FAR={empirical_far*100:.2f}% | Changed Recall={recall*100:.2f}% | ROC-AUC={auc:.4f} | PR-AUC={pr_auc:.4f}")

        results.append({
            "Detector": det.name,
            "Calibrated_Tau": tau,
            "Val_FAR": empirical_far,
            "Changed_Val_Recall": recall,
            "ROC_AUC": auc,
            "PR_AUC": pr_auc,
            "Fit_Time_Sec": round(fit_sec, 2),
        })

    df_res = pd.DataFrame(results)
    
    # Season control delta AUC
    b0_auc = df_res.loc[df_res["Detector"] == "B0_Temperature_Control", "ROC_AUC"].values[0]
    df_res["Delta_AUC_vs_B0"] = df_res["ROC_AUC"] - b0_auc

    out_csv = Path("results/eval/baselines_validation_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"\n=== BASELINE VALIDATION BENCHMARK COMPLETE ===")
    print(f"Results saved to {out_csv}")
    print(df_res.to_string(index=False))
    return df_res


if __name__ == "__main__":
    run_e3_baselines_on_validation()
