"""Sub-Stage F3: Minimum Detectable Frequency Shift Benchmark.

Protocol:
1. Synthetic Proxy: Take clean healthy test recordings and resample time axis by (1 + delta),
   delta in {0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0}%.
   All natural frequencies are shifted uniformly by -delta / (1 + delta).
   (Proxy note: Shifts all modes equally; does not alter mode shapes).
2. Evaluate:
   - Plain Waveform LSTM-AE (B4)
   - Production Stage 2 System (B6)
   - Modal Regression Comparator (B2)
3. Report:
   - Recall at calibrated target FAR (1%) vs delta
   - Smallest delta reaching Recall >= 0.80
   - Theoretical Rayleigh window resolution limit: delta_f = 1 / T_window
4. Plot curves and mark the real KW51 shift (~0.77% / -0.0316 Hz).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.evaluation.loader_protocol_b import ProtocolBSplitsLoader
from app.evaluation.detectors.suite import (
    PlainLSTMAutoencoder_B4,
    FullStage2System_B6,
)
from sklearn.linear_model import LinearRegression

print("=== SUB-STAGE F3: MINIMUM DETECTABLE SHIFT BENCHMARK ===")

# 1. Load data
loader = ProtocolBSplitsLoader()
train_meta = loader.get_recording_ids_for_split("healthy-train")
val_meta = loader.get_recording_ids_for_split("healthy-val")
test_meta = loader.get_recording_ids_for_split("healthy-test-cold")

print(f"Loading recordings: {len(train_meta)} train, {len(val_meta)} val, {len(test_meta)} test...")
train_raw = loader.load_recordings(train_meta)
loader.fit_scaler(train_raw)
train_scaled = loader.transform_recordings(train_raw)

val_raw = loader.load_recordings(val_meta)
val_scaled = loader.transform_recordings(val_raw)

test_raw = loader.load_recordings(test_meta[:30]) # 30 healthy test recordings for sensitivity sweep
test_scaled = loader.transform_recordings(test_raw)

# 2. Fit models on train and calibrate on val (FAR = 1%)
print("Calibrating models...")
m_b4 = PlainLSTMAutoencoder_B4(seq_len=50, hidden=32)
m_b4.fit(train_scaled, train_meta, epochs=6)
tau_b4 = m_b4.calibrate(val_scaled, target_far=0.01)

m_b6 = FullStage2System_B6(seq_len=50, hidden=32)
m_b6.fit(train_scaled, train_meta, epochs=6)
tau_b6 = m_b6.calibrate(val_scaled, target_far=0.01)

# Fit B2 modal comparator on trackedmodes
mat = sio.loadmat("data/kw51/trackedmodes.mat")
modes = mat["modes"]
sdn = modes["sdn"][0, 0].flatten()
f = modes["f"][0, 0]
env = modes["env"][0, 0]

from datetime import datetime, timedelta
def matlab2datetime(d):
    return datetime.fromordinal(int(d)) + timedelta(days=d%1) - timedelta(days=366)

dates = [matlab2datetime(d) for d in sdn]
df_m = pd.DataFrame(f[:, :4], columns=["f1", "f2", "f3", "f4"])
df_m["date"] = dates
df_m["temp"] = env[:, 0]
df_m.dropna(subset=["temp", "f1", "f2", "f3", "f4"], inplace=True)

m_tr = df_m[(df_m["date"] >= "2018-10-10") & (df_m["date"] <= "2018-12-10")]
m_val = df_m[(df_m["date"] >= "2018-12-11") & (df_m["date"] <= "2018-12-31")]

reg = LinearRegression()
reg.fit(m_tr[["temp"]].values, m_tr[["f1", "f2", "f3", "f4"]].values)
res_tr = m_tr[["f1", "f2", "f3", "f4"]].values - reg.predict(m_tr[["temp"]].values)
cov = np.cov(res_tr, rowvar=False)
inv_cov = np.linalg.pinv(cov)

res_val = m_val[["f1", "f2", "f3", "f4"]].values - reg.predict(m_val[["temp"]].values)
d2_val = np.sum(res_val @ inv_cov * res_val, axis=1)
tau_b2 = np.percentile(d2_val, 99.0)

# Deltas to sweep (in %)
deltas_pct = [0.1, 0.25, 0.5, 0.77, 1.0, 2.0, 3.0, 5.0, 10.0]
results = []

print("\n--- Running Resampling Frequency Shift Sweep ---")

for delta in deltas_pct:
    # 1. Evaluate B4 and B6 on resampled vibration recordings
    b4_flagged = 0
    b6_flagged = 0
    total_recs = len(test_scaled)
    
    for r in test_scaled:
        pert = r.copy()
        n_orig = pert.shape[0]
        # Resample time axis
        n_resamp = int(n_orig / (1.0 + delta / 100.0))
        resamp_vib = signal.resample(pert[:, :4], n_resamp, axis=0)
        if len(resamp_vib) < n_orig:
            resamp_vib = np.pad(resamp_vib, ((0, n_orig - len(resamp_vib)), (0, 0)), mode="edge")
        else:
            resamp_vib = resamp_vib[:n_orig]
        pert[:, :4] = resamp_vib
        
        s_b4 = m_b4.score_recording(pert)
        s_b6 = m_b6.score_recording(pert)
        
        if s_b4 >= tau_b4:
            b4_flagged += 1
        if s_b6 >= tau_b6:
            b6_flagged += 1
            
    rec_b4 = b4_flagged / total_recs
    rec_b6 = b6_flagged / total_recs

    # 2. Evaluate B2 on synthetically shifted modal frequencies: f_shifted = f * (1 - delta/100)
    b2_flagged = 0
    m_test_sample = m_val.copy() # use validation points as clean baseline
    shifted_f = m_test_sample[["f1", "f2", "f3", "f4"]].values * (1.0 - delta / 100.0)
    res_b2 = shifted_f - reg.predict(m_test_sample[["temp"]].values)
    d2_b2 = np.sum(res_b2 @ inv_cov * res_b2, axis=1)
    rec_b2 = float(np.mean(d2_b2 >= tau_b2))

    # Theoretical minimum window length needed for resolution: T_win = 1 / delta_f
    # At f0 = 4.06 Hz, delta_f = 4.06 * (delta / 100)
    delta_f_hz = 4.06 * (delta / 100.0)
    theoretical_win_sec = 1.0 / max(1e-4, delta_f_hz)

    results.append({
        "Shift_Delta_Pct": delta,
        "Shift_Delta_Hz (at 4.06 Hz)": round(delta_f_hz, 4),
        "B2_Modal_Recall": round(rec_b2, 4),
        "B4_Plain_AE_Recall": round(rec_b4, 4),
        "B6_Stage2_Recall": round(rec_b6, 4),
        "Theoretical_Window_Sec": round(theoretical_win_sec, 2),
    })
    print(f"Delta = {delta:5.2f}% ({delta_f_hz:+.4f} Hz) | B2: {rec_b2*100:5.1f}% | B4: {rec_b4*100:5.1f}% | B6: {rec_b6*100:5.1f}% | Min Win: {theoretical_win_sec:6.1f}s")

df_res = pd.DataFrame(results)
out_dir = Path("results/eval_v2")
out_dir.mkdir(parents=True, exist_ok=True)
df_res.to_csv(out_dir / "minimum_detectable_shift.csv", index=False)

# 3. Plot Detection Breakdown Curves
plt.figure(figsize=(10, 6))
deltas = df_res["Shift_Delta_Pct"].values

plt.plot(deltas, df_res["B2_Modal_Recall"].values * 100, "o-", color="blue", linewidth=2, label="B2: Modal Regression (f(T) + Mahalanobis)")
plt.plot(deltas, df_res["B6_Stage2_Recall"].values * 100, "s-", color="red", linewidth=2, label="B6: Stage 2 Production (LSTM-AE + EVT)")
plt.plot(deltas, df_res["B4_Plain_AE_Recall"].values * 100, "^--", color="gray", linewidth=1.5, label="B4: Plain Waveform LSTM-AE (1.0s window)")

# Threshold line (80% recall)
plt.axhline(80, color="black", linestyle=":", label="Target Detection Threshold (80% Recall)")

# Mark real KW51 retrofit shift
plt.axvline(0.77, color="darkgreen", linestyle="--", linewidth=2, label="Real KW51 Retrofit Shift (0.77% / -0.0316 Hz)")

plt.xscale("log")
plt.xlabel("Synthetic Frequency Shift δ (%) [Log Scale]", fontsize=11)
plt.ylabel("Recall at Target FAR 1.0% (%)", fontsize=11)
plt.title("Minimum Detectable Frequency Shift Sensitivity Curve", fontsize=12, fontweight="bold")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()

plot_path = out_dir / "minimum_detectable_shift_curve.png"
plt.savefig(plot_path, dpi=150)
print(f"\nSaved sensitivity breakdown curve to {plot_path}")
print(df_res.to_string(index=False))
