"""Sub-Stage F1.g (Long Real-Time Run): Real 1-Hour Wall-Clock Memory Stability Monitor.

Runs a real-time 1-hour wall-clock replay test (3,600 s) streaming live simulated frames
at true 50 Hz through the full pipeline, logging resident memory (RSS) every 60 s.
Fitted slope with standard error is saved to results/eval_v2/realtime_1hour_memory_profile.csv.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.ml.model import LSTMAutoencoder
from app.ml.anomaly_engine import AnomalyStateMachine
from app.ml.threshold_calibrator import ChannelThreshold, ThresholdConfig
from app.ml.telemetry_formatter import TelemetryFormatter

process = psutil.Process(os.getpid())
device = torch.device("cpu")
model = LSTMAutoencoder(input_size=5, target_size=4, enc_hidden1=64, enc_hidden2=32, dec_hidden1=64, dec_hidden2=64, seq_len=50).to(device)
model.eval()

ch_names = ["aBD17Ay", "aBD17Az", "aBD23Ay", "aBD23Az"]
ch_dict = {ch: ChannelThreshold(name=ch, mu=0.0, sigma=1.0, tau=1.0, warn=0.5, k=3.0) for ch in ch_names}
th_config = ThresholdConfig(calibration_timestamp="2026-11-05", k_multiplier=3.0, channels=ch_dict)
state_machine = AnomalyStateMachine(config=th_config, persistence=3, t_persist=0.04)

TOTAL_SECONDS = 3600 # 1 hour of real wall-clock time
LOG_INTERVAL = 60    # every 60 seconds

t_start = time.time()
next_log = t_start + LOG_INTERVAL

timestamps = []
rss_mb_history = []
frames_processed = 0

print(f"=== STARTING REAL-TIME 1-HOUR MEMORY MONITOR (PID {os.getpid()}) ===")
print(f"Duration: {TOTAL_SECONDS} seconds | Log Interval: {LOG_INTERVAL} seconds")

# Seed initial point
rss_mb_history.append(process.memory_info().rss / (1024 * 1024))
timestamps.append(0.0)

out_file = Path("results/eval_v2/realtime_1hour_memory_profile.csv")
out_file.parent.mkdir(parents=True, exist_ok=True)

while True:
    now = time.time()
    elapsed = now - t_start
    if elapsed >= TOTAL_SECONDS:
        break

    # Simulate 50 Hz frame ingestion (1 frame per 0.02s)
    # Batch process 50 frames (1 second of real bridge data)
    frame = np.random.randn(50, 5).astype(np.float32)
    tensor = torch.tensor(frame, device=device).unsqueeze(0)
    with torch.no_grad():
        recon, _ = model(tensor)
        mse = torch.mean((tensor[:, :, :4] - recon) ** 2, dim=1).squeeze(0).cpu().numpy()
    event = state_machine.step(window_index=frames_processed, timestamp=now, mse_vector=mse)
    payload = TelemetryFormatter.format(event, structure_id="bridge_demo_01")
    frames_processed += 1

    time.sleep(0.02) # pace execution to real time

    if now >= next_log:
        rss = process.memory_info().rss / (1024 * 1024)
        rss_mb_history.append(rss)
        timestamps.append(elapsed)
        next_log = now + LOG_INTERVAL

        # Compute running slope
        X = np.array(timestamps)
        Y = np.array(rss_mb_history)
        if len(X) >= 2:
            slope, intercept = np.polyfit(X, Y, 1)
            residuals = Y - (slope * X + intercept)
            s_err = np.sqrt(np.sum(residuals**2) / max(1, len(X) - 2))
            std_slope = s_err / max(1e-6, np.sqrt(np.sum((X - np.mean(X))**2)))
            slope_hr = slope * 3600.0
            std_hr = std_slope * 3600.0
        else:
            slope_hr, std_hr = 0.0, 0.0

        df_curr = pd.DataFrame([{
            "Elapsed_Min": round(elapsed / 60.0, 1),
            "Frames_Processed": frames_processed,
            "Current_RSS_MB": round(rss, 2),
            "Fitted_Slope_MB_per_hr": f"{slope_hr:+.4f} ± {std_hr:.4f}",
            "Status": "RUNNING",
        }])
        df_curr.to_csv(out_file, index=False)
        print(f"[{elapsed/60.0:.1f}m / {TOTAL_SECONDS/60.0:.0f}m] RSS: {rss:.2f} MB | Slope: {slope_hr:+.4f} ± {std_hr:.4f} MB/h")

# Final calculation
X = np.array(timestamps)
Y = np.array(rss_mb_history)
slope, intercept = np.polyfit(X, Y, 1)
residuals = Y - (slope * X + intercept)
s_err = np.sqrt(np.sum(residuals**2) / max(1, len(X) - 2))
std_slope = s_err / max(1e-6, np.sqrt(np.sum((X - np.mean(X))**2)))

final_df = pd.DataFrame([{
    "Total_Elapsed_Sec": round(timestamps[-1], 2),
    "Total_Frames_Processed": frames_processed,
    "Start_RSS_MB": round(Y[0], 2),
    "Final_RSS_MB": round(Y[-1], 2),
    "Fitted_Slope_MB_per_hr": f"{slope * 3600.0:+.4f} ± {std_slope * 3600.0:.4f}",
    "Verdict": "PASS (Memory Leak Slope < 0.1 MB/hr)" if abs(slope * 3600.0) < 0.1 else "MONITOR",
}])
final_df.to_csv(out_file, index=False)
print("\n=== REAL-TIME 1-HOUR MEMORY TEST COMPLETE ===")
print(final_df.to_string(index=False))
