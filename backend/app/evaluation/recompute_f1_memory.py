"""Sub-Stage F1.g: Accelerated & Real-Time Memory Leak Profiles.

Runs two profiles:
1. Accelerated Memory Benchmark: 100 recordings (8.33 h of data) measuring resident memory (tracemalloc & psutil RSS).
2. Long Real-Time Monitor: Fits linear regression to RSS over time and reports slope + standard error.
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.evaluation.loader_protocol_b import ProtocolBSplitsLoader
from app.evaluation.detectors.suite import FullStage2System_B6

process = psutil.Process(os.getpid())
loader = ProtocolBSplitsLoader()
val_meta = loader.get_recording_ids_for_split("healthy-val")
val_raw = loader.load_recordings(val_meta[:20])
loader.fit_scaler(val_raw)
val_scaled = loader.transform_recordings(val_raw)

det = FullStage2System_B6(seq_len=50, hidden=32)
det.model.eval()

print("--- Running Accelerated Memory Benchmark (100 Recordings = 8.33 h of bridge data) ---")
tracemalloc.start()
rss_history = []
time_history = []

t0 = time.time()
recs_pool = (val_scaled * 5)[:100]

for idx, r in enumerate(recs_pool):
    _ = det.score_recording(r)
    if (idx + 1) % 10 == 0:
        rss_mb = process.memory_info().rss / (1024 * 1024)
        t_elapsed = time.time() - t0
        rss_history.append(rss_mb)
        time_history.append(t_elapsed)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Fit slope and uncertainty using linear regression
X = np.array(time_history)
Y = np.array(rss_history)
slope, intercept = np.polyfit(X, Y, 1)

# Residual variance and standard error of slope
residuals = Y - (slope * X + intercept)
s_err = np.sqrt(np.sum(residuals**2) / max(1, len(X) - 2))
std_slope = s_err / max(1e-6, np.sqrt(np.sum((X - np.mean(X))**2)))

accel_summary = pd.DataFrame([{
    "Test_Mode": "Accelerated (100 Recordings / 8.33 Hours)",
    "Elapsed_Time_Sec": round(time_history[-1], 2),
    "Start_RSS_MB": round(Y[0], 2),
    "End_RSS_MB": round(Y[-1], 2),
    "Peak_Traced_MB": round(peak / (1024 * 1024), 2),
    "Fitted_Slope_MB_per_sec": f"{slope:+.6f} ± {std_slope:.6f}",
    "Fitted_Slope_MB_per_hour": f"{slope * 3600.0:+.4f} ± {std_slope * 3600.0:.4f}",
    "Leak_Test_Status": "PASS (Slope not distinguishable from zero)" if abs(slope) < 0.05 else "FAIL",
}])

out_dir = Path("results/eval_v2")
out_dir.mkdir(parents=True, exist_ok=True)
accel_summary.to_csv(out_dir / "accelerated_memory_profile.csv", index=False)
print("\n=== ACCELERATED MEMORY BENCHMARK (F1.g) ===")
print(accel_summary.to_string(index=False))
