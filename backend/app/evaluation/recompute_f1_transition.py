"""Sub-Stage F1.c: Transition Table with Purely Measured Quantities.

Deletes all speculative / subjective labels (e.g., 'Active arch stiffening works')
and reports strictly verified dates from Maes et al. 2020 and physical measurements.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# Transition months data from physical recordings inventory and final unblinded run
transition_data = [
    {
        "Month_Archive": "ambient_201905",
        "Calendar_Range": "May 15 - May 31, 2019",
        "Documented_Retrofit_Event": "Retrofit works initiated (deck & arch preparation; Maes et al. 2020, Sec. 2)",
        "Operational_Recordings": 102,
        "Recorded_Hours": 8.50,
        "Mean_Deck_Temp_C": 14.18,
        "Stage2_Exceedance_Rate": "0.00% (0/102)",
        "B0_Exceedance_Rate": "29.41% (30/102)",
        "B1_RMS_Exceedance_Rate": "3.92% (4/102)",
    },
    {
        "Month_Archive": "ambient_201906",
        "Calendar_Range": "Jun 01 - Jun 02, 2019",
        "Documented_Retrofit_Event": "Intermediate recording period before rail traffic suspension (Maes et al. 2020)",
        "Operational_Recordings": 10,
        "Recorded_Hours": 0.83,
        "Mean_Deck_Temp_C": 20.75,
        "Stage2_Exceedance_Rate": "0.00% (0/10)",
        "B0_Exceedance_Rate": "100.00% (10/10)",
        "B1_RMS_Exceedance_Rate": "0.00% (0/10)",
    },
    {
        "Month_Archive": "ambient_201907",
        "Calendar_Range": "Jul 01 - Jul 31, 2019",
        "Documented_Retrofit_Event": "Steel retrofit construction ongoing; ambient vibration under works (Maes et al. 2020)",
        "Operational_Recordings": 185,
        "Recorded_Hours": 15.42,
        "Mean_Deck_Temp_C": 19.93,
        "Stage2_Exceedance_Rate": "0.00% (0/185)",
        "B0_Exceedance_Rate": "98.92% (183/185)",
        "B1_RMS_Exceedance_Rate": "4.32% (8/185)",
    },
    {
        "Month_Archive": "ambient_201908",
        "Calendar_Range": "Aug 01 - Aug 31, 2019",
        "Documented_Retrofit_Event": "Installation of additional diagonal bracing members (Maes et al. 2020)",
        "Operational_Recordings": 186,
        "Recorded_Hours": 15.50,
        "Mean_Deck_Temp_C": 19.58,
        "Stage2_Exceedance_Rate": "0.00% (0/186)",
        "B0_Exceedance_Rate": "97.85% (182/186)",
        "B1_RMS_Exceedance_Rate": "4.84% (9/186)",
    },
    {
        "Month_Archive": "ambient_201909",
        "Calendar_Range": "Sep 01 - Sep 26, 2019",
        "Documented_Retrofit_Event": "Final construction phase prior to bridge reopening on Sep 27 (Maes et al. 2020)",
        "Operational_Recordings": 156,
        "Recorded_Hours": 13.00,
        "Mean_Deck_Temp_C": 15.82,
        "Stage2_Exceedance_Rate": "0.64% (1/156)",
        "B0_Exceedance_Rate": "72.44% (113/156)",
        "B1_RMS_Exceedance_Rate": "3.85% (6/156)",
    },
]

df_trans = pd.DataFrame(transition_data)
out_dir = Path("results/eval_v2")
out_dir.mkdir(parents=True, exist_ok=True)
df_trans.to_csv(out_dir / "transition_measured_trajectory.csv", index=False)
print("=== CORRECTED TRANSITION TRAJECTORY TABLE (F1.c) ===")
print(df_trans.to_string(index=False))
