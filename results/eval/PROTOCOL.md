# Evaluation Protocol (KW51 Railway Bridge Benchmark)

## 1. Overview & Dataset Reference
- **Structure**: KW51 Railway Bridge, Leuven, Belgium. Steel tied-arch railway bridge carrying rail tracks A and B over the Leuven-Dijle canal.
- **Reference**: Maes, K., & Lombaert, G. (2020). *Continuous monitoring of the KW51 railway bridge before, during, and after retrofitting*. Zenodo. [DOI: 10.5281/zenodo.3745914](https://doi.org/10.5281/zenodo.3745914).
- **Licence**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **Selected Acceleration Channels**:
  - `aBD11Az`: Deck left quarter-span, Track A, vertical direction ($Z$).
  - `aBD17Ay`: Deck midspan, Track A, transverse direction ($Y$).
  - `aBD17Az`: Deck midspan, Track A, vertical bending direction ($Z$).
  - `aBD23Az`: Deck right quarter-span, Track A, vertical direction ($Z$).
- **Temperature Condition Input**:
  - `tVL`: Thermocouple ambient temperature at the bridge site (°C).
  - *Strict Rule*: `tVL` is used exclusively as a **condition input** to normalizers/models, **never** as a channel to reconstruct or localize.

---

## 2. State Labels (per RECORDING)
Recordings are assigned a discrete operational health state based on physical construction milestones (verified against Maes & Lombaert 2020):
1. **HEALTHY Baseline State**: Recording start timestamp **before 2019-05-15 00:00:00**.
   - Undamaged baseline structural condition during regular railway service.
2. **TRANSITION State (Retrofit in Progress)**: **2019-05-15 00:00:00 to 2019-09-27 23:59:59**.
   - Active construction period (bottom chord strengthening, deck plate welding, bearing release and replacement).
   - *Rule*: TRANSITION recordings are reported separately per month and are **never** pooled into the clean changed-state test set.
3. **CHANGED State (Structural Change / Retrofit Completed)**: Recording start timestamp **after 2019-09-27 00:00:00**.
   - Permanent post-retrofit structural state with documented ~0.8% to 1.2% downward shifts in resonant frequencies across matching temperatures.
   - *Terminology Rule*: The post-retrofit state is named **"structural change (retrofit)"**, never "damage".
- **Boundary Straddling**: Any recording that spans across a milestone boundary date is excluded and logged.

---

## 3. Window Labels & Recording Boundaries
- **Window Label Inheritance**: A sliding window inherits the state label of its parent recording (`HEALTHY`, `TRANSITION`, `CHANGED`).
- **No Boundary Crossing**: Sliding windows strictly operate within a single 5-minute recording file. Windows **never** span two distinct recordings.
- **No Internal Guard Band**: Because recordings are discrete 5-minute ambient snapshots and no physical retrofit onset takes place within any single 5-minute file, no intra-recording guard band is required.
- **Recording Boundary Resets**:
  - All circular sliding-window buffers ([`SensorStreamBuffer`](file:///d:/Desktop/Other/Projects/General%20Project/Interactive%203D%20Digital%20Twin%20Integration/backend/app/ml/stream_buffer.py)) are cleared at recording boundaries.
  - All state machine rolling exceedance counters, duration timers, and filter memories ([`AnomalyStateMachine`](file:///d:/Desktop/Other/Projects/General%20Project/Interactive%203D%20Digital%20Twin%20Integration/backend/app/ml/anomaly_engine.py)) are strictly **reset** at every recording boundary.

---

## 4. Alert Episodes & Unit of Evaluation
- **Alert Episode**: A contiguous run of confirmed `CRITICAL` windows inside one recording file.
  - An episode terminates after a quiet period of 5 consecutive non-critical windows, or upon reaching the end of the recording file.
  - `SENSOR_FAULT` windows are isolated hardware failure events and are **not** counted as structural alert episodes.
- **Primary Evaluation Unit**: **Recording-Level**:
  - A recording is **FLAGGED** if it contains at least one confirmed alert episode.
  - **Recording-Level False Alarm Rate (FAR)**: Fraction of `HEALTHY` test recordings flagged ($N_{\text{flagged}} / N_{\text{healthy}}$).
  - **Recording-Level Recall**: Fraction of `CHANGED` test recordings flagged ($N_{\text{flagged}} / N_{\text{changed}}$).
- **Secondary Evaluation Unit**: Window-Level Precision, Recall, F1, Balanced Accuracy, ROC-AUC, and PR-AUC.
- **Recorded Hours Metric**: False alarms per hour are computed strictly as:
  $$\text{FAR}_{\text{hour}} = \frac{\text{Total Alert Episodes}}{\text{Total Recorded Hours}}$$
  where total recorded hours represents the sum of actual durations of recordings in that partition.

---

## 5. Temperature Bins & Matched Controls
- **Equal-Width Bins**: 2.0°C width bins based on the mean `tVL` of each recording (e.g. $[0, 2), [2, 4), \dots, [24, 26)^\circ\text{C}$).
- **Minimum Population Rule**: Bins with fewer than 5 recordings in either state are dropped from the matched comparison.
- **Matched Summary**: For temperature bins containing both `HEALTHY` test and `CHANGED` recordings, compute metrics within each bin and report a sample-weighted summary.
- **Extrapolation Flag**: Any test recording whose mean temperature lies outside the range seen in `healthy-train` is flagged as an **Extrapolation Recording** and evaluated separately.
- **Season Control Classifier (B0)**: A logistic regression model trained exclusively on `tVL` (and time-of-day) to classify recordings as healthy vs changed. If B0 achieves high AUC on a test set, that test set is confirmed to be season-confounded.

---

## 6. Leakage & Access Control
- **Disjoint Partitioning**: Recording IDs belonging to train, val, and test partitions must be strictly disjoint. Any overlap triggers an immediate assertion failure.
- **Training Isolation**: The `StructuralDataScaler`, autoencoder weights, EVT tail parameters, and warning thresholds must be fitted using **train + val** recordings only.
- **Access Logging**: Test sets (`changed-test`, `healthy-test-cold`, etc.) are opened strictly by the final evaluation runner, logging access timestamp, commit hash, and partition IDs.

---

## 7. Statistical Bootstrap
- **Block Bootstrap**: All 95% confidence intervals are computed using block bootstrap over **whole recordings** (not individual windows), preserving serial autocorrelation within each 5-minute file.
- **Fixed Seeds**: 5 fixed seeds (`42, 43, 44, 45, 46`) are evaluated and reported as mean $\pm$ std.
