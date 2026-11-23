# Sub-Stage E3 Baseline Suite Benchmark Report

## Protocol & Experimental Setup
- **Dataset**: KW51 Railway Bridge (KU Leuven, Maes & Lombaert 2020)
- **Protocol**: Protocol B (Strict Causal Ordering)
- **Training Set**: `healthy-train` (2018-10-10 to 2018-12-10, 343 clean 5-minute recordings, 28.58 h)
- **Calibration Set**: `healthy-val` (2018-12-11 to 2018-12-31, 124 clean 5-minute recordings, 10.33 h)
- **Tuning Set**: `changed-val` (2019-09-27 to 2019-10-31, 205 clean 5-minute recordings, 17.08 h)
- **Held-Out Test Sets**: `healthy-test-*` and `changed-test` remain **strictly locked**.
- **Operating Target**: False Alarm Rate (FAR) target $\le 1.0\%$ ($\tau_{0.01}$).

---

## Validation Benchmark Results (B0 through B6)

| Baseline Detector | Calibration Threshold ($\tau$) | Val Empirical FAR (%) | Changed-Val Recall (%) | ROC-AUC | PR-AUC | $\Delta \text{AUC}_{\text{B0}}$ | Fitting Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Temperature Control (Thermometer)** | 0.5720 | 1.61% | 63.41% | 0.9339 | 0.9622 | 0.0000 | 0.00s |
| **B1: RMS + Peak Energy** | 1.5377 | 1.61% | 3.41% | 0.5624 | 0.7054 | -0.3716 | 0.06s |
| **B2: Modal Regression + Mahalanobis (KW51 Published)** | 16.2496 | 1.59% | **100.00%** | **1.0000** | **1.0000** | **+0.0661** | 0.02s |
| **B3: PCA Band Power Reconstruction Error** | 0.0173 | 1.61% | 4.39% | 0.5649 | 0.7004 | -0.3690 | 1.30s |
| **B4: Plain LSTM Autoencoder (Stage 0/1 Baseline)** | 12.4208 | 1.61% | 1.95% | 0.5707 | 0.6988 | -0.3632 | 429.16s |
| **B5: Conditional LSTM-AE (Temperature Conditioned)** | 12.4127 | 1.61% | 1.95% | 0.5708 | 0.6989 | -0.3631 | 422.78s |
| **B6: Full Stage 2 Pipeline (EVT + Persistence)** | 5.9931 | **0.00%** | 0.49% | 0.5648 | 0.6962 | -0.3692 | 423.24s |

---

## Critical Engineering Findings & Scientific Insights

1. **The Physical Retrofit Shift is Purely Frequency-Domain, Not Amplitude-Domain**:
   - The KW51 steel retrofitting (transverse stiffening and arch-tie reinforcement) caused a small, clean **-0.77% downward shift in the primary vertical bending natural frequency** (~4.06 Hz to ~4.03 Hz).
   - The raw ambient vibration energy (RMS) did **not** increase; in fact, ambient RMS was slightly lower in October 2019 (0.00092 m/s²) than October 2018 (0.00142 m/s²) due to train schedule variations.
   - Consequently, amplitude-based autoencoders (B4, B5, B6) and RMS detectors (B1) produce near-chance discrimination ($\text{ROC-AUC} \approx 0.56$–$0.57$) when tasked with raw point-wise waveform reconstruction.

2. **The Modal Baseline (B2) Confirms Why Ku Leuven Tracked Natural Frequencies**:
   - When evaluating natural frequencies with temperature removal ($f(T)$ regression + Mahalanobis novelty distance), **B2 achieves 100.0% recall and 1.0000 ROC-AUC** at 1.59% FAR.
   - This validates the hypothesis pre-registered in O8: operational modal analysis is the gold-standard physical baseline for macroscopic truss retrofits.

3. **Season Control B0 Demonstrates Why Temperature-Aware Splitting is Essential**:
   - The simple thermometer classifier B0 achieved $\text{ROC-AUC} = 0.9339$ between Dec 2018 (mean 4.77°C) and Oct 2019 (mean 13.05°C), proving that naive seasonal comparisons will spuriously classify seasonal warming as bridge changes.
   - B2 is the only baseline that achieves positive $\Delta \text{AUC}_{\text{B0}} = +0.0661$, successfully separating structural retrofit from seasonal thermal drift.
