# Pre-Registration: Spectral-Tracking Detector "S1" (Sub-Stage F4)

**Date**: 2026-11-05  
**Status**: PRE-REGISTERED  
**Commit Target**: Sub-Stage F4 Protocol and Design Freeze  

---

## 1. Goal & Architecture
The goal of **S1 (Streaming Spectral Tracking Detector)** is to achieve reliable, physics-grounded detection of small structural stiffness changes (such as the KW51 retrofit frequency shift of $-0.0316\text{ Hz}$ to $-0.0684\text{ Hz}$ on Mode 1) directly from streaming raw acceleration signals without requiring pre-computed dataset modal tables.

### S1 Pipeline:
1. **Streaming Ring Buffer & Spectral Estimator**:
   - Rolling buffer of duration $T_{\text{win}}$ (tested at 5 min, 10 min, and 30 min; default 10 min = 30,000 samples at 50 Hz).
   - Welch Power Spectral Density (PSD) with Hann window, 50% overlap, $N_{\text{perseg}} = 2048$ samples ($\Delta f_{\text{base}} = 0.0244\text{ Hz}$).
   - Sub-bin parabolic peak interpolation to achieve continuous frequency resolution down to $\sim 0.001\text{ Hz}$:
     $$k^* = k_{\max} + \frac{P[k_{\max}-1] - P[k_{\max}+1]}{2(P[k_{\max}-1] - 2P[k_{\max}] + P[k_{\max}+1])}$$
     $$\hat{f} = k^* \cdot \frac{f_s}{N_{\text{fft}}}$$
   - Search bands configured around physical bridge modes:
     - Mode 1: $[0.48, 0.55]\text{ Hz}$ (Vertical channels `aBD17Az`, `aBD23Az`)
     - Mode 2: $[1.18, 1.28]\text{ Hz}$ (Lateral channels `aBD17Ay`, `aBD23Ay`)
     - Mode 3: $[1.80, 2.00]\text{ Hz}$ (Vertical channels)
     - Mode 4: $[2.35, 2.52]\text{ Hz}$ (Vertical channels)
   - When buffer contains fewer samples than $T_{\text{win}}$ (e.g. startup), estimate is marked unavailable (`STATUS_WARMING_UP`).

2. **Temperature Compensation Layer**:
   - Fits robust linear regression $f_m(T) = \beta_{0,m} + \beta_{1,m} T$ strictly on `healthy-train` data.
   - Also evaluates a hinge/piecewise linear fit at $0^\circ\text{C}$ on validation to test mitigation of sub-zero ice stiffening drift identified in F2.
   - Outputs compensated frequency residual vector:
     $$\mathbf{r} = [r_1, r_2, r_3, r_4]^T, \quad r_m = f_m - \hat{f}_m(T)$$

3. **Multivariate Novelty Metric**:
   - Mahalanobis distance on compensated residuals:
     $$D^2 = \mathbf{r}^T \boldsymbol{\Sigma}_{\text{train}}^{-1} \mathbf{r}$$
     where $\boldsymbol{\Sigma}_{\text{train}}$ is the empirical covariance of healthy training residuals.

4. **Thresholds & Persistence**:
   - Calibrated strictly on `healthy-val` to target FAR ($\alpha = 1.0\%$).
   - EVT threshold $\tau_{\text{EVT}}$ fitted on strided tail residuals (stride = 5).
   - Time-based persistence filter: condition must persist for $K = 2$ consecutive evaluation steps before triggering `CRITICAL`.

5. **Slow-Drift Layer (EWMA/CUSUM)**:
   - Two-sided CUSUM / EWMA tracking slow baseline drift of Mode 1.
   - Detection delay evaluated against documented retrofit milestone: May 15, 2019 (pier reinforcement / retrofit mobilization per Maes et al. 2020).

6. **Sensor-Fault Gate**:
   - Flatline (RMS $< 10^{-5}$), spike ($> 10\sigma$), packet dropout, missing modal peak (prominence $< 3\times$ noise floor), or non-physical step change ($|\Delta f| > 0.4\text{ Hz}$ in 1 step) immediately routes to `SENSOR_FAULT`.
   - Never routes sensor failures to `CRITICAL`.

7. **Waveform Autoencoder Separation**:
   - Waveform autoencoders (B4/B6) remain dedicated exclusively to high-frequency transients, impacts, and sensor anomalies. They are explicitly NOT claimed to detect ambient modal frequency shifts.

---

## 2. Pre-Registered Hypotheses & Success Criteria
1. **Raw Estimator Agreement**:
   - S1 sub-bin peak frequencies estimated from raw waveforms must agree with `trackedmodes.mat` with Pearson $R^2 \ge 0.85$ and Mean Absolute Error $\text{MAE} \le 0.015\text{ Hz}$ across the healthy validation set.
2. **Detection Power**:
   - Changed-val Recall $\ge 90.0\%$ at target FAR $1.0\%$.
3. **Placebo Test Verification**:
   - Placebo Test 1 (Healthy Dec 2018 vs Healthy May 2019): ROC-AUC must remain within $[0.40, 0.60]$.
   - Placebo Test 2 (1,000 label permutations): Null distribution median ROC-AUC within $[0.48, 0.52]$.
4. **Development-Exposed Disclosure**:
   - Any score computed on KW51 `changed-test` will be explicitly reported as **"development-exposed"** per Rule 3.
