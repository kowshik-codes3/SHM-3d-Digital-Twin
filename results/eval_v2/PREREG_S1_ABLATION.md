# Pre-Registration: Systematic Ablation of S1 Spectral Tracker (Sub-Stage F5)

**Date**: 2026-11-05  
**Status**: PRE-REGISTERED  
**Commit Target**: Sub-Stage F5 Ablation Protocol Freeze  

---

## 1. Objective
To systematically evaluate the empirical necessity of each architectural component of the S1 Spectral Tracker detector under the strict decision rules of `PREREG.md`.

---

## 2. Pre-Registered Keep/Drop Decision Rules
1. **Primary Keep Rule (Power)**:
   - A component is kept on detection power grounds **if and only if** the paired block-bootstrap 95% confidence interval on $\Delta \text{Recall}$ lies strictly above zero:
     $$\text{CI}_{95\%}[\Delta \text{Recall}] > 0$$
2. **Secondary Keep Rule (False-Alarm Reduction / Stability)**:
   - Where the recall difference is inconclusive or within the noise floor ($\text{CI}_{95\%}$ spans zero):
     - The component is marked **"Inconclusive on KW51"**.
     - It may be **KEPT** if and only if it demonstrates a statistically significant reduction in Healthy Test FAR (especially under sub-zero winter conditions) or guarantees numerical stability/physical invariance.
     - Otherwise, the component is **DROPPED** in favor of simplicity.

---

## 3. Evaluated Ablation Axes
1. **Temperature Compensation Form**:
   - `None`: Raw uncompensated natural frequencies.
   - `Linear`: Standard linear regression $f(T) = \beta_0 + \beta_1 T$.
   - `Piecewise-Freezing`: Linear fit with sub-zero freezing hinge $f(T) = \beta_0 + \beta_1 T + \beta_2 \max(0, -T)$.
   - `Polynomial_2`: Quadratic polynomial $f(T) = \beta_0 + \beta_1 T + \beta_2 T^2$.
2. **Analysis Window Length**:
   - $5\text{ min}$ (1 single 50 Hz recording, 15,000 samples).
   - $10\text{ min}$ (2 rolling recordings, 30,000 samples).
   - $30\text{ min}$ (6 rolling recordings, 90,000 samples).
3. **Persistence Filter**:
   - $K = 1$: Instantaneous trigger (no persistence).
   - $K = 2$: 2 consecutive over-threshold windows required.
   - $K = 3$: 3 consecutive over-threshold windows required.
4. **Decision Threshold Estimator**:
   - `Empirical Percentile`: Calibrated directly to 99th percentile on validation.
   - `Strided EVT`: Extreme Value Theory generalized Pareto fit on strided tail residuals.
5. **Slow-Drift Layer**:
   - `Instantaneous`: Raw single-step Mahalanobis distance.
   - `EWMA`: Exponentially weighted moving average ($\text{span} = 20$).
   - `CUSUM`: Cumulative sum filter ($k = 0.5\sigma, h = 5\sigma$).
6. **Sensor-Fault Gate**:
   - `Enabled`: Rejects flatlines, spikes, missing peaks to `SENSOR_FAULT`.
   - `Disabled`: Ingestion pipeline routes sensor corruptions directly to detector.
