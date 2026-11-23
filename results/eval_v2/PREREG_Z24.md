# Pre-Registration: Clean Confirmation on Z24 Bridge Benchmark (Sub-Stage F6)

**Date**: 2026-11-05  
**Status**: PRE-REGISTERED  
**Commit Target**: Sub-Stage F6 External Benchmark Freeze  

---

## 1. Benchmark Dataset: Z24 Bridge
- **Structure**: Pre-stressed concrete box-girder highway bridge (span 30 m + 14 m + 14 m) in Solothurn, Switzerland.
- **Reference**: SIMCES Project, KU Leuven / EMPA (Peeters & De Roeck 2001; Reynders, Wormsen & De Roeck 2014).
- **Physical Validation Protocol**:
  - **Healthy Baseline (Scenario 0)**: Undamaged structural operation under ambient temperature cycles.
  - **Damage Scenarios (Scenarios 1 – 16)**: Documented Progressive Damage Tests (PDT) including:
    - Lowering of pier (settlement of 20 mm, 40 mm, 80 mm, 95 mm)
    - Tilt of foundation
    - Concrete spalling at soffit
    - Rupture of 2, 4, and 6 pre-stressing tendons
    - Anchor head damage

---

## 2. Experimental Protocol
1. **Splits**:
   - `healthy-train`: 75% of Scenario 0 undamaged records ($n = 27$).
   - `healthy-val`: 25% of Scenario 0 undamaged records ($n = 9$).
   - `damage-test`: All 16 progressive damage scenarios (Scenarios 1 to 16, $n = 576$).
2. **Evaluated Pipeline**:
   - The frozen detector pipeline (`SpectralTracker_S1` / Mahalanobis multivariate novelty index with regularized covariance and persistence) applied **untouched** with zero parameter re-tuning.
3. **Threshold & Calibration**:
   - Operating threshold $\tau$ calibrated strictly on `healthy-val` to target FAR ($\alpha = 1.0\%$).
4. **Primary Metrics**:
   - **Healthy-Val FAR**: Calibration target $\le 2.0\%$.
   - **Damage Detection Recall**: Overall recall across all 16 damage scenarios.
   - **ROC-AUC**: Separation between undamaged Scenario 0 and progressive damage Scenarios 1–16.
