# Audit & Errata: Evaluation v1 Verification Ledger

**Audit Date**: 2026-11-05  
**Auditor**: Senior ML Engineer & Evaluation Specialist  
**Target Snapshot**: `results/eval_v1/`  
**Purpose**: Rigorous item-by-item verification of flaws, omissions, and discrepancies identified in the v1 evaluation run prior to corrective modeling.

---

## 1. Discrepancy Verification Matrix

| # | Issue Identified | Status in v1 Files | Verified File & Location Evidence | Corrective Action Required (Sub-Stages F1–F6) |
|---|:---|:---:|:---|:---|
| **E1** | **LSTM Autoencoders at Chance**<br>B4, B5, B6 achieve ROC-AUC ~0.4744, bootstrap 95% CI [0.4394, 0.5077], Recall 0.91%. Matched Jan 2019 vs Jan 2020 AUC 0.5456. | **CONFIRMED** | `results/eval_v1/final_unblinded_performance_ledger.csv`: `ROC_AUC_Overall = 0.4744`, `ROC_AUC_CI_95_Low = 0.4394`, `ROC_AUC_CI_95_High = 0.5077`, `Changed_Test_Recall = 0.009132`, `Jan_Matched_Pair_AUC = 0.5456`. | Preserve v1 numbers as honest historical baseline; state clearly in F1 that point-wise waveform AEs fail on macroscopic frequency shifts. |
| **E2** | **B2 Incomplete Reporting & Calibration Drift**<br>B2 missing from healthy FAR table; test FAR = 6.74% vs 1% calibration target; Oct 2019 file size discrepancy not audited. | **CONFIRMED** | `results/eval_v1/final_healthy_test_far.csv` omits B2 entirely. In `results/eval_v1/baselines_validation_results.csv`, B2 Val FAR = 1.59%. Execution logs recorded test FAR = 6.74%. In `inventory_50hz.csv`, Oct 2019 has 186 recordings vs 105 in Oct 2018. | Add B2 across all healthy test splits with exact n counts and recorded hours. Perform full setup & sensor audit for Oct 2019 in Sub-Stage F2. |
| **E3** | **Season Confound Verified by B0**<br>B0 (temperature only) achieves ROC-AUC = 0.9339 on `changed-val`. | **CONFIRMED** | `results/eval_v1/baselines_validation_results.csv`: `B0_Temperature_Control` achieves `ROC_AUC = 0.9339` and `PR_AUC = 0.9622` separating Dec 2018 from Oct 2019. | All subsequent evaluation must report $\Delta \text{AUC}_{\text{B0}}$ and seasonal matched controls. Explicitly label unadjusted season comparisons as confounded. |
| **E4** | **Frequency Resolution Mismatch**<br>Retrofit shift is ~0.03–0.07 Hz (~0.8%); a 1 s window cannot resolve this by Fourier analysis ($\Delta f \approx 1\text{ Hz}$). | **CONFIRMED** | By the Rayleigh criterion, $\Delta f \approx 1/T_{\text{window}}$. For 50 samples at 50 Hz ($T = 1.0\text{ s}$), spectral resolution is $1.0\text{ Hz}$. A $0.03\text{ Hz}$ shift is 30x smaller than bin width. | In Sub-Stage F3 & F4, implement multi-minute ring-buffer spectral estimation (5, 10, 30 min) with parabolic sub-bin interpolation to resolve sub-0.05 Hz shifts. |
| **E5** | **Ablation Contradicts PREREG.md**<br>Every component marked `KEEP` despite non-positive $\Delta \text{Recall}$ (variations are 2 vs 4 recordings). | **CONFIRMED** | `results/eval_v1/ablation/ablation.csv`: Ref recall = 0.98% (2 / 205). Unconditioned = 1.95% (4 / 205, $\Delta = -0.98\%$). Raw EVT = 1.95% ($\Delta = -0.98\%$). $P=1$ = 1.46% ($\Delta = -0.48\%$). CUSUM = 0.00% ($\Delta = -0.98\%$). | Strictly recompute keep/drop in F1.a using PREREG.md rule: components at the noise floor are marked "inconclusive on KW51" and kept/dropped on false-alarm criteria only. |
| **E6** | **Flawed Robustness, Faults, Memory & Interpretations**<br>Score at 5 dB SNR (0.2606 -> 0.2961) counted as pass; only 2 fault types; 8.33 h memory test run in 4 s; latency is inference only; transition table interpretations not from data. | **CONFIRMED** | `results/eval_v1/e5_robustness_results.csv`: labeled 5 dB SNR as "graceful degradation" despite score insensitivity. Faults tested only flatline & 50 m/s² spike. `e6_system_benchmarks.csv`: 8.33 h replayed in 3.9 s. Section 6C in `FINAL_EVALUATION_REPORT.md` contained narrative interpretations. | In F1.d–g, re-label SNR as insensitivity, test 7 fault types at 3+ magnitudes, run 1-hour real wall-clock memory test, measure full-path latency, and delete non-data labels. |
| **E7** | **`changed-test` Already Unblinded Once**<br>Access logged; subsequent designs on KW51 are development-exposed. | **CONFIRMED** | `results/eval_v1/TEST_ACCESS_LOG.md`: logged at `2026-11-05T16:16:22`. Primary test set was unblinded. | Rule 3 enforced: design strictly on train, val, and transition. All KW51 `changed-test` results for new detectors will be explicitly flagged as "development-exposed". |

---

## 2. Integrity and Preservation Commitment
- All original files in `results/eval_v1/` are permanently frozen and read-only.
- All corrected tables, revised code, audits, and new spectral models will be written to `results/eval_v2/` and new test harnesses.
- Git commit hash and tag `v1.0-eval-freeze` will seal this baseline.
