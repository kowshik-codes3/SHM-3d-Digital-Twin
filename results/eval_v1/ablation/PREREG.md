# Pre-Registration Plan for Model Ablation (KW51 Evaluation)

## 1. Context & Purpose
This pre-registration document defines the keep-rules, decision boundaries, primary metrics, and guard metrics for ablating the Stage 2 ML improvements on the KW51 Railway Bridge benchmark **prior** to unblinding the final evaluation.

## 2. Evaluation Partition Rules
- **Tuning and Keep Decisions**: Fitted strictly on `healthy-val` and `changed-val` (October 2019 matched season).
- **Final Test Set Lock**: `changed-test` (November 2019 to January 2020) and `healthy-test-cold` (January 2019) remain completely unopened during ablation tuning.
- Decisions are permanently frozen in `results/eval/FROZEN.md` before `changed-test` is accessed.

## 3. Metrics & Keep-Rules

### Primary Metric
- **Recording-Level Recall** on `changed-val` at the calibrated target FAR ($p=0.001$).

### Guard Metrics
1. **Recording-Level False Alarm Rate (FAR)** across healthy validation and healthy seasonal test splits.
2. **Alert Episodes per Recorded Hour** on healthy validation data.
3. **Season Control Gap**: $\Delta \text{AUC} = \text{AUC}_{\text{method}} - \text{AUC}_{\text{B0}}$.

### Pre-Registered Keep Rule
A component is kept in the production configuration **if and only if**:
1. The paired block-bootstrap 95% confidence interval of its primary metric gain excludes zero ($\Delta \text{Recall} > 0$), **AND**
2. No guard metric degrades beyond a tolerance of $+0.5\text{ episodes/hour}$ or $+2.0\%\text{ FAR}$.

If a component does not meet both criteria on validation, it is declared as not demonstrably helpful on KW51 and marked for removal.

## 4. Components Under Test
- **(a) Temperature Conditioning**: Conditioning on `tVL` without reconstruction.
- **(b) Time-Based Persistence**: $P=5, t_{\text{persist}}=5.0\text{s}$, pre-critical warning buffer.
- **(c) Warning Threshold**: Empirical 99th percentile vs 2-sigma SPC.
- **(d) EVT Decorrelation**: Block-maxima tail fitting vs raw overlapping windows.
- **(e) Top-3 Localization**: Evaluated as descriptive ranking (no ground truth available for scoring).
- **(f) Sensor Fault Gate**: Evaluated via synthetic fault injection sweeps on healthy test data.
- **(g) Slow-Drift Layer**: Hourly EWMA/CUSUM residual change detector.
- **(h) Continuous Severity Score**: Descriptive composite index (0–100).
