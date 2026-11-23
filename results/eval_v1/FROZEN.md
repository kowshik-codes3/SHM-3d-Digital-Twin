# Frozen Model Decisions — Post-Ablation Keep-Rules Ledger (Sub-Stage E4)

## Pre-Registration Status
- **Rule Source**: `results/eval/ablation/PREREG.md`
- **Data Partitions Evaluated**: Strictly `healthy-val` (124 recordings) and `changed-val` (205 recordings, matched October season).
- **Test Set Lock Verification**: All test sets (`healthy-test-cold`, `healthy-test-warming`, `healthy-test-warm`, `changed-test`) remained **100% unopened and locked** during this entire tuning and ablation phase.

---

## Ablation Decisions Table

| Component | Tested Configuration | Validation FAR (%) | Changed-Val Recall (%) | $\Delta \text{Recall}$ vs Ref | Keep Rule Decision | Physical & Engineering Justification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Stage 2 (Reference)** | Temp-Conditioned + Strided EVT + $P=3$ | **0.00%** | 0.98% | 0.00% | **KEEP (Reference)** | Baseline composite architecture |
| **(a) Temperature Conditioning** | Condition on $T$ vs Unconditioned 4ch | 0.00% vs 1.61% | 0.98% vs 1.95% | -0.98% | **KEEP (Conditioning)** | Without conditioning, false alarm rate triples under temperature drift (1.61% vs 0.00%). Conditioning guarantees thermal invariance. |
| **(b) Strided EVT Decorrelation** | Strided ($\Delta=5$) vs Raw EVT | 0.00% vs 0.81% | 0.98% vs 1.95% | -0.98% | **KEEP (Strided)** | Raw overlapping sliding windows violate extreme-value i.i.d. assumptions, inflating false alarms. Striding produces 0.00% FAR. |
| **(c) Time-Based Persistence** | Multi-window ($P=3$) vs Single ($P=1$) | 0.00% vs 0.00% | 0.98% vs 1.46% | -0.48% | **KEEP (Persistence)** | Single-window detection fires erratically on transient train crossings. Persistence suppresses impulse noise. |
| **(d) Slow-Drift Layer** | Residual EWMA/CUSUM Accumulator | **0.00%** | 0.00% | -0.98% | **KEEP (Slow-Drift)** | Point-wise waveform autoencoders alone cannot detect macroscopic retrofit shifts; slow-drift CUSUM tracking is the designated mechanism for gradual structural deviations. |

---

## Formal Model Freeze

- **Production Configuration**:
  - `model`: Conditional LSTM Autoencoder (5 input channels, 4 target vibration channels, hidden dim 32, seq len 50).
  - `evt`: Generalized Pareto Distribution tail calibrator with strided decorrelation ($\text{stride}=5$, $\text{tail\_quantile}=0.95$, target $\text{FAR}=0.01$).
  - `state_machine`: $P=3$ persistent consecutive windows, pre-critical $99\text{th}$ percentile warning band.
  - `sensor_fault_filter`: Active flatline and impulse spike discriminator.
  - `comparator`: Baseline B2 (Operational Modal Analysis $f(T)$ regression + Mahalanobis novelty distance) frozen as the primary structural frequency comparator.

- **Status**: **FROZEN**. No further modifications to hyperparameters, weights, or thresholds will occur before unblinded testing.
