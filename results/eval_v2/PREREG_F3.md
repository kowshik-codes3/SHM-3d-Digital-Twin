# Pre-Registration: Sub-Stage F3 Minimum Detectable Shift Benchmark (E5)

**Date**: 2026-11-05  
**Status**: PRE-REGISTERED  
**Commit Target**: Sub-Stage F3 Protocol Freeze  

---

## 1. Objective and Physical Motivation
Evaluation findings on KW51 established that the retrofit caused a downward shift of approximately $-0.0316\text{ Hz}$ to $-0.0684\text{ Hz}$ ($\sim 0.77\%$) on the bridge's fundamental vertical bending mode ($\approx 4.06\text{ Hz}$), without increasing ambient vibration energy.
Waveform autoencoders operating on $1.0\text{ s}$ windows (50 samples at 50 Hz) have an intrinsic Fourier bin resolution of:
$$\Delta f_{\text{Fourier}} = \frac{1}{T_{\text{win}}} = \frac{1}{1.0\text{ s}} = 1.0\text{ Hz}$$
Because the true physical shift ($0.0316\text{ Hz}$) is $31.6\times$ smaller than a single Fourier bin, waveform autoencoders cannot resolve this shift without severe domain-specific inductive bias.

This benchmark rigorously determines the **breakdown threshold** ($\delta_{\min}$) where detectors transition from noise floor to reliable detection ($\text{Recall} \ge 0.80$).

---

## 2. Experimental Protocol
1. **Baseline Data**:
   - Clean, healthy validation recordings (`healthy-val`) and healthy test recordings (`healthy-test-cold`).
2. **Synthetic Resampling Proxy**:
   - Signals are resampled along the time axis by a scaling factor of $(1 + \delta)$ using Fourier-method band-limited resampling (`scipy.signal.resample`):
     $$t' = t \cdot (1 + \delta)$$
   - This uniformly dilates all frequency components: $f' = f / (1 + \delta) \approx f \cdot (1 - \delta)$.
   - Perturbation sweep $\delta \in \{0.1, 0.25, 0.5, 0.77, 1.0, 2.0, 3.0, 5.0, 10.0\}\%$.
   - **Proxy Disclosure**: Resampling shifts all modal peaks equally. It does not alter mode shapes or simulate local structural stiffness loss.
3. **Detectors Evaluated**:
   - **B4**: Plain Waveform LSTM-AE (1.0 s window, 50 Hz, hidden dim 32).
   - **B6**: Full Stage 2 System (LSTM-AE + temperature conditioning + strided EVT thresholding + persistence).
   - **B2**: Modal Regression Comparator ($f(T)$ linear regression + Mahalanobis distance on tracked modes).
   *(Note: S1 Spectral Tracker will be added in Sub-Stage F4)*.
4. **Calibration & Target Operating Point**:
   - All thresholds calibrated strictly on unperturbed `healthy-val` to a fixed target FAR of $1.0\%$.
5. **Decision Rule & Metrics**:
   - Metric: Recall at target FAR ($1.0\%$) across all recordings for each $\delta$.
   - **Primary Decision Threshold**: Minimum detectable shift $\delta_{80\%}$, defined as the smallest $\delta$ where:
     $$\text{Recall}(\delta) \ge 0.80$$
   - **Theoretical Reference Line**: Minimum window length $T_{\text{win}} = 1 / \Delta f_1$ required to resolve $\Delta f_1 = f_1 \cdot \delta$ by Fourier resolution.
