# SHM Digital Twin — Data-Quality Layer & Sensor Triage (`docs/QUALITY.md`)

`DataQualityLayer` (`backend/app/quality/quality_layer.py`) evaluates every ingested frame **before** samples enter detector buffers so a dead, drifting, or unplugged sensor can never produce a structural `CRITICAL` alarm.

## 1. Safety Principle: Bad Sensors vs. Structural Alerts

- **Structural Alerts (`WARNING`, `CRITICAL`)**: Raised only by structural detectors (`spectral_tracking`, `waveform_ae`) running on verified healthy channels.
- **Sensor Quality Dispositions (`OK`, `DEGRADED`, `FAULT`, `OFFLINE`)**: Tracked per sensor and logged as separate `QualityEvent` records (`/api/v1/structures/{id}/quality/events`).
- **Channel Exclusion**: Any channel marked `FAULT` or `OFFLINE` is stripped from detector inputs (`usable_mask`).
  - If remaining healthy vibration channels `>= min_channels` (and `required_channels` are healthy and temperature is healthy), detectors continue operating seamlessly.
  - If healthy vibration channels fall below `min_channels`, required channels fail, or the temperature sensor is `FAULT`/`OFFLINE`, detectors report `LOW_CONFIDENCE` — **never `CRITICAL`**.
  - When all accelerometers fail or an explicit hardware fault occurs, telemetry reports `SENSOR_FAULT` / `OFFLINE` separately from structural alarms.

## 2. Ten Configurable Quality Checks

All thresholds live in `QualityThresholdsConfig` (`configs/<structure>.yaml`). Baseline statistics (`dc_mean`, `dc_std`, `median`, `mad`, `rms_median`, `rms_p99`, `high_freq_noise_floor`) are calibrated strictly on **healthy training/validation recordings** via `calibrate()` and never tuned on test data.

| # | Check | Method & Fault Criteria | Sensors Applied |
| :- | :--- | :--- | :--- |
| 1 | **Gaps / Dropouts** | Missing sequence numbers (`seq > prev + 1`), timestamp gaps `> max_gap_interpolate_sec`, and zero/NaN fraction (`zero_frac > dropout_zero_fraction` or `finite_frac < 0.5`). | All |
| 2 | **Flatline & Stuck-at-Nonzero** | Block `std <= flatline_min_std` or quantized unique ratio `<= stuck_min_unique_ratio`. | Accelerometers / Strain (**Never Temperature**) |
| 3 | **Saturation / Clipping** | Sample fraction pinned at configured `[range_min, range_max]` rails or flat-top block extrema exceeding `saturation_max_fraction`. | Accelerometers / Strain |
| 4 | **Spikes** | Combined calibrated robust z-score (`> spike_robust_z_threshold`), block excess kurtosis (`> spike_kurtosis_threshold`), and local neighbor-impulse jump check (preserves broadband train passages while catching electrical spikes). | Accelerometers / Strain |
| 5 | **Drift / DC Offset** | Deviation `|mean - calibrated_dc_mean|` and EMA offset exceeding `drift_sigma_multiplier * dc_std` (and calibrated `rms_p99` guard on short frames). | Accelerometers / Strain |
| 6 | **Noise & Gain Change** | High-frequency (>15 Hz) Welch PSD floor ratio (`> noise_floor_ratio`), plus sibling-relative normalized RMS (`gain_ratio > gain_high_ratio` or `< gain_low_ratio`). | Accelerometers / Strain |
| 7 | **Out-of-Order, Duplicate & Late** | Bounded min-heap reorder window (`reorder_window_frames`), seen-key deduplication (`(source_id, seq, t0)`), and hard drop past `late_arrival_tolerance_sec`. | Stream-level |
| 8 | **Clock Offset, Drift, Jump & Sync** | Live wall-vs-frame offset (`clock_offset_max_sec`), cumulative drift rate in ppm (`clock_drift_max_ppm`), step discontinuity (`clock_jump_max_sec`), and optional multi-sensor skew (`clock_sync_skew_max_sec`). Historical replay mode exempts wall-clock offset while retaining backward jump & sync checks. | Stream-level |
| 9 | **Sensor Offline Timeout** | Per-sensor `last_seen_wall` silence exceeding `offline_timeout_sec`, all-NaN frames, or `SOURCE_DOWN` / gateway `offline_sensors` flags. | All |
| 10 | **Temperature Plausibility** | Physical bounds (`temp_range_c`, default `[-40, 70] °C`), rate-of-change (`temp_max_roc_c_per_min`), and packet loss (`finite_frac < 0.5`). | Temperature / Slow Channels |

## 3. Composite Quality Score (`[0, 1]`)

For each sensor block:
$$\text{quality\_score} = \frac{w_c \cdot \text{completeness} + w_t \cdot \text{timing\_score} + w_s \cdot \text{signal\_health}}{w_c + w_t + w_s}$$
Default weights: `weight_completeness = 0.4`, `weight_timing = 0.2`, `weight_signal_health = 0.4`. Any channel fault caps `quality_score <= 0.25` and sets status to `FAULT` (or `0.0` / `OFFLINE`).
