"""Verification script for SHM Digital Twin Phase 1.

Generates healthy baseline and damaged scenarios, runs DSP bandpass filtering, 
validates signal assertions, programmatically checks peak modal frequencies 
at 1.5 Hz and 3.8 Hz, and writes Apache Parquet files to disk.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple

# Add the parent directory of backend/app to path to resolve imports when running from backend/ or the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core import SHMConfig, apply_bandpass_filter, compute_psd
from app.simulation import SHMSignalGenerator


def run_checks_and_save() -> None:
    """Executes all Phase 1 checks and persists Parquet datasets."""
    print("=" * 60)
    print("STARTING PHASE 1 VERIFICATION FOR SHM DIGITAL TWIN")
    print("=" * 60)

    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    healthy_parquet_path = os.path.join(data_dir, "baseline_healthy.parquet")
    damaged_parquet_path = os.path.join(data_dir, "test_damaged.parquet")

    # 2. Instantiate system configuration and generator
    config = SHMConfig()
    generator = SHMSignalGenerator(config)

    print(f"Sampling frequency (fs): {config.FS} Hz")
    print(f"Duration: {config.DURATION_SEC} s ({config.num_samples} samples)")
    print("-" * 60)

    # 3. Generate Healthy Baseline
    print("Generating healthy baseline dataset...")
    df_healthy = generator.generate_healthy_baseline()
    
    # 4. Run Integrity Assertions on healthy data
    print("Verifying healthy baseline data integrity...")
    assert isinstance(df_healthy, pd.DataFrame), "Baseline dataset must be a Pandas DataFrame"
    assert df_healthy.shape == (config.num_samples, 5), f"Incorrect shape: {df_healthy.shape}, expected {(config.num_samples, 5)}"
    assert list(df_healthy.columns) == config.channel_names, f"Columns: {list(df_healthy.columns)} do not match config: {config.channel_names}"
    assert df_healthy.isnull().sum().sum() == 0, "Healthy baseline dataset contains NaN values"
    assert np.isinf(df_healthy.to_numpy()).sum() == 0, "Healthy baseline dataset contains Infinite values"
    
    # Verify temporal step delta is matching sampling period
    delta_t = pd.Series(df_healthy.index).diff().dropna()
    mean_delta = delta_t.mean().total_seconds()
    expected_delta = 1.0 / config.FS
    assert abs(mean_delta - expected_delta) < 1e-6, f"Expected sample interval {expected_delta}s, but got average {mean_delta}s"
    print(" [PASS] Healthy baseline integrity checks completed.")
    print("-" * 60)

    # 5. Peak Frequency Verification for Healthy Baseline
    print("Validating modal spectral peaks for MidSpan Girder in healthy state...")
    # Clean vibration signal (bandpass filter removes sensor noise and temp drift offsets)
    sensor_mid = "Sensor_MidSpan_Girder"
    raw_mid_healthy = df_healthy[sensor_mid].to_numpy()
    filtered_mid_healthy = apply_bandpass_filter(
        raw_mid_healthy,
        fs=config.FS,
        lowcut=config.PASSBAND_HZ[0],
        highcut=config.PASSBAND_HZ[1],
        order=config.FILTER_ORDER
    )
    
    # Calculate PSD to find dominant modal peaks
    freqs, psd = compute_psd(filtered_mid_healthy, fs=config.FS, nperseg=2048)
    
    # Identify Mode 1 peak in [1.0, 2.0] Hz
    m1_mask = (freqs >= 1.0) & (freqs <= 2.0)
    m1_peak_freq = freqs[m1_mask][np.argmax(psd[m1_mask])]
    
    # Identify Mode 2 peak in [3.0, 4.5] Hz
    m2_mask = (freqs >= 3.0) & (freqs <= 4.5)
    m2_peak_freq = freqs[m2_mask][np.argmax(psd[m2_mask])]
    
    print(f"  Detected Mode 1 peak frequency: {m1_peak_freq:.3f} Hz (Expected target: ~1.5 Hz)")
    print(f"  Detected Mode 2 peak frequency: {m2_peak_freq:.3f} Hz (Expected target: ~3.8 Hz)")
    
    # Assert peak values are in tolerances (slight deviations allowed due to temperature variations)
    assert abs(m1_peak_freq - 1.5) < 0.15, f"Mode 1 peak {m1_peak_freq:.2f} Hz out of baseline tolerance!"
    assert abs(m2_peak_freq - 3.8) < 0.25, f"Mode 2 peak {m2_peak_freq:.2f} Hz out of baseline tolerance!"
    print(" [PASS] Healthy baseline modal frequencies verified.")
    print("-" * 60)

    # 6. Generate and Verify Scenario A (Stiffness Loss)
    print("Generating and verifying Scenario A (Damage: Stiffness Loss at Mid-Span)...")
    df_damaged_a = generator.generate_damaged_scenario(scenario="A")
    
    # Integrity check
    assert df_damaged_a.shape == (config.num_samples, 5), "Scenario A shape mismatch"
    assert df_damaged_a.isnull().sum().sum() == 0, "Scenario A contains NaNs"
    
    # Extract post-damage state (final 20% of timeline, t >= 480s to 600s, index 24000 to 30000)
    raw_mid_damaged_a = df_damaged_a[sensor_mid].to_numpy()
    filtered_mid_damaged_a = apply_bandpass_filter(
        raw_mid_damaged_a,
        fs=config.FS,
        lowcut=config.PASSBAND_HZ[0],
        highcut=config.PASSBAND_HZ[1],
        order=config.FILTER_ORDER
    )
    
    # Extract last 120s of simulation (6000 samples)
    post_damage_segment = filtered_mid_damaged_a[-6000:]
    freqs_a, psd_a = compute_psd(post_damage_segment, fs=config.FS, nperseg=1024)
    
    # Peak verification during post-damage
    m1_mask_a = (freqs_a >= 0.8) & (freqs_a <= 1.3)
    m1_peak_freq_damaged = freqs_a[m1_mask_a][np.argmax(psd_a[m1_mask_a])]
    print(f"  Post-damage Detected Mode 1 frequency: {m1_peak_freq_damaged:.3f} Hz (Expected target: ~1.1 Hz)")
    assert abs(m1_peak_freq_damaged - 1.1) < 0.15, f"Scenario A post-damage Mode 1 peak {m1_peak_freq_damaged:.2f} Hz out of tolerance!"
    
    # Verify amplitude magnification: scale factor of 2.5x
    pre_damage_segment_healthy = filtered_mid_healthy[:24000]
    post_damage_segment_damaged = filtered_mid_damaged_a[24000:]
    ratio = np.std(post_damage_segment_damaged) / np.std(pre_damage_segment_healthy)
    print(f"  Mid-Span post-damage amplitude standard deviation ratio (post-damage / healthy-pre): {ratio:.3f}x (Expected: ~2.5x)")
    assert 2.0 <= ratio <= 3.0, f"Scenario A amplitude magnification ratio {ratio:.2f}x is outside expected 2.5x range!"
    print(" [PASS] Scenario A (Stiffness Loss) indicators verified.")
    print("-" * 60)

    # 7. Generate and Verify Scenario B (Pier Settlement / Tilting)
    print("Generating and verifying Scenario B (Damage: Pier Settlement)...")
    df_damaged_b = generator.generate_damaged_scenario(scenario="B")
    
    sensor_pier = "Sensor_Pier_Left"
    raw_pier_b = df_damaged_b[sensor_pier].to_numpy()
    
    # Check baseline offset: final 100 seconds (post-ramp) compared to initial baseline
    post_settlement_ramp = raw_pier_b[24500:] # t >= 490s (settlement ramp complete)
    pre_settlement_baseline = raw_pier_b[:24000]
    
    offset_difference = np.mean(post_settlement_ramp) - np.mean(pre_settlement_baseline)
    print(f"  Detected Pier Settlement static offset shift: {offset_difference:.4f} g (Expected target: ~0.15 g)")
    assert abs(offset_difference - config.SCENARIO_B_PIER_OFFSET) < 0.05, f"Scenario B offset shift {offset_difference:.4f} g out of range!"
    
    # Check noise variance multiplier: 3.0x on standard deviation in the noise-only band high-frequency (10-18 Hz)
    noise_only_pre = apply_bandpass_filter(
        pre_settlement_baseline,
        fs=config.FS,
        lowcut=10.0,
        highcut=18.0,
        order=config.FILTER_ORDER
    )
    noise_only_post = apply_bandpass_filter(
        post_settlement_ramp,
        fs=config.FS,
        lowcut=10.0,
        highcut=18.0,
        order=config.FILTER_ORDER
    )
    std_ratio = np.std(noise_only_post) / np.std(noise_only_pre)
    print(f"  Pier Left post-damage noise std ratio (10-18 Hz band): {std_ratio:.3f}x (Expected target: ~3.0x)")
    assert 2.5 <= std_ratio <= 3.5, f"Scenario B noise multiplier {std_ratio:.2f}x is outside expected 3.0x range!"
    print(" [PASS] Scenario B (Pier Settlement) indicators verified.")
    print("-" * 60)

    # 8. Persist Parquet Datasets
    print("Persisting datasets to Apache Parquet...")
    
    df_healthy.to_parquet(healthy_parquet_path, compression="snappy")
    print(f"  Healthy baseline persisted to: {healthy_parquet_path}")
    print(f"  File size: {os.path.getsize(healthy_parquet_path) / 1024:.2f} kB")
    
    df_damaged_a.to_parquet(damaged_parquet_path, compression="snappy")
    print(f"  Damaged scenario persisted to: {damaged_parquet_path}")
    print(f"  File size: {os.path.getsize(damaged_parquet_path) / 1024:.2f} kB")
    
    print(" [PASS] Parquet persistence checks completed.")
    print("=" * 60)
    print("PHASE 1 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    import scipy.signal
    try:
        run_checks_and_save()
    except AssertionError as e:
        print(f"\n[FAIL] Assertion check failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error occurred during verification: {e}", file=sys.stderr)
        sys.exit(1)
