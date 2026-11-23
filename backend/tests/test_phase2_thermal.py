"""Unit and integration tests for Phase 2 Temperature Decoupling."""

from pathlib import Path
import numpy as np
import pytest

from app.ml.thermal_normalizer import ThermalNormalizer
from app.simulation.signal_generator import SHMSignalGenerator


def test_thermal_normalizer_fit_and_transform():
    """Verify ThermalNormalizer decouples temperature-induced signal drift."""
    # Create synthetic dataset where vibration channels exhibit temperature-dependent thermal expansion
    n_samples = 2000
    temp = np.linspace(15.0, 35.0, n_samples)
    data = np.zeros((n_samples, 5), dtype=np.float32)
    data[:, 4] = temp  # Temperature channel

    # Structural vibration + thermal expansion drift: y_c = vib + 0.05 * (T - 25)^2 + 0.1 * (T - 25)
    for c in range(4):
        vib = np.random.normal(0.0, 0.1, n_samples)
        thermal_drift = 0.05 * ((temp - 25.0) ** 2) + 0.1 * (temp - 25.0)
        data[:, c] = vib + thermal_drift

    normalizer = ThermalNormalizer(vibration_channel_indices=[0, 1, 2, 3], temp_channel_index=4)
    normalizer.fit(data)

    assert normalizer.is_fitted
    assert len(normalizer.models) == 4

    # Apply transform
    normalized = normalizer.transform(data)

    # After normalization, thermal drift is eliminated, variance should be roughly equal to pure vibration noise
    for c in range(4):
        raw_std = np.std(data[:, c])
        norm_std = np.std(normalized[:, c])
        assert norm_std < raw_std, f"Normalized std ({norm_std}) should be less than raw drifted std ({raw_std})"
        # Mean should be near 0
        assert abs(np.mean(normalized[:, c])) < 0.05


def test_thermal_normalizer_serialization(tmp_path):
    """Verify saving and reloading ThermalNormalizer preserves predictions."""
    sig_gen = SHMSignalGenerator()
    df = sig_gen.generate_healthy_baseline()
    data = df.values.astype(np.float32)[:500]

    normalizer = ThermalNormalizer()
    normalizer.fit(data)

    save_path = tmp_path / "thermal_normalizer.joblib"
    normalizer.save(save_path)
    assert save_path.exists()

    loaded = ThermalNormalizer.load(save_path)
    assert loaded.is_fitted
    assert loaded.mean_temp == normalizer.mean_temp

    sample = data[:10]
    out1 = normalizer.transform(sample)
    out2 = loaded.transform(sample)
    np.testing.assert_allclose(out1, out2, rtol=1e-5)
