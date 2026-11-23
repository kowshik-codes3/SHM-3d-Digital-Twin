"""Core SHM modules for configuration and digital signal processing."""

from .config import SHMConfig, SensorConfig
from .dsp import (
    apply_bandpass_filter,
    detrend_signal,
    compute_fft,
    compute_psd
)

__all__ = [
    "SHMConfig",
    "SensorConfig",
    "apply_bandpass_filter",
    "detrend_signal",
    "compute_fft",
    "compute_psd"
]
