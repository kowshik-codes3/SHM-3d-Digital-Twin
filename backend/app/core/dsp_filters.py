"""DSP filters module alias to comply with deliverables representation."""

from .dsp import (
    apply_bandpass_filter,
    detrend_signal,
    compute_fft,
    compute_psd
)

__all__ = [
    "apply_bandpass_filter",
    "detrend_signal",
    "compute_fft",
    "compute_psd"
]
