"""Extreme Value Theory (EVT/POT) and empirical percentile threshold calibration.

Implements Peaks-Over-Threshold (POT) using Generalized Pareto Distribution (GPD)
to extrapolate tail probabilities for extreme anomaly thresholds, alongside
empirical percentile baselines (e.g. 99.9th percentile) and SPC Gaussian limits.
Includes decorrelation support (strided, block maxima) to respect EVT i.i.d. assumptions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import genpareto

from app.ml.threshold_calibrator import ChannelThreshold, ThresholdConfig


@dataclass
class ChannelEVTThreshold:
    """EVT POT parameters and thresholds for a single channel."""

    name: str
    tail_u: float
    gpd_c: float
    gpd_scale: float
    tau_evt: float
    tau_percentile_999: float
    tau_spc_3sigma: float
    warn_spc_2sigma: float
    warn_percentile_99: float = 0.0


@dataclass
class EVTCalibrationConfig:
    """Full calibration configuration comparing EVT, Percentile, and SPC."""

    calibration_timestamp: float
    tail_quantile: float
    target_prob: float
    channels: Dict[str, ChannelEVTThreshold]

    def to_threshold_config(
        self,
        threshold_type: str = "evt",
        warn_type: str = "spc_2sigma",
    ) -> ThresholdConfig:
        """Converts to production ThresholdConfig using chosen threshold and warning variants.

        Args:
            threshold_type: 'evt', 'percentile', or 'spc'
            warn_type: 'percentile_99' or 'spc_2sigma'
        """
        prod_channels: Dict[str, ChannelThreshold] = {}
        for name, ch in self.channels.items():
            if threshold_type == "evt":
                chosen_tau = ch.tau_evt
            elif threshold_type == "percentile":
                chosen_tau = ch.tau_percentile_999
            else:
                chosen_tau = ch.tau_spc_3sigma

            if warn_type == "percentile_99" and ch.warn_percentile_99 > 0:
                chosen_warn = ch.warn_percentile_99
            else:
                chosen_warn = ch.warn_spc_2sigma

            prod_channels[name] = ChannelThreshold(
                name=name,
                mu=ch.tail_u,
                sigma=ch.gpd_scale,
                tau=float(chosen_tau),
                warn=float(chosen_warn),
                k=3.0,
            )

        return ThresholdConfig(
            calibration_timestamp=self.calibration_timestamp,
            k_multiplier=3.0,
            channels=prod_channels,
        )

    def to_json(self) -> dict:
        return {
            "calibration_timestamp": self.calibration_timestamp,
            "tail_quantile": self.tail_quantile,
            "target_prob": self.target_prob,
            "channels": {name: asdict(ch) for name, ch in self.channels.items()},
        }

    def save(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_json(), f, indent=2)


class EVTCalibrator:
    """Calibrates EVT, Percentile, and SPC thresholds from validation residuals."""

    def __init__(
        self,
        tail_quantile: float = 0.95,
        target_prob: float = 0.001,
        decorrelation: str = "none",  # 'none', 'strided', 'block_maxima'
        stride: int = 10,
        block_size: int = 20,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        self.tail_quantile = tail_quantile
        self.target_prob = target_prob
        self.decorrelation = decorrelation
        self.stride = stride
        self.block_size = block_size
        self.channel_names = channel_names

    def _decorrelate(self, col: np.ndarray) -> np.ndarray:
        """Applies decimation or block maxima to break autocorrelation in overlapping windows."""
        if self.decorrelation == "strided":
            return col[:: self.stride]
        elif self.decorrelation == "block_maxima":
            n_blocks = len(col) // self.block_size
            if n_blocks > 0:
                trimmed = col[: n_blocks * self.block_size]
                return np.max(trimmed.reshape(n_blocks, self.block_size), axis=1)
            return col
        return col

    def calibrate(
        self,
        residuals: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ) -> EVTCalibrationConfig:
        """Computes EVT POT, 99.9th percentile, 99th percentile, and 3-sigma thresholds."""
        import time

        ch_names = channel_names or self.channel_names
        assert ch_names is not None
        assert residuals.shape[1] == len(ch_names)

        channels: Dict[str, ChannelEVTThreshold] = {}

        for c, name in enumerate(ch_names):
            col_raw = residuals[:, c]
            mu = float(np.mean(col_raw))
            sigma = float(np.std(col_raw, ddof=1)) if len(col_raw) > 1 else 1e-6
            spc_3sigma = mu + 3.0 * sigma
            spc_2sigma = mu + 2.0 * sigma
            warn_p99 = float(np.percentile(col_raw, 99.0))
            tau_p999 = float(np.percentile(col_raw, 99.9))

            # Decorrelate for EVT tail fitting if requested
            col_evt = self._decorrelate(col_raw)
            n_total = len(col_evt)

            u = float(np.percentile(col_evt, self.tail_quantile * 100.0))
            excesses = col_evt[col_evt > u] - u
            n_excess = len(excesses)

            if n_excess >= 10:
                try:
                    shape_c, _, scale = genpareto.fit(excesses, floc=0)
                    if abs(shape_c) > 1e-4:
                        factor = ((n_total / n_excess) * self.target_prob) ** (-shape_c)
                        tau_evt = float(u + (scale / shape_c) * (factor - 1.0))
                    else:
                        tau_evt = float(u - scale * np.log((n_total / n_excess) * self.target_prob))
                except Exception:
                    shape_c = 0.0
                    scale = sigma
                    tau_evt = tau_p999
            else:
                shape_c = 0.0
                scale = sigma
                tau_evt = tau_p999

            channels[name] = ChannelEVTThreshold(
                name=name,
                tail_u=u,
                gpd_c=float(shape_c),
                gpd_scale=float(scale),
                tau_evt=max(tau_evt, u),
                tau_percentile_999=tau_p999,
                tau_spc_3sigma=spc_3sigma,
                warn_spc_2sigma=spc_2sigma,
                warn_percentile_99=warn_p99,
            )

        return EVTCalibrationConfig(
            calibration_timestamp=time.time(),
            tail_quantile=self.tail_quantile,
            target_prob=self.target_prob,
            channels=channels,
        )
