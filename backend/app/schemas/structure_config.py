"""Pydantic schemas for multi-structure SHM configuration.

Provides strictly-validated, domain-agnostic YAML configurations for any civil structure.
No hardcoded channel counts, sensor indices, or bridge names.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator
import yaml


class SensorType(str, Enum):
    """Supported physical sensor transduction types."""
    ACCEL = "accel"
    TEMP = "temp"
    STRAIN = "strain"
    TILT = "tilt"
    DISPLACEMENT = "displacement"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_lower = value.lower()
            if val_lower in ("accelerometer", "acceleration", "acc"):
                return cls.ACCEL
            if val_lower in ("temperature", "thermal", "degc"):
                return cls.TEMP
            if val_lower in ("strain_gauge", "sg"):
                return cls.STRAIN
        return cls.OTHER


class SensorConfig(BaseModel):
    """Configuration for an individual physical sensor node."""
    id: str
    name: str
    type: SensorType = SensorType.ACCEL
    unit: str = "g"
    axis: Optional[str] = None  # e.g., 'ax', 'ay', 'az', 'temp'
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    position_xyz: Optional[Tuple[float, float, float]] = None
    mesh_name: Optional[str] = None
    channel_index: Optional[int] = None
    description: str = ""

    model_config = {"use_enum_values": True}


class ModeConfig(BaseModel):
    """Natural frequency search band configuration for spectral tracking."""
    name: str
    band_hz: Tuple[float, float]
    channel_type: Optional[str] = "az"  # e.g., 'az', 'ay', or specific sensor_id

    @field_validator("band_hz")
    def validate_band(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"band_hz min ({v[0]}) must be strictly less than max ({v[1]})")
        if v[0] < 0:
            raise ValueError(f"band_hz frequencies must be non-negative, got {v}")
        return v


class DetectorConfig(BaseModel):
    """Configuration for a pluggable anomaly detector instance."""
    detector_id: str
    type: str  # e.g., 'waveform_ae', 'spectral_tracking'
    params: Dict[str, Any] = Field(default_factory=dict)
    update_cadence_seconds: Optional[float] = None
    min_channels: Optional[int] = None
    required_channels: Optional[List[str]] = None


class PersistenceConfig(BaseModel):
    """Temporal persistence and debouncing settings."""
    persistence_k: int = 2
    t_persist: float = 0.04
    warning_debounce_seconds: float = 30.0


class AcquisitionMode(str, Enum):
    """Telemetry acquisition cadence model."""
    CONTINUOUS = "continuous"
    SCHEDULED_RECORDINGS = "scheduled_recordings"


class ResamplingConfig(BaseModel):
    """Edge resampling and anti-alias filtering configuration."""
    enabled: bool = True
    filter_type: str = "butter"  # 'butter' | 'cheby1' | 'none'
    order: int = 8
    cutoff_hz: float = 20.0
    ripple_db: float = 0.5
    mode: str = "streaming_causal"  # 'streaming_causal' | 'block_zero_phase'
    max_gap_interpolate_sec: float = 0.25


class QualityThresholdsConfig(BaseModel):
    """Configurable, baseline-calibratable thresholds for the data-quality layer."""
    flatline_min_std: float = 1e-6
    stuck_min_unique_ratio: float = 0.05
    saturation_max_fraction: float = 0.02
    default_accel_range: Tuple[float, float] = (-20.0, 20.0)
    spike_robust_z_threshold: float = 12.0
    spike_kurtosis_threshold: float = 25.0
    drift_sigma_multiplier: float = 6.0
    gain_high_ratio: float = 2.5
    gain_low_ratio: float = 0.35
    noise_floor_ratio: float = 4.0
    dropout_zero_fraction: float = 0.05
    max_gap_interpolate_sec: float = 0.25
    reorder_window_frames: int = 16
    late_arrival_tolerance_sec: float = 5.0
    clock_offset_max_sec: float = 2.0
    clock_jump_max_sec: float = 1.0
    clock_drift_max_ppm: float = 500.0
    clock_sync_skew_max_sec: float = 0.1
    offline_timeout_sec: float = 10.0
    temp_range_c: Tuple[float, float] = (-40.0, 70.0)
    temp_max_roc_c_per_min: float = 5.0
    weight_completeness: float = 0.4
    weight_timing: float = 0.2
    weight_signal_health: float = 0.4


class IngestionSourceConfig(BaseModel):
    """Per-structure ingestion source configuration."""
    type: str = "simulator"  # simulator | replay | mqtt | http | kafka
    params: Dict[str, Any] = Field(default_factory=dict)
    reset_on_recording_boundary: bool = True
    max_queue_size: int = 512


class StructureConfig(BaseModel):
    """Top-level configuration for an arbitrary civil structure."""
    structure_id: str
    name: str
    structure_type: str = "bridge"  # bridge, building, dam, tower, etc.
    fs: float = 50.0
    native_fs: Optional[float] = None
    detector_fs: Optional[float] = None
    acquisition_mode: AcquisitionMode = AcquisitionMode.CONTINUOUS
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    quality: QualityThresholdsConfig = Field(default_factory=QualityThresholdsConfig)
    ingestion: IngestionSourceConfig = Field(default_factory=IngestionSourceConfig)
    sensors: List[SensorConfig]
    temperature_sensor_id: Optional[str] = None
    detectors: List[DetectorConfig] = Field(default_factory=list)
    modes: Optional[List[ModeConfig]] = None
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    description: str = ""

    model_config = {"use_enum_values": True}

    @property
    def effective_native_fs(self) -> float:
        return float(self.native_fs if self.native_fs is not None else self.fs)

    @property
    def effective_detector_fs(self) -> float:
        return float(self.detector_fs if self.detector_fs is not None else self.fs)

    @property
    def num_channels(self) -> int:
        return len(self.sensors)

    @property
    def channel_names(self) -> List[str]:
        return [s.name for s in self._sorted_sensors()]

    @property
    def sensor_ids(self) -> List[str]:
        return [s.id for s in self._sorted_sensors()]

    def _sorted_sensors(self) -> List[SensorConfig]:
        """Returns sensors sorted by explicit channel_index if present, else original order."""
        if all(s.channel_index is not None for s in self.sensors):
            return sorted(self.sensors, key=lambda s: s.channel_index or 0)
        return self.sensors

    def get_temperature_sensor(self) -> Optional[SensorConfig]:
        if self.temperature_sensor_id:
            for s in self.sensors:
                if s.id == self.temperature_sensor_id:
                    return s
        for s in self.sensors:
            if s.type == SensorType.TEMP.value or s.type == SensorType.TEMP:
                return s
        return None

    def compute_sensor_hash(self) -> str:
        """SHA-256 fingerprint of sorted sensor definitions and sampling rate."""
        payload = [
            {
                "id": s.id,
                "name": s.name,
                "type": str(s.type),
                "unit": s.unit,
                "index": s.channel_index,
            }
            for s in self._sorted_sensors()
        ]
        key = json.dumps({"fs": self.fs, "sensors": payload}, sort_keys=True)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def compute_band_hash(self) -> str:
        """SHA-256 fingerprint of frequency search bands."""
        if not self.modes:
            return ""
        payload = [
            {"name": m.name, "band": list(m.band_hz), "channel_type": m.channel_type}
            for m in self.modes
        ]
        key = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "StructureConfig":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(exclude_none=True), f, sort_keys=False)
