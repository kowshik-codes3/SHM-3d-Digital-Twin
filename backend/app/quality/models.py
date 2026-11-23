"""Pydantic and Dataclass Models for the Stage 4 Data-Quality Layer."""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QualityStatus(str, Enum):
    """Per-sensor and overall structural data-quality disposition."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    OFFLINE = "OFFLINE"


class SensorBaselineStats(BaseModel):
    """Healthy training/validation baseline statistics for one sensor (never tuned on test)."""
    sensor_id: str
    sensor_type: str = "accel"
    dc_mean: float = 0.0
    dc_std: float = 1e-3
    median: float = 0.0
    mad: float = 1e-3
    rms_median: float = 1e-3
    rms_p99: float = 1e-2
    rms_max: float = 5e-2
    high_freq_noise_floor: float = 1e-6
    calibrated_samples: int = 0


class SensorQualityState(BaseModel):
    """Data-quality report for a single sensor channel on one time block."""
    sensor_id: str
    status: QualityStatus = QualityStatus.OK
    quality_score: float = 1.0
    completeness: float = 1.0
    timing_score: float = 1.0
    signal_health: float = 1.0
    reasons: List[str] = Field(default_factory=list)
    fault_type: Optional[str] = None
    last_seen_ns: Optional[int] = None
    metrics: Dict[str, float] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class QualityEvent(BaseModel):
    """Discrete data-quality incident stored separately from structural health alerts."""
    event_id: str
    structure_id: str
    sensor_id: str
    timestamp: float = Field(default_factory=time.time)
    status: QualityStatus
    fault_type: str
    reason: str
    quality_score: float = 0.0
    resolved: bool = False
    resolved_at: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class QualityReport(BaseModel):
    """Structure-wide quality summary produced for every ingested frame/block."""
    structure_id: str
    timestamp: float = Field(default_factory=time.time)
    overall_status: QualityStatus = QualityStatus.OK
    overall_quality: float = 1.0
    usable_channels: List[str] = Field(default_factory=list)
    faulted_channels: Dict[str, str] = Field(default_factory=dict)
    offline_channels: List[str] = Field(default_factory=list)
    temperature_ok: bool = True
    temperature_reason: Optional[str] = None
    sensors: Dict[str, SensorQualityState] = Field(default_factory=dict)
    counters: Dict[str, int] = Field(default_factory=dict)
    gap_stats: Dict[str, float] = Field(default_factory=dict)
    clock_stats: Dict[str, float] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}
