"""Unified Base Detector Interface and Domain Models for Pluggable SHM Detectors.

Defines the contract for all physical and data-driven anomaly detectors:
- fit(train_recordings, val_recordings)
- calibrate(val_recordings, target_far)
- update(frame_or_chunk) -> DetectorOutput
- save(path) / load(path)
- metadata()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class DetectorState(str, Enum):
    """Standardized operating states for anomaly detectors."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SENSOR_FAULT = "SENSOR_FAULT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class DetectorOutput(BaseModel):
    """Standardized output telemetry frame produced by any detector."""
    detector_id: str
    version: str = "v1"
    timestamp: float = Field(default_factory=time.time)
    state: DetectorState = DetectorState.NORMAL
    score: float = 0.0
    tau: float = 0.0
    update_cadence_seconds: float = 1.0
    details: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def get_git_hash() -> str:
    """Retrieve current commit hash for model provenance."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class BaseDetector(ABC):
    """Abstract Base Class for all pluggable anomaly detectors."""

    def __init__(
        self,
        detector_id: str,
        version: str = "v1",
        update_cadence_seconds: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.detector_id = detector_id
        self.version = version
        self.update_cadence_seconds = update_cadence_seconds
        self.params: Dict[str, Any] = params or {}
        self.is_fitted: bool = False
        self.tau: float = 0.0
        self.training_temp_range: Tuple[float, float] = (-999.0, 999.0)
        self.training_time_range: Dict[str, str] = {}
        self.healthy_hours_trained: float = 0.0
        self.last_update_timestamp: float = 0.0

    @abstractmethod
    def fit(
        self,
        train_recordings: List[np.ndarray],
        val_recordings: Optional[List[np.ndarray]] = None,
        train_meta: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Fit model strictly on healthy training data."""
        pass

    @abstractmethod
    def calibrate(
        self,
        val_recordings: List[np.ndarray],
        target_far: float = 0.01,
        **kwargs: Any,
    ) -> float:
        """Calibrate operating decision threshold strictly on healthy validation data."""
        pass

    @abstractmethod
    def update(self, frame_or_chunk: np.ndarray) -> DetectorOutput:
        """Ingest a live raw frame or chunk and produce a DetectorOutput.
        
        Must never throw unhandled exceptions to prevent service crashes.
        """
        pass

    def reset_state(self, clear_persistence: bool = True) -> None:
        """Reset streaming buffers (and optionally persistence state) at recording boundaries."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist model weights and calibrated state to disk."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load model weights and calibrated state from disk."""
        pass

    def compute_params_hash(self) -> str:
        """SHA-256 hash of detector configuration parameters."""
        serialized = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def metadata(self) -> Dict[str, Any]:
        """Produce standardized metadata for auditability and provenance."""
        return {
            "detector_id": self.detector_id,
            "version": self.version,
            "type": self.__class__.__name__,
            "update_cadence_seconds": self.update_cadence_seconds,
            "is_fitted": self.is_fitted,
            "tau": self.tau,
            "training_temp_range": list(self.training_temp_range),
            "training_time_range": self.training_time_range,
            "healthy_hours_trained": self.healthy_hours_trained,
            "params_hash": self.compute_params_hash(),
            "git_hash": get_git_hash(),
        }
