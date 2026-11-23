"""Pydantic schemas and domain models for CAD-to-3D pipeline."""

from __future__ import annotations

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class SupportedCADFormat(str, Enum):
    """Recognized CAD/BIM and 3D mesh formats."""
    IFC = "ifc"
    STEP = "step"
    IGES = "iges"
    GLB = "glb"
    GLTF = "gltf"
    OBJ = "obj"
    FBX = "fbx"
    UNSUPPORTED_2D = "unsupported_2d"
    UNSUPPORTED_PROPRIETARY = "unsupported_proprietary"
    UNKNOWN = "unknown"


class PartInfo(BaseModel):
    """Semantic physical part within the 3D model."""
    mesh_name: str
    part_id: str
    ifc_class: Optional[str] = None
    triangle_count: int = 0
    vertex_count: int = 0
    bounding_box_meters: Dict[str, List[float]] = Field(
        default_factory=lambda: {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "dimensions": [0.0, 0.0, 0.0]}
    )
    center: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    assigned_sensor_id: Optional[str] = None


class ValidationWarning(BaseModel):
    """Non-blocking validation warning."""
    code: str
    message: str
    severity: str = "WARNING"  # WARNING | INFO


class ValidationReport(BaseModel):
    """Quality and dimensional audit for a converted 3D model."""
    is_valid: bool = True
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    triangle_count: int = 0
    vertex_count: int = 0
    mesh_count: int = 0
    bounding_box_meters: Dict[str, List[float]] = Field(
        default_factory=lambda: {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "dimensions": [0.0, 0.0, 0.0]}
    )
    unnamed_meshes_count: int = 0
    warnings: List[ValidationWarning] = Field(default_factory=list)
    hard_blockers: List[str] = Field(default_factory=list)


class JobStatus(str, Enum):
    """State machine lifecycle for background 3D conversion jobs."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    OPTIMIZING = "OPTIMIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ConversionJob(BaseModel):
    """Metadata and execution logs for an individual conversion task."""
    job_id: str
    structure_id: str
    status: JobStatus = JobStatus.PENDING
    progress_percent: float = 0.0
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    source_filename: str
    source_file_type: str
    output_glb_path: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    validation_report: Optional[ValidationReport] = None

    model_config = {"use_enum_values": True}

    def log(self, message: str) -> None:
        timestamp_str = time.strftime("%H:%M:%S", time.localtime())
        self.logs.append(f"[{timestamp_str}] {message}")
