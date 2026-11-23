"""Core domain models for multi-structure SHM configuration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SensorType(str, Enum):
    """Supported physical sensor transduction types."""
    ACCELEROMETER = "accelerometer"
    TEMPERATURE = "temperature"
    STRAIN = "strain"
    TILT = "tilt"
    DISPLACEMENT = "displacement"


class StructureType(str, Enum):
    """Supported civil structure categories."""
    BRIDGE = "bridge"
    BUILDING = "building"
    DAM = "dam"
    TOWER = "tower"


@dataclass(frozen=True)
class SensorDefinition:
    """Configuration for an individual physical or virtual sensor node."""
    id: str
    name: str
    type: SensorType
    unit: str
    position_xyz: Tuple[float, float, float]
    fs: float
    channel_index: int
    description: str = ""


@dataclass(frozen=True)
class StructureConfig:
    """Top-level configuration for an instrumented structure."""
    structure_id: str
    name: str
    structure_type: StructureType
    sensors: List[SensorDefinition]
    window_size: int = 50
    fs: float = 50.0
    description: str = ""

    @property
    def num_channels(self) -> int:
        return len(self.sensors)

    @property
    def channel_names(self) -> List[str]:
        sorted_sensors = sorted(self.sensors, key=lambda s: s.channel_index)
        return [s.name for s in sorted_sensors]

    @property
    def channel_index_map(self) -> Dict[str, int]:
        return {s.name: s.channel_index for s in self.sensors}

    def compute_adjacency_matrix(self, sigma: float = 10.0) -> Dict[str, Any]:
        """Compute Gaussian spatial affinity matrix W_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))
        across accelerometer vibration sensor nodes for physics-informed spatial diffusion.
        """
        import numpy as np

        sorted_sensors = sorted(self.sensors, key=lambda s: s.channel_index)
        n = len(sorted_sensors)
        coords = np.array([s.position_xyz for s in sorted_sensors], dtype=np.float32)

        # Pairwise Euclidean distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)

        # Gaussian kernel
        W = np.exp(-dist_sq / (2.0 * (sigma ** 2)))

        # Mask non-vibration channels (e.g. ambient temperature) from spatial diffusion
        for i, s in enumerate(sorted_sensors):
            if s.type != SensorType.ACCELEROMETER:
                W[i, :] = 0.0
                W[:, i] = 0.0
                W[i, i] = 1.0

        # Row-normalize to conserve total energy
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_norm = W / row_sums

        return {
            "matrix": W_norm,
            "channel_names": [s.name for s in sorted_sensors],
        }


# Default Bridge Definition matching existing 5-channel bridge model
DEFAULT_BRIDGE_CONFIG = StructureConfig(
    structure_id="bridge_demo_01",
    name="Suspension Bridge Prototype",
    structure_type=StructureType.BRIDGE,
    window_size=50,
    fs=50.0,
    description="5-channel multi-harmonic suspension girder bridge prototype",
    sensors=[
        SensorDefinition(
            id="sensor_0",
            name="Sensor_Abutment_Left",
            type=SensorType.ACCELEROMETER,
            unit="g",
            position_xyz=(-14.0, 0.5, 0.0),
            fs=50.0,
            channel_index=0,
            description="Left abutment vertical vibration response",
        ),
        SensorDefinition(
            id="sensor_1",
            name="Sensor_Pier_Left",
            type=SensorType.ACCELEROMETER,
            unit="g",
            position_xyz=(-7.0, 1.2, 0.0),
            fs=50.0,
            channel_index=1,
            description="Left pier coupled vertical & lateral response",
        ),
        SensorDefinition(
            id="sensor_2",
            name="Sensor_MidSpan_Girder",
            type=SensorType.ACCELEROMETER,
            unit="g",
            position_xyz=(0.0, 1.8, 0.0),
            fs=50.0,
            channel_index=2,
            description="Mid-span girder primary flexural vibration mode",
        ),
        SensorDefinition(
            id="sensor_3",
            name="Sensor_Pier_Right",
            type=SensorType.ACCELEROMETER,
            unit="g",
            position_xyz=(7.0, 1.2, 0.0),
            fs=50.0,
            channel_index=3,
            description="Right pier foundation response",
        ),
        SensorDefinition(
            id="sensor_4",
            name="Sensor_Ambient_Temp",
            type=SensorType.TEMPERATURE,
            unit="degC",
            position_xyz=(0.0, 4.0, 0.0),
            fs=50.0,
            channel_index=4,
            description="Diurnal ambient thermal environment",
        ),
    ],
)
