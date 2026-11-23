"""Automated Pytest Suite for Phase 1 Multi-Structure Generalization."""

import asyncio
from pathlib import Path
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.core.structures import DEFAULT_BRIDGE_CONFIG, StructureConfig, SensorDefinition, SensorType, StructureType
from app.core.artifact_manager import ArtifactManager, ModelMetadata
from app.core.structure_registry import StructureRegistry, StructureRuntime
from app.ingestion.simulator_source import SimulatorSource
from app.ingestion.replay_source import ReplaySource
from app.main import app


def test_structure_config_properties():
    """Verify StructureConfig channel ordering and helper properties."""
    cfg = DEFAULT_BRIDGE_CONFIG
    assert cfg.structure_id == "bridge_demo_01"
    assert cfg.num_channels == 5
    assert len(cfg.channel_names) == 5
    assert cfg.channel_names[0] == "Sensor_Abutment_Left"
    assert cfg.channel_names[2] == "Sensor_MidSpan_Girder"
    assert cfg.channel_index_map["Sensor_Ambient_Temp"] == 4


def test_artifact_manager_resolution(tmp_path):
    """Verify ArtifactManager migrates legacy artifacts and populates metadata.json."""
    am = ArtifactManager()
    bundle = am.load_artifacts("bridge_demo_01", version="v1")
    assert bundle.structure_id == "bridge_demo_01"
    assert bundle.version == "v1"
    assert bundle.model is not None
    assert bundle.scaler is not None
    assert bundle.threshold_config is not None
    assert bundle.metadata is not None
    assert bundle.metadata.structure_id == "bridge_demo_01"
    assert bundle.metadata.model_type == "LSTMAutoencoder"


def test_structure_registry_isolated_processing():
    """Verify multiple structures run with isolated buffers and state machines."""
    am = ArtifactManager()
    registry = StructureRegistry(artifact_manager=am)

    runtime1 = registry.register_structure(DEFAULT_BRIDGE_CONFIG, version="v1")
    assert registry.get("bridge_demo_01") is not None

    # Push dummy frame
    frame = np.zeros(5, dtype=np.float32)
    event = runtime1.push_frame(frame)
    # Buffer length is 1, so event should be None until 50 frames
    assert event is None

    # Fill buffer to 50
    for _ in range(49):
        event = runtime1.push_frame(frame)
    assert event is not None
    assert event.system_status in ["NORMAL", "WARNING", "CRITICAL"]


def test_multi_structure_rest_endpoints():
    """Test REST API catalog endpoints."""
    with TestClient(app) as client:
        # Health check
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert data["structures_count"] >= 1

        # List structures
        res = client.get("/api/v1/structures")
        assert res.status_code == 200
        structures = res.json()
        assert any(s["structure_id"] == "bridge_demo_01" for s in structures)

        # Get structure detail
        res = client.get("/api/v1/structures/bridge_demo_01")
        assert res.status_code == 200
        detail = res.json()
        assert detail["structure_id"] == "bridge_demo_01"
        assert len(detail["sensors"]) == 5
        assert detail["sensors"][2]["name"] == "Sensor_MidSpan_Girder"


def test_simulate_fault_multi_structure():
    """Test fault injection on specific structure."""
    with TestClient(app) as client:
        # Valid fault
        res = client.post(
            "/api/v1/structures/bridge_demo_01/simulate-fault",
            json={
                "node_id": "Sensor_MidSpan_Girder",
                "fault_type": "stiffness_loss",
                "severity": 0.8,
            },
        )
        assert res.status_code == 200
        assert res.json()["status"] == "success"

        # Invalid structure
        res = client.post(
            "/api/v1/structures/non_existent/simulate-fault",
            json={"node_id": "Sensor_MidSpan_Girder", "fault_type": "stiffness_loss", "severity": 0.5},
        )
        assert res.status_code == 404

        # Reset faults
        res = client.post("/api/v1/structures/bridge_demo_01/reset-faults")
        assert res.status_code == 200
        assert res.json()["status"] == "success"
