"""Comprehensive Test Suite for Phase 5: Ingestion Adapters & Dynamic Structure Registration."""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.core.artifact_manager import ArtifactManager
from app.core.structure_registry import StructureRegistry
from app.core.structures import (
    DEFAULT_BRIDGE_CONFIG,
    SensorDefinition,
    SensorType,
    StructureConfig,
    StructureType,
)
from app.ingestion.factory import IngestionFactory
from app.ingestion.http_source import HTTPSource
from app.ingestion.kafka_source import KafkaSource
from app.ingestion.mqtt_source import MQTTSource
from app.ingestion.simulator_source import SimulatorSource
from app.main import app


# ---------------------------------------------------------------------------
# 1. MQTTSource Unit Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mqtt_source_payload_parsing_and_queue():
    channels = ["Sensor_Pier", "Sensor_Girder", "Sensor_Temp"]
    source = MQTTSource(channel_names=channels, host="localhost", topic="test/topic")

    # Format 1: values list
    p1 = {"values": [0.12, -0.05, 25.4]}
    f1 = source.parse_payload(p1)
    assert f1 is not None
    assert f1.shape == (3,)
    np.testing.assert_allclose(f1, [0.12, -0.05, 25.4], rtol=1e-5)

    # Format 2: raw json bytes with frame list
    p2 = json.dumps({"frame": [1.0, 2.0, 3.0]}).encode("utf-8")
    f2 = source.parse_payload(p2)
    assert f2 is not None
    np.testing.assert_allclose(f2, [1.0, 2.0, 3.0])

    # Format 3: channel name mapping
    p3 = {"Sensor_Pier": 0.5, "Sensor_Girder": 0.2, "Sensor_Temp": 22.0}
    f3 = source.parse_payload(p3)
    assert f3 is not None
    np.testing.assert_allclose(f3, [0.5, 0.2, 22.0])

    # Format 4: invalid length or missing channel -> None
    assert source.parse_payload({"values": [1.0, 2.0]}) is None
    assert source.parse_payload({"Sensor_Pier": 0.5}) is None

    # Test async emission loop via push_raw_message
    emitted = []

    def on_emit(frame):
        emitted.append(frame)

    await source.start(on_emit)
    assert source.is_running

    ok = source.push_raw_message(p1)
    assert ok
    # Give event loop time to consume queue
    await asyncio.sleep(0.05)
    assert len(emitted) == 1
    np.testing.assert_allclose(emitted[0], [0.12, -0.05, 25.4], rtol=1e-5)

    source.stop()
    assert not source.is_running


# ---------------------------------------------------------------------------
# 2. KafkaSource Unit Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kafka_source_payload_parsing_and_queue():
    channels = ["CH1", "CH2"]
    source = KafkaSource(channel_names=channels, bootstrap_servers="localhost:9092", topic="shm-test")

    # Test bytes payload
    payload = json.dumps({"channels": {"CH1": 3.14, "CH2": 2.71}}).encode("utf-8")
    frame = source.parse_payload(payload)
    assert frame is not None
    np.testing.assert_allclose(frame, [3.14, 2.71], rtol=1e-5)

    emitted = []

    def on_emit(fr):
        emitted.append(fr)

    await source.start(on_emit)
    assert source.is_running

    assert source.push_raw_message(payload)
    await asyncio.sleep(0.05)
    assert len(emitted) == 1
    np.testing.assert_allclose(emitted[0], [3.14, 2.71], rtol=1e-5)

    source.stop()
    assert not source.is_running


# ---------------------------------------------------------------------------
# 3. HTTPSource Unit Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_http_source_direct_push():
    channels = ["Acc_X", "Acc_Y"]
    source = HTTPSource(channel_names=channels)

    emitted = []

    def on_emit(fr):
        emitted.append(fr)

    await source.start(on_emit)
    assert source.is_running

    # Push numpy array
    ok_np = source.push_frame(np.array([0.05, -0.02], dtype=np.float32))
    assert ok_np

    # Push list
    ok_list = source.push_frame([0.10, -0.04])
    assert ok_list

    # Push dict
    ok_dict = source.push_frame({"Acc_X": 0.15, "Acc_Y": -0.06})
    assert ok_dict

    # Invalid channel count
    assert not source.push_frame([0.1])
    assert not source.push_frame({"Acc_X": 0.1})

    await asyncio.sleep(0.05)
    assert len(emitted) == 3
    np.testing.assert_allclose(emitted[0], [0.05, -0.02], rtol=1e-5)
    np.testing.assert_allclose(emitted[1], [0.10, -0.04], rtol=1e-5)
    np.testing.assert_allclose(emitted[2], [0.15, -0.06], rtol=1e-5)

    source.stop()
    assert not source.is_running


# ---------------------------------------------------------------------------
# 4. IngestionFactory Unit Tests
# ---------------------------------------------------------------------------
def test_ingestion_factory():
    ch = ["S1", "S2"]

    # 1. Simulator
    sim = IngestionFactory.create_source("simulator", {"base_temp": 24.0}, ch, fs=100.0)
    assert isinstance(sim, SimulatorSource)
    assert sim.fs == 100.0
    assert sim.base_temp == 24.0

    # 2. MQTT
    mqtt = IngestionFactory.create_source("mqtt", {"host": "broker.emqx.io", "port": 1883}, ch)
    assert isinstance(mqtt, MQTTSource)
    assert mqtt.host == "broker.emqx.io"
    assert mqtt.channel_names == ch

    # 3. Kafka
    kafka = IngestionFactory.create_source("kafka", {"bootstrap_servers": "10.0.0.1:9092"}, ch)
    assert isinstance(kafka, KafkaSource)
    assert kafka.bootstrap_servers == "10.0.0.1:9092"

    # 4. HTTP
    http_src = IngestionFactory.create_source("http", {}, ch)
    assert isinstance(http_src, HTTPSource)
    assert http_src.channel_names == ch

    # 5. Unknown source raises ValueError
    with pytest.raises(ValueError):
        IngestionFactory.create_source("unknown_protocol", {}, ch)


# ---------------------------------------------------------------------------
# 5. ArtifactManager Auto-Synthesis Unit Tests
# ---------------------------------------------------------------------------
def test_artifact_manager_auto_synthesis():
    temp_dir = tempfile.mkdtemp()
    try:
        art_mgr = ArtifactManager(base_artifacts_dir=Path(temp_dir))
        new_config = StructureConfig(
            structure_id="tower_demo_99",
            name="Alpha Transmission Tower",
            structure_type=StructureType.TOWER,
            fs=50.0,
            sensors=[
                SensorDefinition(
                    id="t_s1",
                    name="Tower_Strain_Base",
                    type=SensorType.STRAIN,
                    unit="με",
                    position_xyz=(0.0, 0.0, 1.0),
                    fs=50.0,
                    channel_index=0,
                ),
                SensorDefinition(
                    id="t_s2",
                    name="Tower_Tilt_Top",
                    type=SensorType.TILT,
                    unit="deg",
                    position_xyz=(0.0, 0.0, 80.0),
                    fs=50.0,
                    channel_index=1,
                ),
            ],
        )

        # Artifacts should not exist initially
        vdir = art_mgr.get_structure_version_dir("tower_demo_99", "v1")
        assert not vdir.exists()

        # Call load_artifacts with config -> auto-synthesizes artifacts!
        bundle = art_mgr.load_artifacts(
            structure_id="tower_demo_99",
            version="v1",
            device=torch.device("cpu"),
            config=new_config,
        )

        assert bundle.model is not None
        assert bundle.scaler is not None
        assert bundle.threshold_config is not None
        assert len(bundle.threshold_config.channels) == 2

        # Verify forward pass through synthesized model
        dummy_input = torch.zeros((1, 50, 2), dtype=torch.float32)
        with torch.no_grad():
            output, _ = bundle.model(dummy_input)
        assert output.shape == (1, 50, 2)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. Dynamic Structure Lifecycle REST API Tests
# ---------------------------------------------------------------------------
def test_dynamic_structure_rest_api():
    with TestClient(app) as client:
        # 1. Dynamic structure registration
        req_payload = {
            "structure_id": "test_viaduct_01",
            "name": "Northern Valley Viaduct",
            "structure_type": "BRIDGE",
            "fs": 50.0,
            "sensors": [
                {
                    "channel_name": "Sensor_Pier_North",
                    "sensor_type": "ACCELEROMETER",
                    "coordinates_3d": [10.0, 0.0, -5.0],
                    "threshold_tau": 0.045,
                },
                {
                    "channel_name": "Sensor_Pier_South",
                    "sensor_type": "ACCELEROMETER",
                    "coordinates_3d": [30.0, 0.0, -5.0],
                    "threshold_tau": 0.045,
                },
            ],
            "ingestion": {
                "type": "http",
                "params": {},
            },
        }

        resp = client.post("/api/v1/structures", json=req_payload)
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["structure_id"] == "test_viaduct_01"
        assert data["num_channels"] == 2
        assert data["ingestion_source"] == "HTTP"
        assert data["is_running"] is True

        # 2. Verify duplicate registration fails with 400
        dup_resp = client.post("/api/v1/structures", json=req_payload)
        assert dup_resp.status_code == 400

        # 3. Verify structure appears in GET /api/v1/structures
        list_resp = client.get("/api/v1/structures")
        assert list_resp.status_code == 200
        struct_ids = [s["structure_id"] for s in list_resp.json()]
        assert "test_viaduct_01" in struct_ids

        # 4. Verify detail endpoint GET /api/v1/structures/test_viaduct_01
        det_resp = client.get("/api/v1/structures/test_viaduct_01")
        assert det_resp.status_code == 200
        assert len(det_resp.json()["sensors"]) == 2

        # 5. Push telemetry frame via HTTP POST
        telemetry_payload = {
            "frame": [0.015, -0.022]
        }
        push_resp = client.post("/api/v1/structures/test_viaduct_01/telemetry", json=telemetry_payload)
        assert push_resp.status_code == 200
        assert push_resp.json()["status"] == "accepted"

        # Push telemetry with invalid dimension
        bad_push = client.post("/api/v1/structures/test_viaduct_01/telemetry", json={"frame": [0.015]})
        assert bad_push.status_code == 400

        # 6. Stop ingestion
        stop_resp = client.post("/api/v1/structures/test_viaduct_01/stop")
        assert stop_resp.status_code == 200
        assert "stopped" in stop_resp.json()["message"]

        # 7. Start ingestion
        start_resp = client.post("/api/v1/structures/test_viaduct_01/start")
        assert start_resp.status_code == 200
        assert "started" in start_resp.json()["message"]

        # 8. Prevent deleting default bridge
        del_default = client.delete("/api/v1/structures/bridge_demo_01")
        assert del_default.status_code == 400

        # 9. Cleanly delete registered structure
        del_resp = client.delete("/api/v1/structures/test_viaduct_01")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "success"

        # Verify deleted structure no longer exists in catalog
        list_after = client.get("/api/v1/structures")
        struct_ids_after = [s["structure_id"] for s in list_after.json()]
        assert "test_viaduct_01" not in struct_ids_after

