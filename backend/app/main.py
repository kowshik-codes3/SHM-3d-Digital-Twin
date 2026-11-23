"""Main FastAPI Application Server for Multi-Structure SHM Digital Twin Telemetry.

Manages application lifecycle, StructureRegistry, per-structure artifact loading,
sub-second streaming inference, REST management endpoints, and WebGL WebSocket pub/sub.
"""

import asyncio
import contextlib
import hmac
import logging
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field
import torch

from fastapi import FastAPI, Header, HTTPException, Query, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models3d.job_manager import ModelJobManager
from app.models3d.procedural import ProceduralModelGenerator

from app.core.artifact_manager import ArtifactManager
from app.core.config import settings
from app.core.connection_manager import ConnectionManager
from app.core.structure_registry import StructureRegistry
from app.detectors.registry import StructureRegistry as PluggableStructureRegistry
from app.core.structures import (
    DEFAULT_BRIDGE_CONFIG,
    SensorDefinition,
    SensorType,
    StructureConfig,
    StructureType,
)
from app.db.database import DatabaseManager
from app.db.alert_repository import AlertRepository
from app.ingestion.base import Frame
from app.ingestion.factory import IngestionFactory
from app.ingestion.http_source import HTTPSource
from app.ingestion.protocol import decode_frame_binary
from app.ingestion.simulator_source import SimulatorSource
from app.ml.telemetry_formatter import TelemetryFormatter
import numpy as np

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SHM.Server")

# Global Application State
class AppState:
    artifact_manager: ArtifactManager = ArtifactManager()
    structure_registry: StructureRegistry = StructureRegistry(artifact_manager)
    pluggable_registry: PluggableStructureRegistry = PluggableStructureRegistry()
    http_ingest_sources: Dict[str, HTTPSource] = {}
    model_job_manager: ModelJobManager = ModelJobManager()
    conn_manager: ConnectionManager = ConnectionManager()
    db: DatabaseManager = DatabaseManager()
    alert_repo: AlertRepository = AlertRepository(db)
    device: torch.device = torch.device("cpu")
    start_time: float = time.time()


app_state = AppState()


async def broadcast_telemetry(payload: Dict[str, Any]) -> None:
    """Broadcast telemetry payload to connected clients on structure_id."""
    structure_id = payload.get("structure_id", "bridge_demo_01")
    await app_state.conn_manager.broadcast(payload, structure_id=structure_id)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Context Manager handling startup & shutdown procedures."""
    logger.info("=" * 60)
    logger.info("STARTING MULTI-STRUCTURE SHM DIGITAL TWIN FASTAPI SERVER")
    logger.info("=" * 60)

    # 1. Device selection
    configured_device = settings.DEVICE.lower()
    if configured_device == "cuda" and torch.cuda.is_available():
        app_state.device = torch.device("cuda")
    else:
        app_state.device = torch.device("cpu")
    logger.info(f"PyTorch Execution Device: {app_state.device} (configured: {settings.DEVICE})")

    # 2. Initialise persistence layer
    app_state.db.init()
    logger.info("✅ SQLite database initialised.")

    # 3. Load YAML structure configurations into PluggableRegistry & recover crash spools
    configs_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    loaded_cnt = app_state.pluggable_registry.load_from_configs_dir(configs_dir)
    for sid, p_rt in app_state.pluggable_registry.runtimes.items():
        p_rt.broadcast_fn = broadcast_telemetry
        app_state.http_ingest_sources[sid] = HTTPSource(
            channel_names=p_rt.config.channel_names,
            structure_id=sid,
            fs=p_rt.config.effective_detector_fs,
            queue_maxsize=p_rt.config.ingestion.max_queue_size,
            rate_limit_rps=settings.HTTP_RATE_LIMIT_RPS,
            idempotency_ttl_seconds=settings.IDEMPOTENCY_TTL_SECONDS,
        )
        rec_frames, gap_ev = p_rt.spool.recover()
        if rec_frames or gap_ev:
            logger.info("Recovering %d spooled frames for structure '%s'...", len(rec_frames), sid)
            for rec_item in rec_frames:
                p_rt.push_ingestion_item(rec_item, spool_write=False)
            if gap_ev is not None:
                p_rt.push_ingestion_item(gap_ev, spool_write=False)
            p_rt.spool.clear()
        src_cfg = p_rt.config.ingestion
        if src_cfg.type == "simulator" and not settings.DEMO_MODE:
            logger.info("Skipping simulator source for '%s' because DEMO_MODE=false", sid)
            await p_rt.start_worker()
        elif src_cfg.params.get("autostart", False):
            try:
                src = IngestionFactory.create(
                    src_cfg.type,
                    channel_names=p_rt.config.channel_names,
                    config={**src_cfg.params, "structure_id": sid, "fs": p_rt.config.effective_native_fs},
                )
                p_rt._source_task = asyncio.create_task(p_rt.run_ingestion_source(src))
            except Exception as exc:
                logger.warning("Failed to autostart source for '%s': %s", sid, exc)
                await p_rt.start_worker()
        else:
            await p_rt.start_worker()
    logger.info(f"Loaded {loaded_cnt} YAML structure configs into PluggableRegistry.")

    # 4. Register default bridge structure (Simulator only when DEMO_MODE=true)
    if settings.DEMO_MODE:
        default_source = SimulatorSource(fs=DEFAULT_BRIDGE_CONFIG.fs, base_temp=26.0)
    else:
        default_source = HTTPSource(
            channel_names=DEFAULT_BRIDGE_CONFIG.channel_names,
            structure_id=DEFAULT_BRIDGE_CONFIG.structure_id,
            fs=DEFAULT_BRIDGE_CONFIG.fs,
        )
    runtime = app_state.structure_registry.register_structure(
        config=DEFAULT_BRIDGE_CONFIG,
        version="v1",
        ingestion_source=default_source,
        broadcast_fn=broadcast_telemetry,
        device=app_state.device,
        alert_repository=app_state.alert_repo,
    )
    logger.info("✅ Default bridge registered in StructureRegistry.")

    # 5. Start ingestion for all structures
    await app_state.structure_registry.start_all()
    logger.info("✅ Ingestion pipelines active.")

    yield

    # Shutdown
    logger.info("Halting structure runtimes...")
    for p_rt in app_state.pluggable_registry.runtimes.values():
        await p_rt.stop_pipeline()
    await app_state.structure_registry.stop_all_async()
    app_state.db.close()
    logger.info("FastAPI Server shutdown complete.")


# Instantiate FastAPI Application
app = FastAPI(
    title="Multi-Structure SHM Digital Twin Telemetry API",
    description="Sub-Second Anomaly Detection & WebGL WebSocket Streaming Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for WebGL Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas for Request/Response Validation
class FaultInjectionRequest(BaseModel):
    node_id: Optional[str] = Field("Sensor_MidSpan_Girder", example="Sensor_MidSpan_Girder", description="Target channel node ID")
    fault_type: str = Field("stiffness_loss", example="stiffness_loss", description="Fault type")
    severity: float = Field(0.65, ge=0.1, le=2.0, description="Severity multiplier")


class FaultInjectionResponse(BaseModel):
    status: str
    message: str
    structure_id: str
    node_id: str
    severity: float


class AlertRecordResponse(BaseModel):
    id: int
    structure_id: str
    structure_name: str
    timestamp: float
    system_status: str
    localized_defect_node: Optional[str]
    acknowledged: bool
    created_at: str


class SensorPositionUpdateRequest(BaseModel):
    position_xyz: List[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] coordinate in meters")
    mesh_name: Optional[str] = Field(None, description="Optional target 3D part mesh name")


class AlertDetailResponse(AlertRecordResponse):
    nodes: Dict[str, Any]


class AlertListResponse(BaseModel):
    alerts: List[AlertRecordResponse]
    total: int
    limit: int
    offset: int


class AlertStatsResponse(BaseModel):
    structure_id: str
    total: int
    critical: int
    warning: int
    acknowledged: int
    unacknowledged: int


class SensorCreateSchema(BaseModel):
    channel_name: str
    sensor_type: str = "ACCELEROMETER"
    coordinates_3d: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    threshold_tau: float = 0.04


class IngestionConfigSchema(BaseModel):
    type: str = "simulator"  # simulator | replay | mqtt | kafka | http
    params: Dict[str, Any] = Field(default_factory=dict)


class StructureCreateRequest(BaseModel):
    structure_id: str = Field(..., description="Unique alphanumeric identifier")
    name: str = Field(..., description="Display name for the civil asset")
    structure_type: str = Field("BRIDGE", description="BRIDGE, BUILDING, DAM, WIND_TURBINE, TUNNEL, OFFSHORE")
    fs: float = Field(50.0, ge=1.0, le=1000.0, description="Sampling rate in Hz")
    sensors: List[SensorCreateSchema] = Field(..., min_length=1)
    ingestion: IngestionConfigSchema = Field(default_factory=IngestionConfigSchema)


class StructureActionResponse(BaseModel):
    status: str
    structure_id: str
    message: str


class TelemetryPushRequest(BaseModel):
    frame: Optional[List[float]] = None
    channels: Optional[Dict[str, float]] = None
    values: Optional[List[float]] = None


# REST Endpoints
@app.get("/health", summary="System Health & Diagnostic Status")
async def health_check() -> Dict[str, Any]:
    """Returns backend system status, device configuration, and WebSocket subscriber count."""
    default_rt = app_state.structure_registry.get("bridge_demo_01")
    return {
        "status": "online",
        "device": str(app_state.device),
        "structures_count": len(app_state.structure_registry.list_structures()),
        "active_websockets": app_state.conn_manager.active_count(),
        "inferences_executed": default_rt.inference_count if default_rt else 0,
        "uptime_seconds": round(time.time() - app_state.start_time, 2),
    }


@app.get("/api/v1/structures", summary="List All Registered Structures")
async def list_structures() -> List[Dict[str, Any]]:
    """Returns catalog of all registered civil structures with live status."""
    results = app_state.structure_registry.list_structures()
    existing_ids = {s["structure_id"] for s in results}
    for p_struct in app_state.pluggable_registry.list():
        if p_struct["structure_id"] not in existing_ids:
            results.append({
                "structure_id": p_struct["structure_id"],
                "name": p_struct["name"],
                "structure_type": p_struct["structure_type"],
                "num_channels": p_struct["num_channels"],
                "channels": [s["name"] for s in p_struct["sensors"]],
                "sensors": [
                    {
                        "id": s["id"],
                        "name": s["name"],
                        "channel_name": s["name"],
                        "type": str(s["type"]),
                        "sensor_type": str(s["type"]),
                        "coordinates_3d": s.get("position_xyz") or [0.0, 0.0, 0.0],
                        "position_xyz": s.get("position_xyz") or [0.0, 0.0, 0.0],
                        "threshold_tau": 0.04,
                    }
                    for s in p_struct["sensors"]
                ],
                "detectors": p_struct.get("detectors", []),
                "is_running": True,
                "inferences_executed": 0,
            })
    return results


@app.get("/api/v1/structures/{structure_id}", summary="Get Structure Details & Sensors")
async def get_structure_details(structure_id: str) -> Dict[str, Any]:
    """Returns detailed configuration, sensor coordinates, and status for a structure."""
    # Check pluggable registry first
    p_rt = app_state.pluggable_registry.get(structure_id)
    if p_rt is not None:
        cfg = p_rt.config
        return {
            "structure_id": cfg.structure_id,
            "name": cfg.name,
            "structure_type": cfg.structure_type,
            "description": cfg.description,
            "fs": cfg.fs,
            "sensors": [
                {
                    "id": s.id,
                    "name": s.name,
                    "type": str(s.type),
                    "unit": s.unit,
                    "position_xyz": s.position_xyz,
                    "channel_index": s.channel_index,
                    "description": s.description,
                }
                for s in cfg.sensors
            ],
            "detectors": [
                {
                    "detector_id": d_id,
                    "type": d.__class__.__name__,
                    "cadence_seconds": d.update_cadence_seconds,
                    "is_fitted": d.is_fitted,
                    "tau": d.tau,
                }
                for d_id, d in p_rt.detectors.items()
            ],
            "inferences_executed": p_rt.frame_counter,
            "active_connections": app_state.conn_manager.active_count(structure_id),
        }

    # Fallback to legacy registry
    rt = app_state.structure_registry.get(structure_id)
    if not rt:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")
    cfg = rt.config
    return {
        "structure_id": cfg.structure_id,
        "name": cfg.name,
        "structure_type": cfg.structure_type.value,
        "description": cfg.description,
        "fs": cfg.fs,
        "window_size": cfg.window_size,
        "sensors": [
            {
                "id": s.id,
                "name": s.name,
                "type": s.type.value,
                "unit": s.unit,
                "position_xyz": s.position_xyz,
                "channel_index": s.channel_index,
                "description": s.description,
            }
            for s in cfg.sensors
        ],
        "inferences_executed": rt.inference_count,
        "active_connections": app_state.conn_manager.active_count(structure_id),
    }


@app.get("/api/v1/structures/{structure_id}/detectors", summary="Get Detector Statuses for Structure")
async def get_structure_detectors(structure_id: str) -> Dict[str, Any]:
    """Returns per-detector status, metrics, calibration bounds, and cadence for a structure."""
    p_rt = app_state.pluggable_registry.get(structure_id)
    if p_rt is None:
        legacy_rt = app_state.structure_registry.get(structure_id)
        if not legacy_rt:
            raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")
        return {
            "structure_id": structure_id,
            "detectors": [
                {
                    "detector_id": "waveform_ae_legacy",
                    "type": "WaveformAEDetector",
                    "version": "v1",
                    "cadence_seconds": 1.0,
                    "is_fitted": True,
                    "tau": 0.04,
                    "training_temp_range": [-1.2, 25.1],
                    "healthy_hours_trained": 10.3,
                    "latest_state": "NORMAL",
                    "latest_score": 0.012,
                    "reason": None,
                }
            ],
        }

    detectors_info = []
    for d_id, det in p_rt.detectors.items():
        latest_out = p_rt.latest_outputs.get(d_id)
        detectors_info.append({
            "detector_id": d_id,
            "type": det.__class__.__name__,
            "version": det.version,
            "cadence_seconds": det.update_cadence_seconds,
            "is_fitted": det.is_fitted,
            "tau": det.tau,
            "training_temp_range": list(det.training_temp_range),
            "healthy_hours_trained": det.healthy_hours_trained,
            "latest_state": latest_out.state.value if latest_out else "NORMAL",
            "latest_score": latest_out.score if latest_out else None,
            "reason": latest_out.reason if latest_out else None,
        })

    return {
        "structure_id": structure_id,
        "detectors": detectors_info,
    }


@app.put("/api/v1/structures/{structure_id}/sensors", summary="Update Structure Sensors")
async def update_structure_sensors(structure_id: str, sensors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Updates physical 3D placement coordinates and sensor definitions for a structure."""
    rt = app_state.structure_registry.get(structure_id)
    if not rt:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")

    new_sensor_defs = []
    for idx, s in enumerate(sensors):
        ch_name = s.get("channel_name") or s.get("name") or f"Sensor_Node_{idx+1}"
        s_type_raw = s.get("sensor_type") or s.get("type", "ACCELEROMETER")
        try:
            stype = SensorType[str(s_type_raw).upper()]
        except KeyError:
            stype = SensorType.ACCELEROMETER

        coords = s.get("coordinates_3d") or s.get("position_xyz") or [0.0, 0.0, 0.0]
        coords_tuple = tuple(coords) if len(coords) == 3 else (0.0, 0.0, 0.0)

        new_sensor_defs.append(
            SensorDefinition(
                id=s.get("id") or f"{structure_id}_s{idx+1}",
                name=ch_name,
                type=stype,
                unit="°C" if stype == SensorType.TEMPERATURE else "m/s²",
                position_xyz=coords_tuple,
                fs=rt.config.fs,
                channel_index=idx,
                description=f"Sensor node {ch_name}",
            )
        )

    rt.config.sensors = new_sensor_defs
    if structure_id in app_state.structure_registry._configs:
        app_state.structure_registry._configs[structure_id].sensors = new_sensor_defs
    return {
        "status": "success",
        "structure_id": structure_id,
        "sensors_count": len(new_sensor_defs),
        "message": f"Updated {len(new_sensor_defs)} sensors for {structure_id}.",
    }


# ── 3D Model Import & Geometry Management ────────────────────────────────────────

@app.post("/api/v1/structures/{structure_id}/model", summary="Upload CAD/BIM File for 3D Conversion")
async def upload_structure_model(
    structure_id: str,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Upload CAD file (IFC, STEP, GLB, OBJ, FBX) and launch background conversion job."""
    rt = app_state.pluggable_registry.get(structure_id) or app_state.structure_registry.get(structure_id)
    if rt is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")

    suffix = Path(file.filename).suffix
    temp_dir = Path(tempfile.gettempdir()) / "shm_cad_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_path, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write uploaded file: {exc}")

    job = app_state.model_job_manager.create_job(
        structure_id=structure_id,
        source_filename=file.filename,
        temp_file_path=temp_path,
    )

    return {
        "job_id": job.job_id,
        "structure_id": structure_id,
        "status": job.status,
        "source_filename": job.source_filename,
        "source_file_type": job.source_file_type,
        "error_message": job.error_message,
        "logs": job.logs,
    }


@app.get("/api/v1/structures/{structure_id}/model/status/{job_id}", summary="Get Model Conversion Status")
async def get_model_conversion_status(structure_id: str, job_id: str) -> Dict[str, Any]:
    """Poll job status, progress percentage, conversion logs, and validation report."""
    job = app_state.model_job_manager.get_job(job_id)
    if not job or job.structure_id != structure_id:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found for structure '{structure_id}'.")
    return job.model_dump()


@app.get("/api/v1/structures/{structure_id}/model", summary="Get Active 3D GLB Model")
async def get_structure_model(structure_id: str):
    """Serves the active, optimized GLB 3D model for the structure (or procedural fallback)."""
    active_dir = app_state.model_job_manager.get_active_model_dir(structure_id)
    if active_dir and (active_dir / "model.glb").is_file():
        return FileResponse(
            str(active_dir / "model.glb"),
            media_type="model/gltf-binary",
            filename=f"{structure_id}_model.glb",
        )

    # Generate procedural model fallback
    rt = app_state.pluggable_registry.get(structure_id) or app_state.structure_registry.get(structure_id)
    stype = getattr(rt.config, "structure_type", "bridge") if rt else "bridge"

    procedural_dir = app_state.model_job_manager.base_storage_dir / structure_id / "procedural"
    procedural_glb = procedural_dir / "model.glb"
    parts_json = procedural_dir / "parts.json"

    if not procedural_glb.is_file():
        procedural_dir.mkdir(parents=True, exist_ok=True)
        raw_proc = procedural_dir / "raw.glb"
        ProceduralModelGenerator.generate_for_structure(stype, {}, raw_proc)
        app_state.model_job_manager.optimizer.optimize(raw_proc, procedural_glb, parts_json)
        if raw_proc.exists():
            raw_proc.unlink()

    return FileResponse(
        str(procedural_glb),
        media_type="model/gltf-binary",
        filename=f"{structure_id}_model.glb",
    )


@app.get("/api/v1/structures/{structure_id}/model/parts", summary="Get 3D Model Parts & Validation Report")
async def get_structure_model_parts(structure_id: str) -> Dict[str, Any]:
    """Returns semantic part breakdown and validation report for the structure."""
    active_dir = app_state.model_job_manager.get_active_model_dir(structure_id)
    if not active_dir:
        # Generate procedural if needed
        await get_structure_model(structure_id)
        active_dir = app_state.model_job_manager.base_storage_dir / structure_id / "procedural"

    parts_file = active_dir / "parts.json"
    report_file = active_dir / "validation_report.json"

    parts_data = []
    if parts_file.is_file():
        try:
            parts_data = json.loads(parts_file.read_text())
        except Exception:
            pass

    report_data = {}
    if report_file.is_file():
        try:
            report_data = json.loads(report_file.read_text())
        except Exception:
            pass

    return {
        "structure_id": structure_id,
        "parts": parts_data,
        "validation_report": report_data,
    }


@app.post("/api/v1/structures/{structure_id}/sensors/{sensor_id}/position", summary="Update Sensor 3D Position")
async def update_sensor_position(
    structure_id: str,
    sensor_id: str,
    req: SensorPositionUpdateRequest,
) -> Dict[str, Any]:
    """Update sensor 3D placement coordinates and assigned 3D mesh part name."""
    rt_pluggable = app_state.pluggable_registry.get(structure_id)
    updated = False

    if rt_pluggable:
        for s in rt_pluggable.config.sensors:
            if s.id == sensor_id:
                s.position_xyz = tuple(req.position_xyz)
                if req.mesh_name:
                    s.mesh_name = req.mesh_name
                updated = True
                break

    rt_legacy = app_state.structure_registry.get(structure_id)
    if rt_legacy:
        for s in rt_legacy.config.sensors:
            if s.id == sensor_id:
                s.position_xyz = tuple(req.position_xyz)
                updated = True
                break

    if not updated:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found in structure '{structure_id}'.")

    # Persist change back to configs/<structure_id>.yaml if it exists
    cfg_file = Path("configs") / f"{structure_id}.yaml"
    if not cfg_file.is_file():
        for candidate in Path("configs").glob("*.yaml"):
            try:
                import yaml
                data = yaml.safe_load(candidate.read_text())
                if data and data.get("structure_id") == structure_id:
                    cfg_file = candidate
                    break
            except Exception:
                pass

    if cfg_file.is_file() and rt_pluggable:
        try:
            import yaml
            # Use model_dump_json() then json.loads() to ensure clean standard Python types without !!python/tuple
            cfg_dict = json.loads(rt_pluggable.config.model_dump_json())
            cfg_file.write_text(yaml.dump(cfg_dict, sort_keys=False))
        except Exception as exc:
            logger.warning(f"Could not persist sensor position to {cfg_file}: {exc}")

    return {
        "status": "success",
        "structure_id": structure_id,
        "sensor_id": sensor_id,
        "position_xyz": req.position_xyz,
        "mesh_name": req.mesh_name,
    }


@app.post("/api/v1/structures", status_code=201, summary="Register New Structure Runtime")
async def create_structure(req: StructureCreateRequest) -> Dict[str, Any]:
    """Dynamically register a new structure runtime with configured ingestion source."""
    if app_state.structure_registry.get(req.structure_id) is not None:
        raise HTTPException(status_code=400, detail=f"Structure '{req.structure_id}' is already registered.")

    # Validate structure type
    try:
        st_enum = StructureType[req.structure_type.upper()]
    except KeyError:
        try:
            st_enum = StructureType(req.structure_type.lower())
        except ValueError:
            st_enum = StructureType.BRIDGE

    # Build sensors list
    sensors = []
    for idx, s in enumerate(req.sensors):
        try:
            stype = SensorType[s.sensor_type.upper()]
        except KeyError:
            try:
                stype = SensorType(s.sensor_type.lower())
            except ValueError:
                stype = SensorType.ACCELEROMETER

        coords = tuple(s.coordinates_3d) if len(s.coordinates_3d) == 3 else (0.0, 0.0, 0.0)
        unit = "°C" if stype == SensorType.TEMPERATURE else "m/s²"
        sensors.append(
            SensorDefinition(
                id=f"{req.structure_id}_s{idx+1}",
                name=s.channel_name,
                type=stype,
                unit=unit,
                position_xyz=coords,
                fs=req.fs,
                channel_index=idx,
                description=f"Auto-registered sensor {s.channel_name}",
            )
        )

    config = StructureConfig(
        structure_id=req.structure_id,
        name=req.name,
        structure_type=st_enum,
        fs=req.fs,
        sensors=sensors,
    )

    # Create ingestion source via IngestionFactory
    try:
        source = IngestionFactory.create_source(
            source_type=req.ingestion.type,
            params=req.ingestion.params,
            channel_names=config.channel_names,
            fs=config.fs,
            structure_id=config.structure_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to create ingestion source: {exc}")

    runtime = app_state.structure_registry.register_structure(
        config=config,
        ingestion_source=source,
        broadcast_fn=broadcast_telemetry,
        device=app_state.device,
        alert_repository=app_state.alert_repo,
    )

    await runtime.start()

    return {
        "status": "created",
        "structure_id": config.structure_id,
        "name": config.name,
        "structure_type": config.structure_type.value,
        "num_channels": config.num_channels,
        "channels": config.channel_names,
        "ingestion_source": source.name,
        "is_running": runtime.ingestion_source.is_running,
    }


@app.delete("/api/v1/structures/{structure_id}", response_model=StructureActionResponse, summary="Unregister Structure")
async def delete_structure(structure_id: str) -> StructureActionResponse:
    """Stops ingestion and unregisters a structure from the runtime registry."""
    if structure_id == "bridge_demo_01":
        raise HTTPException(status_code=400, detail="Cannot delete default demo structure 'bridge_demo_01'.")

    ok = app_state.structure_registry.unregister_structure(structure_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")

    return StructureActionResponse(
        status="success",
        structure_id=structure_id,
        message=f"Structure '{structure_id}' unmounted and deleted successfully.",
    )


@app.post("/api/v1/structures/{structure_id}/start", response_model=StructureActionResponse, summary="Start Ingestion Stream")
async def start_structure_ingestion(structure_id: str) -> StructureActionResponse:
    """Resumes or starts telemetry ingestion for a structure."""
    rt = app_state.structure_registry.get(structure_id)
    if not rt:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")
    await rt.start()
    return StructureActionResponse(
        status="success",
        structure_id=structure_id,
        message=f"Ingestion started for '{structure_id}'.",
    )


@app.post("/api/v1/structures/{structure_id}/stop", response_model=StructureActionResponse, summary="Stop Ingestion Stream")
async def stop_structure_ingestion(structure_id: str) -> StructureActionResponse:
    """Pauses or halts telemetry ingestion for a structure."""
    rt = app_state.structure_registry.get(structure_id)
    if not rt:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")
    rt.stop()
    return StructureActionResponse(
        status="success",
        structure_id=structure_id,
        message=f"Ingestion stopped for '{structure_id}'.",
    )


@app.post("/api/v1/structures/{structure_id}/telemetry", summary="Push Telemetry Frame (HTTP)")
async def push_telemetry(structure_id: str, req: TelemetryPushRequest) -> Dict[str, Any]:
    """HTTP push ingestion endpoint for edge DAQs and field gateways."""
    p_rt = app_state.pluggable_registry.get(structure_id)
    if p_rt is not None:
        payload = req.frame or req.values or req.channels
        if payload is None:
            raise HTTPException(status_code=400, detail="Missing telemetry frame, values, or channels in request.")
        if isinstance(payload, dict):
            vals = [float(payload[ch]) for ch in p_rt.config.channel_names if ch in payload]
            if len(vals) != p_rt.config.num_channels:
                raise HTTPException(status_code=400, detail=f"Payload missing required channels: {p_rt.config.channel_names}")
            arr = np.array(vals, dtype=np.float32)
        elif isinstance(payload, list):
            if len(payload) != p_rt.config.num_channels:
                raise HTTPException(status_code=400, detail=f"Expected {p_rt.config.num_channels} values, got {len(payload)}")
            arr = np.array(payload, dtype=np.float32)
        else:
            raise HTTPException(status_code=400, detail="Invalid payload format.")

        p_rt.push_frame(arr)
        return {"status": "accepted", "structure_id": structure_id}

    rt = app_state.structure_registry.get(structure_id)
    if not rt:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")

    payload = req.frame or req.values or req.channels
    if payload is None:
        raise HTTPException(status_code=400, detail="Missing telemetry frame, values, or channels in request.")

    if isinstance(rt.ingestion_source, HTTPSource):
        pushed = rt.ingestion_source.push_frame(payload)
        if not pushed:
            raise HTTPException(status_code=400, detail="Failed to ingest frame: channel mismatch or invalid shape.")
    else:
        if isinstance(payload, dict):
            vals = [float(payload[ch]) for ch in rt.config.channel_names if ch in payload]
            if len(vals) != rt.config.num_channels:
                raise HTTPException(status_code=400, detail=f"Payload missing required channels: {rt.config.channel_names}")
            arr = np.array(vals, dtype=np.float32)
        elif isinstance(payload, list):
            if len(payload) != rt.config.num_channels:
                raise HTTPException(status_code=400, detail=f"Expected {rt.config.num_channels} values, got {len(payload)}")
            arr = np.array(payload, dtype=np.float32)
        else:
            raise HTTPException(status_code=400, detail="Invalid payload format.")

        event = rt.push_frame(arr)
        if event and rt.broadcast_fn:
            formatted = TelemetryFormatter.format(event, structure_id=structure_id)
            asyncio.create_task(rt.broadcast_fn(formatted))

    return {"status": "accepted", "structure_id": structure_id}


def _get_or_create_http_source(structure_id: str, p_rt: Any) -> HTTPSource:
    src = app_state.http_ingest_sources.get(structure_id)
    if src is None:
        src = HTTPSource(
            channel_names=p_rt.config.channel_names,
            structure_id=structure_id,
            fs=p_rt.config.effective_detector_fs,
            queue_maxsize=p_rt.config.ingestion.max_queue_size,
            rate_limit_rps=settings.HTTP_RATE_LIMIT_RPS,
            idempotency_ttl_seconds=settings.IDEMPOTENCY_TTL_SECONDS,
        )
        app_state.http_ingest_sources[structure_id] = src
    return src


@app.post("/api/v1/structures/{structure_id}/frames", summary="Ingest Batch Sensor Frames (HTTP Binary or JSON)")
async def ingest_frames_http(
    structure_id: str,
    request: Request,
) -> Dict[str, Any]:
    """Authenticated, idempotent, size- and rate-limited batch frame endpoint."""
    p_rt = app_state.pluggable_registry.get(structure_id)
    if p_rt is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not registered in pluggable registry.")

    x_api_key = request.headers.get("x-api-key")
    idempotency_key = request.headers.get("idempotency-key")
    expected_key = settings.get_structure_api_key(structure_id)
    if not expected_key or not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), expected_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")

    raw_body = await request.body()
    if len(raw_body) > settings.MAX_FRAME_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Payload {len(raw_body)} bytes exceeds limit {settings.MAX_FRAME_PAYLOAD_BYTES} bytes.",
        )

    http_src = _get_or_create_http_source(structure_id, p_rt)
    if not http_src.check_rate_limit():
        raise HTTPException(status_code=429, detail="HTTP ingestion rate limit exceeded.")

    content_type = request.headers.get("content-type", "").lower()
    frames_to_ingest: List[Frame] = []
    try:
        if "octet-stream" in content_type or raw_body[:4] == b"SHMF":
            frame = decode_frame_binary(
                raw_body,
                max_bytes=settings.MAX_FRAME_PAYLOAD_BYTES,
            )
            if frame.structure_id and frame.structure_id != structure_id:
                raise ValueError(
                    f"Frame structure_id {frame.structure_id!r} != endpoint {structure_id!r}"
                )
            frame.structure_id = structure_id
            if not frame.source_id or frame.source_id == "unknown":
                frame.source_id = "http_binary"
            frames_to_ingest.append(frame)
        else:
            payload = json.loads(raw_body.decode("utf-8"))
            raw_list = payload.get("frames") if isinstance(payload, dict) and "frames" in payload else [payload]
            for idx, item in enumerate(raw_list):
                arr = np.asarray(item["samples"], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                frames_to_ingest.append(
                    Frame(
                        structure_id=item.get("structure_id", structure_id),
                        sensor_ids=list(item.get("sensor_ids") or p_rt.config.channel_names[: arr.shape[1]]),
                        t0=int(item["t0"]),
                        fs=float(item.get("fs", p_rt.config.effective_native_fs)),
                        samples=arr,
                        seq=int(item.get("seq", idx)),
                        source_id=str(item.get("source_id", "http_json")),
                        flags=dict(item.get("flags") or {}),
                    )
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Malformed frame payload: {exc}")

    # Idempotency check & synchronous processing through Stage 4 pipeline
    accepted_count = 0
    duplicate_count = 0
    latest_telemetry: Optional[Dict[str, Any]] = None
    for idx, frame in enumerate(frames_to_ingest):
        item_key = f"{idempotency_key}:{idx}" if (idempotency_key and len(frames_to_ingest) > 1) else idempotency_key
        if item_key and http_src.is_duplicate_key(item_key):
            duplicate_count += 1
            continue
        latest_telemetry = p_rt.push_ingestion_item(frame, spool_write=False)
        accepted_count += 1

    return {
        "status": "duplicate_ignored" if (accepted_count == 0 and duplicate_count > 0) else "accepted",
        "structure_id": structure_id,
        "frames_accepted": accepted_count,
        "duplicates_ignored": duplicate_count,
        "combined_state": latest_telemetry.get("combined_state") if latest_telemetry else None,
    }


@app.get("/api/v1/structures/{structure_id}/ingestion", summary="Get Ingestion Throughput, Lag & Quality Summary")
async def get_structure_ingestion_health(structure_id: str) -> Dict[str, Any]:
    p_rt = app_state.pluggable_registry.get(structure_id)
    if p_rt is not None:
        return p_rt.get_ingestion_health()

    rt = app_state.structure_registry.get(structure_id)
    if rt is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")
    return {
        "structure_id": structure_id,
        "source_type": rt.ingestion_source.__class__.__name__ if rt.ingestion_source else "unknown",
        "source_status": "UP" if rt.is_running else "IDLE",
        "frames_processed": rt.inference_count,
    }


@app.get("/api/v1/structures/{structure_id}/quality/events", summary="Get Isolated Data-Quality Events")
async def get_structure_quality_events(
    structure_id: str,
    limit: int = Query(100, ge=1, le=1000),
    unresolved_only: bool = Query(False),
) -> Dict[str, Any]:
    p_rt = app_state.pluggable_registry.get(structure_id)
    if p_rt is None:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not registered in pluggable registry.")
    events = [e.model_dump() for e in p_rt.quality_layer.get_events(limit=limit, unresolved_only=unresolved_only)]
    return {
        "structure_id": structure_id,
        "count": len(events),
        "events": events,
        "quality": p_rt.last_quality_report.model_dump() if p_rt.last_quality_report else None,
    }


@app.post("/api/v1/structures/{structure_id}/simulate-fault", response_model=FaultInjectionResponse)
async def simulate_fault_for_structure(structure_id: str, req: FaultInjectionRequest) -> FaultInjectionResponse:
    """Injects fault on a specific structure's sensor channel with automatic preset mapping."""
    rt = app_state.structure_registry.get(structure_id)
    if not rt:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")

    if req.fault_type.lower() == "reset":
        if isinstance(rt.ingestion_source, SimulatorSource):
            rt.ingestion_source.clear_faults()
        rt.state_machine.reset()
        return FaultInjectionResponse(
            status="success",
            message=f"All faults reset to baseline for '{structure_id}'.",
            structure_id=structure_id,
            node_id=req.node_id or (rt.config.channel_names[0] if rt.config.channel_names else "all"),
            severity=0.0,
        )

    # Normalize frontend preset names
    fault_map = {
        "structural_degradation": ("Sensor_MidSpan_Girder", "stiffness_loss", 0.65),
        "foundation_settlement":  ("Sensor_Abutment_Left",  "settlement",     0.50),
        "dynamic_impulse":        ("Sensor_MidSpan_Girder", "spike",          0.85),
    }

    if req.fault_type in fault_map:
        default_node, actual_fault, default_sev = fault_map[req.fault_type]
        target_node = req.node_id if (req.node_id and req.node_id in rt.config.channel_names) else default_node
        actual_fault_type = actual_fault
        actual_sev = req.severity if req.severity != 0.5 else default_sev
    else:
        target_node = req.node_id or "Sensor_MidSpan_Girder"
        actual_fault_type = req.fault_type
        actual_sev = req.severity

    if target_node not in rt.config.channel_names:
        target_node = rt.config.channel_names[0]

    if not isinstance(rt.ingestion_source, SimulatorSource):
        raise HTTPException(status_code=400, detail="Fault injection is only supported on Simulator sources.")

    rt.ingestion_source.inject_fault(node_id=target_node, fault_type=actual_fault_type, severity=actual_sev)
    return FaultInjectionResponse(
        status="success",
        message=f"Fault '{actual_fault_type}' injected on {structure_id}:{target_node}",
        structure_id=structure_id,
        node_id=target_node,
        severity=actual_sev,
    )


@app.post("/api/v1/structures/{structure_id}/reset-faults")
async def reset_faults_for_structure(structure_id: str) -> Dict[str, str]:
    """Clears all injected faults on a specific structure."""
    rt = app_state.structure_registry.get(structure_id)
    if not rt:
        raise HTTPException(status_code=404, detail=f"Structure '{structure_id}' not found.")

    if isinstance(rt.ingestion_source, SimulatorSource):
        rt.ingestion_source.clear_faults()
    rt.state_machine.reset()
    return {"status": "success", "message": f"Faults cleared for structure '{structure_id}'."}


# Backwards-compatible legacy fault routes
@app.post("/api/v1/simulate-fault", response_model=FaultInjectionResponse)
async def legacy_simulate_fault(req: FaultInjectionRequest) -> FaultInjectionResponse:
    return await simulate_fault_for_structure("bridge_demo_01", req)


@app.post("/api/v1/reset-faults")
async def legacy_reset_faults() -> Dict[str, str]:
    return await reset_faults_for_structure("bridge_demo_01")


# ── Alert API ────────────────────────────────────────────────────────────────────

@app.get("/api/v1/alerts", response_model=AlertListResponse, summary="List Anomaly Alerts")
async def list_alerts(
    structure_id: Optional[str] = Query(None, description="Filter by structure ID"),
    status: Optional[str] = Query(None, description="Filter by status: WARNING | CRITICAL"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgement state"),
    limit: int = Query(50, ge=1, le=200, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> AlertListResponse:
    """Returns paginated list of alert records with optional filters."""
    records, total = app_state.alert_repo.get_alerts(
        structure_id=structure_id,
        status=status,
        acknowledged=acknowledged,
        limit=limit,
        offset=offset,
    )
    return AlertListResponse(
        alerts=[AlertRecordResponse(**r.__dict__) for r in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/alerts/stats", response_model=List[AlertStatsResponse], summary="Alert Summary Stats")
async def get_alert_stats(
    structure_id: Optional[str] = Query(None, description="Filter by structure ID"),
) -> List[AlertStatsResponse]:
    """Returns alert count breakdown (total, critical, warning, acknowledged) per structure."""
    stats = app_state.alert_repo.get_stats(structure_id=structure_id)
    return [AlertStatsResponse(**s.__dict__) for s in stats]


@app.get("/api/v1/alerts/{alert_id}", response_model=AlertDetailResponse, summary="Get Alert Detail")
async def get_alert_detail(alert_id: int) -> AlertDetailResponse:
    """Returns full alert detail including per-sensor node snapshot."""
    detail = app_state.alert_repo.get_alert(alert_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Alert #{alert_id} not found.")
    return AlertDetailResponse(**detail.__dict__)


@app.post("/api/v1/alerts/{alert_id}/acknowledge", summary="Acknowledge Alert")
async def acknowledge_alert(alert_id: int) -> Dict[str, Any]:
    """Marks an alert as acknowledged (operator reviewed)."""
    ok = app_state.alert_repo.acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Alert #{alert_id} not found.")
    return {"status": "success", "alert_id": alert_id, "acknowledged": True}



# Multi-Structure & Legacy WebSocket Routes
@app.websocket("/ws/{structure_id}/telemetry")
async def websocket_structure_telemetry_standard(websocket: WebSocket, structure_id: str):
    """Standardized WebSocket telemetry route /ws/{structure_id}/telemetry."""
    await websocket_structure_telemetry(websocket, structure_id=structure_id)


@app.websocket("/ws/telemetry/{structure_id}")
async def websocket_structure_telemetry(websocket: WebSocket, structure_id: str):
    """WebSocket streaming real-time JSON telemetry payloads for a given structure."""
    rt = app_state.structure_registry.get(structure_id) or app_state.pluggable_registry.get(structure_id)
    if rt is None:
        await websocket.close(code=1008)
        return

    client_id = await app_state.conn_manager.connect(websocket, structure_id=structure_id)
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected cleanly from {structure_id}.")
    except Exception as err:
        logger.warning(f"WebSocket client loop exception ({client_id}): {err}")
    finally:
        await app_state.conn_manager.disconnect(client_id, structure_id=structure_id)


@app.websocket("/ws/telemetry")
async def legacy_websocket_telemetry(websocket: WebSocket):
    """Backwards-compatible legacy route defaulting to bridge_demo_01."""
    await websocket_structure_telemetry(websocket, structure_id="bridge_demo_01")
