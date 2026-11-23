"""Test suite for STAGE 3B: CAD-to-3D Model Pipeline.

Covers:
1. File type detection & rejection of 2D/proprietary files with clear instructions.
2. Parametric procedural fallbacks (girder bridge, building, tower) with semantic node naming.
3. Mesh optimization (unit scaling, ground-level origin, tiny part pruning, gltf-transform welding).
4. Converters: IFC, STEP (FreeCAD headless), Blender (OBJ/FBX headless), GLB passthrough.
5. Background job execution, versioned disk storage, and REST API endpoints.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
import trimesh

from app.main import app, app_state
from app.models3d.converters import ConverterRegistry
from app.models3d.converters.blend_converter import BlenderConverter
from app.models3d.converters.ifc_converter import IFCConverter
from app.models3d.converters.passthrough_converter import PassthroughConverter
from app.models3d.converters.step_converter import STEPConverter
from app.models3d.detector import FileTypeDetector
from app.models3d.job_manager import ModelJobManager
from app.models3d.optimizer import ModelOptimizer
from app.models3d.procedural import ProceduralModelGenerator
from app.models3d.schemas import JobStatus, SupportedCADFormat


# ── 1. File Type Detection & Rejection ──────────────────────────────────────────

def test_file_type_detector_supported_formats(tmp_path: Path):
    detector = FileTypeDetector(max_upload_size_bytes=10 * 1024 * 1024)

    # GLB
    p_glb = tmp_path / "model.glb"
    p_glb.write_bytes(b"glTF\x02\x00\x00\x00" + b"\x00" * 20)
    fmt, _ = detector.inspect_file(p_glb)
    assert fmt == SupportedCADFormat.GLB

    # IFC
    p_ifc = tmp_path / "structure.ifc"
    p_ifc.write_text("ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');\n")
    fmt, _ = detector.inspect_file(p_ifc)
    assert fmt == SupportedCADFormat.IFC

    # STEP
    p_stp = tmp_path / "beam.stp"
    p_stp.write_text("ISO-10303-21;\nHEADER;\n")
    fmt, _ = detector.inspect_file(p_stp)
    assert fmt == SupportedCADFormat.STEP

    # OBJ
    p_obj = tmp_path / "mesh.obj"
    p_obj.write_text("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n")
    fmt, _ = detector.inspect_file(p_obj)
    assert fmt == SupportedCADFormat.OBJ


def test_file_type_detector_rejects_2d_and_proprietary(tmp_path: Path):
    detector = FileTypeDetector()

    # 2D DWG
    p_dwg = tmp_path / "plan.dwg"
    p_dwg.write_bytes(b"AC1032" + b"\x00" * 50)
    fmt, msg = detector.inspect_file(p_dwg)
    assert fmt == SupportedCADFormat.UNSUPPORTED_2D
    assert "2D DWG/DXF drawings are not 3D BIM/CAD models" in msg
    assert "export your 3D model as IFC" in msg

    # Revit RVT
    p_rvt = tmp_path / "hospital.rvt"
    p_rvt.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 50)
    fmt, msg = detector.inspect_file(p_rvt)
    assert fmt == SupportedCADFormat.UNSUPPORTED_PROPRIETARY
    assert "Autodesk Revit (.RVT)" in msg
    assert "Export -> IFC" in msg

    # SolidWorks SLDPRT
    p_sld = tmp_path / "bracket.sldprt"
    p_sld.write_bytes(b"test" * 10)
    fmt, msg = detector.inspect_file(p_sld)
    assert fmt == SupportedCADFormat.UNSUPPORTED_PROPRIETARY
    assert "SolidWorks" in msg
    assert "Save As -> STEP" in msg


def test_file_type_detector_size_limits(tmp_path: Path):
    detector = FileTypeDetector(max_upload_size_bytes=1024)

    # Empty
    empty_file = tmp_path / "empty.glb"
    empty_file.write_bytes(b"")
    fmt, msg = detector.inspect_file(empty_file)
    assert fmt == SupportedCADFormat.UNKNOWN
    assert "empty" in msg.lower()

    # Oversized
    huge_file = tmp_path / "huge.glb"
    huge_file.write_bytes(b"glTF" + b"\x00" * 2000)
    fmt, msg = detector.inspect_file(huge_file)
    assert fmt == SupportedCADFormat.UNKNOWN
    assert "exceeds maximum upload limit" in msg


# ── 2. Procedural Fallback Generators ──────────────────────────────────────────

def test_procedural_bridge_generator(tmp_path: Path):
    scene = ProceduralModelGenerator.generate_girder_bridge(
        span_meters=40.0,
        width_meters=8.0,
        pier_count=2,
        pier_height=6.0,
    )
    geom_names = list(scene.geometry.keys())
    assert "Deck" in geom_names
    assert "Abutment_Left" in geom_names
    assert "Abutment_Right" in geom_names
    assert "Pier_1" in geom_names
    assert "Pier_2" in geom_names

    glb_path = tmp_path / "procedural_bridge.glb"
    ProceduralModelGenerator.generate_for_structure(
        structure_type="bridge",
        params={"span_meters": 40.0},
        output_glb_path=glb_path,
    )
    assert glb_path.is_file()
    assert glb_path.stat().st_size > 0


def test_procedural_building_and_tower(tmp_path: Path):
    bldg = ProceduralModelGenerator.generate_building(floors=3, bays_x=2, bays_z=2)
    bldg_names = list(bldg.geometry.keys())
    assert "Floor_1" in bldg_names
    assert "Floor_2" in bldg_names
    assert "Floor_3" in bldg_names
    assert "Roof" in bldg_names
    assert "Foundation" in bldg_names

    tower = ProceduralModelGenerator.generate_tower(height_meters=24.0, sections=3)
    tower_names = list(tower.geometry.keys())
    assert "Observation_Platform" in tower_names
    assert "Section_1" in tower_names


# ── 3. Model Optimizer & Audit ────────────────────────────────────────────────

def test_model_optimizer_ground_level_and_pruning(tmp_path: Path):
    # Create scene with 1 large box (span) and 2 tiny screws (< 0.05m)
    scene = trimesh.Scene()
    deck = trimesh.creation.box(extents=[20.0, 1.0, 4.0])
    # Place deck at y = 5.0 to 6.0
    deck.apply_translation([0.0, 5.5, 0.0])
    scene.add_geometry(deck, node_name="Deck_Span")

    # Tiny screws (diameter 0.02m)
    screw1 = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    screw1.apply_translation([1.0, 6.0, 0.0])
    scene.add_geometry(screw1, node_name="Screw_01")

    raw_glb = tmp_path / "raw.glb"
    raw_glb.write_bytes(scene.export(file_type="glb"))

    opt = ModelOptimizer(min_part_size_meters=0.05)
    opt_glb = tmp_path / "optimized.glb"
    parts_json = tmp_path / "parts.json"

    parts, report, logs = opt.optimize(raw_glb, opt_glb, parts_json)

    assert report.is_valid
    assert opt_glb.is_file()
    assert parts_json.is_file()

    # Check ground-level alignment: minimum Y should be at or very close to 0.0
    dims = report.bounding_box_meters["dimensions"]
    min_pt = report.bounding_box_meters["min"]
    assert abs(min_pt[1]) < 1e-3, f"Ground level Y should be 0.0, got {min_pt[1]}"

    # Verify tiny screw was pruned
    mesh_names = [p.mesh_name for p in parts]
    assert "Screw_01" not in mesh_names
    assert any("Pruned 1 tiny parts" in log for log in logs)


def test_model_optimizer_millimeter_auto_scale(tmp_path: Path):
    # Create a box in millimeters: 30000 x 2000 x 8000 mm
    box = trimesh.creation.box(extents=[30000.0, 2000.0, 8000.0])
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="Bridge_In_MM")

    raw_glb = tmp_path / "mm_bridge.glb"
    raw_glb.write_bytes(scene.export(file_type="glb"))

    opt = ModelOptimizer()
    opt_glb = tmp_path / "m_bridge.glb"
    parts_json = tmp_path / "parts.json"

    parts, report, logs = opt.optimize(raw_glb, opt_glb, parts_json)

    assert report.is_valid
    # Bounding box should now be in meters (30m max extent, not 30000m)
    max_extent = max(report.bounding_box_meters["dimensions"])
    assert 25.0 <= max_extent <= 35.0, f"Expected ~30m extent, got {max_extent}"
    assert any("AUTO_SCALE_MM_TO_M" == w.code for w in report.warnings)


# ── 4. Converters Test ────────────────────────────────────────────────────────

def test_ifc_converter_execution(tmp_path: Path):
    ifc_path = Path("backend/tests/fixtures/sample_bridge.ifc")
    if not ifc_path.is_file():
        pytest.skip("sample_bridge.ifc fixture not present")

    conv = IFCConverter()
    out_glb = tmp_path / "ifc_converted.glb"
    res = conv.convert(ifc_path, out_glb)

    assert res.success
    assert out_glb.is_file()
    assert out_glb.stat().st_size > 0
    assert res.duration_seconds < 10.0


def test_step_converter_execution(tmp_path: Path):
    step_path = Path("backend/tests/fixtures/sample_box.step")
    if not step_path.is_file():
        pytest.skip("sample_box.step fixture not present")

    conv = STEPConverter()
    avail, _ = conv.is_available()
    if not avail:
        pytest.skip("FreeCADCmd not found on system")

    out_glb = tmp_path / "step_converted.glb"
    res = conv.convert(step_path, out_glb)

    assert res.success
    assert out_glb.is_file()
    assert out_glb.stat().st_size > 0


def test_blender_converter_execution(tmp_path: Path):
    obj_path = Path("backend/tests/fixtures/test_beam.obj")
    if not obj_path.is_file():
        # Create test OBJ
        box = trimesh.creation.box(extents=[10, 2, 2])
        box.export(str(obj_path))

    conv = BlenderConverter()
    avail, _ = conv.is_available()
    if not avail:
        pytest.skip("Blender not found on system")

    out_glb = tmp_path / "blender_converted.glb"
    res = conv.convert(obj_path, out_glb)

    assert res.success
    assert out_glb.is_file()
    assert out_glb.stat().st_size > 0


# ── 5. Background Job Manager & Versioning ────────────────────────────────────

def test_model_job_manager_lifecycle(tmp_path: Path):
    storage_dir = tmp_path / "test_models"
    mgr = ModelJobManager(base_storage_dir=storage_dir)

    # Create dummy GLB
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=[10, 1, 2]), node_name="Girder")
    glb_data = scene.export(file_type="glb")

    upload_file = tmp_path / "test_upload.glb"
    upload_file.write_bytes(glb_data)

    job = mgr.create_job(
        structure_id="test_struct_01",
        source_filename="test_upload.glb",
        temp_file_path=upload_file,
    )
    assert job.status in (JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.OPTIMIZING, JobStatus.COMPLETED)

    # Wait for completion (max 15s)
    for _ in range(30):
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.5)

    assert job.status == JobStatus.COMPLETED
    assert job.validation_report is not None
    assert job.validation_report.is_valid

    # Check directory structure: test_models/test_struct_01/v1/model.glb
    active_dir = mgr.get_active_model_dir("test_struct_01")
    assert active_dir is not None
    assert (active_dir / "model.glb").is_file()
    assert (active_dir / "parts.json").is_file()
    assert (active_dir / "validation_report.json").is_file()


# ── 6. FastAPI Endpoints ──────────────────────────────────────────────────────

def test_api_structure_model_routes():
    with TestClient(app) as client:
        # 1. GET procedural model for kw51
        res = client.get("/api/v1/structures/kw51/model")
        assert res.status_code == 200
        assert res.headers["content-type"] in ("model/gltf-binary", "application/octet-stream")
        assert len(res.content) > 0

        # 2. GET parts and validation report
        res_parts = client.get("/api/v1/structures/kw51/model/parts")
        assert res_parts.status_code == 200
        data = res_parts.json()
        assert data["structure_id"] == "kw51"
        assert "parts" in data
        assert "validation_report" in data

        # 3. POST sensor position update
        update_payload = {
            "position_xyz": [12.5, 3.2, -1.0],
            "mesh_name": "Deck",
        }
        res_pos = client.post(
            "/api/v1/structures/kw51/sensors/aBD17Ay/position",
            json=update_payload,
        )
        assert res_pos.status_code == 200
        pos_data = res_pos.json()
        assert pos_data["status"] == "success"
        assert pos_data["position_xyz"] == [12.5, 3.2, -1.0]
        assert pos_data["mesh_name"] == "Deck"

        # 4. Upload CAD model via POST
        box_glb = trimesh.creation.box(extents=[15, 2, 3]).export(file_type="glb")
        files = {"file": ("uploaded_span.glb", box_glb, "model/gltf-binary")}
        res_upload = client.post("/api/v1/structures/kw51/model", files=files)
        assert res_upload.status_code == 200
        upload_data = res_upload.json()
        assert "job_id" in upload_data

        # 5. Poll status
        job_id = upload_data["job_id"]
        time.sleep(1.0)
        res_status = client.get(f"/api/v1/structures/kw51/model/status/{job_id}")
        assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ("PROCESSING", "OPTIMIZING", "COMPLETED")
