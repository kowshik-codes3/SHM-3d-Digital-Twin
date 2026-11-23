# CAD-to-3D Model Pipeline & Digital Twin Geometry Guide (Stage 3B)

StructureIQ enables customers to provide real-world architectural and civil engineering CAD/BIM files and automatically obtain a lightweight, WebGL-optimized 3D model (`model.glb`) with semantically named parts (`parts.json`), dimensional validation audit (`validation_report.json`), and interactive raycast sensor placement.

---

## 1. Supported Input Formats

| Format | Extension | Engine / Subprocess | Semantic Part Preservation | Optimization Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **IFC** *(Preferred)* | `.ifc` | `IfcConvert 0.8.5` + `ifcopenshell.geom` | Yes (`IfcBeam`, `IfcColumn`, `IfcSlab`, `IfcFooting`) | Prune screws, weld vertices, simplify to 200k budget |
| **STEP / STP** | `.step`, `.stp` | `FreeCAD 1.1.4` (headless `FreeCADCmd`) | B-Rep / Solid entities | Deflection tessellation (0.02) $\to$ Mesh welding |
| **IGES** | `.iges`, `.igs` | `FreeCAD 1.1.4` (headless `FreeCADCmd`) | Surface entities | OpenCASCADE surface tessellation $\to$ GLB |
| **GLB / glTF** | `.glb`, `.gltf` | Pass-through + `@gltf-transform` | Preserves source node tree | Accessor deduplication, vertex welding, simplification |
| **OBJ / FBX** | `.obj`, `.fbx` | `Blender 5.2.3` (headless background script) | Preserves mesh objects | Blender C++ OBJ importer $\to$ glTF exporter |

### Unsupported Formats & Clear Rejection Guidance
- **2D Drawings (`.dwg`, `.dxf`)**:
  - *Rejection Reason*: 2D vector drawings lack 3D spatial surfaces needed for digital twin simulation.
  - *User Guidance*: In AutoCAD / Civil 3D, run `IFCEXPORT` to export 3D geometry as IFC4 or STEP.
- **Proprietary Binaries (`.rvt`, `.sldprt`, `.sldasm`, `.nwd`)**:
  - *Rejection Reason*: Closed binary formats without open specification.
  - *User Guidance*:
    - **Autodesk Revit**: Open `File -> Export -> IFC` (select IFC4 or IFC2x3 Coordination View) and upload `.ifc`.
    - **SolidWorks**: Open `File -> Save As -> STEP (.stp AP214)` and upload.
    - **Navisworks**: Export underlying CAD assets to IFC or FBX.

---

## 2. Automated Pipeline Architecture

```
User CAD Upload (.ifc / .step / .glb / .obj / .fbx)
        │
        ▼
[ FileTypeDetector ] ──── (Magic bytes & upload size limit check)
        │
        ├─► Rejects 2D DWG / Proprietary RVT with export instructions
        │
        ▼
[ Background Job Executor ] (Status: PENDING ➔ PROCESSING ➔ OPTIMIZING ➔ COMPLETED)
        │
        ├─► IFC       ──► IfcConvert 0.8.5 (--use-element-names)
        ├─► STEP/IGES ──► FreeCADCmd 1.1.4 (MeshPart.meshFromShape)
        ├─► OBJ/FBX   ──► Blender 5.2.3 (bpy.ops.wm.obj_import)
        └─► GLB/glTF  ──► Passthrough / Trimesh Pack
        │
        ▼
   Raw GLB Mesh
        │
        ▼
[ ModelOptimizer ]
        ├─► 1. Unit Normalization: Auto-detects mm (extent > 1000) and scales $\times 0.001$ to meters
        ├─► 2. Ground-Level Origin: Sets Y-up and centers bounding box at ground level (min Y = 0.0)
        ├─► 3. Tiny Part Pruning: Removes bolts/screws/rebar below 0.05 m threshold
        ├─► 4. @gltf-transform CLI: Welds vertices, deduplicates accessors, simplifies to $\le 200\text{k}$ triangles
        └─► 5. Part Catalog: Emits parts.json (bounding boxes, triangle counts, semantic names)
        │
        ▼
[ Validation Audit ]
        ├─► Produces validation_report.json
        └─► Enforces hard quality gates (rejects empty scenes or invalid geometry)
        │
        ▼
[ Versioned On-Disk Storage ]
        models/{structure_id}/v{N}/
           ├── model.glb
           ├── parts.json
           ├── validation_report.json
           └── source.<ext>
```

---

## 3. Parametric Fallback Generator

When no CAD file is available, StructureIQ automatically generates procedural models matching the structure type defined in the YAML config:
- **Girder Bridge (`structure_type: bridge`)**: Parametric spans, width, deck thickness, piers (`Pier_1`, `Pier_2`), abutments (`Abutment_Left`, `Abutment_Right`), and parapet barriers.
- **Building (`structure_type: building`)**: Parametric floor slabs (`Floor_1`, `Floor_2`, `Floor_3`), structural columns, roof, and foundation.
- **Tower (`structure_type: tower`)**: Tapered vertical segments (`Section_1`, `Section_2`), observation platform, and foundation pedestal.

---

## 4. REST API Specification

### 1. Upload CAD Model
`POST /api/v1/structures/{structure_id}/model`
- **Request**: Multipart form data with `file` field.
- **Response**:
  ```json
  {
    "job_id": "a1b2c3d4",
    "structure_id": "kw51",
    "status": "PENDING",
    "source_filename": "bridge_model.ifc",
    "source_file_type": "ifc",
    "logs": ["[21:05:00] Job enqueued for structure 'kw51'"]
  }
  ```

### 2. Poll Conversion Status
`GET /api/v1/structures/{structure_id}/model/status/{job_id}`
- **Response**:
  ```json
  {
    "job_id": "a1b2c3d4",
    "status": "COMPLETED",
    "progress_percent": 100.0,
    "logs": [
      "IfcConvert succeeded in 0.40s (1856 bytes)",
      "Pruned 0 tiny parts below 0.050m threshold.",
      "Aligned model origin to ground-level: offset [-5.0, 0.0, 1.0]",
      "Optimization finished: 12 tris, 1 meshes, 0.00 MB"
    ],
    "validation_report": {
      "is_valid": true,
      "file_size_bytes": 1856,
      "file_size_mb": 0.002,
      "triangle_count": 12,
      "mesh_count": 1,
      "bounding_box_meters": {
        "min": [0.0, 0.0, 0.0],
        "max": [10.0, 2.0, 30.0],
        "dimensions": [10.0, 2.0, 30.0]
      },
      "unnamed_meshes_count": 0,
      "warnings": []
    }
  }
  ```

### 3. Get Active GLB Model
`GET /api/v1/structures/{structure_id}/model`
- **Response**: Binary GLB file (`model/gltf-binary`). Serves the latest version, or auto-generates and caches a procedural fallback model if no CAD file has been uploaded.

### 4. Get Parts Catalog & Validation Report
`GET /api/v1/structures/{structure_id}/model/parts`
- **Response**:
  ```json
  {
    "structure_id": "kw51",
    "parts": [
      {
        "mesh_name": "Main_Girder_Span",
        "part_id": "part_1",
        "triangle_count": 12,
        "vertex_count": 24,
        "bounding_box_meters": {
          "min": [0.0, 0.0, 0.0],
          "max": [10.0, 2.0, 30.0],
          "dimensions": [10.0, 2.0, 30.0]
        },
        "center": [5.0, 1.0, 15.0]
      }
    ],
    "validation_report": { ... }
  }
  ```

### 5. Update Sensor Placement
`POST /api/v1/structures/{structure_id}/sensors/{sensor_id}/position`
- **Request**:
  ```json
  {
    "position_xyz": [12.5, 3.2, -1.0],
    "mesh_name": "Main_Girder_Span"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "structure_id": "kw51",
    "sensor_id": "aBD17Ay",
    "position_xyz": [12.5, 3.2, -1.0],
    "mesh_name": "Main_Girder_Span"
  }
  ```

---

## 5. Frontend Interactive Features

1. **CAD Upload Modal (`ModelUploadModal.jsx`)**:
   - Drag-and-drop dropzone with client-side file inspection.
   - Immediate rejection dialogs for 2D DWG and proprietary RVT/SLDPRT files.
   - Live conversion progress bar and console log streaming.
   - Validation report metrics (triangle count, dimensions, warnings).
2. **Dynamic 3D Model Renderer (`StructureGLBModel.jsx`)**:
   - Loads `/api/v1/structures/{id}/model` using Drei's `useGLTF`.
   - Dynamic structural health tinting:
     - Part matching `sensor.mesh_name` or nearest sensor in 3D Euclidean distance.
     - Neutral dark slate (`#374151`) for `NORMAL`.
     - Glowing amber (`#f59e0b`) for `WARNING`.
     - Pulsing red (`#ef4444`) for `CRITICAL`.
     - Sky blue (`#38bdf8`) for `LOW_CONFIDENCE`.
     - Purple (`#c084fc`) for `SENSOR_FAULT`.
3. **Raycast Click-to-Place Sensor Placement (`SensorPlacementEditor.jsx`)**:
   - Click anywhere on the 3D model surface to raycast $(X, Y, Z)$ coordinates.
   - "Snap to Surface" toggle for precision vertex and grid alignment.
   - Automatically binds clicked mesh node name (`mesh_name`) to the selected sensor.
   - Persists coordinates to both localStorage and backend structure YAML config.
   - Proportional sphere sizing: sensor marker radius scales dynamically relative to the structure bounding box diameter ($\sim 2.5\%$ of radius), ensuring proper scale on any structure.
