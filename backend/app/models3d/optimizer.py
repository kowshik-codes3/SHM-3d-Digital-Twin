"""3D Model optimization, unit conversion, tiny part pruning, and part catalog generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from app.models3d.converters.base import run_subprocess_safe
from app.models3d.schemas import PartInfo, ValidationReport, ValidationWarning


class ModelOptimizer:
    """Cleans up, scales, prunes, simplifies and audits 3D models for StructureIQ digital twin."""

    def __init__(
        self,
        triangle_budget: int = 200_000,
        max_file_size_mb: float = 10.0,
        min_part_size_meters: float = 0.05,
    ) -> None:
        self.triangle_budget = triangle_budget
        self.max_file_size_mb = max_file_size_mb
        self.min_part_size_meters = min_part_size_meters

    def optimize(
        self,
        input_glb: Path,
        output_glb: Path,
        parts_json_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[PartInfo], ValidationReport, List[str]]:
        """Run full optimization pipeline.
        
        Returns:
            Tuple of (parts_list, validation_report, logs)
        """
        options = options or {}
        logs: List[str] = [f"Starting model optimization for {input_glb.name}"]
        warnings: List[ValidationWarning] = []
        hard_blockers: List[str] = []

        output_glb.parent.mkdir(parents=True, exist_ok=True)
        parts_json_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load scene with trimesh to inspect geometry
        try:
            loaded = trimesh.load(str(input_glb), force="scene")
            if not isinstance(loaded, trimesh.Scene):
                scene = trimesh.Scene(loaded)
            else:
                scene = loaded
        except Exception as exc:
            hard_blockers.append(f"Failed to parse GLB scene: {exc}")
            report = ValidationReport(
                is_valid=False,
                hard_blockers=hard_blockers,
            )
            return [], report, logs

        if len(scene.geometry) == 0:
            hard_blockers.append("Model scene contains 0 geometric meshes (empty scene).")
            report = ValidationReport(
                is_valid=False,
                hard_blockers=hard_blockers,
            )
            return [], report, logs

        # 2. Compute bounding box and handle unit scaling (millimeters -> meters)
        bbox = scene.bounding_box
        extents = bbox.extents  # [dx, dy, dz]
        max_dim = float(np.max(extents))
        logs.append(f"Initial bounding box extents: {[round(float(x), 3) for x in extents]} (max {max_dim:.2f})")

        scale_factor = 1.0
        # If dimensions are in thousands of units (typical mm export of a bridge e.g. 50000 mm = 50m)
        if max_dim > 1000.0:
            scale_factor = 0.001
            logs.append(f"Auto-detected millimeter units (max dimension {max_dim:.1f} > 1000). Scaling by 0.001 to meters.")
            scene.apply_scale(scale_factor)
            warnings.append(
                ValidationWarning(
                    code="AUTO_SCALE_MM_TO_M",
                    message=f"Model scaled from millimeters to meters (scale factor 0.001, initial extent {max_dim:.1f}).",
                    severity="INFO",
                )
            )
        elif max_dim < 0.2:
            warnings.append(
                ValidationWarning(
                    code="SUSPICIOUS_SMALL_SCALE",
                    message=f"Model bounding box is very small (max dimension {max_dim:.3f} m). Check export unit settings.",
                    severity="WARNING",
                )
            )

        # 3. Filter tiny parts (bolts, rebar, interior details)
        min_size = options.get("min_part_size_meters", self.min_part_size_meters)
        removed_tiny_parts = 0
        preserved_geometries: Dict[str, trimesh.Trimesh] = {}

        for geom_name, geom in list(scene.geometry.items()):
            if not hasattr(geom, "bounding_box"):
                continue
            diag = float(np.linalg.norm(geom.bounding_box.extents))
            # Keep if size exceeds threshold or if it's the only geometry
            if diag >= min_size or len(scene.geometry) <= 1:
                preserved_geometries[geom_name] = geom
            else:
                removed_tiny_parts += 1

        if removed_tiny_parts > 0:
            logs.append(f"Pruned {removed_tiny_parts} tiny parts below {min_size:.3f}m threshold.")
            scene = trimesh.Scene(preserved_geometries)
        else:
            logs.append(f"No tiny parts below {min_size:.3f}m threshold detected.")

        # 4. Enforce ground-level origin (min Y = 0.0) and center X, Z at 0.0
        bbox_bounds = scene.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        min_pt = bbox_bounds[0]
        max_pt = bbox_bounds[1]

        center_x = (min_pt[0] + max_pt[0]) / 2.0
        center_z = (min_pt[2] + max_pt[2]) / 2.0
        ground_y = min_pt[1]

        translation = [-center_x, -ground_y, -center_z]
        scene.apply_translation(translation)
        logs.append(f"Aligned model origin to ground-level: offset {[round(x, 3) for x in translation]}")

        # Export intermediate normalized GLB
        temp_normalized = output_glb.with_suffix(".norm.glb")
        temp_normalized.write_bytes(scene.export(file_type="glb"))

        # 5. Invoke gltf-transform for deduplication, welding, and simplification
        triangle_budget = options.get("triangle_budget", self.triangle_budget)
        initial_triangles = sum(len(g.faces) for g in scene.geometry.values() if hasattr(g, "faces"))
        logs.append(f"Current total triangle count: {initial_triangles:,} (budget: {triangle_budget:,})")

        optimized_path = self._run_gltf_transform_pipeline(
            input_path=temp_normalized,
            output_path=output_glb,
            initial_triangles=initial_triangles,
            triangle_budget=triangle_budget,
            logs=logs,
        )

        # Clean up temp normalized file
        if temp_normalized.exists() and temp_normalized != output_glb:
            try:
                temp_normalized.unlink()
            except OSError:
                pass

        # 6. Re-inspect optimized model for parts.json and validation report
        final_scene = trimesh.load(str(output_glb), force="scene")
        final_file_size = output_glb.stat().st_size
        final_file_size_mb = final_file_size / (1024 * 1024)

        if final_file_size_mb > self.max_file_size_mb:
            warnings.append(
                ValidationWarning(
                    code="FILE_SIZE_EXCEEDS_BUDGET",
                    message=f"Optimized model size ({final_file_size_mb:.2f} MB) exceeds recommended {self.max_file_size_mb:.1f} MB target.",
                    severity="WARNING",
                )
            )

        parts: List[PartInfo] = []
        unnamed_count = 0
        total_final_triangles = 0
        total_final_vertices = 0

        for idx, (geom_name, geom) in enumerate(final_scene.geometry.items()):
            if not hasattr(geom, "faces") or not hasattr(geom, "vertices"):
                continue

            num_faces = len(geom.faces)
            num_verts = len(geom.vertices)
            total_final_triangles += num_faces
            total_final_vertices += num_verts

            clean_name = str(geom_name).strip()
            if not clean_name or clean_name.startswith("geometry_") or clean_name.isdigit():
                unnamed_count += 1
                part_name = f"Part_{idx + 1}"
            else:
                part_name = clean_name

            g_bounds = geom.bounds
            g_min = [float(x) for x in g_bounds[0]]
            g_max = [float(x) for x in g_bounds[1]]
            g_dims = [float(g_max[i] - g_min[i]) for i in range(3)]
            g_center = [float((g_min[i] + g_max[i]) / 2.0) for i in range(3)]

            parts.append(
                PartInfo(
                    mesh_name=part_name,
                    part_id=f"part_{idx + 1}",
                    triangle_count=num_faces,
                    vertex_count=num_verts,
                    bounding_box_meters={
                        "min": [round(x, 4) for x in g_min],
                        "max": [round(x, 4) for x in g_max],
                        "dimensions": [round(x, 4) for x in g_dims],
                    },
                    center=[round(x, 4) for x in g_center],
                )
            )

        # Write parts.json
        parts_data = [p.model_dump() for p in parts]
        parts_json_path.write_text(json.dumps(parts_data, indent=2))
        logs.append(f"Generated {parts_json_path.name} with {len(parts)} named parts.")

        # Compute scene bounding box in meters
        final_bounds = final_scene.bounds
        f_min = [round(float(x), 4) for x in final_bounds[0]]
        f_max = [round(float(x), 4) for x in final_bounds[1]]
        f_dims = [round(f_max[i] - f_min[i], 4) for i in range(3)]

        if unnamed_count > 0:
            warnings.append(
                ValidationWarning(
                    code="UNNAMED_MESHES",
                    message=f"{unnamed_count} mesh(es) lacked semantic CAD names and were assigned default part names.",
                    severity="INFO",
                )
            )

        report = ValidationReport(
            is_valid=len(hard_blockers) == 0,
            file_size_bytes=final_file_size,
            file_size_mb=round(final_file_size_mb, 3),
            triangle_count=total_final_triangles,
            vertex_count=total_final_vertices,
            mesh_count=len(parts),
            bounding_box_meters={
                "min": f_min,
                "max": f_max,
                "dimensions": f_dims,
            },
            unnamed_meshes_count=unnamed_count,
            warnings=warnings,
            hard_blockers=hard_blockers,
        )

        logs.append(
            f"Optimization finished: {total_final_triangles:,} tris, {len(parts)} meshes, "
            f"{final_file_size_mb:.2f} MB, bbox dims {f_dims}"
        )
        return parts, report, logs

    def _run_gltf_transform_pipeline(
        self,
        input_path: Path,
        output_path: Path,
        initial_triangles: int,
        triangle_budget: int,
        logs: List[str],
    ) -> Path:
        """Run gltf-transform dedup, weld, simplify CLI commands."""
        # Check if npx is available
        which_npx = shutil.which("npx")
        if not which_npx:
            logs.append("npx not found on PATH; copying normalized file directly.")
            shutil.copy2(input_path, output_path)
            return output_path

        current_file = input_path
        step1_weld = output_path.with_suffix(".weld.glb")

        # 1. Weld vertices and deduplicate accessors
        cmd_weld = [
            which_npx,
            "@gltf-transform/cli",
            "weld",
            str(current_file),
            str(step1_weld),
        ]
        logs.append("Running gltf-transform weld...")
        code, _, stderr = run_subprocess_safe(cmd_weld, timeout_seconds=60)
        if code == 0 and step1_weld.is_file() and step1_weld.stat().st_size > 0:
            current_file = step1_weld
        else:
            logs.append(f"Weld step warning (code {code}): {stderr[:200]}")

        # 2. Simplification if triangles exceed budget
        if initial_triangles > triangle_budget:
            ratio = max(0.05, float(triangle_budget) / float(initial_triangles))
            logs.append(f"Simplifying geometry with target ratio {ratio:.3f}...")
            step2_simplify = output_path.with_suffix(".simp.glb")
            cmd_simplify = [
                which_npx,
                "@gltf-transform/cli",
                "simplify",
                str(current_file),
                str(step2_simplify),
                "--ratio",
                str(round(ratio, 4)),
                "--error",
                "0.005",
            ]
            code, _, stderr = run_subprocess_safe(cmd_simplify, timeout_seconds=90)
            if code == 0 and step2_simplify.is_file() and step2_simplify.stat().st_size > 0:
                current_file = step2_simplify
            else:
                logs.append(f"Simplify step warning (code {code}): {stderr[:200]}")

        # Final copy to destination
        shutil.copy2(current_file, output_path)

        # Cleanup intermediate files
        for p in (step1_weld, output_path.with_suffix(".simp.glb")):
            if p.exists() and p != output_path:
                try:
                    p.unlink()
                except OSError:
                    pass

        return output_path
