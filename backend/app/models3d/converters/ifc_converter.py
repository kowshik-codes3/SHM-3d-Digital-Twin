"""IFC to GLB conversion using IfcConvert and native ifcopenshell.geom fallback."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.models3d.converters.base import BaseConverter, ConversionResult, run_subprocess_safe
from app.models3d.schemas import SupportedCADFormat


def find_ifcconvert_binary() -> Optional[Path]:
    """Find IfcConvert executable on local machine or repo tools."""
    # 1. Check tools directory relative to this project
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    local_tool = project_root / "tools" / "ifcconvert" / "IfcConvert.exe"
    if local_tool.is_file():
        return local_tool

    # 2. Check PATH
    which_bin = shutil.which("IfcConvert")
    if which_bin:
        return Path(which_bin)

    return None


class IFCConverter(BaseConverter):
    """Converts Industry Foundation Classes (.ifc) files to GLB."""

    name: str = "IFCConverter"

    def __init__(self, binary_path: Optional[Path] = None) -> None:
        self.binary_path = binary_path or find_ifcconvert_binary()

    def can_convert(self, fmt: SupportedCADFormat) -> bool:
        return fmt == SupportedCADFormat.IFC

    def is_available(self) -> Tuple[bool, str]:
        if self.binary_path and self.binary_path.is_file():
            return True, f"IfcConvert binary found at {self.binary_path}"
        try:
            import ifcopenshell
            import ifcopenshell.geom
            return True, "Native ifcopenshell.geom Python engine available"
        except ImportError:
            return False, "Neither IfcConvert binary nor ifcopenshell.geom is available"

    def convert(
        self,
        input_path: Path,
        output_glb_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConversionResult:
        options = options or {}
        start_time = time.time()
        logs: List[str] = [f"Starting IFC conversion for {input_path.name}"]
        warnings: List[str] = []

        output_glb_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Try IfcConvert CLI if available
        if self.binary_path and self.binary_path.is_file():
            cmd = [
                str(self.binary_path),
                str(input_path),
                str(output_glb_path),
                "-y",  # overwrite without prompt
                "-v",
                "--use-element-names",
            ]
            logs.append(f"Executing: {self.binary_path.name} {input_path.name} -> {output_glb_path.name}")
            code, stdout, stderr = run_subprocess_safe(cmd, timeout_seconds=180)
            if code == 0 and output_glb_path.exists() and output_glb_path.stat().st_size > 0:
                elapsed = time.time() - start_time
                logs.append(f"IfcConvert succeeded in {elapsed:.2f}s ({output_glb_path.stat().st_size} bytes)")
                return ConversionResult(
                    success=True,
                    output_path=str(output_glb_path),
                    duration_seconds=elapsed,
                    logs=logs,
                    warnings=warnings,
                    converter_name="IfcConvert",
                )
            else:
                logs.append(f"IfcConvert returned code {code}. Falling back to Python ifcopenshell.geom. Stderr: {stderr[:300]}")

        # 2. Fallback to Python ifcopenshell.geom
        try:
            import ifcopenshell
            import ifcopenshell.geom
            import trimesh

            logs.append("Parsing IFC file via ifcopenshell Python API...")
            ifc_file = ifcopenshell.open(str(input_path))

            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)
            
            # Extract elements
            products = ifc_file.by_type("IfcProduct")
            logs.append(f"Found {len(products)} IFC products")

            scene = trimesh.Scene()
            element_count = 0

            for product in products:
                if product.is_a("IfcOpeningElement") or product.is_a("IfcSpace"):
                    continue
                if not product.Representation:
                    continue

                try:
                    shape = ifcopenshell.geom.create_shape(settings, product)
                    verts = shape.geometry.verts
                    faces = shape.geometry.faces
                    
                    if not verts or not faces:
                        continue

                    # Reshape verts into (N, 3) and faces into (M, 3)
                    v_arr = np.array(verts, dtype=np.float64).reshape(-1, 3)
                    f_arr = np.array(faces, dtype=np.int32).reshape(-1, 3)

                    # Determine semantic part name
                    elem_name = getattr(product, "Name", None) or f"{product.is_a()}_{product.id()}"
                    clean_name = str(elem_name).strip().replace(" ", "_")

                    mesh = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False)
                    scene.add_geometry(mesh, node_name=clean_name)
                    element_count += 1
                except Exception as shape_err:
                    warnings.append(f"Skipping product {product.id()} ({product.is_a()}): {shape_err}")

            if element_count == 0:
                return ConversionResult(
                    success=False,
                    duration_seconds=time.time() - start_time,
                    logs=logs,
                    warnings=warnings,
                    error="No 3D geometric shapes found in the IFC file.",
                    converter_name="ifcopenshell_python",
                )

            # Export to GLB
            glb_bytes = scene.export(file_type="glb")
            output_glb_path.write_bytes(glb_bytes)

            elapsed = time.time() - start_time
            logs.append(f"Extracted {element_count} elements into GLB in {elapsed:.2f}s ({len(glb_bytes)} bytes)")
            return ConversionResult(
                success=True,
                output_path=str(output_glb_path),
                duration_seconds=elapsed,
                logs=logs,
                warnings=warnings,
                converter_name="ifcopenshell_python",
            )

        except Exception as exc:
            elapsed = time.time() - start_time
            logs.append(f"Python ifcopenshell conversion failed: {exc}")
            return ConversionResult(
                success=False,
                duration_seconds=elapsed,
                logs=logs,
                warnings=warnings,
                error=f"IFC conversion failed: {exc}",
                converter_name=self.name,
            )
