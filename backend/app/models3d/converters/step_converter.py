"""STEP and IGES converter using FreeCAD in headless mode."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from app.models3d.converters.base import BaseConverter, ConversionResult, run_subprocess_safe
from app.models3d.schemas import SupportedCADFormat


def find_freecad_binary() -> Optional[Path]:
    """Find FreeCADCmd or FreeCAD executable."""
    # 1. Local tools folder (including subdirectories like FreeCAD_1.1.4-...)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    local_tools_dir = project_root / "tools" / "freecad"
    if local_tools_dir.is_dir():
        for exe in local_tools_dir.glob("**/FreeCADCmd.exe"):
            if exe.is_file():
                return exe
        for exe in local_tools_dir.glob("**/FreeCAD.exe"):
            if exe.is_file():
                return exe

    # 2. PATH
    for name in ("FreeCADCmd", "freecadcmd", "FreeCAD", "freecad"):
        which_bin = shutil.which(name)
        if which_bin:
            return Path(which_bin)

    # 3. Standard Program Files installations
    program_files = Path(r"C:\Program Files")
    if program_files.is_dir():
        for fc_dir in program_files.glob("FreeCAD*"):
            cmd_exe = fc_dir / "bin" / "FreeCADCmd.exe"
            if cmd_exe.is_file():
                return cmd_exe
            gui_exe = fc_dir / "bin" / "FreeCAD.exe"
            if gui_exe.is_file():
                return gui_exe

    return None


class STEPConverter(BaseConverter):
    """Converts STEP (.step/.stp) and IGES (.iges/.igs) files to GLB via headless FreeCAD."""

    name: str = "STEPConverter"

    def __init__(self, binary_path: Optional[Path] = None) -> None:
        self.binary_path = binary_path or find_freecad_binary()
        self.worker_script = (
            Path(__file__).resolve().parent / "scripts" / "freecad_convert.py"
        )

    def can_convert(self, fmt: SupportedCADFormat) -> bool:
        return fmt in (SupportedCADFormat.STEP, SupportedCADFormat.IGES)

    def is_available(self) -> Tuple[bool, str]:
        if self.binary_path and self.binary_path.is_file():
            return True, f"FreeCAD binary found at {self.binary_path}"
        return False, "FreeCAD binary not found"

    def convert(
        self,
        input_path: Path,
        output_glb_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConversionResult:
        options = options or {}
        start_time = time.time()
        logs: List[str] = [f"Starting FreeCAD conversion for {input_path.name}"]
        warnings: List[str] = []

        output_glb_path.parent.mkdir(parents=True, exist_ok=True)
        deflection = options.get("tessellation_deflection", 0.02)

        if not self.binary_path or not self.binary_path.is_file():
            return ConversionResult(
                success=False,
                duration_seconds=time.time() - start_time,
                logs=logs,
                warnings=warnings,
                error="FreeCAD is not installed or FreeCADCmd was not found.",
                converter_name=self.name,
            )

        # Request FreeCAD to output an intermediate OBJ
        temp_obj_path = output_glb_path.with_suffix(".temp.obj")

        cmd = [
            str(self.binary_path),
            str(self.worker_script),
            "--pass",
            str(input_path.resolve()),
            str(temp_obj_path.resolve()),
            str(deflection),
        ]

        logs.append(f"Executing FreeCAD worker: {self.binary_path.name} with deflection {deflection}")
        code, stdout, stderr = run_subprocess_safe(cmd, timeout_seconds=180)

        for line in stdout.splitlines():
            if "[FreeCADWorker]" in line or "Error" in line or "Warning" in line:
                logs.append(line.strip())

        if code != 0 or not temp_obj_path.is_file() or temp_obj_path.stat().st_size == 0:
            error_msg = f"FreeCAD exited with code {code}. Stderr: {stderr[:300]}"
            logs.append(error_msg)
            if temp_obj_path.exists():
                temp_obj_path.unlink()
            return ConversionResult(
                success=False,
                duration_seconds=time.time() - start_time,
                logs=logs,
                warnings=warnings,
                error=error_msg,
                converter_name=self.name,
            )

        # Convert intermediate OBJ to GLB via trimesh
        try:
            import trimesh

            logs.append(f"Converting FreeCAD OBJ ({temp_obj_path.stat().st_size} bytes) to GLB...")
            mesh_scene = trimesh.load(str(temp_obj_path), force="scene")
            glb_data = mesh_scene.export(file_type="glb")
            output_glb_path.write_bytes(glb_data)
        finally:
            if temp_obj_path.exists():
                try:
                    temp_obj_path.unlink()
                except OSError:
                    pass

        if not output_glb_path.is_file() or output_glb_path.stat().st_size == 0:
            return ConversionResult(
                success=False,
                duration_seconds=time.time() - start_time,
                logs=logs,
                warnings=warnings,
                error="Failed to serialize FreeCAD mesh to GLB",
                converter_name=self.name,
            )

        elapsed = time.time() - start_time
        file_size = output_glb_path.stat().st_size
        logs.append(f"FreeCAD conversion succeeded in {elapsed:.2f}s ({file_size} bytes)")
        return ConversionResult(
            success=True,
            output_path=str(output_glb_path),
            duration_seconds=elapsed,
            logs=logs,
            warnings=warnings,
            converter_name=self.name,
        )
