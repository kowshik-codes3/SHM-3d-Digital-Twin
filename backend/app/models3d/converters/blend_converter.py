"""Blender-based converter for OBJ, FBX, DAE, and STL files."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from app.models3d.converters.base import BaseConverter, ConversionResult, run_subprocess_safe
from app.models3d.schemas import SupportedCADFormat


def find_blender_binary() -> Optional[Path]:
    """Find Blender binary in PATH or standard Program Files locations."""
    which_bin = shutil.which("blender")
    if which_bin:
        return Path(which_bin)

    candidate_paths = [
        Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe"),
    ]
    for p in candidate_paths:
        if p.is_file():
            return p

    # Search Program Files\Blender Foundation recursively if not found in candidate versions
    bf_dir = Path(r"C:\Program Files\Blender Foundation")
    if bf_dir.is_dir():
        for exe in bf_dir.glob("**/blender.exe"):
            if exe.is_file():
                return exe

    return None


class BlenderConverter(BaseConverter):
    """Converts OBJ, FBX, DAE, STL to GLB via headless Blender."""

    name: str = "BlenderConverter"

    def __init__(self, binary_path: Optional[Path] = None) -> None:
        self.binary_path = binary_path or find_blender_binary()
        self.worker_script = (
            Path(__file__).resolve().parent / "scripts" / "blender_convert.py"
        )

    def can_convert(self, fmt: SupportedCADFormat) -> bool:
        return fmt in (
            SupportedCADFormat.OBJ,
            SupportedCADFormat.FBX,
        )

    def is_available(self) -> Tuple[bool, str]:
        if self.binary_path and self.binary_path.is_file():
            return True, f"Blender binary found at {self.binary_path}"
        return False, "Blender binary not found"

    def convert(
        self,
        input_path: Path,
        output_glb_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConversionResult:
        options = options or {}
        start_time = time.time()
        logs: List[str] = [f"Starting Blender conversion for {input_path.name}"]
        warnings: List[str] = []

        output_glb_path.parent.mkdir(parents=True, exist_ok=True)

        # Fast path for OBJ via trimesh if preferred or if blender is unavailable
        ext = input_path.suffix.lower()
        if ext == ".obj" and (not self.binary_path or not self.binary_path.is_file()):
            logs.append("Blender not available; attempting trimesh native OBJ conversion...")
            try:
                import trimesh

                mesh_or_scene = trimesh.load(str(input_path), force="scene")
                glb_bytes = mesh_or_scene.export(file_type="glb")
                output_glb_path.write_bytes(glb_bytes)
                elapsed = time.time() - start_time
                logs.append(f"trimesh OBJ converted to GLB in {elapsed:.2f}s ({len(glb_bytes)} bytes)")
                return ConversionResult(
                    success=True,
                    output_path=str(output_glb_path),
                    duration_seconds=elapsed,
                    logs=logs,
                    warnings=warnings,
                    converter_name="trimesh_obj",
                )
            except Exception as exc:
                warnings.append(f"trimesh OBJ fallback failed: {exc}")

        if not self.binary_path or not self.binary_path.is_file():
            return ConversionResult(
                success=False,
                duration_seconds=time.time() - start_time,
                logs=logs,
                warnings=warnings,
                error="Blender is not installed or could not be found.",
                converter_name=self.name,
            )

        cmd = [
            str(self.binary_path),
            "--background",
            "--python",
            str(self.worker_script),
            "--",
            str(input_path.resolve()),
            str(output_glb_path.resolve()),
        ]

        logs.append(f"Running Blender command: {' '.join(cmd[1:5])} ...")
        code, stdout, stderr = run_subprocess_safe(cmd, timeout_seconds=180)

        for line in stdout.splitlines():
            if "[BlenderWorker]" in line or "Error" in line or "Warning" in line:
                logs.append(line.strip())

        if code != 0 or not output_glb_path.is_file() or output_glb_path.stat().st_size == 0:
            error_msg = f"Blender exited with code {code}. Stderr: {stderr[:300]}"
            logs.append(error_msg)
            return ConversionResult(
                success=False,
                duration_seconds=time.time() - start_time,
                logs=logs,
                warnings=warnings,
                error=error_msg,
                converter_name=self.name,
            )

        elapsed = time.time() - start_time
        file_size = output_glb_path.stat().st_size
        logs.append(f"Blender conversion succeeded in {elapsed:.2f}s ({file_size} bytes)")
        return ConversionResult(
            success=True,
            output_path=str(output_glb_path),
            duration_seconds=elapsed,
            logs=logs,
            warnings=warnings,
            converter_name=self.name,
        )
