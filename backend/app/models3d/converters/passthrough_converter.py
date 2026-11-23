"""Pass-through converter for existing GLB and glTF models."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from app.models3d.converters.base import BaseConverter, ConversionResult
from app.models3d.schemas import SupportedCADFormat


class PassthroughConverter(BaseConverter):
    """Handles GLB and glTF models by copying and validating before optimization."""

    name: str = "PassthroughConverter"

    def can_convert(self, fmt: SupportedCADFormat) -> bool:
        return fmt in (SupportedCADFormat.GLB, SupportedCADFormat.GLTF)

    def is_available(self) -> Tuple[bool, str]:
        return True, "Passthrough converter is always available."

    def convert(
        self,
        input_path: Path,
        output_glb_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConversionResult:
        options = options or {}
        start_time = time.time()
        logs: List[str] = [f"Passthrough processing for {input_path.name}"]
        warnings: List[str] = []

        output_glb_path.parent.mkdir(parents=True, exist_ok=True)
        ext = input_path.suffix.lower()

        if ext == ".glb":
            # Verify magic bytes
            with open(input_path, "rb") as f:
                header = f.read(4)
            if header != b"glTF":
                return ConversionResult(
                    success=False,
                    duration_seconds=time.time() - start_time,
                    logs=logs,
                    warnings=warnings,
                    error=f"File has .glb extension but does not start with glTF magic header: {header!r}",
                    converter_name=self.name,
                )
            shutil.copy2(input_path, output_glb_path)
            logs.append(f"Copied GLB to destination: {output_glb_path.name}")

        elif ext == ".gltf":
            # glTF JSON format -> pack to GLB via trimesh or gltf-transform
            try:
                import trimesh

                logs.append("Packing glTF JSON structure into binary GLB...")
                scene = trimesh.load(str(input_path), force="scene")
                glb_data = scene.export(file_type="glb")
                output_glb_path.write_bytes(glb_data)
                logs.append("Successfully packed glTF into binary GLB")
            except Exception as exc:
                return ConversionResult(
                    success=False,
                    duration_seconds=time.time() - start_time,
                    logs=logs,
                    warnings=warnings,
                    error=f"Failed to pack glTF JSON to GLB: {exc}",
                    converter_name=self.name,
                )
        else:
            return ConversionResult(
                success=False,
                duration_seconds=time.time() - start_time,
                logs=logs,
                warnings=warnings,
                error=f"Unsupported passthrough extension {ext}",
                converter_name=self.name,
            )

        elapsed = time.time() - start_time
        return ConversionResult(
            success=True,
            output_path=str(output_glb_path),
            duration_seconds=elapsed,
            logs=logs,
            warnings=warnings,
            converter_name=self.name,
        )
