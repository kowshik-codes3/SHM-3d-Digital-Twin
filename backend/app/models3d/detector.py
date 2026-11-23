"""File type detection by extension and magic bytes with strict format validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from app.models3d.schemas import SupportedCADFormat

# Magic byte signatures
MAGIC_GLB = b"glTF"
MAGIC_DWG = b"AC"
MAGIC_OLE_COMPOUND = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # RVT, SLDPRT, etc.
MAGIC_STEP_IFC = b"ISO-10303-21"
MAGIC_FBX_BINARY = b"Kaydara FBX Binary"


class FileTypeDetector:
    """Detects 3D/BIM CAD formats, verifies file size limits, and returns clear actionable errors."""

    def __init__(self, max_upload_size_bytes: int = 100 * 1024 * 1024) -> None:
        self.max_upload_size_bytes = max_upload_size_bytes

    def inspect_file(self, file_path: Path) -> Tuple[SupportedCADFormat, str]:
        """Inspects file extension and initial header bytes.
        
        Returns:
            (format_enum, message)
        """
        if not file_path.is_file():
            return SupportedCADFormat.UNKNOWN, f"File not found: {file_path}"

        # 1. Enforce size limit
        file_size = file_path.stat().st_size
        if file_size > self.max_upload_size_bytes:
            max_mb = self.max_upload_size_bytes / (1024 * 1024)
            size_mb = file_size / (1024 * 1024)
            return (
                SupportedCADFormat.UNKNOWN,
                f"File size exceeds maximum upload limit ({size_mb:.1f} MB > {max_mb:.0f} MB). "
                f"Please simplify the CAD export before uploading.",
            )

        if file_size == 0:
            return SupportedCADFormat.UNKNOWN, "Uploaded file is empty (0 bytes)."

        # Read first 1024 bytes for header magic check
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)
        except Exception as exc:
            return SupportedCADFormat.UNKNOWN, f"Unable to read file header: {exc}"

        ext = file_path.suffix.lower()

        # 2. Check 2D DWG / DXF drawings -> REJECT with clear instructions
        if ext in (".dwg", ".dxf") or header.startswith(MAGIC_DWG):
            return (
                SupportedCADFormat.UNSUPPORTED_2D,
                "2D DWG/DXF drawings are not 3D BIM/CAD models. "
                "StructureIQ requires a 3D geometry export. "
                "In AutoCAD / Civil 3D, export your 3D model as IFC (via 'IFCEXPORT') or STEP.",
            )

        # 3. Check proprietary BIM/CAD formats (RVT, SLDPRT, SLDASM, NWD, DGN) -> REJECT
        if ext in (".rvt", ".rfa"):
            return (
                SupportedCADFormat.UNSUPPORTED_PROPRIETARY,
                "Autodesk Revit (.RVT) files are proprietary binary files and cannot be converted directly. "
                "In Revit, open File -> Export -> IFC (choose IFC4 or IFC2x3) and upload the resulting .ifc file.",
            )

        if ext in (".sldprt", ".sldasm"):
            return (
                SupportedCADFormat.UNSUPPORTED_PROPRIETARY,
                "SolidWorks (.SLDPRT / .SLDASM) files are proprietary. "
                "In SolidWorks, open File -> Save As -> STEP (.stp or .step AP214/AP242) and upload the resulting file.",
            )

        if ext in (".nwd", ".nwc"):
            return (
                SupportedCADFormat.UNSUPPORTED_PROPRIETARY,
                "Navisworks (.NWD) files are proprietary caches. "
                "Please export the underlying BIM assets as IFC or FBX and upload.",
            )

        # 4. Check supported 3D formats
        # A) GLB / glTF
        if header.startswith(MAGIC_GLB) or ext == ".glb":
            return SupportedCADFormat.GLB, "Valid GLB binary glTF model."

        if ext == ".gltf":
            return SupportedCADFormat.GLTF, "Valid glTF JSON text model."

        # B) IFC (Industry Foundation Classes)
        if ext == ".ifc" or (MAGIC_STEP_IFC in header and b"IFC" in header.upper()):
            return SupportedCADFormat.IFC, "Valid IFC (Industry Foundation Classes) BIM model."

        # C) STEP / STP
        if ext in (".step", ".stp") or MAGIC_STEP_IFC in header:
            return SupportedCADFormat.STEP, "Valid STEP CAD surface/solid model."

        # D) IGES
        if ext in (".iges", ".igs"):
            return SupportedCADFormat.IGES, "Valid IGES CAD surface model."

        # E) OBJ
        if ext == ".obj":
            return SupportedCADFormat.OBJ, "Valid Wavefront OBJ mesh."

        # F) FBX
        if ext == ".fbx" or header.startswith(MAGIC_FBX_BINARY) or b"Kaydara FBX" in header:
            return SupportedCADFormat.FBX, "Valid Autodesk FBX mesh."

        # Fallback unknown
        return (
            SupportedCADFormat.UNKNOWN,
            f"Unsupported file format '{ext}'. "
            f"Supported 3D formats: IFC (.ifc), STEP (.step/.stp), GLB (.glb), glTF (.gltf), OBJ (.obj), FBX (.fbx).",
        )
