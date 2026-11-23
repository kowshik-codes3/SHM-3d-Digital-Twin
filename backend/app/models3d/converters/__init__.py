"""Registry and factory for 3D/CAD converters."""

from __future__ import annotations

from typing import Dict, List, Optional

from app.models3d.converters.base import BaseConverter, ConversionResult
from app.models3d.converters.blend_converter import BlenderConverter
from app.models3d.converters.ifc_converter import IFCConverter
from app.models3d.converters.passthrough_converter import PassthroughConverter
from app.models3d.converters.step_converter import STEPConverter
from app.models3d.schemas import SupportedCADFormat


class ConverterRegistry:
    """Manages and dispatches conversion tasks to the appropriate converter."""

    def __init__(self) -> None:
        self.converters: List[BaseConverter] = [
            PassthroughConverter(),
            IFCConverter(),
            BlenderConverter(),
            STEPConverter(),
        ]

    def get_converter_for_format(self, fmt: SupportedCADFormat) -> Optional[BaseConverter]:
        for converter in self.converters:
            if converter.can_convert(fmt):
                return converter
        return None

    def get_status_summary(self) -> Dict[str, Dict[str, str]]:
        summary = {}
        for c in self.converters:
            avail, reason = c.is_available()
            summary[c.name] = {"available": str(avail), "status": reason}
        return summary
