"""Base classes and safe subprocess execution primitives for CAD converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.models3d.schemas import SupportedCADFormat


class ConversionResult(BaseModel):
    """Result returned by any CAD-to-GLB converter."""
    success: bool
    output_path: Optional[str] = None
    duration_seconds: float = 0.0
    logs: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    converter_name: str = "base"


def run_subprocess_safe(
    cmd: List[str],
    timeout_seconds: int = 180,
    cwd: Optional[Path] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """Execute external CLI command safely without shell injection risks.
    
    Args:
        cmd: List of command arguments. cmd[0] must be the verified binary path.
        timeout_seconds: Subprocess timeout in seconds (default 180s).
        cwd: Working directory.
        extra_env: Additional environment variables to merge into os.environ.
        
    Returns:
        Tuple of (exit_code, stdout_text, stderr_text)
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # Validate that cmd is a list of strings
    if not isinstance(cmd, (list, tuple)) or not cmd:
        raise ValueError("cmd must be a non-empty list of arguments")
    for arg in cmd:
        if not isinstance(arg, str):
            raise ValueError(f"All command arguments must be strings: {arg}")

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,  # Strict security requirement
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return proc.returncode, stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        elapsed = time.time() - start_time
        return -1, stdout or "", f"Process timed out after {elapsed:.1f}s (limit {timeout_seconds}s)"
    except Exception as exc:
        return -1, "", f"Failed to execute command {' '.join(cmd)}: {exc}"


class BaseConverter(ABC):
    """Abstract interface for all CAD/BIM converters."""

    name: str = "BaseConverter"

    @abstractmethod
    def can_convert(self, fmt: SupportedCADFormat) -> bool:
        """Return True if this converter supports the given format."""
        pass

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """Check if required external binaries or libraries are available."""
        pass

    @abstractmethod
    def convert(
        self,
        input_path: Path,
        output_glb_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> ConversionResult:
        """Convert input CAD file into a GLB file.
        
        Args:
            input_path: Path to input CAD/BIM model.
            output_glb_path: Path where GLB must be written.
            options: Optional conversion settings (tessellation tolerance, filters, etc.).
            
        Returns:
            ConversionResult with success flag, logs, and timing.
        """
        pass
