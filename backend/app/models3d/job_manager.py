"""Background conversion job executor, versioning storage, and state manager."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Optional, Tuple
import uuid

from app.models3d.converters import ConverterRegistry
from app.models3d.detector import FileTypeDetector
from app.models3d.optimizer import ModelOptimizer
from app.models3d.schemas import (
    ConversionJob,
    JobStatus,
    PartInfo,
    SupportedCADFormat,
    ValidationReport,
)


class ModelJobManager:
    """Manages background CAD conversion jobs, versions, and on-disk assets."""

    def __init__(
        self,
        base_storage_dir: Optional[Path] = None,
        max_workers: int = 2,
    ) -> None:
        if base_storage_dir is None:
            # Root models directory: <project_root>/models
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            self.base_storage_dir = project_root / "models"
        else:
            self.base_storage_dir = base_storage_dir

        self.base_storage_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, ConversionJob] = {}
        self.detector = FileTypeDetector()
        self.registry = ConverterRegistry()
        self.optimizer = ModelOptimizer()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def create_job(
        self,
        structure_id: str,
        source_filename: str,
        temp_file_path: Path,
    ) -> ConversionJob:
        """Create a new job, inspect format, and enqueue for background execution."""
        job_id = str(uuid.uuid4())[:8]

        # 1. Inspect format
        fmt, inspect_msg = self.detector.inspect_file(temp_file_path)

        job = ConversionJob(
            job_id=job_id,
            structure_id=structure_id,
            source_filename=source_filename,
            source_file_type=fmt.value,
            status=JobStatus.PENDING,
            progress_percent=5.0,
        )
        job.log(f"Job enqueued for structure '{structure_id}', file '{source_filename}' ({fmt.value})")
        job.log(inspect_msg)

        if fmt in (
            SupportedCADFormat.UNSUPPORTED_2D,
            SupportedCADFormat.UNSUPPORTED_PROPRIETARY,
            SupportedCADFormat.UNKNOWN,
        ):
            job.status = JobStatus.FAILED
            job.error_message = inspect_msg
            job.completed_at = time.time()
            self.jobs[job_id] = job
            return job

        self.jobs[job_id] = job

        # Schedule worker in background thread
        self.executor.submit(self._run_conversion_worker, job_id, temp_file_path, fmt)
        return job

    def get_job(self, job_id: str) -> Optional[ConversionJob]:
        return self.jobs.get(job_id)

    def get_active_model_dir(self, structure_id: str) -> Optional[Path]:
        """Find the directory containing the active/latest version for a structure."""
        struct_dir = self.base_storage_dir / structure_id
        if not struct_dir.is_dir():
            return None

        # Check version pointer file
        current_file = struct_dir / "current_version.txt"
        if current_file.is_file():
            ver = current_file.read_text().strip()
            target = struct_dir / ver
            if target.is_dir() and (target / "model.glb").is_file():
                return target

        # Fallback to highest numeric version directory
        versions = [
            d for d in struct_dir.iterdir()
            if d.is_dir() and (d / "model.glb").is_file()
        ]
        if not versions:
            return None

        versions.sort(key=lambda d: d.name, reverse=True)
        return versions[0]

    def _determine_next_version(self, structure_id: str) -> Tuple[str, Path]:
        """Calculate next version string (v1, v2...) and create directory."""
        struct_dir = self.base_storage_dir / structure_id
        struct_dir.mkdir(parents=True, exist_ok=True)

        existing = [
            d.name for d in struct_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()
        ]
        if not existing:
            next_ver = "v1"
        else:
            nums = [int(v[1:]) for v in existing]
            next_ver = f"v{max(nums) + 1}"

        target_dir = struct_dir / next_ver
        target_dir.mkdir(parents=True, exist_ok=True)
        return next_ver, target_dir

    def _run_conversion_worker(
        self,
        job_id: str,
        temp_input_file: Path,
        fmt: SupportedCADFormat,
    ) -> None:
        """Worker thread executing the conversion, optimization, and auditing pipeline."""
        job = self.jobs[job_id]
        job.status = JobStatus.PROCESSING
        job.progress_percent = 20.0

        try:
            # 1. Determine storage version
            version_str, version_dir = self._determine_next_version(job.structure_id)
            job.log(f"Allocated version {version_str} at {version_dir.name}")

            # Copy source file for archiving
            archived_source = version_dir / f"source{temp_input_file.suffix}"
            shutil.copy2(temp_input_file, archived_source)

            # 2. Select Converter
            converter = self.registry.get_converter_for_format(fmt)
            if not converter:
                raise ValueError(f"No converter registered for format {fmt.value}")

            avail, reason = converter.is_available()
            if not avail:
                raise RuntimeError(f"Converter '{converter.name}' unavailable: {reason}")

            job.log(f"Converting using {converter.name}...")
            job.progress_percent = 40.0

            raw_glb_path = version_dir / "raw_model.glb"
            res = converter.convert(archived_source, raw_glb_path)

            for line in res.logs:
                job.log(line)
            for warn in res.warnings:
                job.log(f"Warning: {warn}")

            if not res.success or not raw_glb_path.is_file():
                raise RuntimeError(res.error or "CAD conversion failed without generating output.")

            job.log("Raw GLB generation complete. Proceeding to optimization stage...")
            job.status = JobStatus.OPTIMIZING
            job.progress_percent = 70.0

            # 3. Optimize and validate
            final_glb_path = version_dir / "model.glb"
            parts_json_path = version_dir / "parts.json"

            parts, report, opt_logs = self.optimizer.optimize(
                input_glb=raw_glb_path,
                output_glb=final_glb_path,
                parts_json_path=parts_json_path,
            )

            for line in opt_logs:
                job.log(line)

            # Write validation_report.json
            report_path = version_dir / "validation_report.json"
            report_path.write_text(json.dumps(report.model_dump(), indent=2))

            job.validation_report = report
            job.output_glb_path = str(final_glb_path)

            # If validation had hard blockers, mark job failed
            if not report.is_valid:
                blockers_str = "; ".join(report.hard_blockers)
                job.status = JobStatus.FAILED
                job.error_message = f"Validation failed with hard blockers: {blockers_str}"
                job.log(f"Model rejected: {blockers_str}")
            else:
                # Update current_version.txt pointer
                current_file = self.base_storage_dir / job.structure_id / "current_version.txt"
                current_file.write_text(version_str)

                job.status = JobStatus.COMPLETED
                job.progress_percent = 100.0
                job.log(f"Model pipeline completed successfully for version {version_str}.")

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            job.log(f"Fatal error during processing: {exc}")
        finally:
            job.completed_at = time.time()
            # Clean up temp file
            if temp_input_file.exists():
                try:
                    temp_input_file.unlink()
                except OSError:
                    pass
