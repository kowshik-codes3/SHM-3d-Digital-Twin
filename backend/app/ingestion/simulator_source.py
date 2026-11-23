"""Simulator Ingestion Source wrapping TelemetrySimulator for demo mode only.

Disabled automatically when DEMO_MODE=false.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator, Callable, List, Optional

import numpy as np

from app.core.config import settings
from app.ingestion.base import EventType, Frame, IngestionSource, IngestionStreamItem, LifecycleEvent
from app.simulation.simulator import TelemetrySimulator


def _is_demo_mode_enabled() -> bool:
    env_val = os.environ.get("DEMO_MODE")
    if env_val is not None:
        return env_val.strip().lower() not in ("0", "false", "no", "off")
    return bool(settings.DEMO_MODE)


class SimulatorSource(IngestionSource):
    """Wraps TelemetrySimulator to generate synthetic modal vibrations for demo mode."""

    def __init__(
        self,
        fs: float = 50.0,
        base_temp: float = 26.0,
        structure_id: str = "bridge_demo_01",
        sensor_ids: Optional[List[str]] = None,
        chunk_samples: int = 50,
    ) -> None:
        super().__init__(name="Simulator", structure_id=structure_id)
        self.simulator = TelemetrySimulator(fs=fs, base_temp=base_temp)
        self.sensor_ids = sensor_ids or [
            "Sensor_Abutment_Left",
            "Sensor_Pier_Left",
            "Sensor_MidSpan_Girder",
            "Sensor_Pier_Right",
            "Sensor_Ambient_Temp",
        ]
        self.chunk_samples = max(1, int(chunk_samples))

    @property
    def fs(self) -> float:
        return self.simulator.fs

    @property
    def base_temp(self) -> float:
        return self.simulator.base_temp

    def inject_fault(self, node_id: str, fault_type: str = "stiffness_loss", severity: float = 0.5) -> None:
        self.simulator.inject_fault(node_id, fault_type, severity)

    def clear_faults(self) -> None:
        self.simulator.clear_faults()

    async def stream(self) -> AsyncIterator[IngestionStreamItem]:
        """Yield synthetic Frames paced to real time; disabled when DEMO_MODE=false."""
        if not _is_demo_mode_enabled():
            self._is_running = False
            yield LifecycleEvent(
                event_type=EventType.SOURCE_DOWN,
                structure_id=self.structure_id,
                source_id=self.source_id,
                reason="SimulatorSource is disabled because DEMO_MODE=false",
            )
            raise RuntimeError("SimulatorSource is disabled when DEMO_MODE=false")

        self._is_running = True
        yield LifecycleEvent(
            event_type=EventType.SOURCE_UP,
            structure_id=self.structure_id,
            source_id=self.source_id,
        )

        # Ensure baseline signal pre-generated
        if getattr(self.simulator, "_baseline_frames", None) is None:
            self.simulator._pregenerate_baseline()

        seq = 0
        dt_chunk = self.chunk_samples / self.simulator.fs
        next_deadline = time.perf_counter()
        t0_ns = time.time_ns()
        sample_ns = int(round(1_000_000_000 / self.simulator.fs))

        try:
            while self._is_running:
                rows = []
                for _ in range(self.chunk_samples):
                    rows.append(self.simulator.next_frame())
                block = np.asarray(rows, dtype=np.float32)

                # Adapt sensor count if custom sensor_ids length differs
                n_s = len(self.sensor_ids)
                if block.shape[1] > n_s:
                    block = block[:, :n_s]
                elif block.shape[1] < n_s:
                    pad = np.zeros((block.shape[0], n_s - block.shape[1]), dtype=np.float32)
                    block = np.hstack([block, pad])

                yield Frame(
                    structure_id=self.structure_id,
                    sensor_ids=self.sensor_ids,
                    t0=t0_ns,
                    fs=self.simulator.fs,
                    samples=block,
                    seq=seq,
                    source_id=self.source_id,
                )
                seq += 1
                t0_ns += self.chunk_samples * sample_ns
                next_deadline += dt_chunk
                sleep_dur = next_deadline - time.perf_counter()
                if sleep_dur > 0:
                    await asyncio.sleep(sleep_dur)
                else:
                    # Resync deadline if fallen behind significantly to prevent runaway catchup burst
                    if sleep_dur < -2.0:
                        next_deadline = time.perf_counter()
                    await asyncio.sleep(0)
        finally:
            self._is_running = False
            yield LifecycleEvent(
                event_type=EventType.SOURCE_DOWN,
                structure_id=self.structure_id,
                source_id=self.source_id,
                reason="Simulator stopped",
            )

    async def start(self, emit_callback: Callable[[np.ndarray], None]) -> None:
        """Legacy callback mode wrapping TelemetrySimulator directly."""
        if not _is_demo_mode_enabled():
            raise RuntimeError("SimulatorSource is disabled when DEMO_MODE=false")
        self._is_running = True
        await self.simulator.start(
            buffer_push_fn=emit_callback,
            inference_trigger_fn=None,
        )

    def stop(self) -> None:
        self._is_running = False
        self.simulator.stop()
