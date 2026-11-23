"""Unified IngestionSource Interface, Frame Contract & Lifecycle Events.

Defines the canonical Stage 4 ingestion contract:
- Async iterator yielding `Frame` chunks ([n_samples, n_sensors]) and `LifecycleEvent` items.
- Lifecycle events: RECORDING_START, RECORDING_END, SOURCE_UP, SOURCE_DOWN, GAP.
- Retains a backwards-compatible `start(emit_callback)` / `stop()` adapter for legacy code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import numpy as np


class EventType(str, Enum):
    """Canonical ingestion stream item types."""
    FRAME = "FRAME"
    RECORDING_START = "RECORDING_START"
    RECORDING_END = "RECORDING_END"
    SOURCE_UP = "SOURCE_UP"
    SOURCE_DOWN = "SOURCE_DOWN"
    GAP = "GAP"


@dataclass
class Frame:
    """Canonical multi-channel telemetry chunk produced by all IngestionSources."""
    structure_id: str
    sensor_ids: List[str]
    t0: int  # Start timestamp in UTC nanoseconds
    fs: float  # Sampling frequency in Hz
    samples: np.ndarray  # Shape (n_samples, n_sensors), float32
    seq: int = 0
    source_id: str = "unknown"
    flags: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Frame.samples must be 2D [n_samples, n_sensors], got shape {arr.shape}")
        if arr.shape[1] != len(self.sensor_ids):
            raise ValueError(
                f"Frame sensor count mismatch: samples has {arr.shape[1]} cols, "
                f"sensor_ids has {len(self.sensor_ids)} entries"
            )
        self.samples = arr

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_sensors(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs if self.fs > 0 else 0.0

    @property
    def t_end(self) -> int:
        if self.fs <= 0:
            return self.t0
        return self.t0 + int(round((self.n_samples / self.fs) * 1_000_000_000))


@dataclass
class LifecycleEvent:
    """Lifecycle and boundary notification emitted inline on the ingestion stream."""
    event_type: EventType
    structure_id: str
    source_id: str
    timestamp_ns: int = field(default_factory=time.time_ns)
    recording_id: Optional[str] = None
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


IngestionStreamItem = Union[Frame, LifecycleEvent]


class IngestionSource(ABC):
    """Abstract telemetry ingestion interface supporting Simulator, Replay, HTTP, MQTT."""

    def __init__(self, name: str, structure_id: str = "default", source_id: Optional[str] = None) -> None:
        self.name = name
        self.structure_id = structure_id
        self.source_id = source_id or f"{name.lower()}_{id(self)}"
        self._is_running = False
        self._compat_task: Optional[asyncio.Task[Any]] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @abstractmethod
    async def stream(self) -> AsyncIterator[IngestionStreamItem]:
        """Yield Frame chunks and LifecycleEvent records asynchronously."""
        if False:  # pragma: no cover
            yield LifecycleEvent(
                event_type=EventType.SOURCE_UP,
                structure_id=self.structure_id,
                source_id=self.source_id,
            )

    def __aiter__(self) -> AsyncIterator[IngestionStreamItem]:
        return self.stream()

    async def start(self, emit_callback: Callable[[np.ndarray], None]) -> None:
        """Legacy compatibility helper: runs stream() in background and forwards rows to emit_callback."""
        if self._is_running and self._compat_task and not self._compat_task.done():
            return
        self._is_running = True

        async def _compat_loop() -> None:
            try:
                async for item in self.stream():
                    if not self._is_running:
                        break
                    if isinstance(item, Frame):
                        for row in item.samples:
                            if not self._is_running:
                                break
                            emit_callback(row)
            except asyncio.CancelledError:
                pass
            finally:
                self._is_running = False

        self._compat_task = asyncio.create_task(_compat_loop())
        await asyncio.sleep(0)

    @abstractmethod
    def stop(self) -> None:
        """Halt telemetry streaming safely."""
        self._is_running = False
        if self._compat_task and not self._compat_task.done():
            self._compat_task.cancel()
