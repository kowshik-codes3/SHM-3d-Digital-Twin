"""HTTP Push Telemetry Ingestion Source for Edge IoT Gateways.

Stage 4 Enhancements:
- Supports canonical Frame payloads (binary SHMF v1 or JSON) and batch uploads.
- Enforces idempotency keys with bounded TTL so HTTP retries never duplicate data.
- Enforces configurable rate limiting and payload size bounds.
- Bounded queue drops oldest + flags GAP under slow consumers.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
import logging
import time
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from app.core.config import settings
from app.ingestion.base import EventType, Frame, IngestionSource, IngestionStreamItem, LifecycleEvent
from app.ingestion.protocol import decode_frame_auto

logger = logging.getLogger("SHM.HTTPSource")


class HTTPSource(IngestionSource):
    """Direct HTTP/REST batch push ingestion source with idempotency & rate limits."""

    def __init__(
        self,
        channel_names: List[str],
        structure_id: str = "bridge_demo_01",
        fs: float = 50.0,
        queue_maxsize: int = 512,
        idempotency_ttl_seconds: Optional[float] = None,
        rate_limit_rps: Optional[float] = None,
        max_payload_bytes: Optional[int] = None,
    ) -> None:
        super().__init__(name="HTTP", structure_id=structure_id)
        self.channel_names = list(channel_names)
        self.num_channels = len(self.channel_names)
        self.fs = float(fs)
        self.queue_maxsize = max(4, int(queue_maxsize))
        self.idempotency_ttl_seconds = float(
            idempotency_ttl_seconds if idempotency_ttl_seconds is not None else settings.IDEMPOTENCY_TTL_SECONDS
        )
        self.rate_limit_rps = float(rate_limit_rps if rate_limit_rps is not None else settings.HTTP_RATE_LIMIT_RPS)
        self.max_payload_bytes = int(max_payload_bytes or settings.MAX_FRAME_PAYLOAD_BYTES)

        self._queue: Optional[asyncio.Queue[IngestionStreamItem]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending_items: Deque[IngestionStreamItem] = deque(maxlen=self.queue_maxsize)

        # Idempotency store: key -> expiry monotonic timestamp
        self._seen_keys: OrderedDict[str, float] = OrderedDict()
        # Token-bucket rate limiter state
        self._tokens: float = max(10.0, self.rate_limit_rps)
        self._last_refill: float = time.monotonic()

        self.duplicate_idempotency_count: int = 0
        self.rate_limited_count: int = 0
        self.dropped_backpressure: int = 0

    def check_rate_limit(self) -> bool:
        """Consume 1 request token from the per-structure HTTP token bucket."""
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        capacity = max(20.0, self.rate_limit_rps * 2.0)
        self._tokens = min(capacity, self._tokens + elapsed * self.rate_limit_rps)
        if self._tokens < 1.0:
            self.rate_limited_count += 1
            return False
        self._tokens -= 1.0
        return True

    def is_duplicate_key(self, idempotency_key: Optional[str]) -> bool:
        """Return True if key has already been accepted within the TTL window; records new keys."""
        if not idempotency_key:
            return False
        now = time.monotonic()
        # Evict expired entries
        while self._seen_keys:
            oldest_k, exp_t = next(iter(self._seen_keys.items()))
            if exp_t <= now or len(self._seen_keys) > 10_000:
                self._seen_keys.popitem(last=False)
            else:
                break

        if idempotency_key in self._seen_keys:
            self.duplicate_idempotency_count += 1
            return True

        self._seen_keys[idempotency_key] = now + self.idempotency_ttl_seconds
        return False

    def parse_payload(self, raw_data: Union[str, bytes, dict, list]) -> Optional[np.ndarray]:
        """Legacy helper: parse payload into 1D vector for the first row."""
        try:
            if isinstance(raw_data, list):
                if len(raw_data) == self.num_channels:
                    return np.array([float(x) for x in raw_data], dtype=np.float32)
                return None
            frame = decode_frame_auto(
                raw_data,
                default_structure_id=self.structure_id,
                default_sensor_ids=self.channel_names,
                default_fs=self.fs,
                max_bytes=self.max_payload_bytes,
            )
            return frame.samples[0]
        except Exception as exc:
            logger.debug(f"Failed to parse HTTP push payload: {exc}")
            return None

    def push_batch_frame(
        self,
        frame: Frame,
        idempotency_key: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Enqueue a validated Frame chunk with rate limiting and idempotency deduplication.

        Returns (accepted, status_reason) where status_reason is one of:
        'accepted' | 'duplicate' | 'rate_limited' | 'structure_mismatch'
        """
        if frame.structure_id != self.structure_id:
            return False, "structure_mismatch"
        if not self.check_rate_limit():
            return False, "rate_limited"
        if self.is_duplicate_key(idempotency_key):
            return True, "duplicate"

        if self._queue is not None and self._loop is not None and not self._loop.is_closed():
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped_backpressure += 1
                except asyncio.QueueEmpty:
                    pass
                frame.flags["gap"] = True
                frame.flags["backpressure_dropped"] = self.dropped_backpressure
            self._queue.put_nowait(frame)
        else:
            if len(self._pending_items) == self._pending_items.maxlen:
                self.dropped_backpressure += 1
                frame.flags["gap"] = True
            self._pending_items.append(frame)

        return True, "accepted"

    def push_frame(self, frame_or_payload: Union[Frame, np.ndarray, list, dict, str, bytes]) -> bool:
        """Push a Frame, numpy array, or JSON/binary payload into the queue (backwards-compatible)."""
        if isinstance(frame_or_payload, Frame):
            ok, _ = self.push_batch_frame(frame_or_payload)
            return ok

        if isinstance(frame_or_payload, np.ndarray):
            if frame_or_payload.ndim == 1 and frame_or_payload.shape[0] == self.num_channels:
                samples = frame_or_payload.reshape(1, -1).astype(np.float32)
            elif frame_or_payload.ndim == 2 and frame_or_payload.shape[1] == self.num_channels:
                samples = frame_or_payload.astype(np.float32)
            else:
                return False
            frame = Frame(
                structure_id=self.structure_id,
                sensor_ids=self.channel_names,
                t0=time.time_ns(),
                fs=self.fs,
                samples=samples,
                source_id=self.source_id,
            )
            ok, _ = self.push_batch_frame(frame)
            return ok

        if isinstance(frame_or_payload, list):
            if len(frame_or_payload) != self.num_channels:
                return False
            arr = np.array([float(x) for x in frame_or_payload], dtype=np.float32).reshape(1, -1)
            frame = Frame(
                structure_id=self.structure_id,
                sensor_ids=self.channel_names,
                t0=time.time_ns(),
                fs=self.fs,
                samples=arr,
                source_id=self.source_id,
            )
            ok, _ = self.push_batch_frame(frame)
            return ok

        try:
            frame = decode_frame_auto(
                frame_or_payload,
                default_structure_id=self.structure_id,
                default_sensor_ids=self.channel_names,
                default_fs=self.fs,
                max_bytes=self.max_payload_bytes,
            )
            ok, _ = self.push_batch_frame(frame)
            return ok
        except Exception:
            return False

    async def stream(self) -> AsyncIterator[IngestionStreamItem]:
        """Yield queued HTTP Frames and LifecycleEvents."""
        self._is_running = True
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)

        while self._pending_items:
            item = self._pending_items.popleft()
            if not self._queue.full():
                self._queue.put_nowait(item)

        yield LifecycleEvent(
            event_type=EventType.SOURCE_UP,
            structure_id=self.structure_id,
            source_id=self.source_id,
        )

        try:
            while self._is_running:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                    self._queue.task_done()
                    yield item
                except asyncio.TimeoutError:
                    continue
        finally:
            self._is_running = False
            yield LifecycleEvent(
                event_type=EventType.SOURCE_DOWN,
                structure_id=self.structure_id,
                source_id=self.source_id,
                reason="HTTP source stopped",
            )

    def stop(self) -> None:
        super().stop()
