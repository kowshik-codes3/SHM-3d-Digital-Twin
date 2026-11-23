"""Production Apache Kafka / Redpanda Telemetry Ingestion Source.

Consumes streaming telemetry records from Kafka topics, decodes JSON payloads,
assembles ordered 1D multi-channel sensor vectors, and feeds them into the structure runtime.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from app.ingestion.base import EventType, Frame, IngestionSource, IngestionStreamItem, LifecycleEvent
import time
from typing import AsyncIterator

logger = logging.getLogger("SHM.KafkaSource")


class KafkaSource(IngestionSource):
    """Production Kafka/Redpanda telemetry ingestion source.

    Parameters
    ----------
    channel_names : List[str]
        Ordered list of sensor channel names expected by the model.
    bootstrap_servers : str
        Kafka broker connection string (default 'localhost:9092').
    topic : str
        Kafka topic name (default 'shm-telemetry').
    group_id : str
        Consumer group ID (default 'shm-consumer-group').
    auto_offset_reset : str
        Offset reset strategy ('latest' or 'earliest').
    queue_maxsize : int
        Maximum buffer size for parsed frames.
    """

    def __init__(
        self,
        channel_names: List[str],
        bootstrap_servers: str = "localhost:9092",
        topic: str = "shm-telemetry",
        group_id: str = "shm-consumer-group",
        auto_offset_reset: str = "latest",
        queue_maxsize: int = 1000,
    ) -> None:
        super().__init__(name="Kafka")
        self.channel_names = list(channel_names)
        self.num_channels = len(self.channel_names)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.queue_maxsize = queue_maxsize

        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=queue_maxsize)
        self._worker_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._use_mock: bool = False

    def parse_payload(self, raw_data: Union[str, bytes, dict]) -> Optional[np.ndarray]:
        """Parse raw Kafka record value into an ordered 1D float32 numpy vector."""
        try:
            if isinstance(raw_data, (bytes, bytearray)):
                data = json.loads(raw_data.decode("utf-8"))
            elif isinstance(raw_data, str):
                data = json.loads(raw_data)
            elif isinstance(raw_data, dict):
                data = raw_data
            else:
                return None

            # 1. Direct array in 'values' or 'frame'
            if "values" in data and isinstance(data["values"], list):
                vals = [float(v) for v in data["values"]]
                if len(vals) == self.num_channels:
                    return np.array(vals, dtype=np.float32)

            if "frame" in data and isinstance(data["frame"], list):
                vals = [float(v) for v in data["frame"]]
                if len(vals) == self.num_channels:
                    return np.array(vals, dtype=np.float32)

            # 2. Channel dictionary
            ch_map = data.get("channels", data)
            if isinstance(ch_map, dict):
                vals = []
                for name in self.channel_names:
                    if name in ch_map:
                        vals.append(float(ch_map[name]))
                    else:
                        return None
                return np.array(vals, dtype=np.float32)

        except Exception as exc:
            logger.debug(f"Failed to parse Kafka payload: {exc}")
            return None

        return None

    def push_raw_message(self, payload: Union[str, bytes, dict]) -> bool:
        """Push a raw message into the queue (used by live consumers or test mocks)."""
        frame = self.parse_payload(payload)
        if frame is None:
            return False

        try:
            self._queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            logger.warning("Kafka ingestion queue full; dropping frame.")
            return False

    async def _worker_loop(self, emit_callback: Callable[[np.ndarray], None]) -> None:
        """Consume parsed frames from the buffer queue and pass to emit_callback."""
        while self._is_running:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                emit_callback(frame)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as err:
                logger.error(f"Error in Kafka worker loop: {err}")

    async def _consumer_loop(self) -> None:
        """Background loop using aiokafka if installed."""
        try:
            from aiokafka import AIOKafkaConsumer

            consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
            )
            await consumer.start()
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}, subscribed to {self.topic}")
            try:
                async for msg in consumer:
                    if not self._is_running:
                        break
                    self.push_raw_message(msg.value)
            finally:
                await consumer.stop()

        except ImportError:
            logger.info("aiokafka not installed; KafkaSource operating in buffered queue mode.")
            self._use_mock = True
        except Exception as err:
            logger.warning(f"Kafka broker connection failed ({err}). Operating in buffered queue mode.")
            self._use_mock = True

    async def stream(self) -> AsyncIterator[IngestionStreamItem]:
        """Yield queued Kafka payloads wrapped as Stage 4 Frames."""
        self._is_running = True
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        seq = 0
        yield LifecycleEvent(
            event_type=EventType.SOURCE_UP,
            structure_id=self.structure_id,
            source_id=self.source_id,
        )
        try:
            while self._is_running:
                try:
                    vec = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                    self._queue.task_done()
                    yield Frame(
                        structure_id=self.structure_id,
                        sensor_ids=self.channel_names,
                        t0=time.time_ns(),
                        fs=50.0,
                        samples=vec.reshape(1, -1),
                        seq=seq,
                        source_id=self.source_id,
                    )
                    seq += 1
                except asyncio.TimeoutError:
                    continue
        finally:
            self._is_running = False
            yield LifecycleEvent(
                event_type=EventType.SOURCE_DOWN,
                structure_id=self.structure_id,
                source_id=self.source_id,
            )

    async def start(self, emit_callback: Callable[[np.ndarray], None]) -> None:
        """Start the Kafka consumer and emission worker."""
        self._is_running = True
        self._worker_task = asyncio.create_task(self._worker_loop(emit_callback))
        self._consumer_task = asyncio.create_task(self._consumer_loop())

    def stop(self) -> None:
        """Halt Kafka consumer and worker task."""
        self._is_running = False
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        logger.info(f"Stopped KafkaSource for topic {self.topic}")
