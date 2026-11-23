"""Production MQTT Telemetry Ingestion Source for Industrial IoT Gateways.

Stage 4 Protocol:
- Topics:
  - `shm/{structure_id}/raw`    (binary SHMF v1 float32 frames or JSON frames)
  - `shm/{structure_id}/meta`   (sensor list, fs, units, firmware metadata)
  - `shm/{structure_id}/status` (gateway online/offline status + Last Will)
- QoS 1 delivery, automatic reconnect with exponential backoff, optional TLS,
  and credentials loaded exclusively from environment settings.
- Backpressure: bounded internal queue drops oldest item and emits GAP if full.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import ssl
import time
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Union

import numpy as np

from app.core.config import settings
from app.ingestion.base import EventType, Frame, IngestionSource, IngestionStreamItem, LifecycleEvent
from app.ingestion.protocol import decode_frame_auto

logger = logging.getLogger("SHM.MQTTSource")


class MQTTSource(IngestionSource):
    """Production MQTT telemetry ingestion source with Stage 4 Frame & Lifecycle support."""

    def __init__(
        self,
        channel_names: List[str],
        structure_id: str = "bridge_demo_01",
        host: Optional[str] = None,
        port: Optional[int] = None,
        topic_prefix: Optional[str] = None,
        topic: Optional[str] = None,  # Legacy compat alias for raw topic
        qos: Optional[int] = None,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls_enabled: Optional[bool] = None,
        ca_certs: Optional[str] = None,
        keepalive: Optional[int] = None,
        queue_maxsize: int = 512,
        fs: float = 50.0,
        max_payload_bytes: Optional[int] = None,
    ) -> None:
        super().__init__(name="MQTT", structure_id=structure_id)
        self.channel_names = list(channel_names)
        self.num_channels = len(self.channel_names)
        self.fs = float(fs)

        # Credentials & broker connection come from env/settings unless explicitly overridden for tests
        self.host = host or settings.MQTT_HOST
        self.port = int(port if port is not None else settings.MQTT_PORT)
        self.qos = int(qos if qos is not None else settings.MQTT_QOS)
        self.client_id = client_id or f"shm_ingestor_{structure_id}_{id(self)}"
        self.username = username if username is not None else (settings.MQTT_USERNAME or None)
        self.password = password if password is not None else (settings.MQTT_PASSWORD or None)
        self.tls_enabled = tls_enabled if tls_enabled is not None else settings.MQTT_TLS_ENABLED
        self.ca_certs = ca_certs if ca_certs is not None else (settings.MQTT_CA_CERTS or None)
        self.keepalive = int(keepalive if keepalive is not None else settings.MQTT_KEEPALIVE)
        self.queue_maxsize = max(4, int(queue_maxsize))
        self.max_payload_bytes = int(max_payload_bytes or settings.MAX_FRAME_PAYLOAD_BYTES)

        prefix = topic_prefix or f"shm/{structure_id}"
        self.raw_topic = topic or f"{prefix}/raw"
        self.meta_topic = f"{prefix}/meta"
        self.status_topic = f"{prefix}/status"
        self.topic = self.raw_topic

        self._queue: Optional[asyncio.Queue[IngestionStreamItem]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending_items: Deque[IngestionStreamItem] = deque(maxlen=self.queue_maxsize)
        self._mqtt_client: Any = None
        self._use_mock: bool = False

        self.gateway_meta: Dict[str, Any] = {}
        self.gateway_status: str = "unknown"
        self.broker_connected: bool = False
        self.dropped_backpressure: int = 0
        self.malformed_count: int = 0

    def parse_payload(self, raw_data: Union[str, bytes, dict]) -> Optional[np.ndarray]:
        """Legacy helper returning 1D numpy vector for the first row of a parsed payload."""
        try:
            frame = decode_frame_auto(
                raw_data,
                default_structure_id=self.structure_id,
                default_sensor_ids=self.channel_names,
                default_fs=self.fs,
                max_bytes=self.max_payload_bytes,
            )
            return frame.samples[0]
        except Exception as exc:
            logger.debug(f"Failed to parse MQTT payload: {exc}")
            return None

    def _enqueue_threadsafe(self, item: IngestionStreamItem) -> bool:
        """Enqueue a Frame or LifecycleEvent safely from any thread with drop-oldest backpressure."""
        if self._queue is not None and self._loop is not None and not self._loop.is_closed():
            def _put() -> None:
                assert self._queue is not None
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self.dropped_backpressure += 1
                    except asyncio.QueueEmpty:
                        pass
                    # Flag a GAP due to queue overflow if item is a Frame
                    if isinstance(item, Frame):
                        item.flags["gap"] = True
                        item.flags["backpressure_dropped"] = self.dropped_backpressure
                self._queue.put_nowait(item)

            self._loop.call_soon_threadsafe(_put)
            return True
        else:
            if len(self._pending_items) == self._pending_items.maxlen:
                self.dropped_backpressure += 1
            self._pending_items.append(item)
            return True

    def push_raw_message(self, payload: Union[str, bytes, dict], topic: Optional[str] = None) -> bool:
        """Parse and push a raw message (used by paho callbacks or unit tests)."""
        tgt = topic or self.raw_topic
        if tgt == self.meta_topic:
            try:
                meta = json.loads(payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload)
                self.gateway_meta = meta
                return True
            except Exception:
                self.malformed_count += 1
                return False

        if tgt == self.status_topic:
            try:
                raw_txt = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
                if raw_txt.strip().startswith("{"):
                    st_doc = json.loads(raw_txt)
                    st = str(st_doc.get("status", "unknown")).lower()
                else:
                    st = raw_txt.strip().lower()
                self.gateway_status = st
                ev_type = EventType.SOURCE_UP if st == "online" else EventType.SOURCE_DOWN
                self._enqueue_threadsafe(
                    LifecycleEvent(
                        event_type=ev_type,
                        structure_id=self.structure_id,
                        source_id=self.source_id,
                        reason=f"Gateway reported status '{st}' (LWT/status)",
                        details={"gateway_status": st},
                    )
                )
                return True
            except Exception:
                self.malformed_count += 1
                return False

        try:
            frame = decode_frame_auto(
                payload,
                default_structure_id=self.structure_id,
                default_sensor_ids=self.channel_names,
                default_fs=self.fs,
                max_bytes=self.max_payload_bytes,
            )
            if frame.structure_id != self.structure_id:
                self.malformed_count += 1
                logger.warning(f"Rejected MQTT frame for mismatched structure_id={frame.structure_id}")
                return False
            self._enqueue_threadsafe(frame)
            return True
        except Exception as exc:
            self.malformed_count += 1
            logger.debug(f"Rejected malformed MQTT payload: {exc}")
            return False

    def _connect_paho_client(self) -> None:
        try:
            import paho.mqtt.client as mqtt

            kwargs: Dict[str, Any] = {"client_id": self.client_id}
            if hasattr(mqtt, "CallbackAPIVersion"):
                kwargs["callback_api_version"] = mqtt.CallbackAPIVersion.VERSION2

            client = mqtt.Client(**kwargs)
            client.reconnect_delay_set(min_delay=1, max_delay=30)

            if self.username and self.password:
                client.username_pw_set(self.username, self.password)

            if self.tls_enabled:
                client.tls_set(ca_certs=self.ca_certs, cert_reqs=ssl.CERT_REQUIRED)

            # Configure Last Will for this ingestor's status if broker dies/client drops
            lwt_payload = json.dumps({"status": "offline", "source_id": self.source_id, "ts_ns": time.time_ns()})
            client.will_set(f"shm/{self.structure_id}/ingestor_status", payload=lwt_payload, qos=self.qos, retain=True)

            def on_connect(c, userdata, flags, reason_code, properties=None):
                rc_val = getattr(reason_code, "value", reason_code)
                if rc_val == 0:
                    self.broker_connected = True
                    logger.info(f"MQTT connected to {self.host}:{self.port}; subscribing QoS={self.qos} to {self.raw_topic}")
                    c.subscribe([(self.raw_topic, self.qos), (self.meta_topic, self.qos), (self.status_topic, self.qos)])
                    self._enqueue_threadsafe(
                        LifecycleEvent(
                            event_type=EventType.SOURCE_UP,
                            structure_id=self.structure_id,
                            source_id=self.source_id,
                            reason=f"Connected to MQTT broker {self.host}:{self.port}",
                        )
                    )
                else:
                    logger.warning(f"MQTT connect failed with rc={reason_code}")

            def on_disconnect(c, userdata, disconnect_flags=None, reason_code=None, properties=None):
                self.broker_connected = False
                if self._is_running:
                    logger.warning(f"MQTT broker disconnected (rc={reason_code}); reconnecting with backoff...")
                    self._enqueue_threadsafe(
                        LifecycleEvent(
                            event_type=EventType.SOURCE_DOWN,
                            structure_id=self.structure_id,
                            source_id=self.source_id,
                            reason=f"MQTT disconnected (rc={reason_code})",
                        )
                    )

            def on_message(c, userdata, msg):
                self.push_raw_message(msg.payload, topic=msg.topic)

            client.on_connect = on_connect
            client.on_disconnect = on_disconnect
            client.on_message = on_message

            try:
                client.connect_async(self.host, self.port, keepalive=self.keepalive)
                client.loop_start()
                self._mqtt_client = client
            except Exception as conn_err:
                logger.warning(f"MQTT broker unreachable ({conn_err}); operating in queued mode until reachable.")
                self._use_mock = True
        except ImportError:
            logger.warning("paho-mqtt not installed; MQTTSource operating in queued mode.")
            self._use_mock = True

    async def stream(self) -> AsyncIterator[IngestionStreamItem]:
        """Yield decoded MQTT Frames and LifecycleEvents continuously."""
        self._is_running = True
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)

        # Drain any pre-buffered messages
        while self._pending_items:
            item = self._pending_items.popleft()
            if not self._queue.full():
                self._queue.put_nowait(item)

        self._connect_paho_client()

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
            self._disconnect_client()

    def _disconnect_client(self) -> None:
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception as err:
                logger.warning(f"Error disconnecting MQTT client: {err}")
            self._mqtt_client = None
        self.broker_connected = False

    def stop(self) -> None:
        super().stop()
        self._disconnect_client()
