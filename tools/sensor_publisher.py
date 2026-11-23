"""Reference Sensor Gateway / Replay Publisher CLI (tools/sensor_publisher.py).

Streams a replay file (HDF5 / Parquet / CSV / MAT) to either MQTT or HTTP
`/api/v1/structures/{id}/frames` at a configurable real-time speed multiplier.

Supports compact binary (`SHMF` float32 little-endian) or JSON framing,
Meta/Status Last-Will MQTT lifecycle publishing, and HTTP idempotency keys.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Ensure backend package is importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ingestion.base import EventType, Frame, LifecycleEvent
from app.ingestion.protocol import encode_frame_binary, encode_frame_json
from app.ingestion.replay_source import ReplaySource


def _http_post_frame(
    url: str,
    api_key: str,
    frame: Frame,
    use_binary: bool,
    timeout_sec: float = 5.0,
) -> int:
    idem_key = f"{frame.source_id}:{frame.seq}:{frame.t0}"
    if use_binary:
        body = encode_frame_binary(frame)
        content_type = "application/octet-stream"
    else:
        body = encode_frame_json(frame)
        content_type = "application/json"

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": content_type,
            "X-API-Key": api_key,
            "Idempotency-Key": idem_key,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return int(resp.status)


async def publish_mqtt(
    source: ReplaySource,
    structure_id: str,
    broker_host: str,
    broker_port: int,
    username: Optional[str],
    password: Optional[str],
    use_binary: bool,
    qos: int,
    max_frames: Optional[int],
    realtime_wall_timestamps: bool,
    tls_ca: Optional[str] = None,
) -> Dict[str, Any]:
    import paho.mqtt.client as mqtt

    base_topic = f"shm/{structure_id}"
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"shm-pub-{structure_id}-{os.getpid()}",
        clean_session=True,
    )
    if username:
        client.username_pw_set(username, password or "")
    if tls_ca:
        client.tls_set(ca_certs=tls_ca)

    lwt_payload = json.dumps({"status": "offline", "source_id": source.source_id, "reason": "lwt"})
    client.will_set(f"{base_topic}/status", payload=lwt_payload, qos=1, retain=True)
    client.connect(broker_host, broker_port, keepalive=15)
    client.loop_start()

    meta_payload = json.dumps(
        {
            "structure_id": structure_id,
            "sensor_ids": source.sensor_ids,
            "fs": source.fs,
            "units": "g",
            "firmware": "ref-publisher-1.0",
        }
    )
    client.publish(f"{base_topic}/meta", meta_payload, qos=1, retain=True)
    client.publish(
        f"{base_topic}/status",
        json.dumps({"status": "online", "source_id": source.source_id}),
        qos=1,
        retain=True,
    )

    sent = 0
    errors = 0
    t_start = time.monotonic()
    try:
        async for item in source:
            if isinstance(item, LifecycleEvent):
                continue
            assert isinstance(item, Frame)
            if realtime_wall_timestamps:
                item.t0 = time.time_ns() - int(round((item.samples.shape[0] / item.fs) * 1e9))
            payload = encode_frame_binary(item) if use_binary else encode_frame_json(item)
            info = client.publish(f"{base_topic}/raw", payload, qos=qos)
            info.wait_for_publish(timeout=5.0)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                errors += 1
            sent += 1
            if max_frames and sent >= max_frames:
                break
    finally:
        client.publish(
            f"{base_topic}/status",
            json.dumps({"status": "offline", "source_id": source.source_id, "reason": "graceful_exit"}),
            qos=1,
            retain=True,
        )
        time.sleep(0.05)
        client.loop_stop()
        client.disconnect()

    elapsed = max(time.monotonic() - t_start, 1e-6)
    return {"frames_sent": sent, "errors": errors, "elapsed_sec": elapsed, "effective_fps": sent / elapsed}


async def publish_http(
    source: ReplaySource,
    structure_id: str,
    http_url: str,
    api_key: str,
    use_binary: bool,
    max_frames: Optional[int],
    realtime_wall_timestamps: bool,
) -> Dict[str, Any]:
    endpoint = f"{http_url.rstrip('/')}/api/v1/structures/{structure_id}/frames"
    sent = 0
    errors = 0
    t_start = time.monotonic()
    async for item in source:
        if isinstance(item, LifecycleEvent):
            continue
        assert isinstance(item, Frame)
        if realtime_wall_timestamps:
            item.t0 = time.time_ns() - int(round((item.samples.shape[0] / item.fs) * 1e9))
        try:
            code = await asyncio.to_thread(_http_post_frame, endpoint, api_key, item, use_binary)
            if code >= 400:
                errors += 1
        except urllib.error.URLError:
            errors += 1
        sent += 1
        if max_frames and sent >= max_frames:
            break
    elapsed = max(time.monotonic() - t_start, 1e-6)
    return {"frames_sent": sent, "errors": errors, "elapsed_sec": elapsed, "effective_fps": sent / elapsed}


def main() -> int:
    parser = argparse.ArgumentParser(description="SHM Reference Sensor Publisher")
    parser.add_argument("--structure-id", required=True, help="Target structure ID (e.g. kw51)")
    parser.add_argument("--file", required=True, help="Path to HDF5, Parquet, CSV, or MAT replay file")
    parser.add_argument("--transport", choices=["mqtt", "http"], default="mqtt")
    parser.add_argument("--format", choices=["binary", "json"], default="binary")
    parser.add_argument("--speed", default="1.0", help="Speed multiplier (e.g. 1.0, 10.0, or max)")
    parser.add_argument("--loop", action="store_true", help="Loop recordings continuously")
    parser.add_argument("--chunk-samples", type=int, default=50, help="Samples per frame (default 50 = 1s at 50 Hz)")
    parser.add_argument("--fs", type=float, default=50.0, help="Sampling rate Hz")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--realtime-ts", action="store_true", help="Stamp frames with current UTC wall clock t0")
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-user", default=os.getenv("MQTT_USERNAME", ""))
    parser.add_argument("--mqtt-pass", default=os.getenv("MQTT_PASSWORD", ""))
    parser.add_argument("--mqtt-qos", type=int, default=1)
    parser.add_argument("--tls-ca", default=os.getenv("MQTT_CA_CERTS", ""))
    parser.add_argument("--http-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.getenv("SHM_API_KEY_DEFAULT", "dev-shm-api-key"))
    args = parser.parse_args()

    speed_val: Any = "max" if str(args.speed).lower() == "max" else float(args.speed)
    source = ReplaySource(
        file_path=args.file,
        structure_id=args.structure_id,
        fs=args.fs,
        chunk_samples=args.chunk_samples,
        speed_multiplier=speed_val,
        loop=args.loop,
    )

    if args.transport == "mqtt":
        stats = asyncio.run(
            publish_mqtt(
                source=source,
                structure_id=args.structure_id,
                broker_host=args.mqtt_host,
                broker_port=args.mqtt_port,
                username=args.mqtt_user or None,
                password=args.mqtt_pass or None,
                use_binary=(args.format == "binary"),
                qos=args.mqtt_qos,
                max_frames=args.max_frames,
                realtime_wall_timestamps=args.realtime_ts,
                tls_ca=args.tls_ca or None,
            )
        )
    else:
        stats = asyncio.run(
            publish_http(
                source=source,
                structure_id=args.structure_id,
                http_url=args.http_url,
                api_key=args.api_key,
                use_binary=(args.format == "binary"),
                max_frames=args.max_frames,
                realtime_wall_timestamps=args.realtime_ts,
            )
        )

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
