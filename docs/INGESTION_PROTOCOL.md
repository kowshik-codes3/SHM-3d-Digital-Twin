# SHM Digital Twin — Ingestion Wire Protocol (`docs/INGESTION_PROTOCOL.md`)

All telemetry sources (`SimulatorSource`, `ReplaySource`, `MQTTSource`, `HTTPSource`, `KafkaSource`) emit a unified async-iterator stream of `Frame` chunks and `LifecycleEvent` records before entering edge resampling and the `DataQualityLayer`.

## 1. Canonical Internal Schema (`Frame` & `LifecycleEvent`)

### `Frame`
| Field | Type | Description |
| :--- | :--- | :--- |
| `structure_id` | `str` | Unique civil asset identifier (e.g., `kw51`, `bridge_demo_01`) |
| `sensor_ids` | `List[str]` | Ordered channel identifiers matching columns of `samples` |
| `t0` | `int` | UTC timestamp of sample 0 in **nanoseconds** since Unix epoch |
| `fs` | `float` | Sampling rate in Hz |
| `samples` | `float32[n_samples, n_sensors]` | Time-major sensor matrix |
| `seq` | `int` | Monotonic per-source frame sequence counter |
| `source_id` | `str` | Gateway / publisher identifier |
| `flags` | `Dict[str, Any]` | Optional metadata (`recording_id`, `historical_replay`, `channel_t0_ns`, `offline_sensors`) |

### `LifecycleEvent`
`event_type` is one of:
- `RECORDING_START`
- `RECORDING_END`
- `SOURCE_UP`
- `SOURCE_DOWN`
- `GAP`

## 2. MQTT Topic Hierarchy

- **Data topic:** `shm/{structure_id}/raw` — Compact binary (`SHMF` v1) or JSON `Frame` payloads. Default **QoS 1**.
- **Metadata topic:** `shm/{structure_id}/meta` — Retained JSON describing gateway identity, active `sensor_ids`, native `fs`, engineering units, and firmware version.
- **Status / LWT topic:** `shm/{structure_id}/status` — Retained JSON `{"status": "online" | "offline", "source_id": ..., "reason": ...}`. Gateways MUST register a broker Last Will & Testament (`status="offline"`) on disconnect.

Recommended production settings (from `.env` / environment):
- `MQTT_QOS=1`, `MQTT_KEEPALIVE=30`
- Optional TLS (`MQTT_TLS_ENABLED=true`, `MQTT_CA_CERTS`, client cert/key) and username/token auth (`MQTT_USERNAME`, `MQTT_PASSWORD`).

## 3. Compact Binary Framing (`SHMF` v1, Little-Endian)

Implemented in `backend/app/ingestion/protocol.py` (`encode_frame_binary` / `decode_frame_binary`) using struct `<4sBBHQfIIQH` (38-byte fixed header + UTF-8 JSON metadata + row-major `float32` payload):

| Offset | Size (Bytes) | Type | Description |
| :--- | :--- | :--- | :--- |
| `0` | `4` | `4s` | Magic header `b"SHMF"` |
| `4` | `1` | `B` | Protocol version (`1`) |
| `5` | `1` | `B` | Bitflags (`0x01`=gap, `0x02`=late, `0x04`=temp_interp) |
| `6` | `2` | `H` | Reserved (`0`) |
| `8` | `8` | `Q` | `t0_ns` — UTC nanoseconds (`uint64`) |
| `16` | `4` | `f` | `fs` — sampling rate Hz (`float32`) |
| `20` | `4` | `I` | `n_samples` (`uint32`) |
| `24` | `4` | `I` | `n_sensors` (`uint32`) |
| `28` | `8` | `Q` | `seq` — monotonic frame sequence counter (`uint64`) |
| `36` | `2` | `H` | `meta_len` (`uint16`) |
| `38` | `meta_len` | UTF-8 JSON | `{"structure_id": ..., "source_id": ..., "sensor_ids": [...], "flags": {...}}` |
| `38 + meta_len` | `4 * N * C` | `float32[]` | Row-major little-endian matrix `[n_samples, n_sensors]` |

## 4. JSON Debug Format

```json
{
  "structure_id": "kw51",
  "sensor_ids": ["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az", "tVL"],
  "t0": 1559347200000000000,
  "fs": 50.0,
  "seq": 42,
  "source_id": "gw-kw51-01",
  "samples": [[0.001, -0.002, 0.004, 0.001, 16.5]],
  "flags": {}
}
```

## 5. HTTP Batch Endpoint

`POST /api/v1/structures/{structure_id}/frames`
- **Headers**:
  - `X-API-Key: <per-structure or SHM_API_KEY_DEFAULT>` (constant-time verification)
  - `Idempotency-Key: <unique-batch-key>` (TTL-cached deduplication so gateway retries never duplicate samples)
  - `Content-Type`: `application/octet-stream` (`SHMF` binary) or `application/json` (single `Frame` or `{"frames": [...]}`)
- **Limits**:
  - Max payload size: `MAX_FRAME_PAYLOAD_BYTES` (default 1 MB -> HTTP `413`)
  - Per-structure rate limit: `HTTP_RATE_LIMIT_RPS` (HTTP `429`)

## 6. Resampling Policy (Native `fs` -> Detector `fs`)

- Anti-alias filtering & decimation runs at the ingestion edge (`app/ingestion/resampler.py`) using configurable filter (`butter` / `cheby1` / `none`, order, cutoff).
- For KW51 (`native_fs = 25600 / 31 ≈ 825.8065 Hz`, `detector_fs = 50.0 Hz`), the exact rational factor `31 / 512` matches the Stage 0 offline HDF5 cache (`max_abs_diff < 1e-9` in `block_zero_phase` mode).
- Irregular / jittery sample timestamps are resampled onto a uniform grid, interpolating only small gaps `<= max_gap_interpolate_sec` (default `0.25 s`) and strictly emitting `NaN` across longer gaps rather than fabricating vibrations.
