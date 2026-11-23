# SHM Digital Twin — Field Gateway Integration Guide (`docs/GATEWAY_GUIDE.md`)

This guide specifies the contract for Raspberry Pi, Linux edge industrial PCs, or ESP32-class DAQ gateways publishing field sensor measurements into the SHM Digital Twin backend.

## 1. Time Synchronization & Timestamping

- **UTC Nanoseconds (`t0`)**: Stamp every chunk with the UTC capture timestamp of sample index `0` in **int64 nanoseconds since Unix epoch**.
- **NTP / PTP Discipline**: Run `chrony` or `systemd-timesyncd` (Raspberry Pi / Linux) or SNTP every 60–300 s (ESP32) to hold gateway clock offset well inside `clock_offset_max_sec` (`< 2.0 s`, target `< 50 ms`) and clock drift inside `clock_drift_max_ppm` (`< 500 ppm`).
- **Simultaneous Sampling**: Sample all accelerometers connected to one structure off a synchronized ADC clock (`channel_t0_ns` skew `< 100 ms`). Never slew timestamps monotonically backwards during step corrections; if NTP steps backward, let `t0` reflect the new true UTC time so the backend flags a clean `clock_jump` once.

## 2. Batching & Cadence

- **Recommended Chunk Size**: Pack **1.0 second per message** (e.g., `50` samples at `50 Hz`, or `~826` samples at `825.8 Hz` if sending raw pre-resampled telemetry).
- **Encoding**: Prefer compact binary `SHMF` (`application/octet-stream` or MQTT `shm/{structure_id}/raw`) to keep payload overhead minimal (~1 KB per 1 s of 5 float32 channels). Use JSON only during lab bring-up.
- **Channel Order**: Match `sensor_ids` in the structure YAML config (`configs/<structure>.yaml`) and announce channel order + engineering units in retained MQTT `shm/{structure_id}/meta` on connect.

## 3. Connectivity, Last Will, and Offline Spooling

1. **MQTT Settings**:
   - Connect with **QoS 1**, keepalive `<= 30 s`, username/password + TLS enabled in production.
   - Register a **Last Will and Testament** on `shm/{structure_id}/status` with retained payload `{"status": "offline", "source_id": "<gw>", "reason": "lwt"}`.
   - Immediately publish `{"status": "online", ...}` (retained) upon connect and on reconnect.
2. **Offline Local Ring Buffer**:
   - When the broker or HTTP endpoint is unreachable (`ping`/`CONNACK` lost), spool unsent `SHMF` frames to local SD/flash storage in bounded FIFO segments (with `seq` incrementing monotonically — **never reset `seq` during an outage**).
   - Flush buffered historical frames with an `Idempotency-Key` (`{source_id}:{seq}:{t0}`) once reconnected. Note: frames older than `late_arrival_tolerance_sec` should either preserve chronological backfill order before newer live frames or trigger explicit gap accounting.
3. **Sensor Unplug / ADC Open-Circuit**:
   - If a single accelerometer wire opens or stops responding at the ADC/SPI bus, list its ID in `flags.offline_sensors` or emit `NaN` in that sensor's column while continuing to stream healthy channels. The backend marks only that channel `OFFLINE`/`SENSOR_FAULT` and keeps structural monitoring active.

## 4. Reference CLI Gateway (`tools/sensor_publisher.py`)

Test ingestion pipelines locally against Mosquitto or HTTP before field deployment:

```bash
# MQTT binary replay at real-time (1x) with live UTC timestamps
python tools/sensor_publisher.py --structure-id kw51 --file data/kw51/cache_50hz/ambient_201906_50hz.h5 --transport mqtt --format binary --speed 1.0 --realtime-ts

# Authenticated HTTP batch push with automatic Idempotency-Key header
python tools/sensor_publisher.py --structure-id kw51 --file data/kw51/cache_50hz/ambient_201906_50hz.h5 --transport http --format binary --speed 10.0 --api-key dev-shm-api-key
```
