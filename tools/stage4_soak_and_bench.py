"""Stage 4 Live MQTT E2E (J.7), Throughput Benchmark (J.11), and 1-Hour Soak Test (J.10) Runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import paho.mqtt.client as mqtt
import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.detectors.registry import StructureRegistry
from app.ingestion.base import EventType, Frame, LifecycleEvent
from app.ingestion.mqtt_source import MQTTSource
from app.ingestion.protocol import encode_frame_binary
from app.schemas.structure_config import SensorConfig, SensorType, StructureConfig


def _load_kw51_cfg() -> StructureConfig:
    return StructureConfig.from_yaml(REPO_ROOT / "configs" / "kw51.yaml")


def _load_kw51_sample_pool() -> np.ndarray:
    h5_path = REPO_ROOT / "data" / "kw51" / "cache_50hz" / "ambient_201810_50hz.h5"
    if h5_path.is_file():
        with h5py.File(h5_path, "r") as hf:
            first_key = sorted(hf["recordings"].keys())[0]
            return hf["recordings"][first_key][:]
    # Fallback deterministic wave if HDF5 absent
    t = np.arange(15000) / 50.0
    return np.column_stack([
        0.002 * np.sin(2 * np.pi * 0.51 * t),
        0.0015 * np.sin(2 * np.pi * 1.23 * t),
        0.002 * np.sin(2 * np.pi * 1.88 * t),
        0.0018 * np.sin(2 * np.pi * 2.42 * t),
        np.full_like(t, 12.0),
    ]).astype(np.float32)


def run_benchmark_j11() -> Dict[str, Any]:
    """J.11: Measure max sustained ingestion + quality + detector throughput (sensors * fs)."""
    proc = psutil.Process(os.getpid())
    results = []
    base_cfg = _load_kw51_cfg()
    t = np.arange(500) / 50.0

    for n_sensors, fs_hz in [(5, 50.0), (16, 100.0), (32, 200.0), (64, 200.0)]:
        sensors = []
        for i in range(n_sensors - 1):
            sensors.append(
                SensorConfig(
                    id=f"s_{i}",
                    name=f"Sensor_{i}",
                    type=SensorType.ACCEL,
                    axis="ay" if i % 4 == 1 else "az",
                    channel_index=i,
                )
            )
        sensors.append(SensorConfig(id="t_env", name="Temp_Env", type=SensorType.TEMP, axis="temp", channel_index=n_sensors - 1))

        cfg = base_cfg.model_copy(deep=True)
        cfg.structure_id = f"bench_{n_sensors}_{int(fs_hz)}"
        cfg.fs = fs_hz
        cfg.native_fs = fs_hz
        cfg.detector_fs = fs_hz
        cfg.sensors = sensors
        cfg.detectors = [d for d in cfg.detectors if d.type == "spectral_tracking"]
        if cfg.detectors:
            cfg.detectors[0].params["fs"] = fs_hz

        reg = StructureRegistry()
        rt = reg.add(cfg, load_artifacts=False)
        n_rows = max(50, int(round(fs_hz)))  # 1s block per frame
        block = np.random.default_rng(1).normal(0, 0.002, size=(n_rows, n_sensors)).astype(np.float32)
        block[:, -1] = 14.0
        n_frames = 40

        t0 = time.perf_counter()
        base_ns = time.time_ns()
        for seq in range(n_frames):
            fr = Frame(
                structure_id=cfg.structure_id,
                sensor_ids=[s.id for s in sensors],
                t0=base_ns + seq * 1_000_000_000,
                fs=fs_hz,
                samples=block,
                seq=seq,
                source_id="bench",
                flags={"received_wall_ns": base_ns + seq * 1_000_000_000},
            )
            rt.push_ingestion_item(fr, spool_write=False)
        dt = max(time.perf_counter() - t0, 1e-6)
        samples_per_sec = (n_frames * n_rows) / dt
        sensor_samples_per_sec = samples_per_sec * n_sensors
        results.append(
            {
                "n_sensors": n_sensors,
                "fs_hz": fs_hz,
                "nominal_sensors_x_fs": n_sensors * fs_hz,
                "frames_per_sec": round(n_frames / dt, 2),
                "sustained_samples_per_sec": round(samples_per_sec, 1),
                "sustained_sensor_samples_per_sec": round(sensor_samples_per_sec, 1),
                "realtime_speedup": round(samples_per_sec / fs_hz, 2),
                "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
            }
        )

    out = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count_logical": os.cpu_count(),
        },
        "benchmarks": results,
    }
    print(json.dumps(out, indent=2))
    return out


async def run_mqtt_e2e_j7(host: str, port: int, test_broker_restart: bool = False) -> Dict[str, Any]:
    """J.7: Live MQTT E2E including binary/JSON frames, LWT offline event, and optional broker restart."""
    cfg = _load_kw51_cfg()
    pool = _load_kw51_sample_pool()
    reg = StructureRegistry()
    rt = reg.add(cfg, load_artifacts=True)
    sids = [s.id for s in cfg.sensors]

    src = MQTTSource(
        channel_names=sids,
        structure_id="kw51",
        host=host,
        port=port,
        qos=1,
        fs=50.0,
        keepalive=3,
    )

    events_seen: List[str] = []
    frames_seen = 0

    async def _consume():
        nonlocal frames_seen
        async for item in src.stream():
            if isinstance(item, LifecycleEvent):
                events_seen.append(item.event_type.value if hasattr(item.event_type, "value") else str(item.event_type))
                rt.handle_lifecycle_event(item)
            elif isinstance(item, Frame):
                frames_seen += 1
                rt.push_ingestion_item(item, spool_write=False)

    consumer_task = asyncio.create_task(_consume())
    for _ in range(40):
        if src.broker_connected:
            break
        await asyncio.sleep(0.1)
    if not src.broker_connected:
        consumer_task.cancel()
        raise RuntimeError(f"Could not connect MQTTSource to {host}:{port}")

    # Publish 10 binary frames from a client with LWT configured
    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="j7-pub-lwt")
    pub.will_set("shm/kw51/status", json.dumps({"status": "offline", "reason": "lwt"}), qos=1, retain=False)
    pub.connect(host, port, keepalive=3)
    pub.loop_start()
    pub.publish("shm/kw51/status", json.dumps({"status": "online"}), qos=1).wait_for_publish(timeout=3.0)

    for seq in range(10):
        fr = Frame(
            structure_id="kw51",
            sensor_ids=sids,
            t0=time.time_ns(),
            fs=50.0,
            samples=pool[(seq * 50) % 10000 : (seq * 50) % 10000 + 50].astype(np.float32),
            seq=seq,
            source_id="j7-pub",
        )
        pub.publish("shm/kw51/raw", encode_frame_binary(fr), qos=1).wait_for_publish(timeout=3.0)
        await asyncio.sleep(0.05)

    # Simulate ungraceful disconnect (abrupt socket drop) -> broker triggers LWT after keepalive
    try:
        pub._sock_close()
    except Exception:
        pass
    pub.loop_stop()

    lwt_received = False
    for _ in range(80):  # up to 8s
        if "SOURCE_DOWN" in events_seen:
            lwt_received = True
            break
        await asyncio.sleep(0.1)

    broker_restarted_ok = False
    if test_broker_restart:
        subprocess.run(["docker", "compose", "--profile", "test", "restart", "mosquitto"], check=True)
        await asyncio.sleep(2.5)
        for _ in range(60):
            if src.broker_connected:
                break
            await asyncio.sleep(0.25)
        pub2 = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="j7-pub-after")
        pub2.connect(host, port, keepalive=10)
        pub2.loop_start()
        pub2.publish("shm/kw51/status", json.dumps({"status": "online"}), qos=1).wait_for_publish(timeout=3.0)
        fr_after = Frame(
            structure_id="kw51",
            sensor_ids=sids,
            t0=time.time_ns(),
            fs=50.0,
            samples=pool[:50].astype(np.float32),
            seq=100,
            source_id="j7-pub",
        )
        pub2.publish("shm/kw51/raw", encode_frame_binary(fr_after), qos=1).wait_for_publish(timeout=3.0)
        await asyncio.sleep(0.3)
        pub2.loop_stop()
        pub2.disconnect()
        broker_restarted_ok = frames_seen >= 11

    src.stop()
    await asyncio.sleep(0.2)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    res = {
        "frames_received": frames_seen,
        "events_seen": events_seen,
        "lwt_detected": lwt_received,
        "broker_restarted_ok": broker_restarted_ok,
        "frames_processed_pipeline": rt.frames_processed,
    }
    print(json.dumps(res, indent=2))
    return res


async def run_soak_j10(host: str, port: int, duration_sec: int, out_json: Optional[str] = None) -> Dict[str, Any]:
    """J.10: Real-time 1x MQTT soak test logging effective rate, lag, dropped frames, and RSS slope +- SE."""
    cfg = _load_kw51_cfg()
    pool = _load_kw51_sample_pool()
    reg = StructureRegistry()
    rt = reg.add(cfg, load_artifacts=True)
    sids = [s.id for s in cfg.sensors]
    proc = psutil.Process(os.getpid())

    src = MQTTSource(
        channel_names=sids,
        structure_id="kw51",
        host=host,
        port=port,
        qos=1,
        fs=50.0,
        keepalive=15,
    )

    async def _consume():
        async for item in src.stream():
            if isinstance(item, LifecycleEvent):
                rt.handle_lifecycle_event(item)
            elif isinstance(item, Frame):
                rt.push_ingestion_item(item, spool_write=False)

    consumer_task = asyncio.create_task(_consume())
    for _ in range(50):
        if src.broker_connected:
            break
        await asyncio.sleep(0.1)
    if not src.broker_connected:
        consumer_task.cancel()
        raise RuntimeError(f"Could not connect MQTTSource to {host}:{port}")

    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"j10-soak-{os.getpid()}")
    pub.connect(host, port, keepalive=30)
    pub.loop_start()
    pub.publish("shm/kw51/status", json.dumps({"status": "online", "source_id": "j10-soak"}), qos=1, retain=True)

    rss_samples: List[tuple[float, float]] = []  # (elapsed_sec, rss_kb)
    lag_samples: List[float] = []
    start_mono = time.monotonic()
    next_deadline = start_mono
    seq = 0
    last_log_t = start_mono
    n_pool = len(pool)

    try:
        while True:
            now_m = time.monotonic()
            elapsed = now_m - start_mono
            if elapsed >= duration_sec:
                break

            idx_start = (seq * 50) % max(1, n_pool - 50)
            chunk = pool[idx_start : idx_start + 50].astype(np.float32)
            fr = Frame(
                structure_id="kw51",
                sensor_ids=sids,
                t0=time.time_ns() - 1_000_000_000,
                fs=50.0,
                samples=chunk,
                seq=seq,
                source_id="j10-soak",
            )
            pub.publish("shm/kw51/raw", encode_frame_binary(fr), qos=1)
            seq += 1

            # Sample RSS & lag every 10s
            if seq == 1 or seq % 10 == 0:
                rss_kb = proc.memory_info().rss / 1024.0
                rss_samples.append((elapsed, rss_kb))
                lag_samples.append(rt.processing_lag_sec)

            if now_m - last_log_t >= 60.0:
                last_log_t = now_m
                print(
                    f"[SOAK t={elapsed:.0f}s/{duration_sec}s] sent={seq} processed={rt.frames_processed} "
                    f"fps={rt.effective_frames_per_sec:.3f} lag={rt.processing_lag_sec:.3f}s "
                    f"dropped={rt.dropped_frames_total} rss_mb={proc.memory_info().rss / (1024*1024):.2f}",
                    flush=True,
                )

            next_deadline += 1.0
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            else:
                await asyncio.sleep(0)

        await asyncio.sleep(0.5)
    finally:
        pub.loop_stop()
        pub.disconnect()
        src.stop()
        await asyncio.sleep(0.2)
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    total_elapsed = max(time.monotonic() - start_mono, 1e-6)
    effective_fps = rt.frames_processed / total_elapsed
    nominal_fps = 1.0

    # OLS linear regression on RSS_KB vs time_hours -> slope (KB/hr) +- SE
    slope_kb_per_hr = 0.0
    slope_se_kb_per_hr = 0.0
    if len(rss_samples) >= 3:
        x_hr = np.array([t / 3600.0 for t, _ in rss_samples], dtype=np.float64)
        y_kb = np.array([kb for _, kb in rss_samples], dtype=np.float64)
        xm = float(np.mean(x_hr))
        ym = float(np.mean(y_kb))
        sxx = float(np.sum((x_hr - xm) ** 2))
        if sxx > 1e-12:
            slope_kb_per_hr = float(np.sum((x_hr - xm) * (y_kb - ym)) / sxx)
            resid = y_kb - (ym + slope_kb_per_hr * (x_hr - xm))
            s2 = float(np.sum(resid ** 2) / max(len(x_hr) - 2, 1))
            slope_se_kb_per_hr = float(np.sqrt(s2 / sxx))

    lag_arr = np.asarray(lag_samples, dtype=np.float64) if lag_samples else np.array([0.0])
    report = {
        "duration_sec": round(total_elapsed, 2),
        "frames_sent": seq,
        "frames_processed": rt.frames_processed,
        "samples_processed": rt.samples_processed,
        "nominal_fps": nominal_fps,
        "effective_fps": round(effective_fps, 4),
        "dropped_frames": rt.dropped_frames_total + src.dropped_backpressure,
        "lag_mean_sec": round(float(np.mean(lag_arr)), 4),
        "lag_max_sec": round(float(np.max(lag_arr)), 4),
        "rss_start_mb": round(rss_samples[0][1] / 1024.0, 2) if rss_samples else 0.0,
        "rss_end_mb": round(rss_samples[-1][1] / 1024.0, 2) if rss_samples else 0.0,
        "rss_slope_kb_per_hour": round(slope_kb_per_hr, 2),
        "rss_slope_se_kb_per_hour": round(slope_se_kb_per_hr, 2),
        "n_rss_samples": len(rss_samples),
        "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
    }
    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4 Live MQTT & Throughput Harness")
    parser.add_argument("--mode", choices=["bench", "mqtt-e2e", "soak"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--duration-sec", type=int, default=3600)
    parser.add_argument("--broker-restart", action="store_true")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if args.mode == "bench":
        run_benchmark_j11()
    elif args.mode == "mqtt-e2e":
        asyncio.run(run_mqtt_e2e_j7(args.host, args.port, test_broker_restart=args.broker_restart))
    elif args.mode == "soak":
        asyncio.run(run_soak_j10(args.host, args.port, args.duration_sec, out_json=args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
