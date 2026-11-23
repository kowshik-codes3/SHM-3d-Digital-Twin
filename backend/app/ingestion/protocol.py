"""Wire-protocol Binary and JSON codecs for SHM Ingestion Frames.

Binary Wire Format (`SHMF` v1, Little-Endian):
  Offset  Size   Field
  0       4      Magic bytes: b"SHMF"
  4       1      Protocol version (uint8 = 1)
  5       1      Bitflags (uint8: bit0=gap, bit1=late, bit2=temp_interp)
  6       2      Reserved (uint16 = 0)
  8       8      t0_ns (uint64 UTC nanoseconds)
  16      4      fs (float32 Hz)
  20      4      n_samples (uint32)
  24      4      n_sensors (uint32)
  28      8      seq (uint64)
  36      2      meta_len (uint16 bytes)
  38      meta_len  UTF-8 JSON header: {"structure_id": ..., "source_id": ..., "sensor_ids": [...], "flags": {...}}
  38+M    4*N*C  Row-major little-endian float32 matrix [n_samples, n_sensors]
"""

from __future__ import annotations

import json
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.ingestion.base import Frame

MAGIC = b"SHMF"
VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sBBHQfIIQH")  # 4+1+1+2+8+4+4+4+8+2 = 38 bytes
HEADER_SIZE = _HEADER_STRUCT.size


def encode_frame_binary(frame: Frame) -> bytes:
    """Encode a Frame into the compact little-endian SHMF v1 binary format."""
    meta = {
        "structure_id": frame.structure_id,
        "source_id": frame.source_id,
        "sensor_ids": frame.sensor_ids,
        "flags": frame.flags,
    }
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    if len(meta_bytes) > 65535:
        raise ValueError(f"Frame metadata too large ({len(meta_bytes)} bytes > 65535)")

    bitflags = 0
    if frame.flags.get("gap"):
        bitflags |= 0x01
    if frame.flags.get("late"):
        bitflags |= 0x02

    header = _HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        bitflags,
        0,
        int(frame.t0),
        float(frame.fs),
        int(frame.n_samples),
        int(frame.n_sensors),
        int(frame.seq),
        len(meta_bytes),
    )
    arr = np.ascontiguousarray(frame.samples, dtype="<f4")
    return header + meta_bytes + arr.tobytes()


def decode_frame_binary(
    payload: bytes,
    max_bytes: int = 4 * 1024 * 1024,
) -> Frame:
    """Decode an SHMF v1 binary payload into a validated Frame."""
    if len(payload) > max_bytes:
        raise ValueError(f"Payload exceeds maximum allowed size ({len(payload)} > {max_bytes})")
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"Truncated binary frame header ({len(payload)} < {HEADER_SIZE})")

    magic, ver, bitflags, _, t0_ns, fs, n_samples, n_sensors, seq, meta_len = _HEADER_STRUCT.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError(f"Invalid binary magic header: {magic!r}")
    if ver != VERSION:
        raise ValueError(f"Unsupported binary protocol version: {ver}")
    if n_samples == 0 or n_sensors == 0 or n_samples > 1_000_000 or n_sensors > 1024:
        raise ValueError(f"Invalid frame dimensions: n_samples={n_samples}, n_sensors={n_sensors}")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"Invalid sampling rate fs={fs}")

    meta_end = HEADER_SIZE + meta_len
    expected_data_bytes = n_samples * n_sensors * 4
    expected_total = meta_end + expected_data_bytes
    if len(payload) != expected_total:
        raise ValueError(f"Payload length mismatch: got {len(payload)} bytes, expected {expected_total}")

    meta = json.loads(payload[HEADER_SIZE:meta_end].decode("utf-8"))
    samples = np.frombuffer(payload, dtype="<f4", count=n_samples * n_sensors, offset=meta_end).reshape(n_samples, n_sensors).copy()

    flags = dict(meta.get("flags") or {})
    if bitflags & 0x01:
        flags["gap"] = True
    if bitflags & 0x02:
        flags["late"] = True

    return Frame(
        structure_id=str(meta["structure_id"]),
        sensor_ids=[str(s) for s in meta["sensor_ids"]],
        t0=int(t0_ns),
        fs=float(fs),
        samples=samples,
        seq=int(seq),
        source_id=str(meta.get("source_id", "binary_stream")),
        flags=flags,
    )


def encode_frame_json(frame: Frame) -> bytes:
    """Encode a Frame into compact JSON bytes (for debugging or HTTP transport)."""
    doc = {
        "structure_id": frame.structure_id,
        "sensor_ids": frame.sensor_ids,
        "t0": int(frame.t0),
        "fs": float(frame.fs),
        "seq": int(frame.seq),
        "source_id": frame.source_id,
        "flags": frame.flags,
        "samples": frame.samples.tolist(),
    }
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def decode_frame_json(
    raw: Union[str, bytes, Dict[str, Any]],
    default_structure_id: Optional[str] = None,
    default_sensor_ids: Optional[List[str]] = None,
    default_fs: float = 50.0,
    max_bytes: int = 4 * 1024 * 1024,
) -> Frame:
    """Decode canonical JSON frame (or legacy debugging dict) into a Frame."""
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > max_bytes:
            raise ValueError(f"JSON payload exceeds max_bytes ({len(raw)} > {max_bytes})")
        data = json.loads(raw.decode("utf-8"))
    elif isinstance(raw, str):
        if len(raw.encode("utf-8")) > max_bytes:
            raise ValueError("JSON string exceeds max_bytes")
        data = json.loads(raw)
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError(f"Unsupported JSON frame input type: {type(raw).__name__}")

    if not isinstance(data, dict):
        raise ValueError("JSON frame payload must be an object")

    structure_id = str(data.get("structure_id") or default_structure_id or "")
    if not structure_id:
        raise ValueError("Missing required field 'structure_id'")

    if "samples" in data:
        sensor_ids = data.get("sensor_ids") or default_sensor_ids
        if not sensor_ids or not isinstance(sensor_ids, list):
            raise ValueError("Missing required list 'sensor_ids'")
        samples_arr = np.asarray(data["samples"], dtype=np.float32)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)
        fs = float(data.get("fs", default_fs))
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Invalid sampling rate fs={fs}")
        t0 = int(data.get("t0") if data.get("t0") is not None else time.time_ns())
        seq = int(data.get("seq", 0))
        source_id = str(data.get("source_id", "json_stream"))
        flags = dict(data.get("flags") or {})
        return Frame(
            structure_id=structure_id,
            sensor_ids=[str(s) for s in sensor_ids],
            t0=t0,
            fs=fs,
            samples=samples_arr,
            seq=seq,
            source_id=source_id,
            flags=flags,
        )

    # Legacy fallback formats: {"values": [...]}, {"frame": [...]}, {"channels": {...}}, or flat {sensor_id: val}
    sensor_ids = data.get("sensor_ids") or default_sensor_ids or []
    vec: Optional[List[float]] = None
    if "values" in data and isinstance(data["values"], list):
        vec = [float(v) for v in data["values"]]
        if sensor_ids and len(vec) != len(sensor_ids):
            raise ValueError(f"Expected {len(sensor_ids)} values, got {len(vec)}")
    elif "frame" in data and isinstance(data["frame"], list):
        vec = [float(v) for v in data["frame"]]
        if sensor_ids and len(vec) != len(sensor_ids):
            raise ValueError(f"Expected {len(sensor_ids)} values in frame, got {len(vec)}")
    elif "channels" in data and isinstance(data["channels"], dict) and sensor_ids:
        vec = [float(data["channels"][sid]) for sid in sensor_ids if sid in data["channels"]]
        if len(vec) != len(sensor_ids):
            raise ValueError("Missing one or more channels in legacy 'channels' dict")
    elif sensor_ids and all(sid in data for sid in sensor_ids):
        vec = [float(data[sid]) for sid in sensor_ids]

    if vec is None:
        raise ValueError("Malformed frame JSON: expected 'samples', 'values', 'frame', or 'channels'")
    if not sensor_ids:
        sensor_ids = [f"ch_{i}" for i in range(len(vec))]

    return Frame(
        structure_id=structure_id,
        sensor_ids=[str(s) for s in sensor_ids],
        t0=int(data.get("t0") or time.time_ns()),
        fs=float(data.get("fs", default_fs)),
        samples=np.asarray(vec, dtype=np.float32).reshape(1, len(vec)),
        seq=int(data.get("seq", 0)),
        source_id=str(data.get("source_id", "legacy_json")),
        flags=dict(data.get("flags") or {}),
    )


def decode_frame_auto(
    raw: Union[bytes, bytearray, str, Dict[str, Any]],
    default_structure_id: Optional[str] = None,
    default_sensor_ids: Optional[List[str]] = None,
    default_fs: float = 50.0,
    max_bytes: int = 4 * 1024 * 1024,
) -> Frame:
    """Automatically detect binary SHMF v1 vs JSON and decode into Frame."""
    if isinstance(raw, (bytes, bytearray)) and len(raw) >= 4 and raw[:4] == MAGIC:
        return decode_frame_binary(bytes(raw), max_bytes=max_bytes)
    return decode_frame_json(
        raw,
        default_structure_id=default_structure_id,
        default_sensor_ids=default_sensor_ids,
        default_fs=default_fs,
        max_bytes=max_bytes,
    )
