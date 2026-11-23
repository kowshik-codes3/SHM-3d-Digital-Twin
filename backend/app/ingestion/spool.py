"""Append-Only Rotating Disk Spool & Restart Recovery for Stage 4 Ingestion.

NOTE: This lightweight spool provides crash/restart buffer recovery (last N minutes)
before full durable time-series storage is introduced in Stage 5.
On restart, the elapsed downtime gap between the last spooled frame and current time
is explicitly emitted as a GAP lifecycle event (never treated as structural data).
"""

from __future__ import annotations

import logging
from pathlib import Path
import struct
import time
from typing import List, Optional, Tuple

from app.ingestion.base import EventType, Frame, LifecycleEvent
from app.ingestion.protocol import decode_frame_binary, encode_frame_binary

logger = logging.getLogger("SHM.Spool")

_LEN_STRUCT = struct.Struct("<I")


class StructureSpool:
    """Two-file rotating binary append-only spool for one structure."""

    def __init__(
        self,
        structure_id: str,
        spool_dir: Path | str = "data/spool",
        max_bytes: int = 10 * 1024 * 1024,
        replay_minutes: float = 10.0,
    ) -> None:
        self.structure_id = structure_id
        self.base_dir = Path(spool_dir) / structure_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(65536, int(max_bytes))
        self.half_max = self.max_bytes // 2
        self.replay_minutes = float(replay_minutes)

        self.active_path = self.base_dir / "spool_active.bin"
        self.backup_path = self.base_dir / "spool_prev.bin"

    def append(self, frame: Frame) -> None:
        """Append binary frame record to active spool file, rotating if over size cap."""
        payload = encode_frame_binary(frame)
        record = _LEN_STRUCT.pack(len(payload)) + payload

        if self.active_path.exists() and (self.active_path.stat().st_size + len(record)) > self.half_max:
            if self.backup_path.exists():
                self.backup_path.unlink()
            self.active_path.rename(self.backup_path)

        with open(self.active_path, "ab") as f:
            f.write(record)

    def _read_file(self, path: Path) -> List[Frame]:
        if not path.is_file():
            return []
        frames: List[Frame] = []
        raw = path.read_bytes()
        offset = 0
        n = len(raw)
        while offset + 4 <= n:
            rec_len = _LEN_STRUCT.unpack_from(raw, offset)[0]
            offset += 4
            if rec_len == 0 or offset + rec_len > n:
                # Truncated tail record on ungraceful shutdown -> ignore tail safely
                break
            try:
                frame = decode_frame_binary(raw[offset : offset + rec_len], max_bytes=self.max_bytes)
                frames.append(frame)
            except Exception as exc:
                logger.warning(f"Skipping corrupted spool record at offset {offset}: {exc}")
                break
            offset += rec_len
        return frames

    def recover(self, now_ns: Optional[int] = None) -> Tuple[List[Frame], Optional[LifecycleEvent]]:
        """Load frames from the last `replay_minutes` and synthesize a downtime GAP event if needed."""
        curr_ns = int(now_ns if now_ns is not None else time.time_ns())
        all_frames = self._read_file(self.backup_path) + self._read_file(self.active_path)
        if not all_frames:
            return [], None

        cutoff_ns = curr_ns - int(self.replay_minutes * 60.0 * 1_000_000_000)
        recent = [f for f in all_frames if f.t_end >= cutoff_ns]
        if not recent:
            recent = all_frames[-1:]

        last_end_ns = max(f.t_end for f in all_frames)
        downtime_sec = max(0.0, (curr_ns - last_end_ns) / 1_000_000_000.0)
        gap_event: Optional[LifecycleEvent] = None
        if downtime_sec > 1.0:
            gap_event = LifecycleEvent(
                event_type=EventType.GAP,
                structure_id=self.structure_id,
                source_id="spool_recovery",
                timestamp_ns=curr_ns,
                reason=f"Downtime GAP of {downtime_sec:.2f}s across restart (spooled up to {last_end_ns})",
                details={
                    "downtime_seconds": downtime_sec,
                    "last_spooled_t_end_ns": last_end_ns,
                    "recovered_frames": len(recent),
                },
            )

        return recent, gap_event

    def clear(self) -> None:
        for p in (self.active_path, self.backup_path):
            if p.is_file():
                p.unlink()
