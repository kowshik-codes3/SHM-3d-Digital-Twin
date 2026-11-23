"""Stage 4 End-to-End Ingestion, Replay Parity, Backpressure, Boundary Reset, Spool & HTTP API Tests (J.1, J.8, J.9)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import h5py
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.detectors.base import DetectorState
from app.detectors.registry import StructureRegistry
from app.detectors.spectral_tracking import SpectralTrackingDetector
from app.ingestion.base import EventType, Frame, LifecycleEvent
from app.ingestion.protocol import encode_frame_binary
from app.ingestion.replay_source import ReplaySource
from app.ingestion.spool import StructureSpool
from app.main import app, app_state
from app.schemas.structure_config import StructureConfig

REPO_ROOT = BACKEND_DIR.parent
KW51_CONFIG_PATH = REPO_ROOT / "configs" / "kw51.yaml"
KW51_CACHE_H5 = REPO_ROOT / "data" / "kw51" / "cache_50hz" / "ambient_201810_50hz.h5"


def _make_kw51_cfg() -> StructureConfig:
    return StructureConfig.from_yaml(KW51_CONFIG_PATH)


@pytest.mark.asyncio
async def test_j1_kw51_replay_matches_offline_spectral_scores():
    """J.1: Streaming 1 KW51 recording through ReplaySource + StructureRuntime matches direct offline extraction."""
    if not KW51_CACHE_H5.is_file():
        pytest.skip("KW51 cache_50hz HDF5 not on disk")

    cfg = _make_kw51_cfg()
    reg = StructureRegistry()
    rt = reg.add(cfg, load_artifacts=False)

    # Load train recordings and 1 clean OK test recording from ambient_201810_50hz.h5
    recs = []
    rec_keys = []
    with h5py.File(KW51_CACHE_H5, "r") as hf:
        grp = hf["recordings"]
        for k in sorted(grp.keys())[:15]:
            recs.append(grp[k][:])
            rec_keys.append(k)

    s1 = rt.detectors["spectral_tracking"]
    assert isinstance(s1, SpectralTrackingDetector)
    rt.quality_layer.calibrate(recs[:5], fs=50.0)
    s1.fit(recs[:5], val_recordings=recs[:5])

    test_idx = next(
        i for i in range(5, len(recs)) if s1.extract_frequencies(recs[i])[2] == "OK"
    )
    test_rec = recs[test_idx]
    rec_key = rec_keys[test_idx]
    expected_freqs, expected_temp, expected_status, _ = s1.extract_frequencies(test_rec)
    assert expected_status == "OK" and expected_freqs is not None
    expected_score, _ = s1._compute_mahalanobis(expected_freqs, expected_temp)

    # Now stream that single 300s recording via ReplaySource -> StructureRuntime (disable boundary clear on END only so we inspect output)
    rt.config.ingestion.reset_on_recording_boundary = False

    # Create temporary 1-recording HDF5 so ReplaySource streams exact recording
    tmp_h5 = Path(__file__).parent / "_tmp_j1_rec.h5"
    channel_ids = [s.id for s in cfg.sensors]
    try:
        with h5py.File(tmp_h5, "w") as out_f:
            grp = out_f.create_group("recordings")
            grp.attrs["channels"] = channel_ids
            grp.create_dataset(rec_key, data=test_rec)

        src = ReplaySource(
            file_path=str(tmp_h5),
            structure_id="kw51",
            sensor_ids=channel_ids,
            fs=50.0,
            chunk_samples=50,
            speed_multiplier="max",
            loop=False,
        )
        await rt.run_ingestion_source(src, use_queue=False)
        out = rt.latest_outputs["spectral_tracking"]
        assert out.state != DetectorState.SENSOR_FAULT
        assert abs(out.score - round(expected_score, 4)) <= 1e-2, f"Live score {out.score} vs offline {expected_score}"
        np.testing.assert_allclose(s1.last_modal_frequencies, expected_freqs, atol=1e-3)
    finally:
        tmp_h5.unlink(missing_ok=True)


def test_j8_backpressure_bounded_queue_drops_oldest_gracefully():
    """J.8: Enqueueing beyond max_queue_size drops oldest frames, records GAP, sets DEGRADED, never crashes."""
    cfg = _make_kw51_cfg()
    cfg.ingestion.max_queue_size = 8
    reg = StructureRegistry()
    rt = reg.add(cfg, load_artifacts=False)

    samples = np.ones((10, cfg.num_channels), dtype=np.float32) * 0.01
    for i in range(25):
        fr = Frame(
            structure_id="kw51",
            sensor_ids=cfg.channel_names,
            t0=1700000000_000_000_000 + i * 200_000_000,
            fs=50.0,
            samples=samples,
            seq=i,
            source_id="burst_test",
        )
        rt.enqueue_item(fr)

    health = rt.get_ingestion_health()
    assert health["queue_depth"] <= 8
    assert health["dropped_frames_total"] == 25 - 8
    assert health["source_status"] == "DEGRADED"


def test_j9_scheduled_recording_boundary_resets_buffers_and_persistence():
    """J.9: RECORDING_END/START clears resampler, detector ring buffers, and persistence; tracks recordings/hours."""
    cfg = _make_kw51_cfg()
    cfg.ingestion.reset_on_recording_boundary = True
    reg = StructureRegistry()
    rt = reg.add(cfg, load_artifacts=False)
    s1 = rt.detectors["spectral_tracking"]
    assert isinstance(s1, SpectralTrackingDetector)
    wf = rt.detectors["waveform_ae"]

    # Simulate partial state accumulated in Recording 1
    s1.consecutive_alerts = 2
    s1.ewma_score = 42.0
    rt.push_ingestion_item(
        LifecycleEvent(
            event_type=EventType.RECORDING_START,
            structure_id="kw51",
            source_id="replay",
            details={"recording_id": "rec_01"},
        )
    )
    chunk = np.random.default_rng(11).normal(0, 0.005, size=(100, cfg.num_channels)).astype(np.float32)
    chunk[:, 4] = 12.0
    rt.push_ingestion_item(
        Frame(
            structure_id="kw51",
            sensor_ids=cfg.channel_names,
            t0=1546300800_000_000_000,
            fs=50.0,
            samples=chunk,
            seq=0,
            source_id="replay",
            flags={"recording_id": "rec_01", "historical_replay": True},
        )
    )
    assert len(s1.ring_buffer) == 100
    assert len(wf.buffer) > 0

    # End Recording 1 -> state and persistence MUST reset cleanly
    rt.push_ingestion_item(
        LifecycleEvent(
            event_type=EventType.RECORDING_END,
            structure_id="kw51",
            source_id="replay",
            details={"recording_id": "rec_01"},
        )
    )
    assert len(s1.ring_buffer) == 0
    assert len(wf.buffer) == 0
    assert s1.consecutive_alerts == 0
    assert s1.ewma_score is None
    assert rt.recordings_completed == 1
    assert rt.hours_observed > 0.0
    health = rt.get_ingestion_health()
    assert health["confirmation_delay"]["recordings"] >= 1
    assert health["confirmation_delay"]["hours"] > 0.0


def test_spool_write_rotation_and_recovery(tmp_path):
    """Append-only rotating spool recovers un-drained binary frames with explicit GAP notification."""
    spool = StructureSpool("kw51", spool_dir=tmp_path, max_bytes=65536)
    samples = np.arange(50, dtype=np.float32).reshape(10, 5) * 0.001
    t_old = time.time_ns() - 10_000_000_000
    for i in range(6):
        fr = Frame(
            structure_id="kw51",
            sensor_ids=["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az", "tVL"],
            t0=t_old + i * 200_000_000,
            fs=50.0,
            samples=samples,
            seq=i,
            source_id="spool_test",
        )
        spool.append(fr)

    frames, gap_ev = spool.recover()
    assert len(frames) == 6
    assert gap_ev is not None and gap_ev.event_type == EventType.GAP
    spool.clear()
    assert spool.recover() == ([], None)


def test_http_frames_auth_idempotency_and_quality_endpoints(monkeypatch):
    """POST /frames enforces X-API-Key, Idempotency-Key dedup, binary/JSON, plus /ingestion & /quality/events."""
    monkeypatch.setenv("SHM_API_KEY_KW51", "stage4-secret-test-key")
    with TestClient(app) as client:
        api_key = settings.get_structure_api_key("kw51")
        frame = Frame(
            structure_id="kw51",
            sensor_ids=["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az", "tVL"],
            t0=int(time.time_ns()),
            fs=50.0,
            samples=np.full((50, 5), 0.001, dtype=np.float32) + np.column_stack([
                0.002 * np.sin(np.linspace(0, 6, 50)),
                0.002 * np.cos(np.linspace(0, 6, 50)),
                0.002 * np.sin(np.linspace(0, 12, 50)),
                0.002 * np.cos(np.linspace(0, 12, 50)),
                np.full(50, 14.0),
            ]).astype(np.float32),
            seq=1,
            source_id="http_test_gw",
        )
        raw_bin = encode_frame_binary(frame)

        # 1. Missing/bad API key -> 401
        r_unauth = client.post("/api/v1/structures/kw51/frames", content=raw_bin, headers={"X-API-Key": "wrong"})
        assert r_unauth.status_code == 401

        # 2. Valid binary frame with Idempotency-Key -> accepted
        hdrs = {
            "X-API-Key": api_key,
            "Idempotency-Key": "gw-batch-1",
            "Content-Type": "application/octet-stream",
        }
        r_ok = client.post("/api/v1/structures/kw51/frames", content=raw_bin, headers=hdrs)
        assert r_ok.status_code == 200
        assert r_ok.json()["frames_accepted"] == 1

        # 3. Retry with same Idempotency-Key -> deduplicated
        r_dup = client.post("/api/v1/structures/kw51/frames", content=raw_bin, headers=hdrs)
        assert r_dup.status_code == 200
        assert r_dup.json()["duplicates_ignored"] == 1
        assert r_dup.json()["frames_accepted"] == 0

        # 4. Ingestion health & quality events endpoints
        r_ing = client.get("/api/v1/structures/kw51/ingestion")
        assert r_ing.status_code == 200
        assert r_ing.json()["frames_processed"] >= 1

        r_qe = client.get("/api/v1/structures/kw51/quality/events")
        assert r_qe.status_code == 200
        assert "events" in r_qe.json()
