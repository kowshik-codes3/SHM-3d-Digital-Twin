"""Stage 4 Data-Quality, Fault Battery & Bad-Sensor Isolation Tests (J.2, J.3, J.4, J.5)."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.detectors.base import DetectorState
from app.detectors.registry import StructureRegistry
from app.detectors.spectral_tracking import SpectralTrackingDetector
from app.ingestion.base import Frame
from app.quality.models import QualityStatus
from app.quality.quality_layer import DataQualityLayer
from app.schemas.structure_config import StructureConfig

REPO_ROOT = BACKEND_DIR.parent
KW51_CONFIG_PATH = REPO_ROOT / "configs" / "kw51.yaml"
KW51_CACHE_DIR = REPO_ROOT / "data" / "kw51" / "cache_50hz"


def _load_kw51_slices(month_file: str, max_recs: int = 8) -> list[np.ndarray]:
    p = KW51_CACHE_DIR / month_file
    if not p.is_file():
        return []
    recs = []
    with h5py.File(p, "r") as hf:
        grp = hf["recordings"]
        for k in sorted(grp.keys())[:max_recs]:
            recs.append(grp[k][:])
    return recs


def _make_kw51_like_config() -> StructureConfig:
    if KW51_CONFIG_PATH.is_file():
        return StructureConfig.from_yaml(KW51_CONFIG_PATH)
    raise FileNotFoundError(KW51_CONFIG_PATH)


def _synthetic_healthy_block(n: int = 500, fs: float = 50.0, temp: float = 15.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    az1 = 0.002 * np.sin(2 * np.pi * 0.51 * t) + 0.001 * np.sin(2 * np.pi * 1.88 * t) + rng.normal(0, 4e-4, n)
    ay1 = 0.0015 * np.sin(2 * np.pi * 1.23 * t) + rng.normal(0, 4e-4, n)
    az2 = 0.0022 * np.sin(2 * np.pi * 0.51 * t) + 0.0011 * np.sin(2 * np.pi * 2.42 * t) + rng.normal(0, 4e-4, n)
    az3 = 0.0019 * np.sin(2 * np.pi * 0.51 * t) + 0.0009 * np.sin(2 * np.pi * 1.88 * t) + rng.normal(0, 4e-4, n)
    tvl = np.full(n, temp)
    return np.column_stack([az1, ay1, az2, az3, tvl]).astype(np.float32)


def test_j2_unplug_sensor_marks_sensor_fault_and_never_critical():
    """J.2: Unplugging 1 of 4 accelerometers (NaN/missing) excludes it and never triggers CRITICAL."""
    cfg = _make_kw51_like_config()
    reg = StructureRegistry()
    rt = reg.add(cfg, load_artifacts=False)

    # Fit spectral detector on multi-modal windows (n >= 4096 so 0.48-0.55 Hz resolves >=3 FFT bins)
    s1 = rt.detectors["spectral_tracking"]
    assert isinstance(s1, SpectralTrackingDetector)
    s1.window_sec = 4096.0 / 50.0
    s1.buffer_size = 4096
    s1.nperseg = 2048
    train_recs = [_synthetic_healthy_block(4096, temp=10.0 + i, seed=i) for i in range(6)]
    rt.quality_layer.calibrate(train_recs, fs=50.0)
    s1.fit(train_recs, val_recordings=train_recs[:4])

    # Unplug channel 0 (aBD11Az = NaN) -> az still has aBD17Az & aBD23Az, ay has aBD17Ay
    unplugged = _synthetic_healthy_block(4096, temp=12.0, seed=99)
    unplugged[:, 0] = np.nan

    f = Frame(
        structure_id="kw51",
        sensor_ids=cfg.channel_names,
        t0=1700000000_000_000_000,
        fs=50.0,
        samples=unplugged,
        seq=1,
        source_id="test_unplug",
    )
    payload = rt.push_ingestion_item(f)
    assert payload is not None
    assert payload["combined_state"] != DetectorState.CRITICAL.value
    # Unplugged channel marked OFFLINE / SENSOR_FAULT in telemetry & quality events
    ch0 = next(s for s in payload["sensors"] if s["id"] == "aBD11Az")
    assert ch0["status"] in ("OFFLINE", "SENSOR_FAULT")
    assert len(rt.quality_layer.get_events()) >= 1

    # Also verify: dropping below min_channels (only 1 healthy accel remaining) -> LOW_CONFIDENCE, NEVER CRITICAL
    severe = unplugged.copy()
    severe[:, 2] = np.nan
    severe[:, 3] = np.nan
    f2 = Frame(
        structure_id="kw51",
        sensor_ids=cfg.channel_names,
        t0=1700000100_000_000_000,
        fs=50.0,
        samples=severe,
        seq=2,
        source_id="test_unplug",
    )
    payload2 = rt.push_ingestion_item(f2)
    assert payload2 is not None
    assert payload2["combined_state"] in (DetectorState.LOW_CONFIDENCE.value, DetectorState.SENSOR_FAULT.value)
    assert payload2["combined_state"] != DetectorState.CRITICAL.value

    # Also verify: temperature sensor unplugged/faulted -> LOW_CONFIDENCE, never CRITICAL
    temp_bad = _synthetic_healthy_block(4096, temp=12.0, seed=101)
    temp_bad[:, 4] = 150.0  # physically out-of-range
    f3 = Frame(
        structure_id="kw51",
        sensor_ids=cfg.channel_names,
        t0=1700000200_000_000_000,
        fs=50.0,
        samples=temp_bad,
        seq=3,
        source_id="test_unplug",
    )
    payload3 = rt.push_ingestion_item(f3)
    assert payload3 is not None
    assert payload3["detectors"]["spectral_tracking"]["state"] == DetectorState.LOW_CONFIDENCE.value
    assert payload3["combined_state"] != DetectorState.CRITICAL.value


def test_j3_seven_fault_battery_and_kw51_clean_zero_false_faults():
    """J.3: Clean KW51 train/val produces 0% false faults; all 7 fault families detected 100%."""
    cfg = _make_kw51_like_config()
    ql = DataQualityLayer(cfg)

    train_recs = _load_kw51_slices("ambient_201810_50hz.h5", max_recs=8)
    val_recs = _load_kw51_slices("ambient_201901_50hz.h5", max_recs=8)
    if len(train_recs) >= 4 and len(val_recs) >= 4:
        ql.calibrate(train_recs, fs=50.0)
        clean_set = val_recs
    else:
        synth_train = [_synthetic_healthy_block(15000, temp=12.0, seed=i) for i in range(6)]
        ql.calibrate(synth_train, fs=50.0)
        clean_set = [_synthetic_healthy_block(15000, temp=13.0, seed=10 + i) for i in range(4)]

    # 1. Clean validation check: 0% false faults across full 5-min clean recordings
    false_faults = 0
    for idx, rec in enumerate(clean_set):
        fr = Frame(
            structure_id="kw51",
            sensor_ids=cfg.channel_names,
            t0=1546300800_000_000_000 + idx * 300 * 1_000_000_000,
            fs=50.0,
            samples=rec.astype(np.float32),
            seq=idx,
            source_id="clean_val",
            flags={"historical_replay": True},
        )
        rep = ql.evaluate_direct(fr)
        if rep.faulted_channels or rep.offline_channels:
            false_faults += 1
    assert false_faults == 0, f"Detected {false_faults} false faults on clean KW51 recordings"

    # 2. All 7 fault types on channel 0 (even the 3 F1.e hard ones: drift, gain, noise)
    base_clean = clean_set[0].copy()
    rng = np.random.default_rng(42)

    fault_vectors = {
        "Flatline": lambda x: np.zeros_like(x),
        "Stuck_Nonzero": lambda x: np.full_like(x, 0.42),
        "Missing_Packet_60pct": lambda x: np.where(np.arange(len(x)) % 5 < 3, np.nan, x),
        "Spike_Burst": lambda x: x + np.where(np.arange(len(x)) == len(x) // 2, 1.5, 0.0),
        "Calibration_Drift_0.2": lambda x: x + 0.2,
        "Calibration_Drift_1.0": lambda x: x + 1.0,
        "Gain_Error_3.0x": lambda x: x * 3.0,
        "Noise_Increase": lambda x: x + rng.normal(0.0, 0.05, size=x.shape),
        "Saturation_Clipping": lambda x: np.clip(x * 10000.0, -0.02, 0.02),
    }

    ql_fault_hits = {}
    for idx, (fname, fn) in enumerate(fault_vectors.items()):
        corrupted = base_clean.copy()
        corrupted[:, 0] = fn(corrupted[:, 0])
        fr = Frame(
            structure_id="kw51",
            sensor_ids=cfg.channel_names,
            t0=1700000000_000_000_000 + idx * 300 * 1_000_000_000,
            fs=50.0,
            samples=corrupted.astype(np.float32),
            seq=idx,
            source_id=f"fault_{fname}",
            flags={"historical_replay": True},
        )
        ql_fresh = DataQualityLayer(cfg)
        ql_fresh.baselines = ql.baselines
        rep = ql_fresh.evaluate_direct(fr)
        detected = ("aBD11Az" in rep.faulted_channels) or ("aBD11Az" in rep.offline_channels)
        ql_fault_hits[fname] = detected
        assert detected, f"DataQualityLayer missed fault {fname}: report={rep.sensors['aBD11Az']}"

    assert all(ql_fault_hits.values())


def test_j4_reorder_duplicates_and_late_frames():
    """J.4: Out-of-order seqs reordered cleanly; duplicate seq+t0 dropped; ultra-late dropped."""
    cfg = _make_kw51_like_config()
    cfg.quality.reorder_window_frames = 4
    cfg.quality.late_arrival_tolerance_sec = 2.0
    ql = DataQualityLayer(cfg)

    base_t0 = 1700000000_000_000_000
    samples = _synthetic_healthy_block(50, temp=12.0, seed=1)

    def _mk(seq: int, offset_s: float) -> Frame:
        return Frame(
            structure_id="kw51",
            sensor_ids=cfg.channel_names,
            t0=base_t0 + int(offset_s * 1e9),
            fs=50.0,
            samples=samples,
            seq=seq,
            source_id="src_ooo",
            flags={"historical_replay": True},
        )

    # Send seq 0 -> flushes immediately
    out0, _ = ql.evaluate_frame(_mk(0, 0.0))
    assert [f.seq for f, _ in out0] == [0]

    # Duplicate (seq=0, t0=0) -> dropped immediately
    out_dup, _ = ql.evaluate_frame(_mk(0, 0.0))
    assert out_dup == []
    assert ql.counters["duplicates_dropped"] == 1

    # Send seq 2 before seq 1 -> buffered until seq 1 arrives, then both flush in order [1, 2]
    out2, _ = ql.evaluate_frame(_mk(2, 2.0))
    assert out2 == []
    out1, _ = ql.evaluate_frame(_mk(1, 1.0))
    assert [f.seq for f, _ in out1] == [1, 2]
    assert ql.counters["out_of_order_frames"] >= 1

    # Advance expected timeline to t=+10s, then send ultra-late frame from t=0s (>2s tolerance) -> dropped
    ql.evaluate_frame(_mk(10, 10.0))
    out_late, _ = ql.evaluate_frame(_mk(3, 0.5))
    assert out_late == []
    assert ql.counters["late_frames_dropped"] == 1


def test_j5_clock_drift_jump_and_sync_skew():
    """J.5: Clock drift, sudden clock jump, and multi-sensor sync skew penalize timing without CRITICAL."""
    cfg = _make_kw51_like_config()
    cfg.quality.clock_jump_max_sec = 0.5
    cfg.quality.clock_drift_max_ppm = 200.0
    cfg.quality.clock_sync_skew_max_sec = 0.05
    ql = DataQualityLayer(cfg)

    samples = _synthetic_healthy_block(50, temp=12.0, seed=5)
    t0_base = 1700000000_000_000_000
    wall_base = 1700000001_000_000_000

    # Baseline live frame 0
    f0 = Frame(
        structure_id="kw51",
        sensor_ids=cfg.channel_names,
        t0=t0_base,
        fs=50.0,
        samples=samples,
        seq=0,
        source_id="gw_clk",
        flags={"received_wall_ns": wall_base},
    )
    rep0 = ql.evaluate_direct(f0, wall_now=wall_base / 1e9)
    assert rep0.clock_stats["jump_sec"] == 0.0

    # Sudden +3.0 s forward jump in frame timestamps without seq gap -> clock jump flagged
    f_jump = Frame(
        structure_id="kw51",
        sensor_ids=cfg.channel_names,
        t0=t0_base + 4_000_000_000,  # expected +1.0s, actual +4.0s -> 3.0s jump
        fs=50.0,
        samples=samples,
        seq=1,
        source_id="gw_clk",
        flags={
            "received_wall_ns": wall_base + 1_000_000_000,
            "channel_t0_ns": {"aBD11Az": t0_base, "aBD17Ay": t0_base + 200_000_000},  # 0.2s sync skew
        },
    )
    rep_jump = ql.evaluate_direct(f_jump, wall_now=(wall_base + 1_000_000_000) / 1e9)
    assert rep_jump.clock_stats["jump_sec"] >= 2.5
    assert rep_jump.clock_stats["sync_skew_sec"] == pytest.approx(0.2, abs=1e-4)
    assert ql.counters["clock_jumps"] >= 1
    assert rep_jump.overall_status == QualityStatus.DEGRADED or rep_jump.overall_status == "DEGRADED"

    # Separate stream: steady +1500 ppm clock drift over 10 consecutive 1s frames without step jump
    ql2 = DataQualityLayer(cfg)
    for k in range(11):
        fk = Frame(
            structure_id="kw51",
            sensor_ids=cfg.channel_names,
            t0=t0_base + int(round(k * 1.0015 * 1e9)),
            fs=50.0,
            samples=samples,
            seq=k,
            source_id="gw_drift",
            flags={"received_wall_ns": wall_base + k * 1_000_000_000},
        )
        rep_d = ql2.evaluate_direct(fk, wall_now=(wall_base / 1e9) + k)
    assert ql2.counters["clock_drift_warnings"] >= 1
    assert rep_d.clock_stats["drift_ppm"] > 200.0
