"""Stage 4 Resampling, Protocol & Guardrail Tests (J.6)."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import scipy.io as sio

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.ingestion.base import EventType, Frame, LifecycleEvent
from app.ingestion.protocol import decode_frame_binary, encode_frame_binary
from app.ingestion.resampler import EdgeResampler
from app.ingestion.simulator_source import SimulatorSource

REPO_ROOT = BACKEND_DIR.parent
RAW_MAT_PATH = REPO_ROOT / "data" / "kw51" / "ambient_201906" / "ambient_20190601_000000.mat"
CACHE_H5_PATH = REPO_ROOT / "data" / "kw51" / "cache_50hz" / "ambient_201906_50hz.h5"


def test_kw51_raw_mat_vs_cached_50hz_parity_j6():
    """J.6: EdgeResampler reproduces cached 50 Hz HDF5 samples from raw 825.8 Hz .mat."""
    if not RAW_MAT_PATH.is_file() or not CACHE_H5_PATH.is_file():
        pytest.skip("KW51 raw .mat or cache_50hz archive not present on disk")

    mat = sio.loadmat(RAW_MAT_PATH)
    pa = mat["predat_a"]
    pe = mat["predat_env"]
    raw_accel_lbl_arr = pa["labels"][0, 0]
    raw_accel_lbls = [str(raw_accel_lbl_arr[i, 0][0]) for i in range(raw_accel_lbl_arr.shape[0])]
    raw_env_lbls = [str(pe["labels"][0, 0][i, 0][0]) for i in range(pe["labels"][0, 0].shape[0])]

    accel_ch = ["aBD11Az", "aBD17Ay", "aBD17Az", "aBD23Az"]
    accel_idx = [raw_accel_lbls.index(ch) for ch in accel_ch]
    raw_vib = pa["tdata"][0, 0][:, accel_idx]
    temp_val = float(np.nanmean(pe["data"][0, 0][:, raw_env_lbls.index("tVL")]))
    raw = np.column_stack([raw_vib, np.full(raw_vib.shape[0], temp_val)])
    sids = accel_ch + ["tVL"]

    native_fs = 25600.0 / 31.0  # 825.8064516129032 Hz
    resampler = EdgeResampler(
        native_fs=native_fs,
        target_fs=50.0,
        filter_type="butter",
        order=8,
        cutoff_hz=20.0,
        mode="block_zero_phase",
        temp_channel_indices=[4],
    )
    frame_in = Frame(
        structure_id="kw51",
        sensor_ids=sids,
        t0=1559347200_000_000_000,
        fs=native_fs,
        samples=raw.astype(np.float32),
        seq=0,
        source_id="raw_mat",
    )
    out_items = resampler.process_frame(frame_in)
    out_frames = [x for x in out_items if isinstance(x, Frame)]
    assert len(out_frames) == 1
    frame_out = out_frames[0]
    assert abs(frame_out.fs - 50.0) < 1e-6

    with h5py.File(CACHE_H5_PATH, "r") as hf:
        cached = hf["recordings/ambient_20190601_000000"][:]

    n_cmp = min(len(cached), frame_out.samples.shape[0])
    assert n_cmp >= 14900
    max_abs_diff = float(np.max(np.abs(frame_out.samples[:n_cmp] - cached[:n_cmp])))
    assert max_abs_diff < 1e-4, f"KW51 resampling parity drift too high: max_abs_diff={max_abs_diff}"


def test_chebyshev_anti_alias_filter_and_streaming_mode():
    """Chebyshev Type I causal streaming resampler attenuates high-freq aliases."""
    native_fs = 200.0
    target_fs = 50.0
    resampler = EdgeResampler(
        native_fs=native_fs,
        target_fs=target_fs,
        filter_type="cheby1",
        order=8,
        cutoff_hz=20.0,
        ripple_db=0.5,
        mode="streaming_causal",
    )

    t = np.arange(400) / native_fs
    # Ch 0: 2 Hz in-band wave; Ch 1: 45 Hz alias tone (above 25 Hz Nyquist -> must be heavily attenuated)
    sig = np.column_stack([np.sin(2 * np.pi * 2.0 * t), np.sin(2 * np.pi * 45.0 * t)]).astype(np.float32)

    out_chunks = []
    for start in range(0, len(sig), 100):
        f = Frame(
            structure_id="test",
            sensor_ids=["c0", "c1"],
            t0=int(start * 1e9 / native_fs),
            fs=native_fs,
            samples=sig[start : start + 100],
            seq=start // 100,
            source_id="test",
        )
        for item in resampler.process_frame(f):
            if isinstance(item, Frame):
                out_chunks.append(item.samples)

    out = np.vstack(out_chunks)
    assert out.shape[0] == 100
    assert float(np.std(out[10:, 0])) > 0.5  # 2 Hz passed
    assert float(np.std(out[10:, 1])) < 0.05  # 45 Hz attenuated


def test_irregular_resampling_never_interpolates_long_gap():
    """J.6: Jittery timestamps interpolate micro-gaps but split at gaps > max_gap_interpolate_sec with GAP."""
    resampler = EdgeResampler(native_fs=50.0, target_fs=50.0, max_gap_interpolate_sec=0.25)

    t1 = np.arange(0.0, 1.0, 0.02) + np.random.default_rng(7).uniform(-0.002, 0.002, size=50)
    t2 = np.arange(2.0, 3.0, 0.02)  # 1.0 s gap between 1.0 and 2.0 (> 0.25 s)
    ts_sec = np.concatenate([t1, t2])
    ts_ns = (ts_sec * 1e9).astype(np.int64)
    vals = np.sin(2 * np.pi * 1.5 * ts_sec).reshape(-1, 1).astype(np.float32)

    items = resampler.resample_irregular(
        timestamps_ns=ts_ns,
        samples=vals,
        structure_id="test",
        sensor_ids=["c0"],
        source_id="jitter",
    )
    frames = [x for x in items if isinstance(x, Frame)]
    gaps = [x for x in items if isinstance(x, LifecycleEvent) and x.event_type == EventType.GAP]
    assert len(gaps) == 1
    assert len(frames) == 2
    assert frames[0].t_end <= frames[1].t0 - int(0.8 * 1e9)


@pytest.mark.asyncio
async def test_simulator_source_disabled_when_demo_mode_false(monkeypatch):
    """SimulatorSource must refuse streaming when DEMO_MODE=false."""
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    src = SimulatorSource(fs=50.0)
    with pytest.raises(RuntimeError, match="DEMO_MODE=false"):
        async for _ in src.stream():
            pass


def test_binary_shmf_wire_roundtrip():
    """Binary SHMF v1 wire protocol round-trips Frame fields accurately."""
    samples = np.arange(15, dtype=np.float32).reshape(3, 5) * 0.01
    frame = Frame(
        structure_id="kw51",
        sensor_ids=["a1", "a2", "a3", "a4", "t1"],
        t0=1700000000_123_456_789,
        fs=50.0,
        samples=samples,
        seq=99,
        source_id="pi-gw-01",
        flags={"recording_id": "rec_01"},
    )
    raw = encode_frame_binary(frame)
    decoded = decode_frame_binary(raw)
    assert decoded.structure_id == "kw51"
    assert decoded.sensor_ids == ["a1", "a2", "a3", "a4", "t1"]
    assert decoded.t0 == frame.t0
    assert decoded.fs == 50.0
    assert decoded.seq == 99
    assert decoded.flags["recording_id"] == "rec_01"
    np.testing.assert_allclose(decoded.samples, samples, atol=1e-7)
