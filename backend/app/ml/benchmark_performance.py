"""
benchmark_performance.py

Micro-second performance benchmarking and memory utilization profiling module for SHM Digital Twin.
Measures:
- Ingestion & Circular Buffer Overhead
- PyTorch LSTM-Autoencoder Inference Latency (1,000 continuous iterations)
- Anomaly State Machine Evaluation
- End-to-End Latency & RAM Utilization Profiling
Generates report at reports/phase9/latency_benchmark.json.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

# Ensure app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.ml.anomaly_engine import AnomalyStateMachine
from app.ml.model import LSTMAutoencoder
from app.ml.scaler_utils import StructuralDataScaler
from app.ml.stream_buffer import SensorStreamBuffer
from app.ml.threshold_calibrator import ThresholdCalibrator

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SHM.Benchmark: %(message)s")
logger = logging.getLogger("SHM.Benchmark")

def resolve_artifacts_dir() -> Path:
    candidates = [
        Path(os.environ.get("ARTIFACTS_DIR", "")),
        Path(__file__).resolve().parent.parent.parent / "artifacts",
        Path(__file__).resolve().parent.parent.parent.parent / "artifacts",
    ]
    for c in candidates:
        if c and c.exists() and (c / "threshold_config.json").exists():
            return c
    return Path(__file__).resolve().parent.parent.parent / "artifacts"


def resolve_report_dir() -> Path:
    report_env = os.environ.get("REPORTS_DIR")
    if report_env:
        p = Path(report_env)
    else:
        p = Path(__file__).resolve().parent.parent.parent / "reports" / "phase9"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


ARTIFACTS_DIR = resolve_artifacts_dir()
REPORT_DIR = resolve_report_dir()


def run_benchmark(num_iterations: int = 1000):
    logger.info("=" * 60)
    logger.info(f"STARTING PHASE 9 PERFORMANCE & LATENCY BENCHMARK ({num_iterations} Iterations)")
    logger.info("=" * 60)

    # 1. Load artifacts
    scaler = StructuralDataScaler.load(str(ARTIFACTS_DIR / "scaler.joblib"))
    thresh_config = ThresholdCalibrator.load(str(ARTIFACTS_DIR / "threshold_config.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder().to(device)
    model.load_state_dict(torch.load(str(ARTIFACTS_DIR / "lstm_autoencoder_best.pt"), map_location=device))
    model.eval()

    state_machine = AnomalyStateMachine(config=thresh_config, persistence=3)
    buffer = SensorStreamBuffer(window_size=50, num_channels=5)

    # Fill buffer initial 50 timesteps
    for _ in range(50):
        buffer.push_frame(np.random.randn(5))

    # Benchmarking lists (milliseconds)
    buf_latencies = []
    model_latencies = []
    eval_latencies = []
    e2e_latencies = []

    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    logger.info(f"Benchmarking streaming inference pipeline over {num_iterations} frames...")

    with torch.no_grad():
        for i in range(num_iterations):
            t_start = time.perf_counter()

            # Step 1: Buffer Ingestion
            t0 = time.perf_counter()
            buffer.push_frame(np.random.randn(5))
            tensor = buffer.get_standardized_tensor(scaler, device)
            t_buf = (time.perf_counter() - t0) * 1000.0

            # Step 2: PyTorch Model Inference
            t0 = time.perf_counter()
            reconstructed, _ = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model = (time.perf_counter() - t0) * 1000.0

            # Step 3: Reconstruction Error & Thresholding
            t0 = time.perf_counter()
            mse_per_channel = torch.mean((tensor - reconstructed) ** 2, dim=1).squeeze(0).cpu().numpy()
            event = state_machine.step(i, time.time(), mse_per_channel)
            t_eval = (time.perf_counter() - t0) * 1000.0

            t_e2e = (time.perf_counter() - t_start) * 1000.0

            buf_latencies.append(t_buf)
            model_latencies.append(t_model)
            eval_latencies.append(t_eval)
            e2e_latencies.append(t_e2e)

    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    ram_delta_mb = ram_end_mb - ram_start_mb

    results = {
        "hardware_device": device.type,
        "iterations": num_iterations,
        "latencies_ms": {
            "ingestion_buffer": {
                "mean": float(np.mean(buf_latencies)),
                "p95": float(np.percentile(buf_latencies, 95)),
                "p99": float(np.percentile(buf_latencies, 99)),
                "target": "< 2.0 ms"
            },
            "pytorch_inference": {
                "mean": float(np.mean(model_latencies)),
                "p95": float(np.percentile(model_latencies, 95)),
                "p99": float(np.percentile(model_latencies, 99)),
                "target": "< 15.0 ms"
            },
            "threshold_state_machine": {
                "mean": float(np.mean(eval_latencies)),
                "p95": float(np.percentile(eval_latencies, 95)),
                "p99": float(np.percentile(eval_latencies, 99)),
                "target": "< 1.0 ms"
            },
            "end_to_end_pipeline": {
                "mean": float(np.mean(e2e_latencies)),
                "p95": float(np.percentile(e2e_latencies, 95)),
                "p99": float(np.percentile(e2e_latencies, 99)),
                "target": "< 250.0 ms"
            }
        },
        "memory_mb": {
            "initial_rss": float(ram_start_mb),
            "final_rss": float(ram_end_mb),
            "delta": float(ram_delta_mb)
        }
    }

    report_path = REPORT_DIR / "latency_benchmark.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  Report exported to:    {report_path}")
    except Exception as exc:
        logger.warning(f"  Note: Report file could not be written ({exc}); logging metrics to console.")

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS SUMMARY:")
    logger.info(f"  Ingestion Buffer Mean: {results['latencies_ms']['ingestion_buffer']['mean']:.4f} ms (p95: {results['latencies_ms']['ingestion_buffer']['p95']:.4f} ms)")
    logger.info(f"  PyTorch Model Mean:    {results['latencies_ms']['pytorch_inference']['mean']:.4f} ms (p95: {results['latencies_ms']['pytorch_inference']['p95']:.4f} ms)")
    logger.info(f"  State Machine Mean:    {results['latencies_ms']['threshold_state_machine']['mean']:.4f} ms (p95: {results['latencies_ms']['threshold_state_machine']['p95']:.4f} ms)")
    logger.info(f"  End-to-End Mean:       {results['latencies_ms']['end_to_end_pipeline']['mean']:.4f} ms (p95: {results['latencies_ms']['end_to_end_pipeline']['p95']:.4f} ms)")
    logger.info(f"  RAM Delta:             {ram_delta_mb:+.2f} MB")
    logger.info(f"  Report exported to:    {report_path}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_benchmark(1000)
