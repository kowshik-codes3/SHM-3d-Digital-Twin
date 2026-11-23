"""Automated runner for SHM Performance & Latency Benchmark."""

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.ml.benchmark_performance import run_benchmark


def main():
    iterations = int(os.environ.get("BENCHMARK_ITERATIONS", "1000"))
    results = run_benchmark(num_iterations=iterations)
    e2e_p95 = results["latencies_ms"]["end_to_end_pipeline"]["p95"]
    assert e2e_p95 < 250.0, f"Benchmark failed: P95 latency {e2e_p95:.2f}ms exceeds 250ms target"
    print(f"\n[OK] Latency target satisfied: P95 = {e2e_p95:.2f} ms (< 250 ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
