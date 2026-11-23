"""Automated End-to-End Integration Test Runner.

Launches the FastAPI backend server, waits for /health readiness,
executes the 4-phase E2E WebSocket & REST integration suite,
and cleanly shuts down the server.
"""

import asyncio
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("BACKEND_PORT", "8000"))
HEALTH_URL = f"http://{HOST}:{PORT}/health"
os.environ["WS_URL"] = f"ws://{HOST}:{PORT}/ws/telemetry"
os.environ["API_BASE"] = f"http://{HOST}:{PORT}/api/v1"


def wait_for_server(timeout_sec: float = 25.0) -> bool:
    start_t = time.time()
    while time.time() - start_t < timeout_sec:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    print(f"[E2E-Runner] Starting FastAPI uvicorn server on {HOST}:{PORT}...")
    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]

    proc = subprocess.Popen(
        server_cmd,
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        print("[E2E-Runner] Awaiting backend initialization...")
        is_ready = wait_for_server(timeout_sec=30.0)
        if not is_ready:
            print("[E2E-Runner] ERROR: FastAPI server failed to become ready within 30s.")
            if proc.poll() is not None:
                out, _ = proc.communicate(timeout=2)
                print(f"[E2E-Runner] Server output:\n{out}")
            return 1

        print("[E2E-Runner] FastAPI is online. Executing E2E scenario suite...")
        from app.ml.test_integration_e2e import run_e2e_suite
        asyncio.run(run_e2e_suite())
        print("[E2E-Runner] All E2E scenarios completed successfully.")
        return 0

    except Exception as exc:
        print(f"[E2E-Runner] E2E Suite execution failed: {exc}")
        return 1

    finally:
        print("[E2E-Runner] Terminating FastAPI server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
        print("[E2E-Runner] Server shutdown completed.")


if __name__ == "__main__":
    sys.exit(main())
