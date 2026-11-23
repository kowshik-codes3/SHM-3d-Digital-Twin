"""Phase 7 Integration Verification Client.

Spawns/connects to the Uvicorn FastAPI server, tests REST health and fault injection APIs,
subscribes to live WebSocket telemetry broadcasts, and verifies 100% contract compliance.
"""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import websockets
except ImportError:
    print("[ERROR] 'websockets' package is required. Install via pip install websockets.")
    sys.exit(1)


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/telemetry"

EXPECTED_NODES = [
    "Sensor_Abutment_Left",
    "Sensor_Pier_Left",
    "Sensor_MidSpan_Girder",
    "Sensor_Pier_Right",
    "Sensor_Ambient_Temp",
]


def check_health(timeout: float = 1.0) -> bool:
    """Check if FastAPI /health endpoint is responsive."""
    try:
        req = urllib.request.urlopen(f"{BASE_URL}/health", timeout=timeout)
        if req.status == 200:
            data = json.loads(req.read().decode("utf-8"))
            return data.get("status") == "online"
    except Exception:
        return False
    return False


def post_json(endpoint: str, payload: dict) -> dict:
    """Helper sending HTTP POST JSON request."""
    url = f"{BASE_URL}{endpoint}"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def run_phase7_verification_async():
    """Async WebSocket test runner."""
    print("=" * 60)
    print("STARTING PHASE 7 VERIFICATION FOR FASTAPI WEBSOCKET TELEMETRY")
    print("=" * 60)

    # 1. Connect to WebSocket Telemetry Stream
    print(f"\n--- [1] Connecting to WebSocket: {WS_URL} ---")
    async with websockets.connect(WS_URL) as ws:
        print("✅ WebSocket connection established successfully.")

        # 2. Receive Baseline Healthy Telemetry Frames
        print("\n--- [2] Listening to Healthy Baseline Stream (5 seconds) ---")
        healthy_frames = 0
        start_t = time.time()
        
        while time.time() - start_t < 4.0:
            msg_str = await asyncio.wait_for(ws.recv(), timeout=3.0)
            payload = json.loads(msg_str)

            # Assert JSON Schema Compliance
            assert "timestamp" in payload, "Missing 'timestamp' key"
            assert "system_status" in payload, "Missing 'system_status' key"
            assert "localized_defect_node" in payload, "Missing 'localized_defect_node' key"
            assert "nodes" in payload, "Missing 'nodes' dict"

            for node in EXPECTED_NODES:
                assert node in payload["nodes"], f"Missing node '{node}' in payload"
                node_data = payload["nodes"][node]
                assert "mse" in node_data, f"Missing 'mse' for {node}"
                assert "threshold" in node_data, f"Missing 'threshold' for {node}"
                assert "status" in node_data, f"Missing 'status' for {node}"

            healthy_frames += 1

        print(f" [PASS] Received {healthy_frames} baseline telemetry frames.")
        print(f"  Sample healthy payload status: '{payload['system_status']}'")

        # 3. Trigger Fault Injection via REST API
        print("\n--- [3] Injecting Fault: 'stiffness_loss' on Sensor_MidSpan_Girder ---")
        resp = post_json(
            "/api/v1/simulate-fault",
            {"node_id": "Sensor_MidSpan_Girder", "fault_type": "stiffness_loss", "severity": 0.6}
        )
        print(f"  API Response: {resp}")
        assert resp["status"] == "success", "Fault injection REST API failed"

        # 4. Listen for Anomaly Exceedance & Localization in WebSocket Stream
        print("\n--- [4] Verifying Real-Time Anomaly Alert & Spatial Localization ---")
        critical_detected = False
        localized_correctly = False

        start_t = time.time()
        while time.time() - start_t < 6.0:
            msg_str = await asyncio.wait_for(ws.recv(), timeout=3.0)
            payload = json.loads(msg_str)

            sys_status = payload["system_status"]
            defect_node = payload["localized_defect_node"]

            if sys_status == "CRITICAL":
                critical_detected = True
                if defect_node == "Sensor_MidSpan_Girder":
                    localized_correctly = True
                    print(f"🔥 CRITICAL ALERT DETECTED! Localized Node: '{defect_node}'")
                    print(f"   MidSpan Girder MSE: {payload['nodes']['Sensor_MidSpan_Girder']['mse']:.6f} "
                          f"(Threshold: {payload['nodes']['Sensor_MidSpan_Girder']['threshold']:.6f})")
                    break

        assert critical_detected, "Failed to receive CRITICAL status after fault injection"
        assert localized_correctly, f"Failed to localize to Sensor_MidSpan_Girder. Got '{defect_node}'"
        print(" [PASS] Real-time anomaly detection and 100% spatial localization verified over WebSocket.")

        # 5. Reset Injected Faults
        print("\n--- [5] Clearing Injected Faults ---")
        post_json("/api/v1/reset-faults", {})
        
        # Listen until stream returns to NORMAL
        start_t = time.time()
        returned_normal = False
        while time.time() - start_t < 4.0:
            msg_str = await asyncio.wait_for(ws.recv(), timeout=3.0)
            payload = json.loads(msg_str)
            if payload["system_status"] == "NORMAL":
                returned_normal = True
                break

        assert returned_normal, "Stream failed to return to NORMAL state after clearing faults"
        print(" [PASS] Telemetry stream returned to NORMAL baseline state.")

    print("=" * 60)
    print("PHASE 7 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def run_phase7_verification():
    """Main verification entry point with automatic server management."""
    server_process = None

    # Check if server is already running
    if not check_health():
        print("[INFO] Launching FastAPI server on http://127.0.0.1:8000...")
        backend_dir = Path(__file__).resolve().parent.parent.parent
        server_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
        ]
        server_process = subprocess.Popen(server_cmd, cwd=str(backend_dir))
        
        # Wait up to 10 seconds for server startup
        started = False
        for _ in range(20):
            time.sleep(0.5)
            if check_health():
                started = True
                break

        if not started:
            if server_process:
                server_process.kill()
            raise RuntimeError("FastAPI server failed to start within 10 seconds.")
        print("✅ FastAPI server started and /health checked.")
    else:
        print("[INFO] Connected to existing FastAPI server.")

    try:
        asyncio.run(run_phase7_verification_async())
    finally:
        if server_process:
            print("[INFO] Terminating temporary FastAPI server process...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    try:
        run_phase7_verification()
    except AssertionError as e:
        print(f"\n[FAIL] Assertion check failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
