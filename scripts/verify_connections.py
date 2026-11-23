import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient
from app.main import app

def main():
    print("Testing backend connectivity...")
    with TestClient(app) as client:
        # 1. Health
        r = client.get("/health")
        assert r.status_code == 200, f"Health failed: {r.status_code}"
        print(f"[OK] /health: 200 OK -> {r.json()['status']}, structures={r.json()['structures_count']}")

        # 2. Structures
        r = client.get("/api/v1/structures")
        assert r.status_code == 200, f"Structures failed: {r.status_code}"
        structs = r.json()
        struct_ids = [s["structure_id"] for s in structs]
        print(f"[OK] /api/v1/structures: 200 OK -> {len(structs)} structures: {struct_ids}")

        # 3. KW51 detail & detectors
        r = client.get("/api/v1/structures/kw51")
        assert r.status_code == 200, f"KW51 failed: {r.status_code}"
        print(f"[OK] /api/v1/structures/kw51: 200 OK -> '{r.json()['name']}' ({len(r.json()['sensors'])} sensors)")

        r = client.get("/api/v1/structures/kw51/detectors")
        assert r.status_code == 200, f"KW51 detectors failed: {r.status_code}"
        det_list = [d["detector_id"] for d in r.json()["detectors"]]
        print(f"[OK] /api/v1/structures/kw51/detectors: 200 OK -> {det_list}")

        # 4. Building Demo
        r = client.get("/api/v1/structures/building_demo")
        assert r.status_code == 200, f"Building Demo failed: {r.status_code}"
        print(f"[OK] /api/v1/structures/building_demo: 200 OK -> '{r.json()['name']}'")

        # 5. 3D Procedural Bridge Model
        r = client.get("/api/v1/structures/bridge_demo_01/model")
        assert r.status_code == 200, f"Model failed: {r.status_code}"
        print(f"[OK] /api/v1/structures/bridge_demo_01/model: 200 OK (GLB size={len(r.content)} bytes)")

        # 6. Alerts
        r = client.get("/api/v1/alerts")
        assert r.status_code == 200, f"Alerts failed: {r.status_code}"
        alerts_data = r.json()
        alerts_cnt = len(alerts_data.get("alerts", [])) if isinstance(alerts_data, dict) else len(alerts_data)
        print(f"[OK] /api/v1/alerts: 200 OK ({alerts_cnt} alerts)")

        # 7. WebSocket Telemetry on bridge_demo_01 (autonomous simulator)
        with client.websocket_connect("/ws/bridge_demo_01/telemetry") as ws:
            msg = ws.receive_json()
            assert "structure_id" in msg, "Missing structure_id in WS message"
            print(f"[OK] /ws/bridge_demo_01/telemetry: Connected & Received frame (structure={msg['structure_id']}, state={msg.get('system_status')})")

        # 8. HTTP Push ingestion -> WebSocket Broadcast on kw51
        with client.websocket_connect("/ws/kw51/telemetry") as ws:
            r = client.post("/api/v1/structures/kw51/telemetry", json={"values": [0.01, 0.02, -0.01, 0.005, 18.5]})
            assert r.status_code == 200, f"Push telemetry failed: {r.status_code}"
            msg = ws.receive_json()
            assert msg.get("structure_id") == "kw51", "Expected kw51 broadcast"
            print(f"[OK] Push ingestion -> /ws/kw51/telemetry: Broadcast received (structure={msg['structure_id']}, state={msg.get('system_status')})")

    print("\nALL BACKEND & WEBSOCKET CONNECTIONS ARE PROPERLY CONNECTED AND FUNCTIONING!")

if __name__ == "__main__":
    main()
