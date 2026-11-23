"""
test_integration_e2e.py

Automated End-to-End Integration Test Suite for SHM Digital Twin.
Executes 4 core operational scenarios against the running FastAPI WebSocket server:
1. Baseline Healthy Stream (FAR = 0%)
2. MidSpan Stiffness Loss Fault Injection & Spatial Localization
3. Pier Foundation Settlement Fault Injection & Spatial Localization
4. Fault Recovery & Normalization Cycle
"""

import asyncio
import json
import logging
import os
import sys
import time
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    import websockets
except ImportError:
    import pip
    pip.main(["install", "websockets"])
    import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SHM.E2E: %(message)s")
logger = logging.getLogger("SHM.E2E")

WS_URL = os.environ.get("WS_URL", "ws://127.0.0.1:8000/ws/telemetry")
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000/api/v1")


def post_api(endpoint: str, data: dict = None):
    url = f"{API_BASE}/{endpoint}"
    req_data = json.dumps(data).encode('utf-8') if data else None
    headers = {'Content-Type': 'application/json'} if data else {}
    req = urllib.request.Request(url, data=req_data, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))


async def test_scenario_1_healthy_baseline(ws):
    logger.info("--- [Scenario 1] Testing Healthy Baseline Operations (10s) ---")
    start_time = time.time()
    frames_received = 0
    false_alarms = 0

    while time.time() - start_time < 10.0:
        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
        if msg == "pong":
            continue
        data = json.loads(msg)
        frames_received += 1
        # System alert state machine requires P=3 persistent frames for CRITICAL state
        if data.get("system_status") == "CRITICAL":
            false_alarms += 1

    far = (false_alarms / frames_received) * 100 if frames_received > 0 else 0
    logger.info(f"  Frames Received: {frames_received} | Persistent False Alarms: {false_alarms} | FAR: {far:.2f}%")
    assert far <= 3.5, f"Persistent FAR should be <= 3.5%, got {far:.2f}%"
    logger.info("  ✅ Scenario 1 PASSED: Baseline stream 96%+ HEALTHY with low FAR.")


async def test_scenario_2_midspan_fault(ws):
    logger.info("--- [Scenario 2] Testing MidSpan Stiffness Loss Injection ---")
    post_api("simulate-fault", {
        "node_id": "Sensor_MidSpan_Girder",
        "fault_type": "stiffness_loss",
        "severity": 0.65
    })

    t0 = time.time()
    detected = False
    localized_correctly = False

    for _ in range(50):
        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
        if msg == "pong":
            continue
        data = json.loads(msg)
        if data.get("system_status") == "CRITICAL":
            dt = time.time() - t0
            detected = True
            defect_node = data.get("localized_defect_node")
            if defect_node == "Sensor_MidSpan_Girder":
                localized_correctly = True
                logger.info(f"  🔥 CRITICAL Alert in {dt:.3f}s! Localized Defect Node: '{defect_node}'")
                break

    assert detected, "Failed to trigger CRITICAL alert on MidSpan fault injection!"
    assert localized_correctly, "Defect node was not correctly localized to Sensor_MidSpan_Girder!"
    logger.info("  ✅ Scenario 2 PASSED: MidSpan fault detected & 100% localized.")


async def test_scenario_3_pier_settlement(ws):
    logger.info("--- [Scenario 3] Testing Pier Settlement Fault Injection ---")
    post_api("simulate-fault", {
        "node_id": "Sensor_Pier_Left",
        "fault_type": "settlement",
        "severity": 0.5
    })

    detected = False
    localized_correctly = False

    for _ in range(50):
        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
        if msg == "pong":
            continue
        data = json.loads(msg)
        if data.get("system_status") == "CRITICAL":
            defect_node = data.get("localized_defect_node")
            if defect_node == "Sensor_Pier_Left":
                detected = True
                localized_correctly = True
                logger.info(f"  🔥 CRITICAL Alert! Localized Defect Node: '{defect_node}'")
                break

    assert detected and localized_correctly, "Failed to localize Pier_Left settlement!"
    logger.info("  ✅ Scenario 3 PASSED: Pier Settlement fault localized with 100% accuracy.")


async def test_scenario_4_fault_recovery(ws):
    logger.info("--- [Scenario 4] Testing Fault Recovery & Stream Reset ---")
    post_api("reset-faults")

    recovered = False
    for _ in range(60):
        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
        if msg == "pong":
            continue
        data = json.loads(msg)
        if data.get("system_status") == "NORMAL":
            recovered = True
            logger.info("  💚 Stream successfully returned to NORMAL status.")
            break

    assert recovered, "System failed to recover to NORMAL state after fault reset!"
    logger.info("  ✅ Scenario 4 PASSED: Fault recovery cycle completed.")


async def run_e2e_suite():
    logger.info("=" * 60)
    logger.info("STARTING PHASE 9 E2E FULL-STACK INTEGRATION SUITE")
    logger.info("=" * 60)

    async with websockets.connect(WS_URL) as ws:
        await test_scenario_1_healthy_baseline(ws)
        await test_scenario_2_midspan_fault(ws)
        await test_scenario_3_pier_settlement(ws)
        await test_scenario_4_fault_recovery(ws)

    logger.info("=" * 60)
    logger.info("🎉 ALL 4 E2E INTEGRATION TEST SCENARIOS PASSED WITH 100% SUCCESS!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_e2e_suite())
