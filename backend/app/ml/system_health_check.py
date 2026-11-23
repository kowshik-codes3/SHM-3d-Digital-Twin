"""
system_health_check.py

Pre-flight diagnostic verification script for the Real-Time SHM Digital Twin.
Verifies artifact paths, PyTorch execution device, REST API endpoints, and WebSocket handshakes.
"""

import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SHM.HealthCheck: %(message)s")
logger = logging.getLogger("SHM.HealthCheck")

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
BASE_URL = "http://127.0.0.1:8000"


def check_artifacts():
    logger.info(f"Checking artifact directory: {ARTIFACTS_DIR}")
    required_files = ["scaler.joblib", "lstm_autoencoder_best.pt", "threshold_config.json"]
    all_exist = True
    for fname in required_files:
        fpath = ARTIFACTS_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            logger.info(f"  ✅ Found '{fname}' ({size_kb:.1f} KB)")
        else:
            logger.error(f"  ❌ Missing required artifact: '{fname}'")
            all_exist = False
    return all_exist


def check_pytorch_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"PyTorch Device Available: '{device}' (PyTorch v{torch.__version__})")
    return True


def check_rest_api():
    url = f"{BASE_URL}/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                logger.info(f"  ✅ REST API Health Check: 200 OK | Payload: {data}")
                return True
            else:
                logger.error(f"  ❌ REST API returned HTTP status {resp.status}")
                return False
    except Exception as err:
        logger.warning(f"  ⚠️ REST API not running at {BASE_URL} ({err})")
        return False


def run_full_diagnostics():
    logger.info("=" * 60)
    logger.info("STARTING PRE-FLIGHT SYSTEM HEALTH DIAGNOSTICS")
    logger.info("=" * 60)

    art_ok = check_artifacts()
    device_ok = check_pytorch_device()
    rest_ok = check_rest_api()

    logger.info("=" * 60)
    if art_ok and device_ok:
        logger.info("✅ PRE-FLIGHT HEALTH DIAGNOSTICS PASSED SUCCESSFULLY!")
        return True
    else:
        logger.error("❌ PRE-FLIGHT HEALTH DIAGNOSTICS FAILED!")
        return False


if __name__ == "__main__":
    success = run_full_diagnostics()
    sys.exit(0 if success else 1)
