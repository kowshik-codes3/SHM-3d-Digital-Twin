"""
run_pipeline.py

Master pipeline orchestrator for the Real-Time SHM Digital Twin Backend.
Runs pre-flight health diagnostics and boots the Uvicorn FastAPI streaming server.
"""

import logging
import os
import sys
import uvicorn
from pathlib import Path

# Ensure backend root directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.system_health_check import run_full_diagnostics

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SHM.Pipeline: %(message)s")
logger = logging.getLogger("SHM.Pipeline")


def main():
    logger.info("=" * 60)
    logger.info("SHM DIGITAL TWIN — MASTER PIPELINE ORCHESTRATOR")
    logger.info("=" * 60)

    # Step 1: Pre-flight System Health Diagnostic
    if not run_full_diagnostics():
        logger.error("Pre-flight system health check failed. Aborting pipeline launch.")
        sys.exit(1)

    logger.info("Starting Uvicorn FastAPI server on http://127.0.0.1:8000...")
    logger.info("WebSocket streaming endpoint: ws://127.0.0.1:8000/ws/telemetry")
    logger.info("Press CTRL+C to terminate.")

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
