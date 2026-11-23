#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"
echo "Starting StructureIQ FastAPI Backend on port 8000..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
