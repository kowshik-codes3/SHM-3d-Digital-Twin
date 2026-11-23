#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================================"
echo "Starting StructureIQ Digital Twin Platform"
echo "========================================================"

# Trap SIGINT/SIGTERM to stop background processes
cleanup() {
    echo "Stopping StructureIQ services..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start backend
cd "$ROOT_DIR/backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 2

# Start frontend
cd "$ROOT_DIR/frontend/shm-dashboard"
npm run dev &
FRONTEND_PID=$!

echo "System Active:"
echo "  - Backend REST & WebSocket: http://localhost:8000"
echo "  - Interactive 3D Frontend:   http://localhost:5173"

wait
