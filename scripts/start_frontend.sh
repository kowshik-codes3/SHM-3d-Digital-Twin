#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/frontend/shm-dashboard"
echo "Starting StructureIQ React Frontend on port 5173..."
npm run dev
