@echo off
title StructureIQ - Backend API (FastAPI)
cd /d "%~dp0\backend"
echo ========================================================
echo Starting StructureIQ FastAPI Telemetry Server on Port 8000
echo ========================================================
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
