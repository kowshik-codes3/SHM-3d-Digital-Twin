@echo off
title StructureIQ - Full Platform Launcher
echo ========================================================
echo Starting StructureIQ Digital Twin (Backend + Frontend)
echo ========================================================

echo [1/2] Launching FastAPI Backend on Port 8000...
start "StructureIQ Backend" cmd /k "cd /d %~dp0\backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 2 /nobreak > nul

echo [2/2] Launching Vite React Frontend on Port 5173...
start "StructureIQ Frontend" cmd /k "cd /d %~dp0\frontend\shm-dashboard && npm run dev"

echo.
echo ========================================================
echo System Active:
echo   - Backend REST & WebSocket: http://localhost:8000
echo   - Interactive 3D Frontend:   http://localhost:5173
echo ========================================================
