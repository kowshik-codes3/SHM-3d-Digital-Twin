@echo off
title StructureIQ Backend
cd /d %~dp0\..\backend
echo Starting StructureIQ FastAPI Backend on port 8000...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
