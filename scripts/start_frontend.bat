@echo off
title StructureIQ Frontend
cd /d %~dp0\..\frontend\shm-dashboard
echo Starting StructureIQ React Frontend on port 5173...
npm run dev
