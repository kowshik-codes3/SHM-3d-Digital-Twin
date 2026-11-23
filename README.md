# StructureIQ — Multi-Structure SHM Digital Twin Platform

[![CI Pipeline](https://github.com/owner/structureiq/actions/workflows/ci.yml/badge.svg)](https://github.com/owner/structureiq/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2.8-61DAFB.svg)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-0.186.1-black.svg)](https://threejs.org/)

StructureIQ is a sub-second Structural Health Monitoring (SHM) Digital Twin platform integrating real-time sensor telemetry ingestion, physics-informed digital signal processing, PyTorch LSTM-Autoencoders for spatio-temporal anomaly detection, and interactive 3D WebGL digital twin rendering.

---

## ⚡ Quick Start (2 Commands)

### 1. Launch FastAPI Backend
```bash
# Windows
start_backend.bat

# Linux / macOS
./scripts/start_backend.sh

# Or directly with Python:
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*API Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs) | Health Check: [http://localhost:8000/health](http://localhost:8000/health)*

### 2. Launch 3D WebGL Frontend
```bash
# Windows
start_frontend.bat

# Linux / macOS
./scripts/start_frontend.sh

# Or directly from root:
npm run dev
```
*Interactive 3D Digital Twin: [http://localhost:5173](http://localhost:5173)*

> **One-Click Windows Launcher**: Run `start_all.bat` (or `scripts\start_all.bat`) to launch both services in separate terminal windows automatically.

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.11+**
- **Node.js 20+** and **npm**

### Step 1: Clone & Configure Environment
```bash
git clone <repo-url>
cd "Interactive 3D Digital Twin Integration"

# Copy environment variables template
cp .env.example .env
```

### Step 2: Install Backend Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
npm --prefix frontend/shm-dashboard ci
```

---

## ⚙️ Configuration & Environment Variables

All parameters are driven by `.env` (with defaults in `backend/app/core/config.py` and `frontend/shm-dashboard/src/config.js`):

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `BACKEND_HOST` | `0.0.0.0` | FastAPI server bind address |
| `BACKEND_PORT` | `8000` | FastAPI HTTP/WebSocket port |
| `FRONTEND_PORT` | `5173` | Vite dev server port |
| `FS` | `50.0` | Sampling frequency (Hz) |
| `WINDOW_SIZE` | `50` | LSTM sliding window length |
| `PERSISTENCE` | `3` | Anti-flapping window persistence ($P \ge 3$) |
| `T_PERSIST` | `0.04` | Min continuous exceedance time (s) |
| `DEVICE` | `cpu` | PyTorch execution device (`cpu` or `cuda`) |
| `DB_PATH` | `data/shm.db` | SQLite database location |
| `ARTIFACTS_DIR`| `artifacts` | Pre-trained models, scalers, thresholds |
| `VITE_API_URL` | `http://localhost:8000` | Frontend backend REST endpoint |
| `VITE_WS_URL`  | `ws://localhost:8000`   | Frontend WebSocket stream endpoint |

---

## 🧪 Testing & Benchmarks

### Run PyTest Suite
```bash
pytest backend/tests -v
```

### Run Latency & Memory Benchmark
```bash
python scripts/run_benchmark.py
```
*Measures circular buffer push, PyTorch forward pass, anomaly state machine, and total end-to-end latency ($< 250\text{ ms}$ SLA).*

### Run Automated End-to-End Integration Suite
```bash
python scripts/run_e2e.py
```
*Tests 4 live scenarios over WebSockets against a running FastAPI instance:*
1. **Healthy Baseline**: Verifies zero false alarms over continuous frames.
2. **MidSpan Stiffness Loss**: Tests 50-ms CRITICAL alarm detection and 100% spatial localization.
3. **Pier Foundation Settlement**: Tests baseline drift detection and spatial localization.
4. **Fault Recovery**: Tests automatic recovery back to NOMINAL state upon clearing faults.

---

## 📁 Repository Structure

```
├── .github/workflows/     # CI Pipeline (Pytest, Benchmark, E2E, Frontend build)
├── backend/
│   ├── app/
│   │   ├── core/          # Config, registry, connection manager, DSP filters
│   │   ├── db/            # SQLite schema, alert repository
│   │   ├── ingestion/     # Adapters (Simulator, HTTP, MQTT, Kafka, Replay)
│   │   ├── ml/            # LSTM Autoencoder, calibrator, anomaly engine
│   │   ├── simulation/    # Physics modal harmonic signal generator
│   │   └── main.py        # FastAPI app, endpoints, WebSocket router
│   ├── tests/             # Automated test suites (Phases 1 - 5)
│   └── requirements.txt   # Pinned Python package dependencies
├── frontend/
│   └── shm-dashboard/     # React 19 + Vite + React Three Fiber + Tailwind v4
│       ├── src/
│       │   ├── components/# 3D Canvas, FleetGrid, AlertsInbox, Editors
│       │   ├── hooks/     # WebSocket telemetry stream hook
│       │   └── config.js  # Centralized environment connection config
│       └── package.json   # Frontend npm package dependencies & scripts
├── docs/                  # Architecture, KW51 dataset, and API documentation
├── scripts/               # Start scripts, benchmark runner, and E2E test runner
├── data/                  # SQLite database and real KW51 bridge dataset
├── artifacts/             # Scalers, neural weights, and threshold JSONs
├── .env.example           # Environment template
└── README.md
```
