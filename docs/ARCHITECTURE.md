# StructureIQ Architecture & Design Overview

StructureIQ is a sub-second Structural Health Monitoring (SHM) Digital Twin platform that connects continuous vibration telemetry with machine learning anomaly detection and interactive 3D WebGL visualization.

## System Components

1. **Ingestion Layer (`backend/app/ingestion/`)**:
   - `SimulatorSource`: Real-time synthetic modal generator producing 1.5 Hz fundamental and 3.8 Hz torsional harmonics alongside temperature cycles.
   - `HTTPSource`: REST push ingestion for edge DAQ gateways.
   - `MQTTSource`: Asynchronous MQTT broker subscriber.
   - `KafkaSource`: Enterprise Kafka event stream subscriber.
   - `ReplaySource`: Historical parquet dataset replayer.

2. **Signal Processing & ML Engine (`backend/app/ml/`)**:
   - `DSP`: 4th-order Butterworth bandpass filter (0.5 – 18.0 Hz) and Welch PSD computation.
   - `ThermalNormalizer`: Diurnal temperature compensation removing ambient thermal drift.
   - `StructuralDataScaler`: Robust channel standardizer.
   - `LSTMAutoencoder`: PyTorch deep encoder-decoder model evaluating temporal-spatial reconstruction error ($50 \times 5$ window).
   - `ThresholdCalibrator`: Statistical process control (warning) and Extreme Value Theory / Peaks-Over-Threshold (critical) limit calibration.
   - `AnomalyStateMachine`: Multi-state persistence filter ($P \ge 3$, $t \ge 0.04\text{s}$) with spatial adjacency localization.

3. **Backend Service (`backend/app/`)**:
   - FastAPI with async lifespan management.
   - Dynamic `StructureRegistry` & `StructureRuntime` for multi-asset fleets.
   - SQLite `AlertRepository` for persistent incident records and operator acknowledgments.
   - WebSocket broadcast (`/ws/telemetry/{structure_id}`).

4. **Frontend 3D Visualizer (`frontend/shm-dashboard/`)**:
   - React 19 + React Three Fiber (`@react-three/fiber`, `@react-three/drei`).
   - Procedural 3D bridge deck with animated sensor spheres that lerp emissive color on alarms.
   - Real-time Recharts telemetry curves.
   - Multi-asset fleet grid, alert console, and 3D sensor placement editor.
